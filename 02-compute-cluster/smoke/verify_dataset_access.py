"""Verify dataset access end-to-end, bypassing AML Studio UI.

Tests four layers independently so you can tell which one is broken:

  1. AAD token        — can we get a token for storage?
  2. Storage container list — can we hit blob data plane directly (firewall + RBAC)?
  3. AML data-asset metadata — can the AML control plane read the asset record?
  4. AML data-asset stream   — can rslex actually read bytes (the path Studio uses)?

Usage:
    python verify_dataset_access.py <data-asset-name> [version]
    python verify_dataset_access.py --list                  # list all assets

Env: reads ../config/germanywest.env via utils.amlauth (same as the notebooks).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

# Make the parent dir importable so we reuse utils.amlauth from the notebooks.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from dotenv import load_dotenv  # noqa: E402

CONFIG_FILE = HERE.parent / "config" / "germanywest.env"
load_dotenv(dotenv_path=CONFIG_FILE, override=True)

from utils.amlauth import AuthHelper  # noqa: E402


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", help="Data asset name")
    parser.add_argument("version", nargs="?", default=None, help="Data asset version (default: latest)")
    parser.add_argument("--list", action="store_true", help="List all data assets and exit")
    args = parser.parse_args()

    settings = AuthHelper.load_settings()
    credential = AuthHelper.test_credential()

    from azure.ai.ml import MLClient
    ml_client = MLClient(credential, settings.subscription_id, settings.resource_group, settings.workspace)

    # ------------------------------------------------------------------
    if args.list:
        banner("Data assets in workspace")
        for a in ml_client.data.list():
            try:
                latest = ml_client.data.get(name=a.name, label="latest")
                print(f"  {a.name:40s}  v{latest.version:6s}  type={latest.type}")
            except Exception as e:  # noqa: BLE001
                print(f"  {a.name:40s}  (error: {e})")
        return 0

    if not args.name:
        parser.error("provide a data asset name, or --list")

    # ------------------------------------------------------------------
    banner("1. AAD token for storage resource")
    try:
        tok = credential.get_token("https://storage.azure.com/.default")
        print(f"   ✅ token acquired, expires in {tok.expires_on - int(__import__('time').time())}s")
    except Exception:  # noqa: BLE001
        print("   ❌ token acquisition failed:")
        traceback.print_exc()
        return 2

    # ------------------------------------------------------------------
    banner("2. Direct blob data-plane access")
    ws = ml_client.workspaces.get(settings.workspace)
    sa_name = ws.storage_account.rsplit("/", 1)[-1]
    blob_url = f"https://{sa_name}.blob.core.windows.net/"
    print(f"   storage account = {sa_name}")
    try:
        from azure.storage.blob import BlobServiceClient
        bsc = BlobServiceClient(account_url=blob_url, credential=credential)
        containers = [c.name for c in bsc.list_containers(results_per_page=20)]
        print(f"   ✅ list_containers OK, {len(containers)} containers")
        for c in containers[:10]:
            print(f"      • {c}")
    except Exception:  # noqa: BLE001
        print("   ❌ list_containers FAILED — firewall, NSP, or RBAC issue:")
        traceback.print_exc()
        # Don't bail; continue to AML control plane to localize the fault.

    # ------------------------------------------------------------------
    banner(f"3. AML data-asset metadata: {args.name} (version={args.version or 'latest'})")
    try:
        if args.version:
            asset = ml_client.data.get(name=args.name, version=args.version)
        else:
            asset = ml_client.data.get(name=args.name, label="latest")
        print(f"   ✅ asset: name={asset.name}  version={asset.version}  type={asset.type}")
        print(f"      path   : {asset.path}")
        print(f"      id     : {asset.id}")
    except Exception:  # noqa: BLE001
        print("   ❌ asset metadata fetch failed:")
        traceback.print_exc()
        return 3

    # ------------------------------------------------------------------
    banner("4. Resolve azureml:// path → blob → list + download a sample")
    #
    # We bypass mltable / azureml-fsspec (which need dotnetcore2, no arm64 wheel).
    # Instead we parse the azureml:// URI on the asset, look up the datastore to
    # find its container, and hit blob via BlobServiceClient. This proves the
    # exact data-plane path AML jobs and Studio preview use.
    from urllib.parse import urlparse
    from azure.storage.blob import BlobServiceClient

    try:
        path = asset.path or ""
        if not path.startswith("azureml://"):
            print(f"   ⚠️  asset path is not an azureml:// URI ({path!r}) — skipping")
            return 0

        # azureml://.../datastores/<dsname>/paths/<inner-path>
        parsed = urlparse(path)
        parts = parsed.path.strip("/").split("/")
        try:
            ds_idx = parts.index("datastores")
            paths_idx = parts.index("paths")
            ds_name = parts[ds_idx + 1]
            inner_path = "/".join(parts[paths_idx + 1:])
        except (ValueError, IndexError):
            print(f"   ❌ could not parse datastore/path from {path!r}")
            return 4

        print(f"   datastore        : {ds_name}")
        print(f"   path in datastore: {inner_path}")

        ds = ml_client.datastores.get(name=ds_name)
        container = getattr(ds, "container_name", None) or getattr(ds, "filesystem", None)
        ds_account  = getattr(ds, "account_name", None) or sa_name
        print(f"   blob container   : {container}  (account={ds_account})")

        bsc = BlobServiceClient(
            account_url=f"https://{ds_account}.blob.core.windows.net/",
            credential=credential,
        )
        container_client = bsc.get_container_client(container)

        blobs = list(container_client.list_blobs(name_starts_with=inner_path.rstrip("/")))
        print(f"\n   ✅ {len(blobs)} blob(s) under prefix:")
        for b in blobs[:10]:
            print(f"      • {b.name}  ({b.size} bytes)")
        if not blobs:
            print("   ⚠️  no blobs found — the asset path may not be uploaded yet")
            return 4

        # Read first 300 bytes of the smallest non-empty blob as a sanity check
        readable = [b for b in blobs if b.size and b.size < 4 * 1024 * 1024]
        if readable:
            target = min(readable, key=lambda b: b.size)
            blob_client = container_client.get_blob_client(target.name)
            head = blob_client.download_blob(offset=0, length=300).readall()
            print(f"\n   head of {target.name}:")
            print(head.decode("utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        print("   ❌ data-plane read FAILED — same path Studio preview uses:")
        traceback.print_exc()
        return 4

    banner("All checks passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
