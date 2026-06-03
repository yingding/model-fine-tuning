"""CPU-side base-model download for offline GPU fine-tuning.

Downloads a HuggingFace model (default: Llama-3-8B-Instruct) and saves it
to `--output_dir` as a flat snapshot (the usual `config.json`,
`model-*.safetensors`, tokenizer files, etc.). The output directory is
then registered as a versioned AML Model asset by the orchestrating
notebook; the GPU training job mounts that asset instead of pulling from
HuggingFace at runtime.

Why this exists:
  - HF Hub is rate-limited and sometimes blocked from corp / managed-VNet
    networks; the GPU job hangs on `AutoModelForCausalLM.from_pretrained`
    or fails with HTTPError 429/403.
  - Downloading once on CPU compute is ~10x cheaper than holding a GPU
    box idle while bytes stream from HF.
  - The cached snapshot is reproducible and versionable as an AML asset.

Two output flavors via `--layout`:
  flat      (default) — writes a directory you can pass directly to
                        `AutoModelForCausalLM.from_pretrained(<dir>)`.
                        Best for the AML "mount this folder" workflow.
  hf_cache  — preserves the HF cache directory structure
              (`models--<org>--<repo>/snapshots/<sha>/...`).
              Use if you want bitwise compatibility with HF's symlink
              layout for downstream tooling that expects it.

Auth: set `HF_TOKEN` env var (or `--hf_token`) for gated models like
official Meta-Llama-3-8B-Instruct. The unsubmitted re-publication
NousResearch/Meta-Llama-3-8B-Instruct does not require a token.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _download_flat(model_id: str, output_dir: str, hf_token: str | None,
                   revision: str | None, allow_patterns: list[str] | None) -> None:
    """Download to local node disk first, then copy to the (BlobFuse) output.

    Downloading directly into the mounted output_dir hits two problems:
      1. huggingface_hub stages files under `<output_dir>/.cache/huggingface/download`
         and renames into place. On BlobFuse that intermediate write + rename
         is slow and sometimes silently fails.
      2. statvfs() on BlobFuse returns 0 bytes free, triggering noisy
         "Not enough free disk space" warnings even when the upload succeeds.

    Compute cluster nodes have fast local SSD scratch space (typically
    100GB+ on A100 SKUs). Staging there and copying once at the end is
    both faster and more reliable than streaming each file through BlobFuse.
    """
    from huggingface_hub import snapshot_download

    # Prefer the AML-provided scratch dir, fall back to system temp.
    scratch_root = os.environ.get("AZ_BATCHAI_JOB_TEMP") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="hf_snap_", dir=scratch_root) as local_dir:
        print(f"Staging to local SSD : {local_dir}")
        print(f"snapshot_download(repo_id={model_id!r}, revision={revision!r}) → {local_dir}")
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=hf_token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )

        # Drop the HF .cache subdir before copy — it's incomplete-file scratch.
        cache_subdir = Path(local_dir) / ".cache"
        if cache_subdir.exists():
            shutil.rmtree(cache_subdir, ignore_errors=True)

        print(f"Copying snapshot to mount: {local_dir} → {output_dir}")
        # copytree(..., dirs_exist_ok=True) needs Python 3.8+; container has 3.12.
        shutil.copytree(local_dir, output_dir, dirs_exist_ok=True)
        print("Copy complete.")


def _download_hf_cache(model_id: str, output_dir: str, hf_token: str | None,
                       revision: str | None) -> None:
    """Use AutoModel/AutoTokenizer .from_pretrained() which writes the HF
    cache layout under `output_dir`.

    Same local-SSD staging as `_download_flat` — see that docstring.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scratch_root = os.environ.get("AZ_BATCHAI_JOB_TEMP") or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="hf_cache_", dir=scratch_root) as local_dir:
        print(f"Staging HF cache to local SSD : {local_dir}")
        os.environ["HF_HOME"] = local_dir
        print(f"AutoTokenizer.from_pretrained({model_id!r}, cache_dir={local_dir!r})")
        AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, token=hf_token,
                                      revision=revision)
        print(f"AutoModelForCausalLM.from_pretrained({model_id!r}, cache_dir={local_dir!r})")
        AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=local_dir, token=hf_token, revision=revision,
            torch_dtype="auto",
        )

        print(f"Copying HF cache to mount: {local_dir} → {output_dir}")
        shutil.copytree(local_dir, output_dir, dirs_exist_ok=True)
        print("Copy complete.")


def _summarize(output_dir: str) -> None:
    root = Path(output_dir)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"\n✅ Downloaded {len(files)} files, {total / 1024 / 1024:.1f} MiB total")
    # Print the top-level structure
    top = sorted({p.relative_to(root).parts[0] for p in files})
    for entry in top[:15]:
        full = root / entry
        if full.is_file():
            print(f"   • {entry}  ({full.stat().st_size / 1024:.1f} KiB)")
        else:
            sub_files = sum(1 for _ in full.rglob("*") if _.is_file())
            sub_size  = sum(p.stat().st_size for p in full.rglob("*") if p.is_file())
            print(f"   • {entry}/  ({sub_files} files, {sub_size / 1024 / 1024:.1f} MiB)")


def main() -> int:
    print(f"Arguments: {sys.argv}")
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="NousResearch/Meta-Llama-3-8B-Instruct",
                   help="HF model id to download")
    p.add_argument("--revision", default=None,
                   help="Branch / tag / commit SHA; omit for the default branch")
    p.add_argument("--layout", choices=("flat", "hf_cache"), default="flat",
                   help="flat = directly loadable by from_pretrained(<dir>); "
                        "hf_cache = preserves HF cache layout")
    p.add_argument("--allow_patterns", nargs="*", default=None,
                   help="Optional glob patterns to restrict download "
                        "(e.g. *.safetensors tokenizer*.json *.json)")
    p.add_argument("--hf_token", default=None,
                   help="HF Hub token; falls back to HF_TOKEN env var")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the snapshot (mounted by AML)")
    args = p.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # AML mounts the output dir; ensure it exists and is writable.
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir : {args.output_dir}")
    print(f"Model id   : {args.model_id}  revision={args.revision or '(default)'}")
    print(f"Layout     : {args.layout}")
    print(f"HF token   : {'set' if hf_token else 'not set'}")

    # Some flat-mode model repos contain duplicate weight files (e.g. both
    # .bin and .safetensors). Default to preferring .safetensors + configs
    # + tokenizer; the user can override via --allow_patterns.
    allow_patterns = args.allow_patterns
    if args.layout == "flat" and allow_patterns is None:
        allow_patterns = [
            "*.safetensors", "*.safetensors.index.json",
            "*.json", "*.txt", "*.model",
            "tokenizer*", "special_tokens_map.json",
            "generation_config.json", "config.json",
        ]
        print(f"   default allow_patterns = {allow_patterns}")

    if args.layout == "flat":
        _download_flat(args.model_id, args.output_dir, hf_token, args.revision,
                       allow_patterns)
    else:
        _download_hf_cache(args.model_id, args.output_dir, hf_token, args.revision)

    _summarize(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
