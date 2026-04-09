# Custom Conda Environment Setup

Lessons learned from automating conda environment provisioning on an AML Compute Instance
via inline creation scripts, including file share constraints under NSP, CUDA forward-compatibility,
and common packaging pitfalls.

> **Related docs:**
> [01-Quick-Start.md](01-Quick-Start.md) — local environment and notebook workflow ·
> [02-About-Compute-Instance.md](02-About-Compute-Instance.md) — CI connection, storage layout, and lifecycle management

---

## 1. Creation Script Hooks

AML supports two script hooks on a Compute Instance:

| Hook | When it runs | Patchable after creation? |
|------|-------------|--------------------------|
| `creationScript` | Once, at provisioning | No |
| `startupScript` | Every boot | Yes |

A one-time conda install requires `creationScript`.

---

## 2. Script Delivery: ScriptReference vs Inline

### Option A — `ScriptReference` (SDK)

```python
from azure.ai.ml.entities import SetupScripts, ScriptReference

SetupScripts(
    creation_script=ScriptReference(
        path="Users/<username>/setup_conda_env.sh",  # relative to the Notebooks file share
        timeout_minutes=20,
    )
)
```

- `path` resolves against the AML Notebooks file share (`code-<guid>`).
- The file **must exist on the share before CI creation**.
- **Not usable from a local machine** when the storage account has a Network Security
  Perimeter (NSP) or private endpoint — see Section 3.

### Option B — Inline via ARM REST API

```python
import base64, requests

script_b64 = base64.b64encode(script_content.encode()).decode()

arm_body["properties"]["properties"]["setupScripts"] = {
    "scripts": {
        "creationScript": {
            "scriptSource": "inline",
            "scriptData": script_b64,
            "timeout": "25m",
        }
    }
}
```

- No file share or blob download required — the script is delivered by AML at provisioning time.
- **Timeout format**: `"<float>m"` only. ISO 8601 (`"PT20M"`) is rejected.
- **Hybrid approach**: build the `ComputeInstance` object with the Python SDK, serialize with
  `ci._to_rest_object().serialize()`, inject `setupScripts`, then `PUT` via `requests`.

---

## 3. File Share Access Constraints (NSP / Private Endpoint)

### Storage architecture

```
AML Workspace
  └── Storage Account
        ├── Blob container  : azureml-blobstore-<guid>   ← blob endpoint
        └── File share      : code-<guid>                ← file endpoint
```

### Access methods from a local machine

| Method | Protocol | NSP / Private Endpoint | Works? |
|--------|----------|------------------------|--------|
| SMB mount | SMB (445) | Blocked | ❌ |
| Azure Files REST + key auth | HTTPS (443) | Key auth may be disabled | ❌ |
| Azure Files REST + OAuth (`token_intent="backup"`) | HTTPS (443) | See below | ❌ |
| **Blob REST + OAuth** | **HTTPS (443)** | **Allowed** | **✅** |

#### Why `token_intent="backup"` fails

`token_intent="backup"` is reserved for Azure Backup service. The Azure Files data plane
rejects OAuth tokens from regular callers — even with `Storage File Data Privileged Contributor`
assigned and NSP allowing port 443. This is a service-level restriction, not RBAC or network.

#### Share name discovery

`service_client.list_shares()` fails with the same `AuthorizationFailure`.
Use the ARM control plane instead:

```python
from azure.mgmt.storage import StorageManagementClient

mgmt = StorageManagementClient(credential, subscription_id)
share_name = next(
    s.name for s in mgmt.file_shares.list(resource_group, storage_account_name)
    if s.name.startswith("code-")
)
```

Requires `Microsoft.Storage/storageAccounts/fileServices/fileshares/read`
(included in `Contributor` or `Storage Account Contributor`).

---

## 4. Recommended Approach (NSP Setup)

### Step 1 — Upload reference copies to blob (optional, for audit)

```python
from azure.storage.blob import BlobServiceClient

blob_svc = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net",
    credential=credential,
)
with open("setup_conda_env.sh", "rb") as f:
    blob_svc.get_blob_client(container, "setup-scripts/setup_conda_env.sh").upload_blob(f, overwrite=True)
```

### Step 2 — Create CI with inline script (yaml embedded as heredoc)

```python
combined_script = f"""#!/bin/bash
set -e
mkdir -p /tmp/aml-setup
cat > /tmp/aml-setup/slm_conda_sft.yaml << 'YAML_HEREDOC'
{yaml_content.rstrip()}
YAML_HEREDOC

{setup_body}
"""

script_b64 = base64.b64encode(combined_script.encode()).decode()

ci = ComputeInstance(name=ci_name, size=ci_size, ...)
arm_body = ci._to_rest_object().serialize()
arm_body["location"] = workspace_location
arm_body["properties"]["properties"]["setupScripts"] = {
    "scripts": {
        "creationScript": {
            "scriptSource": "inline",
            "scriptData": script_b64,
            "timeout": "25m",
        }
    }
}

token = credential.get_token("https://management.azure.com/.default").token
requests.put(arm_url, headers={"Authorization": f"Bearer {token}", ...}, json=arm_body)
```

### Required RBAC

| Role | Scope | Purpose |
|------|-------|---------|
| `Contributor` (or `Storage Account Contributor`) | Storage account | ARM share discovery |
| `Storage Blob Data Contributor` | Storage account | Blob upload (reference copies) |
| `Storage File Data Privileged Contributor` | Storage account | File share data-plane — **not usable from local** (see Section 3) |

---

## 5. CUDA / PyTorch Forward Compatibility

### Environment

| Component | Value |
|-----------|-------|
| VM SKU | `Standard_NC4as_T4_v3` (T4 GPU) |
| NVIDIA driver | `535.274.02` (pre-installed) |
| Driver CUDA version | **12.2** |
| Python | 3.14 (conda) |

### Problem

PyTorch ≥ 2.6.0 dropped cu121 wheels. The lowest available variant is cu126, which bundles
CUDA 12.6 runtime libraries. Driver 535 only exposes CUDA 12.2 — so `torch.cuda.is_available()`
returns `False`.

| Torch version | Available CUDA variants |
|---------------|------------------------|
| ≥ 2.6.0 | cu126+ only |
| 2.5.x | cu118, **cu121**, cu124 (last native match for driver 535) |

### Solution: `cuda-compat-12-6`

Install the CUDA forward-compatibility shim — no driver upgrade or reboot needed:

```bash
sudo apt-get install -y cuda-compat-12-6
```

This places shim libraries in `/usr/local/cuda-12.6/compat/` that let the CUDA 12.6 runtime
work on top of driver 535.

> If PyTorch still cannot find CUDA, ensure `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.6/compat/`.
> On most AML images, `cuda-compat` registers itself automatically.

### Why not upgrade the driver?

- Kernel module — cannot install via conda.
- `nvidia-driver-560` conflicts with pre-installed `nvidia-driver-535` packages.
- Requires a reboot, which is unsupported during `creationScript` execution.

### Why not downgrade to PyTorch 2.5.x (cu121)?

- PyTorch 2.5.1 does not ship wheels for Python 3.14.
- Downgrading Python to 3.12 risks incompatibility with newer `transformers`, `trl`, and `peft`.

---

## 6. Known Issues & Fixes

### Anaconda ToS — `CondaToSNonInteractiveError`

Conda requires ToS acceptance for `defaults` channels in non-interactive mode.
Accept explicitly in the setup script **before** `conda env create`:

```bash
source /anaconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -f /tmp/aml-setup/slm_conda_sft.yaml
```

### Timeout format

`setupScripts.creationScript.timeout` accepts `"<float>m"` only, max `"25m"`.

| Value | Result |
|-------|--------|
| `"PT30M"` | ❌ Invalid format |
| `"30m"` | ❌ Exceeds maximum |
| `"25m"` | ✅ Valid |

### scriptData size limit

The inline `scriptData` field (base64-encoded) is limited to **4096 characters**.
Current combined script size: ~2500 base64 characters.

### Other packaging issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| `tensorflow-serving-apt` repo | `apt-get update` fails (HTTP 403) | Remove broken repo files before `apt-get update` |
| `ipywidgets` missing | tqdm shows `IProgress not found` warning | Add `ipywidgets>=8.1.8` to conda deps |
| `ipykernel` via pip | Jupyter infra less reliable via pip | Move to conda dependencies |
| `--index-url` for PyTorch | Replaces PyPI; non-torch packages fail | Use `--extra-index-url` (keeps PyPI as fallback) |