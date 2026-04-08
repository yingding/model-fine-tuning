# AML Compute Instance — Custom Conda Environment Setup

## Goal

Create an AML `ComputeInstance` that runs a custom conda environment install script
(`setup_conda_env.sh` + `environments/slm_conda_sft.yaml`) at provisioning time.

---

## AML Setup Script Options

AML supports two script hooks on a `ComputeInstance`:

| Hook | When it runs | Can be patched after creation? |
|------|-------------|-------------------------------|
| `creationScript` | Once, immediately after provisioning | ❌ No — provisioning-time only |
| `startupScript` | Every boot | ✅ Yes — can be added/updated via PATCH |

`creationScript` is required for a one-time conda env install (not `startupScript`).

---

## How to Pass the Script: `ScriptReference` vs Inline

### Option A — `ScriptReference(path=...)` (SDK)

```python
from azure.ai.ml.entities import SetupScripts, ScriptReference

SetupScripts(
    creation_script=ScriptReference(
        path="Users/<username>/setup_conda_env.sh",  # path on the Notebooks file share
        timeout_minutes=20, # must be <= 25min
    )
)
```

- `path` resolves relative to the AML Notebooks **file share** (`code-<guid>`), mounted at
  `~/cloudfiles/code/` on every CI in the workspace.
- The file **must already exist on the file share** before CI creation.
- **Cannot be used from a local machine** when the storage account uses a Network Security
  Perimeter (NSP) or private endpoint — see the file share access section below.

### Option B — Inline via ARM REST API (no file share required)

```python
import base64, requests

script_b64 = base64.b64encode(script_content.encode()).decode()

arm_body["properties"]["properties"]["setupScripts"] = {
    "scripts": {
        "creationScript": {
            "scriptSource": "inline",
            "scriptData": script_b64,   # base64-encoded bash script
            "timeout": "25m",           # format: "<float>m" (NOT ISO 8601 PT20M) and must be <=25m
        }
    }
}
```

- The script is delivered by AML at provisioning time — no file share or blob download needed.
- **Timeout format**: plain `"20m"` (float + `m`). ISO 8601 `"PT30M"` is rejected with:
  `The specified script timeout PT20M is invalid. It should be a floating point number followed by suffix 'm' for minutes`
- Use the SDK-first hybrid: build `ComputeInstance` with the Python SDK, serialize via
  `ci._to_rest_object().serialize()`, inject `setupScripts`, then PUT via `requests`.

---

## File Share Access from a Local Machine

### Architecture

```
AML Workspace
  └── Storage Account (amlworkspaceyw3405831058)
        ├── Blob container  : azureml-blobstore-<guid>   ← blob endpoint
        └── File share      : code-<guid>                ← file endpoint
              └── mounted on every CI at ~/cloudfiles/code/
```

### Why the File Share Is Inaccessible from Local

| Method | Protocol | Port | NSP / Private Endpoint | Works locally? |
|--------|----------|------|------------------------|----------------|
| SMB mount | SMB | 445 | Blocked by NSP | ❌ |
| Azure Files REST + key auth | HTTPS | 443 | NSP must allow; key auth may be disabled | ❌ |
| Azure Files REST + OAuth (`token_intent="backup"`) | HTTPS | 443 | NSP allows 443, but... | ❌ |
| Blob REST + OAuth | HTTPS | 443 | NSP allows 443 | ✅ |

#### Why `token_intent="backup"` Fails (403 `AuthorizationFailure`)

`token_intent="backup"` is a **Microsoft-internal feature reserved for Azure Backup service**.
The Azure Files service rejects OAuth tokens from regular callers even when:
- `Storage File Data Privileged Contributor` RBAC is correctly assigned at the storage account scope, and
- The NSP allows outbound/inbound on port 443.

This is not a RBAC or network issue — it is a service-level restriction.

#### Why Blob Works

The Blob REST API fully supports OAuth (`Storage Blob Data Contributor` role) over HTTPS.
The blob endpoint (`*.blob.core.windows.net`) is accessible through NSP on port 443.

### File Share Discovery

The `code-*` share name cannot be discovered via `service_client.list_shares()` from local
(same `AuthorizationFailure`). Use the ARM control plane instead:

```python
from azure.mgmt.storage import StorageManagementClient

mgmt = StorageManagementClient(credential, subscription_id)
share_name = next(
    s.name for s in mgmt.file_shares.list(resource_group, storage_account_name)
    if s.name.startswith("code-")
)
```

Requires `Microsoft.Storage/storageAccounts/fileServices/fileshares/read` (included in
`Contributor` or `Storage Account Contributor`).

---

## Recommended Approach (NSP / Private Endpoint Setup)

### Step 1 — Upload reference copies to blob (optional, for audit/backup)

```python
from azure.storage.blob import BlobServiceClient

blob_svc = BlobServiceClient(
    account_url=f"https://{storage_account_name}.blob.core.windows.net",
    credential=credential,
)
with open("setup_conda_env.sh", "rb") as f:
    blob_svc.get_blob_client(container, "setup-scripts/setup_conda_env.sh").upload_blob(f, overwrite=True)
```

### Step 2 — Embed yaml as heredoc and create CI with inline script

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

---

## Required RBAC

| Role | Scope | Used for |
|------|-------|---------|
| `Contributor` (or `Storage Account Contributor`) | Storage account | ARM share discovery via `file_shares.list()` |
| `Storage Blob Data Contributor` | Storage account | Blob upload (reference copies) |
| `Storage File Data Privileged Contributor` | Storage account | File share data-plane — **not usable from local machine** (see above) |

---

## Known Issues & Fixes

### Anaconda Terms of Service — `CondaToSNonInteractiveError`

When the conda env yaml includes the `defaults` channel (`pkgs/main`, `pkgs/r`), conda
requires ToS acceptance before resolving packages. In a non-interactive provisioning
context this throws:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for the following channels.
    - https://repo.anaconda.com/pkgs/main
    - https://repo.anaconda.com/pkgs/r
```

**Fix**: Accept the ToS explicitly in `setup_conda_env.sh` **before** `conda env create`:

```bash
source /anaconda/etc/profile.d/conda.sh

# Accept Anaconda ToS for non-interactive provisioning
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f /tmp/aml-setup/slm_conda_sft.yaml
```

The `defaults` channel can then remain in `environments/slm_conda_sft.yaml`:

```yaml
channels:
  - conda-forge
  - defaults
```

### Timeout Format Error

The `timeout` field in `setupScripts.creationScript` must be `"<float>m"`, **not** ISO 8601.
Maximum allowed value is `"25m"`.

| Value | Result |
|-------|--------|
| `"PT30M"` | ❌ Rejected — invalid format |
| `"30m"` | ❌ Rejected — exceeds 25-minute maximum |
| `"25m"` | ✅ Valid |

---

## CUDA / PyTorch Compatibility on AML Compute Instance

### Environment

| Component | Value |
|-----------|-------|
| VM SKU | `Standard_NC4as_T4_v3` (T4 GPU) |
| OS | Ubuntu 22.04 (jammy) |
| NVIDIA driver | `535.274.02` (pre-installed on AML image) |
| Driver CUDA version | **12.2** |
| Python | 3.14 (conda) |

### PyTorch CUDA wheel availability (cu121 dropped since v2.6.0)

| Torch version | Available CUDA variants |
|---------------|------------------------|
| 2.11.0 | cu126, cu128, cu130 |
| 2.10.0 | cu126, cu128, cu130 |
| 2.9.x | cu126, cu128, cu130 |
| 2.8.0 | cu126, cu128, cu129 |
| 2.7.x | cu118, cu126, cu128 |
| 2.6.0 | cu118, cu124, cu126 |
| **2.5.1** | **cu118, cu121, cu124** |
| **2.5.0** | **cu118, cu121, cu124** |

**cu121 was the last CUDA variant that works natively with driver 535 (CUDA 12.2).**
It was dropped starting with PyTorch 2.6.0. Versions 2.5.x and earlier have cu121 wheels
but likely lack Python 3.14 support.

### The problem

- Driver 535 exposes CUDA 12.2 at the kernel level.
- PyTorch cu126 wheels bundle CUDA 12.6 runtime libraries and require the driver to
  advertise CUDA ≥ 12.6.
- Without a fix, `torch.cuda.is_available()` returns `False` and CUDA operations fail.

### Solution: `cuda-compat-12-6` (forward compatibility shim)

Instead of upgrading the kernel-level NVIDIA driver (which requires a reboot and risks
package conflicts), install the **CUDA forward-compatibility package**:

```bash
sudo apt-get install -y cuda-compat-12-6
```

This installs shim libraries into `/usr/local/cuda-12.6/compat/` that let the CUDA 12.6
runtime work on top of driver 535. No driver removal, no reboot needed.

> **Note**: `LD_LIBRARY_PATH` may need to include `/usr/local/cuda-12.6/compat/` if PyTorch
> still cannot find CUDA after install. In practice, `cuda-compat` registers itself
> automatically on most AML images.

### Why not upgrade the NVIDIA driver?

- The driver is a **kernel module** — cannot be installed via conda.
- `nvidia-driver-560` (for CUDA 12.6) conflicts with the pre-installed `nvidia-driver-535`
  packages on the AML image.
- Even with `apt-get purge`, dependency tangles and held packages make it unreliable.
- A driver upgrade requires a **reboot**, which is not supported during CI `creationScript`
  execution.

### Why not downgrade PyTorch to 2.5.x with cu121?

- PyTorch 2.5.1 (last version with cu121) was released in October 2024.
- It does **not** ship wheels for Python 3.14 (released 2025).
- Downgrading Python to 3.12 would work but risks incompatibility with newer versions of
  `transformers`, `trl`, `peft`, and other packages in the environment.

### Other issues fixed in the setup script

| Issue | Symptom | Fix |
|-------|---------|-----|
| `tensorflow-serving-apt` repo | `apt-get update` fails with HTTP 403 | Remove broken repo files before `apt-get update` |
| `ipywidgets` missing | tqdm progress bars show `IProgress not found` warning | Add `ipywidgets>=8.1.8` to conda dependencies |
| `ipykernel` via pip | Jupyter infra packages more reliable via conda | Moved to conda dependencies section |
| `--index-url` for PyTorch | Replaces PyPI entirely; non-torch packages fail to resolve | Use `--extra-index-url` instead (keeps PyPI as fallback) |

### scriptData size constraint

The inline `creationScript` field (`scriptData`) is base64-encoded and limited to
**4096 characters**. The combined script (yaml heredoc + shell setup) must stay under
this limit. Current size: ~2500 base64 characters.

---

## SSH Access to a Running CI

```bash
# Get public IP and SSH port from AML portal or:
az ml compute show --name <ci-name> --workspace-name <ws> --resource-group <rg>

# Connect (default admin user is azureuser, port is 50000)
ssh -i ~/.ssh/<private_key> -p 50000 azureuser@<public-ip>

# File share is mounted at:
~/cloudfiles/code/
# User notebooks directory:
~/cloudfiles/code/Users/<aml-username>/
```