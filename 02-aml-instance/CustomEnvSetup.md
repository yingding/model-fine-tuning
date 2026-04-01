# AML Compute Instance — Custom Conda Environment Setup

## Goal

Create an AML `ComputeInstance` that runs a custom conda environment install script
(`2_setup_conda_env.sh` + `environments/slm_conda_sft.yaml`) at provisioning time.

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
        path="Users/<username>/2_setup_conda_env.sh",  # path on the Notebooks file share
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
with open("2_setup_conda_env.sh", "rb") as f:
    blob_svc.get_blob_client(container, "setup-scripts/2_setup_conda_env.sh").upload_blob(f, overwrite=True)
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

**Fix**: Accept the ToS explicitly in `2_setup_conda_env.sh` **before** `conda env create`:

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