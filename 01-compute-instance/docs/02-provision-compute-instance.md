# 02 — Provision a GPU Compute Instance

This step creates an AML **Compute Instance** with a GPU and an **inline
creation script** that automatically installs a custom conda environment with all
fine-tuning dependencies.

The corresponding notebook is **`aml_ci_create.ipynb`**.

---

## Why Inline Creation Script?

AML supports two script hooks on a Compute Instance:

| Hook | When it runs | Can be patched after creation? |
|------|-------------|-------------------------------|
| `creationScript` | Once, immediately after provisioning | No |
| `startupScript` | Every boot | Yes |

We use `creationScript` to install the conda environment **once** at provisioning
time. The inline approach embeds the script directly in the ARM request so it is
not dependent on file-share access (important for NSP / private-endpoint setups).

---

## Step by Step

### 1. Load Configuration

```python
import os
from dotenv import load_dotenv
from azure.ai.ml import MLClient
from utils.amlauth import AuthHelper

config_file_path = os.path.join(".", "config", ".env")
load_dotenv(dotenv_path=config_file_path, override=True)

settings = AuthHelper.load_settings()
credential = AuthHelper.test_credential()

ml_client = MLClient(
    credential,
    settings.subscription_id,
    settings.resource_group,
    settings.workspace,
)

workspace_location = ml_client.workspaces.get(settings.workspace).location
print(f"Workspace location: {workspace_location}")
```

### 2. Load SSH Public Key

```python
ssh_public_key_path = os.path.expanduser(
    os.path.join("~", ".ssh", settings.ssh_pub_key_name)
)
with open(ssh_public_key_path, "r") as f:
    ssh_public_key_content = f.read().strip()
```

### 3. Prepare the Inline Script

The creation script combines:

- **`environments/slm_conda_sft.yaml`** — the conda environment definition
- **`setup_conda_env.sh`** — the shell script that creates the env and registers
  the Jupyter kernel

They are combined into a single bash script using a **heredoc** so the YAML is
embedded inline:

```python
import base64

# Read the conda YAML and shell script
with open("environments/slm_conda_sft.yaml", "r") as f:
    yaml_content = f.read()
with open("setup_conda_env.sh", "r") as f:
    setup_body = f.read()

# Combine into a single script with heredoc
combined_script = f"""#!/bin/bash
set -e
mkdir -p /tmp/aml-setup
cat > /tmp/aml-setup/slm_conda_sft.yaml << 'YAML_HEREDOC'
{yaml_content.rstrip()}
YAML_HEREDOC

{setup_body}
"""

script_b64 = base64.b64encode(combined_script.encode()).decode()
print(f"Script size: {len(script_b64)} base64 chars (limit: 4096)")
```

!!! warning "Script size limit"
    The inline `scriptData` field is limited to **4096 base64 characters**. Keep
    the combined script concise.

### 4. Create the Compute Instance (SDK-first Hybrid)

The Python SDK does not expose `creationScript` with inline mode directly, so we
use a **hybrid approach**: build the `ComputeInstance` with the SDK, serialize it
to an ARM body, inject the inline script, and PUT via the REST API.

```python
import requests, time
from azure.ai.ml.entities import (
    ComputeInstance,
    ComputeInstanceSshSettings,
)

ci_name = "t4-phi4-ft"
ci_size = "Standard_NC4as_T4_v3"   # T4 GPU, 4 vCPUs, 28 GB RAM

# Build via SDK
ci = ComputeInstance(
    name=ci_name,
    size=ci_size,
    ssh_settings=ComputeInstanceSshSettings(
        ssh_public_access="Enabled",
        admin_public_key=ssh_public_key_content,
    ),
    idle_time_before_shutdown="PT15M",  # auto-stop after 15 min idle
)

# Serialize to ARM
arm_body = ci._to_rest_object().serialize()
arm_body["location"] = workspace_location

# Inject inline creation script
arm_body["properties"]["properties"]["setupScripts"] = {
    "scripts": {
        "creationScript": {
            "scriptSource": "inline",
            "scriptData": script_b64,
            "timeout": "25m",
        }
    }
}

# PUT via ARM REST
arm_url = (
    f"https://management.azure.com/subscriptions/{settings.subscription_id}"
    f"/resourceGroups/{settings.resource_group}"
    f"/providers/Microsoft.MachineLearningServices"
    f"/workspaces/{settings.workspace}"
    f"/computes/{ci_name}?api-version=2024-10-01"
)

token = credential.get_token("https://management.azure.com/.default").token
resp = requests.put(
    arm_url,
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    json=arm_body,
)
resp.raise_for_status()
print(f"Provisioning started: {resp.status_code}")
```

### 5. Poll for Completion

```python
while True:
    ci_state = ml_client.compute.get(ci_name)
    state = ci_state.state if hasattr(ci_state, "state") else "Unknown"
    print(f"  State: {state}")
    if state in ("Running", "Failed", "Canceled"):
        break
    time.sleep(30)

print(f"Compute Instance '{ci_name}' is now {state}.")
```

---

## VM SKU Selection

| SKU | GPU | vCPUs | RAM | Best for |
|-----|-----|-------|-----|----------|
| `Standard_NC4as_T4_v3` | 1× T4 (16 GB) | 4 | 28 GB | QLoRA fine-tuning of SLMs |
| `Standard_NC6s_v3` | 1× V100 (16 GB) | 6 | 112 GB | Larger models / faster training |
| `Standard_NC24ads_A100_v4` | 1× A100 (80 GB) | 24 | 220 GB | Full fine-tuning |

For Phi-4-mini-instruct with QLoRA, the **T4** (`Standard_NC4as_T4_v3`) is
sufficient and cost-effective.

---

## List / Check / Delete Compute Instances

```python
# List
for item in ml_client.compute.list():
    if item.type == "computeinstance":
        print(item.name, item.state)

# Stop
ml_client.compute.begin_stop(ci_name).wait()

# Start
ml_client.compute.begin_start(ci_name).wait()

# Delete
ml_client.compute.begin_delete(ci_name).wait()
```

---

Next: [03 — Custom Conda Environment](03-custom-conda-environment.md)
