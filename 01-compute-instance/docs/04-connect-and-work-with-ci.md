# 04 — Connect and Work with the Compute Instance

Once your Compute Instance is **Running**, you can connect to it in several ways.

---

## Connection Options

### Option A — SSH from Local Machine

```bash
# Get CI details
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group>

# SSH in (default user: azureuser, default port: 50000)
ssh -i ~/.ssh/<private_key> -p 50000 azureuser@<public-ip>
```

### Option B — JupyterLab / VS Code (Web)

1. Open **Azure ML Studio** → **Compute** → **Compute instances**.
2. Click on the running CI name.
3. Choose **JupyterLab**, **Jupyter**, or **VS Code (Web)**.

### Option C — VS Code Desktop via Remote-SSH

1. SSH into the CI (Option A).
2. Clone the repo under `~/localfiles` (local disk):

```bash
cd ~/localfiles
git clone <repo-url>
cd <repo-name>
code .
```

!!! tip "Why `~/localfiles` instead of `~/cloudfiles/code/`?"
    The `~/cloudfiles/code/` path is an SMB-mounted Azure Files share. Git
    operations, file watchers, and VS Code file indexing are **significantly
    slower** over SMB. Use the local disk for development.

---

## Directory Layout

| Path | Description |
|------|-------------|
| `/home/azureuser/` | Home directory |
| `~/localfiles/` | Local disk — recommended for git repos and VS Code |
| `~/cloudfiles/code/` | AML Notebooks file share mount (SMB) |
| `~/cloudfiles/code/Users/<username>/` | Your personal notebooks directory |
| `/tmp/aml-setup/` | Inline creation script artifacts (conda yaml) |

---

## Activate the Conda Environment

```bash
source /anaconda/etc/profile.d/conda.sh
conda env list
conda activate sft-notebook
```

### Verify GPU Access

```bash
conda activate sft-notebook
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

Expected output on `Standard_NC4as_T4_v3`:

```
CUDA available: True
Device: Tesla T4
```

!!! note
    If `torch.cuda.is_available()` returns `False`, verify that `cuda-compat-12-6`
    is installed. See [03 — Custom Conda Environment](03-custom-conda-environment.md).

### Select the Kernel in Jupyter

The `sft-notebook` kernel is registered during provisioning. Select it from the
kernel picker in JupyterLab or VS Code.

If the kernel does not appear:

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook
python -m ipykernel install --user --name sft-notebook --display-name "sft-notebook"
```

---

## Manage CI Lifecycle

### Check Status

```python
ci_state = ml_client.compute.get("<ci-name>")
print(ci_state.state)
```

Or via CLI:

```bash
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group> \
  --query "{name:name, state:state, vmSize:size}" -o table
```

### Stop / Start / Delete

```python
# Stop (saves cost — auto-stop is set to 15 min idle)
ml_client.compute.begin_stop("<ci-name>").wait()

# Start
ml_client.compute.begin_start("<ci-name>").wait()

# Delete
ml_client.compute.begin_delete("<ci-name>").wait()
```

!!! tip "Auto-shutdown"
    The CI is provisioned with `idle_time_before_shutdown: PT15M`. It
    auto-stops after 15 minutes of inactivity.

---

## File Share Access from Local (NSP / Private Endpoint)

If your workspace storage uses a **Network Security Perimeter** or private
endpoints, the file share is not accessible from your local machine via SMB or
Azure Files REST.

**Workaround:** Use the **Blob REST API** (which supports OAuth over HTTPS) or
upload through the AML Studio UI.

| Method | Protocol | Works locally? |
|--------|----------|----------------|
| SMB mount | SMB / 445 | No (blocked by NSP) |
| Azure Files REST + key auth | HTTPS / 443 | No (key auth disabled) |
| Blob REST + OAuth | HTTPS / 443 | **Yes** |
| AML Studio Upload | HTTPS / 443 | **Yes** |

---

Next: [05 — Prepare Training Data](05-prepare-training-data.md)
