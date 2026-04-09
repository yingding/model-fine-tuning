# Working with an AML Compute Instance

## Prerequisites

- A provisioned Compute Instance created via `aml_ci_create.ipynb`
- The custom conda environment (`sft-notebook`) installed via the inline creation script
- SSH key pair configured (public key was embedded at CI creation time)

---

## Connect to the Compute Instance

### Option A — SSH from Local Machine

```bash
# Get CI connection details from AML portal or CLI
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group>

# SSH in (default user: azureuser, default port: 50000)
ssh -i ~/.ssh/<private_key> -p 50000 azureuser@<public-ip>
```

### Option B — JupyterLab / VS Code (Web) via AML Portal

1. Go to **Azure ML Studio** → **Compute** → **Compute instances**
2. Click on the running CI name
3. Choose **JupyterLab**, **Jupyter**, or **VS Code (Web)** from the application links

### Option C — VS Code (Desktop) via Remote-SSH

1. SSH into the CI (see Option A)
2. Clone your repo under `~/localfiles` (local disk, **not** the SMB file share):

```bash
cd ~/localfiles
git clone <repo-url>
```

3. Open the folder in VS Code:

```bash
cd ~/localfiles/<repo-name>
code .
```

VS Code Desktop will connect via the Remote-SSH extension and open the workspace.

> **Why `~/localfiles` instead of `~/cloudfiles/code/`?**
> The `~/cloudfiles/code/` path is an SMB-mounted Azure Files share. Git operations, file
> watchers, and VS Code's file indexing are significantly slower over SMB. Using the local
> disk (`~/localfiles`) avoids these performance issues and provides a native file-system
> experience for development.

---

## Directory Layout on the CI

| Path | Description |
|------|-------------|
| `/home/azureuser/` | Home directory |
| `~/localfiles/` | Local disk — recommended for git repos and VS Code workspaces |
| `~/cloudfiles/code/` | AML Notebooks file share mount (SMB) |
| `~/cloudfiles/code/Users/<username>/` | Your personal notebooks directory |
| `/tmp/aml-setup/` | Inline creation script artifacts (conda yaml, etc.) |

---

## Activate the Custom Conda Environment

The `sft-notebook` conda environment is installed during CI provisioning.

```bash
# Activate conda
source /anaconda/etc/profile.d/conda.sh

# List available environments
conda env list

# Activate the fine-tuning environment
conda activate sft-notebook
```

### Verify GPU Access

```bash
conda activate sft-notebook
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

> If `torch.cuda.is_available()` returns `False`, ensure `cuda-compat-12-6` is installed.
> See [CustomEnvSetup.md](CustomEnvSetup.md) for details on the CUDA forward-compatibility shim.

### Use the Environment in Jupyter

The `sft-notebook` kernel is registered with Jupyter during provisioning. Select it from the
kernel picker in JupyterLab or VS Code:

- Kernel name: **sft-notebook**

If the kernel does not appear, re-register it:

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook
python -m ipykernel install --user --name sft-notebook --display-name "sft-notebook"
```

---

## Upload Notebooks and Data

### From a Local Machine (via Blob)

The AML file share is not directly accessible from local machines behind NSP / private
endpoints. Use blob storage as an intermediary or upload through the AML Studio UI.

```bash
# Upload via AML CLI
az ml data create --name my-dataset \
  --path ./local-data/ \
  --type uri_folder \
  --workspace-name <workspace> \
  --resource-group <resource-group>
```

### From the CI itself

```bash
# Files placed under the file share are visible in AML Studio → Notebooks
cp my_notebook.ipynb ~/cloudfiles/code/Users/<username>/
```

---

## Manage CI Lifecycle

### Check CI Status

Use `aml_ci_status.ipynb` or the CLI:

```bash
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group> \
  --query "{name:name, state:state, vmSize:size}" -o table
```

### List All Compute Instances

```bash
az ml compute list \
  --workspace-name <workspace> \
  --resource-group <resource-group> \
  --type ComputeInstance -o table
```

### Stop (Deallocate)

Stops billing for the VM while preserving the OS disk and environment.

```python
# From aml_ci_create.ipynb or a Python session:
ml_client.compute.begin_stop("<ci-name>").wait()
```

```bash
# Or via CLI:
az ml compute stop --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group>
```

### Start

```python
ml_client.compute.begin_start("<ci-name>").wait()
```

```bash
az ml compute start --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group>
```

### Delete

```python
ml_client.compute.begin_delete("<ci-name>").wait()
```

```bash
az ml compute delete --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group> --yes
```

> The CI is configured with `idle_time_before_shutdown: PT15M` — it will auto-stop after
> 15 minutes of inactivity.

---

## View Creation Script Logs

If the custom conda environment setup failed or you need to debug provisioning:

1. **AML Studio** → **Compute** → click on the CI → **Boot Diagnostics** → **Serial log**
2. Or SSH in and check:

```bash
# Creation script stdout/stderr logs
cat /var/log/aml_custom_setup/std_log.txt 2>/dev/null
# or
ls /var/log/cloud-init*.log
```

---

## Install Additional Packages

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook

# pip install
pip install <package-name>

# conda install
conda install -c conda-forge <package-name>
```

> Changes are persisted across stop/start cycles but lost on delete/recreate.
> To make packages permanent, add them to `environments/slm_conda_sft.yaml` and recreate.

---

## Reference

- [CustomEnvSetup.md](CustomEnvSetup.md) — Custom conda environment provisioning details
- [QuickStart.md](QuickStart.md) — Local development environment setup
- [AML Compute Instance docs](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)
- [Manage Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-compute-instance)
- [Customize Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-customize-compute-instance?view=azureml-api-2)
