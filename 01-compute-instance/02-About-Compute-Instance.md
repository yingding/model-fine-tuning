# About the AML Compute Instance

An AML **Compute Instance (CI)** is a managed cloud VM with GPU access, pre-installed
NVIDIA drivers, and a built-in Jupyter environment. It serves as your remote
development machine for fine-tuning.

---

## Prerequisites

Before connecting, make sure you have:

- A running Compute Instance provisioned via `aml_ci_create.ipynb`
- The `sft-notebook` conda environment installed (done automatically by the inline creation script)
- Your SSH **private** key matching the public key embedded at CI creation time

---

## 1. Connect to the Compute Instance

### Option A — SSH

```bash
# Look up CI connection info
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group>

# Connect (default user: azureuser, default port: 50000)
ssh -i ~/.ssh/<private_key> -p 50000 azureuser@<public-ip>
```

### Option B — JupyterLab / VS Code (Web)

1. Open **Azure ML Studio → Compute → Compute instances**.
2. Click on the running CI.
3. Choose **JupyterLab**, **Jupyter**, or **VS Code (Web)**.

### Option C — VS Code Desktop (Remote-SSH)

1. SSH into the CI (Option A).
2. Clone your repo under `~/localfiles`:

```bash
cd ~/localfiles
git clone <repo-url>
cd <repo-name> && code .
```

VS Code Desktop connects via the Remote-SSH extension.

---

## 2. Storage: Local Disk vs. File Share

The CI has two storage areas. Choosing the right one matters for performance.

| Path | Type | Speed | Persists across stop/start? | Use for |
|------|------|-------|-----------------------------|---------|
| `~/localfiles/` | Local SSD | Fast | Yes | Git repos, VS Code workspaces, model checkpoints |
| `~/cloudfiles/code/` | Azure Files (SMB) | Slow | Yes (shared across CIs) | Sharing notebooks via AML Studio |

> **Key insight:** Git operations, file watchers, and VS Code indexing are
> **significantly slower** on the SMB mount. Always develop under `~/localfiles/`.

> **Storage identity note:** Compute Instances currently do not support
> identity-based storage account access at creation time. Configure the workspace
> with **storage account key access** first, create the CI, then switch to
> identity-based access.
> ([GitHub Issue](https://github.com/Azure-Samples/ai-studio-in-a-box/issues/7#issuecomment-2273346002)
> · [FAQ / Workaround](https://github.com/Azure-Samples/ai-studio-in-a-box?tab=readme-ov-file#faq))

### Full directory layout

| Path | Description |
|------|-------------|
| `/home/azureuser/` | Home directory |
| `~/localfiles/` | Local disk — **recommended for development** |
| `~/cloudfiles/code/` | AML Notebooks file share mount (SMB) |
| `~/cloudfiles/code/Users/<username>/` | Your personal notebooks directory (visible in AML Studio) |
| `/tmp/aml-setup/` | Inline creation script artifacts (conda yaml) |

### Jupyter notebook mount paths

| Variable | Value |
|----------|-------|
| Default mount path | `/mnt/batch/tasks/shared/LS_root/mounts/clusters/<ci-name>` |
| User data home | `code/Users/<username>` |
| Full notebook path | `/mnt/batch/tasks/shared/LS_root/mounts/clusters/<ci-name>/code/Users/<username>` |

### Upload files to the CI

**From the CI itself** — copy files to the file share so they appear in AML Studio:

```bash
cp my_notebook.ipynb ~/cloudfiles/code/Users/<username>/
```

**From a local machine behind NSP / private endpoint** — the SMB share is not
directly accessible. Use blob storage or the AML Studio UI instead:

```bash
az ml data create --name my-dataset \
  --path ./local-data/ \
  --type uri_folder \
  --workspace-name <workspace> \
  --resource-group <resource-group>
```

---

## 3. Activate the Conda Environment

The `sft-notebook` environment is installed during CI provisioning.

```bash
source /anaconda/etc/profile.d/conda.sh
conda env list               # verify sft-notebook exists
conda activate sft-notebook
```

### Verify GPU access

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

> If `torch.cuda.is_available()` returns `False`, ensure `cuda-compat-12-6` is
> installed. See [03-Custom-Env-Setup.md](03-Custom-Env-Setup.md) for the CUDA
> forward-compatibility fix.

### Select the Jupyter kernel

The kernel **sft-notebook** is registered during provisioning. Select it from the
kernel picker in JupyterLab or VS Code.

If it does not appear, re-register:

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook
python -m ipykernel install --user --name sft-notebook --display-name "sft-notebook"
```

---

## 4. Manage CI Lifecycle

### Check status

```bash
az ml compute show --name <ci-name> \
  --workspace-name <workspace> \
  --resource-group <resource-group> \
  --query "{name:name, state:state, vmSize:size}" -o table
```

Or use `aml_ci_status.ipynb`.

### List all Compute Instances

```bash
az ml compute list \
  --workspace-name <workspace> \
  --resource-group <resource-group> \
  --type ComputeInstance -o table
```

### Stop / Start / Delete

| Action | Python SDK | CLI |
|--------|-----------|-----|
| **Stop** | `ml_client.compute.begin_stop("<ci-name>").wait()` | `az ml compute stop --name <ci-name> ...` |
| **Start** | `ml_client.compute.begin_start("<ci-name>").wait()` | `az ml compute start --name <ci-name> ...` |
| **Delete** | `ml_client.compute.begin_delete("<ci-name>").wait()` | `az ml compute delete --name <ci-name> ... --yes` |

> **Auto-shutdown:** The CI is configured with `idle_time_before_shutdown: PT15M`
> — it stops automatically after 15 minutes of inactivity.

> **Cost tip:** Stopping deallocates the VM (no compute charges) but preserves the
> OS disk and conda environment. Delete when you no longer need it.

---

## 5. Debug Provisioning Issues

If the custom conda environment setup failed:

1. **AML Studio → Compute →** click the CI **→ Boot Diagnostics → Serial log**
2. Or SSH in and inspect the logs:

```bash
cat /var/log/aml_custom_setup/std_log.txt 2>/dev/null
ls /var/log/cloud-init*.log
```

---

## 6. Install Additional Packages

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook

pip install <package-name>
# or
conda install -c conda-forge <package-name>
```

> Packages persist across stop/start cycles but are **lost on delete/recreate**.
> To make them permanent, add them to `environments/slm_conda_sft.yaml` and
> reprovision.

---

## Useful Commands

Show the SSH public key on Windows:

```powershell
Get-Content C:\Users\<user_alias>\.ssh\id_rsa_xxx.pub
```

---

## Reference

- [03-Custom-Env-Setup.md](03-Custom-Env-Setup.md) — Conda environment and CUDA compatibility details
- [01-Quick-Start.md](01-Quick-Start.md) — Local development environment setup
- [Custom Conda Env on CI (example script)](https://github.com/Azure/azureml-examples/blob/main/setup/setup-ci/setup-custom-conda-env.sh)
- [AML Compute Instance — Concepts](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)
- [Create Compute Instance (Python SDK)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-compute-instance?view=azureml-api-2&tabs=python)
- [Customize Compute Instance (startup scripts)](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-customize-compute-instance?view=azureml-api-2#create-the-setup-script)
- [Manage Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-compute-instance)
- [Customize Compute Instance](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-customize-compute-instance?view=azureml-api-2)
