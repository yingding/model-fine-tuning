# 03 — Custom Conda Environment

The creation script installs a fully configured conda environment called
**`sft-notebook`** on the Compute Instance at provisioning time. This page
explains what the environment contains and how to troubleshoot issues.

---

## Conda Environment Spec

The environment is defined in `environments/slm_conda_sft.yaml`:

```yaml
name: sft-notebook
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.14
  - pip>=24.2
  - ipywidgets>=8.1.8
  - ipykernel>=6.0
  - pip:
    - --extra-index-url https://download.pytorch.org/whl/cu126
    - torch==2.11.0+cu126
    - torchaudio==2.11.0+cu126
    - torchvision==0.26.0+cu126
    - transformers==5.5.0
    - huggingface-hub==1.9.2
    - datasets==4.8.4
    - mlflow>=3.9.0
    - azureml-mlflow==1.62.0.post2
    - azure-ai-ml==1.32.0
    - python-dotenv==1.2.2
    - pydantic-settings==2.13.1
    - pydantic==2.12.5
    - safetensors==0.7.0
    - accelerate==1.13.0
    - trl==1.0.0
    - peft==0.18.1
    - bitsandbytes==0.49.2
    - applyllm==0.0.10
```

Key packages and their roles:

| Package | Purpose |
|---------|---------|
| `torch` + `cu126` | PyTorch with CUDA 12.6 support |
| `transformers` | Hugging Face model loading & tokenizers |
| `peft` | Parameter-Efficient Fine-Tuning (LoRA / QLoRA) |
| `bitsandbytes` | 4-bit quantisation (NF4) |
| `trl` | `SFTTrainer` for supervised fine-tuning |
| `accelerate` | Mixed-precision and distributed training |
| `mlflow` + `azureml-mlflow` | Experiment tracking in AML |
| `datasets` | Hugging Face dataset loading |
| `applyllm` | GPU accelerator helpers |

---

## The Setup Script

`setup_conda_env.sh` runs as the **creation script** on the Compute Instance:

```bash
#!/bin/bash
set -e
ENV_NAME="${ENV_NAME:-sft-notebook}"

# 1. Fix broken apt repos and update
sudo rm -f /etc/apt/sources.list.d/*tensorflow*
sudo apt-get update -y || true

# 2. Install GitHub CLI (gh)
# ... (see setup_conda_env.sh for full commands)

# 3. Install CUDA 12.6 forward-compat shim
sudo apt-get install -y cuda-compat-12-6

# 4. Create conda env and register Jupyter kernel
source /anaconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda env create -f /tmp/aml-setup/slm_conda_sft.yaml
conda activate "$ENV_NAME"
conda install -y ipykernel

sudo -u azureuser -i ENV_NAME="$ENV_NAME" bash << 'EOF'
source /anaconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME"
EOF
```

---

## CUDA / PyTorch Compatibility

### The Problem

| Component | Value |
|-----------|-------|
| VM SKU | `Standard_NC4as_T4_v3` (T4 GPU) |
| Pre-installed NVIDIA driver | `535.274.02` |
| Driver CUDA version | **12.2** |

Since PyTorch 2.6.0, `cu121` wheels are no longer published. The minimum CUDA
variant is `cu124` or `cu126`:

| Torch version | Available CUDA variants |
|---------------|------------------------|
| 2.11.0 | cu126, cu128, cu130 |
| 2.8.0 | cu126, cu128, cu129 |
| 2.5.1 | cu118, **cu121**, cu124 |

### The Solution: `cuda-compat-12-6`

The **CUDA forward-compatibility shim** allows a newer CUDA toolkit to run on an
older driver without replacing the driver or rebooting:

```bash
sudo apt-get install -y cuda-compat-12-6
```

After installing `cuda-compat-12-6`, PyTorch `cu126` wheels work correctly on the
driver 535 stack.

---

## Troubleshooting

### Anaconda Terms of Service Error

```
CondaToSNonInteractiveError: Anaconda Terms of Service have not been accepted.
```

**Fix:** Accept the ToS before creating the environment:

```bash
source /anaconda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### Creation Script Timeout

The inline script timeout must be in `"<float>m"` format (e.g. `"25m"`), **not**
ISO 8601 (`"PT20M"`). Maximum value is `"25m"`.

### Kernel Not Visible in Jupyter

Re-register the kernel:

```bash
source /anaconda/etc/profile.d/conda.sh
conda activate sft-notebook
python -m ipykernel install --user --name sft-notebook --display-name "sft-notebook"
```

### View Creation Script Logs

```bash
cat /var/log/aml_custom_setup/std_log.txt 2>/dev/null
ls /var/log/cloud-init*.log
```

---

Next: [04 — Connect and Work with the CI](04-connect-and-work-with-ci.md)
