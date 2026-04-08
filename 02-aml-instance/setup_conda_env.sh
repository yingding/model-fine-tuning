#!/bin/bash
set -e
ENV_NAME="${ENV_NAME:-sft-notebook}"

# 1. Remove broken tensorflow-serving-apt repo (403) and update apt
sudo rm -f /etc/apt/sources.list.d/*tensorflow*
sudo sed -i '/tensorflow-serving-apt\|storage.googleapis.com.*tensorflow/d' /etc/apt/sources.list 2>/dev/null || true
sudo apt-get update -y || true

# 2. Install CUDA 12.6 forward-compat for driver 535.274.02 cuda 12.2 (no driver removal/reboot needed)
sudo apt-get install -y cuda-compat-12-6

# 3. Create conda env and register Jupyter kernel
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