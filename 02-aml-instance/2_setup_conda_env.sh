#!/bin/bash
set -e

# set the name of the conda environment to create, default to "sft-notebook" if not set
ENV_NAME="${ENV_NAME:-sft-notebook}"

# This script creates a custom conda environment and kernel based on a sample yml file.
source /anaconda/etc/profile.d/conda.sh

# Accept Anaconda Terms of Service for non-interactive provisioning.
# Required when the 'defaults' channel (pkgs/main, pkgs/r) is listed in the conda env yaml.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create -f /tmp/aml-setup/slm_conda_sft.yaml

echo "Activating new conda environment: $ENV_NAME"
conda activate "$ENV_NAME"
conda install -y ipykernel

# Pass ENV_NAME explicitly — sudo -i starts a clean login shell that doesn't inherit env vars
sudo -u azureuser -i ENV_NAME="$ENV_NAME" bash << 'EOF'
echo "Installing kernel for env: $ENV_NAME"
source /anaconda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME"
echo "Conda environment setup successfully."
EOF