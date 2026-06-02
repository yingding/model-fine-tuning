# NGC PyTorch 26.05: torch 2.12.0a0 + CUDA 13.2.1 + Python 3.12.
# Also ships: cuDNN, NCCL, apex, transformer-engine, deepspeed (NVIDIA-tested),
# flash-attn, triton. We layer only the SFT-specific HF stack on top.
#
# Release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-05.html
# Requires NVIDIA driver R580+ on the AML compute cluster host (CUDA 13.x).
# If the cluster driver is older, swap FROM to nvcr.io/nvidia/pytorch:25.06-py3
# (torch 2.8 + CUDA 12.9) or nvcr.io/nvidia/pytorch:24.12-py3 (torch 2.6 + CUDA 12.6).
FROM nvcr.io/nvidia/pytorch:26.05-py3

# bitsandbytes 0.49.2 ships prebuilt kernels for cu11.8 through cu13.0.
# NGC pytorch:26.05 ships CUDA 13.2 (libcudart.so.13) — NOT .so.12 — so we
# explicitly tell bnb to load its cu130 binary, which is linked against
# libcudart.so.13 (already on disk). Without this, bnb would try its
# default (matching torch.version.cuda) which is cu132 — a build that
# doesn't exist in this wheel, falling back to cu126 which needs .so.12.
#
# Diagnostic confirmed (smoke job patient_gyro_*):
#   /usr/local/cuda-13.2/lib/libcudart.so.13   ✅ present
#   libcudart.so.12                            ❌ not on system
#   .../bitsandbytes/libbitsandbytes_cuda130.so ✅ exists in wheel (3.5 MB)
ENV BNB_CUDA_VERSION=130

# NGC uses /etc/pip/constraint.txt to pin torch/triton/cuda/etc. so users don't
# accidentally overwrite the tested versions. We don't need to override any of
# those — just install the HF stack additively.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Deepspeed launcher needs passwordless ssh for multi-node; NGC base is bare
# Ubuntu under the hood.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server openssh-client \
    && rm -rf /var/lib/apt/lists/*
