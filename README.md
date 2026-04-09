# Model Fine-Tuning

Hands-on tutorials for fine-tuning language models on Azure and local hardware.

> Disclaimer: This is a learning/sample artifact — not production hardened.

## 📚 Contents

- [Model Fine-Tuning](#model-fine-tuning)
  - [📚 Contents](#-contents)
  - [📝 Overview](#-overview)
  - [🚀 Get Started](#-get-started)
  - [📦 Tutorials](#-tutorials)
    - [1. Azure Compute Instance (Phi SLM)](#1-azure-compute-instance-phi-slm)
    - [2. Azure Compute Cluster (LLama)](#2-azure-compute-cluster-llama)
    - [3. Foundry Serverless (Phi)](#3-foundry-serverless-phi)
    - [4. Local MPS Accelerator (LLama)](#4-local-mps-accelerator-llama)
  - [🤝 Author](#-author)
  - [📄 License](#-license)

## 📝 Overview

| Tutorial | Model | Compute | Status |
|----------|-------|---------|--------|
| 1. Azure Compute Instance | Phi-4-mini-instruct | 1× T4 GPU | **Available** |
| 2. Azure Compute Cluster | LLama | Multi-node | Coming soon |
| 3. Foundry Serverless | Phi | Serverless | Coming soon |
| 4. Local MPS Accelerator | LLama | Apple Silicon | Coming soon |

## 🚀 Get Started

- **[Tutorial](index.md)** — Step-by-step guide from workspace setup to model deployment
- **[Technical Background](TechBackGround.md)** — References, model availability, and related resources

## 📦 Tutorials

### 1. Azure Compute Instance (Phi SLM)

Fine-tune **Phi-4-mini-instruct** for under **$3** in about **2 hours** on a single T4 GPU in Azure.

No massive clusters. No expensive API calls. Just QLoRA, a small dataset, and an AML Compute Instance.

**What You Will Learn**

- Provision and manage **Azure Machine Learning GPU Compute Instances** with the Python SDK
- Set up a custom **conda environment** with an inline creation script (no file-share dependency)
- Handle **CUDA forward compatibility** (`cuda-compat-12-6`) on older GPU drivers
- Fine-tune a 3.8B-parameter SLM using **QLoRA** (4-bit NF4 quantisation + LoRA adapters)
- Track experiments with **MLflow** integrated into AML
- Register and deploy models through the **AML Model Registry**

**Tech Stack**

| Category | Library / Tool | Version |
|----------|---------------|---------|
| Deep Learning | `torch` | 2.11.0+cu126 |
| Model Hub | `transformers` | 5.5.0 |
| Quantisation | `bitsandbytes` | 0.49.2 |
| LoRA / QLoRA | `peft` | 0.18.1 |
| SFT Trainer | `trl` | 1.0.0 |
| Distributed / Mixed Precision | `accelerate` | 1.13.0 |
| Dataset Loading | `datasets` | 4.8.4 |
| Experiment Tracking | `mlflow` + `azureml-mlflow` | 3.9+ / 1.62.0 |
| AML SDK | `azure-ai-ml` | 1.32.0 |
| GPU Helpers | `applyllm` | 0.0.10 |

➡️ [Go to tutorial](01-compute-instance/README.md)

### 2. Azure Compute Cluster (LLama)

*Coming soon.* — Fine-tune LLama models on Azure Machine Learning Compute Clusters with multi-node distributed training.

### 3. Foundry Serverless (Phi)

*Coming soon.* — Serverless fine-tuning using Azure AI Foundry's managed compute with no infrastructure to manage.

### 4. Local MPS Accelerator (LLama)

*Coming soon.* — Fine-tune models locally on Apple Silicon using the Metal Performance Shaders (MPS) backend.

## 🤝 Author

Yingding Wang

## 📄 License

See [LICENSE](LICENSE).