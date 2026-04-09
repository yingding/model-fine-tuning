# Fine-Tuning Phi-4 with AML Compute Instance

Fine-tune **Phi-4-mini-instruct** on a T4 GPU using QLoRA — under $3 in about 2 hours.

## 📚 Contents

- [Fine-Tuning Phi-4 with AML Compute Instance](#fine-tuning-phi-4-with-aml-compute-instance)
  - [📚 Contents](#-contents)
  - [📝 Overview](#-overview)
  - [🚀 Quick Start](#-quick-start)
  - [📖 Documentation](#-documentation)
  - [📓 Tutorial](#-tutorial)
  - [📄 License](#-license)

## 📝 Overview

This directory contains everything needed to fine-tune Phi-4-mini-instruct on an
Azure Machine Learning GPU Compute Instance:

- Notebooks to **provision**, **monitor**, and **fine-tune** on a T4 GPU
- An inline creation script that installs a custom **conda environment** with CUDA compatibility
- Configuration templates and utility modules for AML authentication

## 🚀 Quick Start

See [01-Quick-Start.md](01-Quick-Start.md) for the full setup guide.

```bash
cd model-fine-tuning/01-compute-instance
python3.14 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r deploy_requirements.txt
cp config/.env.example config/.env   # edit with your values
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](01-Quick-Start.md) | Set up local environment and run the notebooks |
| [About Compute Instance](02-About-Compute-Instance.md) | Connect, storage, conda env, lifecycle management |
| [Custom Env Setup](03-Custom-Env-Setup.md) | Inline creation script, CUDA compat, troubleshooting |

## 📓 Tutorial

Step-by-step MkDocs tutorial pages under [`docs/`](docs/):

| Step | Page |
|------|------|
| 00 | [Prerequisites](docs/00-prerequisites.md) |
| 01 | [Create AML Workspace](docs/01-create-aml-workspace.md) |
| 02 | [Provision Compute Instance](docs/02-provision-compute-instance.md) |
| 03 | [Custom Conda Environment](docs/03-custom-conda-environment.md) |
| 04 | [Connect and Work with CI](docs/04-connect-and-work-with-ci.md) |
| 05 | [Prepare Training Data](docs/05-prepare-training-data.md) |
| 06 | [Fine-Tune Phi-4](docs/06-fine-tune-phi4.md) |
| 07 | [Evaluate the Model](docs/07-evaluate-model.md) |
| 08 | [Deploy and Inference](docs/08-deploy-and-inference.md) |
| — | [Reference](docs/reference.md) |

## 📄 License

See root [LICENSE](../LICENSE).