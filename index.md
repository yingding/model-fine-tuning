# Model Fine-Tuning Tutorials

Step-by-step tutorials for fine-tuning language models on Azure and local hardware.

> **Disclaimer:** These are learning/sample artifacts — not production hardened.

---

## Tutorials

### 1. [Finetuning with Azure Compute Instance (Phi SLM)](01-compute-instance/docs/00-prerequisites.md)

Fine-tune **Phi-4-mini-instruct** on a T4 GPU Compute Instance using **QLoRA**.
Covers workspace setup, CI provisioning with custom conda environments,
supervised fine-tuning with SFTTrainer, MLflow tracking, and model registration.

| Aspect | Detail |
|--------|--------|
| Model | Phi-4-mini-instruct (~3.8B params) |
| Technique | QLoRA (4-bit NF4 + LoRA) |
| Hardware | NVIDIA T4 (16 GB) — `Standard_NC4as_T4_v3` |
| Tracking | MLflow → AML |

### 2. Finetuning with Azure Compute Cluster (LLama)

*Coming soon.*

### 3. Finetuning with Foundry Serverless

*Coming soon.*

### 4. Finetuning with Local MPS Accelerator

*Coming soon.*

---

## Repository Structure

```
01-compute-instance/   ← Phi-4 fine-tuning on AML Compute Instance
02-compute-cluster/    ← LLama fine-tuning on AML Compute Cluster (coming soon)
03-foundry-serverless/ ← Foundry serverless fine-tuning (coming soon)
04-local-mps/          ← Local Apple MPS fine-tuning (coming soon)
```
