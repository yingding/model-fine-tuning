# Quick Start

Set up a local Python environment to provision and manage an AML GPU Compute Cluster for Llama3 fine-tuning.

> Disclaimer: This is a learning/sample artifact — not production hardened.

---

## Prerequisites

- Python 3.14+
- Azure CLI (`az login` authenticated)
- An SSH key pair (see [docs/00-prerequisites.md](docs/00-prerequisites.md))

---

## 1. Create Local Environment

### Linux / macOS

```bash
# 1. Navigate to the compute cluster directory
cd model-fine-tuning/02-compute-cluster

# 2. Create venv, activate, and install dependencies
python3.14 -m venv .venv && source .venv/bin/activate
python3.14 -m pip install --upgrade pip
python3.14 -m pip install -r deploy_requirements.txt

# 3. Register a Jupyter kernel
python3.14 -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

### Windows (PowerShell)

```powershell
# 1. Navigate to the compute cluster directory
cd model-fine-tuning\02-compute-cluster

# 2. Create venv, activate, and install dependencies
python3.14 -m venv .venv
.\.venv\Scripts\activate
python3.14 -m pip install --upgrade pip
python3.14 -m pip install -r deploy_requirements.txt

# 3. Register a Jupyter kernel
python3.14 -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

---

## 2. Configure Environment

```bash
cp config/.env.example config/.env
```

Edit `config/.env` with your Azure subscription, resource group, workspace, and SSH key name.

---

## 3. Run the Notebooks

Open the notebooks in VS Code or JupyterLab using the **Python (.venv)** kernel:

| Notebook | Purpose |
|----------|---------|
| `aml_cc_create.ipynb` | Provision a GPU Compute Cluster with custom conda env |
| `aml_cc_status.ipynb` | Check Compute Cluster status |
| `aml_cc_finetung_llama3.ipynb` | Fine-tune Llama3 (run **on the CC**, not locally) |

### Workflow

```mermaid
flowchart LR
    subgraph Local["Local Machine"]
        A["aml_cc_create.ipynb"] --> B["aml_cc_status.ipynb"]
    end

    subgraph Azure["AML Compute Cluster (A100 GPU)"]
        C["sft-notebook conda env"]
        D["aml_cc_finetung_llama3.ipynb"]
        E["Fine-tuned Llama3 Model"]
    end

    A -- "provisions" --> C
    B -- "checks status" --> C
    Local -- "SSH / VS Code Web" --> D
    D -- "QLoRA + SFTTrainer" --> E
```

---

## License

See root [LICENSE](../LICENSE).