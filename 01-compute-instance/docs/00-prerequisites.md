# 00 — Prerequisites

Before you begin, make sure you have the following tools and access in place.

---

## Azure Subscription

- An active [Azure subscription](https://azure.microsoft.com/free/).
- **Owner** or **Contributor** role on the resource group you will use.
- Sufficient quota for a GPU VM SKU (e.g. `Standard_NC4as_T4_v3`).

!!! tip "Check GPU quota"
    In the Azure Portal go to **Subscriptions → Usage + quotas** and filter for
    *NCasv3* family in your target region.

---

## Local Development Machine

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14+ | Local SDK notebooks |
| Azure CLI | latest | `az login` authentication |
| Git | latest | Clone the repository |
| SSH key pair | RSA | SSH into the Compute Instance |

### Install the Azure CLI

```bash
# macOS
brew install azure-cli

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

### Generate an SSH Key (if you don't have one)

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_aml -C "aml-compute-instance"
```

You will embed the **public key** (`~/.ssh/id_rsa_aml.pub`) when creating the
Compute Instance.

---

## Azure Authentication

Log in to Azure before running any notebook:

```bash
az login
```

The project code uses `DefaultAzureCredential`, which picks up your `az login`
session automatically. If that fails it falls back to
`InteractiveBrowserCredential`.

---

## Clone the Repository

```bash
git clone https://github.com/yingding/model-fine-tuning.git
cd model-fine-tuning/01-compute-instance
```

---

## Create the Local Python Environment

```bash
python3.14 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .\.venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install -r deploy_requirements.txt

# Register a Jupyter kernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

The `deploy_requirements.txt` installs only the packages needed to **create and
manage** the Compute Instance from your local machine:

```text
azure-ai-ml==1.32.0
azure-identity==1.25.3
ipykernel==7.2.0
pydantic==2.12.5
pydantic_settings==2.13.1
python-dotenv==1.2.2
azure-mgmt-storage==24.0.1
```

---

## Environment Configuration

A `.env.example` template is provided under `config/`. Copy it to create your
own `.env` file:

```bash
cp config/.env.example config/.env
```

Then edit `config/.env` with your values:

```env
# Enter details of your AML workspace
SUBSCRIPTION_ID = "<SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<RESOURCE_GROUP>"
WORKSPACE = "<AML_WORKSPACE_NAME>"
SSH_PUB_KEY_NAME = "id_rsa.pub"
```

!!! warning "Do not commit `.env` files"
    The `.env` file contains your real credentials. Only the `.env.example`
    template should be checked into version control.

Region-specific example files (`swedencentral.env`, `germanywest.env`) are also
provided as references.

These values are loaded by `utils/amlauth.py` via **pydantic-settings**.

---

## Summary

After completing these steps you should have:

- [x] An Azure subscription with GPU quota
- [x] `az login` authenticated
- [x] The repo cloned and the local `.venv` activated
- [x] A `config/.env` file (copied from `.env.example`) pointing at your AML Workspace
- [x] An SSH key pair ready

Next: [01 — Create AML Workspace](01-create-aml-workspace.md)
