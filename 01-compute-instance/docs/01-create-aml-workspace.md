# 01 — Create an AML Workspace

An **Azure Machine Learning (AML) Workspace** is the top-level resource that
organises compute, data, experiments, and models. You need one before creating a
Compute Instance.

---

## Option A — Azure Portal

1. Go to the [Azure Portal](https://portal.azure.com).
2. Search for **Machine Learning** and click **+ Create**.
3. Fill in:
   - **Subscription** / **Resource group** — use (or create) the ones from your `config/.env` file.
   - **Workspace name** — e.g. `aml-workspace-yw-uno`.
   - **Region** — pick a region with GPU quota (e.g. *Sweden Central*).
4. Leave defaults for Storage, Key Vault, and Application Insights.
5. Click **Review + Create → Create**.

---

## Option B — Azure CLI

```bash
az ml workspace create \
  --name aml-workspace-yw-uno \
  --resource-group rg-aml-yw-uno \
  --location swedencentral
```

---

## Option C — Python SDK

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()

ws = Workspace(
    name="aml-workspace-yw-uno",
    location="swedencentral",
)

ml_client = MLClient(
    credential,
    subscription_id="<subscription-id>",
    resource_group_name="rg-aml-yw-uno",
)
ml_client.workspaces.begin_create_or_update(ws).result()
```

---

## Storage Account — Key Access vs Identity-Based Access

!!! warning "Compute Instance creation requires key-based storage access"
    At the time of writing, AML Compute Instances **do not** support
    identity-based storage account access during provisioning. If your workspace
    is configured for identity-based access you will see:

    > *"storage can't be connected, proxy error"*

    **Workaround:** Create the workspace with storage-account-key access first,
    provision the Compute Instance, then switch the workspace to identity-based
    access afterwards.

    - [GitHub Issue #7](https://github.com/Azure-Samples/ai-studio-in-a-box/issues/7#issuecomment-2273346002)
    - [FAQ / Workaround](https://github.com/Azure-Samples/ai-studio-in-a-box?tab=readme-ov-file#faq)

---

## Verify the Workspace

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="rg-aml-yw-uno",
    workspace_name="aml-workspace-yw-uno",
)

ws = ml_client.workspaces.get("aml-workspace-yw-uno")
print(f"Workspace: {ws.name}, Location: {ws.location}")
```

---

## Update Your `.env` File

If you haven't already, copy the template and fill in your values:

```bash
cp config/.env.example config/.env
```

Make sure `config/.env` reflects the workspace you just created:

```env
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP = "rg-aml-yw-uno"
WORKSPACE = "aml-workspace-yw-uno"
SSH_PUB_KEY_NAME = "id_rsa.pub"
```

---

Next: [02 — Provision Compute Instance](02-provision-compute-instance.md)
