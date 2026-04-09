# 08 — Deploy and Inference

After evaluating the fine-tuned model, register it in the AML Model Registry for
versioning, sharing, and deployment.

The corresponding notebook is **`aml_ci_finetung_phi.ipynb`**, sections 10–11.

---

## 1. Register the Model in AML

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

FINETUNED_MODEL = "phi-4-mini-instruct-finetuned"
full_model_path = f"./{FINETUNED_MODEL}/full"

registered_model = ml_client.models.create_or_update(
    Model(
        path=full_model_path,
        name=FINETUNED_MODEL,
        type=AssetTypes.CUSTOM_MODEL,
        description="Phi-4-mini-instruct fine-tuned on medical Q&A with QLoRA",
    )
)

print(f"Registered: {registered_model.name}, version: {registered_model.version}")
```

The model is now visible in **Azure ML Studio → Models**.

---

## 2. Deployment Options

Once registered, the model can be deployed as a real-time endpoint:

### Option A — AML Managed Online Endpoint

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="phi4-medical-endpoint",
    auth_mode="key",
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# Deploy
deployment = ManagedOnlineDeployment(
    name="phi4-medical-1",
    endpoint_name="phi4-medical-endpoint",
    model=f"azureml:{FINETUNED_MODEL}:{registered_model.version}",
    instance_type="Standard_NC4as_T4_v3",
    instance_count=1,
)
ml_client.online_deployments.begin_create_or_update(deployment).result()
```

### Option B — Local Inference on the CI

For testing, run inference directly on the Compute Instance without deploying:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

model = AutoModelForCausalLM.from_pretrained(
    full_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(full_model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "What are the common symptoms of type 2 diabetes?"},
]

output = pipe(messages)
print(output[0]["generated_text"][-1]["content"])
```

### Option C — Download and Run Locally

Download the registered model to your local machine:

```bash
az ml model download \
  --name phi-4-mini-instruct-finetuned \
  --version 1 \
  --download-path ./downloaded-model \
  --workspace-name <workspace> \
  --resource-group <resource-group>
```

---

## 3. Model Artifacts Summary

| Artifact | Description |
|----------|-------------|
| **LoRA adapter** | ~13 MB — lightweight, can be applied on top of any Phi-4-mini-instruct base |
| **Merged full model** | ~7.6 GB — standalone, no dependency on base model |
| **Registered model** | Versioned in AML Model Registry |

---

## 4. Key Takeaways

| Aspect | Detail |
|--------|--------|
| Base model | `microsoft/Phi-4-mini-instruct` (~3.8B params) |
| Technique | QLoRA (4-bit NF4 + LoRA r=8) |
| Trainable parameters | ~0.09% of total |
| Training hardware | 1× NVIDIA T4 (16 GB) |
| Training time | ~15–30 min for 1 epoch / 1000 samples |
| Dataset | `ruslanmv/ai-medical-chatbot` (1000 rows) |
| Tracking | MLflow integrated with AML |

---

## Clean Up

When you are done, stop or delete the Compute Instance to save costs:

```python
# Stop (can be restarted later)
ml_client.compute.begin_stop("<ci-name>").wait()

# Delete (permanent)
ml_client.compute.begin_delete("<ci-name>").wait()
```

---

Previous: [07 — Evaluate the Model](07-evaluate-model.md) |
[Reference](reference.md)
