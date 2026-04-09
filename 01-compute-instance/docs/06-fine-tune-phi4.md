# 06 — Fine-Tune Phi-4

This is the core step: fine-tuning **Phi-4-mini-instruct** using **QLoRA** on a
T4 GPU. The corresponding notebook is **`aml_ci_finetung_phi.ipynb`**.

---

## Overview

| Technique | Description |
|-----------|-------------|
| **QLoRA** | 4-bit quantised base model + LoRA adapters (trainable) |
| **SFTTrainer** | Supervised fine-tuning trainer from the `trl` library |
| **MLflow** | Experiment tracking integrated with AML |

---

## 1. Environment Setup & GPU Check

```python
from applyllm.accelerators import AcceleratorHelper, DirectorySetting

aml_dir_setting = DirectorySetting(
    home_dir="/home/azureuser",
    transformers_cache_home="localfiles/models",
    huggingface_token_file="localfiles/models/.huggingface_token",
)

uuids = AcceleratorHelper.nvidia_device_uuids_filtered_by(is_mig=False)
AcceleratorHelper.init_torch_env(
    accelerator="cuda",
    dir_setting=aml_dir_setting,
    uuids=uuids,
)
```

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"GPU:             {torch.cuda.get_device_name(0)}")
print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Enable TF32 on Ampere+ GPUs for faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
```

---

## 2. Authentication & MLflow

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
```

---

## 3. Training Parameters

```python
# Model
BASE_MODEL = "microsoft/Phi-4-mini-instruct"
FINETUNED_MODEL = "phi-4-mini-instruct-finetuned"

# Data
FINETUNE_DATASET = "ruslanmv/ai-medical-chatbot"
NUM_DATA_ROWS = 1000
EVAL_SIZE = 0.1

# Training
NUM_EPOCHS = 1
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

# LoRA
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

---

## 4. Load Model with 4-bit Quantisation

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
```

!!! info "Why 4-bit quantisation?"
    Phi-4-mini-instruct has ~3.8B parameters. In full precision (FP16) it
    requires ~7.6 GB just for weights. 4-bit NF4 quantisation reduces this to
    ~2 GB, leaving room for activations, gradients, and the optimizer on the
    T4's 16 GB VRAM.

---

## 5. Configure LoRA

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing

lora_config = LoraConfig(
    r=LORA_R,                    # Rank
    lora_alpha=LORA_ALPHA,       # Scaling factor
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
```

Expected output:

```
trainable params: 3,407,872 || all params: 3,840,xxx,xxx || trainable%: 0.089%
```

Only **~0.09%** of parameters are trainable — the rest are frozen in 4-bit.

---

## 6. Training with SFTTrainer

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=f"./{FINETUNED_MODEL}",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="mlflow",
    run_name=FINETUNED_MODEL,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

trainer.train()
```

### Key Training Arguments

| Argument | Value | Notes |
|----------|-------|-------|
| `bf16=True` | BFloat16 | Mixed precision — saves memory |
| `optim="paged_adamw_8bit"` | 8-bit Adam | Further memory savings |
| `gradient_accumulation_steps=4` | Effective batch size 4 | Compensates for small per-device batch |
| `lr_scheduler_type="cosine"` | Cosine decay | Standard for fine-tuning |

---

## 7. Monitor Training

Training metrics are logged to **MLflow** and visible in AML Studio under
**Jobs → Experiments**.

You can also print the loss trend from the trainer:

```python
for entry in trainer.state.log_history:
    if "loss" in entry:
        print(f"Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")
    if "eval_loss" in entry:
        print(f"Step {entry.get('step', '?')}: eval_loss={entry['eval_loss']:.4f}")
```

---

Next: [07 — Evaluate the Model](07-evaluate-model.md)
