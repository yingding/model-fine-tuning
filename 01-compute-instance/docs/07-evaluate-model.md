# 07 — Evaluate the Model

After training, evaluate the fine-tuned model with test inference and review
the training metrics.

The corresponding notebook is **`aml_ci_finetung_phi.ipynb`**, sections 8–10.

---

## 1. Save the Fine-Tuned Model

### Save LoRA Adapter

```python
adapter_path = f"./{FINETUNED_MODEL}/adapter"
trainer.model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")
```

The adapter is **very small** (~13 MB) compared to the full model (~7.6 GB).

### Merge and Save Full Model

For deployment, merge the LoRA weights back into the base model:

```python
from peft import AutoPeftModelForCausalLM

# Reload adapter in FP16 for merging
merged_model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Merge LoRA weights into base model
merged_model = merged_model.merge_and_unload()

# Save merged model
full_model_path = f"./{FINETUNED_MODEL}/full"
merged_model.save_pretrained(full_model_path, safe_serialization=True)
tokenizer.save_pretrained(full_model_path)
print(f"Full merged model saved to {full_model_path}")
```

### Saved Artifacts

| Artifact | Path | Size |
|----------|------|------|
| LoRA adapter | `./phi-4-mini-instruct-finetuned/adapter/` | ~13 MB |
| Merged full model | `./phi-4-mini-instruct-finetuned/full/` | ~7.6 GB |
| Tokenizer + config | Included in both paths | ~1 MB |

---

## 2. Test Inference

Load the fine-tuned model and run a sample prompt:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = f"./{FINETUNED_MODEL}/full"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Format a test prompt
messages = [
    {
        "role": "system",
        "content": "You are a helpful medical assistant.",
    },
    {
        "role": "user",
        "content": "I have been experiencing frequent headaches and dizziness. "
                   "What could be the cause?",
    },
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## 3. Review Training Metrics

### From the Trainer Log

```python
train_losses = [
    (e["step"], e["loss"])
    for e in trainer.state.log_history
    if "loss" in e
]
eval_losses = [
    (e["step"], e["eval_loss"])
    for e in trainer.state.log_history
    if "eval_loss" in e
]

print("Training Loss:")
for step, loss in train_losses:
    print(f"  Step {step}: {loss:.4f}")

print("\nEval Loss:")
for step, loss in eval_losses:
    print(f"  Step {step}: {loss:.4f}")
```

### From MLflow in AML Studio

1. Go to **Azure ML Studio** → **Jobs** → find your experiment.
2. Click on the run to see logged metrics: `train/loss`, `eval/loss`, learning
   rate, etc.
3. Compare runs side-by-side if you experiment with different hyperparameters.

---

## 4. Evaluation Checklist

- [ ] **Training loss** decreases over steps — model is learning.
- [ ] **Eval loss** decreases (or stabilises) — no overfitting.
- [ ] **Sample outputs** are coherent and relevant to the domain.
- [ ] **Response format** follows the expected chat template.
- [ ] **No regressions** on general knowledge (compare with base model).

!!! tip "Quick sanity check"
    Run 5–10 diverse prompts from your domain and compare outputs between the
    base model and the fine-tuned model. The fine-tuned version should give
    more domain-specific, accurate answers.

---

Next: [08 — Deploy and Inference](08-deploy-and-inference.md)
