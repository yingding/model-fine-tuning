# 05 — Prepare Training Data

This step loads and formats the training dataset for supervised fine-tuning (SFT)
of Phi-4-mini-instruct.

The corresponding notebook is **`aml_ci_finetung_phi.ipynb`**, sections 3–4.

---

## Dataset

We use the [ruslanmv/ai-medical-chatbot](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)
dataset from Hugging Face — a collection of medical Q&A conversations.

| Parameter | Value |
|-----------|-------|
| Dataset | `ruslanmv/ai-medical-chatbot` |
| Rows used | 1 000 (sampled) |
| Eval split | 10 % |
| Max sequence length | 512 tokens |

---

## Step 1 — Load the Tokenizer

```python
from transformers import AutoTokenizer

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

---

## Step 2 — Load and Split the Dataset

```python
from datasets import load_dataset

FINETUNE_DATASET = "ruslanmv/ai-medical-chatbot"
NUM_DATA_ROWS = 1000
EVAL_SIZE = 0.1

dataset = load_dataset(FINETUNE_DATASET, split="all")
dataset = dataset.shuffle(seed=42).select(range(NUM_DATA_ROWS))
dataset = dataset.train_test_split(test_size=EVAL_SIZE, seed=42)

print(f"Train: {len(dataset['train'])} rows")
print(f"Eval:  {len(dataset['test'])} rows")
```

---

## Step 3 — Format as Chat Messages

Phi-4-mini-instruct uses a chat template. We convert each row into the
`messages` format expected by the tokenizer:

```python
def format_chat_template(row):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful medical assistant. Answer the patient's "
                       "question based on your medical knowledge.",
        },
        {"role": "user", "content": row["Patient"]},
        {"role": "assistant", "content": row["Doctor"]},
    ]
    row["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
)
```

### Example Output

```text
<|system|>
You are a helpful medical assistant. Answer the patient's question based on your medical knowledge.
<|user|>
I have been experiencing headaches for the past week...
<|assistant|>
Based on your symptoms, I would recommend...
```

---

## Data Quality Tips

!!! tip "Quality over quantity"
    For QLoRA fine-tuning of SLMs, a **small, high-quality** dataset (500–2000
    examples) often outperforms a large noisy dataset. Focus on:

    - Consistent formatting
    - Accurate and complete answers
    - Diverse question types within your domain

!!! warning "Sequence length"
    Examples longer than `MAX_SEQ_LENGTH` (512) are truncated. If your domain
    requires longer responses, increase this — but be mindful of GPU memory
    on the T4 (16 GB).

---

## Custom Dataset Format

If you bring your own data, format it as a Hugging Face `Dataset` with a `text`
column containing the full chat-formatted string. You can also use JSONL files:

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Load with:

```python
dataset = load_dataset("json", data_files={"train": "train.jsonl", "test": "eval.jsonl"})
```

---

Next: [06 — Fine-Tune Phi-4](06-fine-tune-phi4.md)
