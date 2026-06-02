"""SFT fine-tune Llama-3 with QLoRA (bitsandbytes 4-bit NF4) + PEFT + TRL.

Consumes a pre-formatted dataset from disk (produced by `prepare_dataset.py`
on a cheap CPU job and registered as an AML Data asset). The GPU job
therefore does no HF download and no tokenizer.map() — pure training.

Mounted at `--data_dir`, the dataset folder is expected to contain `train/`
and `test/` subfolders written via `datasets.DatasetDict.save_to_disk`.

Saves to ./outputs/ which AML uploads as job outputs automatically:
  outputs/<finetuned_model>/         LoRA adapter + tokenizer
  outputs/<finetuned_model>_config/  LoRA config
  outputs/<finetuned_model>_full/    Merged full model + tokenizer
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import mlflow
import torch
from accelerate import Accelerator
from datasets import load_from_disk
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# ───────────────────────────────────────────────────────────────────────────
# Memory / numerics knobs
# ───────────────────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# AML MLflow rejects param values > 500 chars — truncate before logging.
_AML_MAX_PARAM_LEN = 500
if not getattr(mlflow.log_params, "_is_truncated_patch", False):
    _orig_log_params = mlflow.log_params

    def _truncated_log_params(params, **kwargs):
        out = {}
        for k, v in params.items():
            s = str(v)
            out[k] = s[:_AML_MAX_PARAM_LEN] if len(s) > _AML_MAX_PARAM_LEN else v
        return _orig_log_params(out, **kwargs)

    _truncated_log_params._is_truncated_patch = True
    mlflow.log_params = _truncated_log_params


def load_prepared_dataset(data_dir: str):
    """Load DatasetDict written by prepare_dataset.py:save_to_disk."""
    print(f"Loading prepared dataset from: {data_dir}")
    ds = load_from_disk(data_dir)
    print(f"Splits     : {list(ds.keys())}")
    print(f"Train rows : {len(ds['train'])}")
    print(f"Test  rows : {len(ds['test'])}")
    return ds


def do_training(base_model: str, dataset, tokenizer, finetuned_model: str,
                hf_cache: str, output_dir: str,
                num_epochs: int, max_steps: int, batch_size: int,
                grad_accum: int, learning_rate: float,
                lora_r: int, lora_alpha: int, lora_dropout: float):

    use_bf16 = False
    if torch.cuda.is_available():
        try:
            major_cc, _ = torch.cuda.get_device_capability(0)
            use_bf16 = major_cc >= 8
        except Exception:
            use_bf16 = False

    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if torch.cuda.is_available():
        device_map = {"": Accelerator().process_index}
    else:
        device_map = "auto"
        qlora_config = None

    print(f"Loading model: {base_model}  (bf16={use_bf16}, 4-bit={qlora_config is not None})")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=qlora_config,
        device_map=device_map,
        attn_implementation="eager",
        cache_dir=hf_cache,
    )

    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    # Llama-3 LoRA targets: all attention + MLP projections.
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, finetuned_model),
        per_device_train_batch_size=max(1, batch_size),
        per_device_eval_batch_size=max(1, batch_size),
        gradient_accumulation_steps=grad_accum,
        optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=learning_rate,
        fp16=False,
        bf16=use_bf16,
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="mlflow",
        dataloader_pin_memory=False,
        save_steps=500,
        save_total_limit=2,
        eval_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        disable_tqdm=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training ...")
    trainer.train()

    adapter_path = os.path.join(output_dir, finetuned_model)
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA adapter + tokenizer → {adapter_path}")

    config_path = os.path.join(output_dir, f"{finetuned_model}_config")
    peft_config.save_pretrained(config_path)
    print(f"LoRA config              → {config_path}")

    full_path = os.path.join(output_dir, f"{finetuned_model}_full")
    model.merge_and_unload().save_pretrained(full_path)
    tokenizer.save_pretrained(full_path)
    print(f"Merged full model        → {full_path}")

    return model, tokenizer, trainer


def quick_inference(model, tokenizer):
    messages = [{
        "role": "user",
        "content": ("Hello doctor, I get red blotches on my skin whenever "
                    "I'm next to a cat. What can I do?"),
    }]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


if __name__ == "__main__":
    print(f"Arguments: {sys.argv}")
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="Mounted folder produced by prepare_dataset.py")
    p.add_argument("--base_model", default="NousResearch/Meta-Llama-3-8B-Instruct")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps",  type=int, default=-1,
                   help="-1 = honour num_epochs; small int for quick smoke test")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--lora_r",       type=int, default=8)
    p.add_argument("--lora_alpha",   type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--finetuned_model", default="llama3-8b-chat-doctor")
    p.add_argument("--hf_cache",   default="/mounts/models",
                   help="HF cache for the base model only (dataset is pre-built)")
    p.add_argument("--output_dir", default="./outputs",
                   help="AML uploads ./outputs/ automatically after the job")
    args = p.parse_args()

    print(f"Script path: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"data_dir   : {args.data_dir}")
    print(f"hf_cache   : {args.hf_cache}")
    print(f"output_dir : {args.output_dir}")
    os.makedirs(args.hf_cache,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer for the SFTTrainer (uses base_model's tokenizer; must match
    # the tokenizer used by prepare_dataset.py to build the chat template).
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.hf_cache)
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_prepared_dataset(args.data_dir)

    model, tokenizer, trainer = do_training(
        base_model=args.base_model,
        dataset=dataset,
        tokenizer=tokenizer,
        finetuned_model=args.finetuned_model,
        hf_cache=args.hf_cache,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    if Accelerator().process_index == 0:
        print("\n──── sample generation ────")
        print(quick_inference(model, tokenizer))
