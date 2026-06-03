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
    """Load DatasetDict written by prepare_dataset.py:save_to_disk.

    The dataset folder is typically mounted READ-ONLY by AML. TRL's
    SFTTrainer._prepare_dataset internally calls `dataset.map(...)` to add
    EOS tokens, and `datasets.map` writes its Arrow cache next to the source
    files — on a read-only mount that raises `OSError: [Errno 30] Read-only
    file system`. Workaround: materialize each split in-memory (small
    overhead for the curated medical-chatbot dataset, ~30 MB on disk).
    """
    print(f"Loading prepared dataset from: {data_dir}")
    ds = load_from_disk(data_dir)
    # Re-load each split into RAM so subsequent .map calls cache in memory,
    # not next to the read-only mount.
    for split in list(ds.keys()):
        ds[split] = ds[split].map(
            lambda x: x,
            keep_in_memory=True,
            load_from_cache_file=False,
            desc=f"materialize {split} in memory",
        )
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


def quick_inference(model, tokenizer, prompts=None):
    """Run a few sample prompts on the in-memory trained model.

    Returns a list of {prompt, response} dicts so the caller can log them
    as a job artifact (visible in the AML run's Outputs + logs).
    """
    if prompts is None:
        prompts = [
            ("Hello doctor, I get red blotches on my skin whenever I'm next "
             "to a cat. What can I do?"),
            ("I have been having severe headaches for the past three days, "
             "especially in the mornings. Should I be worried?"),
            ("My 5-year-old has a fever of 39°C and a sore throat. Is it "
             "safe to give ibuprofen?"),
        ]
    device = next(model.parameters()).device
    results = []
    for content in prompts:
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=200, num_return_sequences=1,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        results.append({"prompt": content, "response": response})
    return results


def register_artifacts(args, finetuned_model, output_dir, sample_outputs=None,
                       trainer=None):
    """Register adapter + merged model in the AML model registry via SDK v2.

    Matches the pattern used by 01-compute-instance/aml_ci_finetune_phi.ipynb:
    `ml_client.models.create_or_update(Model(path=..., name=..., type=...))`.

    Workspace coords inside an AML job are injected as env vars
    (AZUREML_ARM_SUBSCRIPTION / AZUREML_ARM_RESOURCEGROUP / AZUREML_ARM_WORKSPACE_NAME).
    Auth uses DefaultAzureCredential, which on the compute cluster picks up
    the cluster's system-assigned managed identity (Step 3 of the prep
    notebook grants it the required RBAC).
    """
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.constants import AssetTypes
        from azure.ai.ml.entities import Model
        from azure.identity import DefaultAzureCredential
    except ImportError as e:
        print(f"[register] azure-ai-ml not available ({e}) — skipping registry upload.")
        return

    sub_id  = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg_name = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws_name = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    if not (sub_id and rg_name and ws_name):
        print("[register] AZUREML_ARM_* env vars not set — likely a local/debug run; "
              "skipping registry upload.")
        return

    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=sub_id,
            resource_group_name=rg_name,
            workspace_name=ws_name,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[register] MLClient init failed: {e}")
        return

    # Common description block with training context.
    desc_base = (
        f"Fine-tuned {args.base_model} with QLoRA + SFTTrainer\n"
        f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}\n"
        f"Training: epochs={args.num_epochs}, max_steps={args.max_steps}, "
        f"lr={args.learning_rate}, batch={args.batch_size}x{args.grad_accum} grad_accum"
    )
    if trainer is not None:
        log_history = trainer.state.log_history
        train_logs = [l for l in log_history if "loss" in l]
        eval_logs  = [l for l in log_history if "eval_loss" in l]
        if train_logs:
            desc_base += f"\nFinal train loss: {train_logs[-1]['loss']:.4f}"
        if eval_logs:
            desc_base += f"\nFinal eval loss: {eval_logs[-1]['eval_loss']:.4f}"

    common_tags = {
        "base_model":     args.base_model,
        "lora_r":         str(args.lora_r),
        "lora_alpha":     str(args.lora_alpha),
        "lora_dropout":   str(args.lora_dropout),
        "max_seq_length": str(args.max_seq_length),
        "num_epochs":     str(args.num_epochs),
        "learning_rate":  str(args.learning_rate),
        "quantization":   "QLoRA-4bit-NF4",
    }

    # 1) LoRA adapter (small, reusable on top of compatible base models)
    adapter_dir = os.path.join(output_dir, finetuned_model)
    print(f"[register] Registering LoRA adapter: {adapter_dir}")
    try:
        registered_lora = ml_client.models.create_or_update(Model(
            path=adapter_dir,
            name=f"{finetuned_model}-lora",
            type=AssetTypes.CUSTOM_MODEL,
            description=f"{desc_base}\nKind: LoRA adapter + tokenizer",
            tags={**common_tags, "kind": "lora-adapter"},
        ))
        print(f"[register]   name={registered_lora.name}  version={registered_lora.version}")
    except Exception as e:  # noqa: BLE001
        print(f"[register]   ❌ adapter registration failed: {e}")

    # 2) Merged full model (deployable as-is)
    full_dir = os.path.join(output_dir, f"{finetuned_model}_full")
    print(f"[register] Registering merged full model: {full_dir}")
    try:
        registered_full = ml_client.models.create_or_update(Model(
            path=full_dir,
            name=f"{finetuned_model}-full",
            type=AssetTypes.CUSTOM_MODEL,
            description=f"{desc_base}\nKind: Merged full model + tokenizer",
            tags={**common_tags, "kind": "merged"},
        ))
        print(f"[register]   name={registered_full.name}  version={registered_full.version}")
    except Exception as e:  # noqa: BLE001
        print(f"[register]   ❌ merged-model registration failed: {e}")

    # 3) Persist sample inference results so they ship with job outputs.
    if sample_outputs:
        import json
        sample_path = os.path.join(output_dir, "sample_inference.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
        print(f"[register] Wrote sample inference results → {sample_path}")

    print("[register] Done.")


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
        sample_outputs = quick_inference(model, tokenizer)
        for i, item in enumerate(sample_outputs, 1):
            print(f"\n[{i}] PROMPT  : {item['prompt']}")
            print(f"[{i}] RESPONSE: {item['response']}")

        print("\n──── registering model artifacts ────")
        register_artifacts(
            args=args,
            finetuned_model=args.finetuned_model,
            output_dir=args.output_dir,
            sample_outputs=sample_outputs,
            trainer=trainer,
        )
