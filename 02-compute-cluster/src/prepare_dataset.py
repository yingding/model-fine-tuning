"""CPU-side dataset preparation for Llama-3 SFT.

Downloads `ruslanmv/ai-medical-chatbot` (or any compatible Patient/Doctor
HF dataset), formats rows with the target tokenizer's chat template,
train/test-splits, and writes to `--output_dir` via `dataset.save_to_disk`.

Designed to run as a small AML command job on a CPU compute (no GPU needed
for tokenizer-only operations). The output directory is then registered
as a versioned AML Data asset by the orchestrating notebook.

Why this lives on a CPU box, not the GPU training job:
  - HF download is network-bound, not GPU-bound
  - `tokenizer.apply_chat_template` is CPU work
  - GPU compute is ~10x the cost per hour; doing prep there is wasteful
"""

from __future__ import annotations

import argparse
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer


def main() -> int:
    print(f"Arguments: {sys.argv}")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="ruslanmv/ai-medical-chatbot",
                   help="HF dataset id with 'Patient' and 'Doctor' columns")
    p.add_argument("--tokenizer_model", default="NousResearch/Meta-Llama-3-8B-Instruct",
                   help="Model whose chat template will format the conversations")
    p.add_argument("--num_data_rows", type=int, default=10000)
    p.add_argument("--eval_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=65)
    p.add_argument("--hf_cache", default=None,
                   help="Optional HF cache dir (defaults to HF default)")
    p.add_argument("--output_dir", required=True,
                   help="Where to write the save_to_disk() output (mounted by AML)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir : {args.output_dir}")
    print(f"Dataset    : {args.dataset_name}")
    print(f"Tokenizer  : {args.tokenizer_model}")

    # Load tokenizer for chat template only — no model weights downloaded.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model, cache_dir=args.hf_cache,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Download + subsample.
    ds = load_dataset(args.dataset_name, split="all", cache_dir=args.hf_cache)
    n = min(args.num_data_rows, len(ds))
    ds = ds.shuffle(seed=args.seed).select(range(n))
    print(f"Loaded {n} rows from {args.dataset_name}")

    def format_chat_template(row):
        messages = [
            {"role": "user",      "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]},
        ]
        row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return row

    ds = ds.map(format_chat_template, num_proc=4)

    # Split.
    splits = ds.train_test_split(test_size=args.eval_size, seed=args.seed)
    print(f"Train: {len(splits['train'])}   Eval: {len(splits['test'])}")

    # Save to disk — produces train/ and test/ subfolders (parquet shards).
    splits.save_to_disk(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")

    # Sanity-print one example.
    print("\n──── sample (train[0].text, first 400 chars) ────")
    print(splits["train"][0]["text"][:400])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
