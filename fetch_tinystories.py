#!/usr/bin/env python3
"""
Fetch roneneldan/TinyStories-3M (a small *real* causal LM) plus the GPT-Neo
tokenizer it uses, and collate them into one self-contained folder you can
upload for morse_stego's --model.

Run on a machine WITH Hugging Face access:

    pip install "transformers>=4.30" torch safetensors
    python fetch_tinystories.py

Produces ./tinystories_3m/ with weights + tokenizer, all files < 30 MB.
Then upload every file in that folder, and run:

    python morse_stego.py "secret message" --model /path/to/tinystories_3m
"""

import os
import sys

OUT = os.path.abspath("tinystories_3m")
MODEL = "roneneldan/TinyStories-3M"
TOKENIZER = "EleutherAI/gpt-neo-125M"   # TinyStories models use this tokenizer
LIMIT = 30 * 1024 * 1024


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"downloading {MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    print(f"downloading tokenizer {TOKENIZER} ...")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    os.makedirs(OUT, exist_ok=True)
    # safe_serialization=True writes model.safetensors (what --model expects first)
    model.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)

    # Quick sanity check: the model should produce logits for a prompt.
    ids = tok("The weather today is", return_tensors="pt")["input_ids"]
    import torch
    with torch.no_grad():
        logits = model(ids).logits
    print(f"\nsanity: vocab={model.config.vocab_size}, "
          f"logits shape={tuple(logits.shape)} (OK)\n")

    print(f"collated into {OUT}:")
    oversize = []
    for name in sorted(os.listdir(OUT)):
        path = os.path.join(OUT, name)
        size = os.path.getsize(path)
        flag = "  <-- OVER 30MB" if size > LIMIT else ""
        if size > LIMIT:
            oversize.append(name)
        print(f"  {size / 1e6:8.2f} MB  {name}{flag}")

    if oversize:
        print(f"\nWARNING: {oversize} exceed 30 MB; split with:")
        print("  split -b 28m tinystories_3m/model.safetensors "
              "tinystories_3m/model.safetensors.part-")
        sys.exit(1)
    print("\nAll files under 30 MB. Upload the whole folder.")


if __name__ == "__main__":
    main()
