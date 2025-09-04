#!/usr/bin/env python3
"""
make_longbench_pairs.py

Generate TokenSIM “tokens” workload pairs from LongBench-v2:

  example.json → [
    [prompt_len_1, gen_len_1],
    [prompt_len_2, gen_len_2],
    …
  ]
"""

import json
from datasets import load_dataset
from transformers import AutoTokenizer
from TokenSim.utils import get_generation_lens

def main():
    # ——— CONFIGURATION ———
    model_name    = "mistralai/Mistral-7B-v0.3"   # or "mistralai/Mistral-7B", "internlm/internlm2-7b"
    split         = "train"                 # LongBench-v2 only publishes a train split (503 examples)
    output_file   = "dataset/mistral7b-longbench.json"
    gen_distribution = "uniform"            # "uniform" | "exponential" | "capped_exponential" | "burst"
    gen_mean      = 100                     # mean # tokens to simulate generating
    gen_range     = 50                      # +/- range around gen_mean
    # —————————————————————

    # 1) Load LongBench-v2
    ds = load_dataset("THUDM/LongBench-v2", split=split)
    print(f"Loaded {len(ds)} examples from LongBench-v2 split='{split}'")

    # 2) Load the model’s tokenizer (so counts match exactly what the model sees)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Some tokenizers (like Llama’s) may not have a pad token
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    # 3) Tokenize each context → prompt_lens
    prompt_lens = []
    for idx, ex in enumerate(ds):
        ct = ex["context"]
        tokens = tok.encode(ct, add_special_tokens=False)
        prompt_lens.append(len(tokens))
        if (idx + 1) % 50 == 0 or idx == len(ds)-1:
            print(f"  → processed {idx+1}/{len(ds)} contexts")

    # 4) Generate synthetic generation lengths
    gen_lens = get_generation_lens(
        distribution=gen_distribution,
        len_mean=gen_mean,
        len_range=gen_range,
        num_prompt=len(prompt_lens),
    )

    # 5) Zip into [[prompt_len, gen_len], …] and write JSON
    pairs = [[p, g] for p, g in zip(prompt_lens, gen_lens)]
    with open(output_file, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Wrote {len(pairs)} prompt/gen pairs → {output_file}")

if __name__ == "__main__":
    main()

