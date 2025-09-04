#!/usr/bin/env python3
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from TokenSim.utils import get_generation_lens  # make sure TokenSim is importable

def main():
    # 1) Load the LongBench-v2 “test” split
    ds = load_dataset("THUDM/LongBench-v2", split="train")
    print(f"Loaded {len(ds)} examples")

    # 2) Tokenizer for counting context tokens
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})

    # 3) Measure prompt lengths
    prompt_lens = []
    for ex in ds:
        ct = ex["context"]
        tokens = tok.encode(ct, add_special_tokens=False)
        prompt_lens.append(len(tokens))

    # 4) Generate “gen_len” for each example (uniform around 100±50 here)
    gen_lens = get_generation_lens(
        distribution="uniform",
        len_mean=100,
        len_range=50,
        num_prompt=len(prompt_lens),
    )

    # 5) Zip into pairs and dump
    pairs = [[p, g] for p, g in zip(prompt_lens, gen_lens)]
    with open("example.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Wrote {len(pairs)} pairs → example.json")

if __name__ == "__main__":
    main()

