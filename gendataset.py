#!/usr/bin/env python3
"""
gendataset.py

Unified script to generate TokenSim prompt–generation length pairs for various corpora:
- LongBench-v2
- Needle-in-a-Haystack (OpenCompass/NeedleBench)
- ShareGPT (RyokoAI/ShareGPT52K) via custom loader
- BookCorpus (with trust_remote_code)
- Wikipedia (multi-config with slicing)

Handles non-string and nested types by casting and flattening before tokenization.
"""
import os
import sys
import json
import glob
import argparse
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from TokenSim.utils import get_generation_lens


def load_sharegpt_conversations():
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = os.path.join(cache_root, "datasets--RyokoAI--ShareGPT52K*", "**", "*.json")
    contexts = []
    for path in glob.glob(pattern, recursive=True):
        try:
            data = json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            continue
        for ex in data:
            msg = ex.get('conversations') or ex.get('messages') or ex.get('conversation')
            # flatten list inputs
            if isinstance(msg, list):
                texts = []
                for m in msg:
                    if isinstance(m, dict):
                        texts.append(m.get('value') or m.get('text') or str(m))
                    else:
                        texts.append(str(m))
                contexts.append(" ".join(texts))
            else:
                contexts.append(str(msg))
    return contexts


def main():
    parser = argparse.ArgumentParser(description="Generate TokenSim prompt/gen pairs.")
    parser.add_argument("--model",   required=True, help="HF model ID for tokenizer counts")
    parser.add_argument("--dataset", required=True, help="HF dataset ID or local path (CSV/JSON)")
    parser.add_argument("--config",  default=None,   help="Config name for multi-config HF datasets")
    parser.add_argument("--split",   default="train", help="Split or slice (e.g. train[:1%])")
    parser.add_argument("--field",   required=True, help="Column or key containing text contexts")
    parser.add_argument("--local",   default=None,  help="Local file path instead of HF dataset")
    parser.add_argument("--out",     default="pairs.json", help="Output JSON path for pairs")
    parser.add_argument("--dist",    default="uniform", choices=["uniform","exponential","capped_exponential","burst"], help="Distribution for gen lengths")
    parser.add_argument("--mean",    type=int, default=100, help="Mean gen length to sample")
    parser.add_argument("--rng",     type=int, default=50,  help="Range around mean for sampling")
    args = parser.parse_args()

    # 1) Load contexts
    print(f"\n>> Loading {args.dataset} (config={args.config}, split={args.split})")
    if args.dataset == "RyokoAI/ShareGPT52K":
        contexts = load_sharegpt_conversations()
    elif args.local:
        import pandas as pd
        ext = args.local.split('.')[-1].lower()
        df = pd.read_json(args.local) if ext == 'json' else pd.read_csv(args.local)
        contexts = df[args.field].astype(str).tolist()
    else:
        load_kwargs = {}
        # BookCorpus requires trust_remote_code
        if args.dataset.lower() == 'bookcorpus':
            load_kwargs['trust_remote_code'] = True
        # streaming flag can be added here if desired
        if args.config:
            raw = load_dataset(args.dataset, args.config, **load_kwargs)
        else:
            raw = load_dataset(args.dataset, split=args.split, **load_kwargs)
        if isinstance(raw, DatasetDict):
            ds = raw[args.split] if args.split in raw else next(iter(raw.values()))
        elif isinstance(raw, Dataset):
            ds = raw
        else:
            raise ValueError(f"Unexpected dataset type: {type(raw)}")
        contexts = ds[args.field]
    print(f"✅ Loaded {len(contexts)} contexts")

    # 2) Load tokenizer
    print(f"\n>> Loading tokenizer for {args.model}...")
    tok_kwargs = {"use_fast": True}
    if "internlm" in args.model:
        tok_kwargs["trust_remote_code"] = True
    if "llama" in args.model.lower():
        tok_kwargs["legacy"] = False
    tok = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    print("✅ Tokenizer loaded")

    # 3) Tokenize contexts
    print("\n>> Tokenizing contexts...")
    prompt_lens = []
    for idx, ct in enumerate(contexts, start=1):
        # ensure a string input
        text = ct if isinstance(ct, str) else str(ct)
        try:
            length = len(tok.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f"❌ Failed to tokenize item {idx} (type={type(ct)}): {e}")
            length = 0
        prompt_lens.append(length)
        if idx % 50 == 0 or idx == len(contexts):
            print(f"   • {idx}/{len(contexts)}")
    print("✅ Tokenization complete")

    # 4) Sample generation lengths
    print(f"\n>> Sampling generation lengths (dist={args.dist}, mean={args.mean}, rng={args.rng})")
    gen_lens = get_generation_lens(distribution=args.dist, len_mean=args.mean, len_range=args.rng, num_prompt=len(prompt_lens))
    print("✅ Generation lengths sampled")

    # 5) Write output
    pairs = [[p, g] for p, g in zip(prompt_lens, gen_lens)]
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)
    print(f"\n✅ Wrote {len(pairs)} pairs → {args.out}\n")

if __name__ == '__main__':
    main()

