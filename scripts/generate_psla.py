#!/usr/bin/env python3
"""
scripts/generate_psla.py

Generate PSLA JSON files for each model×dataset combination based on your 
`dataset_masuqur/` directory structure, writing into `data/psla/masuqur/`.
"""
import os
import json
import statistics

# 1) Fill in your measured latencies per model
hardware_profiles = {
    "internlm2-7b": {
        "first_token_latency": {"p50": 2.0, "p99": 3.0, "max": 10.0},
        "decode_token_latency": {"p50": 0.02, "p99": 0.05, "max": 1.0},
    },
    "mistral-7b": {
        "first_token_latency": {"p50": 1.8, "p99": 2.5, "max": 8.0},
        "decode_token_latency": {"p50": 0.015, "p99": 0.04, "max": 1.0},
    },
    "llama-2-7b": {
        "first_token_latency": {"p50": 2.1, "p99": 3.5, "max": 12.0},
        "decode_token_latency": {"p50": 0.018, "p99": 0.05, "max": 1.0},
    }
}

# 2) Explicit mapping from filename prefix to hardware_profiles key
#    and to the exact model name used in hardware_models.json
prefix_to_model = {
    "internlm2-7b": "internlm2-7b",
    "mistral7b":   "mistral-7b",
    "mistral-7b":  "mistral-7b",
    "llama2-7b":   "llama-2-7b",
    "llama-2-7b":  "llama-2-7b"
}
# Map filename prefix to the canonical model name in hardware_models.json
prefix_to_model_name = {
    "internlm2-7b": "InternLM2-7B",
    "mistral7b":   "Mistral-7B",
    "mistral-7b":  "Mistral-7B",
    "llama2-7b":   "LLaMa-7B",
    "llama-2-7b":  "LLaMa-7B"
}

# 3) Locate project directories
script_dir   = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
dataset_dir  = os.path.join(project_root, "dataset_masuqur")
out_dir      = os.path.join(project_root, "data", "psla", "masuqur")
os.makedirs(out_dir, exist_ok=True)

print(f"Reading datasets from: {dataset_dir}")
print(f"Writing PSLA files to:  {out_dir}")

# 4) Traverse each dataset folder
for ds_name in sorted(os.listdir(dataset_dir)):
    ds_path = os.path.join(dataset_dir, ds_name)
    if not os.path.isdir(ds_path):
        continue

    for fname in sorted(os.listdir(ds_path)):
        if not fname.endswith(".json"):
            continue

        # Parse "<prefix>_<suffix>.json"
        base = fname[:-5]  # strip '.json'
        if "_" not in base:
            print(f"Skipping unexpected file: {fname}")
            continue
        prefix, suffix = base.split("_", 1)

        # Map prefix -> profile key
        if prefix not in prefix_to_model:
            print(f"No profile mapping for '{prefix}' in {fname}")
            continue
        model_key = prefix_to_model[prefix]
        profile   = hardware_profiles[model_key]

        # Load prompt–generation pairs
        file_path = os.path.join(ds_path, fname)
        with open(file_path, "r") as f:
            data = json.load(f)
        if not data:
            print(f"Empty dataset: {fname}")
            continue

        # Extract lengths
        first = data[0]
        if isinstance(first, list):
            prompts = [p for p, g in data]
            gens    = [g for p, g in data]
        elif isinstance(first, dict):
            prompts = [rec["prompt_len"] for rec in data]
            gens    = [rec["gen_len"]   for rec in data]
        else:
            print(f"Unsupported format in {fname}")
            continue

        # Compute stats
        p_mean  = statistics.mean(prompts)
        p_range = max(prompts) - min(prompts)
        g_mean  = statistics.mean(gens)
        g_range = max(gens)    - min(gens)

        # Round means to integers so PSLAConfig validation passes
        p_mean_int = int(round(p_mean))
        g_mean_int = int(round(g_mean))
        p_range_int = int(p_range)
        g_range_int = int(g_range)

        # Build PSLA JSON
        psla = {
            "name":     model_key,
            "model":    model_key,
            "qps":      1000,
            "distribution": "empirical",
            "prompt_lens_mean": p_mean_int,
            "prompt_lens_range": p_range_int,
            "generation_lens_mean": g_mean_int,
            "generation_lens_range": g_range_int,
            "generation_lens_distribution": "empirical",
            "first_token_latency": profile["first_token_latency"],
            "decode_token_latency": profile["decode_token_latency"],
            "shared_cache": True,
            "debug": False
        }

        # Write out
        out_fname = f"{prefix}_{suffix}.json"
        out_path  = os.path.join(out_dir, out_fname)
        with open(out_path, "w") as outf:
            json.dump(psla, outf, indent=2)
        print(f"Wrote PSLA → {out_path}")

