import os
import json
import yaml
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse

# --------------------------
# CLI Argument for YAML Path
# --------------------------
parser = argparse.ArgumentParser(description="Token per session sweep runner")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
args = parser.parse_args()

# --------------------------
# Load YAML Config
# --------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

models = config["models"]
batching = config["batching"]
block_size = str(config["block_size"])
swap_policy = config["swap_policy"]
prompt_count = str(config["prompt_count"])
qps = str(config["qps"])
eviction_policy = config["eviction_policy"]
token_per_session_list = config["token_per_session_list"]  # e.g. [512, 1024, 2048, 4096, 8192]

# --------------------------
# Run Benchmarks
# --------------------------
result_root = Path("results/benchmark")
for model in models:
    name = model["name"]
    psla = model["psla"]
   # dataset = model["dataset_json_path"]
    dataset_name = "Custom"
    model_dir = result_root / dataset_name / name / eviction_policy / f"qps_{qps}"
    model_dir.mkdir(parents=True, exist_ok=True)

    for tps in token_per_session_list:
        prompt_len = tps // 2
        gen_len = tps - prompt_len
        result_path = model_dir / f"{tps}_tps.json"

        if result_path.exists():
            print(f"[✓] Skipping existing result: {result_path}")
            continue

        print(f"[+] Running: {name} with {tps} tokens/session")

        cmd = [
            "python3", "benchmark.py",
            "--batching", batching,
            "--block_size", block_size,
            "--swap_policy", swap_policy,
            "--prompt_count", prompt_count,
            "--prompt_lens_mean", str(prompt_len),
            "--generation_lens_mean", str(gen_len),
            "--qps", qps,
            "--eviction_policy", eviction_policy,
            "--psla", psla,
            "--cluster", model["cluster"], 
            "--tps", str(tps),
        ]

        #subprocess.run(cmd)

# --------------------------
# Collect Results for Plotting
# --------------------------
latency_data = defaultdict(list)
throughput_data = defaultdict(list)
cache_hit_data = defaultdict(list)
x_labels = [str(tps) for tps in token_per_session_list]
x = np.arange(len(token_per_session_list))

# Load collected JSONs
for model in models:
    name = model["name"]
    dataset_name = "Custom"#//Path(model["dataset_json_path"]).parts[1]
    model_dir = result_root / dataset_name / name / eviction_policy / f"qps_{qps}"

    for tps in token_per_session_list:
        result_path = model_dir / f"tps_{int(tps)}"
        for file in result_path.glob("*.json"):
                with open(file) as f:
                    result = json.load(f)
                    latency_data[name].append(result["total_latency"])
                    throughput_data[name].append(result["output_token_ps"])
                    cache_hit_data[name].append(result.get("cache_stats", {}).get("hit_rate", 0))

# --------------------------
# Plot Figures to PDF
# --------------------------
fig_dir = Path("results/benchmark/figure")
fig_dir.mkdir(parents=True, exist_ok=True)
bar_width = 0.25
offsets = np.linspace(-bar_width, bar_width, len(models))
model_rename = {
            'llama-7b': 'LLaMA2',
            'mistral-7b': 'Mistral',
            'internlm2-7b': 'Intern',
        }
# Figure 9: Latency
plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})
plt.figure(figsize=(3.3, 2.2))
for i, model in enumerate(models):
    name = model["name"]
    print(f"Processing {name} latency data: {latency_data[name]}")
    #latency_ms = [v * 1000 for v in latency_data[name]]
    latency_ms = [v * 1000 for v in latency_data[name]]  # Convert to ms

    plt.bar(x + offsets[i], latency_ms, width=bar_width, label=model_rename.get(name, name))

plt.xticks(x, x_labels, rotation=45)
plt.xlabel("Tokens per Session")
plt.ylabel("Latency(ms)")
from matplotlib.ticker import FuncFormatter
def format_k(val, pos):
            return f'{val/1000:.1f}k'
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_k))
#plt.title("Figure 9: Latency vs Tokens per Session")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "figure9_latency_vs_tokens.pdf", bbox_inches='tight', pad_inches=0.01)

# Figure 10: Throughput
plt.figure(figsize=(3.3, 2.2))
for i, model in enumerate(models):
    name = model["name"]
    plt.bar(x + offsets[i], throughput_data[name], width=bar_width, label=model_rename.get(name, name))
plt.xticks(x, x_labels, rotation=45)
plt.xlabel("Tokens per Session")
plt.ylabel("Throughput (tokens/s)")
from matplotlib.ticker import FuncFormatter
def format_k(val, pos):
            return f'{val/1000:.1f}k'
plt.gca().yaxis.set_major_formatter(FuncFormatter(format_k))
#plt.title("Figure 10: Throughput vs Tokens per Session")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "figure10_throughput_vs_tokens.pdf", bbox_inches='tight', pad_inches=0.01)

# Figure 11: Cache Hit Rate
plt.figure(figsize=(3.3, 2.2))
for i, model in enumerate(models):
    name = model["name"]
    positions = x + i * bar_width
    plt.bar(x + offsets[i], cache_hit_data[name], width=bar_width, label=model_rename.get(name, name))
plt.xticks(x, x_labels, rotation=45)
plt.xlabel("Tokens per Session")
plt.ylabel("Cache Hit Rate(%)")
#plt.title("Figure 11: Cache Hit Rate vs Tokens per Session")
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "figure11_cache_hit_vs_tokens.pdf", bbox_inches='tight', pad_inches=0.01)

print("[✓] Finished: Figures saved to PDF.")
