import os
import json
import yaml
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import argparse
from matplotlib.ticker import FuncFormatter
import csv

# --------------------------
# CLI Argument for YAML Path
# --------------------------
parser = argparse.ArgumentParser(description="Token per session benchmark")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--outdir', type=str, default="results/benchmark", help='Output directory for results (CSV and plots)')
args = parser.parse_args()

# --------------------------
# Load YAML Config
# --------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

models = config["models"]
batching = config["batching"]
block_size = int(config["block_size"])
swap_policy = config["swap_policy"]
prompt_count = str(config["prompt_count"])
qps = str(config["qps"])
eviction_policy = config["eviction_policy"]
cluster_path = config["cluster"]

# --------------------------
# Load cluster for cache size calculation
# --------------------------
with open(cluster_path, "r") as f:
    cluster_json = json.load(f)

def compute_cache_size_gb(psla_model: str) -> float:
    try:
        with open("TransformerRoofline/hardware_models.json", "r") as f:
            hardware_models = json.load(f)

        gpu_list = hardware_models["hardware"]
        matched_model = next(
            (m for m in gpu_list if m["Name"].lower() in psla_model.lower()),
            None
        )
        worker_groups = cluster_json.get("worker_groups", [])
        total_capacity_blocks = 0

        for wg in worker_groups:
            hw_name = wg["hardware"]
            num = wg["num_workers"]
            matched_hw = next((h for h in gpu_list if h["Name"] == hw_name), None)
            if matched_hw:
                cap = matched_hw.get("Capacity", 0)
                total_capacity_blocks += cap * num

        d_model = matched_model["Dmodel"] if matched_model else 4096
        bytes_per_token = 2 * d_model
        total_cache_bytes = total_capacity_blocks * block_size * bytes_per_token
        return round(total_cache_bytes / (1024 ** 3), 2)

    except Exception as e:
        print(f"[!] Error computing cache size: {e}")
        return 0

# --------------------------
# Run Benchmarks
# --------------------------
result_root = Path(args.outdir)
for model in models:
    name = model["name"]
    psla_list = model["psla_files"]
    dataset_json_path = model["dataset_json_path"]
    dataset_name = Path(dataset_json_path).parts[1]

    for psla_path in psla_list:
        psla_filename = Path(psla_path).stem
        variant = "shared" if "shared" in Path(psla_path).stem else "isolated"
        psla_model = name
        cache_size_gb = compute_cache_size_gb(psla_model)
        cache_tag = f"{variant}_{cache_size_gb}_GB"
         
        model_dir = result_root / dataset_name / name / eviction_policy / f"qps_{qps}" / cache_tag
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[+] Running: {name} with  ({cache_tag})")

        cmd = [
            "python3", "benchmark.py",
            "--batching", batching,
            "--block_size", str(block_size),
            "--swap_policy", swap_policy,
            "--prompt_count", prompt_count,
            "--prompt_lens_mean", str(config["prompt_lens_mean"]),
            "--generation_lens_mean", str(config["generation_lens_mean"]),
            "--qps", qps,
            "--eviction_policy", eviction_policy,
            "--psla", psla_path,
            "--cluster", cluster_path,
            "--dataset_json_path", dataset_json_path,
        ]

        subprocess.run(cmd)

# --------------------------
# Aggregate Results (shared/isolated)
# --------------------------
latency_data = defaultdict(lambda: defaultdict(list))
throughput_data = defaultdict(lambda: defaultdict(list))

for model in models:
    name = model["name"]
    for psla_path in model["psla_files"]:
        psla_filename = Path(psla_path).stem
        variant = "shared" if "shared" in psla_filename else "isolated"
        cache_size = '40'  # Assumed fixed for now
        cache_tag = f"{variant}_{cache_size}GB"

        dataset_name = "longbench"
        model_dir = result_root / dataset_name / name / eviction_policy / f"qps_{qps}"
        result_path = model_dir / f"{cache_tag}.json"
        print(f"[+] Collecting results from: {result_path}")
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
                print(f"[✓] Loaded {variant} for {name}: latency={result['total_latency']}, throughput={result['output_token_ps']}")
                latency_data[variant][name].append(result["total_latency"] * 1000)
                throughput_data[variant][name].append(result["output_token_ps"])

# --------------------------
# Plot Shared vs Isolated (x-axis)
# --------------------------
fig_dir = result_root / "figure"
fig_dir.mkdir(parents=True, exist_ok=True)
bar_width = 0.25
model_rename = {
    'llama-7b': 'LLaMA2',
    'mistral-7b': 'Mistral',
    'internlm2-7b': 'Intern',
}
variants = ["shared", "isolated"]
x = np.arange(len(variants))
offsets = np.linspace(-bar_width, bar_width, len(models))

# Mean aggregation
mean_latency = defaultdict(lambda: defaultdict(float))
mean_throughput = defaultdict(lambda: defaultdict(float))

for variant in variants:
    for model in models:
        name = model["name"]
        lats = latency_data[variant][name]
        thps = throughput_data[variant][name]
        if lats:
            mean_latency[variant][name] = sum(lats) / len(lats)
        if thps:
            mean_throughput[variant][name] = sum(thps) / len(thps)

plt.rcParams.update({
    'font.size': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

# === Figure: Latency ===
plt.figure(figsize=(3.3, 2.2))
for i, model in enumerate(models):
    name = model["name"]
    data = [mean_latency[v][name] for v in variants]
    plt.bar(x + offsets[i], data, width=bar_width, label=model_rename.get(name, name))
plt.xticks(x, variants)
plt.xlabel("Cache Architecture")
plt.ylabel("Latency (ms)")
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/1000:.1f}k'))
plt.legend()
plt.tight_layout()
latency_fig_path = fig_dir / "figure5_latency_shared_vs_isolated.pdf"
plt.savefig(latency_fig_path, bbox_inches='tight', pad_inches=0.01)
print(f"[✓] Saved: {latency_fig_path}")

# === Figure: Throughput ===
plt.figure(figsize=(3.3, 2.2))
for i, model in enumerate(models):
    name = model["name"]
    data = [mean_throughput[v][name] for v in variants]
    plt.bar(x + offsets[i], data, width=bar_width, label=model_rename.get(name, name))
plt.xticks(x, variants)
plt.xlabel("Cache Architecture")
plt.ylabel("Throughput (tokens/s)")
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{val/1000:.1f}k'))
plt.legend()
plt.tight_layout()
throughput_fig_path = fig_dir / "figure6_throughput_shared_vs_isolated.pdf"
plt.savefig(throughput_fig_path, bbox_inches='tight', pad_inches=0.01)
print(f"[✓] Saved: {throughput_fig_path}")

# --------------------------
# CSV Output
# --------------------------
csv_dir = result_root / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / "shared_vs_isolated_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Variant", "Latency(ms)", "Throughput(tokens/s)"])
    for variant in variants:
        for model in models:
            name = model["name"]
            latency = round(mean_latency[variant][name], 2)
            throughput = round(mean_throughput[variant][name], 2)
            writer.writerow([name, variant, latency, throughput])
print(f"[✓] Saved: {csv_path}")

print("[✓] Finished: Plots and CSV generated for shared vs. isolated analysis.")
