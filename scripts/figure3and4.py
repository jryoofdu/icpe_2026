import os
import json
import yaml
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import numpy as np
import csv
from matplotlib.ticker import MaxNLocator

# -------------------------
# Compute Cache Size Label
# -------------------------
def compute_cache_size(cluster_path: str, psla_model: str, block_size: int) -> str:
    try:
        with open("TransformerRoofline/hardware_models.json", "r") as f:
            hardware_models = json.load(f)
        with open(cluster_path, "r") as f:
            cluster_json = json.load(f)

        gpu_list = hardware_models["hardware"]
        matched_model = next((m for m in gpu_list if m["Name"].lower() in psla_model.lower()), None)

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
        return f"{total_capacity_blocks}GB"
    except Exception as e:
        print(f"[!] Error computing cache size: {e}")
        return "unknown"

# -------------------------
# Plotting Function
# -------------------------
 

def plot_combined_metric(data_dict, ylabel, filename, fig_dir):
    eviction_policies = list(next(iter(data_dict.values())).keys())
    model_names = list(next(iter(next(iter(data_dict.values())).values())).keys())
    datasets = list(data_dict.keys())

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axs = axs.flatten()

    bar_width = 0.2
    x = np.arange(len(eviction_policies))
    offset = np.linspace(-bar_width, bar_width, len(model_names))

    # Calculate global max y value for uniform scaling
    max_y = 0
    for dataset in datasets:
        for ep in eviction_policies:
            for model in model_names:
                val = data_dict[dataset][ep].get(model, 0)
                max_y = max(max_y, val)

    # Round up to next multiple of 10 for cleaner Y-axis
    max_y = (int(max_y / 10) + 1) * 10

    for i, dataset in enumerate(datasets):
        ax = axs[i]
        for j, model in enumerate(model_names):
            values = [data_dict[dataset][ep].get(model, 0) for ep in eviction_policies]
            ax.bar(x + offset[j], values, width=bar_width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels([ep.upper() for ep in eviction_policies])
        ax.set_title(f"({chr(97+i)}) {dataset.replace('_', ' ').title()}", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max_y)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=9)
        #ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Remove extra plot cell if fewer than 6 datasets
    if len(datasets) < 6:
        for k in range(len(datasets), 6):
            fig.delaxes(axs[k])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_dir / filename, bbox_inches='tight', pad_inches=0.01)
    print(f"[✓] Saved: {fig_dir / filename}")

# -------------------------
# Main Execution
# -------------------------
parser = argparse.ArgumentParser(description="Eviction Policy Benchmark + Plot + CSV")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--outdir', type=str, default="results/benchmark", help='Output directory for results (CSV and plots)')
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

eviction_policies = config["eviction_policies"]
base_args = config["base_args"]
models = config["models"]
datasets = config["dataset"]
results_base_path = Path(args.outdir)
fig_dir = results_base_path / "figure"
fig_dir.mkdir(parents=True, exist_ok=True)
csv_dir = results_base_path / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Delete all previous JSON result files
# -------------------------
deleted_files = 0
for dataset in datasets:
    dataset_name = Path(dataset["dataset_json_path"]).parts[1]
    for model in models:
        model_name = model["name"]
        for eviction_policy in eviction_policies:
            result_dir = results_base_path / dataset_name / model_name / eviction_policy / f"qps_{base_args['qps']}"
            if result_dir.exists():
                for file in result_dir.glob("*.json"):
                    file.unlink()
                    deleted_files += 1
print(f"[✓] Deleted {deleted_files} old JSON result files.")

# -------------------------
# Run Benchmarks
# -------------------------
for model in models:
    model_name = model["name"]
    psla = model["psla"]
    for dataset in datasets:
        dataset_json_path = dataset["dataset_json_path"]
        dataset_name = Path(dataset_json_path).parts[1]
        for eviction_policy in eviction_policies:
            result_dir = results_base_path / dataset_name / model_name / eviction_policy / f"qps_{base_args['qps']}"
            result_dir.mkdir(parents=True, exist_ok=True)
            cache_size_str = compute_cache_size(base_args["cluster"], psla, base_args["block_size"])
            filename = f"shared_{cache_size_str}.json"
            result_file = result_dir / filename
            if result_file.exists():
                print(f"[!] Skipping existing: {result_file}")
                continue
            print(f"[+] Running {model_name} on {dataset_name} with eviction={eviction_policy}")
            cmd = [
                "python3", "benchmark.py",
                "--batching", base_args["batching"],
                "--block_size", str(base_args["block_size"]),
                "--swap_policy", base_args["swap_policy"],
                "--prompt_count", str(base_args["prompt_count"]),
                "--prompt_lens_mean", str(base_args["prompt_lens_mean"]),
                "--generation_lens_mean", str(base_args["generation_lens_mean"]),
                "--qps", str(base_args["qps"]),
                "--eviction_policy", eviction_policy,
                "--psla", psla,
                "--dataset_json_path", dataset_json_path,
                "--cluster", base_args["cluster"],
            ]
            subprocess.run(cmd)

# -------------------------
# Collect Data, Write CSV, and Plot
# -------------------------
hitrate_combined = defaultdict(lambda: defaultdict(dict))
decode_combined = defaultdict(lambda: defaultdict(dict))
hitrate_csv_path = csv_dir / "cache_hit_rates_by_eviction.csv"
decode_csv_path = csv_dir / "token_generation_by_eviction.csv"

with open(hitrate_csv_path, "w", newline="") as f1, open(decode_csv_path, "w", newline="") as f2:
    hitrate_writer = csv.writer(f1)
    decode_writer = csv.writer(f2)
    hitrate_writer.writerow(["Dataset", "Model", "EvictionPolicy", "CacheHitRate(%)"])
    decode_writer.writerow(["Dataset", "Model", "EvictionPolicy", "TokenGenTime(ms)"])
    for dataset in datasets:
        dataset_name = Path(dataset["dataset_json_path"]).parts[1]
        for model in models:
            model_name = model["name"]
            psla = model["psla"]
            for eviction_policy in eviction_policies:
                cache_size_str = compute_cache_size(base_args["cluster"], psla, base_args["block_size"])
                filename = f"shared_{cache_size_str}.json"
                result_file = results_base_path / dataset_name / model_name / eviction_policy / f"qps_{base_args['qps']}" / filename
                hitrate = 0.0
                decode_p50_ms = 0.0
                if result_file.exists():
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    stats = data.get("cache_stats", {})
                    hitrate = stats.get("hit_rate", 0.0)
                    decode_p50 = data.get("decode_time", {}).get("p50", 0.0)
                    decode_p50_ms = round(decode_p50 * 1000, 3)
                hitrate_writer.writerow([dataset_name, model_name, eviction_policy, round(hitrate, 2)])
                decode_writer.writerow([dataset_name, model_name, eviction_policy, decode_p50_ms])
                hitrate_combined[dataset_name][eviction_policy][model_name] = round(hitrate, 2)
                decode_combined[dataset_name][eviction_policy][model_name] = decode_p50_ms

# Final plots
plot_combined_metric(hitrate_combined, "Cache Hit Rate (%)", "figure3_combined_hit_rate.pdf", fig_dir)
plot_combined_metric(decode_combined, "Avg Token Gen (ms)", "figure4_combined_token_gen.pdf", fig_dir)
# Print CSV paths
print(f"[✓] Saved: {hitrate_csv_path}")
print(f"[✓] Saved: {decode_csv_path}")