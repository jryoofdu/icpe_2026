# This script runs benchmarks for all (dataset x model x capacity) combinations
# and then generates two combined plots (latency and throughput) in a 2x2+1 layout

import os
import json
import yaml
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import FuncFormatter
import numpy as np

# -------------------------
# Parse CLI args for YAML
# -------------------------
parser = argparse.ArgumentParser(description="Run benchmarks with YAML config")
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--outdir', type=str, default="results/benchmark", help='Output directory for results (CSV and plots)')

args = parser.parse_args()
base_out_dir = Path(args.outdir)
fig_out = base_out_dir / "figure" / "combined"
csv_out_dir = base_out_dir / "csv"

fig_out.mkdir(parents=True, exist_ok=True)
csv_out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load YAML Config
# -------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

total_capacities = config["total_capacities"]
batching = config["batching"]
block_size = str(config["block_size"])
swap_policy = config["swap_policy"]
prompt_count = str(config["prompt_count"])
prompt_lens_mean = str(config["prompt_lens_mean"])
generation_lens_mean = str(config["generation_lens_mean"])
qps = str(config["qps"])
eviction_policy = config["eviction_policy"]
hardware_name = config["hardware_name"]
models = config["models"]
datasets = config["dataset"]
print("Loaded config with the following parameters:")
# === Load Hardware Info ===
with open("TransformerRoofline/hardware_models.json") as f:
    hw_data = json.load(f)

hardware_list = hw_data.get("hardware", [])
fixed_hw = next((h for h in hardware_list if h["Name"] == hardware_name), None)

if not fixed_hw:
    raise ValueError(f"Hardware '{hardware_name}' not found in hardware_models.json")

base_unit = fixed_hw["Capacity"] * fixed_hw.get("Card_Num", 1)

# === Prepare Cluster Config Directory ===
cluster_dir = Path("./data/clusters")
cluster_dir.mkdir(parents=True, exist_ok=True)

# === Benchmarking ===
def make_cluster_config(hw):
    name = hw["Name"]
    card_num = hw["Card_Num"]
    return {
        "num_workers": card_num,
        "networks": {"netw1": "ethernet100Gb"},
        "worker_groups": [
            {
                "role": "homo",
                "hardware": name,
                "num_workers": card_num,
                "network": "netw1"
            }
        ]
    }

for dataset in datasets:
    dataset_json_path = dataset["dataset_json_path"]
    dataset_name = Path(dataset_json_path).parts[1]

    for model in models:
        model_name = model["name"]
        psla = model["psla"]

        results_path = Path(f"results/benchmark/{dataset_name}/{model_name}/{eviction_policy}/qps_{qps}")
        results_path.mkdir(parents=True, exist_ok=True)

        for total in total_capacities:
            if total % base_unit != 0:
                continue

            num_units = total // base_unit
            hw_copy = fixed_hw.copy()
            hw_copy["Card_Num"] = num_units

            cluster_path = cluster_dir / f"{model_name.replace('-', '')}_{total}GB.json"
            config_data = make_cluster_config(hw_copy)
            with open(cluster_path, "w") as f:
                json.dump(config_data, f, indent=4)

            result_file = results_path / f"shared_{total}GB.json"
            if result_file.exists():
                print(f"[!] Skipping existing: {result_file}")
                continue

            print(f"[+] Running benchmark: {dataset_name} × {model_name} @ {total}GB")

            cmd = [
                "python3", "benchmark.py",
                "--batching", batching,
                "--block_size", block_size,
                "--swap_policy", swap_policy,
                "--prompt_count", prompt_count,
                "--prompt_lens_mean", prompt_lens_mean,
                "--generation_lens_mean", generation_lens_mean,
                "--qps", qps,
                "--eviction_policy", eviction_policy,
                "--psla", psla,
                "--dataset_json_path", dataset_json_path,
                "--cluster", str(cluster_path),
            ]
            subprocess.run(cmd)

# === Plotting ===
x = np.arange(len(total_capacities))
bar_width = 0.25
model_rename = {
    "llama-7b": "LLaMA2",
    "mistral-7b": "Mistral",
    "internlm2-7b": "Intern",
}
offset_map = np.linspace(-bar_width, bar_width, len(models))
#fig_out = Path("results/benchmark/figure/combined")
#fig_out.mkdir(parents=True, exist_ok=True)

def plot_combined(plot_type: str, filename: str):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(3.3, 6.6))
    axs = axs.flatten()

    for idx, ds in enumerate(datasets):
        dataset_path = Path(ds["dataset_json_path"])
        dataset_name = dataset_path.parts[1]
        ax = axs[idx]

        for i, model in enumerate(models):
            model_name = model["name"]
            result_path = Path(f"results/benchmark/{dataset_name}/{model_name}/{eviction_policy}/qps_{qps}")
            values = []

            for total in total_capacities:
                result_file = result_path / f"shared_{total}GB.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    value = data["request_time"]["p99"] * 1000 if plot_type == "latency" else data["output_token_ps"]
                else:
                    value = 0
                values.append(value)

            ax.bar(x + offset_map[i], values, width=bar_width, label=model_rename.get(model_name, model_name))

        ax.set_title(dataset_name.replace("_", " ").title(), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in total_capacities], rotation=45)
        ylabel = "Prompt Latency (p99, ms)" if plot_type == "latency" else "Throughput (tokens/s)"
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([str(cs) for cs in total_capacities], fontsize=7, rotation=45)
        ax.set_xlabel("Cache Size", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f'{val/1000:.1f}k'))

    fig.delaxes(axs[-1])
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(fig_out / filename, bbox_inches='tight', pad_inches=0.01)
    print(f"[✓] Saved: {fig_out / filename}")

# Generate both plots
plot_combined("latency", "figure_combined_latency.pdf")
plot_combined("throughput", "figure_combined_throughput.pdf")
import csv

 
csv_path = csv_out_dir / "benchmark_summary.csv"

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dataset", "Model", "CacheSize", "Latency(ms)", "Throughput(tokens/s)"])

    for model in models:
        model_name = model["name"]
        renamed_model = model_rename.get(model_name, model_name)

        for dataset in datasets:
            dataset_path = Path(dataset["dataset_json_path"])
            dataset_name = dataset_path.parts[1]
            result_path = Path(f"results/benchmark/{dataset_name}/{model_name}/{eviction_policy}/qps_{qps}")

            for total in total_capacities:
                result_file = result_path / f"shared_{total}GB.json"
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    latency = round(data["request_time"]["p99"] * 1000, 2)
                    throughput = round(data["output_token_ps"], 2)
                else:
                    latency = "NA"
                    throughput = "NA"
                writer.writerow([dataset_name, renamed_model, total, latency, throughput])

print(f"[✓] Combined CSV written: {csv_path}")


