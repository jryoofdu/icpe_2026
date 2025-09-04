import subprocess
import yaml
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path
from collections import defaultdict
from itertools import product
import matplotlib.ticker as ticker

def get_dataset_name(dataset_path):
    parts = Path(dataset_path).parts
    if "dataset" in parts:
        idx = parts.index("dataset")
        if len(parts) > idx + 1:
            return parts[idx + 1]
    return "unknown_dataset"

def run_benchmarks(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    qps_values = config['qps_values']
    datasets = config['dataset']
    base_args = config['base_args']
    policy = base_args.get("eviction_policy", "lru")

    for qps, entry in product(qps_values, datasets):
        model_name = entry['name']
        dataset_name = get_dataset_name(entry["dataset_json_path"])

        cmd = ["python", "benchmark.py"]
        for k, v in base_args.items():
            if k != "results_path":
                cmd.extend([f"--{k}", str(v)])
        cmd.extend([
            "--psla", entry["psla"],
            "--dataset_json_path", entry["dataset_json_path"],
            "--qps", str(qps),
            "--results_path", ""
        ])

        print(f"🚀 Running model={model_name}, qps={qps}, dataset={dataset_name}")
        subprocess.run(cmd)

    return config

def plot_results(config):
    qps_values = config["qps_values"]
    datasets = config["dataset"]
    policy = config["base_args"].get("eviction_policy", "lru")

    result_dict = defaultdict(dict)
    latency_rows = []  # For CSV: (Dataset, QPS, Latency ms)

    for entry in datasets:
        model_name = entry["name"]
        dataset_name = get_dataset_name(entry["dataset_json_path"])

        for qps in qps_values:
            result_path = Path(args.outdir) / dataset_name / model_name / policy / f"qps_{qps}"
            if not result_path.exists():
                continue

            for file in result_path.glob("*.json"):
                with open(file) as f:
                    result = json.load(f)

                throughput = result.get("throughput", result.get("output_token_ps"))
                latency_sec = result.get("request_time", {}).get("p99", 0)
                latency_ms = round(latency_sec * 1000, 2)

                if latency_ms is not None:
                    result_dict[dataset_name][qps] = latency_ms
                latency_rows.append([dataset_name, qps, latency_ms])
                break  # only take the first JSON file per result_path

    # Prepare combined data
    combined_data = {
        dataset_name: [(qps, result_dict[dataset_name].get(qps, 0)) for qps in qps_values]
        for dataset_name in result_dict
    }

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 6,
        'axes.labelsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
    })

    # Short names for datasets
    short_dataset_names = {
        "wikipedia_structured": "Wiki",
        "needle_in_a_haystack": "Needle",
        "sharegpt": "ShareGPT",
        "longbench": "LongB",
        "bookcorpus": "Book"
    }

    x = np.arange(len(qps_values))
    num_datasets = len(combined_data)
    bar_width = 0.8 / num_datasets

    for idx, (dataset_name, results) in enumerate(combined_data.items()):
        qps_vals, latency_vals  = zip(*results)
        positions = x + idx * bar_width
        label = short_dataset_names.get(dataset_name, dataset_name)
        ax.bar(positions, latency_vals , bar_width, label=label)

    ax.set_xlabel("Sessions")
    ax.set_ylabel("Request Latency P99(ms)")
    ax.set_xticks(x + (bar_width * (num_datasets - 1) / 2))
    ax.set_xticklabels(qps_values)

    # Use 'k' suffix for thousands on y-axis
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

    # Legend on the right
    ax.legend(
    loc='center left',
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    fontsize=6,          # smaller legend font
    handlelength=1.0,    # shorten legend marker lines
    labelspacing=0.4     # tighten vertical spacing
)

    plt.tight_layout()
    fig_dir = Path(args.outdir) /"figure"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "figure7b.pdf"
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved bar chart to: {fig_path}")
    plt.close()


    # Save CSV of latency
    csv_path = fig_dir / "Table6_latency_data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "QPS", "Latency_ms"])
        writer.writerows(latency_rows)
    print(f"✅ Saved latency data CSV to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="results/benchmark", help="Output directory for results")
    args = parser.parse_args()

    config = run_benchmarks(args.config)
    plot_results(config)
