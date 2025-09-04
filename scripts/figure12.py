import subprocess
import yaml
import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np

def get_dataset_name(dataset_path):
    parts = Path(dataset_path).parts
    return parts[1] if "dataset" in parts else "unknown_dataset"


def run_benchmarks(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    qps_values = config['qps_values']
    models = config['models']
    base_args = config['base_args']

    for qps, model in product(qps_values, models):
        cmd = ["python", "benchmark.py"]
        for k, v in base_args.items():
            if k != "results_path":
                cmd.extend([f"--{k}", str(v)])
        cmd.extend([
            "--psla", model["psla"],
            "--dataset_json_path", model["dataset_json_path"],
            "--qps", str(qps),
            "--results_path", ""
        ])
        print(f"🚀 Running {model['name']} at QPS={qps}")
        #subprocess.run(cmd)

    return config


def plot_latency_figure12(config):
    qps_values = config["qps_values"]
    models = config["models"]
    policy = config["base_args"]["eviction_policy"]
    dataset_name = get_dataset_name(models[0]["dataset_json_path"])
    results_dir = Path("results/benchmark") / dataset_name

    model_names = [m["name"] for m in models]
    latency_matrix = []

    for model in models:
        latencies = []
        for qps in qps_values:
            result_path = results_dir / model["name"] / policy / f"qps_{qps}"
            latency = 0
            for file in result_path.glob("*.json"):
                with open(file) as f:
                    result = json.load(f)
                latency = result.get("request_time", {}).get("p99", 0) * 1000  # ms
                break
            latencies.append(latency)
        latency_matrix.append(latencies)

    latency_matrix = np.array(latency_matrix)  # shape: (models, qps)
    print(f"✅ Loaded latency data for {len(models)} models at {len(qps_values)} QPS values.")
   
    model_rename = {
            'llama-7b': 'LLaMA2',
            'mistral-7b': 'Mistral',
            'internlm2-7b': 'Intern',
        }
    model_names = [m["name"] for m in models]
    renamed_models = [model_rename.get(name, name) for name in model_names]
    # Plot true stacked line chart
    plt.figure(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })
    plt.stackplot(qps_values, latency_matrix, labels=renamed_models, alpha=0.7)

    plt.xlabel("QPS")
    plt.ylabel("Cumulative p99 Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.6)

    def to_k(x, _):
        if x >= 1000:
            return f"{x/1000:.0f}k"
        return str(int(x))
    import matplotlib.ticker as ticker
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_k))
    plt.legend(loc="upper left")
    plt.tight_layout()

    fig_dir = Path("results/benchmark/figure")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "figure12_latency_vs_model_stacked.pdf"
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.01)
    print(f"✅ Saved true stacked line chart to: {fig_path}")
    plt.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = run_benchmarks(args.config)
    plot_latency_figure12(config)
