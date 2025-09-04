import subprocess
import yaml
import os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from itertools import product
import matplotlib.ticker as mticker

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
    models = config['models']
    base_args = config['base_args']
    policy = base_args.get("eviction_policy", "lru")

    for qps, model in product(qps_values, models):
        model_name = model['name']
        dataset_name = get_dataset_name(model["dataset_json_path"])

        cmd = ["python", "benchmark.py"]
        for k, v in base_args.items():
            if k != "results_path":
                cmd.extend([f"--{k}", str(v)])
        cmd.extend([
            "--psla", model["psla"],
            "--dataset_json_path", model["dataset_json_path"],
            "--qps", str(qps),
            "--results_path", ""  # intentionally empty to trigger export_result logic
        ])

        print(f"🚀 Running model={model_name}, qps={qps}, dataset={dataset_name}")
        subprocess.run(cmd)

    return config

def plot_results(config):
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    from collections import defaultdict

    qps_values = config["qps_values"]
    models = config["models"]
    policy = config["base_args"].get("eviction_policy", "lru")

    data_by_dataset = defaultdict(lambda: defaultdict(list))

    # Gather data
    for model in models:
        model_name = model["name"]
        dataset_name = get_dataset_name(model["dataset_json_path"])
        for qps in qps_values:
            result_path = Path(args.outdir) / dataset_name / model_name / policy / f"qps_{qps}"
            if not result_path.exists():
                continue
            for file in result_path.glob("*.json"):
                with open(file) as f:
                    result = json.load(f)
                throughput = result.get("throughput", result.get("output_token_ps"))
                if throughput is not None:
                    data_by_dataset[dataset_name][model_name].append((qps, throughput))

    # Plot
    for dataset, model_data in data_by_dataset.items():
        plt.figure(figsize=(3.3, 2.2))
        plt.rcParams.update({
            'font.size': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 6,
        })

        bar_width = 0.2
        x = np.arange(len(qps_values))

        model_rename = {
            'llama-7b': 'LLaMA2',
            'mistral-7b': 'Mistral',
            'internlm2-7b': 'Intern',
        }

        for i, (model, results) in enumerate(model_data.items()):
            qps_to_throughput = {q: t for q, t in results}
            throughput_vals = [qps_to_throughput.get(q, 0) for q in qps_values]
            positions = x + i * bar_width
            plt.bar(positions, throughput_vals, bar_width, label=model_rename.get(model, model))

        plt.xlabel("Sessions")
        plt.ylabel("Throughput (tokens/sec)")
        from matplotlib.ticker import FuncFormatter
        def format_k(val, pos):
            return f'{val/1000:.1f}k'
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_k))
        plt.xticks(x + bar_width * (len(model_data) - 1) / 2, qps_values)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()

        fig_dir = Path(args.outdir) /"figure"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / "figure8a.pdf"
        plt.savefig(fig_path, bbox_inches='tight', pad_inches=0.01)
        print(f"✅ Saved bar chart to: {fig_path}")
        plt.close()

     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="results/benchmark", help="Output directory for results")
    args = parser.parse_args()

    config = run_benchmarks(args.config)
    plot_results(config)
