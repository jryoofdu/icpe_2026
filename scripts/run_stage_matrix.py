#!/usr/bin/env python3
"""
Run TokenSim stage-latency benchmarks for a single dataset across all PSLA models.

For a given dataset (e.g., "bookcorpus"), this script:
  1. Scans `--dataset-root` for per-model JSONs named `<model>_<dataset>.json`.
  2. Scans `--psla-root` for matching PSLA JSONs `<model>_<dataset>.json`.
  3. Invokes `benchmark.py` for each `<model>-<dataset>` pair if results are missing.
  4. Collects each run's `stage_latency_summary` into a single
     `results/stage_summary_matrix.json` of the form:

   {
     "<dataset>": {
       "<model1>": { ...stage_latency_summary... },
       "<model2>": { ... },
       ...
     }
   }

Usage:
  chmod +x scripts/run_stage_matrix.py
  scripts/run_stage_matrix.py \
    --dataset-root dataset_masuqur/bookcorpus \
    --dataset-name bookcorpus \
    --psla-root data/psla/masuqur \
    --cluster data/clusters/1_a100/h1.json \
    [--results-root results] [--qps 100] [--batching paged-attn] \
    [--block-size 64] [--swap-policy eager] [--eviction-policy lru]
"""
import argparse
import subprocess
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Run per-stage benchmarks for one dataset across all PSLA models"
    )
    # paths
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Directory holding <model>_<dataset>.json files")
    p.add_argument("--dataset-name", required=True,
                   help="Name of dataset suffix (e.g. 'bookcorpus')")
    p.add_argument("--psla-root", type=Path, required=True,
                   help="Directory with PSLA JSONs named <model>_<dataset>.json")
    p.add_argument("--cluster", type=Path, required=True,
                   help="Cluster configuration JSON for benchmark.py")
    # optional overrides
    p.add_argument("--results-root", type=Path, default=Path("results"),
                   help="Root output directory (default: results)")
    p.add_argument("--qps", type=int, default=100, help="Queries per second")
    p.add_argument("--batching", default="paged-attn", help="Batching strategy")
    p.add_argument("--block-size", type=int, default=64, help="Block size")
    p.add_argument("--swap-policy", default="eager", help="Swap policy: eager | lazy")
    p.add_argument("--eviction-policy", default="lru", help="Cache eviction: lru | lfu")
    p.add_argument("--prompt-count", type=int, default=50, help="Number of requests to simulate")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    bench_py = repo_root / "benchmark.py"
    ds = args.dataset_name

    # Gather PSLA files matching this dataset
    psla_pattern = f"*_{ds}.json"
    psla_files = sorted(args.psla_root.glob(psla_pattern))
    if not psla_files:
        print(f"⚠️  No PSLA files matching '{psla_pattern}' in {args.psla_root}")
        sys.exit(1)

    matrix = {ds: {}}
    for ps in psla_files:
        model = ps.stem.replace(f"_{ds}", "")
        ds_json = args.dataset_root / ps.name
        if not ds_json.exists():
            print(f"⚠️  Missing dataset JSON for model '{model}': {ds_json}")
            continue

        # expected results folder
        out_dir = (
            args.results_root / "benchmark" / ds / model /
            args.eviction_policy / f"qps_{args.qps}"
        )
        out_file = next(out_dir.glob("shared_*.json"), None)
        if not out_file:
            cmd = [
                sys.executable, str(bench_py),
                "--batching", args.batching,
                "--block_size", str(args.block_size),
                "--swap_policy", args.swap_policy,
                "--prompt_count", str(args.prompt_count),
                "--cluster", str(args.cluster),
                "--dataset_json_path", str(ds_json),
                "--psla", str(ps),
                "--results_path", str(args.results_root),
                "--qps", str(args.qps),
                "--eviction_policy", args.eviction_policy,
            ]
            print(f"▶️ Running: model={model}, dataset={ds}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Benchmark failed for {model}: {e}")
                continue
            out_file = next(out_dir.glob("shared_*.json"), None)
            if not out_file:
                print(f"⚠️  No output JSON for {model}")
                continue

        data = json.loads(out_file.read_text())
        matrix[ds][model] = data.get("stage_latency_summary", {})

    # Write combined matrix
    dst = args.results_root / "stage_summary_matrix.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(matrix, indent=2))
    print(f"✅ Stage-latency matrix written to {dst}")


if __name__ == '__main__':
    main()

