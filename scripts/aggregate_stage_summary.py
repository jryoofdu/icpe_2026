#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    # adjust this if your results path is different
    benchmark_root = Path("results/benchmark/dataset_masuqur")
    matrix: dict[str, dict[str, dict]] = {}

    for combo in benchmark_root.iterdir():
        if not combo.is_dir():
            continue

        # combo.name is like "internlm2-7b_bookcorpus"
        try:
            model, dataset = combo.name.split("_", 1)
        except ValueError:
            # skip any odd directories
            continue

        # find the first shared_*.json under this tree
        shared = next(combo.glob("**/shared_*.json"), None)
        if shared is None:
            print(f"⚠️  no shared_*.json in {combo}")
            continue

        data = json.loads(shared.read_text())
        summary = data.get("stage_latency_summary")
        if summary is None:
            print(f"⚠️  no stage_latency_summary in {shared}")
            continue

        matrix.setdefault(dataset, {})[model] = summary

    # write out
    out = Path("results/stage_summary_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2))
    print(f"✅ Wrote combined stage_summary_matrix → {out}")

if __name__ == "__main__":
    main()

