#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES = ["preprocessing", "inference", "cache_access", "postprocessing"]

def collect_stage_data(results_root: Path) -> pd.DataFrame:
    records = []
    for shared_json in results_root.rglob("shared_*.json"):
        ds_mod = shared_json.parent.parent.parent.name
        if "_" not in ds_mod:
            continue
        model, dataset = ds_mod.split("_", 1)
        data = json.loads(shared_json.read_text())
        summary = data.get("stage_latency_summary", {})
        rec = {
            "dataset": dataset,
            "model": model,
        }
        for s in STAGES:
            rec[s] = summary.get(s, {}).get("p50", np.nan)
            rec[s + "_max"] = summary.get(s, {}).get("max", np.nan)
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    df.set_index(["dataset", "model"], inplace=True)
    return df.sort_index()

def table_11_12(df: pd.DataFrame, out_dir: Path):
    tbl11 = df[STAGES].copy()
    tbl11.to_csv(out_dir / "table11_p50.csv")
    print("\nTable 11: Median (p50) latency per stage (ms)\n", tbl11)

    # Table 12: Percentage
    tbl12 = tbl11.div(tbl11.sum(axis=1), axis=0) * 100
    tbl12.to_csv(out_dir / "table12_percent.csv")
    print("\nTable 12: Percentage contribution per stage (%)\n", tbl12)
    return tbl11, tbl12

def table_13(df: pd.DataFrame, out_dir: Path):
    # Table 13: Max vs. median
    rows = []
    for s in STAGES:
        rows.append({"stage": s, "median_p50": df[s].median(), "max_overall": df[s + "_max"].max()})
    tbl13 = pd.DataFrame(rows).set_index("stage")
    tbl13.to_csv(out_dir / "table13_max_vs_median.csv")
    print("\nTable 13: Max vs. median latency per stage\n", tbl13)
    return tbl13

def table_14(df: pd.DataFrame, out_dir: Path):
    # Table 14: Deviation across datasets
    tbl14 = pd.DataFrame({s: df[s].groupby(level=0).median().std() for s in STAGES}, index=["stddev_p50"]).T
    tbl14.to_csv(out_dir / "table14_deviation.csv")
    print("\nTable 14: Std. dev. of median latency per stage\n", tbl14)
    return tbl14

def table_15(df: pd.DataFrame, out_dir: Path):
    # Table 15: Cache access p50 & max
    d = []
    for idx, row in df.iterrows():
        d.append({
            "dataset": idx[0],
            "model": idx[1],
            "cache_p50": row["cache_access"],
            "cache_max": row["cache_access_max"]
        })
    tbl15 = pd.DataFrame(d)
    tbl15.to_csv(out_dir / "table15_cache_latency.csv", index=False)
    print("\nTable 15: Cache access latency (p50 & max)\n", tbl15)
    return tbl15

def figure_13(df: pd.DataFrame, out_dir: Path):
    datasets = sorted(set(df.index.get_level_values(0)))
    models = sorted(set(df.index.get_level_values(1)))
    n_ds, n_m = len(datasets), len(models)
    bar_w = 0.7 / n_m
    x = np.arange(n_ds)
    colors = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=(max(10, n_ds*1.8), 6))
    for mi, m in enumerate(models):
        bottoms = np.zeros(n_ds)
        for si, stage in enumerate(STAGES):
            vals = [df.loc[(ds, m), stage] if (ds, m) in df.index else 0 for ds in datasets]
            ax.bar(x + mi * bar_w, vals, bar_w, bottom=bottoms, label=stage if mi == 0 else None, color=colors[si])
            bottoms += vals
    ax.set_xticks(x + bar_w * (n_m - 1) / 2)
    ax.set_xticklabels(datasets, rotation=18, ha="right")
    ax.set_ylabel("Latency (ms, p50)")
    ax.legend(title="Stage", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Figure 13: Latency per Stage (Stacked Bar Chart)")
    fig.tight_layout()
    fig.savefig(out_dir / "figure13_stage_stacked.png")
    plt.close(fig)

def figure_14(df: pd.DataFrame, out_dir: Path):
    tbl12 = df[STAGES].div(df[STAGES].sum(axis=1), axis=0) * 100
    avg = tbl12.mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(avg, labels=STAGES, autopct='%1.1f%%', colors=plt.get_cmap("tab20").colors[:len(STAGES)])
    ax.set_title("Figure 14: Percentage Time Distribution Across Stages (Pie Chart)")
    fig.tight_layout()
    fig.savefig(out_dir / "figure14_stage_percent_pie.png")
    plt.close(fig)

def figure_15(df: pd.DataFrame, out_dir: Path):
    models = sorted(set(df.index.get_level_values(1)))
    avg = df.groupby("model")[STAGES].mean()
    n_stages, n_m = len(STAGES), len(models)
    bar_w = 0.7 / n_m
    x = np.arange(n_stages)
    fig, ax = plt.subplots(figsize=(max(10, n_stages*2), 6))
    for mi, m in enumerate(models):
        y = [avg.loc[m, s] if m in avg.index else 0 for s in STAGES]
        ax.bar(x + mi * bar_w, y, bar_w, label=m)
    ax.set_xticks(x + bar_w * (n_m - 1) / 2)
    ax.set_xticklabels(STAGES, rotation=18, ha="right")
    ax.set_ylabel("Avg Latency (ms, p50)")
    ax.legend(title="Model", bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Figure 15: Model-wise Stage Latency Comparison (Grouped Bar Chart)")
    fig.tight_layout()
    fig.savefig(out_dir / "figure15_model_stage.png")
    plt.close(fig)

def figure_16(df: pd.DataFrame, out_dir: Path):
    # Dataset-wise latency per stage (grouped bar chart)
    datasets = sorted(set(df.index.get_level_values(0)))
    avg = df.groupby("dataset")[STAGES].mean()
    n_stages, n_d = len(STAGES), len(datasets)
    bar_w = 0.7 / n_d
    x = np.arange(n_stages)
    fig, ax = plt.subplots(figsize=(max(10, n_stages*2), 6))
    for di, ds in enumerate(datasets):
        y = [avg.loc[ds, s] if ds in avg.index else 0 for s in STAGES]
        ax.bar(x + di * bar_w, y, bar_w, label=ds)
    ax.set_xticks(x + bar_w * (n_d - 1) / 2)
    ax.set_xticklabels(STAGES, rotation=18, ha="right")
    ax.set_ylabel("Avg Latency (ms, p50)")
    ax.legend(title="Dataset", bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Figure 16: Dataset-wise Latency Contribution per Stage")
    fig.tight_layout()
    fig.savefig(out_dir / "figure16_dataset_stage.png")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Plot stage-latency breakdown")
    p.add_argument("--results-root", "-r", type=Path, required=True, help="Root of per-dataset results")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("results/stage_latency_plots"), help="Where to write tables & figures")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_stage_data(args.results_root)

    table_11_12(df, args.output_dir)
    table_13(df, args.output_dir)
    table_14(df, args.output_dir)
    table_15(df, args.output_dir)
    figure_13(df, args.output_dir)
    figure_14(df, args.output_dir)
    figure_15(df, args.output_dir)
    figure_16(df, args.output_dir)

    print(f"\n✅ All tables & figures saved under `{args.output_dir}`.")

if __name__ == "__main__":
    main()

