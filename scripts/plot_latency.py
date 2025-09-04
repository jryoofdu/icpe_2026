#!/usr/bin/env python3
import argparse
import json
import matplotlib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import matplotlib.axes as maxes
# override the normal set_title so nothing ever gets drawn
maxes.Axes.set_title = lambda self, *args, **kwargs: None

STAGES = ["preprocessing", "inference", "cache_access", "postprocessing"]

def collect_stage_data(results_root: Path) -> pd.DataFrame:
    records = []
    for shared_json in results_root.rglob("shared_*.json"):
        # shared_json: .../<model>_<dataset>/lru/qps_*/shared_*.json
        ds_mod = shared_json.parent.parent.parent.name
        if "_" not in ds_mod:
            continue
        model, dataset = ds_mod.split("_", 1)
        data = json.loads(shared_json.read_text())
        summary = data.get("stage_latency_summary", {})
        rec = {"dataset": dataset, "model": model}
        for s in STAGES:
            rec[s] = summary.get(s, {}).get("p50", np.nan)
            rec[s + "_max"] = summary.get(s, {}).get("max", np.nan)
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    df.set_index(["dataset", "model"], inplace=True)
    return df.sort_index()

def table_11_12(df: pd.DataFrame, out_dir: Path):
    tbl11 = df[STAGES].copy()
    print("\nTable 11: Median (p50) latency per stage (ms)\n")
    print(tbl11.to_markdown())
    tbl11.to_csv(out_dir/"table11_p50.csv")

    tbl12 = tbl11.div(tbl11.sum(axis=1), axis=0) * 100
    print("\nTable 12: Percentage contribution per stage (%)\n")
    print(tbl12.to_markdown())
    #tbl12.to_csv(out_dir/"table12_percent.csv")
    tbl12.to_csv(
        out_dir/"table12_percent.csv",
        float_format="%.2f",
    )
    
    return tbl11, tbl12

def table_13__(df: pd.DataFrame, out_dir: Path):
    """
    Table 13: Max vs. median latency per stage, 
    computed over ALL dataset–model combinations.
    """
    # 1) flatten the MultiIndex so we get one row per (dataset,model)
    flat = df.reset_index()

    # 2) build our rows by taking the median p50 and the absolute max
    rows = []
    for stage in STAGES:
        med = flat[stage].median()
        mx  = flat[stage + "_max"].max()
        rows.append({
            "stage": stage,
            "median_p50": med,
            "max_overall": mx
        })

    # 3) assemble into a DataFrame and write out
    tbl13 = pd.DataFrame(rows).set_index("stage")
    print("\nTable 13: Max vs. median latency per stage (all models & datasets)\n")
    print(tbl13.to_markdown())

    tbl13.to_csv(out_dir / "table13.csv", float_format="%.3f")
    return tbl13

def table_13(df: pd.DataFrame, out_dir: Path):
    """
    Table 13: Per‐dataset, per‐stage p50 and max latency for the llama2-7b model.
    Produces 20 rows (5 datasets × 4 stages).
    """
    # 1) pick out only the llama2-7b entries
    llama_df = df.xs("llama2-7b", level="model")

    # 2) assemble one row per (dataset,stage)
    rows = []
    for dataset in llama_df.index:
        for stage in STAGES:
            p50   = llama_df.loc[dataset, stage]
            mkey  = f"{stage}_max"
            max_v = llama_df.loc[dataset, mkey]
            rows.append({
                "dataset": dataset,
                "stage":    stage,
                "p50_ms":   round(p50,   3),
                "max_ms":   round(max_v, 3),
            })

    # 3) turn into a DataFrame with a hierarchical index
    tbl13 = (
        pd.DataFrame(rows)
          .set_index(["dataset","stage"])
          .sort_index()
    )

    # 4) print & write out
    print("\nTable 13: Latency (p50 & max) per stage for llama2-7b across all datasets\n")
    print(tbl13.to_markdown())

    tbl13.to_csv(out_dir / "table13_llama2-7b_by_dataset_stage.csv", float_format="%.3f")
    return tbl13

def table_13_(df: pd.DataFrame, out_dir: Path):
    rows = []
    for s in STAGES:
        rows.append({
            "stage": s,
            "median_p50": df[s].median(),
            "max_overall": df[s + "_max"].max()
        })
    tbl13 = pd.DataFrame(rows).set_index("stage")
    print("\nTable 13: Max vs. median latency per stage\n")
    print(tbl13.to_markdown())
    tbl13.to_csv(out_dir/"table13_max_vs_median.csv")
    return tbl13

def table_14_(df: pd.DataFrame, out_dir: Path):
    # Std dev of per‐dataset medians
    medians = df[STAGES].groupby(level=0).median()
    tbl14 = pd.DataFrame({
        s: medians[s].std() for s in STAGES
    }, index=["stddev_p50"]).T
    print("\nTable 14: Std. dev. of median latency per stage\n")
    print(tbl14.to_markdown())
    tbl14.to_csv(out_dir/"table14_deviation.csv")
    return tbl14

def table_14(df: pd.DataFrame, out_dir: Path):
    """
    Table 14: For each stage (rows) show the stddev of p50 latency
    across the three models for each dataset (columns).

    If you only want to “consider only longbench”, we filter down to that
    column at the end.
    """
    # 1) compute per‐dataset stddev across models
    std_by_dataset = df[STAGES].groupby(level=0).std()

    # std_by_dataset is a DataFrame:
    # index: dataset, columns: stages

    # 2) pivot so that stages are rows and datasets are columns
    tbl14 = std_by_dataset.T

    # 3) if you really only want the 'longbench' column, uncomment this line:
    # tbl14 = tbl14[['longbench']]

    print("\nTable 14: Std. dev. of p50 latency per stage (rows) × dataset (cols)\n")
    print(tbl14.to_markdown())

    tbl14.to_csv(out_dir / "table14_deviation.csv")
    return tbl14


def table_15(df: pd.DataFrame, out_dir: Path):
    d = []
    for idx, row in df.iterrows():
        d.append({
            "dataset": idx[0],
            "model": idx[1],
            "cache_p50": row["cache_access"],
            "cache_max": row["cache_access_max"]
        })
    tbl15 = pd.DataFrame(d)
    print("\nTable 15: Cache access latency (p50 & max)\n")
    print(tbl15.to_markdown(index=False))
    tbl15.to_csv(out_dir/"table15_cache_latency.csv", index=False)
    return tbl15


def figure_13(df: pd.DataFrame, out_dir: Path):
    """
    Figure 13: For each dataset (x‐group), stack 4 stages,
    with 3 models side by side per group.
    Stages → solid face‐colors.
    Models → distinct edge‐colors on each bar.
    """
    # 1) Global small‐font settings
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'legend.title_fontsize': 7,
    })

    datasets = sorted(df.index.get_level_values(0).unique())
    models   = sorted(df.index.get_level_values(1).unique())
    n_ds, n_m = len(datasets), len(models)

    bar_w = 0.8 / n_m
    x = np.arange(n_ds)

    stage_colors = plt.get_cmap("Set2").colors[:len(STAGES)]
    model_edgecolors = plt.get_cmap("Dark2").colors[:n_m]

    # 2) One figure call at the target small size
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    for mi, model in enumerate(models):
        bottom = np.zeros(n_ds)
        for si, stage in enumerate(STAGES):
            vals = [
                df.loc[(ds, model), stage] if (ds, model) in df.index else 0
                for ds in datasets
            ]
            ax.bar(
                x + mi*bar_w,
                vals,
                bar_w,
                bottom=bottom,
                facecolor=stage_colors[si],
                edgecolor=model_edgecolors[mi],
                linewidth=1.0,
                label=stage if mi == 0 else None
            )
            bottom += vals

    ax.set_xticks(x + bar_w*(n_m-1)/2)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Latency (ms, p50)")
    ax.set_title("Figure 13: Latency per Stage\n(Stacked Bar Chart)", fontsize=7)

    # build custom legends
    stage_patches = [
        plt.Line2D([0], [0],
                   marker='s',
                   color='w',
                   markerfacecolor=stage_colors[i],
                   markersize=6,
                   label=STAGES[i])
        for i in range(len(STAGES))
    ]
    model_patches = [
        plt.Line2D([0], [0],
                   marker='s',
                   color=model_edgecolors[i],
                   markerfacecolor='w',
                   markersize=6,
                   label=models[i])
        for i in range(len(models))
    ]

    # 3) place legends with small text
    leg1 = ax.legend(
        handles=stage_patches,
        title="Stage",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        frameon=False
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=model_patches,
        title="Model",
        bbox_to_anchor=(1.02, 0.6),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        out_dir/"figure13.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

def figure_13_works(df: pd.DataFrame, out_dir: Path):
    """
    Figure 13: For each dataset (x‐group), stack 4 stages,
    with 3 models side by side per group.
    Stages → solid face‐colors.
    Models → distinct edge‐colors on each bar.
    """
    datasets = sorted(df.index.get_level_values(0).unique())
    models   = sorted(df.index.get_level_values(1).unique())
    n_ds, n_m = len(datasets), len(models)

    # width of each bar within a dataset group
    bar_w = 0.8 / n_m
    x = np.arange(n_ds)

    # pick 4 solid colors for the stages
    stage_cmap = plt.get_cmap("Set2")
    stage_colors = stage_cmap.colors[:len(STAGES)]

    # pick n_m edge‐colors for the models
    model_cmap = plt.get_cmap("Dark2")
    model_edgecolors = model_cmap.colors[:n_m]

    fig, ax = plt.subplots(figsize=(max(10, n_ds*1.2), 6))

    for mi, model in enumerate(models):
        bottoms = np.zeros(n_ds)
        for si, stage in enumerate(STAGES):
            # gather the p50 for this (ds,model,stage)
            vals = [df.loc[(ds, model), stage] if (ds, model) in df.index else 0
                    for ds in datasets]

            bars = ax.bar(
                x + mi*bar_w,
                vals,
                bar_w,
                bottom=bottoms,
                facecolor=stage_colors[si],
                edgecolor=model_edgecolors[mi],
                linewidth=1.5,
                label=stage if mi==0 else None
            )
            bottoms += vals

    # x‐axis labels in the middle of each group
    ax.set_xticks(x + bar_w*(n_m-1)/2)
    ax.set_xticklabels(datasets, rotation=30, ha="right")

    ax.set_ylabel("Latency (ms, p50)")
    ax.set_title("Figure 13: Latency per Stage (Stacked Bar Chart)")

    # one legend for the face‐colors (stages)
    stage_patches = [
        plt.Line2D([0],[0], marker='s', color='w',
                   markerfacecolor=stage_colors[i], markersize=10,
                   label=STAGES[i])
        for i in range(len(STAGES))
    ]
    # one legend for the edge‐colors (models)
    model_patches = [
        plt.Line2D([0],[0], marker='s', color=model_edgecolors[i],
                   markerfacecolor='w', markersize=10,
                   label=models[i])
        for i in range(len(models))
    ]

    # place both legends
    leg1 = ax.legend(handles=stage_patches, title="Stage",
                     bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=model_patches, title="Model",
              bbox_to_anchor=(1.02, 0.6), loc="upper left")

    fig.tight_layout()
    #fig.savefig(out_dir/"figure13_stage_stacked.png")
    plt.figure(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })
    fig.savefig(out_dir/"figure13.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def figure_13_(df: pd.DataFrame, out_dir: Path):
    datasets = sorted(set(df.index.get_level_values(0)))
    models   = sorted(set(df.index.get_level_values(1)))
    n_ds, n_m = len(datasets), len(models)
    bar_w = 0.8 / n_m
    x = np.arange(n_ds)

    # one color per model
    model_colors = plt.get_cmap("tab10").colors
    # one hatch per stage
    hatches = ["", "//", "xx", ".."]

    fig, ax = plt.subplots(figsize=(max(10, n_ds*1.5), 6))
    for mi, m in enumerate(models):
        bottom = np.zeros(n_ds)
        color = model_colors[mi % len(model_colors)]
        for si, stage in enumerate(STAGES):
            vals = [
                df.loc[(ds, m), stage]
                if (ds, m) in df.index else 0
                for ds in datasets
            ]
            ax.bar(
                x + mi*bar_w,
                vals,
                bar_w,
                bottom=bottom,
                color=color,
                edgecolor="black",
                hatch=hatches[si],
                label=m if si==0 else None,
            )
            bottom += vals

    # Models legend
    model_handles = [
        mpatches.Patch(facecolor=model_colors[i], edgecolor="black", label=models[i])
        for i in range(len(models))
    ]
    leg1 = ax.legend(
        handles=model_handles,
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    # Stages legend
    stage_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=hatches[i], label=STAGES[i])
        for i in range(len(STAGES))
    ]
    ax.add_artist(leg1)
    ax.legend(
        handles=stage_handles,
        title="Stage",
        bbox_to_anchor=(1.02, 0.6),
        loc="upper left",
    )

    ax.set_xticks(x + bar_w*(n_m-1)/2)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms, p50)")
    ax.set_title("Figure 13: Latency per Stage (Stacked Bar Chart)")
    fig.tight_layout()
    fig.savefig(out_dir/"figure13_stage_stacked.png")
    plt.close(fig)

def figure_14_(df: pd.DataFrame, out_dir: Path):
    tbl12 = df[STAGES].div(df[STAGES].sum(axis=1), axis=0) * 100
    avg = tbl12.mean()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(
        avg,
        labels=STAGES,
        autopct="%1.1f%%",
        colors=plt.get_cmap("tab20").colors[:len(STAGES)]
    )
    ax.set_title("Figure 14: Percentage Time Distribution Across Stages (Pie Chart)")
    fig.tight_layout()
    #fig.savefig(out_dir/"figure14_stage_percent_pie.png")
    plt.figure(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })
    fig.savefig(out_dir/"figure14.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def figure_14(df: pd.DataFrame, out_dir: Path):
    tbl12 = df[STAGES].div(df[STAGES].sum(axis=1), axis=0) * 100
    avg = tbl12.mean()

    # 1) update rcParams up front (optional, but will apply globally in this function)
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })

    # 2) create figure & axis
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    # 3) draw pie, passing textprops to control label & autopct font size
    wedges, texts, autotexts = ax.pie(
        avg,
        labels=STAGES,
        autopct="%1.1f%%",
        textprops={'fontsize': 7},           # <— label & pct text size
        colors=plt.get_cmap("tab20").colors[:len(STAGES)]
    )

    # 4) explicitly set title fontsize
    ax.set_title(
        "Figure 14: Percentage Time Distribution Across Stages (Pie Chart)",
        fontsize=7
    )

    fig.tight_layout()
    fig.savefig(out_dir/"figure14.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def figure_15(df: pd.DataFrame, out_dir: Path):
    # 1) Global style for everything in this plot
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'legend.title_fontsize': 7,
    })

    models = sorted(df.index.get_level_values(1).unique())
    avg = df.groupby("model")[STAGES].mean()
    n_m = len(models)
    bar_w = 0.8 / n_m
    x = np.arange(len(STAGES))

    # 2) Make the figure small (3.3×2.2 inches, same as figure14)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    # 3) Draw bars
    for mi, m in enumerate(models):
        y = [avg.loc[m, s] if m in avg.index else 0 for s in STAGES]
        ax.bar(
            x + mi*bar_w,
            y,
            bar_w,
            label=m
        )

    # 4) Ticks and labels (fontsize now comes from rcParams)
    ax.set_xticks(x + bar_w*(n_m-1)/2)
    ax.set_xticklabels(STAGES, rotation=20, ha="right")  # uses xtick.labelsize=7
    ax.set_ylabel("Avg Latency (ms, p50)")             # uses axes.labelsize=7

    # 5) Title
    ax.set_title(
        "Figure 15: Model-wise Stage Latency Comparison\n(Grouped Bar Chart)",
        fontsize=7
    )

    # 6) Legend (fontsize and title_fontsize from rcParams)
    ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        out_dir/"figure15.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

def figure_15_(df: pd.DataFrame, out_dir: Path):
    models = sorted(set(df.index.get_level_values(1)))
    avg = df.groupby("model")[STAGES].mean()
    n_m = len(models)
    bar_w = 0.8 / n_m
    x = np.arange(len(STAGES))

    fig, ax = plt.subplots(figsize=(max(10, len(STAGES)*1.5), 6))
    for mi, m in enumerate(models):
        y = [avg.loc[m, s] if m in avg.index else 0 for s in STAGES]
        ax.bar(x + mi*bar_w, y, bar_w, label=m)

    ax.set_xticks(x + bar_w*(n_m-1)/2)
    ax.set_xticklabels(STAGES, rotation=20, ha="right")
    ax.set_ylabel("Avg Latency (ms, p50)")
    ax.set_title("Figure 15: Model-wise Stage Latency Comparison (Grouped Bar Chart)")
    ax.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout()
    #fig.savefig(out_dir/"figure15_model_stage.png")
    plt.figure(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })
    fig.savefig(out_dir/"figure15.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def figure_16(df: pd.DataFrame, out_dir: Path):
    # 1) Global style for this plot
    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'legend.title_fontsize': 7,
    })

    datasets = sorted(df.index.get_level_values(0).unique())
    avg = df.groupby("dataset")[STAGES].mean()
    n_d = len(datasets)
    bar_w = 0.8 / n_d
    x = np.arange(len(STAGES))

    # 2) One figure call at final small size
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    # 3) Draw bars
    for di, ds in enumerate(datasets):
        y = [avg.loc[ds, s] if ds in avg.index else 0 for s in STAGES]
        ax.bar(x + di*bar_w, y, bar_w, label=ds)

    # 4) Ticks & labels (font sizes from rcParams)
    ax.set_xticks(x + bar_w*(n_d-1)/2)
    ax.set_xticklabels(STAGES, rotation=20, ha="right")
    ax.set_ylabel("Avg Latency (ms, p50)")

    # 5) Title
    ax.set_title(
        "Figure 16: Dataset-wise\nLatency Contribution per Stage",
        fontsize=7
    )

    # 6) Legend (uses legend.fontsize & legend.title_fontsize)
    ax.legend(
        title="Dataset",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    fig.tight_layout()
    fig.savefig(
        out_dir/"figure16.pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

def figure_16_(df: pd.DataFrame, out_dir: Path):
    datasets = sorted(set(df.index.get_level_values(0)))
    avg = df.groupby("dataset")[STAGES].mean()
    n_d = len(datasets)
    bar_w = 0.8 / n_d
    x = np.arange(len(STAGES))

    fig, ax = plt.subplots(figsize=(max(10, len(STAGES)*1.5), 6))
    for di, ds in enumerate(datasets):
        y = [avg.loc[ds, s] if ds in avg.index else 0 for s in STAGES]
        ax.bar(x + di*bar_w, y, bar_w, label=ds)

    ax.set_xticks(x + bar_w*(n_d-1)/2)
    ax.set_xticklabels(STAGES, rotation=20, ha="right")
    ax.set_ylabel("Avg Latency (ms, p50)")
    ax.set_title("Figure 16: Dataset-wise Latency Contribution per Stage")
    ax.legend(title="Dataset", bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout()
    #fig.savefig(out_dir/"figure16_dataset_stage.png")
    plt.figure(figsize=(3.3, 2.2))
    plt.rcParams.update({
        'font.size': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
    })
    fig.savefig(out_dir/"figure16.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Plot stage-latency breakdown")
    p.add_argument(
        "--results-root", "-r",
        type=Path, required=True,
        help="Root of per-dataset results, e.g. results/benchmark/dataset_masuqur"
    )
    p.add_argument(
        "--output-dir", "-o",
        type=Path, default=Path("results/stage_latency_plots"),
        help="Where to write tables & figures"
    )
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_stage_data(args.results_root)

    table_11_12(df, args.output_dir)
    table_13  (df, args.output_dir)
    table_14  (df, args.output_dir)
    table_15  (df, args.output_dir)
    figure_13 (df, args.output_dir)
    figure_14 (df, args.output_dir)
    figure_15 (df, args.output_dir)
    figure_16 (df, args.output_dir)

    print(f"\n✅ All tables & figures saved under `{args.output_dir}`.")

if __name__=="__main__":
    main()

