import json
import os
from pathlib import Path

MPLCONFIGDIR = Path("results/plots/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_ROOT = Path("results/selfplay")
LAYOUT_ORDER = ["simple", "unident_s", "random1", "random0", "random3"]
ALGO_ORDER = ["ppo", "a2c"]
ALGO_COLORS = {
    "ppo": "#3b6ea8",
    "a2c": "#d38b2a",
}


def ordered_values(values, preferred_order):
    present = set(values)
    ordered = [value for value in preferred_order if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def collect_evaluations(root: Path) -> pd.DataFrame:
    rows = []

    for eval_path in root.rglob("evaluation.json"):
        with eval_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data["algo"] not in ALGO_ORDER:
            continue

        rows.append({
            "algo": data["algo"],
            "layout": data["layout"],
            "seed": data["seed"],
            "mean_return": data["mean_return"],
            "std_return": data["std_return"],
            "median_return": data["median_return"],
            "min_return": data["min_return"],
            "max_return": data["max_return"],
            "run_dir": str(eval_path.parent),
        })

    if not rows:
        raise FileNotFoundError("No evaluation.json found under results/selfplay")

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["algo", "layout"])["mean_return"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def plot_algo_layout_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    layouts = ordered_values(summary["layout"].unique(), LAYOUT_ORDER)
    algos = ordered_values(summary["algo"].unique(), ALGO_ORDER)

    pivot_mean = (
        summary.pivot(index="layout", columns="algo", values="mean")
        .reindex(index=layouts, columns=algos)
    )
    pivot_std = (
        summary.pivot(index="layout", columns="algo", values="std")
        .reindex(index=layouts, columns=algos)
        .fillna(0.0)
    )

    colors = [ALGO_COLORS.get(algo, None) for algo in pivot_mean.columns]

    ax = pivot_mean.plot(
        kind="bar",
        yerr=pivot_std,
        capsize=4,
        figsize=(10, 6),
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
    )

    ax.set_title("Deterministic Evaluation Return by Layout")
    ax.set_xlabel("Layout")
    ax.set_ylabel("Mean Return Across Seeds")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output = output_dir / "evaluation_summary_bar.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved figure to {output}")


def plot_seed_scatter(df: pd.DataFrame, output_dir: Path) -> None:
    layouts = ordered_values(df["layout"].unique(), LAYOUT_ORDER)
    algos = ordered_values(df["algo"].unique(), ALGO_ORDER)
    layout_to_x = {layout: idx for idx, layout in enumerate(layouts)}

    fig, ax = plt.subplots(figsize=(10, 6))

    offsets = {
        algo: (idx - (len(algos) - 1) / 2) * 0.18
        for idx, algo in enumerate(algos)
    }
    for algo in algos:
        sub = df[df["algo"] == algo]
        x = [layout_to_x[layout] + offsets[algo] for layout in sub["layout"]]
        ax.scatter(
            x,
            sub["mean_return"],
            label=algo,
            alpha=0.85,
            s=64,
            color=ALGO_COLORS.get(algo),
            edgecolor="#333333",
            linewidth=0.5,
        )

    ax.set_title("Evaluation Return by Seed")
    ax.set_xlabel("Layout")
    ax.set_ylabel("Mean Return")
    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels(layouts, rotation=25, ha="right")
    ax.legend(title="Algorithm")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output = output_dir / "evaluation_seed_scatter.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved figure to {output}")


def plot_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    layouts = ordered_values(summary["layout"].unique(), LAYOUT_ORDER)
    algos = ordered_values(summary["algo"].unique(), ALGO_ORDER)
    pivot = (
        summary.pivot(index="layout", columns="algo", values="mean")
        .reindex(index=layouts, columns=algos)
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    image = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_title("Mean Evaluation Return Heatmap")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row_idx, layout in enumerate(pivot.index):
        for col_idx, algo in enumerate(pivot.columns):
            value = pivot.loc[layout, algo]
            label = "" if pd.isna(value) else f"{value:.1f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center")

    fig.colorbar(image, ax=ax, label="Mean return across seeds")
    plt.tight_layout()

    output = output_dir / "evaluation_mean_heatmap.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved figure to {output}")


def plot_box_by_algo(df: pd.DataFrame, output_dir: Path) -> None:
    algos = ordered_values(df["algo"].unique(), ALGO_ORDER)
    values = [df[df["algo"] == algo]["mean_return"].values for algo in algos]

    fig, ax = plt.subplots(figsize=(7, 5))
    box = ax.boxplot(values, tick_labels=algos, patch_artist=True)
    for patch, algo in zip(box["boxes"], algos):
        patch.set_facecolor(ALGO_COLORS.get(algo, "#cccccc"))
        patch.set_alpha(0.75)

    ax.set_title("Distribution of Evaluation Returns")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Return per Run")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()

    output = output_dir / "evaluation_algo_boxplot.png"
    plt.savefig(output, dpi=200)
    plt.close()
    print(f"Saved figure to {output}")


def main() -> None:
    df = collect_evaluations(RESULT_ROOT)
    summary = summarize(df)

    csv_output = RESULT_ROOT / "evaluation_summary.csv"
    df.to_csv(csv_output, index=False)
    print(f"Saved table to {csv_output}")

    summary_output = RESULT_ROOT / "evaluation_grouped_summary.csv"
    summary.to_csv(summary_output, index=False)
    print(f"Saved grouped table to {summary_output}")
    print(summary.sort_values(["layout", "algo"]).to_string(index=False))

    plot_algo_layout_summary(summary, RESULT_ROOT)
    plot_seed_scatter(df, RESULT_ROOT)
    plot_heatmap(summary, RESULT_ROOT)
    plot_box_by_algo(df, RESULT_ROOT)


if __name__ == "__main__":
    main()
