"""
plot_results.py
Generate plots for the APS Coverage Path Planning report.

Reads CSVs from results/ and produces 3 PNGs in results/:
  1. coverage_bars.png       Bar chart: mean coverage with std error bars
                             across all approved grid sizes.
  2. coverage_distribution.png  Box+strip plot: distribution of per-episode
                             coverage for each size (shows bimodality).
  3. obstacle_density_20x20.png  Effect of obstacle density on the 20x20:
                             compares bigtwenty easy(16) vs medium(32)
                             vs hard(48) obstacles.

Usage:
    python plot_results.py

Requires: pandas, matplotlib, seaborn  (already in requirements.txt)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

# Consistent color palette
PALETTE = {
    "5x5":     "#2E86AB",
    "10x10":   "#06A77D",
    "15x15":   "#F18F01",
    "20x20":   "#C73E1D",
    "easy":    "#52B788",
    "medium":  "#F4A261",
    "hard":    "#E76F51",
}


def _safe_load(path: Path) -> pd.DataFrame | None:
    """Load eval CSV. Returns None if missing or empty."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "coverage" not in df.columns:
        return None
    return df


def _summary(df: pd.DataFrame) -> dict:
    cov = df["coverage"].astype(float)
    return {
        "mean": cov.mean(),
        "std": cov.std(),
        "min": cov.min(),
        "max": cov.max(),
        "full_rate": (cov >= 0.999).mean(),
        "n": len(cov),
    }


def _pick_latest(pattern: str) -> Path | None:
    """Return the most recently modified file matching the glob pattern, or
    None if there are no matches."""
    matches = list(RESULTS_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Plot 1 — coverage bars with std error bars
# ---------------------------------------------------------------------------
def plot_coverage_bars(out_path: Path) -> None:
    """Mean coverage per size with std error bars."""
    sizes = ["5x5", "10x10", "15x15", "20x20"]
    rows = []
    for s in sizes:
        df = _safe_load(RESULTS_DIR / f"cpp_{s}_approved_eval.csv")
        if df is None:
            continue
        st = _summary(df)
        rows.append({"size": s, **st})

    if not rows:
        print("[plot_coverage_bars] no eval CSVs found, skipping")
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [PALETTE.get(s, "#888888") for s in df["size"]]
    bars = ax.bar(
        df["size"], df["mean"] * 100,
        yerr=df["std"] * 100,
        capsize=8, color=colors, edgecolor="black", linewidth=0.7,
    )
    ax.axhline(90, color="red", linestyle="--", alpha=0.6, label="threshold 90%")
    ax.set_ylabel("Mean coverage (%)")
    ax.set_xlabel("Grid size")
    ax.set_title("Cobertura média por tamanho de grid (200 episódios cada)", pad=15)
    ax.set_ylim(0, 130)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")

    # Annotate bars with the actual value (placed below top of bar to avoid title)
    for bar, m, s in zip(bars, df["mean"], df["std"]):
        # Place the annotation just above the error-bar top, but cap so it
        # never overlaps the title region above 120.
        y_top = (m * 100) + (s * 100) + 2
        if y_top > 118:
            y_top = 118
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_top,
            f"{m:.1%}\n±{s:.1%}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_coverage_bars] saved to {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — per-episode coverage distribution (shows bimodality)
# ---------------------------------------------------------------------------
def plot_coverage_distribution(out_path: Path) -> None:
    """Distribution of per-episode coverage per size — exposes bimodality."""
    sizes = ["5x5", "10x10", "15x15", "20x20"]
    frames = []
    for s in sizes:
        df = _safe_load(RESULTS_DIR / f"cpp_{s}_approved_eval.csv")
        if df is None:
            continue
        df = df.copy()
        df["size"] = s
        frames.append(df[["size", "coverage"]])

    if not frames:
        print("[plot_coverage_distribution] no eval CSVs found, skipping")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df["coverage_pct"] = all_df["coverage"] * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.violinplot(
        data=all_df, x="size", y="coverage_pct",
        hue="size", legend=False,
        palette=[PALETTE[s] for s in sizes if s in all_df["size"].unique()],
        inner=None, cut=0, ax=ax,
    )
    sns.stripplot(
        data=all_df, x="size", y="coverage_pct",
        color="black", alpha=0.25, size=2.5, jitter=0.25, ax=ax,
    )
    ax.axhline(90, color="red", linestyle="--", alpha=0.6, label="threshold 90%")
    ax.set_ylabel("Coverage por episódio (%)")
    ax.set_xlabel("Grid size")
    ax.set_title("Distribuição de cobertura por episódio (200 episódios cada)")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_coverage_distribution] saved to {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — effect of obstacle density on the 20x20
# ---------------------------------------------------------------------------
def plot_obstacle_density_20x20(out_path: Path) -> None:
    """Compare bigtwenty easy(16) vs medium(32) vs hard(48) on 20x20."""
    spec = [
        ("easy",   16, "cpp_20x20_bigtwenty_easy_*_eval.csv"),
        ("medium", 32, "cpp_20x20_bigtwenty_medium_*_eval.csv"),
        ("hard",   48, "cpp_20x20_bigtwenty_hard_*_eval.csv"),
    ]
    rows = []
    distros = []
    for label, n_obs, pat in spec:
        path = _pick_latest(pat)
        if path is None:
            continue
        df = _safe_load(path)
        if df is None:
            continue
        st = _summary(df)
        rows.append({"label": label, "n_obs": n_obs, **st})
        d = df.copy()
        d["label"] = f"{label} ({n_obs} obs)"
        distros.append(d[["label", "coverage"]])

    if not rows:
        print("[plot_obstacle_density_20x20] no bigtwenty CSVs found, skipping")
        return

    summary = pd.DataFrame(rows)
    all_dist = pd.concat(distros, ignore_index=True)
    all_dist["coverage_pct"] = all_dist["coverage"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: bar chart with std error bars
    ax = axes[0]
    colors = [PALETTE.get(r["label"], "#888888") for _, r in summary.iterrows()]
    labels = [f"{r['label']}\n({r['n_obs']} obs)" for _, r in summary.iterrows()]
    bars = ax.bar(
        labels, summary["mean"] * 100,
        yerr=summary["std"] * 100,
        capsize=8, color=colors, edgecolor="black", linewidth=0.7,
    )
    ax.axhline(90, color="red", linestyle="--", alpha=0.6, label="threshold 90%")
    ax.set_ylabel("Mean coverage (%)")
    ax.set_title("Cobertura média no 20x20 vs densidade de obstáculos", pad=15)
    ax.set_ylim(0, 130)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    for bar, m, s in zip(bars, summary["mean"], summary["std"]):
        y_top = (m * 100) + (s * 100) + 2
        if y_top > 118:
            y_top = 118
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_top,
            f"{m:.1%}\n±{s:.1%}",
            ha="center", va="bottom", fontsize=9,
        )

    # Right: distribution
    ax = axes[1]
    label_order = [f"{r['label']} ({r['n_obs']} obs)" for _, r in summary.iterrows()]
    palette = [PALETTE.get(r["label"], "#888888") for _, r in summary.iterrows()]
    sns.violinplot(
        data=all_dist, x="label", y="coverage_pct",
        order=label_order,
        hue="label", hue_order=label_order, legend=False,
        palette=palette,
        inner=None, cut=0, ax=ax,
    )
    sns.stripplot(
        data=all_dist, x="label", y="coverage_pct",
        order=label_order,
        color="black", alpha=0.25, size=2.5, jitter=0.25, ax=ax,
    )
    ax.axhline(90, color="red", linestyle="--", alpha=0.6)
    ax.set_ylabel("Coverage por episódio (%)")
    ax.set_xlabel("")
    ax.set_title("Distribuição de cobertura — 20x20 por densidade de obstáculos")
    ax.set_ylim(-5, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot_obstacle_density_20x20] saved to {out_path}")


# ---------------------------------------------------------------------------
# Plot 4 — comparison table CSV (summary numbers for the report)
# ---------------------------------------------------------------------------
def write_comparison_csv(out_path: Path) -> None:
    """Write a one-row-per-config CSV with the headline numbers."""
    rows = []

    for s in ["5x5", "10x10", "15x15", "20x20"]:
        df = _safe_load(RESULTS_DIR / f"cpp_{s}_approved_eval.csv")
        if df is None:
            continue
        st = _summary(df)
        rows.append({
            "config": f"{s} (approved)",
            "size": s,
            "n_episodes": st["n"],
            "mean_coverage": round(st["mean"], 4),
            "std_coverage": round(st["std"], 4),
            "min_coverage": round(st["min"], 4),
            "max_coverage": round(st["max"], 4),
            "full_coverage_rate": round(st["full_rate"], 4),
        })

    for label, n_obs, pat in [
        ("easy",   16, "cpp_20x20_bigtwenty_easy_*_eval.csv"),
        ("medium", 32, "cpp_20x20_bigtwenty_medium_*_eval.csv"),
        ("hard",   48, "cpp_20x20_bigtwenty_hard_*_eval.csv"),
    ]:
        path = _pick_latest(pat)
        if path is None:
            continue
        df = _safe_load(path)
        if df is None:
            continue
        st = _summary(df)
        rows.append({
            "config": f"20x20 bigtwenty {label} ({n_obs} obs)",
            "size": "20x20",
            "n_episodes": st["n"],
            "mean_coverage": round(st["mean"], 4),
            "std_coverage": round(st["std"], 4),
            "min_coverage": round(st["min"], 4),
            "max_coverage": round(st["max"], 4),
            "full_coverage_rate": round(st["full_rate"], 4),
        })

    if not rows:
        print("[write_comparison_csv] no data to write")
        return
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[write_comparison_csv] saved to {out_path}")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
    RESULTS_DIR.mkdir(exist_ok=True)

    plot_coverage_bars(RESULTS_DIR / "coverage_bars.png")
    plot_coverage_distribution(RESULTS_DIR / "coverage_distribution.png")
    plot_obstacle_density_20x20(RESULTS_DIR / "obstacle_density_20x20.png")
    write_comparison_csv(RESULTS_DIR / "comparison_table.csv")

    print("\nAll plots saved to results/")
    print("Files generated:")
    for f in ["coverage_bars.png", "coverage_distribution.png",
              "obstacle_density_20x20.png", "comparison_table.csv"]:
        p = RESULTS_DIR / f
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
