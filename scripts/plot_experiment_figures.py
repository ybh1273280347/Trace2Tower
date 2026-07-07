from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  # 注册 science/no-latex 等 matplotlib styles。
import seaborn as sns


METHOD_LABELS = {
    "no_skill": "No Skill",
    "trace2tower": "Trace2Tower",
    "skillx_official": "SkillX",
    "skilllens_official": "SkillLens",
}

GRANULARITY_ORDER = [
    "low",
    "mid",
    "high",
    "planning",
    "functional",
    "atomic",
    "skilllens_skill",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_dir", help="Directory produced by analyze_experiment_table.py.")
    parser.add_argument("--prefix", default="", help="Only use files with this prefix when provided.")
    parser.add_argument("--output-dir", default="", help="Figure output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_paper_style()
    generated: list[str] = []
    runs = _read_analysis_frame(analysis_dir, suffix="runs", prefix=args.prefix)
    granularities = _read_analysis_frame(analysis_dir, suffix="skill_granularity", prefix=args.prefix)

    if not runs.empty:
        generated.extend(_plot_skill_efficiency(runs, output_dir))
        generated.extend(_plot_deployment_metrics(runs, output_dir))
    if not granularities.empty:
        generated.extend(_plot_skill_granularity(granularities, output_dir))

    manifest = {
        "analysis_dir": str(analysis_dir),
        "prefix": args.prefix,
        "generated": generated,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _set_paper_style() -> None:
    plt.style.use(["science", "no-latex", "grid"])
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="colorblind",
        rc={
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def _read_analysis_frame(analysis_dir: Path, *, suffix: str, prefix: str) -> pd.DataFrame:
    pattern = f"{prefix}_{suffix}.csv" if prefix else f"*_{suffix}.csv"
    paths = sorted(analysis_dir.glob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 1]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_skill_efficiency(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    required = {"method", "skill_count", "avg_skill_token_cost"}
    if not required.issubset(frame.columns):
        return []

    data = _method_frame(frame).dropna(subset=["skill_count", "avg_skill_token_cost"])
    if data.empty:
        return []

    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    sns.scatterplot(
        data=data,
        x="avg_skill_token_cost",
        y="skill_count",
        hue="method_label",
        s=48,
        ax=ax,
    )
    ax.set_xlabel("Avg. skill tokens")
    ax.set_ylabel("Skill count")
    _clean_legend(ax, data["method_label"].nunique())
    return _save_figure(fig, output_dir / "skill_efficiency")


def _plot_deployment_metrics(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    required = {"method", "success_rate", "avg_total_token_cost"}
    if not required.issubset(frame.columns):
        return []

    data = _method_frame(frame)
    data = data[data.get("row_type", "") == "run"] if "row_type" in data.columns else data
    data = data.dropna(subset=["success_rate", "avg_total_token_cost"])
    if data.empty:
        return []

    data = data.assign(success_rate_percent=100.0 * data["success_rate"].astype(float))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.15), sharex=False)
    sns.barplot(data=data, x="method_label", y="success_rate_percent", ax=axes[0], color="#4C72B0")
    sns.barplot(data=data, x="method_label", y="avg_total_token_cost", ax=axes[1], color="#55A868")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Success rate (%)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Avg. total tokens")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    return _save_figure(fig, output_dir / "deployment_metrics")


def _plot_skill_granularity(frame: pd.DataFrame, output_dir: Path) -> list[str]:
    required = {"method", "granularity", "skill_count"}
    if not required.issubset(frame.columns):
        return []

    data = _method_frame(frame).dropna(subset=["granularity", "skill_count"])
    if data.empty:
        return []

    pivot = (
        data.pivot_table(
            index="method_label",
            columns="granularity",
            values="skill_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=[item for item in GRANULARITY_ORDER if item in set(data["granularity"])])
        .sort_index()
    )
    if pivot.empty:
        return []

    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
    ax.set_xlabel("")
    ax.set_ylabel("Skill count")
    ax.legend(title="Level", frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.tick_params(axis="x", rotation=25)
    return _save_figure(fig, output_dir / "skill_granularity")


def _method_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["method_label"] = [
        METHOD_LABELS.get(str(method), str(method).replace("_", " ").title())
        for method in data["method"]
    ]
    return data


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    fig.tight_layout(pad=0.4)
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return [str(path) for path in paths]


def _clean_legend(ax: plt.Axes, item_count: int) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    if item_count <= 1:
        legend.remove()
        return
    legend.set_title("")
    legend.set_frame_on(False)


if __name__ == "__main__":
    main()
