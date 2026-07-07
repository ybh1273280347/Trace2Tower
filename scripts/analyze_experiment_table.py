from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


RUN_MARKDOWN_COLUMNS = [
    "row_type",
    "method",
    "episodes",
    "success_rate",
    "avg_reward",
    "avg_steps",
    "avg_total_token_cost",
    "avg_token_cost",
    "avg_retrieval_embedding_token_cost",
    "success_per_1k_tokens",
    "skill_count",
    "refined_skill_count",
    "split_child_count",
    "merged_skill_count",
    "promoted_high_skill_count",
    "reinforced_high_skill_count",
    "avg_skill_token_cost",
    "avg_skill_support",
    "avg_retrieved_skills",
    "avg_deployment_retrieved_skills",
]

SELECTOR_MARKDOWN_COLUMNS = [
    "selector",
    "model_method",
    "skill_count",
    "future_records",
    "future_utility_pearson",
    "future_utility_spearman",
    "positive_utility_auc",
    "regret",
    "avg_selected_utility",
    "avg_selected_token_cost",
]

GRANULARITY_MARKDOWN_COLUMNS = [
    "method",
    "granularity",
    "skill_count",
    "avg_support",
    "avg_token_cost",
    "avg_success_rate",
]

DISPLAY_COLUMNS = {
    "row_type": "Type",
    "method": "Method",
    "experiment_dir": "Run",
    "episodes": "N",
    "success_rate": "SR (%)",
    "avg_reward": "Reward",
    "avg_steps": "Steps",
    "avg_total_token_cost": "Tok.",
    "avg_token_cost": "LLM Tok.",
    "avg_retrieval_embedding_token_cost": "Emb. Tok.",
    "success_per_1k_tokens": "SR/1k Tok.",
    "skill_count": "#Skill",
    "refined_skill_count": "#Refined",
    "split_child_count": "Split",
    "merged_skill_count": "Merge",
    "promoted_high_skill_count": "Promote",
    "reinforced_high_skill_count": "Reinforce",
    "avg_skill_token_cost": "Skill Tok.",
    "avg_skill_support": "Supp.",
    "avg_retrieved_skills": "Ret.",
    "avg_deployment_retrieved_skills": "Deploy Ret.",
    "selector": "Selector",
    "model_method": "Model",
    "future_records": "Future N",
    "future_utility_pearson": "Pearson",
    "future_utility_spearman": "Spearman",
    "positive_utility_auc": "AUC",
    "regret": "Regret",
    "avg_selected_utility": "Sel. Util.",
    "avg_selected_token_cost": "Sel. Tok.",
    "batch_index": "Batch",
    "granularity": "Level",
    "avg_support": "Supp.",
    "avg_success_rate": "SR (%)",
}

PERCENT_COLUMNS = {
    "success_rate",
    "avg_success_rate",
}

INTEGER_COLUMNS = {
    "episodes",
    "skill_count",
    "refined_skill_count",
    "split_child_count",
    "merged_skill_count",
    "promoted_high_skill_count",
    "reinforced_high_skill_count",
    "future_records",
    "batch_index",
}

DISPLAY_VALUES = {
    "row_type": {
        "run": "Deploy",
        "mining_model": "Model",
    },
    "method": {
        "no_skill": "No Skill",
        "trace2tower": "Trace2Tower",
        "skillx_official": "SkillX",
        "skilllens_official": "SkillLens",
    },
    "model_method": {
        "no_skill": "No Skill",
        "trace2tower": "Trace2Tower",
        "skillx_official": "SkillX",
        "skilllens_official": "SkillLens",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="Experiment roots, run directories, config files, or summary json files.",
    )
    parser.add_argument("--output-dir", default="experiments/analysis")
    parser.add_argument("--prefix", default="comparison")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect the exact directories passed in paths.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    discovered = _discover_inputs(
        [Path(path) for path in args.paths],
        recursive=not args.no_recursive,
    )

    run_rows: list[dict[str, Any]] = []
    skill_rows: list[dict[str, Any]] = []
    for experiment_dir in _sorted_paths(discovered["run_dirs"]):
        row, skills = _run_row(experiment_dir)
        if row:
            run_rows.append(row)
            skill_rows.extend(skills)

    for model_dir in _sorted_paths(discovered["model_dirs"]):
        row, skills = _model_summary_row(model_dir)
        if row:
            run_rows.append(row)
            skill_rows.extend(skills)

    selector_rows: list[dict[str, Any]] = []
    selector_batch_rows: list[dict[str, Any]] = []
    for selector_path in _sorted_paths(discovered["selector_files"]):
        rows, batch_rows = _selector_rows(selector_path)
        selector_rows.extend(rows)
        selector_batch_rows.extend(batch_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    generated.extend(
        _write_dataframe_set(
            output_dir=output_dir,
            name=f"{args.prefix}_runs",
            rows=run_rows,
            markdown_columns=RUN_MARKDOWN_COLUMNS,
        )
    )
    generated.extend(
        _write_dataframe_set(
            output_dir=output_dir,
            name=f"{args.prefix}_selectors",
            rows=selector_rows,
            markdown_columns=SELECTOR_MARKDOWN_COLUMNS,
        )
    )
    generated.extend(
        _write_dataframe_set(
            output_dir=output_dir,
            name=f"{args.prefix}_selector_batches",
            rows=selector_batch_rows,
            markdown_columns=["batch_index", *SELECTOR_MARKDOWN_COLUMNS],
        )
    )
    generated.extend(
        _write_dataframe_set(
            output_dir=output_dir,
            name=f"{args.prefix}_skills",
            rows=skill_rows,
            markdown_columns=[],
        )
    )

    granularity_rows = _granularity_rows(skill_rows)
    generated.extend(
        _write_dataframe_set(
            output_dir=output_dir,
            name=f"{args.prefix}_skill_granularity",
            rows=granularity_rows,
            markdown_columns=GRANULARITY_MARKDOWN_COLUMNS,
        )
    )

    manifest = {
        "inputs": [str(path) for path in args.paths],
        "recursive": not args.no_recursive,
        "generated": generated,
        "row_counts": {
            "runs": len(run_rows),
            "selectors": len(selector_rows),
            "selector_batches": len(selector_batch_rows),
            "skills": len(skill_rows),
            "skill_granularity": len(granularity_rows),
        },
    }
    (output_dir / f"{args.prefix}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _discover_inputs(paths: list[Path], *, recursive: bool) -> dict[str, set[Path]]:
    discovered = {
        "run_dirs": set(),
        "model_dirs": set(),
        "selector_files": set(),
    }
    for path in paths:
        _discover_path(path, recursive=recursive, discovered=discovered)
    return discovered


def _discover_path(path: Path, *, recursive: bool, discovered: dict[str, set[Path]]) -> None:
    if path.is_file():
        _discover_file(path, discovered=discovered)
        return
    if not path.is_dir():
        return

    candidates = list(path.rglob("*.json")) if recursive else list(path.glob("*.json"))
    for candidate in candidates:
        if candidate.name in {
            "summary.json",
            "model_summary.json",
            "selector_metrics.json",
            "deployment_summary.json",
            "mining_summary.json",
        }:
            _discover_file(candidate, discovered=discovered)


def _discover_file(path: Path, *, discovered: dict[str, set[Path]]) -> None:
    if path.name == "summary.json":
        discovered["run_dirs"].add(path.parent)
        return
    if path.name == "model_summary.json":
        discovered["model_dirs"].add(path.parent)
        return
    if path.name == "selector_metrics.json":
        discovered["selector_files"].add(path)
        return
    if path.name == "deployment_summary.json":
        for run in _read_json(path).get("runs", []):
            experiment_dir = run.get("experiment_dir")
            if experiment_dir:
                discovered["run_dirs"].add(Path(experiment_dir))
        return
    if path.name == "mining_summary.json":
        for model in _read_json(path).get("models", []):
            model_dir = model.get("model_dir")
            if model_dir:
                discovered["model_dirs"].add(Path(model_dir))
        return

    data = _read_json(path)
    output_dir = data.get("runtime", {}).get("output_dir") if isinstance(data, dict) else None
    if output_dir:
        discovered["run_dirs"].add(Path(output_dir))


def _run_row(experiment_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        return {}, []

    summary = _read_json(summary_path)
    model_path = experiment_dir / "model.json"
    model = _read_json(model_path) if model_path.exists() else {}
    row = {
        "row_type": "run",
        "experiment_dir": str(experiment_dir),
        "summary_path": str(summary_path),
        "model_path": str(model_path) if model_path.exists() else "",
    }
    _merge_scalars(row, summary)

    method = row.get("deployment_model_method") or row.get("model_method") or model.get("method", "")
    row["method"] = method
    _merge_model_stats(row, model)
    _merge_refinement_stats(row, experiment_dir / "refined_model.json")
    row["official_output_path"] = model.get("metadata", {}).get("official_output_path", "")
    row.setdefault("avg_retrieved_skills", _avg_retrieved(experiment_dir / "retrieval.jsonl"))
    row.setdefault(
        "avg_deployment_retrieved_skills",
        _avg_retrieved(experiment_dir / "deployment_retrieval.jsonl"),
    )
    row["success_per_1k_tokens"] = _success_per_1k_tokens(row)
    return row, _skill_rows(model, experiment_dir=experiment_dir, method=str(method), row_type="run")


def _model_summary_row(model_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = model_dir / "model_summary.json"
    if not summary_path.exists():
        return {}, []

    summary = _read_json(summary_path)
    model_path = Path(summary.get("model_path") or model_dir / "model.json")
    model = _read_json(model_path) if model_path.exists() else {}
    row = {
        "row_type": "mining_model",
        "experiment_dir": str(model_dir),
        "summary_path": str(summary_path),
        "model_path": str(model_path) if model_path.exists() else "",
    }
    _merge_scalars(row, summary)

    method = row.get("method") or model.get("method", "")
    row["method"] = method
    _merge_model_stats(row, model)
    _merge_refinement_stats(row, model_dir / "refined_model.json")
    row["official_output_path"] = (
        row.get("official_output_path")
        or model.get("metadata", {}).get("official_output_path", "")
    )
    row["success_per_1k_tokens"] = _success_per_1k_tokens(row)
    return row, _skill_rows(model, experiment_dir=model_dir, method=str(method), row_type="mining_model")


def _selector_rows(selector_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = _read_json(selector_path)
    split = data.get("split", {})
    rows = []
    for selector, metrics in data.get("selectors", {}).items():
        row = {
            "selector_metrics_path": str(selector_path),
            "selector": selector,
            "config": data.get("config", ""),
            "records": data.get("records", ""),
            "miner": data.get("miner", ""),
            "model_method": data.get("model_method", ""),
            "top_k": data.get("top_k", 0),
        }
        row.update({key: value for key, value in split.items() if _is_scalar(value)})
        row.update(metrics)
        rows.append(row)

    batch_rows = []
    for batch in data.get("batches", []):
        for selector, metrics in batch.get("selectors", {}).items():
            row = {
                "selector_metrics_path": str(selector_path),
                "selector": selector,
                "config": data.get("config", ""),
                "records": data.get("records", ""),
                "miner": data.get("miner", ""),
                "model_method": data.get("model_method", ""),
                "top_k": data.get("top_k", 0),
                "batch_index": batch.get("batch_index", 0),
                "task_start": batch.get("task_start", 0),
                "task_end": batch.get("task_end", 0),
                "task_count": batch.get("task_count", 0),
            }
            row.update({key: value for key, value in split.items() if _is_scalar(value)})
            row.update(metrics)
            batch_rows.append(row)
    return rows, batch_rows


def _skill_rows(
    model: dict[str, Any],
    *,
    experiment_dir: Path,
    method: str,
    row_type: str,
) -> list[dict[str, Any]]:
    rows = []
    for skill in model.get("skills", []):
        metadata = skill.get("metadata", {})
        rows.append(
            {
                "row_type": row_type,
                "experiment_dir": str(experiment_dir),
                "method": method,
                "skill_id": skill.get("skill_id", ""),
                "name": skill.get("name", ""),
                "granularity": skill.get("granularity", ""),
                "support": _float(metadata.get("support", len(skill.get("members", [])))),
                "token_cost": _float(metadata.get("token_cost", 0.0)),
                "success_rate": _float(metadata.get("success_rate", 0.0)),
                "avg_reward": _float(metadata.get("avg_reward", 0.0)),
                "coverage": _float(metadata.get("coverage", 0.0)),
            }
        )
    return rows


def _granularity_rows(skill_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not skill_rows:
        return []

    frame = pd.DataFrame(skill_rows)
    grouped = (
        frame.groupby(["method", "experiment_dir", "granularity"], dropna=False)
        .agg(
            skill_count=("skill_id", "count"),
            avg_support=("support", "mean"),
            avg_token_cost=("token_cost", "mean"),
            avg_success_rate=("success_rate", "mean"),
            avg_reward=("avg_reward", "mean"),
            avg_coverage=("coverage", "mean"),
        )
        .reset_index()
    )
    return _records(grouped)


def _merge_scalars(row: dict[str, Any], data: dict[str, Any]) -> None:
    for key, value in data.items():
        if key == "components" and isinstance(value, dict):
            for component_name, component_value in value.items():
                row[f"component_{component_name}"] = component_value
        elif _is_scalar(value):
            row[key] = value


def _merge_model_stats(row: dict[str, Any], model: dict[str, Any]) -> None:
    skills = model.get("skills", [])
    metadata = [skill.get("metadata", {}) for skill in skills]
    token_costs = [_float(item.get("token_cost", 0.0)) for item in metadata]
    supports = [_float(item.get("support", 0.0)) for item in metadata]
    success_rates = [_float(item.get("success_rate", 0.0)) for item in metadata]
    rewards = [_float(item.get("avg_reward", 0.0)) for item in metadata]
    coverages = [_float(item.get("coverage", 0.0)) for item in metadata]

    row.setdefault("skill_count", len(skills))
    row.setdefault("avg_skill_token_cost", _mean(token_costs))
    row.setdefault("total_skill_token_cost", sum(token_costs))
    row.setdefault("avg_skill_support", _mean(supports))
    row.setdefault("avg_skill_success_rate", _mean(success_rates))
    row.setdefault("avg_skill_reward", _mean(rewards))
    row.setdefault("avg_skill_coverage", _mean(coverages))


def _merge_refinement_stats(row: dict[str, Any], refined_model_path: Path) -> None:
    if not refined_model_path.exists():
        return
    refined = _read_json(refined_model_path)
    refinement = refined.get("metadata", {}).get("refinement", {})
    structural = refinement.get("structural_updates", {})
    row["refined_skill_count"] = len(refined.get("skills", []))
    row["split_child_count"] = structural.get("split_child_count", 0)
    row["merged_skill_count"] = structural.get("merged_skill_count", 0)
    row["promoted_high_skill_count"] = structural.get("promoted_high_skill_count", 0)
    row["reinforced_high_skill_count"] = structural.get("reinforced_high_skill_count", 0)


def _write_dataframe_set(
    *,
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    markdown_columns: list[str],
) -> list[str]:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [column for column in ["row_type", "method", "selector", "experiment_dir"] if column in frame]
        if sort_columns:
            frame = frame.sort_values(sort_columns)

    csv_path = output_dir / f"{name}.csv"
    json_path = output_dir / f"{name}.json"
    markdown_path = output_dir / f"{name}.md"
    latex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(_records(frame), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(markdown_path, frame, columns=markdown_columns)
    _write_latex(latex_path, frame, columns=markdown_columns)
    return [str(csv_path), str(json_path), str(markdown_path), str(latex_path)]


def _write_markdown(path: Path, frame: pd.DataFrame, *, columns: list[str]) -> None:
    if frame.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
        return

    selected = [column for column in columns if column in frame.columns] if columns else list(frame.columns)
    view = frame[selected]
    headers = [_display_column(column) for column in selected]
    alignments = [_markdown_alignment(column) for column in selected]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(_markdown_cell(row[column], column=column) for column in selected) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(path: Path, frame: pd.DataFrame, *, columns: list[str]) -> None:
    if frame.empty:
        path.write_text("% No rows.\n", encoding="utf-8")
        return

    selected = [column for column in columns if column in frame.columns] if columns else list(frame.columns)
    view = _latex_view(frame, selected)
    headers = {
        column: _latex_escape(_display_column(column))
        for column in selected
    }
    column_format = "".join("S" if _latex_numeric_column(frame, column) else "l" for column in selected)
    formatters = {
        headers[column]: _latex_number_formatter(column)
        for column in selected
        if _latex_numeric_column(frame, column)
    }
    latex = (
        view.rename(columns=headers)
        .style
        .hide(axis="index")
        .format(formatters, na_rep="", escape="latex")
        .to_latex(
            hrules=True,
            siunitx=True,
            column_format=column_format,
        )
    )
    preamble = [
        "% Requires: \\usepackage{booktabs}",
        "% Numeric columns use siunitx S alignment: \\usepackage{siunitx}",
    ]
    path.write_text("\n".join(preamble) + "\n" + latex, encoding="utf-8")


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notnull(frame), None)
    return clean.to_dict(orient="records")


def _latex_view(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    view = pd.DataFrame()
    for column in columns:
        if _latex_numeric_column(frame, column):
            values = pd.to_numeric(frame[column], errors="coerce")
            if column in PERCENT_COLUMNS:
                values = values * 100.0
            view[column] = values
        else:
            view[column] = [
                _display_value(column, value) or str(value).replace("\n", " ")
                for value in frame[column]
            ]
    return view


def _latex_numeric_column(frame: pd.DataFrame, column: str) -> bool:
    if column in DISPLAY_VALUES:
        return False
    values = pd.to_numeric(frame[column], errors="coerce")
    return bool(values.notna().any())


def _latex_number_formatter(column: str) -> str:
    if column in PERCENT_COLUMNS:
        return "{:.1f}"
    if column in INTEGER_COLUMNS:
        return "{:.0f}"
    return "{:.3g}"


def _avg_retrieved(path: Path) -> float:
    records = _read_jsonl(path)
    if not records:
        return 0.0
    counts = [len(record.get("retrieved_skills", [])) for record in records]
    return _mean([float(count) for count in counts])


def _success_per_1k_tokens(row: dict[str, Any]) -> float:
    token_cost = _float(row.get("avg_total_token_cost", 0.0))
    if token_cost <= 0:
        token_cost = _float(row.get("avg_token_cost", 0.0))
    if token_cost <= 0:
        return 0.0
    return 1000.0 * _float(row.get("success_rate", 0.0)) / token_cost


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _float(value: Any) -> float:
    return float(value or 0.0)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _sorted_paths(paths: set[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: str(path))


def _markdown_cell(value: Any, *, column: str) -> str:
    if value is None or pd.isna(value):
        return ""
    display_value = _display_value(column, value)
    if display_value is not None:
        return display_value
    if column in INTEGER_COLUMNS:
        return str(int(round(float(value))))
    if column in PERCENT_COLUMNS:
        return f"{100.0 * float(value):.1f}"
    if isinstance(value, float):
        return _format_number(value)
    return str(value).replace("\n", " ").replace("|", "\\|")


def _display_column(column: str) -> str:
    return DISPLAY_COLUMNS.get(column, column.replace("_", " ").title())


def _display_value(column: str, value: Any) -> str | None:
    values = DISPLAY_VALUES.get(column)
    if not values:
        return None
    text = str(value)
    return values.get(text, text.replace("_", " ").title())


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    absolute = abs(value)
    if absolute >= 1000:
        return f"{value / 1000.0:.2g}k"
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _markdown_alignment(column: str) -> str:
    if column in INTEGER_COLUMNS or column in PERCENT_COLUMNS:
        return "---:"
    return "---:" if column.startswith("avg_") or column in {"regret", "success_per_1k_tokens"} else "---"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in value)


if __name__ == "__main__":
    main()
