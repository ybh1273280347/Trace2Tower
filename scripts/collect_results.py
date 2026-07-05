from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Experiment directories or config files.")
    parser.add_argument("--output-json", default="experiments/results_summary.json")
    parser.add_argument("--output-csv", default="experiments/results_summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = [_collect_path(Path(path)) for path in args.paths]
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError("No experiment summaries found.")

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_path(path: Path) -> dict[str, Any]:
    experiment_dir = _resolve_experiment_dir(path)
    summary_path = experiment_dir / "summary.json"
    if not summary_path.exists():
        return {}

    # summary 给任务指标，model 给技能数量，retrieval 给平均每题实际取出的技能数。
    summary = _read_json(summary_path)
    model = _read_json(experiment_dir / "model.json") if (experiment_dir / "model.json").exists() else {}
    retrieval_records = _read_jsonl(experiment_dir / "retrieval.jsonl")
    deployment_retrieval_records = _read_jsonl(experiment_dir / "deployment_retrieval.jsonl")
    avg_retrieved = (
        sum(len(record.get("retrieved_skills", [])) for record in retrieval_records) / len(retrieval_records)
        if retrieval_records
        else 0.0
    )
    avg_deployment_retrieved = (
        sum(len(record.get("retrieved_skills", [])) for record in deployment_retrieval_records)
        / len(deployment_retrieval_records)
        if deployment_retrieval_records
        else 0.0
    )
    row = {
        "experiment_dir": str(experiment_dir),
        "model_method": summary.get("model_method", model.get("method", "")),
        "skill_count": summary.get("skill_count", len(model.get("skills", []))),
        "avg_retrieved_skills": avg_retrieved,
        "avg_deployment_retrieved_skills": avg_deployment_retrieved,
    }
    for key, value in summary.items():
        if key == "components":
            # 把嵌套 components 展平成 CSV 列，方便直接做论文表或 pandas 分析。
            for component_name, component_value in value.items():
                row[f"component_{component_name}"] = component_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    return row


def _resolve_experiment_dir(path: Path) -> Path:
    if path.suffix == ".json" and path.exists():
        config = _read_json(path)
        return Path(config["runtime"]["output_dir"])
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
