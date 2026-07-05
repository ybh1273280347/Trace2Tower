from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from trace2tower.config import load_config
from trace2tower.io import write_json, write_jsonl_dicts
from trace2tower.registry import build_miner, build_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Miner configs to run on the same segments.")
    parser.add_argument("--segments", required=True, help="Shared training segments.jsonl.")
    parser.add_argument("--records", default="", help="Optional shared records.jsonl for retrieval previews.")
    parser.add_argument("--output-root", required=True, help="Directory containing one model folder per config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    segments = _read_jsonl(Path(args.segments))
    records = _read_jsonl(Path(args.records)) if args.records else []
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for config_path in args.configs:
        config = load_config(config_path)
        model_dir = output_root / Path(config_path).stem
        model_dir.mkdir(parents=True, exist_ok=True)

        miner_config = dict(config.miner_config)
        miner_config.setdefault("runtime_output_dir", str(model_dir))
        model = build_miner(config.miner_name, miner_config).mine(segments)
        write_json(model_dir / "model.json", model)

        retrieval_records = []
        if records:
            retriever = build_retriever(config.retriever_name, config.retriever_config)
            retrieval_records = [
                {
                    "task_id": record.get("task_id"),
                    "retrieved_skills": retriever.retrieve(
                        model,
                        {
                            "goal": record.get("goal", ""),
                            "env": record.get("env", ""),
                            "segments": record.get("segments", []),
                        },
                    ),
                }
                for record in records
            ]
            write_jsonl_dicts(model_dir / "retrieval.jsonl", retrieval_records)

        summary = _model_summary(
            config_path=Path(config_path),
            model_dir=model_dir,
            model=model,
            segment_count=len(segments),
            retrieval_records=retrieval_records,
        )
        write_json(model_dir / "model_summary.json", summary)
        summaries.append(summary)

    write_json(output_root / "mining_summary.json", {"models": summaries})


def _model_summary(
    *,
    config_path: Path,
    model_dir: Path,
    model: dict[str, Any],
    segment_count: int,
    retrieval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    skills = model.get("skills", [])
    token_costs = [float(skill.get("metadata", {}).get("token_cost", 0.0) or 0.0) for skill in skills]
    supports = [float(skill.get("metadata", {}).get("support", 0.0) or 0.0) for skill in skills]
    retrieved_counts = [len(record.get("retrieved_skills", [])) for record in retrieval_records]
    return {
        "config": str(config_path),
        "model_dir": str(model_dir),
        "model_path": str(model_dir / "model.json"),
        "method": model.get("method", ""),
        "skill_count": len(skills),
        "segment_count": segment_count,
        "avg_skill_token_cost": _mean(token_costs),
        "total_skill_token_cost": sum(token_costs),
        "avg_skill_support": _mean(supports),
        "avg_retrieved_skills": _mean(retrieved_counts),
        "official_output_path": model.get("metadata", {}).get("official_output_path", ""),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
