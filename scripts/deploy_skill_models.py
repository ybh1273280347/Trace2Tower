from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from trace2tower.agent_factory import build_agent
from trace2tower.config import load_config
from trace2tower.env_factory import build_env
from trace2tower.evaluation import Evaluator
from trace2tower.execution import run_episodes
from trace2tower.io import read_json, write_json, write_jsonl, write_jsonl_dicts
from trace2tower.registry import build_retriever, build_segmenter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Deployment config for env/agent/retriever/runtime.")
    parser.add_argument("--output-root", required=True, help="Directory containing one deployment folder per model.")
    parser.add_argument("--models", nargs="*", default=[], help="model.json paths or label=path entries.")
    parser.add_argument("--include-no-skill", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    specs = []
    if args.include_no_skill:
        specs.append(("no_skill", None))
    specs.extend(_parse_model_specs(args.models))
    if not specs:
        raise ValueError("Provide --models and/or --include-no-skill.")

    summaries = []
    for label, model_path in specs:
        model = read_json(Path(model_path)) if model_path else None
        run_dir = output_root / _slug(label)
        summary = _deploy_one(config, model=model, model_path=model_path, run_dir=run_dir)
        summaries.append(summary)

    write_json(output_root / "deployment_summary.json", {"runs": summaries})


def _deploy_one(
    config: Any,
    *,
    model: dict[str, Any] | None,
    model_path: str | None,
    run_dir: Path,
) -> dict[str, Any]:
    env = build_env(
        config.env_name,
        config.env_mode,
        config.num_products,
        alfworld_config_path=str(config.alfworld_config_path),
        alfworld_data_dir=str(config.alfworld_data_dir),
        webshop_root=str(config.webshop_root),
    )
    agent = build_agent(config.agent_name, config.agent_config)
    segmenter = build_segmenter(config.segmenter_name, config.segmenter_config)
    retriever = build_retriever(config.retriever_name, config.retriever_config)
    evaluator = Evaluator(metrics=config.raw["evaluator"]["metrics"])

    output = run_episodes(
        env=env,
        agent=agent,
        segmenter=segmenter,
        retriever=retriever,
        env_name=config.env_name,
        episodes=config.episodes,
        max_steps=config.max_steps,
        deployment_model=model,
    )
    raw_records = output["records"]
    deployment_retrieval = output["deployment_retrieval"]
    summary = evaluator.summarize(raw_records)
    summary["components"] = {
        "agent": config.agent_name,
        "segmenter": config.segmenter_name,
        "retriever": config.retriever_name,
    }
    summary["deployment_model_path"] = model_path or ""
    summary["deployment_model_method"] = model.get("method", "no_skill") if model else "no_skill"
    summary["deployment_skill_count"] = len(model.get("skills", [])) if model else 0
    summary["avg_deployment_retrieved_skills"] = _avg_retrieved(deployment_retrieval)

    write_jsonl(run_dir / "trajectories.jsonl", output["trajectories"])
    write_jsonl_dicts(run_dir / "records.jsonl", raw_records)
    write_jsonl_dicts(run_dir / "segments.jsonl", output["segments"])
    write_jsonl_dicts(run_dir / "deployment_retrieval.jsonl", deployment_retrieval)
    write_json(run_dir / "model.json", model or _no_skill_model())
    write_json(run_dir / "summary.json", summary)
    return {"experiment_dir": str(run_dir), **summary}


def _parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).parent.name or Path(path).stem
        specs.append((label, path))
    return specs


def _avg_retrieved(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(len(record.get("retrieved_skills", [])) for record in records) / len(records)


def _no_skill_model() -> dict[str, Any]:
    return {
        "method": "no_skill",
        "description": "No deployment skill model.",
        "nodes": [],
        "edges": [],
        "skills": [],
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return slug or "run"


if __name__ == "__main__":
    main()
