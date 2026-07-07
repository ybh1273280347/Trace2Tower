from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2tower.io import read_json, write_json
from trace2tower.mining.refinement import refine_skill_tower


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Input model.json.")
    parser.add_argument("--records", required=True, help="Deployment records.jsonl.")
    parser.add_argument("--deployment-retrieval", required=True, help="Deployment retrieval.jsonl.")
    parser.add_argument("--output", required=True, help="Output refined model path.")
    parser.add_argument("--config", default="", help="Optional refinement config json.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = read_json(Path(args.model))
    records = _read_jsonl(Path(args.records))
    deployment_retrieval = _read_jsonl(Path(args.deployment_retrieval))
    config = read_json(Path(args.config)) if args.config else {}
    refined = refine_skill_tower(
        model,
        records=records,
        deployment_retrieval=deployment_retrieval,
        config=config,
    )
    write_json(Path(args.output), refined)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
