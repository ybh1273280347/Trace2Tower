from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trace2tower.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--miner-configs", nargs="+", required=True)
    parser.add_argument("--deployment-config", required=True)
    parser.add_argument(
        "--train-config",
        default="",
        help="Config used to collect shared training records when --segments is not provided.",
    )
    parser.add_argument("--segments", default="")
    parser.add_argument("--records", default="")
    parser.add_argument("--exclude-no-skill", action="store_true")
    parser.add_argument("--skip-selectors", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--selector-future-ratio", type=float, default=0.4)
    parser.add_argument("--selector-top-k", type=int, default=3)
    parser.add_argument("--selector-batch-size", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    commands = []
    segments = Path(args.segments) if args.segments else None
    records = Path(args.records) if args.records else None
    if segments is None:
        if not args.train_config:
            raise ValueError("Provide --segments or --train-config.")
        train_config = load_config(args.train_config)
        command = [sys.executable, "-m", "trace2tower.runtime.run", "--config", args.train_config]
        _run(command)
        commands.append(command)
        segments = train_config.output_dir / "segments.jsonl"
        records = train_config.output_dir / "records.jsonl"
    elif records is None:
        raise ValueError("Provide --records when using --segments.")

    models_root = output_root / "models"
    mine_command = [
        sys.executable,
        "scripts/mine_skill_models.py",
        *args.miner_configs,
        "--segments",
        str(segments),
        "--records",
        str(records),
        "--output-root",
        str(models_root),
    ]
    _run(mine_command)
    commands.append(mine_command)

    selectors_root = output_root / "selectors"
    if not args.skip_selectors:
        for config_path in args.miner_configs:
            selector_command = [
                sys.executable,
                "scripts/evaluate_selectors.py",
                "--config",
                config_path,
                "--records",
                str(records),
                "--output-dir",
                str(selectors_root / Path(config_path).stem),
                "--future-ratio",
                str(args.selector_future_ratio),
                "--top-k",
                str(args.selector_top_k),
                "--batch-size",
                str(args.selector_batch_size),
            ]
            _run(selector_command)
            commands.append(selector_command)

    deployment_root = output_root / "deployment"
    model_specs = [
        f"{Path(config_path).stem}={models_root / Path(config_path).stem / 'model.json'}"
        for config_path in args.miner_configs
    ]
    if not args.skip_deploy:
        deploy_command = [
            sys.executable,
            "scripts/deploy_skill_models.py",
            "--config",
            args.deployment_config,
            "--output-root",
            str(deployment_root),
            "--models",
            *model_specs,
        ]
        if not args.exclude_no_skill:
            deploy_command.append("--include-no-skill")
        _run(deploy_command)
        commands.append(deploy_command)
        if not args.skip_refine:
            commands.extend(_refine_deployment_runs(deployment_root))

    analysis_root = output_root / "analysis"
    if not args.skip_analyze:
        analyze_command = [
            sys.executable,
            "scripts/analyze_experiment_table.py",
            str(output_root),
            "--output-dir",
            str(analysis_root),
        ]
        _run(analyze_command)
        commands.append(analyze_command)
        if not args.skip_plot:
            plot_command = [
                sys.executable,
                "scripts/plot_experiment_figures.py",
                str(analysis_root),
                "--prefix",
                "comparison",
            ]
            _run(plot_command)
            commands.append(plot_command)

    manifest = {
        "output_root": str(output_root),
        "segments": str(segments),
        "records": str(records),
        "models_root": str(models_root),
        "selectors_root": str(selectors_root),
        "deployment_root": str(deployment_root),
        "analysis_root": str(analysis_root),
        "commands": commands,
    }
    (output_root / "suite_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _refine_deployment_runs(deployment_root: Path) -> list[list[str]]:
    commands = []
    for run_dir in sorted(path for path in deployment_root.iterdir() if path.is_dir()):
        required = [
            run_dir / "model.json",
            run_dir / "records.jsonl",
            run_dir / "deployment_retrieval.jsonl",
        ]
        if not all(path.exists() for path in required):
            continue
        command = [
            sys.executable,
            "scripts/refine_skill_tower.py",
            "--model",
            str(run_dir / "model.json"),
            "--records",
            str(run_dir / "records.jsonl"),
            "--deployment-retrieval",
            str(run_dir / "deployment_retrieval.jsonl"),
            "--output",
            str(run_dir / "refined_model.json"),
        ]
        _run(command)
        commands.append(command)
    return commands


if __name__ == "__main__":
    main()
