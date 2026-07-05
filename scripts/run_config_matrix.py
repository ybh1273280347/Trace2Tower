from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Config files to run in order.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run trace2tower.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for config in args.configs:
        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(config_path)

        subprocess.run(
            [
                args.python,
                "-m",
                "trace2tower.run",
                "--config",
                str(config_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
