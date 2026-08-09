"""Command-line entrypoint.

Subcommands land as their phases do. `env` works today because diagnosing a
wrong Colab runtime is the first thing that goes wrong in this project.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from alpr import __version__
from alpr.env import detect_gpus, in_colab


def _cmd_env(_: argparse.Namespace) -> int:
    print(f"alpr {__version__}")
    print(f"python  {sys.version.split()[0]}")
    print(f"colab   {in_colab()}")

    gpus = detect_gpus()
    if not gpus:
        print("gpu     none visible (CPU-only runtime)")
    else:
        for i, gpu in enumerate(gpus):
            print(f"gpu[{i}]  {gpu}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"alpr {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("env", help="report interpreter, Colab, and GPU status")
    env.set_defaults(func=_cmd_env)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
