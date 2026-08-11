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


def _cmd_train(args: argparse.Namespace) -> int:
    # Imported here so `alpr env` does not pay for loading torch.
    from alpr.train import TrainConfig, TrainingError, train

    try:
        config = TrainConfig.from_yaml(args.config)
        train(config, args.data, require_gpu=not args.allow_cpu)
    except TrainingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from alpr.data.schema import Region
    from alpr.detect import PlateDetector
    from alpr.ocr import PlateReader
    from alpr.pipeline import Pipeline, PipelineConfig
    from alpr.sources import SourceError, open_source

    try:
        detector = PlateDetector(args.weights, device=args.device)
        config = PipelineConfig(
            ocr_every=args.ocr_every,
            region=Region(args.region) if args.region else None,
            confidence=args.confidence,
        )
        pipeline = Pipeline(detector, PlateReader(), config)
        with open_source(args.source) as source:
            stats = pipeline.run(source, args.out, max_frames=args.max_frames)
    except (SourceError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(stats.report())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alpr", description=__doc__)
    parser.add_argument("--version", action="version", version=f"alpr {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env", help="report interpreter, Colab, and GPU status")
    env.set_defaults(func=_cmd_env)

    train_cmd = sub.add_parser("train", help="train the plate detector")
    train_cmd.add_argument("--config", default="configs/detector.yaml", help="training config")
    train_cmd.add_argument("--data", default="data/yolo/data.yaml", help="dataset config")
    train_cmd.add_argument(
        "--allow-cpu",
        action="store_true",
        help="do not require a CUDA GPU (a CPU run will not finish; for smoke tests only)",
    )
    train_cmd.set_defaults(func=_cmd_train)

    run_cmd = sub.add_parser("run", help="detect, read and log plates from a source")
    run_cmd.add_argument(
        "--source",
        required=True,
        help="video path, camera index (0), or rtsp:// url",
    )
    run_cmd.add_argument("--out", default="plates.xlsx", help="Excel log to write")
    run_cmd.add_argument("--weights", default="best.pt", help="trained detector weights")
    run_cmd.add_argument(
        "--device", default=None, help="cuda index, 'mps' or 'cpu' (auto-detected)"
    )
    run_cmd.add_argument(
        "--region",
        choices=["IN", "DE"],
        default=None,
        help="restrict plate parsing to one country's grammar",
    )
    run_cmd.add_argument(
        "--ocr-every",
        type=int,
        default=3,
        dest="ocr_every",
        help="read each track once every N frames (1 is most accurate, slowest)",
    )
    run_cmd.add_argument("--confidence", type=float, default=0.25)
    run_cmd.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    run_cmd.set_defaults(func=_cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
