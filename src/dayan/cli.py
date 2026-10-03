"""CLI: python -m dayan <command>.  Currently: augment datasets."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dayan",
        description="DaYan (大衍): thought-graph reasoning analysis + verified data augmentation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    aug = sub.add_parser("augment", help="Run Logical DA over a dataset JSON file")
    aug.add_argument("--dataset", required=True, help="JSON array of source records")
    aug.add_argument("--output", required=True, help="Where to write augmented records")
    aug.add_argument("--solver", default="auto", help="auto|prover9|rules|csp|z3|llm")
    aug.add_argument("--max-records", type=int, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "augment":
        from pathlib import Path

        from dayan.augment import PipelineConfig, SimpleWorkflow, run_dataset
        from dayan.augment.data import load_dataset
        from dayan.llm import ChatClient

        dataset = load_dataset(Path(args.dataset))
        workflow = SimpleWorkflow(ChatClient(), solver=args.solver)
        config = PipelineConfig(symbolic_solver=args.solver, max_records=args.max_records)
        result = run_dataset(dataset, workflow, config)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"rows": result.rows, "report": result.report}, handle, indent=2, ensure_ascii=False)
        print(f"Augmented {len(result.rows)} records -> {args.output}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
