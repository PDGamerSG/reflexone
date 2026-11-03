"""CLI entry points for reproducible data, training, evaluation and inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit(result, output: Path | None = None) -> None:
    value = json.dumps(result, indent=2, allow_nan=False, default=str) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(value)
    else:
        print(value, end="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reflexone", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("profiles", help="Describe model families and supported use cases")
    prepare = sub.add_parser("prepare", help="Download and audit support data")
    prepare.add_argument("--data", type=Path, default=Path("data/support"))
    prepare.add_argument("--seed", type=int, default=17)
    train = sub.add_parser("train", help="Fine-tune the support encoder and parallel heads")
    train.add_argument("--data", type=Path, default=Path("data/support"))
    train.add_argument("--artifacts", type=Path, default=Path("artifacts/support"))
    train.add_argument("--config", type=Path, default=Path("configs/support.json"))
    train.add_argument("--seed", type=int)
    train.add_argument("--output", type=Path)
    evaluate = sub.add_parser("evaluate", help="Re-evaluate frozen support weights")
    evaluate.add_argument("--data", type=Path, default=Path("data/support"))
    evaluate.add_argument("--artifacts", type=Path, default=Path("artifacts/support"))
    evaluate.add_argument("--output", type=Path)
    predict = sub.add_parser("predict", help="Return validated parallel support decisions")
    predict.add_argument("--artifacts", type=Path, default=Path("artifacts/support"))
    predict.add_argument("--domain", choices=("banking", "hotels"), required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument(
        "--selected-only", action="store_true", help="Only display selected intent modules"
    )
    benchmark = sub.add_parser("benchmark", help="Measure warmed CPU support inference")
    benchmark.add_argument("--artifacts", type=Path, default=Path("artifacts/support"))
    benchmark.add_argument("--repeats", type=int, default=120)
    benchmark.add_argument("--output", type=Path)
    for verb in ("prepare-task", "train-task", "predict-task", "evaluate-task", "benchmark-task"):
        command = sub.add_parser(verb, help=f"{verb.replace('-', ' ')} commands or spam profile")
        command.add_argument("task", choices=("commands", "spam"))
        command.add_argument("--data", type=Path)
        command.add_argument("--artifacts", type=Path)
        command.add_argument("--output", type=Path)
        if verb in ("prepare-task", "train-task"):
            command.add_argument("--seed", type=int, default=17 if verb == "prepare-task" else 42)
        if verb == "predict-task":
            command.add_argument("--text", required=True)
        if verb == "benchmark-task":
            command.add_argument("--repeats", type=int, default=120)
    serve = sub.add_parser("serve", help="Serve all installed profiles using FastAPI")
    serve.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "profiles":
            result = {
                "support": {
                    "domains": ["banking", "hotels"],
                    "intents": 62,
                    "slot_types": 17,
                    "algorithm": "Fine-tuned MiniLM, sigmoid intent heads, independent BIO slot heads",
                },
                "commands": {
                    "intents": 150,
                    "oos_evaluation": True,
                    "algorithm": "TF-IDF logistic regression + temperature + validated rejection",
                },
                "spam": {
                    "classes": ["ham", "spam"],
                    "algorithm": "TF-IDF Complement Naive Bayes + temperature",
                },
            }
        elif args.command == "prepare":
            from reflexone.data import prepare_dataset

            audit = prepare_dataset(args.data, args.seed)
            result = {k: audit[k] for k in ("canonical_records", "fingerprint", "supports")}
        elif args.command == "train":
            from reflexone.training import SupportConfig, train_support

            config = json.loads(args.config.read_text())
            if args.seed is not None:
                config["seed"] = args.seed
            result = train_support(args.data, args.artifacts, SupportConfig(**config))
        elif args.command == "evaluate":
            from reflexone.training import evaluate_support

            result = evaluate_support(args.artifacts, args.data)
        elif args.command == "predict":
            from reflexone.inference import SupportPredictor

            result = (
                SupportPredictor(args.artifacts)
                .predict(args.text, args.domain)
                .model_dump(mode="json")
            )
            if args.selected_only:
                result["intents"] = {k: v for k, v in result["intents"].items() if v["value"]}
        elif args.command == "benchmark":
            from reflexone.inference import benchmark_support

            result = benchmark_support(args.artifacts, args.repeats)
        elif args.command.endswith("-task"):
            from reflexone import tasks
            from reflexone.schema import CategoricalDecision

            data = args.data or Path("data") / args.task
            artifact = args.artifacts or Path("artifacts") / args.task
            if args.command == "prepare-task":
                result = tasks.prepare_task(args.task, data, args.seed)
            elif args.command == "train-task":
                result = tasks.train_task(args.task, data, artifact, args.seed)
            else:
                predictor = tasks.TaskPredictor(artifact)
                if predictor.task != args.task:
                    raise ValueError(
                        f"Artifact profile is '{predictor.task}', requested '{args.task}'"
                    )
                if args.command == "predict-task":
                    result = CategoricalDecision.model_validate(
                        predictor.predict(args.text)
                    ).model_dump(mode="json")
                elif args.command == "evaluate-task":
                    result = tasks.evaluate_task(artifact)
                else:
                    result = tasks.benchmark_task(artifact, args.repeats)
        else:
            import uvicorn

            from reflexone.api import create_app

            uvicorn.run(create_app(args.artifacts_root), host=args.host, port=args.port)
            return 0
        _emit(result, getattr(args, "output", None))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
