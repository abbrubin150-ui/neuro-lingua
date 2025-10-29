"""Run Neuro-Lingua training across multiple configurations.

Each configuration is defined in a JSON file under `configs/` and contains the
hyperparameters, dataset pointers, and metadata required to execute the
TypeScript training script (`scripts/train.ts`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = ROOT / "configs"
DEFAULT_METRICS_DIR = ROOT / "docs" / "experiments" / "runs"
TRAIN_SCRIPT = ROOT / "scripts" / "train.ts"


@dataclass
class ExperimentConfig:
    name: str
    description: str
    corpus_path: Path
    hyperparameters: Dict[str, Any]
    tokenizer: Dict[str, Any]
    metrics_path: Path
    model_path: Path
    extra_env: Dict[str, str]


def load_config(path: Path, metrics_dir: Path, model_dir: Path) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = payload.get("name")
    if not name:
        raise ValueError(f"Configuration at {path} is missing the 'name' field")
    description = payload.get("description", "")
    corpus_path = Path(payload["corpus_path"]).expanduser()
    hyperparameters = payload.get("hyperparameters", {})
    tokenizer = payload.get("tokenizer", {})
    metrics_filename = payload.get("metrics_filename", f"{name}.json")
    metrics_path = metrics_dir / metrics_filename
    model_filename = payload.get("model_filename", f"{name}.json")
    model_path = model_dir / model_filename
    extra_env = {str(key): str(value) for key, value in payload.get("env", {}).items()}
    return ExperimentConfig(
        name=name,
        description=description,
        corpus_path=corpus_path,
        hyperparameters=hyperparameters,
        tokenizer=tokenizer,
        metrics_path=metrics_path,
        model_path=model_path,
        extra_env=extra_env,
    )


def _as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_experiment(config: ExperimentConfig, node_runner: Sequence[str]) -> dict:
    env = os.environ.copy()
    env.update(config.extra_env)
    env["CORPUS_PATH"] = str(config.corpus_path.resolve())
    env["METRICS_PATH"] = str(config.metrics_path.resolve())
    env["MODEL_EXPORT_PATH"] = str(config.model_path.resolve())
    env["EXPERIMENT_NAME"] = config.name

    for key, value in config.hyperparameters.items():
        env[key.upper()] = _as_env_value(value)

    tokenizer_mode = config.tokenizer.get("mode")
    if tokenizer_mode:
        env["TOKENIZER_MODE"] = tokenizer_mode
        if tokenizer_mode == "custom" and config.tokenizer.get("pattern"):
            env["TOKENIZER_PATTERN"] = config.tokenizer["pattern"]
    if config.tokenizer.get("use_ascii"):
        env["USE_ASCII_TOKENIZER"] = "true"

    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    command = [*node_runner, str(TRAIN_SCRIPT)]
    print(f"\n▶ Running {config.name} ...")
    completed = subprocess.run(command, check=False, env=env, capture_output=True, text=True)
    Path(ROOT / "logs").mkdir(exist_ok=True)
    log_path = ROOT / "logs" / f"{config.name}.log"
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Experiment {config.name} failed. See {log_path} for details.")

    metrics = json.loads(config.metrics_path.read_text(encoding="utf-8"))
    return metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch runner for Neuro-Lingua experiments")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Configuration files to execute (defaults to all JSON files in configs/)",
    )
    parser.add_argument(
        "--node-runner",
        nargs="+",
        default=["npx", "tsx"],
        help="Command used to execute the TypeScript training script",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=DEFAULT_METRICS_DIR,
        help="Directory for metrics artifacts",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=ROOT / "models" / "experiments",
        help="Directory for experiment-specific model exports",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)

    metrics_dir = args.metrics_dir
    models_dir = args.models_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    config_paths: List[Path]
    if args.configs:
        config_paths = [path if path.is_absolute() else (ROOT / path) for path in args.configs]
    else:
        config_paths = sorted(DEFAULT_CONFIG_DIR.glob("*.json"))

    if not config_paths:
        raise SystemExit("No configuration files found.")

    summaries = []
    for config_path in config_paths:
        config = load_config(config_path, metrics_dir, models_dir)
        metrics = run_experiment(config, node_runner=args.node_runner)
        summaries.append(
            {
                "name": config.name,
                "description": config.description,
                "metrics_path": str(config.metrics_path.relative_to(ROOT)),
                "model_path": str(config.model_path.relative_to(ROOT)),
                "loss": metrics["metrics"]["loss"],
                "accuracy": metrics["metrics"]["accuracy"],
                "epochs": metrics["hyperparameters"]["epochs"],
                "hiddenSize": metrics["hyperparameters"]["hiddenSize"],
            }
        )

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    summary_report = {
        "generated_at": timestamp,
        "runs": summaries,
    }
    report_path = metrics_dir / "summary.json"
    report_path.write_text(json.dumps(summary_report, indent=2), encoding="utf-8")
    print("\n✅ Experiments completed. Summary saved to", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
