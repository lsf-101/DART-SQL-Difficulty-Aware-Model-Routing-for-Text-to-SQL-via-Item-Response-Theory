#!/usr/bin/env python3
"""Run the BIRD calibration, positioning, routing, and direct generation stages."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "dart_sql"
sys.path.insert(0, str(CODE))
from run_direct_sql import safe_name


def path(value):
    if not value or "your path" in str(value).lower():
        raise ValueError("Replace every required 'your path' value in your local configuration before running.")
    return (ROOT / Path(value).expanduser()).resolve()


def read_json(file):
    return json.loads(file.read_text(encoding="utf-8-sig"))


def write_json(file, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(script, *args, allowed=(0,)):
    command = [sys.executable, str(script), *map(str, args)]
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode not in allowed:
        result.check_returncode()


def prepare_models(config, work):
    model_config = read_json(path(config["models"]))
    defaults = model_config.get("defaults", {})
    entries = {}
    for alias, spec in model_config["models"].items():
        if not spec.get("enabled", True):
            continue
        entry = {**defaults, **spec}
        entry["llm_kwargs"] = {**defaults.get("llm_kwargs", {}), **spec.get("llm_kwargs", {})}
        entry.pop("enabled", None)
        if entry.get("api_key"):
            raise ValueError(f"{alias}: use api_key_env instead of a stored key")
        if entry.get("backend", "vllm") == "vllm":
            entry["model_path"] = str(path(entry["model_path"]))
        entries[alias] = entry
    if not entries:
        raise ValueError("Enable at least one model in the models configuration.")
    write_json(work / "models.resolved.json", {"models": entries})
    write_json(work / "candidates.json", {"policy": "local_only", "models": list(entries)})
    return entries


def schema_args(data, split):
    if data.get(f"{split}_schema_dir"):
        return ["--schema-dir", path(data[f"{split}_schema_dir"])]
    return ["--tables-json", path(data[f"{split}_tables_json"])]


def calibrate(config, work, history):
    data, training = config["data"], config["training"]
    args = [
        "--dataset-name", "bird", "--train-json", path(data["train_json"]),
        "--db-root", path(data["train_db_root"]), *schema_args(data, "train"),
        "--output-dir", work / "calibration", "--model-results-dir", history,
        "--sql-timeout", training["sql_timeout"],
        "--max-result-rows", training["max_result_rows"],
    ]
    if training.get("training_config"):
        args.extend(["--training-config", path(training["training_config"])])
    if training["reuse_history"]:
        if not any(history.glob("*_results.csv")):
            raise FileNotFoundError(f"No historical *_results.csv files in {history}")
        args.append("--results-only")
    else:
        ids = training["history_gpu_ids"]
        args.extend(["--models-config", work / "models.resolved.json",
                     "--gpu-ids", ids, "--num-gpus", len(ids.split(","))])
    run(CODE / "2PL.py", *args)


def position(config, work):
    data = config["data"]
    run(CODE / "test_2PL.py", "--dataset-name", "bird",
        "--test-json", path(data["dev_json"]), *schema_args(data, "dev"),
        "--train-artifact-dir", work / "calibration", "--output-dir", work / "positions",
        "--anchor-num", config["training"]["anchor_count"])


def route(config, work, history):
    positions = work / "positions"
    knobs = config["routing"]
    flags = []
    for name, value in knobs.items():
        flags.extend(["--" + name.replace("_", "-"), value])
    run(CODE / "model_selector.py", "--dataset-name", "bird",
        "--test-json", positions / "bird_dev_with_anchor_position.json",
        "--train-json", positions / "bird_train_with_anchor_position.json",
        "--test-skeleton-embeddings", positions / "test_question_skeleton_embeddings.pkl",
        "--train-skeleton-embeddings", positions / "train_question_skeleton_embeddings.pkl",
        "--model-results-dir", history, "--local-model-config", work / "candidates.json",
        "--output-json", work / "selected.json", "--report-json", work / "routing_report.json",
        *flags)


def generate(config, work, entries, model, selection, dry_run):
    data = config["data"]
    if model and model not in entries:
        raise ValueError(f"Unknown or disabled model alias: {model}")
    name = f"fixed_{safe_name(model)}" if model else "routed"
    run_dir = work / "generation" / name
    args = ["--fixed-model", model] if model else ["--selection-json", selection]
    schema = ["--schema-dir", path(data["dev_schema_dir"])] if data.get("dev_schema_dir") else []
    options = config["generation"]
    for key in ("temperature", "max_tokens", "batch_size", "seed"):
        args.extend(["--" + key.replace("_", "-"), options[key]])
    if dry_run:
        args.append("--dry-run")
    run(CODE / "run_direct_sql.py", "generate", "--mode", "fixed" if model else "routed",
        "--dataset-name", "bird", "--dev-json", path(data["dev_json"]),
        "--db-root", path(data["dev_db_root"]), *schema,
        "--models-config", work / "models.resolved.json", "--output-dir", run_dir, *args,
        allowed=(0, 2))
    return run_dir / f"{name}.json", run_dir / "token_usage.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "position", "route", "generate", "all"))
    parser.add_argument("--config", default="configs/experiment.local.json")
    parser.add_argument("--model", help="Use this fixed model alias during generation.")
    parser.add_argument("--selection", help="Use an existing routed selection JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Validate direct generation inputs.")
    args = parser.parse_args()
    if args.model and args.selection:
        parser.error("--model and --selection are mutually exclusive")
    if args.dry_run and args.stage != "generate":
        parser.error("--dry-run requires generate")

    config = read_json(path(args.config))
    work = path(config["work_dir"])
    selection = path(args.selection) if args.selection else work / "selected.json"

    entries = prepare_models(config, work)
    if args.stage == "calibrate" or (args.stage == "all" and not (args.model or args.selection)):
        calibrate(config, work, path(config["history_dir"]))
    if args.stage == "position" or (args.stage == "all" and not (args.model or args.selection)):
        position(config, work)
    if args.stage == "route" or (args.stage == "all" and not (args.model or args.selection)):
        route(config, work, path(config["history_dir"]))
    if args.stage in ("generate", "all"):
        generate(config, work, entries, args.model, selection, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
