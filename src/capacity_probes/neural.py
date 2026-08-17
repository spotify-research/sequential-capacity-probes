from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .dependency import ensure_dependency
from .io import csv_rows, environment_manifest, immutable_json, sha256
from .neural_contract import validate_upstream_contract


JOBS = {
    ("beauty", "sasrec_plus"): ("s3_beauty", "configs/paper/loo/s3_vanilla.yaml", 0),
    ("beauty", "esasrec"): ("s3_beauty", "configs/paper/loo/s3_beauty_ligr.yaml", 0),
    ("sports", "sasrec_plus"): ("s3_sports", "configs/paper/loo/s3_vanilla.yaml", 0),
    ("sports", "esasrec"): ("s3_sports", "configs/paper/loo/s3_sports_ligr.yaml", 0),
    ("toys", "sasrec_plus"): ("s3_toys", "configs/paper/loo/s3_vanilla.yaml", 0),
    ("toys", "esasrec"): ("s3_toys", "configs/paper/loo/s3_toys_ligr.yaml", 0),
    ("ml1m", "sasrec_plus"): ("ml_1m", "configs/paper/loo/ml_1m.yaml", 0),
    ("ml1m", "esasrec"): ("ml_1m", "configs/paper/loo/ml_1m.yaml", 1),
    ("ml20m", "sasrec_plus"): ("ml_20m", "configs/paper/loo/ml_20m.yaml", 0),
    ("ml20m", "esasrec"): ("ml_20m", "configs/paper/loo/ml_20m.yaml", 2),
}


def _link_data(checkout: Path, data_root: Path, dataset: str) -> None:
    destination = checkout / "data" / dataset / "leave_one_out"
    source = (data_root / dataset / "leave_one_out").resolve()
    if destination.is_symlink() and destination.resolve() == source:
        return
    if (destination / "train.csv").is_file() and (destination / "holdout.csv").is_file():
        for name in ("train.csv", "holdout.csv"):
            consumed = destination / name
            verified = source / name
            if sha256(consumed) != sha256(verified):
                raise RuntimeError(
                    f"Upstream {name} does not match the verified split: {consumed}"
                )
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to replace existing dependency data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)


PARAMETER_NAMES = {
    "factors": "n_factors",
    "max_length": "session_max_len",
    "blocks": "n_blocks",
    "heads": "n_heads",
    "dropout": "dropout_rate",
    "ff_multiplier": "transformer_layers_kwargs.ff_factors_multiplier",
    "loss": "loss",
    "negatives": "n_negatives",
}


def _upstream_environment(checkout: Path) -> dict[str, str]:
    """Make the pinned upstream ``src`` package importable by its script."""
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(checkout), existing) if part
    )
    return environment


def _build_config(
    checkout: Path,
    dataset: str,
    model: str,
    selected: dict,
    defaults: dict,
    output: Path,
) -> Path:
    _, source_name, index = JOBS[(dataset, model)]
    source = yaml.safe_load((checkout / source_name).read_text())
    config = copy.deepcopy(source)
    upstream_dataset = JOBS[(dataset, model)][0]
    config["datasets"] = [upstream_dataset]
    config["val_schemes"] = ["leave_one_out"]
    config["models"] = [config["models"][index]]
    report_name = f"table1_{dataset}_{model}"
    model_spec = config["models"][0]
    model_spec["report_file_name"] = report_name
    parameters = {item["name"]: item for item in model_spec["search_parameters"]}
    for local_name, value in selected.items():
        if local_name == "layers":
            continue
        upstream_name = PARAMETER_NAMES[local_name]
        if upstream_name not in parameters:
            raise RuntimeError(f"Upstream config lacks frozen parameter {upstream_name}")
        parameters[upstream_name]["choices"] = [value]
    for upstream_name, value in (
        ("lr", defaults["learning_rate"]),
        ("batch_size", defaults["batch_size"]),
    ):
        if upstream_name in parameters:
            parameters[upstream_name]["choices"] = [value]
        else:
            item = {"name": upstream_name, "choices": [value]}
            model_spec["search_parameters"].append(item)
            parameters[upstream_name] = item
    has_ligr = "transformer_layers_type" in parameters
    if has_ligr != (selected["layers"] == "LiGR"):
        raise RuntimeError("Frozen layer type does not match the pinned upstream config")
    trainer = model_spec["fixed_parameters"]["get_trainer_func"]
    expected_trainer = (
        "get_trainer_200_epochs"
        if upstream_dataset.startswith("s3_")
        else "get_trainer"
    )
    if not trainer.endswith(expected_trainer):
        raise RuntimeError(f"Pinned upstream config has unexpected trainer: {trainer}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output


def run_neural(
    root: Path,
    dataset: str,
    model: str,
    selected: dict,
    protocol: dict,
    data_root: Path,
    output_root: Path,
) -> dict:
    checkout = ensure_dependency(root)
    defaults = protocol["neural_defaults"]
    upstream_dataset = JOBS[(dataset, model)][0]
    _link_data(checkout, data_root, upstream_dataset)
    config_path = _build_config(
        checkout,
        dataset,
        model,
        selected,
        defaults,
        output_root / "upstream_configs" / f"{dataset}_{model}.yaml",
    )
    generated = yaml.safe_load(config_path.read_text())
    upstream_contract = validate_upstream_contract(
        checkout, generated, dataset, protocol
    )
    subprocess.check_call(
        [
            sys.executable,
            "-u",
            "src/evaluation/holdout_from_params.py",
            "--config_file",
            str(config_path.resolve()),
        ],
        cwd=checkout,
        env=_upstream_environment(checkout),
    )
    report_name = f"table1_{dataset}_{model}.csv"
    report = (
        checkout
        / "reports"
        / "leave_one_out"
        / upstream_dataset
        / "holdout"
        / report_name
    )
    rows = pd.read_csv(report)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one upstream result row in {report}, got {len(rows)}")
    row = rows.iloc[0]
    split_root = data_root / upstream_dataset / "leave_one_out"
    train_path, holdout_path = split_root / "train.csv", split_root / "holdout.csv"
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset": dataset,
        "config": {"model": selected, "training": defaults},
        "official_metrics": {
            "ndcg@10": float(row["ndcg@10"]),
            "recall@10": float(row["recall@10"]),
        },
        "model_params": json.loads(row["model_params"]),
        "data": {
            "train": {
                "path": str(train_path),
                "sha256": sha256(train_path),
                "rows": csv_rows(train_path),
            },
            "holdout": {
                "path": str(holdout_path),
                "sha256": sha256(holdout_path),
                "rows": csv_rows(holdout_path),
            },
        },
        "upstream": {
            "commit": str(row["commit"]),
            "config": str(config_path),
            "checkpoint": str(row.get("ckpt", "")),
            "report": str(report),
        },
        "protocol": {
            "full_catalogue": True,
            "filter_viewed": True,
            "k": 10,
            **upstream_contract,
        },
        "environment": environment_manifest(),
    }
    immutable_json(output_root / dataset / f"{model}.json", result)
    return result
