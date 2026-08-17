from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from capacity_probes.neural import _build_config, _link_data, _upstream_environment
from capacity_probes.neural_contract import validate_upstream_contract


def _model(parameters: list[tuple[str, object]]) -> dict:
    return {
        "report_file_name": "upstream",
        "cls": "SASRecModel",
        "comment": "_",
        "fixed_parameters": {
            "deterministic": True,
            "get_trainer_func": "src.models.transformers.trainer.get_trainer_200_epochs",
        },
        "search_parameters": [
            {"name": name, "choices": [value]} for name, value in parameters
        ],
    }


def test_local_sasrec_parameters_override_pinned_upstream(tmp_path: Path) -> None:
    source = tmp_path / "configs/paper/loo/s3_vanilla.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "datasets": ["s3_beauty", "s3_sports", "s3_toys"],
                "val_schemes": ["leave_one_out"],
                "models": [
                    _model(
                        [
                            ("n_factors", 256),
                            ("session_max_len", 50),
                            ("n_blocks", 1),
                            ("n_heads", 1),
                            ("dropout_rate", 0.2),
                            ("loss", "sampled_softmax"),
                            ("n_negatives", 256),
                        ]
                    )
                ],
            }
        )
    )
    selected = {
        "layers": "SASRec",
        "factors": 64,
        "max_length": 50,
        "blocks": 1,
        "heads": 1,
        "dropout": 0.2,
        "loss": "sampled_softmax",
        "negatives": 256,
    }
    defaults = {"learning_rate": 0.001, "batch_size": 128}
    output = _build_config(
        tmp_path,
        "beauty",
        "sasrec_plus",
        selected,
        defaults,
        tmp_path / "job.yaml",
    )
    job = yaml.safe_load(output.read_text())
    assert job["datasets"] == ["s3_beauty"]
    assert len(job["models"]) == 1
    parameters = {
        item["name"]: item["choices"] for item in job["models"][0]["search_parameters"]
    }
    assert parameters["n_factors"] == [64]
    assert parameters["lr"] == [0.001]
    assert parameters["batch_size"] == [128]
    assert job["models"][0]["report_file_name"] == "table1_beauty_sasrec_plus"


def test_layer_type_mismatch_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "configs/paper/loo/s3_vanilla.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump({"models": [_model([])]}))
    try:
        _build_config(
            tmp_path,
            "beauty",
            "sasrec_plus",
            {"layers": "LiGR"},
            {"learning_rate": 0.001, "batch_size": 128},
            tmp_path / "job.yaml",
        )
    except RuntimeError as error:
        assert "layer type" in str(error)
    else:
        raise AssertionError("Mismatched transformer layer type was accepted")


def test_upstream_checkout_is_first_on_pythonpath(
    tmp_path: Path, monkeypatch
) -> None:
    checkout = tmp_path / "transformer_benchmark"
    monkeypatch.setenv("PYTHONPATH", "/pinned/dependencies:/publication/src")
    environment = _upstream_environment(checkout)
    assert environment["PYTHONPATH"].split(":") == [
        str(checkout),
        "/pinned/dependencies",
        "/publication/src",
    ]


def test_existing_upstream_splits_must_match_verified_copies(tmp_path: Path) -> None:
    checkout = tmp_path / "dependency"
    data_root = tmp_path / "processed"
    upstream = checkout / "data/s3_beauty/leave_one_out"
    verified = data_root / "s3_beauty/leave_one_out"
    upstream.mkdir(parents=True)
    verified.mkdir(parents=True)
    for name in ("train.csv", "holdout.csv"):
        (upstream / name).write_text("user,item\n1,2\n")
        (verified / name).write_text("user,item\n1,2\n")

    _link_data(checkout, data_root, "s3_beauty")
    (upstream / "train.csv").write_text("user,item\n1,3\n")

    with pytest.raises(RuntimeError, match="does not match the verified split"):
        _link_data(checkout, data_root, "s3_beauty")


def test_frozen_upstream_training_contract_is_checked(tmp_path: Path) -> None:
    trainer = tmp_path / "src/models/transformers/trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text(
        "RECALL_K = 10\nPATIENCE = 50\nMAX_EPOCHS = 100\n"
        "def get_trainer():\n    return Trainer(max_epochs=MAX_EPOCHS)\n"
        "def get_trainer_200_epochs():\n    return Trainer(max_epochs=200)\n"
    )
    utilities = tmp_path / "src/utils.py"
    utilities.write_text("def setup_deterministic(random_seed=32):\n    pass\n")
    generated = {
        "models": [
            {
                "fixed_parameters": {
                    "deterministic": True,
                    "get_val_mask_func": "src.trainer.get_val_mask_func_all",
                    "get_trainer_func": "src.trainer.get_trainer_200_epochs",
                },
                "search_parameters": [{"name": "n_factors", "choices": [64]}],
            }
        ]
    }
    protocol = {
        "neural_seed": 32,
        "neural_defaults": {
            "validation_metric": "Recall@10",
            "early_stopping_patience": 50,
            "amazon_max_epochs": 200,
            "movielens_max_epochs": 100,
        },
    }
    actual = validate_upstream_contract(tmp_path, generated, "beauty", protocol)
    assert actual["seed"] == 32
    assert actual["max_epochs"] == 200
