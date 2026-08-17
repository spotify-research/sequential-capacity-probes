from __future__ import annotations

import json
from pathlib import Path

import capacity_probes.data as data_module
from capacity_probes.config import load_table_config
from capacity_probes.data import verify_processed
from capacity_probes.reporting import DATASET_ORDER, MODEL_ORDER, verify_table
from capacity_probes.models import fmc, fmc_plus, mc, pctm, seqrules
from tests.result_factory import result_payload


ROOT = Path(__file__).resolve().parents[1]


def test_table_contains_exactly_seven_models_and_five_datasets() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    assert tuple(config["datasets"]) == DATASET_ORDER
    assert all(tuple(row) == MODEL_ORDER for row in config["reported_ndcg10"].values())
    policy = config["verification"]
    covered = set(policy["exact_4dp_models"]) | set(
        policy["absolute_ndcg10_tolerance"]
    )
    assert covered == set(MODEL_ORDER)


def test_recorded_tie_policies_match_model_implementations() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    protocol = config["protocol"]
    assert mc.TIE_BREAK == protocol["mc_tie_break"]
    assert seqrules.TIE_BREAK == protocol["seqrules_tie_break"]
    assert pctm.TIE_BREAK == protocol["pctm_tie_break"]
    assert fmc.TIE_BREAK == protocol["factorized_tie_break"]
    assert fmc_plus.TIE_BREAK == protocol["factorized_tie_break"]


def test_classical_models_have_explicit_modules() -> None:
    model_root = ROOT / "src/capacity_probes/models"
    visible_modules = {
        path.name for path in model_root.glob("*.py") if not path.name.startswith(".")
    }
    assert visible_modules == {
        "__init__.py",
        "mc.py",
        "fmc.py",
        "fmc_plus.py",
        "seqrules.py",
        "pctm.py",
    }


def test_paper_pctm_hyperparameters_are_frozen() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    assert config["protocol"]["mc_tie_break"] == "ascending external item ID"
    assert config["datasets"]["toys"]["pctm"] == {
        "counts": "exp:20:0.7",
        "tau": 15000.0,
        "kernel": "tail:7:0.8:0.6",
        "pop_boost": 0.0,
    }
    assert config["datasets"]["ml20m"]["pctm"]["tau"] == 225.0


def test_factorized_refit_configs_are_frozen() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    expected = {
        "beauty": (
            ("sampled_bce", 256, 0.003, 0.0, 5),
            ("full_ce", 128, 0.003, 0.0, 16),
        ),
        "sports": (
            ("sampled_bce", 256, 0.003, 0.0, 4),
            ("full_ce", 256, 0.003, 0.0, 9),
        ),
        "toys": (
            ("sampled_bce", 256, 0.003, 0.2, 8),
            ("full_ce", 256, 0.003, 0.0, 12),
        ),
        "ml1m": (
            ("sampled_bce", 256, 0.003, 0.0, 29),
            ("full_ce", 64, 0.01, 0.0, 21),
        ),
        "ml20m": (
            ("sampled_bce", 256, 0.003, 0.0, 7),
            ("full_ce", 128, 0.003, 0.0, 45),
        ),
    }
    for dataset, pair in expected.items():
        for model, frozen in zip(("fmc", "fmc_plus"), pair):
            selected = config["datasets"][dataset][model]
            actual = tuple(
                selected[key]
                for key in ("objective", "factors", "learning_rate", "dropout", "epochs")
            )
            assert actual == frozen
            assert selected["weight_decay"] == 0.0
    assert config["protocol"]["refit_seed"] == 20270710
    assert config["verification"]["absolute_ndcg10_tolerance"] == {
        "fmc": 0.001,
        "fmc_plus": 0.001,
        "sasrec_plus": 0.005,
        "esasrec": 0.005,
    }


def test_transformer_configs_are_frozen() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    defaults = config["protocol"]["neural_defaults"]
    assert defaults == {
        "learning_rate": 0.001,
        "batch_size": 128,
        "early_stopping_patience": 50,
        "validation_metric": "Recall@10",
        "amazon_max_epochs": 200,
        "movielens_max_epochs": 100,
    }
    assert config["protocol"]["neural_seed"] == 32
    assert config["datasets"]["beauty"]["esasrec"]["ff_multiplier"] == 4
    assert config["datasets"]["sports"]["esasrec"]["ff_multiplier"] == 2
    assert config["datasets"]["toys"]["esasrec"]["ff_multiplier"] == 1
    assert config["datasets"]["ml1m"]["sasrec_plus"]["factors"] == 64
    assert config["datasets"]["ml20m"]["esasrec"]["factors"] == 256


def test_paper_seqrules_hyperparameters_are_frozen() -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    assert config["protocol"]["seqrules_tie_break"] == "ascending external item ID"
    assert config["datasets"]["beauty"]["seqrules"] == {
        "rules": {
            "steps": 10,
            "weighting": "log",
            "pruning": 100,
            "idf_weight": False,
        },
        "history_length": 10,
        "history_weighting": "quadratic",
    }
    assert config["datasets"]["sports"]["seqrules"]["rules"]["pruning"] == 0
    assert config["datasets"]["toys"]["seqrules"]["rules"]["pruning"] == 0
    assert config["datasets"]["ml1m"]["seqrules"]["rules"]["idf_weight"] is True
    assert config["datasets"]["ml20m"]["seqrules"]["history_length"] == 5


def test_seqrules_subset_verification(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    for dataset in DATASET_ORDER:
        path = tmp_path / dataset / "seqrules.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = config["reported_ndcg10"][dataset]["seqrules"]
        path.write_text(json.dumps(result_payload(config, dataset, "seqrules", value)))
    report = verify_table(config, tmp_path, DATASET_ORDER, ("seqrules",))
    assert report["status"] == "pass"
    assert "SeqRules" in (tmp_path / "RESULTS.md").read_text()


def test_factorized_tolerance_accepts_close_and_rejects_distant(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    expected = config["reported_ndcg10"]["beauty"]["fmc"]
    path = tmp_path / "beauty" / "fmc.json"
    path.parent.mkdir(parents=True)
    payload = result_payload(config, "beauty", "fmc", expected + 0.0009)
    path.write_text(json.dumps(payload))
    report = verify_table(config, tmp_path, ("beauty",), ("fmc",))
    assert report["status"] == "pass"
    assert report["checks"][0]["criterion"].endswith("<= 0.001")
    payload = result_payload(config, "beauty", "fmc", expected + 0.0011)
    path.write_text(json.dumps(payload))
    try:
        verify_table(config, tmp_path, ("beauty",), ("fmc",))
    except RuntimeError as error:
        assert "did not meet acceptance" in str(error)
    else:
        raise AssertionError("FMC result outside tolerance was accepted")


def test_neural_tolerance_accepts_close_and_rejects_distant(
    tmp_path: Path,
) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    expected = config["reported_ndcg10"]["beauty"]["sasrec_plus"]
    path = tmp_path / "beauty" / "sasrec_plus.json"
    path.parent.mkdir(parents=True)
    payload = result_payload(config, "beauty", "sasrec_plus", expected + 0.0049)
    path.write_text(json.dumps(payload))
    report = verify_table(config, tmp_path, ("beauty",), ("sasrec_plus",))
    assert report["status"] == "pass"
    assert report["checks"][0]["criterion"].endswith("<= 0.005")
    payload = result_payload(config, "beauty", "sasrec_plus", expected + 0.0051)
    path.write_text(json.dumps(payload))
    try:
        verify_table(config, tmp_path, ("beauty",), ("sasrec_plus",))
    except RuntimeError as error:
        assert "did not meet acceptance" in str(error)
    else:
        raise AssertionError("Neural result outside tolerance was accepted")


def test_processed_manifest_rejects_wrong_content(tmp_path: Path) -> None:
    data = tmp_path / "s3_beauty" / "leave_one_out"
    data.mkdir(parents=True)
    (data / "train.csv").write_text("wrong\n")
    (data / "holdout.csv").write_text("wrong\n")
    try:
        verify_processed(ROOT, tmp_path, ("s3_beauty",))
    except RuntimeError as error:
        assert "Checksum mismatch" in str(error)
    else:
        raise AssertionError("Bad data was accepted")


def test_table_verifier_checks_four_decimal_results(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            path = tmp_path / dataset / f"{model}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            value = config["reported_ndcg10"][dataset][model]
            path.write_text(json.dumps(result_payload(config, dataset, model, value)))
    report = verify_table(config, tmp_path)
    assert report["status"] == "pass"
    assert (tmp_path / "table1.csv").is_file()
    assert (tmp_path / "RESULTS.md").is_file()


def test_table_verifier_rejects_wrong_artifact_identity(tmp_path: Path) -> None:
    config = load_table_config(ROOT / "configs/table1.json")
    value = config["reported_ndcg10"]["beauty"]["pctm"]
    payload = result_payload(config, "beauty", "pctm", value)
    payload["dataset"] = "sports"
    path = tmp_path / "beauty/pctm.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    try:
        verify_table(config, tmp_path, ("beauty",), ("pctm",))
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("Mislabeled result artifact was accepted")


def test_public_builder_restores_tracked_upstream_statistics(
    tmp_path: Path, monkeypatch
) -> None:
    checkout = tmp_path / "dependency"
    dataset = checkout / "data" / "s3_beauty"
    dataset.mkdir(parents=True)
    statistics = dataset / "statistics.csv"
    statistics.write_text("original\n")

    def fake_builder(*_args, **_kwargs) -> None:
        statistics.write_text("generated\n")
        split = dataset / "leave_one_out"
        split.mkdir()
        (split / "train.csv").write_text("header\nrow\n")
        (split / "holdout.csv").write_text("header\nrow\n")

    monkeypatch.setattr(data_module, "ensure_dependency", lambda _root: checkout)
    monkeypatch.setattr(data_module, "_download", lambda *_args: None)
    monkeypatch.setattr(data_module.subprocess, "check_call", fake_builder)
    monkeypatch.setattr(data_module, "verify_processed", lambda *_args: None)
    output = tmp_path / "output"
    data_module.build_public(ROOT, output, ("s3_beauty",))
    assert statistics.read_text() == "original\n"
    assert (output / "s3_beauty" / "statistics.csv").read_text() == "generated\n"
