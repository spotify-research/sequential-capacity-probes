from __future__ import annotations

import importlib
from math import log2
from pathlib import Path

import numpy as np
from scipy import sparse

from capacity_probes.models.fmc import build_fmc, recommend_fmc
from capacity_probes.models.fmc_plus import build_fmc_plus, recommend_fmc_plus
from capacity_probes.models.mc import build_mc, recommend_mc
from capacity_probes.models.pctm import (
    PCTMConfig,
    PCTMModel,
    build_pctm,
    recommend_pctm,
)
from capacity_probes.models.seqrules import build_seqrules, recommend_seqrules
from capacity_probes.sequences import EvaluationSet, SequenceStore


def _case() -> tuple[SequenceStore, EvaluationSet]:
    store = SequenceStore(
        catalog=np.arange(10, 30, dtype=np.int64),
        users=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        items=np.asarray([0, 1, 2, 0, 1, 3], dtype=np.int32),
        group_users=np.asarray([1, 2], dtype=np.int64),
        starts=np.asarray([0, 3], dtype=np.int64),
        ends=np.asarray([3, 6], dtype=np.int64),
        user_to_group={1: 0, 2: 1},
    )
    evaluation = EvaluationSet(
        store,
        np.asarray([1, 2], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int32),
        (np.asarray([4], dtype=np.int32), np.asarray([5], dtype=np.int32)),
    )
    return store, evaluation


def _assert_recommendations_are_unseen(frame, store: SequenceStore) -> None:
    assert len(frame) > 0
    for user, rows in frame.groupby("user_id"):
        seen = set(store.catalog[store.group_items(store.user_to_group[int(user)])])
        assert set(rows["item_id"]).isdisjoint(seen)


def test_every_model_module_has_explicit_lifecycle() -> None:
    names = ("mc", "fmc", "fmc_plus", "seqrules", "pctm")
    for name in names:
        module = importlib.import_module(f"capacity_probes.models.{name}")
        assert callable(getattr(module, f"build_{name}"))
        assert callable(getattr(module, f"recommend_{name}"))


def test_mc_build_and_recommend_live_in_mc(tmp_path: Path) -> None:
    store, evaluation = _case()
    model = build_mc(store, tmp_path / "mc")
    metrics, frame = recommend_mc(model, evaluation)
    assert model.config.transition_distance == 1
    assert metrics.users == 2
    _assert_recommendations_are_unseen(frame, store)


def test_pctm_build_and_recommend_live_in_pctm(tmp_path: Path) -> None:
    store, evaluation = _case()
    values = {"counts": "u:2", "tau": 5.0, "kernel": "last", "pop_boost": 0.0}
    model = build_pctm(store, tmp_path / "pctm", values)
    metrics, frame = recommend_pctm(model, evaluation, max_history=3)
    assert model.config.tau == 5.0
    assert metrics.users == 2
    _assert_recommendations_are_unseen(frame, store)


def test_pctm_recommendation_implements_algorithm_2_scores() -> None:
    store = SequenceStore(
        catalog=np.asarray([10, 20, 30, 40, 50], dtype=np.int64),
        users=np.asarray([1, 1], dtype=np.int64),
        items=np.asarray([0, 1], dtype=np.int32),
        group_users=np.asarray([1], dtype=np.int64),
        starts=np.asarray([0], dtype=np.int64),
        ends=np.asarray([2], dtype=np.int64),
        user_to_group={1: 0},
    )
    evaluation = EvaluationSet(
        store,
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        (np.asarray([3], dtype=np.int32),),
    )
    evidence = sparse.csr_matrix(
        (
            np.asarray([1.0, 0.5], dtype=np.float32),
            (np.asarray([1, 1]), np.asarray([2, 3])),
        ),
        shape=(5, 5),
    )
    log_popularity = np.asarray([-1, -1, -1, -2, -4], dtype=np.float32)
    model = PCTMModel(PCTMConfig("u:1", 1.0, "last", -0.2), evidence, log_popularity)
    metrics, frame = recommend_pctm(model, evaluation, max_history=1)
    np.testing.assert_array_equal(frame["item_id"], [30, 40, 50])
    np.testing.assert_allclose(frame["score"], [1.2, 0.9, 0.8], atol=1e-6)
    assert np.isclose(metrics.ndcg10, 1 / log2(3))


def test_seqrules_build_and_recommend_live_in_seqrules(tmp_path: Path) -> None:
    store, evaluation = _case()
    values = {
        "rules": {"steps": 2, "weighting": "div", "pruning": 0, "idf_weight": False},
        "history_length": 2,
        "history_weighting": "quadratic",
    }
    model = build_seqrules(store, tmp_path / "seqrules", values)
    metrics, frame = recommend_seqrules(model, evaluation, max_history=3)
    assert model.config.rules.steps == 2
    assert metrics.users == 2
    _assert_recommendations_are_unseen(frame, store)


def test_fmc_build_and_recommend_live_in_fmc() -> None:
    store, evaluation = _case()
    values = {
        "objective": "sampled_bce",
        "factors": 4,
        "learning_rate": 0.003,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "epochs": 0,
    }
    model = build_fmc(store, 3, values, "cpu", seed=7)
    metrics, frame = recommend_fmc(model, evaluation)
    assert model.network.source.sparse
    assert metrics.users == 2
    _assert_recommendations_are_unseen(frame, store)


def test_fmc_plus_build_and_recommend_live_in_fmc_plus() -> None:
    store, evaluation = _case()
    values = {
        "objective": "full_ce",
        "factors": 4,
        "learning_rate": 0.003,
        "dropout": 0.0,
        "weight_decay": 0.0,
        "epochs": 0,
    }
    model = build_fmc_plus(store, 3, values, "cpu", seed=7)
    metrics, frame = recommend_fmc_plus(model, evaluation)
    assert not model.network.source.sparse
    assert metrics.users == 2
    _assert_recommendations_are_unseen(frame, store)


def test_core_runner_contains_no_model_formula_implementation() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src/capacity_probes/core_runner.py"
    ).read_text()
    for model_detail in (
        "row_probabilities",
        "smoothed_log_evidence",
        "RuleCache",
        "sampled_bce_epoch",
        "full_ce_epoch",
    ):
        assert model_detail not in source


def test_recommendation_pipelines_are_owned_by_model_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    ranking = (root / "src/capacity_probes/ranking.py").read_text()
    assert "def evaluate(" not in ranking
    assert "context_scores" not in ranking
    assert "group_items" not in ranking
    assert not (root / "src/capacity_probes/fmc_ranking.py").exists()
    for model in ("mc", "pctm", "seqrules"):
        source = (root / f"src/capacity_probes/models/{model}.py").read_text()
        assert f"def _recommend_{model}_rows(" in source
        assert "RecommendationCollector" in source
        assert "blocked[seen]" in source
        assert "topk(" in source
    fmc = (root / "src/capacity_probes/models/fmc.py").read_text()
    assert "def _recommend_factorized(" in fmc
    assert "network.all_scores" in fmc
    assert "scores[rows, items] = -torch.inf" in fmc
