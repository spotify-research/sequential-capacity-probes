from __future__ import annotations

import numpy as np
import torch

from capacity_probes.models.fmc import (
    FMCConfig,
    FactorizedMarkovChain,
    make_optimizer,
)
from capacity_probes.models.fmc_plus import full_ce_epoch
from capacity_probes.fmc_data import TransitionData
from capacity_probes.sequences import SequenceStore


def store() -> SequenceStore:
    return SequenceStore(
        catalog=np.asarray([10, 11, 12, 13, 14], dtype=np.int64),
        users=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int64),
        items=np.asarray([0, 1, 2, 0, 1, 3], dtype=np.int32),
        group_users=np.asarray([1, 2], dtype=np.int64),
        starts=np.asarray([0, 3], dtype=np.int64),
        ends=np.asarray([3, 6], dtype=np.int64),
        user_to_group={1: 0, 2: 1},
    )


def test_transition_window_is_causal_and_recent() -> None:
    data = TransitionData.from_store(store(), 1)
    np.testing.assert_array_equal(data.sources, [1, 1])
    np.testing.assert_array_equal(data.targets, [2, 3])


def test_negative_sampler_excludes_complete_history() -> None:
    data = TransitionData.from_store(store(), 2)
    groups = np.asarray([0] * 500 + [1] * 500, dtype=np.int32)
    negatives = data.sample_unseen(groups, np.random.default_rng(7))
    assert set(negatives[:500]).isdisjoint({0, 1, 2})
    assert set(negatives[500:]).isdisjoint({0, 1, 3})


def test_source_and_target_embeddings_are_unshared() -> None:
    model = FactorizedMarkovChain(5, 3, sparse=True)
    assert model.source.weight.data_ptr() != model.target.weight.data_ptr()


def test_full_ce_matches_expanded_transition_loss() -> None:
    data = TransitionData.from_store(store(), 2)
    config = FMCConfig("full_ce", 4, 0.0, 0.0, 0.0, 1)
    model = FactorizedMarkovChain(data.num_items, config.factors)
    with torch.no_grad():
        logits = model.source.weight @ model.target.weight.T
        expected = -torch.log_softmax(logits, dim=1)[
            torch.as_tensor(data.sources, dtype=torch.long),
            torch.as_tensor(data.targets, dtype=torch.long),
        ].mean()
    loss = full_ce_epoch(
        model,
        data,
        make_optimizer(model, config),
        config,
        torch.device("cpu"),
        2,
        np.random.default_rng(1),
    )
    assert np.isclose(loss, float(expected), atol=1e-5)
