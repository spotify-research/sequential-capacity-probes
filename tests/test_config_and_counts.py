from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from rectools import Columns
from scipy import sparse

from capacity_probes.models.pctm import count_weights, kernel_weights
from capacity_probes.models.pctm import smoothed_log_evidence
from capacity_probes.models.mc import row_probabilities
from capacity_probes.ranking import stable_hash_ranks, topk
from capacity_probes.sequences import EvaluationSet, SequenceStore


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            Columns.User: [2, 1, 1, 2, 1],
            Columns.Item: [12, 10, 11, 10, 12],
            Columns.Datetime: pd.to_datetime([2, 1, 2, 1, 2], unit="s"),
            "_source_order": np.arange(5),
        }
    )


def test_count_and_history_kernels_are_frozen() -> None:
    np.testing.assert_allclose(count_weights("exp:3:0.5"), [1.0, 0.5, 0.25])
    np.testing.assert_allclose(count_weights("inv:3:1"), [1.0, 0.5, 1 / 3])
    assert kernel_weights("last", 3).tolist() == [1.0, 0.0, 0.0]
    assert np.isclose(kernel_weights("tail:2:0.8:0.5", 4).sum(), 1.0)


def test_sequence_order_is_timestamp_then_source_order() -> None:
    store = SequenceStore.from_frame(frame())
    assert store.catalog.tolist() == [10, 11, 12]
    assert store.group_items(store.user_to_group[1]).tolist() == [0, 1, 2]
    assert store.group_items(store.user_to_group[2]).tolist() == [0, 2]


def test_sparse_pctm_term_differs_from_full_log_probability_by_row_constant() -> None:
    counts = sparse.csr_matrix(np.asarray([[2.0, 0.0, 1.0], [0.0, 4.0, 0.0]]))
    tau = 5.0
    sparse_term = smoothed_log_evidence(counts, tau).toarray()
    full = np.empty_like(sparse_term)
    for source in range(2):
        for candidate in range(3):
            full[source, candidate] = log(
                (counts[source, candidate] + tau / 3) / (counts[source].sum() + tau)
            )
    differences = full - sparse_term
    np.testing.assert_allclose(differences[:, 0], differences[:, 1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(differences[:, 1], differences[:, 2], rtol=0, atol=1e-6)


def test_mc_is_unsmoothed_row_conditional() -> None:
    counts = sparse.csr_matrix(np.asarray([[2.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(row_probabilities(counts).toarray(), [[2 / 3, 1 / 3], [0, 0]])


def test_hash_tie_break_is_stable_and_catalog_based() -> None:
    catalog = np.asarray([100, 3, 42, 7], dtype=np.int64)
    ranks = stable_hash_ranks(catalog)
    ids = np.arange(4, dtype=np.int32)
    selected = topk(ids, np.zeros(4, dtype=np.float32), 3, ranks)
    assert selected.tolist() == np.argsort(ranks)[:3].tolist()


def test_holdout_drops_unknown_items_and_users() -> None:
    store = SequenceStore.from_frame(frame())
    holdout = pd.DataFrame(
        {
            Columns.User: [1, 2, 3],
            Columns.Item: [12, 999, 10],
            Columns.Datetime: pd.to_datetime([3, 3, 1], unit="s"),
            "_source_order": [0, 1, 2],
        }
    )
    evaluation = EvaluationSet.from_holdout(store, holdout)
    assert evaluation.users.tolist() == [1]

