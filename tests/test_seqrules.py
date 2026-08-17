from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rectools import Columns

from capacity_probes.counts import OffsetCache
from capacity_probes.ranking import tie_ranks
from capacity_probes.models.seqrules import (
    RuleCache,
    RuleSpec,
    history_matrix,
    history_weights,
    rule_weights,
)
from capacity_probes.sequences import EvaluationSet, SequenceStore


def _store() -> SequenceStore:
    rows = [
        (1, 10, "2020-01-01"),
        (1, 11, "2020-01-02"),
        (1, 12, "2020-01-03"),
        (1, 13, "2020-01-04"),
        (2, 10, "2020-01-01"),
        (2, 11, "2020-01-02"),
        (2, 14, "2020-01-03"),
    ]
    frame = pd.DataFrame(rows, columns=[Columns.User, Columns.Item, Columns.Datetime])
    frame[Columns.Datetime] = pd.to_datetime(frame[Columns.Datetime])
    frame[Columns.Weight] = 1.0
    frame["_source_order"] = range(len(frame))
    return SequenceStore.from_frame(frame)


def _literal_rules(store: SequenceStore, steps: int, weighting: str) -> np.ndarray:
    result = np.zeros((store.num_items, store.num_items), dtype=np.float32)
    weights = rule_weights(steps, weighting)
    for group in range(len(store.group_users)):
        sequence = store.group_items(group)
        for target_pos, target in enumerate(sequence):
            for distance in range(1, min(steps, target_pos) + 1):
                result[sequence[target_pos - distance], target] += weights[distance - 1]
    return result


def test_sparse_rules_match_literal_definition(tmp_path: Path) -> None:
    store = _store()
    offsets = OffsetCache(tmp_path, store)
    offsets.build(3)
    actual = RuleCache(offsets, store).matrix(RuleSpec(3, "div", 0, False))
    np.testing.assert_allclose(actual.toarray(), _literal_rules(store, 3, "div"))


def test_multi_item_history_scoring_matches_literal_sum(tmp_path: Path) -> None:
    store = _store()
    offsets = OffsetCache(tmp_path, store)
    offsets.build(3)
    rules = RuleCache(offsets, store).matrix(RuleSpec(3, "div", 0, False))
    evaluation = EvaluationSet(
        store,
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int32),
        (np.asarray([4], dtype=np.int32),),
    )
    actual = (history_matrix(evaluation, 3, "div", 10) @ rules).toarray()[0]
    sequence, weights = store.group_items(0), history_weights(3, "div")
    expected = sum(
        weights[index] * rules.getrow(sequence[-index - 1]).toarray()[0]
        for index in range(3)
    )
    np.testing.assert_allclose(actual, expected)


def test_pruning_keeps_strongest_rule_with_item_id_ties(tmp_path: Path) -> None:
    store = _store()
    offsets = OffsetCache(tmp_path, store)
    offsets.build(3)
    cache = RuleCache(offsets, store)
    full = cache.matrix(RuleSpec(3, "same", 0, False))
    pruned = cache.matrix(RuleSpec(3, "same", 1, False))
    assert np.all(np.diff(pruned.indptr) <= 1)
    for row in range(full.shape[0]):
        begin, end = full.indptr[row : row + 2]
        if begin == end:
            continue
        expected = full.indices[begin:end][
            np.lexsort((full.indices[begin:end], -full.data[begin:end]))[0]
        ]
        assert pruned.getrow(row).indices[0] == expected


def test_idf_changes_source_rows_only(tmp_path: Path) -> None:
    store = _store()
    offsets = OffsetCache(tmp_path, store)
    offsets.build(2)
    cache = RuleCache(offsets, store)
    plain = cache.matrix(RuleSpec(2, "same", 0, False))
    weighted = cache.matrix(RuleSpec(2, "same", 0, True))
    for row in range(store.num_items):
        plain_row, weighted_row = plain.getrow(row), weighted.getrow(row)
        if not len(weighted_row.data):
            continue
        np.testing.assert_array_equal(weighted_row.indices, plain_row.indices)
        ratios = weighted_row.data / plain_row.data
        np.testing.assert_allclose(ratios, np.full(len(ratios), ratios[0]))


def test_publication_ties_use_ascending_external_item_id() -> None:
    catalog = np.asarray([50, 10, 30], dtype=np.int64)
    np.testing.assert_array_equal(tie_ranks(catalog, "external_id"), [2, 0, 1])
