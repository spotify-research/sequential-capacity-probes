"""Model-agnostic ranking and result-collection primitives."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .metrics import (
    RankingMetrics,
    gains_for,
    recommendation_frame,
    update_totals,
)
from .sequences import EvaluationSet


def stable_hash_ranks(catalog: np.ndarray) -> np.ndarray:
    digests = [hashlib.sha256(str(item).encode("utf-8")).digest() for item in catalog]
    order = sorted(range(len(catalog)), key=lambda i: (digests[i], str(catalog[i])))
    ranks = np.empty(len(catalog), dtype=np.int64)
    ranks[np.asarray(order, dtype=np.int64)] = np.arange(len(catalog), dtype=np.int64)
    return ranks


def tie_ranks(catalog: np.ndarray, tie_break: str) -> np.ndarray:
    if tie_break == "sha256":
        return stable_hash_ranks(catalog)
    if tie_break == "external_id":
        order = np.argsort(catalog, kind="stable")
        ranks = np.empty(len(catalog), dtype=np.int64)
        ranks[order] = np.arange(len(catalog), dtype=np.int64)
        return ranks
    raise ValueError(f"Unknown tie break: {tie_break}")


def topk(ids: np.ndarray, scores: np.ndarray, k: int, ranks: np.ndarray) -> np.ndarray:
    """Return positions ordered by descending score and the supplied tie ranks."""
    if len(ids) <= k:
        return np.lexsort((ranks[ids], -scores))
    partition = np.argpartition(scores, len(scores) - k)[-k:]
    threshold = scores[partition].min()
    greater = np.flatnonzero(scores > threshold)
    equal = np.flatnonzero(scores == threshold)
    need = k - len(greater)
    if need < len(equal):
        equal = equal[np.argsort(ranks[ids[equal]], kind="stable")[:need]]
    selected = np.concatenate((greater, equal))
    return selected[np.lexsort((ranks[ids[selected]], -scores[selected]))]


class RecommendationCollector:
    """Accumulate metrics and the publication recommendation table."""

    def __init__(self, evaluation: EvaluationSet, k: int = 10) -> None:
        self.evaluation, self.k = evaluation, k
        self.totals = np.zeros(4, dtype=np.float64)
        self.reachable = 0.0
        self.users: list[np.ndarray] = []
        self.items: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []
        self.ranks: list[np.ndarray] = []

    def add(
        self,
        row: int,
        item_ids: np.ndarray,
        scores: np.ndarray,
        reachable: bool,
    ) -> None:
        targets = self.evaluation.targets[row]
        update_totals(self.totals, gains_for(item_ids, targets), targets, self.k)
        self.reachable += float(reachable)
        size = len(item_ids)
        self.users.append(np.full(size, self.evaluation.users[row], dtype=np.int64))
        self.items.append(self.evaluation.store.catalog[item_ids])
        self.scores.append(scores)
        self.ranks.append(np.arange(1, size + 1, dtype=np.int32))

    def finish(self) -> tuple[RankingMetrics, pd.DataFrame]:
        count = len(self.evaluation.users)
        metrics = RankingMetrics(
            *(self.totals / count), self.reachable / count, count
        )
        return metrics, recommendation_frame(
            self.users, self.items, self.scores, self.ranks
        )


def merge_results(
    outputs: list[tuple[RankingMetrics, pd.DataFrame]],
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Merge independently ranked user shards without changing their order."""
    total_users = sum(metrics.users for metrics, _ in outputs)
    values = sum(
        np.asarray(
            (m.ndcg10, m.recall10, m.hit_rate10, m.precision10, m.reachable)
        )
        * m.users
        for m, _ in outputs
    )
    metrics = RankingMetrics(*(values / total_users), total_users)
    return metrics, pd.concat([frame for _, frame in outputs], ignore_index=True)
