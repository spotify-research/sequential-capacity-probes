"""Empirical Markov Chain construction and recommendation."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..counts import OffsetCache
from ..metrics import RankingMetrics
from ..ranking import RecommendationCollector, merge_results, tie_ranks, topk
from ..sequences import EvaluationSet, SequenceStore


TIE_BREAK = "ascending external item ID"


@dataclass(frozen=True)
class MCConfig:
    transition_distance: int = 1
    smoothing: None = None


@dataclass
class MCModel:
    config: MCConfig
    evidence: sparse.csr_matrix


def row_probabilities(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    counts = counts.tocsr().astype(np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    inverse = np.zeros_like(totals, dtype=np.float32)
    nonzero = totals > 0
    inverse[nonzero] = 1.0 / totals[nonzero]
    result = (sparse.diags(inverse) @ counts).tocsr().astype(np.float32)
    result.sort_indices()
    return result


def build_mc(store: SequenceStore, cache_dir: Path) -> MCModel:
    """Fit the empirical first-order Markov chain."""
    config = MCConfig()
    cache = OffsetCache(cache_dir, store)
    cache.build(config.transition_distance)
    counts = cache.combine("u:1", np.ones(1, dtype=np.float32))
    return MCModel(config, row_probabilities(counts))


def recommend_mc(
    model: MCModel,
    evaluation: EvaluationSet,
) -> tuple[RankingMetrics, pd.DataFrame | None]:
    """Produce full-catalogue MC recommendations for the evaluation users."""
    histories = evaluation.history_matrix(
        model.config.transition_distance,
        lambda length: np.ones(length, dtype=np.float32),
    )
    jobs = min(int(os.environ.get("PCTM_EVAL_JOBS", "1")), len(evaluation.users))
    if jobs <= 1 or len(evaluation.users) < 20_000:
        return _recommend_mc_rows(model, evaluation, histories)
    edges = np.linspace(0, len(evaluation.users), jobs + 1, dtype=np.int64)
    bounds = [(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:])]
    global _MC_STATE
    _MC_STATE = model, evaluation, histories
    try:
        with mp.get_context("fork").Pool(jobs) as pool:
            return merge_results(pool.map(_recommend_mc_shard, bounds))
    finally:
        _MC_STATE = None


def _recommend_mc_rows(
    model: MCModel,
    evaluation: EvaluationSet,
    histories: sparse.csr_matrix,
    k: int = 10,
    batch_size: int = 512,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Score MC transitions, filter history, and rank the full catalogue."""
    ranks = tie_ranks(evaluation.store.catalog, "external_id")
    background_order = np.argsort(ranks)
    blocked = np.zeros(evaluation.store.num_items, dtype=bool)
    collector = RecommendationCollector(evaluation, k)
    for start in range(0, len(evaluation.users), batch_size):
        stop = min(start + batch_size, len(evaluation.users))
        batch_scores = (histories[start:stop] @ model.evidence).tocsr()
        for local_row, row in enumerate(range(start, stop)):
            begin, end = batch_scores.indptr[local_row : local_row + 2]
            candidate_ids = batch_scores.indices[begin:end]
            candidate_scores = batch_scores.data[begin:end]
            seen = np.unique(evaluation.store.group_items(int(evaluation.groups[row])))
            blocked[seen] = True
            keep = ~blocked[candidate_ids]
            candidate_ids, candidate_scores = candidate_ids[keep], candidate_scores[keep]
            reachable = bool(np.any(np.isin(evaluation.targets[row], candidate_ids)))
            blocked[candidate_ids] = True
            background = []
            for item in background_order:
                if not blocked[item]:
                    background.append(int(item))
                    if len(background) == k:
                        break
            background_ids = np.asarray(background, dtype=np.int32)
            all_ids = np.concatenate((candidate_ids, background_ids))
            scores = np.concatenate(
                (candidate_scores, np.zeros(len(background_ids), dtype=np.float32))
            )
            selected = topk(all_ids, scores, k, ranks)
            collector.add(row, all_ids[selected], scores[selected], reachable)
            blocked[seen] = False
            blocked[candidate_ids] = False
    return collector.finish()


_MC_STATE: tuple | None = None


def _recommend_mc_shard(bounds: tuple[int, int]):
    if _MC_STATE is None:
        raise RuntimeError("MC worker state was not initialized")
    model, evaluation, histories = _MC_STATE
    start, stop = bounds
    shard = EvaluationSet(
        evaluation.store,
        evaluation.users[start:stop],
        evaluation.groups[start:stop],
        evaluation.targets[start:stop],
    )
    return _recommend_mc_rows(model, shard, histories[start:stop])
