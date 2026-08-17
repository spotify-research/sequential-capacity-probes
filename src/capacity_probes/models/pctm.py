"""PCTM configuration, construction, scoring, and recommendation."""

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


TIE_BREAK = "ascending SHA-256 digest of UTF-8 external item ID"


@dataclass(frozen=True)
class PCTMConfig:
    counts: str
    tau: float
    kernel: str
    pop_boost: float


@dataclass
class PCTMModel:
    config: PCTMConfig
    evidence: sparse.csr_matrix
    log_popularity: np.ndarray


def count_weights(spec: str) -> np.ndarray:
    kind, *parts = spec.split(":")
    window = int(parts[0])
    offsets = np.arange(window, dtype=np.float32)
    if kind == "u":
        return np.ones(window, dtype=np.float32)
    if kind == "exp":
        return np.power(float(parts[1]), offsets, dtype=np.float32)
    if kind == "inv":
        return 1.0 / np.power(offsets + 1.0, float(parts[1]), dtype=np.float32)
    raise ValueError(f"Unknown count specification: {spec}")


def kernel_weights(spec: str, length: int) -> np.ndarray:
    if spec == "last":
        weights = np.zeros(length, dtype=np.float32)
        weights[0] = 1.0
        return weights
    kind, *parts = spec.split(":")
    ages = np.arange(length, dtype=np.float32)
    if kind == "exp":
        weights = np.power(float(parts[0]), ages, dtype=np.float32)
    elif kind == "inv":
        weights = 1.0 / np.power(ages + 1.0, float(parts[0]), dtype=np.float32)
    elif kind == "tail":
        head = min(int(parts[0]), length)
        recent_mass, beta = float(parts[1]), float(parts[2])
        weights = np.zeros(length, dtype=np.float32)
        recent = np.power(beta, np.arange(head, dtype=np.float32), dtype=np.float32)
        weights[:head] = recent_mass * recent / recent.sum()
        if length > head:
            weights[head:] = (1.0 - recent_mass) / (length - head)
        else:
            weights /= weights.sum()
    else:
        raise ValueError(f"Unknown history kernel: {spec}")
    return weights / weights.sum()


def smoothed_log_evidence(counts: sparse.csr_matrix, tau: float) -> sparse.csr_matrix:
    """Return the candidate-dependent term of log P(a|b)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    result = counts.copy().astype(np.float32)
    result.data = np.log1p(result.data * counts.shape[1] / tau).astype(np.float32)
    return result


def log_popularity(store: SequenceStore) -> np.ndarray:
    frequency = store.item_frequency()
    return np.log((frequency + 1.0) / (frequency.sum() + store.num_items)).astype(
        np.float32
    )


def build_pctm(store: SequenceStore, cache_dir: Path, values: dict) -> PCTMModel:
    """Fit PCTM transition evidence and its popularity background."""
    config = PCTMConfig(**values)
    weights = count_weights(config.counts)
    cache = OffsetCache(cache_dir, store)
    cache.build(len(weights))
    counts = cache.combine(config.counts, weights)
    evidence = smoothed_log_evidence(counts, config.tau)
    return PCTMModel(config, evidence, log_popularity(store))


def recommend_pctm(
    model: PCTMModel,
    evaluation: EvaluationSet,
    max_history: int,
) -> tuple[RankingMetrics, pd.DataFrame | None]:
    """Produce full-catalogue PCTM recommendations for the evaluation users."""
    histories = evaluation.history_matrix(
        max_history,
        lambda length: kernel_weights(model.config.kernel, length),
    )
    jobs = min(int(os.environ.get("PCTM_EVAL_JOBS", "1")), len(evaluation.users))
    if jobs <= 1 or len(evaluation.users) < 20_000:
        return _recommend_pctm_rows(model, evaluation, histories)
    edges = np.linspace(0, len(evaluation.users), jobs + 1, dtype=np.int64)
    bounds = [(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:])]
    global _PCTM_STATE
    _PCTM_STATE = model, evaluation, histories
    try:
        with mp.get_context("fork").Pool(jobs) as pool:
            return merge_results(pool.map(_recommend_pctm_shard, bounds))
    finally:
        _PCTM_STATE = None


def _recommend_pctm_rows(
    model: PCTMModel,
    evaluation: EvaluationSet,
    histories: sparse.csr_matrix,
    k: int = 10,
    batch_size: int = 512,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Implement Algorithm 2 scoring, filtering, and deterministic ranking."""
    ranks = tie_ranks(evaluation.store.catalog, "sha256")
    boost = model.config.pop_boost
    if boost < 0:
        background_order = np.lexsort((ranks, model.log_popularity))
    elif boost > 0:
        background_order = np.lexsort((ranks, -model.log_popularity))
    else:
        background_order = np.argsort(ranks)
    blocked = np.zeros(evaluation.store.num_items, dtype=bool)
    collector = RecommendationCollector(evaluation, k)
    for start in range(0, len(evaluation.users), batch_size):
        stop = min(start + batch_size, len(evaluation.users))
        context_scores = (histories[start:stop] @ model.evidence).tocsr()
        for local_row, row in enumerate(range(start, stop)):
            begin, end = context_scores.indptr[local_row : local_row + 2]
            candidate_ids = context_scores.indices[begin:end]
            candidate_scores = context_scores.data[begin:end]
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
            scores += np.float32(boost) * model.log_popularity[all_ids]
            selected = topk(all_ids, scores, k, ranks)
            collector.add(row, all_ids[selected], scores[selected], reachable)
            blocked[seen] = False
            blocked[candidate_ids] = False
    return collector.finish()


_PCTM_STATE: tuple | None = None


def _recommend_pctm_shard(bounds: tuple[int, int]):
    if _PCTM_STATE is None:
        raise RuntimeError("PCTM worker state was not initialized")
    model, evaluation, histories = _PCTM_STATE
    start, stop = bounds
    shard = EvaluationSet(
        evaluation.store,
        evaluation.users[start:stop],
        evaluation.groups[start:stop],
        evaluation.targets[start:stop],
    )
    return _recommend_pctm_rows(model, shard, histories[start:stop])
