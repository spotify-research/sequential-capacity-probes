"""Sequential Rules construction and recommendation."""

from __future__ import annotations

import multiprocessing as mp
import os
import re
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
class RuleSpec:
    steps: int
    weighting: str
    pruning: int
    idf_weight: bool


@dataclass(frozen=True)
class SeqRulesConfig:
    rules: RuleSpec
    history_length: int
    history_weighting: str

    @classmethod
    def from_dict(cls, values: dict) -> "SeqRulesConfig":
        return cls(
            RuleSpec(**values["rules"]),
            values["history_length"],
            values["history_weighting"],
        )


@dataclass
class SeqRulesModel:
    config: SeqRulesConfig
    rules: sparse.csr_matrix


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def positional_weights(weighting: str, positions: np.ndarray) -> np.ndarray:
    positions = positions.astype(np.float32, copy=False)
    if weighting == "same":
        return np.ones(len(positions), dtype=np.float32)
    if weighting == "div":
        return 1.0 / positions
    if weighting == "quadratic":
        return 1.0 / np.square(positions)
    if weighting == "log":
        return 1.0 / np.log10(positions + np.float32(1.7))
    if weighting == "linear":
        return np.where(positions <= 100, 1.0 - 0.1 * positions, 0.0)
    raise ValueError(f"Unknown SeqRules weighting: {weighting}")


def rule_weights(steps: int, weighting: str) -> np.ndarray:
    return positional_weights(weighting, np.arange(1, steps + 1, dtype=np.float32))


def history_weights(length: int, weighting: str) -> np.ndarray:
    result = np.ones(length, dtype=np.float32)
    if length > 1:
        result[1:] = positional_weights(
            weighting, np.arange(2, length + 1, dtype=np.float32)
        )
    return result


def history_matrix(
    evaluation: EvaluationSet,
    history_length: int,
    history_weighting: str,
    max_history: int,
) -> sparse.csr_matrix:
    limit = min(history_length, max_history)
    return evaluation.history_matrix(
        limit,
        lambda length: history_weights(length, history_weighting),
    )


def _idf_weights(store: SequenceStore) -> np.ndarray:
    raw = np.log(len(store.group_users) / store.item_frequency())
    low, high = raw.min(), raw.max()
    if high == low:
        return np.ones(store.num_items, dtype=np.float32)
    return ((raw - low) / (high - low)).astype(np.float32)


def _prune_rows(matrix: sparse.csr_matrix, limit: int) -> sparse.csr_matrix:
    if limit <= 0:
        return matrix
    matrix = matrix.tocsr()
    keep = np.zeros(matrix.nnz, dtype=bool)
    row_counts = np.empty(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        begin, end = matrix.indptr[row : row + 2]
        count = end - begin
        if count <= limit:
            keep[begin:end], row_counts[row] = True, count
            continue
        order = np.lexsort((matrix.indices[begin:end], -matrix.data[begin:end]))[:limit]
        keep[begin + order], row_counts[row] = True, limit
    indptr = np.empty(matrix.shape[0] + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])
    return sparse.csr_matrix(
        (matrix.data[keep], matrix.indices[keep], indptr),
        shape=matrix.shape,
        dtype=np.float32,
    )


class RuleCache:
    def __init__(self, offset_cache: OffsetCache, store: SequenceStore):
        self.offset_cache, self.store = offset_cache, store
        self.directory = offset_cache.directory / "seqrules_v1"
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, spec: RuleSpec, base: bool) -> Path:
        stem = f"s{spec.steps}_{_slug(spec.weighting)}_idf{int(spec.idf_weight)}"
        name = f"base_{stem}.npz" if base else f"rules_{stem}_p{spec.pruning}.npz"
        return self.directory / name

    def _load(self, path: Path) -> sparse.csr_matrix:
        result = sparse.load_npz(path).tocsr()
        if result.shape != (self.store.num_items, self.store.num_items):
            raise ValueError(f"Stale SeqRules cache has shape {result.shape}: {path}")
        return result

    def _base(self, spec: RuleSpec) -> sparse.csr_matrix:
        path = self._path(spec, True)
        if path.exists():
            return self._load(path)
        if spec.idf_weight:
            plain = RuleSpec(spec.steps, spec.weighting, 0, False)
            result = self._base(plain).multiply(_idf_weights(self.store)[:, None])
        else:
            result = None
            for offset, weight in enumerate(rule_weights(spec.steps, spec.weighting), 1):
                if weight == 0:
                    continue
                current = sparse.load_npz(self.offset_cache.offset_path(offset)).tocsr()
                weighted = current.multiply(np.float32(weight))
                result = weighted if result is None else result + weighted
            if result is None:
                shape = (self.store.num_items, self.store.num_items)
                result = sparse.csr_matrix(shape, dtype=np.float32)
        result = result.tocsr().astype(np.float32)
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        sparse.save_npz(path, result, compressed=False)
        return result

    def matrix(self, spec: RuleSpec) -> sparse.csr_matrix:
        if spec.pruning <= 0:
            return self._base(spec)
        path = self._path(spec, False)
        if path.exists():
            return self._load(path)
        result = _prune_rows(self._base(spec), spec.pruning)
        result.eliminate_zeros()
        result.sort_indices()
        sparse.save_npz(path, result, compressed=False)
        return result


def build_seqrules(
    store: SequenceStore,
    cache_dir: Path,
    values: dict,
) -> SeqRulesModel:
    """Fit the configured directed sequential-rule matrix."""
    config = SeqRulesConfig.from_dict(values)
    offsets = OffsetCache(cache_dir, store)
    offsets.build(config.rules.steps)
    rules = RuleCache(offsets, store).matrix(config.rules)
    return SeqRulesModel(config, rules)


def recommend_seqrules(
    model: SeqRulesModel,
    evaluation: EvaluationSet,
    max_history: int,
) -> tuple[RankingMetrics, pd.DataFrame | None]:
    """Produce full-catalogue Sequential Rules recommendations."""
    histories = history_matrix(
        evaluation,
        model.config.history_length,
        model.config.history_weighting,
        max_history,
    )
    jobs = min(int(os.environ.get("PCTM_EVAL_JOBS", "1")), len(evaluation.users))
    if jobs <= 1 or len(evaluation.users) < 20_000:
        return _recommend_seqrules_rows(model, evaluation, histories)
    edges = np.linspace(0, len(evaluation.users), jobs + 1, dtype=np.int64)
    bounds = [(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:])]
    global _SEQRULES_STATE
    _SEQRULES_STATE = model, evaluation, histories
    try:
        with mp.get_context("fork").Pool(jobs) as pool:
            return merge_results(pool.map(_recommend_seqrules_shard, bounds))
    finally:
        _SEQRULES_STATE = None


def _recommend_seqrules_rows(
    model: SeqRulesModel,
    evaluation: EvaluationSet,
    histories: sparse.csr_matrix,
    k: int = 10,
    batch_size: int = 512,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Score sequential rules, filter history, and rank the full catalogue."""
    ranks = tie_ranks(evaluation.store.catalog, "external_id")
    background_order = np.argsort(ranks)
    blocked = np.zeros(evaluation.store.num_items, dtype=bool)
    collector = RecommendationCollector(evaluation, k)
    for start in range(0, len(evaluation.users), batch_size):
        stop = min(start + batch_size, len(evaluation.users))
        batch_scores = (histories[start:stop] @ model.rules).tocsr()
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


_SEQRULES_STATE: tuple | None = None


def _recommend_seqrules_shard(bounds: tuple[int, int]):
    if _SEQRULES_STATE is None:
        raise RuntimeError("SeqRules worker state was not initialized")
    model, evaluation, histories = _SEQRULES_STATE
    start, stop = bounds
    shard = EvaluationSet(
        evaluation.store,
        evaluation.users[start:stop],
        evaluation.groups[start:stop],
        evaluation.targets[start:stop],
    )
    return _recommend_seqrules_rows(model, shard, histories[start:stop])
