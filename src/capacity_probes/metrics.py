from __future__ import annotations

from dataclasses import dataclass
from math import log2

import numpy as np
import pandas as pd
from rectools import Columns
from rectools.metrics import NDCG, Recall, calc_metrics


@dataclass(frozen=True)
class RankingMetrics:
    ndcg10: float
    recall10: float
    hit_rate10: float
    precision10: float
    reachable: float
    users: int


def gains_for(predicted: np.ndarray, targets: np.ndarray) -> np.ndarray:
    target_ids, target_counts = np.unique(targets, return_counts=True)
    positions = np.searchsorted(target_ids, predicted)
    valid = positions < len(target_ids)
    matching = np.zeros(len(predicted), dtype=bool)
    valid_positions = np.flatnonzero(valid)
    matching[valid_positions] = (
        target_ids[positions[valid_positions]] == predicted[valid_positions]
    )
    gains = np.zeros(len(predicted), dtype=np.int32)
    gains[matching] = target_counts[positions[matching]]
    return gains


def update_totals(totals: np.ndarray, gains: np.ndarray, targets: np.ndarray, k: int) -> None:
    hits = int(gains.sum())
    ideal = min(k, len(targets))
    idcg = sum(1.0 / log2(rank + 2) for rank in range(ideal))
    dcg = sum(
        int(gain) / log2(rank + 2) for rank, gain in enumerate(gains) if gain
    )
    totals += (dcg / idcg, hits / len(targets), hits > 0, hits / k)


def official_metrics(
    recommendations: pd.DataFrame,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    metrics = {f"recall@{k}": Recall(k=k), f"ndcg@{k}": NDCG(k=k, divide_by_achievable=True)}
    result = calc_metrics(
        metrics=metrics,
        reco=recommendations,
        interactions=holdout.drop(columns="_source_order", errors="ignore"),
        prev_interactions=train.drop(columns="_source_order", errors="ignore"),
        catalog=train[Columns.Item].unique(),
    )
    return {name: float(value) for name, value in result.items()}


def recommendation_frame(
    users: list[np.ndarray],
    items: list[np.ndarray],
    scores: list[np.ndarray],
    ranks: list[np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            Columns.User: np.concatenate(users),
            Columns.Item: np.concatenate(items),
            Columns.Score: np.concatenate(scores),
            Columns.Rank: np.concatenate(ranks),
        }
    )

