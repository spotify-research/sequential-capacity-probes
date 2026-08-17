"""Factorized Markov Chain construction, training, and recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..fmc_data import TransitionData
from ..metrics import (
    RankingMetrics,
    gains_for,
    recommendation_frame,
    update_totals,
)
from ..sequences import EvaluationSet, SequenceStore


TIE_BREAK = "PyTorch top-k; equal-score order is backend-dependent"


@dataclass(frozen=True)
class FMCConfig:
    objective: str
    factors: int
    learning_rate: float
    dropout: float
    weight_decay: float
    epochs: int


@dataclass
class FMCModel:
    config: FMCConfig
    network: FactorizedMarkovChain
    device: torch.device


class FactorizedMarkovChain(nn.Module):
    def __init__(self, num_items: int, factors: int, sparse: bool = False) -> None:
        super().__init__()
        self.source = nn.Embedding(num_items, factors, sparse=sparse)
        self.target = nn.Embedding(num_items, factors, sparse=sparse)
        nn.init.normal_(self.source.weight, std=0.01)
        nn.init.normal_(self.target.weight, std=0.01)

    def all_scores(self, sources: torch.Tensor) -> torch.Tensor:
        return self.source(sources) @ self.target.weight.T


def make_optimizer(model: FactorizedMarkovChain, config: FMCConfig):
    if model.source.sparse:
        if config.weight_decay:
            raise ValueError("Sparse FMC does not support weight decay")
        return torch.optim.SparseAdam(
            model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98)
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )


def sampled_bce_epoch(
    model: FactorizedMarkovChain,
    data: TransitionData,
    optimizer,
    config: FMCConfig,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = rng.permutation(data.size)
    total = 0.0
    for start in range(0, data.size, batch_size):
        indices = order[start : start + batch_size]
        sources_np, groups_np = data.sources[indices], data.groups[indices]
        sources = torch.as_tensor(sources_np, device=device, dtype=torch.long)
        positives = torch.as_tensor(data.targets[indices], device=device, dtype=torch.long)
        negatives = torch.as_tensor(
            data.sample_unseen(groups_np, rng), device=device, dtype=torch.long
        )
        vectors = F.dropout(model.source(sources), p=config.dropout, training=True)
        positive_scores = (vectors * model.target(positives)).sum(dim=1)
        negative_scores = (vectors * model.target(negatives)).sum(dim=1)
        loss = (F.softplus(-positive_scores) + F.softplus(negative_scores)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / data.size


def _refit_fmc(
    data: TransitionData,
    config: FMCConfig,
    device: torch.device,
    seed: int,
    bce_batch_size: int = 32768,
) -> FactorizedMarkovChain:
    if config.objective != "sampled_bce":
        raise ValueError(f"FMC requires sampled_bce, got {config.objective}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FactorizedMarkovChain(data.num_items, config.factors, sparse=True).to(device)
    optimizer = make_optimizer(model, config)
    for epoch in range(1, config.epochs + 1):
        loss = sampled_bce_epoch(
            model,
            data,
            optimizer,
            config,
            device,
            bce_batch_size,
            rng,
        )
        print(
            f"FMC_REFIT objective={config.objective} epoch={epoch}/{config.epochs} "
            f"loss={loss:.7f}",
            flush=True,
        )
    return model


def build_fmc(
    store: SequenceStore,
    max_history: int,
    values: dict,
    device_name: str,
    seed: int,
) -> FMCModel:
    """Fit FMC with sampled binary cross-entropy."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = FMCConfig(**values)
    device = torch.device(device_name)
    data = TransitionData.from_store(store, max_history)
    return FMCModel(config, _refit_fmc(data, config, device, seed), device)


def recommend_fmc(
    model: FMCModel,
    evaluation: EvaluationSet,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Produce full-catalogue FMC recommendations."""

    return _recommend_factorized(model.network, evaluation, model.device)


def _mask_seen(
    scores: torch.Tensor,
    evaluation: EvaluationSet,
    start: int,
    stop: int,
) -> None:
    row_parts, item_parts = [], []
    for local_row, row in enumerate(range(start, stop)):
        seen = np.unique(evaluation.store.group_items(int(evaluation.groups[row])))
        row_parts.append(np.full(len(seen), local_row, dtype=np.int64))
        item_parts.append(seen.astype(np.int64, copy=False))
    rows = torch.as_tensor(np.concatenate(row_parts), device=scores.device)
    items = torch.as_tensor(np.concatenate(item_parts), device=scores.device)
    scores[rows, items] = -torch.inf


def _recommend_factorized(
    network: FactorizedMarkovChain,
    evaluation: EvaluationSet,
    device: torch.device,
    batch_size: int = 512,
    k: int = 10,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Rank full-catalogue factorized transition scores for FMC variants."""
    network.eval()
    totals = np.zeros(4, dtype=np.float64)
    reco_users, reco_items, reco_scores, reco_ranks = [], [], [], []
    effective_k = min(k, evaluation.store.num_items)
    with torch.no_grad():
        for start in range(0, len(evaluation.users), batch_size):
            stop = min(start + batch_size, len(evaluation.users))
            last_items = np.fromiter(
                (
                    evaluation.store.group_items(int(group))[-1]
                    for group in evaluation.groups[start:stop]
                ),
                dtype=np.int64,
                count=stop - start,
            )
            scores = network.all_scores(torch.as_tensor(last_items, device=device))
            _mask_seen(scores, evaluation, start, stop)
            values, top_ids = torch.topk(scores, effective_k, dim=1, sorted=True)
            top_ids_np, values_np = top_ids.cpu().numpy(), values.cpu().numpy()
            for local_row, row in enumerate(range(start, stop)):
                gains = gains_for(top_ids_np[local_row], evaluation.targets[row])
                update_totals(totals, gains, evaluation.targets[row], k)
            reco_users.append(np.repeat(evaluation.users[start:stop], effective_k))
            reco_items.append(evaluation.store.catalog[top_ids_np.ravel()])
            reco_scores.append(values_np.ravel())
            reco_ranks.append(
                np.tile(np.arange(1, effective_k + 1, dtype=np.int32), stop - start)
            )
    users = len(evaluation.users)
    metrics = RankingMetrics(*(totals / users), 1.0, users)
    return metrics, recommendation_frame(
        reco_users, reco_items, reco_scores, reco_ranks
    )
