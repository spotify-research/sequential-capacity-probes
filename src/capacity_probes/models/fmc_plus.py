"""FMC+ full-catalogue training, construction, and recommendation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..fmc_data import TransitionData
from ..metrics import RankingMetrics
from ..sequences import EvaluationSet, SequenceStore
from .fmc import FMCConfig, FactorizedMarkovChain, _recommend_factorized, make_optimizer


TIE_BREAK = "PyTorch top-k; equal-score order is backend-dependent"


@dataclass
class FMCPlusModel:
    config: FMCConfig
    network: FactorizedMarkovChain
    device: torch.device


def full_ce_epoch(
    model: FactorizedMarkovChain,
    data: TransitionData,
    optimizer,
    config: FMCConfig,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    """Train FMC+ for one full-catalogue cross-entropy epoch."""
    model.train()
    counts = data.count_matrix()
    row_totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    active = np.flatnonzero(row_totals).astype(np.int32)
    rng.shuffle(active)
    batches = (len(active) + batch_size - 1) // batch_size
    gradient_scale = float(row_totals.sum()) / batches
    total = 0.0
    for start in range(0, len(active), batch_size):
        rows = active[start : start + batch_size]
        sources = torch.as_tensor(rows, device=device, dtype=torch.long)
        vectors = F.dropout(model.source(sources), p=config.dropout, training=True)
        logits = vectors @ model.target.weight.T
        totals = torch.as_tensor(row_totals[rows], device=device)
        normalizers = totals * torch.logsumexp(logits, dim=1)
        lengths = counts.indptr[rows + 1] - counts.indptr[rows]
        local_rows = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
        columns = np.concatenate(
            [counts.indices[counts.indptr[row] : counts.indptr[row + 1]] for row in rows]
        )
        values = np.concatenate(
            [counts.data[counts.indptr[row] : counts.indptr[row + 1]] for row in rows]
        )
        positive = (
            logits[
                torch.as_tensor(local_rows, device=device),
                torch.as_tensor(columns, device=device, dtype=torch.long),
            ]
            * torch.as_tensor(values, device=device)
        ).sum()
        objective = normalizers.sum() - positive
        loss = objective / gradient_scale
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(objective.detach())
    return total / float(row_totals.sum())


def _refit_fmc_plus(
    data: TransitionData,
    config: FMCConfig,
    device: torch.device,
    seed: int,
    batch_size: int = 256,
) -> FactorizedMarkovChain:
    if config.objective != "full_ce":
        raise ValueError(f"FMC+ requires full_ce, got {config.objective}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FactorizedMarkovChain(data.num_items, config.factors).to(device)
    optimizer = make_optimizer(model, config)
    for epoch in range(1, config.epochs + 1):
        loss = full_ce_epoch(
            model,
            data,
            optimizer,
            config,
            device,
            batch_size,
            rng,
        )
        print(
            f"FMC_REFIT objective={config.objective} epoch={epoch}/{config.epochs} "
            f"loss={loss:.7f}",
            flush=True,
        )
    return model


def build_fmc_plus(
    store: SequenceStore,
    max_history: int,
    values: dict,
    device_name: str,
    seed: int,
) -> FMCPlusModel:
    """Fit FMC+ with full-catalogue softmax cross-entropy."""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = FMCConfig(**values)
    device = torch.device(device_name)
    data = TransitionData.from_store(store, max_history)
    return FMCPlusModel(config, _refit_fmc_plus(data, config, device, seed), device)


def recommend_fmc_plus(
    model: FMCPlusModel,
    evaluation: EvaluationSet,
) -> tuple[RankingMetrics, pd.DataFrame]:
    """Produce full-catalogue FMC+ recommendations."""
    return _recommend_factorized(model.network, evaluation, model.device)
