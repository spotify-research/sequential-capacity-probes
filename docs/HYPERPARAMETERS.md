# Frozen hyperparameters

This document records the final parameters used for Table 1. They are also
machine-readable in `configs/table1.json`; runners read that file directly.
Values were selected using an inner split of the training data. The published
holdout was used once for final evaluation.

Dataset order throughout is Beauty, Sports, Toys, MovieLens 1M, and MovieLens
20M. Maximum history length is 50 for the Amazon datasets and 200 for
MovieLens.

## MC

MC has no tuned parameter. It uses distance-one transition counts, no
smoothing, the last observed item, no popularity term, and ascending external
item ID for score ties.

## FMC

| Data | Factors | Learning rate | Dropout | Weight decay | Epochs |
|---|---:|---:|---:|---:|---:|
| Beauty | 256 | .003 | 0 | 0 | 5 |
| Sports | 256 | .003 | 0 | 0 | 4 |
| Toys | 256 | .003 | .2 | 0 | 8 |
| ML-1M | 256 | .003 | 0 | 0 | 29 |
| ML-20M | 256 | .003 | 0 | 0 | 7 |

FMC uses separate source and target embeddings. Its objective is binary
cross-entropy with one uniformly sampled item outside the user's complete
training history per positive. Positive transitions are formed from the most
recent 50 events per Amazon user and 200 events per MovieLens user. It uses
SparseAdam with betas (.9, .98), batch size 32,768, and refit seed 20270710.
Ranking uses PyTorch full-catalogue `topk`; ordering of exact score ties is
backend-dependent.

## FMC+

| Data | Factors | Learning rate | Dropout | Weight decay | Epochs |
|---|---:|---:|---:|---:|---:|
| Beauty | 128 | .003 | 0 | 0 | 16 |
| Sports | 256 | .003 | 0 | 0 | 9 |
| Toys | 256 | .003 | 0 | 0 | 12 |
| ML-1M | 64 | .01 | 0 | 0 | 21 |
| ML-20M | 128 | .003 | 0 | 0 | 45 |

FMC+ uses the same unshared embedding architecture and the full-catalogue
softmax cross-entropy over the same capped positive-transition set. It uses
Adam with betas (.9, .98), 256 active source items per batch, and refit seed
20270710. Its exact-tie behavior is the same as FMC and is recorded in result
artifacts rather than presented as deterministic.

## Sequential Rules

| Data | Rule window | Rule weight | Top rules/source | IDF | History length | History weight |
|---|---:|---|---:|:---:|---:|---|
| Beauty | 10 | logarithmic | 100 | no | 10 | quadratic |
| Sports | 30 | logarithmic | unpruned | no | 20 | inverse |
| Toys | 30 | logarithmic | unpruned | no | 10 | inverse |
| ML-1M | 10 | inverse | 200 | yes | 10 | quadratic |
| ML-20M | 20 | inverse | 100 | yes | 5 | quadratic |

“Inverse” is $1/d$, “quadratic” is $1/d^2$, and “logarithmic” is
$1/\log_{10}(d + 1.7)$, where $d$ starts at one. IDF is min-max normalized
and multiplies source rows before pruning. Ties use ascending external item ID.

## PCTM

The count kernel notation is `kind:window:decay`; the history kernel notation
is `tail:head:recent_mass:decay` or `exp:decay`.

| Data | Count kernel | Tau | History kernel | Popularity coefficient |
|---|---|---:|---|---:|
| Beauty | inverse:10:.5 | 11,250 | exponential:.6 | -.04 |
| Sports | inverse:10:.5 | 15,000 | tail:10:.8:.7 | 0 |
| Toys | exponential:20:.7 | 15,000 | tail:7:.8:.6 | 0 |
| ML-1M | inverse:20:1 | 150 | tail:7:.8:.6 | -.4 |
| ML-20M | exponential:10:.5 | 225 | tail:7:.8:.6 | -.5 |

PCTM uses a uniform background prior and ascending SHA-256 digest of the UTF-8
external item ID for score ties.

## SASRec+ and eSASRec

Both methods run through the original eSASRec repository at commit
`2b039927ceb4c7654131d7ac7c43ea124b49d240` and RecTools 0.13.0. SASRec+ uses
the standard SASRec block; eSASRec uses LiGR layers with a key-padding mask.

| Data | Model | Factors | Length | Blocks | Heads | Dropout | LiGR FF multiplier |
|---|---|---:|---:|---:|---:|---:|---:|
| Beauty | SASRec+ | 64 | 50 | 1 | 1 | .2 | — |
| Beauty | eSASRec | 64 | 50 | 1 | 1 | .2 | 4 |
| Sports | SASRec+ | 64 | 50 | 1 | 1 | .2 | — |
| Sports | eSASRec | 64 | 50 | 1 | 1 | .2 | 2 |
| Toys | SASRec+ | 64 | 50 | 1 | 1 | .2 | — |
| Toys | eSASRec | 64 | 50 | 1 | 1 | .2 | 1 |
| ML-1M | SASRec+ | 64 | 200 | 2 | 1 | .1 | — |
| ML-1M | eSASRec | 64 | 200 | 2 | 1 | .1 | default |
| ML-20M | SASRec+ | 256 | 200 | 4 | 8 | .2 | — |
| ML-20M | eSASRec | 256 | 200 | 4 | 8 | .2 | default |

Shared settings are sampled-softmax loss, 256 negatives, learning rate .001,
batch size 128, deterministic seed 32, and early stopping on Recall@10 with
patience 50. The maximum is 200 epochs for Amazon and 100 for MovieLens; the
best Recall@10 checkpoint is loaded for final ranking. The remaining RecTools
defaults are train minimum length 2, learnable inverse positional encoding,
causal attention, catalogue-uniform negative sampling, and no data-loader
workers.

## Evaluation constants

All methods use full-catalogue top-10 recommendation, filter previously viewed
items, and report RecTools 0.13.0 NDCG@10 with `divide_by_achievable=True`.
There are no sampled evaluation negatives.
