# Sequential Capacity Probes

[![Tests](https://github.com/spotify-research/sequential-capacity-probes/actions/workflows/tests.yml/badge.svg)](https://github.com/spotify-research/sequential-capacity-probes/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Official code for **“Do Sequential Recommendation Benchmarks Really Require
Higher-Order Sequence Modelling?”**, accepted at the
[20th ACM Conference on Recommender Systems (RecSys 2026)](https://recsys.acm.org/recsys26/),
Marriott City Center, Minneapolis, Minnesota, USA, September 28–October 2,
2026.

**Official publication code from [Spotify Research](https://research.atspotify.com/).**

[Aleksandr V. Petrov](https://scholar.google.com/citations?user=Cw7DY8IAAAAJ),
[Praveen Chandar](https://scholar.google.com/citations?user=phLOBVYAAAAJ),
[Paul Bennett](https://scholar.google.com/citations?user=AIncPrIAAAAJ),
[Hugues Bouchard](https://scholar.google.com/scholar?q=author%3A%22Hugues+Bouchard%22),
and [Mounia Lalmas](https://scholar.google.com/citations?user=wAr9G5sAAAAJ).

This repository provides the paper implementations of MC, FMC, FMC+, Sequential
Rules (SeqRules), and PCTM, plus pinned launchers for SASRec+ and eSASRec from
the authors' original [eSASRec code](https://github.com/blondered/transformer_benchmark).

## Reported results

Full-catalogue NDCG@10 from Table 1:

| Data | MC | FMC | FMC+ | SAS+ | eSAS | SeqRules | PCTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beauty | .0492 | .0481 | .0531 | .0537 | .0524 | .0605 | .0635 |
| Sports | .0251 | .0244 | .0271 | .0315 | .0324 | .0371 | .0368 |
| Toys | .0548 | .0554 | .0588 | .0575 | .0533 | .0730 | .0738 |
| ML-1M | .1240 | .1124 | .1206 | .1662 | .1739 | .1505 | .1815 |
| ML-20M | .1050 | .0807 | .1034 | .1806 | .1969 | .1115 | .1431 |

## Install

The reference platform is Linux with Python 3.10.12. The complete experiment
requires Git, about 10 GiB of free disk, and an NVIDIA GPU with at least 24 GiB
of exclusive memory. MC, SeqRules, and PCTM can run on CPU.

```bash
git clone https://github.com/spotify-research/sequential-capacity-probes.git
cd sequential-capacity-probes
make setup
make dependency
```

`make setup` installs the complete transitive environment recorded in
`requirements-lock.txt`, then installs this checkout in editable mode. The
lock and build tooling are fixed for CPython 3.10.12, including the
platform-marked CUDA 12.6/Triton stack used on Linux; the direct dependency
contract remains visible in `pyproject.toml`.

`make dependency` checks out the original eSASRec repository at commit
`2b039927ceb4c7654131d7ac7c43ea124b49d240`. The dependency remains separate
under `.external/`; this repository does not vendor or modify it.

## Prepare the data

The original eSASRec release documents a manual route for obtaining and placing
its prepared benchmark archives. To support automated and independently
verifiable runs, we traced those benchmarks to their public upstream sources
and reproduced the published splits with:

```bash
make data-public
```

It downloads the three processed Amazon sequence files from the S3Rec project
and MovieLens 1M/20M from GroupLens, runs the pinned eSASRec builders, and
checks all raw and processed hashes in `data/manifest.json`. MovieLens ZIPs are
unpacked with Python's standard library, so no system `unzip` package is needed.

This path exactly reproduces the eSASRec data-preparation protocol rather than
implementing a compatible split independently: it invokes the dataset builders
from the pinned eSASRec checkout. All ten generated train/holdout files must
then match the SHA-256 hashes of the published eSASRec release byte for byte;
we have verified this equality across all five datasets.

Researchers who already have the five original eSASRec processed archives can
instead place `s3_beauty.zip`, `s3_sports.zip`, `s3_toys.zip`, `ml_1m.zip`, and
`ml_20m.zip` in `data/downloads/` and run `make data`. That path is deliberately
manual and verifies the archive and processed-split hashes before use.

## Reproduce Table 1 from the paper

Start from empty `results/`, `cache/`, and `logs/` directories:

```bash
make test
make run
make verify
```

The run writes one immutable JSON artifact per model/data pair, `table1.csv`,
`RESULTS.md`, and `verification.json` under `results/`. Existing model JSON is
never overwritten; use `--resume` to skip completed cells after interruption.
The final verifier does not accept a matching number alone: it checks model and
dataset identity, the complete frozen configuration, split hashes and row
counts, experiment-source digest, clean Git commit when available, package
versions, official-versus-independent metric parity, ranking protocol, and the
pinned neural dependency contract.

The suites can also be run separately in clean output directories:

```bash
make run-deterministic  # MC + SeqRules + PCTM, all five datasets, CPU
make run-factorized     # FMC + FMC+, all five datasets
make run-transformers   # SASRec+ + eSASRec, all five datasets
make run-sasrec         # SASRec+ only
make run-esasrec        # eSASRec only
```

For a focused run, use the common entry point:

```bash
.venv/bin/python scripts/run_all.py \
  --models pctm,seqrules --datasets beauty,ml1m --device cpu

.venv/bin/python scripts/run_all.py \
  --models sasrec_plus,esasrec --datasets all

.venv/bin/python scripts/run_all.py --resume
```

### Reproducibility expectations

MC, SeqRules, and PCTM are deterministic under the frozen data and evaluation
protocol and are required to reproduce the reported four-decimal values
exactly. A fresh H100 clean-room run reproduced all 15 values. FMC and FMC+
use fixed configurations and seeds and reproduced all ten values within the
declared absolute NDCG@10 tolerance of `0.001`. Their PyTorch full-catalogue
`topk` ordering for exactly equal scores is backend-dependent, so no stronger
cross-accelerator tie guarantee is claimed.

SASRec+ and eSASRec are trained by the pinned upstream eSASRec code rather than
reimplemented here. The wrapper freezes the upstream commit, configuration,
seed, software environment, and deterministic trainer mode. Nevertheless, our
clean-room experiments show small GPU- and run-dependent numerical variation
in this inherited neural training path. The paper therefore states that the
matched-protocol reproduction is within **0.0033 absolute NDCG@10** across all
ten SASRec+/eSASRec model-data pairs; it does not claim bit-exact neural
checkpoints. Our fresh H100 clean-room run had a maximum absolute difference of
`0.0027740228`, so all ten neural values were within that published bound.

Accordingly, `make verify` requires exact four-decimal parity for MC, SeqRules,
and PCTM and absolute NDCG@10 error at most `0.001` for FMC and FMC+. For
SASRec+ and eSASRec it allows `0.005`: a conservative cross-run and
cross-accelerator margin above both the paper's observed `0.0033` maximum and
our H100 clean-room maximum. This is a pragmatic reproducibility tolerance,
not a confidence interval. All data hashes, hyperparameters, seeds, candidate
sets, filtering rules, and evaluation invariants remain exact for every method.

The official metrics for MC, FMC, FMC+, SeqRules, and PCTM are not a local
reimplementation. They use RecTools 0.13.0 `Recall(k=10)` and
`NDCG(k=10, divide_by_achievable=True)` through `calc_metrics`, exactly as the
pinned eSASRec evaluator does. Each local model also computes the same metrics
independently from its ranked lists and stores them as `custom_metrics`; this
second implementation is a non-authoritative cross-check used to detect
ranking or evaluator regressions. SASRec+ and eSASRec are trained and evaluated
entirely by the pinned upstream runner, and this repository reads its result
CSV. Consequently, local-model evaluation needs RecTools but does not need to
import the eSASRec source tree; the eSASRec checkout remains required for data
preparation and neural-model execution.

### Published clean-room evidence

The complete result matrix, independent verification report, and all 35
per-model metadata records are available in
[`artifacts/table1-clean-room`](artifacts/table1-clean-room). Each record
contains the exact source commit and digest, split hashes, frozen parameters,
environment, official metrics, and independent cross-checks. The evidence
contains no datasets or model checkpoints.

## Code map

The five paper implementations requested for direct inspection are in
[`src/capacity_probes/models`](src/capacity_probes/models):

| File | Method | Training objective |
|---|---|---|
| `mc.py` | MC | empirical first-order conditional |
| `fmc.py` | FMC | sampled binary cross-entropy |
| `fmc_plus.py` | FMC+ | full-catalogue softmax cross-entropy |
| `seqrules.py` | Sequential Rules | weighted and optionally pruned rules |
| `pctm.py` | PCTM | smoothed multi-distance transition evidence |

Each module exposes an explicit `build_<model>()` and `recommend_<model>()`
API and owns that method's configuration, fitting, score construction,
candidate generation, seen-item filtering, background completion, and tie
policy. The small shared `ranking.py` module contains only deterministic top-k
and result-collection primitives. `core_runner.py` only handles data loading,
dispatch, official metric calculation, and artifact writing.

SASRec+ and eSASRec are dependency-backed by design. `neural.py` constructs a
single frozen upstream job from `configs/table1.json`; the dedicated launchers
are `scripts/run_sasrec.py` and `scripts/run_esasrec.py`.
Before training, the launcher validates that the pinned upstream seed, trainer,
epoch limit, early-stopping patience, validation metric, deterministic mode,
and all-users validation mask agree with the local publication contract.

See [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md) for every frozen model
setting, [docs/PROTOCOL.md](docs/PROTOCOL.md) for formulas and evaluation, and
[docs/DATA.md](docs/DATA.md) for provenance and licensing.

## Support and contributing

Use [GitHub issues](https://github.com/spotify-research/sequential-capacity-probes/issues)
for reproducibility questions and bug reports. Contributions are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security vulnerabilities through
the process in [SECURITY.md](SECURITY.md), not through a public issue.

## Citation

If you use this code or the results in your work, please cite the paper:

```bibtex
@inproceedings{petrov2026sequential,
  author    = {Aleksandr V. Petrov and Praveen Chandar and Paul Bennett and
               Hugues Bouchard and Mounia Lalmas},
  title     = {Do Sequential Recommendation Benchmarks Really Require
               Higher-Order Sequence Modelling?},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
  series    = {RecSys '26},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  location  = {Minneapolis, MN, USA},
  note      = {To appear}
}
```

Machine-readable citation metadata is also available in
[CITATION.cff](CITATION.cff). The BibTeX entry will be updated with the DOI and
page numbers when the ACM proceedings are published.

## License

Code in this repository is licensed under Apache-2.0. Dataset licenses remain
with their providers.
