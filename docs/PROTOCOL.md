# Frozen evaluation protocol

## Data boundary

For each benchmark, `train.csv` is the complete model-fitting input and
`holdout.csv` is the published final evaluation split. Hyperparameter choices
were made previously on an inner leave-one-out split of `train.csv`; the frozen
choices are recorded in `configs/table1.json`. No final labels are used during
fitting, configuration selection, candidate construction, or tie-breaking.

Rows are ordered stably by `(user_id, datetime, source row order)`. Candidates
outside the training catalogue and final users absent from training are handled
as in eSASRec. All previously viewed items are filtered before full-catalogue
top-10 ranking.

## Models

MC is the unsmoothed empirical first-order conditional

$$
P(a \mid b) = \frac{N_{ba}}{\sum_i N_{bi}},
$$

where $b$ is the last training item.

FMC assigns

$$
\mathrm{score}(a,b) = \langle \mathbf{s}_b, \mathbf{t}_a \rangle
$$

using unshared source and target embedding
tables and only the last item. FMC uses one uniformly sampled item outside the
user's complete training history per positive and binary cross-entropy. FMC+
uses the same architecture and the full-catalogue softmax cross-entropy. Both
form positive transitions from each user's most recent 50 events on Amazon or
200 events on MovieLens and are refitted for the frozen epoch count using seed
`20270710`.

For PCTM, distance-weighted causal counts are

$$
C_{ba} = \sum_{d=1}^{W} \gamma_d N^{(d)}_{ba},
\qquad
\widehat{P}(a \mid b)
= \frac{C_{ba} + \tau / |\mathcal{I}|}{C_{b\cdot} + \tau}.
$$

For a history ordered newest-first, its final score is

$$
S(a \mid h) = \sum_j w_j \log \widehat{P}(a \mid h_j) + \lambda \log p_{\mathrm{pop}}(a).
$$

The implementation stores only
$\log(1 + C_{ba}|\mathcal{I}|/\tau)$. The omitted term
$\log(\tau/|\mathcal{I}|) - \log(C_{b\cdot} + \tau)$ is constant over
candidate $a$ for each source and therefore cannot change a rank. Zero-count
candidates receive the uniform-prior background exactly.

Sequential Rules accumulates a directed item rule at each distance $d$ up to
the frozen window $W$,

$$
R_{ba} = \sum_{d=1}^{W} g(d) N^{(d)}_{ba}.
$$

Depending on the frozen dataset configuration, source rows are optionally
multiplied by min-max-normalized item IDF and then pruned to their strongest
$K$ targets. The prediction score combines rules from the newest $L$ history
items as

$$
S(a \mid h) = \sum_j v(j) R_{h_j a}.
$$

Rule and history weighting, window, IDF, pruning, and history length are all
frozen in `configs/table1.json`.

SAS+ and eSAS use the pinned upstream eSASRec configurations. The wrapper
selects vanilla SASRec for SAS+ and LiGR for eSAS, fixes the dataset, and uses
64 factors for the three Amazon runs, matching the reported configurations.

## Metrics and determinism

Official values come from RecTools 0.13.0 `NDCG(k=10,
divide_by_achievable=True)` and `Recall(k=10)`. Each non-neural implementation
passes its recommendation table to `calc_metrics`, exactly as the pinned
eSASRec evaluator does; the official metric is therefore not reimplemented in
this repository. Each non-neural implementation also computes the
leave-one-out metrics independently from its ranked lists and records both
values. That local implementation is a non-authoritative audit designed to
catch ranking and evaluator integration errors, and the tests enforce formula
parity. SASRec+ and eSASRec use the pinned upstream evaluation end to end and
the wrapper reads their result CSV.

PCTM, MC, and SeqRules are deterministic. PCTM equal scores use ascending
SHA-256 digest of the UTF-8 external item ID, then the external ID string. MC
and SeqRules preserve their reported runs' ascending-external-item-ID tie rule.
MC, PCTM, and SeqRules are deterministic and must match the four-decimal table
exactly. Factorized and neural runs pin their seeds, software, and training
configuration, but GPU kernels, exact-score ties, or different accelerators can
still introduce small numerical variation. FMC and FMC+ must have absolute
NDCG@10 error at most `0.001` from the reported value. The paper observed a
maximum `0.0033` reproduction gap across the ten inherited SASRec+/eSASRec
model-data pairs. Their automated acceptance tolerance is conservatively set
to `0.005` to allow additional cross-run and cross-accelerator variation; this
margin is pragmatic rather than a statistical confidence interval. These
tolerances cover numerical variation only; all data, configuration, epoch,
seed, ranking, and filtering invariants remain exact.
FMC and FMC+ use PyTorch full-catalogue `topk`; ordering among exactly equal
scores is backend-dependent and no stronger tie guarantee is claimed.

## Artifact contract

Each result JSON includes the selected configuration, official and independent
metrics, exact input hashes and row counts, package versions, and runtime
platform. `verification.json` hashes all 35 result JSON files. A complete run
never replaces an existing result file.

Verification rejects an artifact unless its model/data identity, frozen
configuration, input hashes, experiment-source digest, Git commit and
cleanliness when available, package versions, metric cross-check, and ranking
protocol all agree with the release contract. The source digest covers every
Python implementation/runner file plus the table configuration, data manifest,
package metadata, complete dependency lock, and experiment Makefile. This also
makes provenance verifiable from a source archive that does not contain `.git`
metadata.
Neural artifacts additionally validate the resolved upstream model parameters,
training contract, and pinned eSASRec commit.
