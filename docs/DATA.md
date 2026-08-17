# Dataset provenance

The paper uses the processed leave-one-out splits released with eSASRec. The
original release documents a manual route for obtaining and placing those
prepared archives. For automated clean-room runs, we traced the benchmark
inputs to public upstream sources and reproduced the same splits. Both
supported paths must produce the hashes in `data/manifest.json`.

## Public raw sources (recommended)

`python scripts/prepare_data.py --source public` downloads:

- Beauty, Sports and Outdoors, and Toys and Games sequences from the
  [S3Rec repository](https://github.com/RUCAIBox/CIKM2020-S3Rec), pinned at
  commit `2a81540ae18615d88ef88227b0c066e5b74781e5`;
- MovieLens 1M and 20M archives from
  [GroupLens](https://grouplens.org/datasets/movielens/).

Raw files are checked before use. The pinned eSASRec dataset builders then
produce only the leave-one-out splits required by Table 1. Generated splits are
copied to `data/processed/` and checked again. The wrapper uses Python's
standard library to unpack the hash-verified MovieLens ZIPs, avoiding an
undeclared operating-system `unzip` dependency.
This is the eSASRec preparation implementation itself, not an independently
rewritten approximation. The resulting ten train/holdout files have been
verified to match the SHA-256 hashes of the published eSASRec release byte for
byte across all five datasets.

The Amazon benchmark files are already filtered sequential datasets published
by S3Rec; they are not rebuilt from the complete Amazon review dumps. This is
necessary to match the benchmark population and item identifiers exactly.

## Exact released splits

If you already have the original eSASRec processed archives, preserve these
names in `data/downloads/`:

- `s3_beauty.zip`
- `s3_sports.zip`
- `s3_toys.zip`
- `ml_1m.zip`
- `ml_20m.zip`

Then run `python scripts/prepare_data.py --source release`. This path performs
no network download. It verifies each archive before extracting only
`statistics.csv`, `leave_one_out/train.csv`, and `leave_one_out/holdout.csv`;
macOS metadata and unrelated datasets are ignored.

Run `python scripts/prepare_data.py --source verify` at any time. Model runners
perform the same verification gate before reading a split.

## Licensing

This repository does not redistribute data. MovieLens and Amazon/S3Rec data are
subject to their providers' terms. Researchers are responsible for confirming
that their use complies with those terms.
