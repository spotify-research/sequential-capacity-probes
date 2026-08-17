# Table 1 clean-room evidence

This directory contains the inspectable outputs from the independent H100
clean-room run completed on 16 August 2026:

- `table1.csv`: full-precision NDCG@10 values for all 35 model/data pairs;
- `RESULTS.md`: the rendered result matrix and overall status;
- `verification.json`: every acceptance check and SHA-256 digest;
- `model-metadata/`: the 35 immutable per-run JSON records; and
- `independent-verification.log`: the independent hash and metric audit.

The run used a clean checkout at commit
`35d5e8f60c2c831c346c5b56ebcbe3d139f67327`, recorded in every model JSON,
with source-tree digest
`83b755a44f64cd61009da84c9f7e8d77869d34b62410cd978542890ee909322a`.
It passed the artifact-count, provenance, data-hash, environment, independent
metric, and Table 1 acceptance gates.

Between that run and the v1.0.0 publication tree, the model implementations,
training objectives, model hyperparameters, metric definitions, reported
targets, and acceptance thresholds did not change. The intervening release
work removed automatic access to the manual archive source, added input-hash
and network-timeout guards, removed an unused protocol field and console entry
point, clarified documentation, and adjusted README branding. The evidence is
therefore preserved under its actual generating commit rather than relabelled
as output produced by the later release commit.

Paths beginning with `/workspace/` in the metadata describe the isolated
container filesystem used by the clean-room run. No datasets, checkpoints, or
user histories are included.
