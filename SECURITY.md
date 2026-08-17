# Security Policy

Spotify takes security and responsible disclosure seriously.

## Reporting a vulnerability

Do not report a suspected vulnerability through a public GitHub issue. Submit
it privately through Spotify's
[HackerOne bug-bounty programme](https://hackerone.com/spotify).

Include enough information to reproduce and assess the issue, but do not expose
sensitive data, credentials, or active exploits publicly. Spotify will review
the report and coordinate disclosure as appropriate.

## Research-prototype trust boundary

This repository is a research prototype intended to reproduce the experiments
described in the accompanying paper. Run it in an isolated environment and use
only the documented public datasets, pinned upstream source, and checkpoints
created locally by the same run. Do not load checkpoints or other serialized
artifacts from untrusted sources.

The frozen reproduction environment retains PyTorch 2.7.1 and PyTorch Lightning
2.5.6 because changing the numerical stack would change the environment used to
validate the reported results. The repository does not call `torch.jit.script`,
`torch.lstm_cell`, or `torch.nn.utils.rnn.unpack_sequence`; its Lightning path
loads only a checkpoint produced locally during the current trusted run. These
constraints are part of the supported security boundary, not a claim that the
pinned third-party versions are suitable for processing untrusted inputs.
