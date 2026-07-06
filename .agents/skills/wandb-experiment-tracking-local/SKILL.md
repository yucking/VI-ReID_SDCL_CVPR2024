---
name: wandb-experiment-tracking-local
description: Local fallback for W&B experiment tracking guidance in SDCL2 when official W&B Skills cannot be installed. Use for offline-safe PyTorch experiment logging plans, hyperparameter/Git association, artifact/checkpoint policy, and multi-run comparison design without forcing online W&B upload.
---

# W&B Experiment Tracking Local Fallback

Use this only because the official W&B Skills install requires `npx`, which is unavailable in this server environment.

## Rules

- Do not require `wandb login`.
- Do not force online uploads in training code.
- Prefer optional `WANDB_MODE=offline` or disabled-by-default integration.
- Record config, seed, command, code state, checkpoint paths, summary paths, and environment.
- Treat W&B as experiment bookkeeping, not proof of research contribution.

## Recommended Metadata

- `experiment_id`, version, dataset, protocol, seed.
- Git commit or `UNVERIFIED` if the directory is not a Git repo.
- Baseline commit and log path.
- Full command and environment variables.
- All-search and indoor metrics.
- Artifacts: config, log, summary, checkpoints, monitor log.
- Decision: keep, revise, or reject.
