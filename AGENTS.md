# SDCL2 Agent Rules

## Project Goal

This project should not keep producing v40, v41, and v42 by stacking small Stage-2 tweaks. The goal is to freeze a trusted baseline, diagnose the real bottleneck, and obtain explainable, reproducible, multi-seed improvements with a paper-grade problem, method, mechanism, and experiment loop.

## Required Skills

- Before any algorithm change or formal training, use `sdcl-research-gate` and `sdcl-experiment-design`.
- After PyTorch, clustering, pseudo-label, memory, trainer, checkpoint, or evaluator changes, use `sdcl-implementation-audit` and `sdcl-paper-reviewer`.
- When analyzing logs or updating result claims, use `sdcl-log-and-result-audit`.
- Use `wandb-experiment-tracking-local` only as offline-safe W&B guidance unless official W&B Skills are installed.

## Research Workflow

Every algorithm change must follow:

`problem diagnosis -> falsifiable hypothesis -> single-variable plan -> implementation audit -> smoke test -> full experiment -> multi-seed comparison -> ablation and mechanism validation`

## Forbidden

- Do not modify algorithm code without an experiment contract.
- Do not mix multiple new mechanisms in one experiment.
- Do not claim improvement from one seed.
- Do not report a best trial as a 10-trial average.
- Do not change SYSU or RegDB evaluation protocol without saying so.
- Do not hardcode version parameters inside training loops.
- Do not create endless version scripts by copying files without a contract.
- Do not skip syntax checks, static checks, and smoke tests after code changes.
- Do not declare monitoring successful until the monitor has been simulated or completed.
- Do not declare training complete before logs show completion and summary parsing succeeds.
- Do not record only positive results; failed runs go into the registry.
- Do not let the same agent be the only implementer and final reviewer.
- Do not start full SYSU-MM01 or RegDB training for infrastructure-only tasks.
- Do not edit `~/.codex/config.toml`, `codex_apps`, MCP transport, datasets, pretrained weights, or unrecoverable checkpoints unless explicitly requested.

## Done Definition For Algorithm Changes

An algorithm change is complete only when all are true:

- Experiment contract is complete.
- Research gate output is saved in the contract or adjacent artifact.
- Preflight output is saved and referenced by the registry row.
- Diff is reviewable.
- Syntax check passes.
- Static check passes.
- Unit tests pass.
- Single-batch forward/backward smoke test passes.
- Short training smoke test passes.
- Monitor simulation passes.
- Result parser passes.
- Independent reviewer completes.
- Multi-seed plan is explicit.
- Result is entered in `experiments/registry.csv`.

## Current Infrastructure

- Research docs: `docs/research/`.
- Experiment registry: `experiments/registry.csv`.
- Formal contracts: `experiments/contracts/`.
- Preflight gate: `python scripts/research_preflight.py --contract <file> --experiment-id <id> --log-dir <dir> --train-script <script>`.
- Setup validator: `python scripts/validate_skill_setup.py`.
