# SDCL2 Research Workflow

## 1. Baseline Audit

- Identify the trusted baseline log, command, checkpoint, seed, dataset, and protocol.
- Mark baseline commit as `UNVERIFIED` if Git is unavailable.
- Record repeated baseline variance if available.

## 2. Failure Diagnosis

- Use `sdcl-log-and-result-audit`.
- Cite raw log evidence.
- Separate Stage-1 weakness, clustering/pseudo-label quality, Stage-2 association, optimization, and evaluation issues.

## 3. Hypothesis

- State one falsifiable hypothesis.
- Name the mechanism and the intermediate metric expected to move before final metrics.

## 4. Experiment Contract

- Create one file under `experiments/contracts/`.
- Fill every required field in `EXPERIMENT_CONTRACT_TEMPLATE.md`.
- Register the planned experiment in `experiments/registry.csv`.

## 5. Implementation

- Pass `scripts/research_preflight.py`.
- Modify only files named in the contract.
- Keep one primary independent variable.

## 6. Independent Review

- Use `sdcl-implementation-audit`.
- Use `sdcl-paper-reviewer`.
- Do not let the implementer be the only final reviewer.

## 7. Smoke Test

- Syntax check changed scripts.
- Static check targeted files.
- Run unit tests that do not require GPU.
- Run single-batch forward/backward when algorithm code changes.
- Run monitor simulation before relying on monitoring.

## 8. Full Run

- Use `scripts/run_experiment_screen.sh` and `scripts/watch_experiment_screen.sh`.
- Store command, log path, monitor path, summary path, and checkpoints.
- Do not overwrite existing log directories.

## 9. Multi-seed

- Treat seed-1 as diagnostic.
- Compare at least three seeds for promising changes.
- Compare to repeated baseline natural fluctuation.

## 10. Ablation

- Remove each new mechanism component.
- Keep protocol and seed list fixed.
- Report negative ablations.

## 11. Mechanism Validation

- Inspect cluster counts, outliers, coverage, CRA confidence, bridge/teacher diagnostics, loss scale, and Stage-1/Stage-2 trajectories.
- Validate that the claimed mechanism moved before final metrics improved.

## 12. Paper-level Decision

- Use `keep`, `revise`, or `reject`.
- Keep only mechanisms with stable gains and interpretable diagnostics.
- Reject changes that only tune selection, thresholds, or schedules without mechanism evidence.
