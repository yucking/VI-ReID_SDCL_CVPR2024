---
name: sdcl-research-gate
description: Mandatory research gate before any SDCL2 Stage-1, Stage-2, clustering, pseudo-label, loss, memory, model, sampler, evaluation, or training-strategy change. Use when Codex proposes or implements algorithmic changes, starts training, or evaluates whether an experiment is worth running.
---

# SDCL Research Gate

Before editing algorithm code or launching training, require a written gate decision.

## Required Inputs

- Current trusted baseline and evidence path.
- Current failure evidence from real logs, not file names.
- Exact scope: Stage-1 representation, clustering/pseudo-labels, Stage-2 cross-modal association, optimization, evaluation, or engineering.
- One falsifiable hypothesis.
- One primary independent variable.
- Fixed variables, including dataset, protocol, seed, cmlabel, training length, checkpoint source, and evaluation protocol.

## Hard Blocks

Do not allow implementation or training when any item is true:

- No experiment contract exists.
- Baseline is `UNKNOWN` and the change is not explicitly a diagnostic.
- The plan combines multiple mechanisms.
- The plan only adds a threshold, weight, schedule, post-processing branch, or unexplained extra loss.
- The plan changes the evaluation protocol without declaring it.
- The expected mechanism does not name an intermediate metric.
- A single seed is used to claim improvement.
- Results are compared only to best trial, not repeated-baseline variance.
- Failure experiments will not be registered.

## Gate Output

Return this structure:

```text
Baseline:
Evidence:
Problem class:
Why current methods failed:
Hypothesis:
Independent variable:
Fixed variables:
Expected intermediate metric:
Falsification result:
Artifact paths:
Stored gate decision path:
Required smoke tests:
Full-run plan:
Multi-seed plan:
Gate decision: PASS | FAIL
Reason:
```

If `FAIL`, do not modify algorithm code and do not start training.
