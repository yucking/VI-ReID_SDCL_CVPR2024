---
name: sdcl-experiment-design
description: Design SDCL2 paper-grade experiment plans, ablations, and mechanism validation. Use before formal SYSU/RegDB runs, when choosing the next Stage-1 or Stage-2 research direction, or when writing an experiment contract.
---

# SDCL Experiment Design

Prefer diagnosing Stage-1 representation quality, clustering quality, and pseudo-label reliability before more Stage-2 tuning.

## Required Plan Fields

- Research question.
- Baseline commit or `UNVERIFIED` with reason.
- Failure mode and log evidence.
- Falsifiable hypothesis.
- Primary independent variable.
- Fixed variables.
- Main metrics: all-search Rank-1/mAP/mINP and indoor Rank-1/mAP/mINP.
- Mechanism diagnostics: cluster counts, outliers, pseudo-label coverage, CRA confidence, cross-modal nearest-neighbor stability, loss ranges, checkpoint trajectory.
- Required ablations.
- Seed count and seed list.
- Success criteria and failure criteria.
- Compute budget and early-stop rules.
- Paper contribution potential.
- Strongest opposing explanation.
- How to disprove the method.

## Quality Rules

- Treat single-seed gains as diagnostic only.
- Compare against repeated baseline variance when available.
- Separate best epoch, final epoch, and single-metric highest epoch.
- Do not use stage2-only results as proof for full-chain claims.
- Do not propose v40 or a new version until the contract passes `sdcl-research-gate`.
