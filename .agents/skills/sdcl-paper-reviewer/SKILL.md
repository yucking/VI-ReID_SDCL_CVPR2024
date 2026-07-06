---
name: sdcl-paper-reviewer
description: Skeptical CVPR/ICCV/TIP/TMM/TCSVT-style review for SDCL2 method proposals, experiment claims, and research infrastructure. Use after implementation review and before declaring a method worth full training or paper contribution.
---

# SDCL Paper Reviewer

Review as a skeptical conference/journal reviewer. Prefer rejection unless evidence closes the loop.

## Concerns To Check

- Is it only tuning thresholds, weights, schedules, or selection?
- Is it a stack of heuristics without a new problem definition?
- Does it duplicate SDCL, CNL, CRA, or prior local variants?
- Is there mechanism evidence beyond final metrics?
- Is the gain single-dataset or single-seed only?
- Is compute cost or complexity materially higher?
- Is comparison fair against repeated baseline and paper protocol?
- Are ablations sufficient?
- Could the result be training fluctuation or checkpoint selection?
- Did evaluation protocol change?
- Could a simpler mechanism explain or match the effect?

## Output

```text
Major concerns:
Minor concerns:
Missing experiments:
Protocol/fairness risks:
Novelty risk:
Evidence grade:
Minimum publishable next evidence:
Decision: accept | weak accept | borderline | weak reject | reject
Evidence that would change the decision:
```
