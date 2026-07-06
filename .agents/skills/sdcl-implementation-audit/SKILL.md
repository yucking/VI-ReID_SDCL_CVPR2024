---
name: sdcl-implementation-audit
description: Independent implementation audit after SDCL2 PyTorch, DDP, memory, clustering, pseudo-label, loss, checkpoint, trainer, or evaluator edits. Use before accepting any algorithmic code change or before starting a full experiment.
---

# SDCL Implementation Audit

Audit the implementation as a silent-bug review, not just syntax review. The implementer cannot be the only final approver.

## Required Checks

- Tensor shapes at every new loss, memory, label, and feature path.
- Device placement and local-rank behavior; flag raw `.cuda()` in new code.
- DataParallel/DDP wrapper access and whether parameters participate in backward.
- Optimizer coverage for all new trainable parameters.
- Whether `detach`, `.data`, or `torch.no_grad()` cuts intended gradients.
- Memory update direction and feature normalization.
- RGB to IR and IR to RGB label-map direction.
- Outlier `-1` handling before cross entropy, memory indexing, and masking.
- Cluster center ordering versus pseudo-label IDs.
- Instance feature order versus dataset sorted order and `nameMap_*`.
- Scheduler, optimizer, RNG, and checkpoint restore behavior.
- Whether training loop overwrites `args` or hardcodes version parameters.
- Eval protocol unchanged for SYSU all-search, indoor-search, RegDB.
- NaN, Inf, empty cluster, no positive sample, and all-filtered sample boundaries.
- Single-GPU and multi-GPU behavior differences.

## Output

Report:

```text
Files reviewed:
Changed symbols:
High-risk paths:
Shape/device findings:
Pseudo-label/memory findings:
Checkpoint/eval findings:
Tests run:
Required fixes:
Audit decision: PASS | FAIL
```

If any high-risk path lacks a targeted smoke test, return `FAIL`.
