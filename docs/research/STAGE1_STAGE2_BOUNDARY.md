# Stage-1 And Stage-2 Boundary

## Boundary

- `--cmlabel` defines the stage boundary; current default is `30`.
- Stage-1 is `epoch < cmlabel`.
- Stage-2 begins at `epoch == cmlabel`.
- In `sdcl_sysu_v39_ema_teacher.py`, the boundary is implemented in `main_worker_stage2()`:
  - `trainer.cmlabel = int(args.cmlabel)` at line 573.
  - `if (epoch == trainer.cmlabel)` at line 822.
  - `20model_best.pth.tar` is loaded at lines 823-825.

## Stage-1 Behavior

- Each epoch extracts RGB and IR features.
- DBSCAN separately clusters RGB and IR features.
- Separate `ClusterMemory` instances are built for IR and RGB.
- `Memory_wise_v3` stores instance-level features for neighbor structure.
- Stage-1 saves `20model_best.pth.tar` when `epoch < cmlabel` and the Stage-1 selector improves.
- v38 `ProtoBridge` affects Stage-1 only when `--enable-proto-bridge` and `epoch < cmlabel`.

## Stage-2 Behavior

- At `epoch == cmlabel`, the model reloads Stage-1 `20model_best.pth.tar`.
- RGB clustering `k1` changes from `args.k1` to `args.stage2_k1`.
- CRA/soft structure smoothing creates transferred labels.
- Shared memory is anchored to IR pseudo-label space with `num_cluster_ir` classes.
- Stage-2 softweight and tailtrim are optional flags.

## Rebuilt State

- Cluster labels and memories are rebuilt every epoch from extracted features.
- Stage-2 handoff in v4base/v39 loads model weights; optimizer and scheduler are not restored from Stage-1 unless using v35 full-state resume.
- v35 adds `stage2_handoff_full_state.pth.tar` to capture optimizer, scheduler, RNG, and best selector state.

## Historical Scope

- v16-v34 mostly changed Stage-2 loss, pseudo-label filtering, topology, or post-handoff logic.
- v35 changed Stage-2 state handoff.
- v37 changed Stage-1 handoff selection and is diagnostic, not a Stage-1 mechanism.
- v38 added a Stage-1 prototype bridge mechanism.
- v39 added an EMA teacher for clustering/CRA feature extraction.

## Suitable Stage-1 Research Locations

- Feature extraction before DBSCAN in `main_worker_stage2()`.
- DBSCAN inputs and cluster diagnostics.
- `generate_cluster_features()` and cluster center construction.
- Stage-1 memory and loss assignment before `epoch >= cmlabel`.
- `20model_best.pth.tar` selector and diagnostics.

## Implementation Risks

- Raw `.cuda()` usage makes DDP/local-rank behavior fragile.
- Outlier `-1` labels must not reach memory CE.
- Cluster ID order must match center order from sorted labels.
- A better Stage-1 checkpoint selector can diagnose, but it cannot create better Stage-1 representation.

## Stage-2 Label-space Invariant

Stage-2 shared memory is anchored to IR pseudo-label space. RGB cluster labels must be transferred into that space before shared-memory CE:

`pseudo_labels_rgb -> cluster_label_rgb_ir -> pseudo_labels_all -> shared_memory(num_cluster_ir)`

Do not create RGB-only shared identities unless the experiment contract explicitly makes that the independent variable and includes an ablation.

## Checkpoint Audit Rules

| path | handoff behavior | acceptance rule |
| --- | --- | --- |
| v4base/v39 default | loads only `20model_best.pth.tar` model state at `epoch == cmlabel` | Do not claim optimizer/scheduler/RNG continuity. |
| v35 full-state | saves/restores model, optimizer, scheduler, RNG, and best selector state | Use for fair full-state resume comparisons. |
| stage2-only | starts from a Stage-1 checkpoint but does not reproduce full-chain state | Treat as diagnostic unless equivalence is proven. |

Before comparing Stage-2 variants, record whether optimizer state, scheduler state, RNG state, and Stage-1 best selector were preserved.
