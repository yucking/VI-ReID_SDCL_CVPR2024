# SYSU Experiment Decisions

This file is a decision ledger, not a complete experiment table. Final judgments
should come from `log.txt`, `evaluate_sysu_v4_push_log.py`, and key grep lines.

## Target

- Indoor target: mAP/mINP >= 76.90/73.50
- All-search target: mAP/mINP >= 63.24/51.06
- Best uploaded reproduction: `logs/0620/log复现最优.txt`
  - Best epoch: 38
  - All-search mAP/mINP: 63.41/50.38
  - Indoor mAP/mINP: 76.58/73.12
- Remaining gap from best uploaded reproduction:
  - All-search mAP is already above target.
  - All-search mINP needs about +0.68.
  - Indoor mAP needs about +0.32.
  - Indoor mINP needs about +0.38.

## Reproduction Evidence

The three uploaded reproduction logs use the same nominal training recipe, but
land on very different stage2 trajectories:

| Log | Best epoch | All mAP/mINP | Indoor mAP/mINP | Decision |
| --- | ---: | ---: | ---: | --- |
| `log复现最优.txt` | 38 | 63.41/50.38 | 76.58/73.12 | Current reproduction anchor |
| `log复现测试3.txt` | 44 | 62.22/49.09 | 74.98/71.55 | Same recipe can fall to baseline-level |
| `log复现测试1.txt` | 47 | 61.26/48.23 | 73.39/69.79 | Same recipe can fail badly |

Conclusion: the main problem is not checkpoint selection. In the best log,
epoch 38 is best for all tracked metrics. The bigger issue is late-stage
trajectory variance and weight instability.

## Stop List

### XMB matched sampler

Do not continue this direction.

Evidence:
- `log_sysu_v19_fromW_on_seed2_tba0.03_xmb0.2_cmw0.0.txt`
- `[XMB-PLAN]` kept `rgb_batch=96 ir_batch=96 target_batch=96`, so the batch-size
  bug was fixed.
- Even after preserving batch size, epoch 35 dropped sharply after XMB started.

Why it is not worth continuing:
- Matched sampling changes the core stage2 training distribution.
- The failure remains after fixing the obvious batch-size issue.
- More weight sweeps would optimize around a harmful data distribution.

### XMB legacy queue / partial overlay

Do not continue this direction.

Evidence:
- `log_sysu_v19_fromW_on_seed2_tba0.03_xmb0.2_legacy_cmw0.0.txt`
- It avoided the matched-sampler crash, but best result stayed around
  all mAP 62.44, indoor 74.87/71.28.
- This is below the off seed2 baseline: all mAP 62.75, indoor 74.97/71.39.

Why it is not worth continuing:
- The queue signal is too weak or too noisy to improve the tail.
- It adds code complexity without beating the clean baseline.

### CM-triplet / `cm_w` / `cms`

Do not resume parameter sweeps here.

Evidence:
- v18 and later cm-triplet branches did not improve mINP.
- Multiple `cm_w` / `cms` combinations stayed in the same 74/71 region.

Why it is not worth continuing:
- Local cross-modal positives inside pseudo labels do not reliably fix the
  worst-rank tail.
- The loss adds another noisy objective on top of already noisy stage2 labels.

### TBA / BPT / SOT label topology rewrites

Do not continue label-flip topology rewrites unless there is a new diagnostic
showing label topology is the bottleneck.

Evidence:
- v15/v15.1 greedy BPT was too aggressive.
- v19 TBA was made conservative, but the v19 line still did not produce a gain.

Why it is not worth continuing:
- The current v4base CRA/softweight/tailtrim path is already close to target.
- Extra assignment/transport logic mostly injects label noise or training
  distribution drift.

### Shared sampler edits

Do not edit `RandomMultipleGallerySampler` for these experiments.

Why it is not worth continuing:
- It is shared by many training paths.
- Previous failures point to distribution shift, not to lack of matched samples.
- A shared sampler change has a large blast radius for a small expected gain.

## Current Clean Direction

Use clean v4base as the base path:

- `sdcl_sysu_v4base.py`
- `train_sysu_v4base.sh`
- `clustercontrast/trainers_source_softweight.py`

Removed dead branch:

- `sdcl_sysu_v19.py`
- `train_sysu_v19_from_warmstart.sh`
- `clustercontrast/models/cra_label_transfer.py`
- XMB/cm-triplet code inside `clustercontrast/trainers_source_softweight.py`

## Stopped / Opt-in: Late EMA

Rationale:
- Best uploaded reproduction peaks at epoch 38.
- Other runs with the same nominal recipe fall far below it.
- This suggests late-stage weight noise and trajectory variance are real
  bottlenecks.

Observed result:
- `logs/0619/log_sysu_v4base_lateema_seed2.txt`
- Best epoch: 37
- All-search mAP: 61.70
- Indoor mAP/mINP: 75.30/71.80
- Status: weak rerun, below the best reproduction and below target.

Implementation:
- `sdcl_sysu_v4base.py` now supports:
  - `--enable-late-ema`
  - `--late-ema-start`
  - `--late-ema-decay`
- `train_sysu_v4base.sh` keeps Late EMA off by default.
- Enable it only with `ENABLE_LATE_EMA=1`.
- The run saves:
  - `model_best.pth.tar`
  - `model_late_ema.pth.tar`

Clean default test:

```bash
cd /home/lhp/project/SDCL2
sed -i 's/\r$//' train_sysu_v4base.sh
export SEED=2
bash train_sysu_v4base.sh
python evaluate_sysu_v4_push_log.py /home/lhp/project/SDCL2/logs/sysu_v4base_clean_seed2/log.txt
```

EMA checkpoint evaluation:

```bash
ENABLE_LATE_EMA=1 SEED=2 bash train_sysu_v4base.sh
python test_sysu.py \
  --data-dir /home/lhp/project/DATASETS/SYSU-MM01 \
  --logs-dir /home/lhp/project/SDCL2/logs/sysu_v4base_lateema_seed2 \
  --checkpoint-name model_late_ema.pth.tar
```

Decision rule:
- Late EMA is not a default direction after the seed2 weak rerun.
- Keep it only as a cheap diagnostic/opt-in checkpoint smoother.
- Do not combine it with XMB/TBA/cm-triplet.

## Next Focus

The seed2 Late EMA log shows the run was already weak before late-stage smoothing:

- Epoch 29: all mAP 53.30, indoor 66.70/62.70
- Epoch 34: all mAP 61.00, indoor 74.40/70.90
- Best epoch 37: all mAP 61.70, indoor 75.30/71.80

Conclusion:
- Late smoothing cannot rescue a weak stage1 / early stage2 trajectory.
- The next useful direction should improve or select the early pseudo-label
  trajectory before stage2 settles.
- Candidate directions should be judged by epoch 29/34 diagnostics first, not
  by late epoch sweeps.

Do not spend more runs on:
- Late EMA decay sweeps.
- More stage2 auxiliary losses.
- More sampler/queue/topology branches.

## Stopped: TCR Reliability Weighting

Evidence:
- `logs/0619/log_sysu_v4base_tcr_seed2.txt`
- Epoch 29: all mAP 51.20, indoor 64.70/60.50.
- Epoch 34: all mAP 60.40, indoor 72.76/69.27.
- This is materially below the Late EMA weak rerun at epoch 34:
  all mAP 61.00, indoor 74.40/70.90.

Conclusion:
- Downweighting pseudo-labelled samples reduces effective early supervision.
- Do not sweep TCR history, minimum weight, power, or confidence components.
- TCR code and its training entrypoint were removed.

## Current Main Experiment: MSM

MSM means Multi-Subcenter Memory.

Goal:
- Preserve every accepted pseudo-labelled sample.
- Represent large shared pseudo identities with two feature-space prototypes
  instead of forcing pose, view, and modality modes into a single center.
- Improve tail retrieval and mINP without changing the sampler or hard-flipping
  RGB labels.

Mechanism:
- Stage2 builds the normal shared pseudo identity labels as before.
- Large identities are conservatively split into at most two feature-space
  subcenters.
- The memory loss treats every subcenter of the same pseudo identity as a
  positive through log-sum-exp multi-positive contrast, not as a negative.
- Small or unstable splits collapse back to a single center.

Implementation:
- `clustercontrast/models/multi_center_memory.py`
- `sdcl_sysu_v4base.py`
  - `--enable-msm`
  - `--msm-min-cluster-size`
  - `--msm-max-centers`
  - `--msm-iterations`
- `train_sysu_v4base.sh` enables MSM by default.
- Default log dir: `sysu_v4base_msm_seed${SEED}`.

Default run:

```bash
cd /home/lhp/project/SDCL2
sed -i 's/\r$//' train_sysu_v4base.sh
export SEED=2
bash train_sysu_v4base.sh
python evaluate_sysu_v4_push_log.py /home/lhp/project/SDCL2/logs/sysu_v4base_msm_seed2/log.txt
```

Early decision rule:
- Verify `[MSM]` logs: centers must exceed identities, while train IR/RGB counts
  remain close to the clean run.
- Epoch 34 must beat the weak Late EMA run: indoor 74.40/70.90.
- A meaningful signal is indoor mINP >= 72.0 by epoch 34.
- If MSM fails this gate, stop it rather than increasing the number of centers.
