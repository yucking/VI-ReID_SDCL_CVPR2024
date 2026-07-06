# sdcl-research-gate: v39_ema_teacher_delay30_seed1_fullchain

Baseline:
Trusted reproduction anchor `/mnt/data/lhp/SDCL2/logs-part1/sysu_train_ori_test2/summary.md`: best epoch 38, all 65.55/63.41/50.38, indoor 71.12/76.58/73.12.

Evidence:
`logs/0706/log_sysu_v39_ema_teacher_seed1_fullchain.txt:107` enables EMA teacher with `start_epoch=0`; `:111` uses `clustering_source=ema_teacher` at epoch 0. At epoch 1, v39 has IR/RGB outliers 2456/2829 at `:376-377`, while the trusted reproduction log has 1368/1693 at `/mnt/data/lhp/SDCL2/logs-part1/sysu_train_ori_test2/log.txt:327-328`. v39 summary reports best/final epoch 22 all 12.31/13.22/6.29 and indoor 14.08/21.49/18.01.

Problem class:
Stage-1 representation and clustering/pseudo-label engineering bug.

Why current methods failed:
The EMA teacher is used for Stage-1 clustering from epoch 0 with decay 0.999. The teacher remains close to the initial model while the student changes, so Stage-1 DBSCAN uses stale features and outliers stay high instead of dropping after epoch 1.

Hypothesis:
Falsifiable: if stale EMA teacher clustering caused v39 collapse, delaying EMA teacher use until `cmlabel=30` should restore Stage-1 epoch 1-7 outlier and mAP trajectory near v38/trusted reproduction.

Independent variable:
One primary variable: default `EMA_TEACHER_START` in `train_sysu_v39_ema_teacher.sh` changes from 0 to 30.

Fixed variables:
SYSU-MM01 full-chain all-search plus indoor protocol, seed 1 diagnostic, seeds 1/2/3 planned, cmlabel 30, epochs 60, iters 200, batch 96, num-instances 16, eps 0.6, stage2 softweight and tailtrim unchanged, evaluation protocol unchanged.

Expected intermediate metric:
Stage-1 epoch 1-7 IR/RGB outliers should fall toward baseline/v38 ranges, and all-search mAP should exceed 28 by epoch 2 and 40 by epoch 7.

Falsification result:
Reject if epoch 2 all-search mAP remains below 25, epoch 7 all-search mAP remains below 38, or epoch 7 RGB/IR outliers remain above 1500/1000 without loss/OOM errors.

Artifact paths:
Contract `experiments/contracts/v39_ema_teacher_delay30_seed1_fullchain.md`; log dir `/home/lhp/project/SDCL2/logs/0706/sysu_v39_ema_teacher_delay30_seed1_fullchain`; raw log `/home/lhp/project/SDCL2/logs/0706/log_sysu_v39_ema_teacher_delay30_seed1_fullchain.txt`; summary `/home/lhp/project/SDCL2/logs/0706/sysu_v39_ema_teacher_delay30_seed1_fullchain/summary.md`.

Stored gate decision path:
`experiments/contracts/v39_ema_teacher_delay30_seed1_fullchain.gate.md`

Required smoke tests:
`bash -n train_sysu_v39_ema_teacher.sh`; `python -m py_compile sdcl_sysu_v39_ema_teacher.py scripts/parse_sysu_log.py`; `python scripts/research_preflight.py ...`; no formal training before preflight passes.

Full-run plan:
Launch by `scripts/run_experiment_screen.sh` and `scripts/watch_experiment_screen.sh` after code review and smoke tests pass.

Multi-seed plan:
Treat seed 1 as diagnostic only. If Stage-1 is restored and final metrics are competitive, repeat seeds 1, 2, and 3 against the trusted reproduction baseline variance.

Gate decision: PASS

Reason:
This is a single-variable bugfix for a logged Stage-1 collapse, not a new stacked mechanism or a result claim.
