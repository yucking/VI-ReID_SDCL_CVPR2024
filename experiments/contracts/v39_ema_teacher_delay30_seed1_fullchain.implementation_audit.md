# sdcl-implementation-audit: v39_ema_teacher_delay30_seed1_fullchain

Files reviewed:
`train_sysu_v39_ema_teacher.sh`, `sdcl_sysu_v39_ema_teacher.py`, `logs/0706/log_sysu_v39_ema_teacher_seed1_fullchain.txt`, `/mnt/data/lhp/SDCL2/logs-part1/sysu_train_ori_test2/log.txt`.

Changed symbols:
No Python function, class, method, tensor path, loss, memory, evaluator, model, or DDP wrapper was changed. The shell defaults changed: `LOG_DIR` target and `EMA_TEACHER_START`.

High-risk paths:
`sdcl_sysu_v39_ema_teacher.py` lines using `cluster_model = teacher_model` after `epoch >= args.ema_teacher_start`. Risk is stale teacher features for DBSCAN/CRA pseudo-labels. This run delays that path until `cmlabel=30`.

Shape/device findings:
No tensor shape or device code was changed. The existing teacher model remains `nn.DataParallel(...).cuda()` and is loaded from student state dict at creation and at handoff.

Pseudo-label/memory findings:
Before the fix, v39 log shows epoch 1 clustering still from EMA teacher and outliers IR/RGB 2456/2829. Trusted reproduction has epoch 1 IR/RGB outliers 1368/1693. The fix makes Stage-1 use student clustering by default, preserving the stable pseudo-label source until handoff.

Checkpoint/eval findings:
Evaluation protocol and checkpoint names are unchanged. Log directory default changes to a new delay30 directory to avoid overwriting the failed v39 run.

Tests run:
`bash -n train_sysu_v39_ema_teacher.sh`; `bash -n scripts/run_experiment_screen.sh`; `bash -n scripts/watch_experiment_screen.sh`; `conda activate sdcl && python -m py_compile sdcl_sysu_v39_ema_teacher.py scripts/parse_sysu_log.py`; `python scripts/validate_skill_setup.py`; `python -m pytest tests/test_research_preflight.py -q`; `python scripts/research_preflight.py ...`.

Required fixes:
None before launching the diagnostic run. Independent final approval is still required before any keep claim.

Audit decision: PASS for diagnostic launch only.
