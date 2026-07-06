# Baseline Freeze

## Current Original SDCL Baseline

- Code path: `sdcl_sysu_v4base.py`, `train_sysu_v4base.sh`.
- Trainer: `clustercontrast/trainers_source_softweight.py`.
- Dataset: SYSU-MM01.
- Protocol: all-search and indoor-search, 10 gallery trials.
- Local commit: `UNVERIFIED` because `/home/lhp/project/SDCL2` is not currently a Git repository.
- Candidate GitHub project: `git@github.com:yucking/VI-ReID_SDCL_CVPR2024.git`.
- Remote HEAD verified on 2026-07-06: `2bb095050aba41cb252ca7d3a607283034f3cb23`.

## Current Most Trusted Reproduction

- Actual raw-log directory verified on 2026-07-06: `/mnt/data/lhp/SDCL2/logs-part1/sysu_train_ori_test2/`.
- Project symlink path: `logs-part1/sysu_train_ori_test2/`.
- Raw logs: `log.txt` and `log复现最优.txt`.
- Parser summary generated: `logs-part1/sysu_train_ori_test2/summary.md`.
- Parsed best epoch: 38.
- Parsed metrics:
  - all-search Rank-1/mAP/mINP: `65.55 / 63.41 / 50.38`.
  - indoor Rank-1/mAP/mINP: `71.12 / 76.58 / 73.12`.
- Final epoch 49 metrics: all-search `64.29 / 62.74 / 50.10`, indoor `70.81 / 76.11 / 72.67`.
- Checkpoint hashes:
  - `checkpoint.pth.tar`: `b1dc67e036fdc2c1bad5beb6715a0b93daffa4ee0b942b1f1ddccff3a91ea55e`
  - `model_best.pth.tar`: `0eb64daa69ca9eb993fdf722654c21eca6d3c1ecef73335813eb5f615ed6fb9b`
  - `20model_best.pth.tar`: `39dcea0bebf2f8de1583078c4462528796962fb0a3dc64db1af199f835c8fdcd`
  - `log.txt`: `57ff8ede3c345613421af8170ed4054114155e58e8b443b927615fc3992f9c7b`
- Status: raw-log verified, local Git commit still `UNVERIFIED`.

## Paper Targets

- all-search mAP/mINP: `63.24 / 51.06`.
- indoor mAP/mINP: `76.90 / 73.50`.
- Rank-1 targets: `UNVERIFIED` in current docs.

## Known Biases

- Several later runs are stage2-only and cannot prove full-chain gains.
- Some summaries compare against a missing reproduction-best raw log.
- Current server Python environment lacks PyTorch, so algorithm smoke tests need the `sdcl` training environment.

## Next Baseline Actions

- Run at least three clean v4base full-chain seeds with screen monitor.
- Register each baseline run in `experiments/registry.csv`.
- Recover Git metadata by cloning or reconnecting `git@github.com:yucking/VI-ReID_SDCL_CVPR2024.git` in a controlled way, without overwriting this working directory.
