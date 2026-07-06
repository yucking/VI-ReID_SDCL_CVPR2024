# SDCL2 Codebase Map

This map is based on files present on 2026-07-06 in `/home/lhp/project/SDCL2`.

## Training Entrypoints

- `sdcl_sysu_v4base.py:main()` parses arguments and calls `main_worker_stage2(args, log_name)`.
- `sdcl_sysu_v35_fullstate.py:main()` adds full-state handoff and resume logic.
- `sdcl_sysu_v36_cmbs.py:main()` and `sdcl_sysu_v39_ema_teacher.py:main()` are later variants.
- Shell launchers include `train_sysu_v4base.sh`, `train_sysu_v35_fullstate.sh`, `train_sysu_v36_cmbs.sh`, `train_sysu_v39_ema_teacher.sh`, and archived `main_try/train_sysu_v37_s1handoff.sh`, `main_try/train_sysu_v38_proto_bridge.sh`.

## Stage-1 And Stage-2 Loop

- Main loop: `sdcl_sysu_v39_ema_teacher.py:819` iterates `for epoch in range(args.epochs)`.
- Stage boundary: `trainer.cmlabel = int(args.cmlabel)` at `sdcl_sysu_v39_ema_teacher.py:573`; default parser value is `--cmlabel 30`.
- Stage-2 handoff: `sdcl_sysu_v39_ema_teacher.py:822-825` loads `20model_best.pth.tar` when `epoch == trainer.cmlabel`.
- Stage-2 RGB k1 changes at `sdcl_sysu_v39_ema_teacher.py:899-904`.
- Checkpoint selection writes `checkpoint.pth.tar`, `model_best.pth.tar`, and Stage-1 `20model_best.pth.tar` around `sdcl_sysu_v39_ema_teacher.py:1409-1468`.

## Model And Trainer

- Model factory: `clustercontrast/model_vit_cmrefine/make_model.py` is imported as `make_model`.
- Trainer class: `clustercontrast/trainers_source_softweight.py:63` defines `ClusterContrastTrainer_SDCL`.
- Training step: `ClusterContrastTrainer_SDCL.train()` starts at `clustercontrast/trainers_source_softweight.py:94`.
- Data parsing: `_parse_data_rgb`, `_parse_data_ir`, and `_forward` are at `clustercontrast/trainers_source_softweight.py:424`, `428`, and `434`.

## Clustering And Pseudo-labels

- DBSCAN is constructed in `sdcl_sysu_v39_ema_teacher.py:843-845`.
- Feature extraction for clustering uses `clustercontrast/evaluators.py:29 extract_features`.
- Jaccard distance uses `clustercontrast/utils/faiss_rerank.py:compute_jaccard_distance`, called at `sdcl_sysu_v39_ema_teacher.py:897` and `904`.
- Pseudo labels are `pseudo_labels_ir` and `pseudo_labels_rgb` from DBSCAN at `sdcl_sysu_v39_ema_teacher.py:898` and `905`.

## Memory

- Cluster memory: `clustercontrast/models/cm.py:108 ClusterMemory`.
- Instance memory: `clustercontrast/models/cm.py:208 Memory` and `Memory_wise_v3` later in the same file.
- Memory updates happen in custom autograd functions `CM.backward()` and `EM.backward()` at `clustercontrast/models/cm.py:21` and `190`.
- Stage-1 separate IR/RGB memories are assigned at `sdcl_sysu_v39_ema_teacher.py:915-924`.
- Stage-2 shared memory uses IR label space at `sdcl_sysu_v39_ema_teacher.py:1196-1211`.

## CNL, CRA, And Cross-modal Association

- CNL and CRA logic is embedded in `sdcl_sysu_v*.py` and `clustercontrast/trainers_source_softweight.py` rather than isolated in one module.
- Cross-neighbor losses are in `ClusterContrastTrainer_SDCL.train()` around `clustercontrast/trainers_source_softweight.py:217-350`.
- v38 prototype bridge logic is active only when `--enable-proto-bridge` and `epoch < cmlabel`, around `sdcl_sysu_v39_ema_teacher.py:1223-1263`.
- v39 EMA teacher uses a teacher model for clustering/CRA feature extraction at `sdcl_sysu_v39_ema_teacher.py:558-569` and `831-839`.

## Evaluation

- SYSU helper functions are in `test_sysu.py`: `process_query_sysu`, `process_gallery_sysu`, `extract_query_feat`, `extract_gall_feat`, `eval_sysu`.
- Training-loop SYSU evaluation prints all-search averages and indoor averages in `sdcl_sysu_v39_ema_teacher.py:1350-1402`.
- RegDB evaluation helpers are in `test_regdb.py`: `process_test_regdb`, `eval_regdb`, `extract_query_feat`, `extract_gall_feat`.
- `scripts/parse_sysu_log.py` parses SYSU logs and compares against the hardcoded reproduction best and paper targets.

## Monitoring And Logs

- Screen launcher: `scripts/run_experiment_screen.sh`.
- Screen monitor: `scripts/watch_experiment_screen.sh`.
- SYSU summary parser: `scripts/parse_sysu_log.py`.
- Current logs are under symlink `logs -> /mnt/data/lhp/SDCL2/logs`.
- v39 running log: `logs/0706/log_sysu_v39_ema_teacher_seed1_fullchain.txt`.

## Key Tensor Shapes

- Backbone feature dimension used by memory is 768.
- Clustering concatenates normal and shallow/deep paired features to 1536 for distance, then uses 768 original features for memory centers.
- `features_rgb` and `features_ir` are ordered by `sorted(dataset_*.train)`.
- Memory labels are pseudo-label cluster IDs; outliers use `-1` and must be filtered before memory CE.

## Configuration Sources

- CLI parser in each `sdcl_sysu_v*.py`.
- `vit_base_ics_288.yml`.
- Shell environment variables in `train_sysu_v*.sh`.
- Runtime `--logs-dir` controls checkpoint and `log.txt` location.

## Known Hardcoding And Risks

- Many paths and constants are embedded in training files.
- New code often uses raw `.cuda()`, which is risky for DDP/local-rank portability.
- Evaluation protocol is duplicated in training files and `test_sysu.py`.
- The project copy is not a Git repo in this server state, so commit-based reproducibility is currently `UNVERIFIED`.

## Audit Addendum: CNL/CRA Dataflow

- Stage-2 remaps RGB pseudo-labels into IR pseudo-label space before shared memory training.
- Operational path: `pseudo_labels_rgb -> cluster_label_rgb_ir -> pseudo_labels_all -> shared_memory(num_cluster_ir)`.
- `clustercontrast/trainers_source_softweight.py:217-350` alternates cross-neighbor losses by epoch parity:
  - even epochs compute RGB-to-IR neighbor structure;
  - odd epochs compute IR-to-RGB neighbor structure;
  - modality-internal RGB/RGB and IR/IR losses are also logged.
- Failure cases to audit: RGB labels left in RGB cluster ID space; `-1` labels entering CE; CRA/tailtrim changing sample counts without matching weights; Stage-2 logic active before `cmlabel`.

## Audit Addendum: Clustering And Memory Invariants

- `generate_cluster_features()` skips `-1` and sorts cluster IDs before stacking centers.
- DBSCAN labels are expected to be non-negative IDs suitable for CE after outlier removal.
- Audit empty clusters, non-contiguous labels, center ordering, and whether target IDs are `< num_cluster_*`.
- `ClusterMemory.forward()` requires every target in `[0, num_samples)`.
- `Memory_wise_v3` depends on instance index order; trainer maps filenames through `nameMap_ir` and `nameMap_rgb`.
- Feature order must match `sorted(dataset_rgb.train)` and `sorted(dataset_ir.train)`.
- Memory features must be normalized after initialization and momentum update.

## Audit Addendum: SYSU And RegDB Protocol

- SYSU all-search and indoor-search must be reported separately.
- Each SYSU mode averages 10 gallery trials and prints `All Average`.
- Keep Rank-1, mAP, and mINP separate; do not compare a single trial to the 10-trial average.
- RegDB evaluation is in `test_regdb.py`, outside the SYSU training loop.
- RegDB direction, visible-to-thermal or thermal-to-visible, and trial ID must be recorded.

## Audit Addendum: Device/DDP Contract

- Current training files use `nn.DataParallel`, not true DDP.
- Existing code contains many raw `.cuda()` calls; new code should prefer the owning module/input device.
- Any DDP migration must audit rank-safe feature extraction, distributed sampler behavior, all-rank pseudo-label consistency, and memory synchronization.
- New buffers must be registered on modules or explicitly moved with the owning module.
