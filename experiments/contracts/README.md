# Experiment Contracts

Every formal SDCL2 experiment must have one contract file in this directory before code changes or training start.

Required process:

1. Copy `docs/research/EXPERIMENT_CONTRACT_TEMPLATE.md`.
2. Fill every field; no required field may remain `UNVERIFIED`.
3. Register the `experiment_id` in `experiments/registry.csv`.
4. Run `python scripts/research_preflight.py --contract experiments/contracts/<id>.md --experiment-id <id> --log-dir <log_dir> --train-script <train_script>`.
5. Only start training after preflight prints `PASS`.

After a run:

1. Verify monitor completion and readable `monitor.log`.
2. Run `python scripts/parse_sysu_log.py <log_file> > <summary.md>`.
3. Verify `checkpoint.pth.tar`, `model_best.pth.tar`, and `20model_best.pth.tar`.
4. Update `experiments/registry.csv` with metrics, summary, monitor, checkpoint paths, evidence grade, and decision.
5. Record negative results as `reject` instead of omitting them.
