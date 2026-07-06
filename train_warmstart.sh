#!/usr/bin/env bash
set -euo pipefail

# Produce a reusable, fixed stage1 warm-start `W` (20model_best.pth.tar).
#   Run this 2-3 times with different SEED, then keep the run whose epoch29
#   indoor_mAP is highest as the fixed warm-start for the clean A/B protocol.
#
# Usage:
#   SEED=1 bash train_warmstart.sh
#   SEED=2 bash train_warmstart.sh
#   SEED=3 bash train_warmstart.sh
# The warm-start lives at:  ${LOG_DIR}/20model_best.pth.tar

export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_warmstart_seed${SEED}}"
SCRIPT="sdcl_sysu_v18.py"

python - "$LOG_DIR" "$SCRIPT" "$SEED" <<'PY'
import os
import subprocess
import sys

log_dir = sys.argv[1]
script = sys.argv[2]
seed = sys.argv[3]
os.makedirs(log_dir, exist_ok=True)

cmd = [
    sys.executable,
    script,
    "-b", "96",
    "-a", "agw",
    "-d", "sysu_all",
    "--iters", "200",
    "--epochs", "60",
    "--momentum", "0.1",
    "--eps", "0.6",
    "--cmlabel", "30",
    "--num-instances", "16",
    "--best-select-mode", "full",
    "--stage1-best-select-mode", "legacy",
    "--trainer-backend", "source",
    "--grad-accum-steps", "1",
    "--seed", seed,
    # stage1-only: train epochs [0, cmlabel) then exit, producing 20model_best.
    "--stage1-only",
    "--logs-dir", log_dir,
]

print("[RUN] " + " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd)
proc.wait()
sys.exit(proc.returncode if proc.returncode is not None else 0)
PY

echo "[WARMSTART] done. fixed warm-start checkpoint: ${LOG_DIR}/20model_best.pth.tar"
