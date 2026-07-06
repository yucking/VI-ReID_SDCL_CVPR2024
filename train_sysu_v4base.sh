#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_v4base_seed${SEED}}"
WORKERS="${WORKERS:-8}"
ENABLE_RAS="${ENABLE_RAS:-0}"
RAS_ARGS=()

if [[ "${ENABLE_RAS}" == "1" ]]; then
    RAS_ARGS+=(--enable-ras)
fi

mkdir -p "${LOG_DIR}"

python sdcl_sysu_v4base.py \
    -b 96 \
    -j "${WORKERS}" \
    -a agw \
    -d sysu_all \
    --iters 200 \
    --epochs 60 \
    --momentum 0.1 \
    --eps 0.6 \
    --cmlabel 30 \
    --num-instances 16 \
    --best-select-mode full \
    --stage1-best-select-mode legacy \
    --trainer-backend source \
    --grad-accum-steps 1 \
    --seed "${SEED}" \
    --enable-stage2-softweight \
    --stage2-softweight-min 0.80 \
    --stage2-softweight-power 1.0 \
    --enable-stage2-tailtrim \
    --stage2-tailtrim-delay 3 \
    --stage2-tailtrim-warmup 4 \
    --stage2-tailtrim-pct 0.02 \
    --stage2-tailtrim-decay-after -1 \
    --logs-dir "${LOG_DIR}" \
    "${RAS_ARGS[@]}"

python evaluate_sysu_v4_push_log.py "${LOG_DIR}/log.txt"
