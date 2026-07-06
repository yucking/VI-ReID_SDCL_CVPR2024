#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
DATA_DIR="${DATA_DIR:-/home/lhp/project/DATASETS/SYSU-MM01}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_v37_s1handoff_seed${SEED}_fullchain}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-66}"
CMLABEL="${CMLABEL:-36}"
STEP_SIZE="${STEP_SIZE:-23}"
STAGE1_SELECT_START="${STAGE1_SELECT_START:-28}"
STAGE1_BEST_SELECT_MODE="${STAGE1_BEST_SELECT_MODE:-tail}"
SAVE_STAGE1_CANDIDATES="${SAVE_STAGE1_CANDIDATES:-0}"

if [ "${SEED}" != "1" ]; then
    echo "Reference SYSU experiment is pinned to seed=1; got SEED=${SEED}." >&2
    exit 2
fi
if [ "${STAGE1_SELECT_START}" -ge "${CMLABEL}" ]; then
    echo "STAGE1_SELECT_START must be smaller than CMLABEL." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

extra_args=()
if [ "${SAVE_STAGE1_CANDIDATES}" = "1" ]; then
    extra_args+=(--stage1-save-candidates)
fi

echo "==> v37 S1 handoff: cmlabel=${CMLABEL} stage1_window=[${STAGE1_SELECT_START}, ${CMLABEL}) selector=${STAGE1_BEST_SELECT_MODE} step_size=${STEP_SIZE}"
echo "==> v37 full-chain only: Stage-1 is part of the experiment; do not use old Stage-2-only checkpoints."

python sdcl_sysu_v37_s1handoff.py \
    -b 96 \
    -j "${WORKERS}" \
    -a agw \
    -d sysu_all \
    --data-dir "${DATA_DIR}" \
    --iters 200 \
    --epochs "${EPOCHS}" \
    --step-size "${STEP_SIZE}" \
    --momentum 0.1 \
    --eps 0.6 \
    --cmlabel "${CMLABEL}" \
    --num-instances 16 \
    --best-select-mode full \
    --stage1-best-select-mode "${STAGE1_BEST_SELECT_MODE}" \
    --stage1-select-start "${STAGE1_SELECT_START}" \
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
    "${extra_args[@]}"
