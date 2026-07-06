#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
DATA_DIR="${DATA_DIR:-/home/lhp/project/DATASETS/SYSU-MM01}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_v9_gprd_seed${SEED}_fullchain}"
STAGE2_STATE="${STAGE2_STATE:-}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-60}"
GPRD_START="${GPRD_START:-34}"
EARLY_STOP="${EARLY_STOP:-1}"

if [ "${SEED}" != "1" ]; then
    echo "GPRD reference experiment is pinned to seed=1; got SEED=${SEED}." >&2
    exit 2
fi
if [ -n "${STAGE2_STATE}" ] && [ ! -f "${STAGE2_STATE}" ]; then
    echo "Missing complete Stage-2 state: ${STAGE2_STATE}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

extra_args=()
if [ -n "${STAGE2_STATE}" ]; then
    extra_args+=(--resume-stage2-state "${STAGE2_STATE}")
fi
if [ "${EARLY_STOP}" = "0" ]; then
    extra_args+=(--disable-early-stop)
fi

python sdcl_sysu_v9_gprd.py \
    -b 96 \
    -j "${WORKERS}" \
    -a agw \
    -d sysu_all \
    --data-dir "${DATA_DIR}" \
    --iters 200 \
    --epochs "${EPOCHS}" \
    --momentum 0.1 \
    --eps 0.6 \
    --num-instances 16 \
    --cmlabel 30 \
    --seed "${SEED}" \
    --gprd-start "${GPRD_START}" \
    --gprd-warmup 4 \
    --gprd-weight 0.05 \
    --gprd-topk 24 \
    --gprd-positive-count 4 \
    --gprd-hard-negative-count 12 \
    --gprd-gallery-per-camera 2048 \
    --gprd-chunk-size 1024 \
    --gprd-teacher-temperature 0.07 \
    --gprd-student-temperature 0.05 \
    --gprd-confidence-floor 0.20 \
    --logs-dir "${LOG_DIR}" \
    "${extra_args[@]}"
