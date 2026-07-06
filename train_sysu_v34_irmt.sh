#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
DATA_DIR="${DATA_DIR:-/home/lhp/project/DATASETS/SYSU-MM01}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_v34_irmt_seed${SEED}_stage2only}"
STAGE2_INIT="${STAGE2_INIT:-}"
STAGE2_ONLY="${STAGE2_ONLY:-1}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-50}"
EARLY_STOP="${EARLY_STOP:-0}"

IRMT_MERGE_FRAC="${IRMT_MERGE_FRAC:-0.90}"
IRMT_RECIPROCAL_TOPK="${IRMT_RECIPROCAL_TOPK:-5}"
IRMT_MIN_SIM="${IRMT_MIN_SIM:-0.35}"
IRMT_MARGIN_FLOOR="${IRMT_MARGIN_FLOOR:-0.02}"
IRMT_FORCE_SIM="${IRMT_FORCE_SIM:-0.75}"
IRMT_MIN_CRA_PURITY="${IRMT_MIN_CRA_PURITY:-0.45}"

if [ "${SEED}" != "1" ]; then
    echo "Reference SYSU experiment is pinned to seed=1; got SEED=${SEED}." >&2
    exit 2
fi
if [ "${STAGE2_ONLY}" = "1" ] && [ -z "${STAGE2_INIT}" ]; then
    echo "STAGE2_ONLY=1 requires STAGE2_INIT=/path/to/20model_best.pth.tar" >&2
    exit 2
fi
if [ -n "${STAGE2_INIT}" ] && [ ! -f "${STAGE2_INIT}" ]; then
    echo "Missing model-only Stage-2 checkpoint: ${STAGE2_INIT}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

extra_args=()
if [ "${STAGE2_ONLY}" = "1" ]; then
    extra_args+=(--stage2-only --stage2-init "${STAGE2_INIT}")
fi
if [ "${EARLY_STOP}" = "0" ]; then
    extra_args+=(--disable-early-stop)
fi

python sdcl_sysu_v34_irmt.py \
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
    --best-select-mode full \
    --stage1-best-select-mode legacy \
    --trainer-backend source \
    --grad-accum-steps 1 \
    --enable-irmt \
    --irmt-merge-frac "${IRMT_MERGE_FRAC}" \
    --irmt-reciprocal-topk "${IRMT_RECIPROCAL_TOPK}" \
    --irmt-min-sim "${IRMT_MIN_SIM}" \
    --irmt-margin-floor "${IRMT_MARGIN_FLOOR}" \
    --irmt-force-sim "${IRMT_FORCE_SIM}" \
    --irmt-min-cra-purity "${IRMT_MIN_CRA_PURITY}" \
    --logs-dir "${LOG_DIR}" \
    "${extra_args[@]}"
