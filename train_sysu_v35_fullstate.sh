#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
DATA_DIR="${DATA_DIR:-/home/lhp/project/DATASETS/SYSU-MM01}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_v35_fullstate_seed${SEED}_fullchain}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-50}"

FULL_CHAIN="${FULL_CHAIN:-1}"
STAGE2_RESUME_FULL_STATE="${STAGE2_RESUME_FULL_STATE:-}"
STAGE2_ONLY="${STAGE2_ONLY:-0}"
STAGE2_INIT="${STAGE2_INIT:-}"
NO_RESUME_RNG="${NO_RESUME_RNG:-0}"
DISABLE_STAGE2_HANDOFF_SAVE="${DISABLE_STAGE2_HANDOFF_SAVE:-0}"
STAGE2_HANDOFF_NAME="${STAGE2_HANDOFF_NAME:-stage2_handoff_full_state.pth.tar}"

if [ "${SEED}" != "1" ]; then
    echo "Reference SYSU experiment is pinned to seed=1; got SEED=${SEED}." >&2
    exit 2
fi
if [ -n "${STAGE2_RESUME_FULL_STATE}" ] && [ "${STAGE2_ONLY}" = "1" ]; then
    echo "STAGE2_RESUME_FULL_STATE and STAGE2_ONLY=1 are mutually exclusive." >&2
    exit 2
fi
if [ -n "${STAGE2_RESUME_FULL_STATE}" ] && [ ! -f "${STAGE2_RESUME_FULL_STATE}" ]; then
    echo "Missing full-state Stage-2 handoff: ${STAGE2_RESUME_FULL_STATE}" >&2
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
if [ "${FULL_CHAIN}" != "1" ] && [ -z "${STAGE2_RESUME_FULL_STATE}" ] && [ "${STAGE2_ONLY}" != "1" ]; then
    echo "FULL_CHAIN=0 requires STAGE2_RESUME_FULL_STATE or STAGE2_ONLY=1." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

extra_args=()
if [ -n "${STAGE2_RESUME_FULL_STATE}" ]; then
    extra_args+=(--stage2-resume-full-state "${STAGE2_RESUME_FULL_STATE}")
fi
if [ "${STAGE2_ONLY}" = "1" ]; then
    extra_args+=(--stage2-only --stage2-init "${STAGE2_INIT}")
fi
if [ "${NO_RESUME_RNG}" = "1" ]; then
    extra_args+=(--no-resume-rng)
fi
if [ "${DISABLE_STAGE2_HANDOFF_SAVE}" = "1" ]; then
    extra_args+=(--disable-stage2-handoff-save)
fi

if [ -n "${STAGE2_RESUME_FULL_STATE}" ]; then
    echo "==> v35 mode: full-state Stage-2 resume"
elif [ "${STAGE2_ONLY}" = "1" ]; then
    echo "==> v35 mode: model-only Stage-2 compatibility"
else
    echo "==> v35 mode: full-chain from epoch 0"
fi

python sdcl_sysu_v35_fullstate.py \
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
    --stage2-handoff-name "${STAGE2_HANDOFF_NAME}" \
    --logs-dir "${LOG_DIR}" \
    "${extra_args[@]}"
