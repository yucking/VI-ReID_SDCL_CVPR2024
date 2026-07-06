#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
DATA_DIR="${DATA_DIR:-/home/lhp/project/DATASETS/SYSU-MM01}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/0705/sysu_v38_proto_bridge_seed${SEED}_fullchain}"
WORKERS="${WORKERS:-8}"
EPOCHS="${EPOCHS:-60}"
CMLABEL="${CMLABEL:-30}"
ENABLE_PROTO_BRIDGE="${ENABLE_PROTO_BRIDGE:-1}"
PROTO_BRIDGE_WEIGHT="${PROTO_BRIDGE_WEIGHT:-0.05}"
PROTO_BRIDGE_MIN_SIM="${PROTO_BRIDGE_MIN_SIM:-0.42}"
PROTO_BRIDGE_MIN_MARGIN="${PROTO_BRIDGE_MIN_MARGIN:-0.015}"
PROTO_BRIDGE_MIN_CLUSTER_SIZE="${PROTO_BRIDGE_MIN_CLUSTER_SIZE:-4}"
PROTO_BRIDGE_MAX_PAIRS="${PROTO_BRIDGE_MAX_PAIRS:-512}"
PROTO_BRIDGE_TEMP="${PROTO_BRIDGE_TEMP:-0.05}"
PROTO_BRIDGE_MOMENTUM="${PROTO_BRIDGE_MOMENTUM:-0.1}"

if [ "${SEED}" != "1" ]; then
    echo "Reference SYSU experiment is pinned to seed=1; got SEED=${SEED}." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"

bridge_args=()
if [ "${ENABLE_PROTO_BRIDGE}" = "1" ]; then
    bridge_args+=(
        --enable-proto-bridge
        --proto-bridge-weight "${PROTO_BRIDGE_WEIGHT}"
        --proto-bridge-min-sim "${PROTO_BRIDGE_MIN_SIM}"
        --proto-bridge-min-margin "${PROTO_BRIDGE_MIN_MARGIN}"
        --proto-bridge-min-cluster-size "${PROTO_BRIDGE_MIN_CLUSTER_SIZE}"
        --proto-bridge-max-pairs "${PROTO_BRIDGE_MAX_PAIRS}"
        --proto-bridge-temp "${PROTO_BRIDGE_TEMP}"
        --proto-bridge-momentum "${PROTO_BRIDGE_MOMENTUM}"
    )
fi

echo "==> v38 Stage-1 prototype bridge: enabled=${ENABLE_PROTO_BRIDGE} weight=${PROTO_BRIDGE_WEIGHT} min_sim=${PROTO_BRIDGE_MIN_SIM} min_margin=${PROTO_BRIDGE_MIN_MARGIN}"
echo "==> logs_dir: ${LOG_DIR}"

python sdcl_sysu_v38_proto_bridge.py \
    -b 96 \
    -j "${WORKERS}" \
    -a agw \
    -d sysu_all \
    --data-dir "${DATA_DIR}" \
    --iters 200 \
    --epochs "${EPOCHS}" \
    --momentum 0.1 \
    --eps 0.6 \
    --cmlabel "${CMLABEL}" \
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
    "${bridge_args[@]}"

python evaluate_sysu_v4_push_log.py "${LOG_DIR}/log.txt"
