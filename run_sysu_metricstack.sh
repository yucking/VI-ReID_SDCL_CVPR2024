#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/lhp/project/SDCL2}"
CHECKPOINT="${CHECKPOINT:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/logs/sysu_metricstack_repro_best}"
WORKERS="${WORKERS:-4}"
FORCE_EXTRACT="${FORCE_EXTRACT:-0}"
extra_cli_args=("$@")

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Set CHECKPOINT to the frozen model_best.pth.tar before running." >&2
    exit 2
fi

extra_args=()
if [[ "${FORCE_EXTRACT}" == "1" ]]; then
    extra_args+=(--force-extract)
fi

python test_sysu_metricstack.py \
    --config-file vit_base_ics_288.yml \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${WORKERS}" \
    "${extra_args[@]}" \
    "${extra_cli_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/log.txt"
