#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_experiment_screen.sh EXP_NAME TRAIN_SCRIPT LOG_DIR LOG_FILE [-- TRAIN_ARGS...]

Example:
  CUDA_VISIBLE_DEVICES=0,1 SEED=1 \
  scripts/run_experiment_screen.sh \
    v38_proto_bridge \
    train_sysu_v38_proto_bridge.sh \
    /home/lhp/project/SDCL2/logs/0705/sysu_v38_proto_bridge_seed1_fullchain \
    /home/lhp/project/SDCL2/logs/0705/log_sysu_v38_proto_bridge_seed1_fullchain.txt
EOF
}

if [ "$#" -lt 4 ]; then
    usage >&2
    exit 2
fi

EXP_NAME="$1"
TRAIN_SCRIPT="$2"
LOG_DIR="$3"
LOG_FILE="$4"
shift 4
if [ "${1:-}" = "--" ]; then
    shift
fi

if ! command -v screen >/dev/null 2>&1; then
    echo "screen is not installed or not in PATH." >&2
    exit 127
fi

SESSION_NAME="train_$(printf '%s' "${EXP_NAME}" | tr -cs 'A-Za-z0-9_.-' '_')"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT}"
if [[ "${TRAIN_SCRIPT_PATH}" != /* ]]; then
    TRAIN_SCRIPT_PATH="${PROJECT_DIR}/${TRAIN_SCRIPT_PATH}"
fi
if [ ! -f "${TRAIN_SCRIPT_PATH}" ]; then
    echo "Missing train script: ${TRAIN_SCRIPT_PATH}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
if [[ "${LOG_FILE}" != /* ]]; then
    LOG_FILE="${LOG_DIR}/${LOG_FILE}"
fi
mkdir -p "$(dirname "${LOG_FILE}")"

if screen -ls | grep -Eq "[[:space:]][0-9]+\\.${SESSION_NAME}[[:space:]]"; then
    echo "screen session already exists: ${SESSION_NAME}" >&2
    echo "Refusing to overwrite. Attach with: screen -r ${SESSION_NAME}" >&2
    exit 3
fi

STARTED_AT="$(date -Iseconds)"
META_FILE="${LOG_DIR}/run_screen.meta"
ARGS_QUOTED=""
for arg in "$@"; do
    ARGS_QUOTED+=" $(printf '%q' "${arg}")"
done

TRAIN_CMD="cd $(printf '%q' "${PROJECT_DIR}") && LOG_DIR=$(printf '%q' "${LOG_DIR}") bash $(printf '%q' "${TRAIN_SCRIPT_PATH}")${ARGS_QUOTED} 2>&1 | tee $(printf '%q' "${LOG_FILE}")"

{
    echo "experiment=${EXP_NAME}"
    echo "screen_session=${SESSION_NAME}"
    echo "started_at=${STARTED_AT}"
    echo "project_dir=${PROJECT_DIR}"
    echo "train_script=${TRAIN_SCRIPT_PATH}"
    echo "log_dir=${LOG_DIR}"
    echo "log_file=${LOG_FILE}"
    echo "command=${TRAIN_CMD}"
} > "${META_FILE}"

screen -dmS "${SESSION_NAME}" bash -lc "${TRAIN_CMD}"

echo "Started training screen session: ${SESSION_NAME}"
echo "Log file: ${LOG_FILE}"
echo "Metadata: ${META_FILE}"
echo "Attach: screen -r ${SESSION_NAME}"
