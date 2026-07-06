#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/watch_experiment_screen.sh EXP_NAME TRAIN_SESSION LOG_DIR LOG_FILE [INTERVAL]

Optional environment:
  LOSS_WARN_THRESHOLD=70
  LOSS_CRITICAL_THRESHOLD=100
  TRAIN_DEAD_CONFIRMATIONS=2
  LOG_STABLE_CONFIRMATIONS=1

Example:
  scripts/watch_experiment_screen.sh \
    v38_proto_bridge \
    train_v38_proto_bridge \
    /home/lhp/project/SDCL2/logs/0705/sysu_v38_proto_bridge_seed1_fullchain \
    /home/lhp/project/SDCL2/logs/0705/log_sysu_v38_proto_bridge_seed1_fullchain.txt \
    300
EOF
}

if [ "$#" -lt 4 ]; then
    usage >&2
    exit 2
fi

EXP_NAME="$1"
TRAIN_SESSION="$2"
LOG_DIR="$3"
LOG_FILE="$4"
INTERVAL="${5:-${INTERVAL:-300}}"
LOSS_WARN_THRESHOLD="${LOSS_WARN_THRESHOLD:-70}"
LOSS_CRITICAL_THRESHOLD="${LOSS_CRITICAL_THRESHOLD:-100}"
TRAIN_DEAD_CONFIRMATIONS="${TRAIN_DEAD_CONFIRMATIONS:-2}"
LOG_STABLE_CONFIRMATIONS="${LOG_STABLE_CONFIRMATIONS:-1}"

if ! command -v screen >/dev/null 2>&1; then
    echo "screen is not installed or not in PATH." >&2
    exit 127
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONITOR_SESSION="watch_$(printf '%s' "${EXP_NAME}" | tr -cs 'A-Za-z0-9_.-' '_')"
mkdir -p "${LOG_DIR}"
if [[ "${LOG_FILE}" != /* ]]; then
    LOG_FILE="${LOG_DIR}/${LOG_FILE}"
fi
MONITOR_LOG="${LOG_DIR}/monitor.log"
SUMMARY_FILE="${LOG_DIR}/summary.md"

if screen -ls | grep -Eq "[[:space:]][0-9]+\\.${MONITOR_SESSION}[[:space:]]"; then
    echo "monitor screen session already exists: ${MONITOR_SESSION}" >&2
    echo "Attach with: screen -r ${MONITOR_SESSION}" >&2
    exit 3
fi

MONITOR_CMD=$(cat <<EOF
set -u
PROJECT_DIR=$(printf '%q' "${PROJECT_DIR}")
TRAIN_SESSION=$(printf '%q' "${TRAIN_SESSION}")
LOG_DIR=$(printf '%q' "${LOG_DIR}")
LOG_FILE=$(printf '%q' "${LOG_FILE}")
MONITOR_LOG=$(printf '%q' "${MONITOR_LOG}")
SUMMARY_FILE=$(printf '%q' "${SUMMARY_FILE}")
INTERVAL=$(printf '%q' "${INTERVAL}")
LOSS_WARN_THRESHOLD=$(printf '%q' "${LOSS_WARN_THRESHOLD}")
LOSS_CRITICAL_THRESHOLD=$(printf '%q' "${LOSS_CRITICAL_THRESHOLD}")
TRAIN_DEAD_CONFIRMATIONS=$(printf '%q' "${TRAIN_DEAD_CONFIRMATIONS}")
LOG_STABLE_CONFIRMATIONS=$(printf '%q' "${LOG_STABLE_CONFIRMATIONS}")
last_size=-1
dead_count=0
stable_count=0
echo "[\$(date -Iseconds)] monitor started for session=\${TRAIN_SESSION} log=\${LOG_FILE} loss_warn=\${LOSS_WARN_THRESHOLD} loss_critical=\${LOSS_CRITICAL_THRESHOLD} dead_confirmations=\${TRAIN_DEAD_CONFIRMATIONS} stable_confirmations=\${LOG_STABLE_CONFIRMATIONS}" >> "\${MONITOR_LOG}"
while true; do
    now="\$(date -Iseconds)"
    if [ -f "\${LOG_FILE}" ]; then
        size="\$(stat -c '%s' "\${LOG_FILE}" 2>/dev/null || echo 0)"
    else
        size=0
    fi
    if [ "\${last_size}" -ge 0 ]; then
        if [ "\${size}" -gt "\${last_size}" ]; then
            growth="growing +\$((size - last_size)) bytes"
            stable_count=0
        else
            growth="not_growing"
            stable_count=\$((stable_count + 1))
        fi
    else
        growth="initial_size=\${size}"
        stable_count=0
    fi
    last_size="\${size}"

    train_alive=0
    if screen -ls | grep -Eq "[[:space:]][0-9]+\\.\${TRAIN_SESSION}[[:space:]]"; then
        train_alive=1
    fi
    if [ "\${train_alive}" -eq 1 ]; then
        dead_count=0
    else
        dead_count=\$((dead_count + 1))
    fi

    errors="none"
    loss_status="loss=unavailable"
    latest_loss=""
    max_recent_loss=""
    if [ -f "\${LOG_FILE}" ]; then
        if grep -Eqi 'Traceback|CUDA out of memory|out of memory|\\bNaN\\b|RuntimeError' "\${LOG_FILE}"; then
            errors="detected"
        fi
        latest_loss="\$(grep -Eo 'Loss[[:space:]]+[0-9]+([.][0-9]+)?' "\${LOG_FILE}" | awk '{print \$2}' | tail -1 || true)"
        max_recent_loss="\$(grep -Eo 'Loss[[:space:]]+[0-9]+([.][0-9]+)?' "\${LOG_FILE}" | awk '{print \$2}' | tail -200 | awk 'BEGIN{m=0} {if(\$1+0>m)m=\$1+0} END{if(NR>0)print m}' || true)"
        if [ -n "\${latest_loss}" ]; then
            loss_status="loss_latest=\${latest_loss}"
        fi
        if [ -n "\${max_recent_loss}" ]; then
            loss_status="\${loss_status} loss_recent_max=\${max_recent_loss}"
            if awk "BEGIN{exit !(\${max_recent_loss} >= \${LOSS_CRITICAL_THRESHOLD})}"; then
                errors="detected"
                loss_status="\${loss_status} loss_state=critical"
                printf '[%s] CRITICAL loss explosion: recent_max=%s threshold=%s log=%s\n' "\${now}" "\${max_recent_loss}" "\${LOSS_CRITICAL_THRESHOLD}" "\${LOG_FILE}" > "\${LOG_DIR}/LOSS_ALERT"
            elif awk "BEGIN{exit !(\${max_recent_loss} >= \${LOSS_WARN_THRESHOLD})}"; then
                loss_status="\${loss_status} loss_state=warn"
                printf '[%s] WARNING high loss: recent_max=%s threshold=%s log=%s\n' "\${now}" "\${max_recent_loss}" "\${LOSS_WARN_THRESHOLD}" "\${LOG_FILE}" > "\${LOG_DIR}/LOSS_ALERT"
            else
                loss_status="\${loss_status} loss_state=ok"
            fi
        fi
    fi

    ckpt="checkpoint=\$([ -f "\${LOG_DIR}/checkpoint.pth.tar" ] && echo yes || echo no)"
    best="model_best=\$([ -f "\${LOG_DIR}/model_best.pth.tar" ] && echo yes || echo no)"
    s1best="20model_best=\$([ -f "\${LOG_DIR}/20model_best.pth.tar" ] && echo yes || echo no)"
    echo "[\${now}] train_alive=\${train_alive} dead_count=\${dead_count}/\${TRAIN_DEAD_CONFIRMATIONS} log=\${growth} stable_count=\${stable_count}/\${LOG_STABLE_CONFIRMATIONS} errors=\${errors} \${loss_status} \${ckpt} \${best} \${s1best}" >> "\${MONITOR_LOG}"

    if [ "\${errors}" = "detected" ]; then
        echo "[\${now}] error pattern detected; last matches:" >> "\${MONITOR_LOG}"
        grep -Ein 'Traceback|CUDA out of memory|out of memory|\\bNaN\\b|RuntimeError' "\${LOG_FILE}" | tail -20 >> "\${MONITOR_LOG}"
        if [ -f "\${LOG_DIR}/LOSS_ALERT" ]; then
            cat "\${LOG_DIR}/LOSS_ALERT" >> "\${MONITOR_LOG}"
        fi
    fi

    if [ "\${train_alive}" -eq 0 ] && [ "\${dead_count}" -ge "\${TRAIN_DEAD_CONFIRMATIONS}" ] && [ "\${stable_count}" -ge "\${LOG_STABLE_CONFIRMATIONS}" ]; then
        echo "[\${now}] training session ended and log is stable; parsing log" >> "\${MONITOR_LOG}"
        if [ -f "\${LOG_FILE}" ]; then
            cd "\${PROJECT_DIR}" && python scripts/parse_sysu_log.py "\${LOG_FILE}" > "\${SUMMARY_FILE}" 2>&1
            cat "\${SUMMARY_FILE}" >> "\${MONITOR_LOG}"
        else
            echo "log file missing: \${LOG_FILE}" >> "\${MONITOR_LOG}"
        fi
        exit 0
    elif [ "\${train_alive}" -eq 0 ]; then
        echo "[\${now}] training session missing but waiting for confirmations/log stability before parsing" >> "\${MONITOR_LOG}"
    fi
    sleep "\${INTERVAL}"
done
EOF
)

screen -dmS "${MONITOR_SESSION}" bash -lc "${MONITOR_CMD}"

echo "Started monitor screen session: ${MONITOR_SESSION}"
echo "Monitor log: ${MONITOR_LOG}"
echo "Summary file after completion: ${SUMMARY_FILE}"
echo "Attach: screen -r ${MONITOR_SESSION}"
