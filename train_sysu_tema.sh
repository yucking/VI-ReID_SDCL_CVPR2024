#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SEED="${SEED:-1}"
LOG_ROOT="${LOG_ROOT:-/home/lhp/project/SDCL2/logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/sysu_tema_seed${SEED}}"
SCRIPT="sdcl_sysu_tema.py"

python - "$LOG_DIR" "$SCRIPT" "$SEED" <<'PY'
import os
import re
import subprocess
import sys
import time

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
    "--best-select-mode", "rank1",
    "--stage1-best-select-mode", "rank1",
    "--trainer-backend", "source",
    "--grad-accum-steps", "1",
    "--seed", seed,
    "--enable-tema",
    "--tema-start", "20",
    "--tema-momentum", "0.60",
    "--tema-warmup", "2",
    "--enable-stage2-softweight",
    "--stage2-softweight-min", "0.80",
    "--stage2-softweight-power", "1.0",
    "--enable-stage2-tailtrim",
    "--stage2-tailtrim-delay", "3",
    "--stage2-tailtrim-warmup", "4",
    "--stage2-tailtrim-pct", "0.02",
    "--stage2-tailtrim-decay-after", "-1",
    "--logs-dir", log_dir,
]

metric_pattern = re.compile(
    r"Finished epoch\s+(\d+).*?all_mAP:\s*([0-9.]+)%\s+indoor_mAP:\s*([0-9.]+)%\s+indoor_mINP:\s*([0-9.]+)%",
    re.S,
)
log_path = os.path.join(log_dir, "log.txt")

print("[RUN] " + " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd)
checked = set()
best_after_50 = None
best_stage1 = None
stopped_early = False


def terminate(message):
    global stopped_early
    print(message, flush=True)
    stopped_early = True
    proc.terminate()
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return "stopped"


def scan_log():
    global best_after_50, best_stage1
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    for match in metric_pattern.finditer(text):
        epoch = int(match.group(1))
        all_map = float(match.group(2))
        indoor_map = float(match.group(3))
        indoor_minp = float(match.group(4))

        if epoch < 30:
            score = indoor_map + indoor_minp
            if best_stage1 is None or score > best_stage1[0]:
                best_stage1 = (score, epoch, all_map, indoor_map, indoor_minp)

        if epoch >= 50:
            score = indoor_map + indoor_minp
            if best_after_50 is None or score > best_after_50[0]:
                best_after_50 = (score, epoch, all_map, indoor_map, indoor_minp)

        if epoch == 29 and epoch not in checked:
            checked.add(epoch)
            _, best_epoch, best_all, best_map, best_minp = best_stage1
            print(
                "[MONITOR] epoch=29 current all_mAP={:.2f} indoor_mAP={:.2f} indoor_mINP={:.2f}; "
                "best_stage1 all_mAP={:.2f} indoor_mAP={:.2f} indoor_mINP={:.2f}@{} hard=65.5/61.5 target=67.0/63.0".format(
                    all_map, indoor_map, indoor_minp, best_all, best_map, best_minp, best_epoch
                ),
                flush=True,
            )
            if "[TEMA]" not in text:
                print("[WARN] no [TEMA] records found before epoch29; check --enable-tema and log capture", flush=True)
            if best_map < 65.5 or best_minp < 61.5:
                return terminate("[STOP] best_stage1 below relaxed epoch29 hard stop for TEMA")
            if best_map < 67.0 or best_minp < 63.0:
                print("[MONITOR] best_stage1 below target but above hard stop; continue to stage2", flush=True)

        elif epoch == 34 and epoch not in checked:
            checked.add(epoch)
            print(
                "[MONITOR] epoch=34 all_mAP={:.2f} indoor_mAP={:.2f} indoor_mINP={:.2f} stop=74.0/70.3 pass=75.0/71.3".format(
                    all_map, indoor_map, indoor_minp
                ),
                flush=True,
            )
            if indoor_map < 74.0 or indoor_minp < 70.3:
                return terminate("[STOP] below epoch34 hard stop for TEMA")
            if indoor_map < 75.0 or indoor_minp < 71.3:
                print("[MONITOR] epoch34 in gray zone; continue to epoch44", flush=True)

        elif epoch == 44 and epoch not in checked:
            checked.add(epoch)
            print(
                "[MONITOR] epoch=44 all_mAP={:.2f} indoor_mAP={:.2f} indoor_mINP={:.2f} hard=75.4/71.8".format(
                    all_map, indoor_map, indoor_minp
                ),
                flush=True,
            )
            if all_map < 61.5:
                print("[WARN] epoch44 all_mAP below 61.5; indoor/all-search imbalance", flush=True)
            if indoor_map < 75.4 or indoor_minp < 71.8:
                return terminate("[STOP] below epoch44 hard stop for TEMA")

        elif epoch == 52 and best_after_50 is not None and epoch not in checked:
            checked.add(epoch)
            _, best_epoch, best_all, best_map, best_minp = best_after_50
            print(
                "[MONITOR] best_after_50 all_mAP={:.2f} indoor={:.2f}/{:.2f}@{}".format(
                    best_all, best_map, best_minp, best_epoch
                ),
                flush=True,
            )
            if best_map < 75.9 or best_minp < 72.4:
                return terminate("[STOP] epoch52 best_after_50 below 75.9/72.4 for TEMA")
    return None


while proc.poll() is None:
    if scan_log() == "stopped":
        break
    time.sleep(60)

scan_log()
sys.exit(0 if stopped_early else (proc.returncode if proc.returncode is not None else 0))
PY

python - "${LOG_DIR}/log.txt" <<'PY'
import os
import re
import sys

path = sys.argv[1]
if not os.path.exists(path):
    raise SystemExit(0)

text = open(path, "r", encoding="utf-8", errors="replace").read()
cluster = {}
for mod, epoch, clusters, outliers in re.findall(
    r"Statistics for (IR|RGB) epoch\s+(\d+):\s+(\d+) clusters outlier\s+(\d+)", text
):
    cluster.setdefault(int(epoch), {})[mod] = (int(clusters), int(outliers))

metrics = {
    int(epoch): (float(all_map), float(indoor_map), float(indoor_minp))
    for epoch, all_map, indoor_map, indoor_minp in re.findall(
        r"Finished epoch\s+(\d+).*?all_mAP:\s*([0-9.]+)%\s+indoor_mAP:\s*([0-9.]+)%\s+indoor_mINP:\s*([0-9.]+)%",
        text,
        re.S,
    )
}

print("[SUMMARY] epoch IR_cl/IR_out RGB_cl/RGB_out all_mAP indoor_mAP indoor_mINP")
for epoch in range(20, 30):
    ir = cluster.get(epoch, {}).get("IR")
    rgb = cluster.get(epoch, {}).get("RGB")
    metric = metrics.get(epoch)
    if not (ir and rgb and metric):
        continue
    print(
        "[SUMMARY] {:02d} {:4d}/{:<4d} {:4d}/{:<4d} {:6.2f} {:6.2f} {:6.2f}".format(
            epoch, ir[0], ir[1], rgb[0], rgb[1], metric[0], metric[1], metric[2]
        )
    )
PY

python evaluate_sysu_v4_push_log.py "${LOG_DIR}/log.txt"
