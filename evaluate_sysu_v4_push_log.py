# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import re


METRIC_RE = re.compile(
    r"Finished epoch\s+(\d+).*?all_mAP:\s*([0-9.]+)%\s+"
    r"indoor_mAP:\s*([0-9.]+)%\s+indoor_mINP:\s*([0-9.]+)%",
    re.S,
)
BEST_RE = re.compile(
    r"model_best from epoch\s+(\d+).*?stored all_mAP=([0-9.]+)%\s+"
    r"indoor_mAP=([0-9.]+)%\s+indoor_mINP=([0-9.]+)%",
    re.S,
)


def parse_metrics(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    records = []
    for match in METRIC_RE.finditer(text):
        records.append(
            {
                "epoch": int(match.group(1)),
                "all_map": float(match.group(2)),
                "indoor_map": float(match.group(3)),
                "indoor_minp": float(match.group(4)),
            }
        )

    best = None
    match = BEST_RE.search(text)
    if match:
        best = {
            "epoch": int(match.group(1)) - 1,
            "all_map": float(match.group(2)),
            "indoor_map": float(match.group(3)),
            "indoor_minp": float(match.group(4)),
        }
    elif records:
        best = max(records, key=lambda item: (item["indoor_map"] + item["indoor_minp"], item["epoch"]))

    return records, best


def classify(best, paper_map, paper_minp, strong_map, strong_minp, weak_map, weak_minp):
    if best is None:
        return "no-metrics"
    if best["indoor_map"] >= paper_map and best["indoor_minp"] >= paper_minp:
        return "paper-level"
    if best["indoor_map"] >= strong_map and best["indoor_minp"] >= strong_minp:
        return "strong-rerun"
    if best["indoor_map"] >= weak_map and best["indoor_minp"] >= weak_minp:
        return "valid-but-needs-v2"
    return "weak-rerun"


def main():
    parser = argparse.ArgumentParser(description="Summarize SYSU v4_push/v2 indoor metrics from log.txt.")
    parser.add_argument("log_path")
    parser.add_argument("--paper-map", type=float, default=76.90)
    parser.add_argument("--paper-minp", type=float, default=73.50)
    parser.add_argument("--strong-map", type=float, default=76.70)
    parser.add_argument("--strong-minp", type=float, default=73.20)
    parser.add_argument("--weak-map", type=float, default=75.50)
    parser.add_argument("--weak-minp", type=float, default=72.00)
    args = parser.parse_args()

    records, best = parse_metrics(args.log_path)
    status = classify(
        best,
        args.paper_map,
        args.paper_minp,
        args.strong_map,
        args.strong_minp,
        args.weak_map,
        args.weak_minp,
    )

    print("log: {}".format(args.log_path))
    print("epochs_with_metrics: {}".format(len(records)))
    if best is None:
        print("status: {}".format(status))
        return

    print(
        "best: epoch={} all_mAP={:.2f} indoor_mAP={:.2f} indoor_mINP={:.2f}".format(
            best["epoch"], best["all_map"], best["indoor_map"], best["indoor_minp"]
        )
    )
    print("status: {}".format(status))


if __name__ == "__main__":
    main()
