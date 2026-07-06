#!/usr/bin/env python3
import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PAPER = {
    "all": {"mAP": 63.24, "mINP": 51.06},
    "indoor": {"mAP": 76.90, "mINP": 73.50},
}

FALLBACK_BASELINE = {
    "all": {"rank1": 65.55, "mAP": 63.41, "mINP": 50.38},
    "indoor": {"rank1": 71.12, "mAP": 76.58, "mINP": 73.12},
}


@dataclass
class Metrics:
    rank1: float
    mAP: float
    mINP: float

    def fmt(self) -> str:
        return f"{self.rank1:.2f} / {self.mAP:.2f} / {self.mINP:.2f}"


@dataclass
class EpochRecord:
    epoch: int
    all: Metrics
    indoor: Metrics
    score: Optional[float] = None
    is_best: bool = False


FC_RE = re.compile(
    r"Rank-1:\s*([0-9.]+)%.*?mAP:\s*([0-9.]+)%\|?\s*mINP:\s*([0-9.]+)%",
    re.IGNORECASE,
)
FINISHED_RE = re.compile(r"Finished epoch\s+(\d+).*?(?:score:\s*([0-9.]+))?.*?(\*)?\s*$")


def parse_fc(line: str) -> Optional[Metrics]:
    match = FC_RE.search(line)
    if not match:
        return None
    return Metrics(*(float(match.group(i)) for i in range(1, 4)))


def parse_log(path: str) -> Tuple[List[EpochRecord], Optional[Tuple[Metrics, Metrics]], Optional[int]]:
    records: List[EpochRecord] = []
    pending_all: Optional[Metrics] = None
    pending_indoor: Optional[Metrics] = None
    best_eval_all: Optional[Metrics] = None
    best_eval_indoor: Optional[Metrics] = None
    best_model_epoch: Optional[int] = None
    mode: Optional[str] = None
    in_best_eval = False

    with open(path, "r", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if "Test with the best model" in line:
                in_best_eval = True
                mode = None
                continue
            model_best = re.search(r"model_best from epoch\s+(\d+)", line)
            if model_best:
                best_model_epoch = int(model_best.group(1)) - 1

            low = line.lower()
            if line == "all" or "all search all average" in low:
                mode = "all"
                continue
            if "indoor all average" in low:
                mode = "indoor"
                continue
            if line == "All Average:" and mode is None:
                mode = "all"
                continue

            metrics = parse_fc(line)
            if metrics and mode == "all":
                if in_best_eval:
                    best_eval_all = metrics
                else:
                    pending_all = metrics
                mode = None
                continue
            if metrics and mode == "indoor":
                if in_best_eval:
                    best_eval_indoor = metrics
                else:
                    pending_indoor = metrics
                mode = None
                continue

            finished = FINISHED_RE.search(line)
            if finished and pending_all and pending_indoor:
                score = float(finished.group(2)) if finished.group(2) else None
                records.append(
                    EpochRecord(
                        epoch=int(finished.group(1)),
                        all=pending_all,
                        indoor=pending_indoor,
                        score=score,
                        is_best="*" in line,
                    )
                )
                pending_all = None
                pending_indoor = None

    best_eval = None
    if best_eval_all and best_eval_indoor:
        best_eval = (best_eval_all, best_eval_indoor)
    return records, best_eval, best_model_epoch


def choose_best(records: List[EpochRecord]) -> Optional[EpochRecord]:
    if not records:
        return None
    starred = [record for record in records if record.is_best]
    if starred:
        return starred[-1]
    scored = [record for record in records if record.score is not None]
    if scored:
        return max(scored, key=lambda record: record.score or -1.0)
    return max(records, key=lambda record: (record.all.mAP, record.indoor.mAP, record.all.mINP, record.indoor.mINP))


def single_metric_highs(records: List[EpochRecord]) -> Dict[str, Tuple[int, float]]:
    highs: Dict[str, Tuple[int, float]] = {}
    for side in ("all", "indoor"):
        for key in ("rank1", "mAP", "mINP"):
            best = max(records, key=lambda record: getattr(getattr(record, side), key))
            highs[f"{side}_{key}"] = (best.epoch, getattr(getattr(best, side), key))
    return highs


def load_baseline(repo_root: str) -> Dict[str, Metrics]:
    baseline_path = os.path.join(repo_root, "logs", "0620", "log复现最优.txt")
    if os.path.exists(baseline_path):
        records, best_eval, _ = parse_log(baseline_path)
        if best_eval:
            return {"all": best_eval[0], "indoor": best_eval[1]}
        best = choose_best(records)
        if best:
            return {"all": best.all, "indoor": best.indoor}
    return {
        "all": Metrics(**FALLBACK_BASELINE["all"]),
        "indoor": Metrics(**FALLBACK_BASELINE["indoor"]),
    }


def gap(metrics: Metrics, base: Metrics) -> str:
    return (
        f"all {metrics.rank1 - base.rank1:+.2f} / "
        f"{metrics.mAP - base.mAP:+.2f} / {metrics.mINP - base.mINP:+.2f}"
    )


def paper_status(all_metrics: Metrics, indoor_metrics: Metrics) -> str:
    passed = []
    failed = []
    checks = [
        ("all mAP", all_metrics.mAP, PAPER["all"]["mAP"]),
        ("all mINP", all_metrics.mINP, PAPER["all"]["mINP"]),
        ("indoor mAP", indoor_metrics.mAP, PAPER["indoor"]["mAP"]),
        ("indoor mINP", indoor_metrics.mINP, PAPER["indoor"]["mINP"]),
    ]
    for name, value, target in checks:
        (passed if value >= target else failed).append(name)
    if not failed:
        return "全部超过"
    if passed:
        return "超过 " + ", ".join(passed) + "; 未超过 " + ", ".join(failed)
    return "未超过"


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SYSU SDCL training logs.")
    parser.add_argument("log_file")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    log_file = os.path.abspath(args.log_file)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    records, best_eval, best_model_epoch = parse_log(log_file)
    baseline = load_baseline(repo_root)
    name = args.name or os.path.splitext(os.path.basename(log_file))[0]

    if not records and not best_eval:
        print(f"# {name}\n")
        print(f"No complete SYSU epoch or best-model metrics found in `{log_file}`.")
        return

    best_record = choose_best(records)
    final_record = records[-1] if records else None
    best_all, best_indoor = best_eval if best_eval else (best_record.all, best_record.indoor)  # type: ignore[union-attr]

    print(f"# {name}\n")
    print("| 实验名 | best all Rank-1/mAP/mINP | best indoor Rank-1/mAP/mINP | 相对复现最优差距 | 是否超过论文 |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| {name} | {best_all.fmt()} | {best_indoor.fmt()} | "
        f"{gap(best_all, baseline['all'])}; indoor "
        f"{best_indoor.rank1 - baseline['indoor'].rank1:+.2f} / "
        f"{best_indoor.mAP - baseline['indoor'].mAP:+.2f} / "
        f"{best_indoor.mINP - baseline['indoor'].mINP:+.2f} | "
        f"{paper_status(best_all, best_indoor)} |"
    )

    if best_model_epoch is not None:
        print(f"\n- best model checkpoint epoch: {best_model_epoch}")
    elif best_record:
        print(f"\n- best epoch by log score: {best_record.epoch}")
    if final_record:
        print(f"- final epoch: {final_record.epoch}, all {final_record.all.fmt()}, indoor {final_record.indoor.fmt()}")
    if records:
        print("- single-metric highest epochs:")
        for key, (epoch, value) in single_metric_highs(records).items():
            print(f"  - {key}: epoch {epoch}, {value:.2f}")
    print(
        "\nBaseline: reproduction best all "
        f"{baseline['all'].fmt()}, indoor {baseline['indoor'].fmt()}."
    )
    print(
        "Paper targets: all mAP/mINP "
        f"{PAPER['all']['mAP']:.2f}/{PAPER['all']['mINP']:.2f}; indoor mAP/mINP "
        f"{PAPER['indoor']['mAP']:.2f}/{PAPER['indoor']['mINP']:.2f}."
    )


if __name__ == "__main__":
    main()
