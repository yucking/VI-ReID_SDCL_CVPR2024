#!/usr/bin/env python3
"""Preflight gate for SDCL2 research experiments."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

REQUIRED_CONTRACT_FIELDS = [
    "## Experiment Name",
    "## Date",
    "## Baseline Commit",
    "## Baseline Configuration",
    "## Problem",
    "## Evidence",
    "## Gate Decision",
    "## Gate Output Path",
    "## Hypothesis",
    "## Mechanism",
    "## Changed Files",
    "## Independent Variable",
    "## Fixed Variables",
    "## Intermediate Metric",
    "## Falsification Result",
    "## Dataset And Protocol",
    "## Seeds",
    "## Baseline Variance",
    "## Budget",
    "## Training Command",
    "## Monitor Command",
    "## Monitor Log Path",
    "## Log Directory",
    "## Summary Path",
    "## Parser Command",
    "## Registry Row",
    "## Checkpoint Plan",
    "## Environment",
    "## Determinism",
    "## Smoke Test",
    "## Success Criteria",
    "## Failure Criteria",
    "## Strongest Opposing Explanation",
    "## Disproof Plan",
    "## Ablation",
    "## Rollback",
]

REQUIRED_REGISTRY_FIELDS = [
    "experiment_id",
    "status",
    "contract_path",
    "log_path",
    "summary_path",
    "monitor_log_path",
    "checkpoint_path",
    "model_best_path",
    "stage1_best_path",
    "baseline_comparator",
    "baseline_variance",
    "evidence_grade",
]


def run_git(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def add_result(results: list[tuple[str, str, str]], ok: bool, key: str, detail: str) -> None:
    results.append(("PASS" if ok else "FAIL", key, detail))


def contract_field_body(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    rest = text[start + len(heading) :]
    next_heading = rest.find("\n## ")
    if next_heading >= 0:
        rest = rest[:next_heading]
    return rest.strip()


def registry_row(path: Path, experiment_id: str) -> tuple[dict[str, str] | None, list[str]]:
    if not path.is_file():
        return None, REQUIRED_REGISTRY_FIELDS
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [field for field in REQUIRED_REGISTRY_FIELDS if field not in (reader.fieldnames or [])]
        for row in reader:
            if row.get("experiment_id") == experiment_id:
                return row, missing
    return None, missing


def log_dir_available(log_dir: Path, allow_existing: bool) -> bool:
    if not log_dir.exists():
        return True
    if allow_existing:
        return True
    if not log_dir.is_dir():
        return False
    return not any(log_dir.iterdir())


def contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def extract_ints(text: str) -> list[int]:
    return [int(item) for item in re.findall(r"\b\d+\b", text)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--registry", default=Path("experiments/registry.csv"), type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--train-script", required=True, type=Path)
    parser.add_argument("--allow-existing-log-dir", action="store_true")
    parser.add_argument("--allow-no-git", action="store_true")
    args = parser.parse_args()

    results: list[tuple[str, str, str]] = []

    code, commit = run_git(["rev-parse", "--short", "HEAD"])
    add_result(
        results,
        code == 0 or args.allow_no_git,
        "git_commit",
        commit if code == 0 else f"{commit}; allow_no_git={args.allow_no_git}",
    )

    code, status = run_git(["status", "--short"])
    clean = code == 0 and status == ""
    add_result(
        results,
        clean or args.allow_no_git,
        "git_worktree",
        "clean" if clean else (status or "git unavailable"),
    )

    contract_ok = args.contract.is_file()
    add_result(results, contract_ok, "contract_exists", str(args.contract))
    text = args.contract.read_text(encoding="utf-8") if contract_ok else ""
    for heading in REQUIRED_CONTRACT_FIELDS:
        body = contract_field_body(text, heading)
        ok = bool(body) and "UNVERIFIED" not in body and "UNKNOWN" not in body
        add_result(results, ok, f"contract:{heading[3:]}", body.splitlines()[0] if body else "missing")

    row, missing_registry_fields = registry_row(args.registry, args.experiment_id)
    add_result(results, not missing_registry_fields, "registry_schema", ", ".join(missing_registry_fields) or "complete")
    add_result(results, row is not None, "registry_experiment_id", args.experiment_id)
    if row is not None:
        for field in ["contract_path", "log_path", "summary_path", "monitor_log_path", "checkpoint_path"]:
            value = row.get(field, "")
            add_result(results, bool(value) and value not in {"UNKNOWN", "UNVERIFIED"}, f"registry:{field}", value)
    add_result(
        results,
        log_dir_available(args.log_dir, args.allow_existing_log_dir),
        "log_dir_available",
        str(args.log_dir),
    )
    add_result(results, args.train_script.is_file(), "train_script_exists", str(args.train_script))

    sections = {heading[3:]: contract_field_body(text, heading) for heading in REQUIRED_CONTRACT_FIELDS}
    add_result(results, contains_any(sections.get("Gate Decision", ""), ["pass"]), "gate_decision_pass", sections.get("Gate Decision", "missing"))
    add_result(
        results,
        contains_any(sections.get("Evidence", ""), ["logs/", ".log", ".txt", "summary.md"]),
        "evidence_path_present",
        sections.get("Evidence", "missing"),
    )
    add_result(
        results,
        contains_any(sections.get("Hypothesis", ""), ["falsifiable", "falsify", "证伪"]),
        "hypothesis_falsifiable",
        sections.get("Hypothesis", "missing"),
    )
    add_result(
        results,
        contains_any(sections.get("Independent Variable", ""), ["one primary", "single primary", "一个主要"]),
        "single_primary_variable",
        sections.get("Independent Variable", "missing"),
    )
    add_result(
        results,
        contains_any(sections.get("Intermediate Metric", ""), ["cluster", "outlier", "coverage", "confidence", "loss", "map", "minp"]),
        "intermediate_metric_named",
        sections.get("Intermediate Metric", "missing"),
    )
    add_result(
        results,
        contains_any(sections.get("Falsification Result", ""), ["reject", "fail", "falsify", "证伪"]),
        "falsification_result_named",
        sections.get("Falsification Result", "missing"),
    )
    seed_ints = sorted(set(extract_ints(sections.get("Seeds", ""))))
    add_result(results, len(seed_ints) >= 3, "multi_seed_list", ",".join(map(str, seed_ints)) or "missing")
    add_result(
        results,
        contains_any(sections.get("Baseline Variance", ""), ["baseline", "variance", "std", "波动"]),
        "baseline_variance_reference",
        sections.get("Baseline Variance", "missing"),
    )
    add_result(
        results,
        args.train_script.name in sections.get("Training Command", "") or str(args.train_script) in sections.get("Training Command", ""),
        "training_command_matches_script",
        sections.get("Training Command", "missing"),
    )
    add_result(
        results,
        str(args.log_dir) in sections.get("Log Directory", ""),
        "log_dir_matches_contract",
        sections.get("Log Directory", "missing"),
    )
    add_result(
        results,
        "watch_experiment_screen.sh" in sections.get("Monitor Command", ""),
        "monitor_command_screen",
        sections.get("Monitor Command", "missing"),
    )
    add_result(
        results,
        "monitor.log" in sections.get("Monitor Log Path", ""),
        "monitor_log_path_present",
        sections.get("Monitor Log Path", "missing"),
    )
    add_result(
        results,
        "parse_sysu_log.py" in sections.get("Parser Command", ""),
        "parser_command_present",
        sections.get("Parser Command", "missing"),
    )
    add_result(
        results,
        args.experiment_id in sections.get("Registry Row", ""),
        "registry_row_mentions_experiment",
        sections.get("Registry Row", "missing"),
    )
    add_result(
        results,
        contains_any(sections.get("Checkpoint Plan", ""), ["checkpoint.pth.tar", "model_best.pth.tar", "20model_best.pth.tar"]),
        "checkpoint_plan_present",
        sections.get("Checkpoint Plan", "missing"),
    )

    failed = [item for item in results if item[0] == "FAIL"]
    for status_text, key, detail in results:
        print(f"{status_text} {key}: {detail}")
    print("PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
