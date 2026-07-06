#!/usr/bin/env python3
"""Validate SDCL2 local skills and research infrastructure."""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = [
    "sdcl-research-gate",
    "sdcl-implementation-audit",
    "sdcl-experiment-design",
    "sdcl-log-and-result-audit",
    "sdcl-paper-reviewer",
]
DOCS = [
    "docs/research/CODEBASE_MAP.md",
    "docs/research/STAGE1_STAGE2_BOUNDARY.md",
    "docs/research/RESEARCH_WORKFLOW.md",
    "docs/research/EXPERIMENT_CONTRACT_TEMPLATE.md",
    "docs/research/BASELINE_FREEZE.md",
    "docs/research/EXPERIMENT_REGISTRY.md",
    "docs/research/NEGATIVE_RESULTS.md",
    "docs/research/REPRODUCIBILITY_CHECKLIST.md",
    "docs/research/NEXT_CHANGE_CHECKLIST.md",
]
REGISTRY_FIELDS = [
    "experiment_id",
    "version",
    "date",
    "git_commit",
    "baseline_commit",
    "dataset",
    "protocol",
    "seed",
    "multi_seed_group",
    "baseline_comparator",
    "baseline_variance",
    "stage1_change",
    "stage2_change",
    "hypothesis",
    "status",
    "completed",
    "rank1_all",
    "map_all",
    "minp_all",
    "rank1_indoor",
    "map_indoor",
    "minp_indoor",
    "best_epoch",
    "final_epoch",
    "contract_path",
    "preflight_output_path",
    "gate_output_path",
    "log_path",
    "summary_path",
    "parser_command",
    "parser_version",
    "monitor_status",
    "monitor_log_path",
    "monitor_exit_status",
    "monitor_started_at",
    "monitor_completed_at",
    "monitor_error_detected",
    "monitor_parsed_summary",
    "checkpoint_path",
    "model_best_path",
    "stage1_best_path",
    "checkpoint_hashes",
    "stage1_checkpoint_source",
    "full_state_restore",
    "optimizer_state",
    "scheduler_state",
    "evidence_grade",
    "reviewer",
    "decision",
    "notes",
]


def check_frontmatter(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.S)
    if not match:
        return [f"{path}: missing YAML frontmatter"]
    fm = match.group(1)
    if not re.search(r"^name:\s*\S+", fm, flags=re.M):
        errors.append(f"{path}: missing name")
    if not re.search(r"^description:\s*.+", fm, flags=re.M):
        errors.append(f"{path}: missing description")
    return errors


def main() -> int:
    errors: list[str] = []

    agents = ROOT / "AGENTS.md"
    if not agents.is_file():
        errors.append("AGENTS.md missing")
    elif agents.stat().st_size > 20000:
        errors.append("AGENTS.md too large")

    for skill in SKILLS:
        skill_file = ROOT / ".agents" / "skills" / skill / "SKILL.md"
        if not skill_file.is_file():
            errors.append(f"skill missing: {skill}")
        else:
            errors.extend(check_frontmatter(skill_file))

    for doc in DOCS:
        if not (ROOT / doc).is_file():
            errors.append(f"doc missing: {doc}")

    registry = ROOT / "experiments" / "registry.csv"
    if not registry.is_file():
        errors.append("experiments/registry.csv missing")
    else:
        with registry.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [field for field in REGISTRY_FIELDS if field not in (reader.fieldnames or [])]
            if missing:
                errors.append(f"registry missing fields: {', '.join(missing)}")

    bad_global = False
    global_cfg = Path.home() / ".codex" / "config.toml"
    if global_cfg.is_file():
        text = global_cfg.read_text(encoding="utf-8", errors="replace")
        bad_global = "[mcp_servers.codex_apps]" in text and "enabled = false" in text
    if bad_global:
        errors.append("global Codex config disables codex_apps")

    if errors:
        for error in errors:
            print(f"FAIL {error}")
        return 1
    print("PASS skill setup valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
