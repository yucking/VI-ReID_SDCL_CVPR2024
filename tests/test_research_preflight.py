from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research_preflight.py"


def write_contract(path: Path, *, experiment_id: str = "exp_ok") -> None:
    path.write_text(
        f"""# Contract

## Experiment Name
{experiment_id}

## Date
2026-07-06

## Baseline Commit
abc1234

## Baseline Configuration
v4base fixed config

## Problem
Stage-1 feature bottleneck evidence.

## Evidence
logs/example.txt shows the failure.

## Gate Decision
PASS

## Gate Output Path
experiments/contracts/exp_ok.gate.txt

## Hypothesis
One falsifiable hypothesis that can be falsified by no cluster coverage gain.

## Mechanism
One mechanism.

## Changed Files
sdcl_sysu_v4base.py

## Independent Variable
one primary variable: bridge memory enabled

## Fixed Variables
dataset, protocol, seed, cmlabel, epochs fixed

## Intermediate Metric
cluster coverage and outlier rate should improve first.

## Falsification Result
reject if cluster coverage does not improve or final mINP fails.

## Main Metrics
all-search and indoor metrics

## Diagnostic Metrics
cluster count and outliers

## Dataset And Protocol
SYSU all-search and indoor-search

## Seeds
Diagnostic seed: 1
Full seed list: 1,2,3
Repeated-baseline comparison source: baseline variance table

## Budget
one smoke test and planned full run

## Baseline Variance
baseline variance from repeated v4base seeds is recorded.

## Training Command
bash train.sh

## Monitor Command
bash scripts/watch_experiment_screen.sh

## Monitor Log Path
logs/test_exp/monitor.log

## Log Directory
{{path_log_dir}}

## Summary Path
logs/test_exp/summary.md

## Parser Command
python scripts/parse_sysu_log.py logs/test_exp/log.txt > logs/test_exp/summary.md

## Registry Row
exp_ok is registered as planned.

## Checkpoint Plan
checkpoint.pth.tar, model_best.pth.tar, and 20model_best.pth.tar will be checked.

## Environment
Python, CUDA, PyTorch, GPU, and Git state captured.

## Determinism
Python/NumPy/Torch/CUDA seeds, worker seed, sampler seed, cudnn and TF32 recorded.

## Smoke Test
single batch

## Success Criteria
multi-seed improvement beyond baseline variance

## Failure Criteria
no diagnostic improvement

## Strongest Opposing Explanation
training variance explains any gain.

## Disproof Plan
falsify with ablation and multi-seed comparison.

## Ablation
remove the single variable

## Rollback
revert experiment branch

## Reviewer Conclusion
pending independent review
""",
        encoding="utf-8",
    )


def write_registry(path: Path, experiment_id: str) -> None:
    fields = [
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {field: "x" for field in fields}
            | {
                "experiment_id": experiment_id,
                "contract_path": "contract.md",
                "log_path": "logs/test_exp/log.txt",
                "summary_path": "logs/test_exp/summary.md",
                "monitor_log_path": "logs/test_exp/monitor.log",
                "checkpoint_path": "logs/test_exp/checkpoint.pth.tar",
            }
        )


def run_preflight(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--contract",
            str(tmp_path / "contract.md"),
            "--registry",
            str(tmp_path / "registry.csv"),
            "--experiment-id",
            "exp_ok",
            "--log-dir",
            str(tmp_path / "logs" / "exp_ok"),
            "--train-script",
            str(tmp_path / "train.sh"),
            "--allow-no-git",
            *extra,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def test_preflight_passes_for_complete_contract(tmp_path: Path) -> None:
    write_contract(tmp_path / "contract.md")
    text = (tmp_path / "contract.md").read_text(encoding="utf-8")
    (tmp_path / "contract.md").write_text(
        text.replace("{path_log_dir}", str(tmp_path / "logs" / "exp_ok")),
        encoding="utf-8",
    )
    write_registry(tmp_path / "registry.csv", "exp_ok")
    (tmp_path / "train.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    proc = run_preflight(tmp_path)

    assert proc.returncode == 0, proc.stdout
    assert proc.stdout.rstrip().endswith("PASS")


def test_preflight_fails_when_contract_has_unknown(tmp_path: Path) -> None:
    write_contract(tmp_path / "contract.md")
    text = (tmp_path / "contract.md").read_text(encoding="utf-8")
    (tmp_path / "contract.md").write_text(
        text.replace("{path_log_dir}", str(tmp_path / "logs" / "exp_ok")).replace("One mechanism.", "UNKNOWN"),
        encoding="utf-8",
    )
    write_registry(tmp_path / "registry.csv", "exp_ok")
    (tmp_path / "train.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    proc = run_preflight(tmp_path)

    assert proc.returncode == 1
    assert "FAIL contract:Mechanism" in proc.stdout


def test_preflight_fails_for_unregistered_experiment(tmp_path: Path) -> None:
    write_contract(tmp_path / "contract.md")
    text = (tmp_path / "contract.md").read_text(encoding="utf-8")
    (tmp_path / "contract.md").write_text(
        text.replace("{path_log_dir}", str(tmp_path / "logs" / "exp_ok")),
        encoding="utf-8",
    )
    write_registry(tmp_path / "registry.csv", "other_exp")
    (tmp_path / "train.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    proc = run_preflight(tmp_path)

    assert proc.returncode == 1
    assert "FAIL registry_experiment_id" in proc.stdout
