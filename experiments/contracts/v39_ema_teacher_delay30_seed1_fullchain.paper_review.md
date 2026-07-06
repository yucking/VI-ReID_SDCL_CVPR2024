# sdcl-paper-reviewer: v39_ema_teacher_delay30_seed1_fullchain

Major concerns:
This is a corrective diagnostic for v39, not a new paper method. Delaying EMA teacher may simply restore baseline behavior rather than provide a contribution.

Minor concerns:
Only seed 1 is being launched first. Baseline variance remains unmeasured in this repository.

Missing experiments:
If Stage-1 recovers, compare `EMA_TEACHER_START=30`, `ENABLE_EMA_TEACHER=0`, and the failed `EMA_TEACHER_START=0`; then repeat seeds 1, 2, and 3.

Protocol/fairness risks:
Evaluation protocol is unchanged. The log directory is changed only to prevent overwriting the failed run.

Novelty risk:
High. EMA teacher delay is an engineering correction and cannot be claimed as a contribution without mechanism evidence and ablations.

Evidence grade:
Raw-log diagnosis plus preflight and syntax checks. No completed fixed-run metrics yet.

Minimum publishable next evidence:
Recovered Stage-1 cluster/outlier trajectory, final metrics competitive with the trusted reproduction, multi-seed stability, and ablation showing EMA teacher after handoff adds value beyond student-only clustering.

Decision: borderline for diagnostic execution, reject as a paper claim.

Evidence that would change the decision:
Consistent multi-seed improvement over repeated baseline plus mechanism evidence that delayed EMA teacher improves Stage-2 pseudo-label stability without harming Stage-1.
