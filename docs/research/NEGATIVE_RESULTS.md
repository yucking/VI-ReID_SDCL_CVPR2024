# Negative Results

## Parser-backed Failures

| version / idea | evidence | failure type | conclusion |
| --- | --- | --- | --- |
| v31 GPRD-PLG | parser on `logs/0703/log_sysu_v31_gprd_plg_seed1_stage2only.txt` gives best 64.18/62.67/50.06 and 70.28/76.03/72.66 | below trusted reproduction | Reject. |
| v34 IRMT | parser gives best 63.90/62.62/50.22 and 70.23/75.86/72.65 | restored but did not improve | Reject threshold-only continuation. |
| v38 prototype bridge | parser gives best 63.27/61.57/48.85 and 69.73/75.43/71.73 | Stage-1 mechanism did not improve final result | Reject current implementation. |

## Historical Notes Only

| version / idea | evidence | failure type | conclusion |
| --- | --- | --- | --- |
| XMB matched sampler | `SYSU_EXPERIMENTS.md` names v19 logs and says epoch 35 dropped after XMB started | mechanism or distribution shift | Do not continue matched-sampler edits without new diagnostics. |
| XMB legacy queue | `SYSU_EXPERIMENTS.md` reports best all mAP around 62.44 and below seed2 baseline | mechanism failure | Queue signal too weak/noisy. |
| CM-triplet / `cm_w` / `cms` | `SYSU_EXPERIMENTS.md` says v18 and later branches did not improve mINP | evidence from notes only | Do not resume weight sweeps. |
| TBA / BPT / SOT label topology | `SYSU_EXPERIMENTS.md` says aggressive or conservative variants did not produce gain | mechanism failure | Do not repeat label-flip topology rewrites without new evidence. |
| Late EMA | `SYSU_EXPERIMENTS.md` reports seed2 best all mAP 61.70, indoor 75.30/71.80 | weak trajectory | Smoothing cannot rescue weak Stage-1/early Stage-2. |
| TCR reliability weighting | `SYSU_EXPERIMENTS.md` reports epoch 34 below Late EMA weak rerun | supervision reduction | Do not sweep TCR variants. |
| v33 CMGC | notes report best 61.25/59.02/45.63 and 66.37/72.27/68.60 | wrong objective | Reject RGB-only joint identity direction until raw parser summary is regenerated. |
| v37 handoff selector | user and notes identify it as diagnostic selector only | no new representation mechanism | Not a breakthrough experiment. |

## Repeated Stage-2 Directions To Avoid

- More auxiliary losses without mechanism proof.
- More threshold or weight sweeps around CRA/IRMT/PQLC.
- More sampler or queue changes without distribution diagnostics.
- More checkpoint selector changes as if they improve representation.

## Still Needs Multi-seed Verification

- Clean v4base baseline variance on this server.
- v35 full-state handoff equivalence.
- Any future Stage-1 representation change.
