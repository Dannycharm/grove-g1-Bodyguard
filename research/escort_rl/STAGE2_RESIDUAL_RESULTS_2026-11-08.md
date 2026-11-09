# Frozen-base residual avoidance: executed diagnostic results

**Decision: retain this architecture for further controlled imitation diagnostics;
DO NOT start PPO or promote Stage 2.** The residual learned both development
maneuvers and improved unseen crossing/diagonal performance, while exactly preserving
Stage 1. Generalization, particularly head-on encounters, remains insufficient.

## What was implemented

- `diagnostics/stage2_residual.py`: frozen selected Stage 1 base; exactly validity-
  gated residual; causal internal risk features; observable-history supervision;
  clipping-aware loss; progressive BC/DAgger; strict checkpoint/evaluation handling.
- `diagnostics/stage2_residual_v1.yaml`: model, loss, phase sampling, independent
  seeds, two development cases and the bounded 3×4,000-update budget.
- `diagnostics/evaluate_observable_teacher.py`: separate fixed-suite revalidation
  of the teacher variant, with no training-data export.
- `diagnostics/plot_residual_diagnostic.py`: read-only development plot utility.
- `tests/test_stage2_residual.py`: nine contract/provenance/numerical regression tests.
- `STAGE2_RESIDUAL_V1.md`: architecture, exact input schema/commands and limitations.

This adapts existing Stage2Env, frozen gait, scenario generator and evaluation/gates.
It does not replace them or modify runtime/controller assets, rewards or observations.
All base policy parameters are frozen, including actor/critic/log sigma. The new
branch has 23,555 trainable parameters and 51 INTERNAL features; external inputs
remain the same causal 40-vector. Base pedestrian inputs are always zeroed. No-ped
rows bypass residual computation entirely, guaranteeing exact applied base actions.

## Teacher audit before training

Audit used the existing training-partition teacher data (not fixed validation).
For 5,394 pedestrian observations:

- Original sequential teacher replay: max absolute label discrepancy **0.0**.
- Holding observation fixed and changing five plausible stored-velocity values
  changed actions by >0.05 normalized L2 in **59.58%** of stress probes; maximum
  difference **1.898**. This does not prove this many real labels were inconsistent.
- Zero conflicting EXACT duplicate observations were found. No near-neighbour
  consistency/convergence guarantee is claimed.
- Reconstructing stored velocity from the already-observed previous action changed
  only **one of 5,394** original labels by >0.05 L2 (maximum 0.375).

The observable teacher mode is explicitly versioned, not silently substituted.
After training diagnostics it was independently revalidated: **60/60** pedestrian
successes, twenty each crossing/head-on/diagonal, zero geometric intrusions or
critical failures. Its mean formation error was 0.119289 m. This removes a known
hidden-history dependency while retaining measured expert feasibility. It does not
identify the sole cause of earlier failed imitation.

## Bounded overfit and iterative DAgger

Executed run: `runs/residual_overfit_v1b_20261108/`.
Cases were predeclared training-partition crossing seed 904000 and diagonal seed
904004, excluding reserved final-test combinations.

| Stage | Cumulative residual BC updates | Crossing | Diagonal |
|---|---:|---|---|
| Zero residual / selected frozen base | 0 | Intrusion | Intrusion |
| BC on observable teacher trajectories | 4,000 | Success | Success |
| First student-only DAgger aggregation | 8,000 | Success | Success |
| Second student-only DAgger aggregation | 12,000 | Success | Success |

Both DAgger collections executed the student on every step, with zero teacher
execution overrides. Labels were queried on actual visited states. Dataset sizes
were 311 -> 622 -> 933 rows, all from the SAME two development cases. These are
not hundreds of independent scenarios. Final training-label action MSE by iteration:
0.002829, 0.002263, 0.002018. Final action-bound fraction on fitting rows was zero.
No independent supervised holdout is claimed for an explicit overfit test; held-out
closed-loop validation below is the generalization test.

The completed diagnostic executed fourteen full-G1 episodes: two teacher, two
initial-base, six post-fit student tests and four student-only collection episodes.
That is **1,998 active transitions / 199,800 physics steps / 19,980 gait calls**.
Initial teacher execution accounts for 311 steps; all later executed actions are
student commands. Warmup physics is additional and not included in active counters.

An initial run `runs/residual_overfit_20261108/` solved both cases after 4k updates
but stopped after checkpoint saving due to a relative-path registry bug. That bug
was fixed and tested; the complete experiment restarted in a FRESH directory with
the same seeds, preserving old artifacts/source snapshot. The interrupted run adds
4k supervised updates and 754 active transitions to total research compute, not to
this final model's update count. No original archive was overwritten or strict hash
check bypassed. Full diagnostic wall-clock throughput was not recorded, so none
is inferred from older bare-runner/PPO benchmark numbers.

## Fixed validation: final predeclared 12k residual

These cases were not collected as fitting data, and final-test data was not used.
The final iteration was selected by the predeclared budget, not by searching
validation performance across residual checkpoints.

| Fixed suite | Frozen Stage 1-derived baseline | Final residual |
|---|---:|---:|
| Crossing | 0/20 | **12/20** |
| Diagonal | 0/20 | **17/20** |
| Head-on | 20/20 | **9/20** |
| Pedestrian total | 20/60 | **38/60 (63.33%)** |
| Geometric intrusions | 40 | **22** |
| Stage 1 total | 60/60 | **60/60** |
| Stage 1 mean formation error | 0.0563095368 m | **0.0563095368 m** |

All twenty-two pedestrian failures were geometric proxy intrusions: eight crossing,
three diagonal and eleven head-on. No recorded fall/supervisor/physics failure.
Pedestrian mean formation error was 0.103723 m; that low mean does not excuse
intrusions. Contact semantics remain a swept pelvis-disc/pedestrian-disc surrogate,
not physical person contacts or a real-human safety claim.

Beyond matching aggregate Stage 1 metrics, **all 11,892 observations and actions
across the sixty original Stage 1 validation traces match the selected 5k policy
EXACTLY**. Evidence: `stage1_exact_replay.json`. This is stronger than a success-rate
tie and demonstrates the intended no-pedestrian preservation in full dynamics.
It is not a claim that every hidden MuJoCo integration-state array was compared.

The unchanged gate rejects this residual for overall/per-family pedestrian success
and geometric intrusions. Stage 1 retention passes. Diagonal success exceeds its
80% target, but crossing/head-on/overall do not; zero-intrusion and two-round gates
also remain unmet. `validation_decision.json` records the decision. No promotion
or guarded PPO continuation is authorized by these results.

## What these experiments show, and what they do not

The combination of a frozen base, residual targets, observable teacher state,
internal risk features and a longer progressive overfit budget produced real
crossing/diagonal successes that earlier whole-MLP fits did not achieve. No single
factor was isolated by ablation; do not attribute the full gain solely to residuals
or label the earlier architecture inherently incapable.

No-pedestrian forgetting is structurally removed for deterministic actions and
verified in full dynamics. That does NOT preserve head-on behavior: a pedestrian
is valid there, so the residual is active and may damage the previously successful
base. This run never trained head-on or broad approach-side/speed variations.
Only two development encounters are insufficient supervision for general safety.

The next controlled milestone should broaden TRAINING-ONLY encounter coverage,
including head-on/safe-following examples and both approach sides, then continue
residual-only BC/DAgger with the base frozen and both gates unchanged. Do not turn
fixed validation failures into training trajectories or add handcrafted exceptions
to pass particular validation cases. Additional datasets/case support must use an
explicit versioned transition from these recorded residual artifacts; the current
v1 CLI is deliberately the two-family overfit diagnostic, not a general trainer.

PPO remains unimplemented for the residual. Only after reliable imitation should
we specify its stochastic residual law, critic, optimizer/RNG transition and KL
controls while preserving the base freeze. These `.pt` files are not SB3 `.zip`
resume archives. Broad training and Stage 3 are outside this milestone.

## Artifacts and integrity

Final residual:
`research/escort_rl/runs/residual_overfit_v1b_20261108/residual_2.pt`

SHA-256:
`2ff02f723de84308674a5b4d9ff656e4b7bac5331b1cf530e00a55f13f08d56c`.

Frozen selected base SHA-256:
`470fef05cc1add9ceb1557d863c4b0d5506504d0f915790a5e5271378d323c8d`.

The base retains **5,000 inherited PPO transitions**. The residual has **12,000
supervised updates, zero PPO updates**. Preserve the adjacent manifest, resolved
config, original source snapshots, teacher audit, labeled datasets, fitting records,
closed-loop traces, RNG/optimizer state and fixed-suite reports. The registry is
updated at `docs/bodyguard/MODEL_WEIGHTS.md`; entries do not back up binary files.

Development plot: `development_comparison.png`, visually inspected. Bottom-row base
commands are the base queried on residual-visited states, not actually executed
base-only commands. No plot or dataset is loaded as privileged actor input.

Final full regression: **159 passed, zero failed/skipped**, 14 third-party warnings,
100.83 seconds. Real-checkpoint unit tests cover exact inactive gating, frozen base,
clipping-aware gradients, risk features, teacher determinism and save/load integrity.
New full-dynamics experiments are explicitly described above. Live ROS/DDS was not
launched or changed; no system dependencies, commits or pushes were performed.
Full numerical live-Grove parity and real-person safety remain unverified.
