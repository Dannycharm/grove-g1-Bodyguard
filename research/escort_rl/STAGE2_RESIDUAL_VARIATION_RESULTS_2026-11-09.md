# Targeted crossing/diagonal variation v3 — executed results

**Decision: strongest residual so far on the fixed suites, but NOT promoted and
NOT ready for PPO under the unchanged safety gate.** Coverage variation improved
pedestrian success from 45/60 to 57/60 while retaining exact Stage 1 behavior.
One geometric intrusion and two client losses remain.

## What changed and what stayed fixed

Added `diagnostics/stage2_residual_variation_v3.yaml` and extended the existing
coverage driver to select predefined training-only parameter strata. Existing v2
selection still works and is regression-tested; old experiment/source snapshots
and weights are intact. The v1 inference/loader/source, teacher, architecture,
features, loss, phase weights, gait, physics, command limits, reward, observations
and gates were not changed. Parent summary hashes are additionally checked where
recorded. There is no new framework or PPO adapter.

The 20k residual parent, SHA-256
`7410ee0ede3e0c69d16dac065928bc9ca7a775915ed5167b7765631c421404de`, was restored
with its genuine Adam moments, Torch RNG and saved local batch RNG. The frozen
selected Stage 1 archive remains unchanged. Training kept the previous per-phase
cap of 8,000 updates, rather than extending the old same-case run without a coverage
change. Final cumulative residual BC is 28,000; residual PPO updates are zero,
base PPO counter remains 5,000.

This is a practical continuation, not a matched same-parent/no-new-data control.
The data expansion and further fitting are both present, so gains cannot be
attributed causally to data alone. Only one residual initialization was tested.

## Predeclared training variation

All eight previous cases, including both head-on cases, were retained. Eight new
cases were selected BEFORE outcomes and without fixed-validation IDs:

- Positive-side crossing: low/high pedestrian peak speed crossed with early/late
  encounter fraction, four new cases.
- Diagonal: both generator sides crossed with shallow/oblique approach angle,
  four new cases.

All ranges are within the existing Stage 2 generator limits. Seeds are in the
training partition and exclude reserved final-test route combinations. Metadata
is only for generation/batching/evaluation, never actor input. Family/side/case
mass remains balanced; head-on keeps 1/3 of batch probability despite only two
retained cases. `cases.json`, `coverage_parameters.csv` and `parameter_coverage.png`
record the actual parameter coverage.

## Development and student-state aggregation

Executed run: `runs/residual_variation_v3_20261109/`.

| Stage | Cumulative residual BC updates | Development successes |
|---|---:|---:|
| Observable teacher | — | 16/16 |
| Incoming v2 residual | 20,000 | 12/16 |
| First new fit | 24,000 | 14/16 |
| Student-only DAgger | 24,000 | 14/16 |
| Second new fit | 28,000 | 16/16 |

The incoming model's four failures and the intermediate model's two failures
were geometric intrusions, not critical physics/supervisor failures. All cases
were kept; no failures were replaced with easier seeds. DAgger executed all 3,087
actions from the student with zero teacher overrides, retaining teacher command
labels on actual visited states, including failed trajectories.

The original aggregate had 3,891 rows. Teacher collection added 3,307, giving
7,198 fitting rows; DAgger added 3,087, giving 10,285. These represent sixteen
distinct family/seed cases with repeated visits, not thousands of independent
scenarios. Final label MSE was 0.004398 on the enlarged aggregate; this is not a
holdout/generalization score. Closed-loop fixed validation is reported below.

## Fixed validation and comparison

Final iteration was predeclared by the budget, not chosen by searching checkpoints
on validation. The SAME original sixty pedestrian and sixty Stage 1 cases were
used; no traces became training labels and no final-test episode was evaluated.

| Metric | V2 coverage | V3 variation |
|---|---:|---:|
| Crossing | 12/20 | **20/20** |
| Head-on | 19/20 | **20/20** |
| Diagonal | 14/20 | **17/20** |
| Pedestrian total | 45/60 (75%) | **57/60 (95%)** |
| Geometric intrusions | 15 | **1** |
| Client losses | 0 | **2** |
| Stage 1 total | 60/60 | **60/60** |
| Stage 1 mean formation error | 0.0563095368 m | **0.0563095368 m** |

No recorded fall, supervisor-failure or physics-failure termination. All **11,892
Stage 1 observation/action pairs across sixty episodes exactly match the selected
original 5k archive**; `stage1_exact_replay.json` records the comparison. This is
exact observed/action replay, not comparison of every hidden integration-state
array or numerical parity with live Grove.

Case changes relative to v2: crossing gained eight successes and lost none; head-on
gained one and lost none; diagonal gained four and lost one. Thus the new model
improves every aggregate family score, but does not dominate v2 on every individual
case. Pedestrian mean formation error rose 0.11400 -> 0.13531 m; this must not be
hidden by the higher success rate.

## Gate decision and remaining failures

Overall/per-family success targets now pass: 95% overall, 100% crossing/head-on,
85% diagonal. Stage 1 retention and critical-failure conditions pass. However,
**the zero-geometric-intrusion gate fails**, so `round_pass=false` and promotion
is NOT approved. This is not one of the two required consecutive passing rounds.
No gate was weakened to accept 95% success.

All three failures are positive-generator-side oblique diagonals:

| Validation seed (evaluator only) | Outcome | Min surface clearance | Terminal client distance |
|---|---|---:|---:|
| 3000037 | Geometric intrusion | -0.0301 m | 2.641 m |
| 3000081 | Client lost | +0.0752 m | 3.338 m |
| 3000001 | Client lost | +0.0607 m | 3.632 m |

Their approach angles are approximately 136.5–145 degrees and pedestrian peak
speeds 0.260–0.302 m/s. These IDs are evidence ONLY, not a proposed demonstration
list. Future data must use NEW training-residue seeds.

Offline teacher queries on the last two seconds suggest two related weaknesses:
insufficient passing maneuver in the intrusion case (student mean lateral command
-0.021 m/s versus teacher -0.255), and insufficient forward/recovery progress in the
client-loss cases. One loss has student forward +0.232 versus teacher +0.397 m/s;
the other has student -0.032 versus teacher +0.151 and a stronger teacher lateral
passing request. These are causal command labels on recorded states, not executed
counterfactual recovery or proof that last-moment recovery remained possible.

Next proposed controlled work is positive-side oblique diagonal TRAINING-only
variation at slower pedestrian speeds, with early/late timing and recovery states,
while retaining current crossing/head-on replay. Do not copy validation trajectories,
raise the distance limit, relax the grace/clearance gate, change the gait or start
PPO. The teacher's prior sixty-case result shows these cases are feasible for the
expert; it does not provide a neural safety guarantee.

## Measured compute and software checks

Eighty full-G1 development episodes: sixteen teacher, sixteen initial student,
thirty-two post-fit student tests and sixteen DAgger episodes. Total active
work: **15,513 transitions / 1,551,300 physics steps / 155,130 gait calls**.
Warmup integration is additional. The initial teacher executed 3,307 actions;
every later action was student-controlled. Diagnostic wall time was **516.56 s**,
of which fitting took **7.86 s**, excluding subsequent fixed-suite validation and
initial model loading/case resolution. The mixed diagnostic rate was **30.03 active
transitions/s**, not a vectorized PPO throughput estimate.

Full regression: **168 passed**, zero failures/skips, fourteen third-party
Matplotlib/pyparsing warnings, 101.96 s. Three added tests verify explicit parameter
cells/head-on retention, invalid-cell rejection and real 20k parent/optimizer/RNG
restoration. The original v2 coverage and exact no-pedestrian tests still pass.
No dependencies, live ROS/DDS, Docker lifecycle actions, commits or pushes were used.

## Useful artifact and limitations

Final residual:
`research/escort_rl/runs/residual_variation_v3_20261109/residual_coverage_1.pt`

SHA-256:
`a8bbd73a5a6aee06f23384822f8b7350e148dfc95fdc22f6cdf863b431ff79da`.

All residual Adam step counters are exactly 28,000. Preserve adjacent manifest,
continuation/source/config snapshots, parent archives, RNG/optimizer state,
dataset hashes, parameter coverage exports, fixed-suite traces and gate/replay
records. Architecture stays frozen Stage 1 + mask-gated 51-input residual with an
external 40-vector. These are NOT SB3 PPO ZIP resumes or a promoted policy.

The experiment retains the conservative client/gait limits and simulated-pose
observation condition. People are scripted nonreactive proxies; overlap is geometric,
not physical person contact. Live Grove numerical dynamics parity and real-person
safety remain unverified. Fixed validation has been reused for engineering decisions;
untouched final-test evaluation and additional policy seeds remain future work.
