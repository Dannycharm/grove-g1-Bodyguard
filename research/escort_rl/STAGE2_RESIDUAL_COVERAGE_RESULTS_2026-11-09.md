# Broader residual coverage v2 — bounded continuation results

**Decision: useful head-on/generalization improvement, but Stage 2 gates still fail.
No PPO or promotion.** Completed the bounded continuation and both fixed suites;
there is no need to rerun these experiments to obtain this result.

## Implementation and continuation integrity

New files: `diagnostics/stage2_residual_coverage.py`, its versioned YAML
`stage2_residual_coverage_v2.yaml`, six focused tests and the coverage guide.
Existing v1 inference/loader/teacher/environment/gait/rewards/scenarios were not edited.

The driver loaded the pinned 12k residual through its original strict loader,
restored residual weights, genuine Adam moments, Torch RNG and local NumPy batch RNG,
and verified its three label sources. Missing optimizer/RNG is an explicit failure.
All base parameters remain frozen. This was NOT fresh initialization or continuation
from the failed whole-MLP diagnostic. New training lineage/config/source/data hashes
are stored in continuation metadata while v1 inference format remains unchanged.

Family, generator-side and case mass are now balanced: each family 1/3, each side
within that family 1/6 overall, and equal mass per case within a side. Existing phase
weights prioritize avoidance/anticipation/recovery/stopping inside each case.
This changes the supervised sampling condition; it does not modify the PPO scenario
mixture, reward or external 40-value observation. No metadata becomes actor input.

## Executed development sequence

Run: `runs/residual_coverage_v2_20261109/`.
The predeclared eight training-only cases retain crossing 904000 and diagonal 904004,
and add one example per side for crossing/head-on/diagonal. Case selection excludes
reserved final-test route combinations and precedes observing rollout outcomes.
The exact identities are in `cases.json` and the guide. No fixed validation case is
used for fitting or student-state labels.

| Stage | Residual cumulative supervised updates | Development success |
|---|---:|---:|
| Observable teacher, before updates | — | 8/8 |
| V1 residual parent | 12,000 | 7/8 |
| First continuation fit | 16,000 | 8/8 |
| Student-only DAgger collection | 16,000 | 8/8 |
| Second continuation fit | 20,000 | 8/8 |

Initial failure was one head-on intrusion. The DAgger collector executed student
actions on ALL 1,476 steps; there were zero teacher overrides. Parent labels (933
rows) plus expanded teacher data (1,482 rows) formed 2,415 fitting rows. Adding the
new DAgger data yielded 3,891 rows, all from the same eight distinct cases. These
are repeated state visits, not thousands of independent scenarios.

Actual optimizer state confirms every residual Adam parameter step reaches 20,000.
The local batch RNG fingerprint at the second block exactly continues the first
block's final state. The inherited base PPO counter is unchanged at 5,000;
residual PPO updates remain ZERO. New supervised updates: 8,000.

## Fixed held-out validation

The final 20k checkpoint was fixed by the prospective budget, not selected by trying
multiple residual checkpoints on validation. Both existing suites and gates remain
unchanged. Final-test cases were not evaluated.

| Fixed suite | V1 residual | Broader v2 |
|---|---:|---:|
| Crossing | 12/20 | **12/20** |
| Head-on | 9/20 | **19/20** |
| Diagonal | 17/20 | **14/20** |
| Pedestrian total | 38/60 (63.3%) | **45/60 (75%)** |
| Geometric intrusions | 22 | **15** |
| Stage 1 | 60/60 | **60/60** |
| Stage 1 mean formation error | 0.0563095368 m | **0.0563095368 m** |

All fifteen failures were geometric intrusions: eight crossing, six diagonal, one
head-on. No recorded fall/supervisor/physics-failure termination. These proxies are
noncolliding scripted actors; intrusion is NOT a physical person contact.

All **11,892 observation/action pairs in all sixty Stage 1 traces exactly match the
selected original 5k actor**. This proves preservation for these observed/action
trajectories, not numerical equivalence of every hidden integration-state array or
parity with live Grove. Evidence: `stage1_exact_replay.json`.

Pedestrian mean formation error is 0.11400 m, slightly above v1's 0.10372 m. A low
formation error does not excuse geometric intrusion. The unchanged gate fails for
overall success (75% <80%), crossing/diagonal success and intrusions. Head-on exceeds
its 80% success target but one intrusion still violates the safety gate.
`validation_decision.json` records the complete decision. No promotion is approved.

## Failure structure and what changed

Case churn versus v1: crossing gained two successes and lost two (same net count);
head-on gained ten and lost none; diagonal gained two and lost five. Thus a higher
overall score must not hide diagonal regression.

| Family / generator side | Success |
|---|---:|
| Crossing -1 | 8/8 |
| Crossing +1 | 4/12 |
| Head-on -1 | 8/8 |
| Head-on +1 | 11/12 |
| Diagonal -1 | 6/8 |
| Diagonal +1 | 8/12 |

All crossing failures occur on the positive generator side. This supports broadening
training-only variation on that side; it does not authorize copying failed validation
case identities or labels into training. One new positive-side example solved at
training time is insufficient generalization across the whole geometry distribution.

Offline teacher queries on the final two seconds of failed student traces suggest
continued under-correction. Crossing mean forward command is +0.340 m/s versus
queried teacher +0.270; mean absolute lateral command 0.227 versus teacher 0.426.
Diagonal mean forward command is +0.338 versus teacher -0.022; lateral magnitude
0.284 versus teacher 0.366. These are suggestions at already-failed visited states,
not executed recovery experiments or proof that recovery was still possible.

The most supported next milestone is additional TRAINING-ONLY crossing/diagonal
variation across both sides, approach speeds/timing and route headings, retaining
head-on examples/anchors and balanced sampling. Continue only the residual and keep
the base frozen. Evaluate after a bounded round; do not blindly extend a large budget,
change the gait/reward, or claim that a larger MLP is necessary. This experiment
changes coverage AND supervised sampling/optimization exposure; it does not isolate
which of those caused each improvement or regression. Only one residual initialization
has been tested; no training-seed robustness claim is made.

## Actual compute and testing

The development experiment executed forty full-G1 episodes across teacher collection,
initial evaluation, two post-fit evaluations and one student-only DAgger collection:
**7,276 active transitions, 727,600 physics steps, 72,760 gait calls**. Warmup physics
is additional. Wall time was **247.44 s**, including fit/collection/reset/logging,
excluding subsequent fixed-suite evaluations and initial loading/case resolution.
Fitting itself took **7.60 s**. The mixed diagnostic rate is **29.41 active transitions/s**;
it is NOT a vectorized PPO throughput benchmark or an estimate of optimizer training time.

Full regression: **165 passed**, zero failed/skipped, fourteen third-party deprecation
warnings, 100.81 seconds. The six added tests exercise case coverage, sampling mass,
real-parent weight/Adam/RNG continuation, residual-only updates, exact inactive gating,
missing-state rejection and invalid configurations. No new dependencies, live ROS/DDS,
Docker lifecycle action, original controller edits, commits or pushes occurred.

## Useful artifact

Final residual:
`research/escort_rl/runs/residual_coverage_v2_20261109/residual_coverage_1.pt`

SHA-256:
`7410ee0ede3e0c69d16dac065928bc9ca7a775915ed5167b7765631c421404de`.

Preserve the adjacent `.manifest.json`, continuation manifest/source/config snapshots,
parent archive/sidecars, dataset hashes, fitting records, fixed-suite results and exact
replay/gate records. Architecture remains frozen selected Stage 1 + mask-gated residual;
external observation remains 40 values. This is not an SB3 PPO archive or a promoted
policy. All original weights remain unchanged and the registry retains v1 for comparison.
Live Grove full numerical dynamics parity and real-person safety remain unverified.
