# Stage 2 continuation after critic preparation

Reviewed existing artifacts in `runs/stage2_after_critic_3OEFLy/pilot_5k`;
no additional training or simulation was launched for this review.

## Results

5,000 new PPO transitions completed: 15,000 cumulative, of which 10,000 are
Stage 2. The earlier critic preparation separately collected 10,680 transitions;
these are not included in the PPO counter. This continuation executed 500,000
physics steps and 50,000 gait calls, at 58.39 training transitions/s over 85.63 s
(excludes final evaluation wall time).

| Fixed validation | Original Stage 2 pilot | After critic + 5k |
|---|---:|---:|
| Crossing | 0/20 | 0/20 |
| Head-on | 20/20 | 20/20 |
| Diagonal | 0/20 | 0/20 |
| Pedestrian total | 20/60 | 20/60 |
| Geometric intrusions | 40 | 40 |
| Stage 1 | 60/60 | 60/60 |
| Stage 1 mean formation error | 0.05822 m | 0.06067 m |

All fixed-suite failures were geometric pedestrian intrusions, not physical
person contacts. No fall/supervisor failure was recorded in either suite.
Stage 1 formation increased about 2.45 mm, within its configured gate.
Pedestrian formation error of 0.05406 m does not establish safe accompaniment:
crossing/diagonal trajectories terminate on intrusion while still close to formation.
Stage 2 promotion failed for overall/family success and geometric intrusions.

## Learning diagnostics

The two logged optimization summaries show explained variance 0.938 and 0.931,
versus roughly 0.01--0.02 in the original pilot. Value losses were 62.60/70.32;
KL was 0.00312/0.00783. These are different on-policy rollout batches, not a
paired critic-error benchmark. They nevertheless support that value calibration
did not immediately collapse. Approximate policy sigma stayed 0.049995.

Completed training episodes: 19 successes, 16 geometric intrusions. All eleven
crossing and five diagonal episodes failed; the other four families all succeeded
(7 straight, 5 turns, 3 stops, 4 head-on). Worker scenario logs contain 37 distinct
seeds across 37 starts (19/18 per worker), including two active incomplete episodes.
Uniqueness alone is not a complete RNG continuation proof, but no repeated seeds
appear in this chunk.

## Interpretation and next experiment

Better critic predictions alone have not produced avoidance in this bounded run.
The policy preserves its following behavior, but neither training nor validation
contains a completed successful crossing/diagonal maneuver. With normalized sigma
about 0.05, velocity-command noise has standard deviation about 0.03 m/s in x/y
and 0.045 rad/s in yaw before clipping. Such perturbations may fail to discover
a sustained early yield. This is a plausible discovery bottleneck, not proof of
an exploration-only cause. No equal-budget unprepared-critic continuation was run,
so this is not a controlled causal estimate of critic preparation's effect.

Do not launch a long continuation or repeat critic fitting. Recommended next
milestone: collect and validate causal pedestrian-aware diagnostic demonstrations,
starting with the existing yielding controller for crossings. Resolve diagonal
feasibility with development-only cases before claiming a teacher works there.
Only if these demonstrations are reliable, propose a small separately versioned
pedestrian-aware imitation bootstrap with Stage 1 replay and actor-preservation
checks, followed by bounded PPO and both unchanged validation suites. This would
change the training method and must be reported explicitly; it has not been done.
An exploration-only alternative can remain a control, but simply increasing
independent action noise is not established as sufficient.

Keep the original reward as the control: the previous matched audit found tested
successful maneuvers had better discounted returns than collisions, though early
avoidance had delayed credit. These results neither prove the reward ideal nor
justify declaring it the sole problem. No evidence here implicates MLP capacity
or low-level gait changes as a required fix. Live Grove dynamics parity and robust
diagonal avoidance remain open. Final-test scenarios remain untouched.

## Checkpoint

`runs/stage2_after_critic_3OEFLy/pilot_5k/final.zip`

SHA-256: `c456882468a00a3dedbf19e192375a8eee7b54d02417c6e07918f98f81d7f8ac`.
Retain as a diagnostic continuation, not a promoted pedestrian-aware policy.
