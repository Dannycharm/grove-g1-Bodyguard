# Intervention-balanced imitation v3 review

Reviewed `runs/stage2_distill_v3_yY56tc/weighted/`. Two other recent runs,
`stage2_distill_v3_d4k8UE` and `stage2_distill_v3_rSmP7K`, have identical actor
hashes and results. They are deterministic reruns, not independent training seeds.
No further fitting, simulator rollout or PPO was launched in this review. Teacher
queries below are offline labels on saved observations, not executed counterfactuals.

## Results

| Fixed suite | Previous unweighted fit | Weighted v3 |
|---|---:|---:|
| Crossing | 0/20 | 0/20 |
| Diagonal | 0/20 | 0/20 |
| Head-on | 20/20 | 14/20 |
| Pedestrian total | 20/60 | 14/60 |
| Geometric intrusions | 40 | 41 |
| Stage 1 success | 60/60 | 60/60 |
| Stage 1 mean formation error | 0.08030 m | 0.17805 m |

V3 also recorded five recovery timeouts. No fall/supervisor/physics failure was
recorded in the fixed-suite outcome counts. Stage 1 completed episodes but failed
its mean formation gate (limit 0.10631 m). All failures/gates remain enforced.
`prepared.zip` was correctly not saved. The diagnostic actor is retained, with
5,000 inherited Stage 1 PPO transitions, zero new PPO transitions and 3,000
supervised updates (selected update 2,700).

## What the intervention balancing changed

Balanced teacher-state holdout MSE improved 0.09574 -> 0.01723. Intervention-only
holdout errors improved:

| Encounter | Before fitting | Selected fit |
|---|---:|---:|
| Crossing | 0.50139 | 0.04589 |
| Diagonal | 0.55608 | 0.07268 |
| Head-on | 0.06751 | 0.04768 |

This is evidence that weighting changes what the model fits, but it is not evidence
of task completion. Ordinary-following errors rose, and the closed-loop results
show a preservation cost. Do not equate a lower imitation objective with a better
escort controller or keep increasing the avoidance weight blindly.

Querying the same causal teacher on saved student-visited validation states gives
intervention-state MSE 0.26485 crossing, 0.31692 diagonal and 0.08941 head-on.
The first two are approximately 5.8x and 4.4x their teacher-state holdout errors.
These are different state distributions and teacher histories, not a randomized
causal experiment. The gap supports testing student-state dataset aggregation;
rare-event weighting alone did not solve closed-loop transfer.

In the final two seconds of crossing/diagonal episodes, student mean forward
commands were +0.251/+0.260 m/s; measured forward velocities were +0.189/+0.199 m/s.
The queried teacher instead requested mean forward commands -0.250/-0.273 m/s
and mean absolute lateral commands 0.384/0.421 m/s; the student's mean absolute
lateral commands were only 0.031/0.027 m/s. These late teacher queries do not prove
recovery remained possible, but show that the actor still failed to produce the
teacher's braking/avoidance behavior. The gait was executing continuing forward
commands; this is not evidence of a refusal to stop under zero commands.

## Next bounded experiment

Do not run PPO or repeat the identical weighted fit. Collect one DAgger-style
batch of 18 training/development episodes (three per family), executing this
student deterministically while querying causal teacher command labels at each
visited state. Default beta=0 means zero teacher overrides; failures terminate
normally and their labels remain explicitly student-state labels, not successful
teacher returns. The collector records execution ownership and actual physics/gait
counts. Do not use any fixed validation traces as training data.

Review collection, then fit the aggregate of the original demonstrations and this
new development batch from the same Stage 1 parent, preserving the original
source-wise holdout and Stage 1 replay. This is a test of covariate-shift mitigation,
not a guaranteed cure. Gate both fixed suites before considering critic calibration
or PPO. Do not broaden architecture/reward changes in this comparison.

## Diagnostic actor

`runs/stage2_distill_v3_yY56tc/weighted/candidate_actor.npz`

SHA-256 `70b888fef2a64417681a33e0305627753640ae5b3c9a980fe70c5c0567a4a000`.
Its checked `candidate.json` is required. This is actor-only diagnostic data, not a
PPO resume archive. All original weights remain unchanged. Final-test data remains
untouched and full numerical live-Grove parity remains unverified.
