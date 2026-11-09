# Residual PPO 5k: first passing validation round

Run: `runs/residual_ppo_recovered_9htyqV/pilot`.

| Metric | Prepared imitation initial | Residual PPO 5k |
|---|---:|---:|
| Crossing success | 20/20 | 20/20 |
| Head-on success | 20/20 | 20/20 |
| Diagonal success | 17/20 | 20/20 |
| All pedestrian success | 57/60 | 60/60 |
| Geometric intrusions | 1 | 0 |
| Minimum surface clearance | −0.03012 m | +0.05524 m |
| Mean pedestrian-suite formation error | 0.13531 m | 0.13269 m |
| Stage 1 success | 60/60 | 60/60 |
| Stage 1 mean formation error | 0.0563095368 m | 0.0563095368 m |

Read-only inspection loaded actual initial/final archives with strict configuration,
runtime/source and archive-hash checks. Saved final counter is 5,000. Residual
parameters changed; frozen base parameters are identical. The complete Stage 1
summary objects match exactly. Thus the small PPO continuation improved these
three previously failed diagonal validation cases while preserving the tested
client-only behavior. This does not establish which particular action change
caused improvement, generalization to new routes, or performance across seeds.

All 23 completed training episodes succeeded. Collection executed 500,000 physics
steps and 50,000 gait calls. Two hundred optimizer steps took 0.354 s; measured
training wall time was 156.38 s (31.97 transitions/s), excluding preparation and
validation. Runtime is overwhelmingly collection/associated overhead rather than
optimizer compute; the available timing does not separately resolve physics,
inference and logging. Action saturation was 0.1%; latent clipping was zero.

Reward remains Stage 2 tracking/clearance v1. The successful change was a bounded,
small-noise, small-learning-rate residual-only PPO continuation of a strong
imitation policy, with a calibrated critic and frozen Stage 1 base. It was not PPO
from scratch, a larger network, new gait, changed physics, or reward retuning.

The final archive is now the leading research candidate, registered with hash and
companions in `docs/bodyguard/MODEL_WEIGHTS.md`. Existing gates pass this round,
but require two consecutive passing full rounds. No promotion is granted.
Minimum clearance is only 5.5 cm under the geometric proxy definition, so these
results are not a real-person safety claim.

Recommended next step: preserve this candidate and verify a second full paired
round in a fresh process. A repeat of a deterministic fixed suite checks
reproducibility, not independent statistical generalization; label it accordingly.
Additional training-only development routes or another bounded continuation can
then test robustness, with explicit authorization/configuration for any increased
training budget. Do not launch 100k, change rewards or unlock the Stage 1 actor
on the strength of this single round. Final-test data remains untouched and live
Grove numerical dynamics parity remains unverified.

## Second round: fresh-process confirmation

Operator output: `runs/residual_ppo_confirmation_fw9EtU`. Read-only inspection
verified both complete saved summary objects match the first round exactly.
Recomputed configured gate: PASS, no reasons. Pedestrian 60/60 (20/20 each),
Stage 1 60/60 (20/20 each), no geometric intrusions or critical failures; mean
formation errors remain 0.1326855206 m and 0.0563095368 m respectively.

The selected residual PPO 5k checkpoint has now passed two consecutive full paired
rounds on the fixed suites and satisfies the configured repeatability criterion.
These are deterministic repeated evaluations of one checkpoint on the same 120
cases, not 240 distinct scenarios or independent policy-seed replication. The
per-round gate utility intentionally leaves automatic promotion false. This
supports Stage 2 fixed-suite acceptance for this model seed and preparation of
the next milestone; it does not authorize automatic Stage 3 implementation or
training. Broader development-route/seed robustness, narrow geometric clearance
and unverified live-Grove parity remain limitations. Final tests remain untouched.
