# Stage 2 first 5k pilot review

**Stage 1 preserved; no successful crossing or diagonal avoidance learned yet. Do not
promote or launch a blind 100k run.** This review inspected existing saved models,
complete validation traces, training diagnostics, scenario logs and RNG sidecars. It
performed inference-only observation ablations and critic checks, not new training or
simulation rollouts. No controller, reward, scenario, observation or training source changed.

Run: `runs/stage2_pilot_YfC5S4/pilot_5k/`.
Parent: registered `stage2_readiness_20261107/transition_ready/initial.zip`.

## Common-suite results

| Metric | Before Stage 2 training | After 5k Stage 2 transitions |
|---|---:|---:|
| Single-pedestrian success | 20/60 | **20/60** |
| Crossing | 0/20 | **0/20** |
| Head-on opposing-lane passes | 20/20 | **20/20** |
| Diagonal | 0/20 | **0/20** |
| Geometric pedestrian intrusions | 40 | **40** |
| Original Stage 1 success | 60/60 | **60/60** |
| Stage 1 mean formation error | .05631 m | **.05822 m** |
| Critical failures | 0 | **0** |

The 1.91 mm Stage 1 formation increase is small and well inside the .10631 m configured
regression bound. The single-pedestrian gate fails correctly on aggregate/per-family
success and geometric intrusion count. Head-on cases were already successful before
training, so their 20/20 score is not evidence of newly learned avoidance.

All forty encounter failures are disc-footprint proxy intrusions, not MuJoCo physical
person contacts. Mean minimum surface clearance in crossing changes -.03685 -> -.03587 m;
diagonal -.05455 -> -.04886 m; head-on +.16097 -> +.16586 m. The millimetre changes do not
establish useful avoidance: every crossing/diagonal still intrudes, and measured penetration
also depends on where the fixed high-level interval ends after the intrusion latch.

## Collection and integrity

- **5,000 new Stage 2 transitions**, from 5k to **10k total** along the selected Stage 1
  5k lineage. This is not 10k Stage 2 training, nor a continuation of the separate Stage 1
  10k experimental branch.
- Actual executed active counts: **500,000 physics steps, 50,000 gait calls**.
- **83.185 s**, **60.11 transitions/s**, including collection/reset/optimizer/logging but
  excluding the two full validation suites. No guessed training throughput.
- **200 new Adam steps**, 194 -> **394 cumulative**. Slot weight norms are nonzero:
  actor .01050, critic .08597. The new observation connections are updating.
- **34 resets / 34 distinct seeds**. Independently advancing each saved initial worker
  RNG through scenario generation and reserved-combination rejection exactly reproduces
  both complete logged scenario-seed sequences. Final worker episode counters 29 and 31
  match the inherited thirteen each plus sixteen/eighteen new resets. No replay bug found.
- Final checkpoint loads under exact current Stage 2 source/config/runtime provenance.

Completed training episodes: **32**, with fifteen successes and seventeen intrusions.
Every crossing (12) and diagonal (5) failed. Every completed replay episode succeeded
(4 straight, 4 gentle-turn, 5 stop/start), as did both head-on episodes. Two episodes were
still active at the budget boundary. Falling aggregate training reward should not be read
as policy collapse: the fraction of completed hard encounters changes across rollout blocks.

The proposed mixture is an **episode sampling** mixture, not a transition quota. Actual
transition percentages were straight 20.66%, turns 17.04%, stops 25.86%, crossing 18.86%,
head-on 7.70%, diagonal 9.88%. Replay therefore occupied **63.56% of collected transitions**,
and crossing plus diagonal only **28.74%**, because failed encounters end much sooner.
This is not proof that the sampler is wrong, but it matters when interpreting the small budget.

## Policy response versus gait response

For already recorded validation observations with surface clearance below one metre,
compare the final actor mean with the SAME observation except pedestrian slot 22:28 is
zero. This is an inference-only sensitivity diagnostic; it is not a physically rolled-out
counterfactual and does not expose future information to the actor.

| Family | Mean absolute pedestrian-induced vx command change | Mean applied vx near pedestrian | Prior mean applied vx |
|---|---:|---:|---:|
| Crossing | .00410 m/s | .23612 m/s | .24035 m/s |
| Head-on | .00194 m/s | .25167 m/s | .25450 m/s |
| Diagonal | .00275 m/s | .24158 m/s | .24613 m/s |

Lateral command sensitivity is also small: .00121 m/s crossing, .00087 diagonal. The
actor has acquired a small response, but not the substantial braking/deviation needed
to clear the conflict. The observed primary failure is continued high-level motion into
the proxy path. There is no fall/supervisor evidence suggesting low-level gait collapse.
This review did not execute a new controlled braking/yielding feasibility trial, so it
does not establish which avoidance maneuver is sufficient for every generated encounter.

## Optimization and critic

Available logged PPO blocks have KL .00531 and .00445, clipping .0653 and .0301. Sigma
remains approximately (.04999,.05001,.04997). There is no evidence of the huge actor update
that destroyed the earlier unguarded warm start. **The last optimizer block is not flushed
to CSV by the current Stage 2 trainer**, so these values must not be presented as all three
blocks. Checkpoint counters confirm that last block ran. No source fix was applied during
this review, preserving strict resume compatibility.

Reported explained variance .0103 -> .0155, value loss 209.7 -> 354.2. The critic inherited
successful Stage 1 return predictions and is not yet predicting new failure outcomes well.
An inference-only diagnostic on the final two seconds of each deterministic validation
trajectory gives:

| Family | Mean predicted remaining value | Actual discounted remaining return |
|---|---:|---:|
| Crossing | +7.54 | -95.36 |
| Head-on | +7.82 | +19.03 |
| Diagonal | +7.55 | -95.31 |

Targets use actual recorded rewards and gamma .99, bootstrapping zero at termination.
The critic was trained on stochastic rather than deterministic continuation, so this is
not an exact same-distribution value calibration test. However, stochastic training also
had zero crossing/diagonal successes. Together the observations support substantial new-task
value mismatch; they do not prove network capacity is insufficient.

Sigma .05 is only .03 m/s command noise per translational axis. Around a .24 m/s mean,
strong braking toward .05 m/s is over six such standard deviations away in a single
sample. Policy-gradient learning can gradually change the mean, but few successful
avoidance trajectories and conservative updates make discovering a new maneuver difficult.
This is a plausible bottleneck, not evidence that simply raising all noise is safe or sufficient.

## Recommended next steps

1. Keep Stage 1 weights/BC baseline and this failed-gate pilot as separate artifacts. Do not
   promote, enlarge the network, change reward weights or start 100k on this evidence.
2. First run a small **evaluation-only causal yielding feasibility test**: can a simple
   braking/sidestep diagnostic, with the same current observations, gait, caps and outcome
   rules, safely complete matched crossing/diagonal trials and recover formation? This
   distinguishes discoverability from impossible timing or outcome constraints. It is not
   automatically a new teacher or the final Reactive Baseline A.
3. Check/calibrate the Stage 2 critic on training-only mixed rollouts with the actor frozen,
   using whole-episode holdout and retaining replay. Compare to a constant baseline and
   inspect near-conflict values. Any resulting branch is explicit and versioned, not a
   silent replacement of this checkpoint or modification of its source manifest.
4. Then choose a bounded matched comparison: unchanged guarded continuation versus a
   justified preparation/discovery adjustment. Another 5k unchanged can be a trend check,
   but the present evidence does not justify assuming it will discover avoidance. If a
   causal diagnostic succeeds and PPO still sees no safe encounters, a separately approved
   pedestrian-aware imitation warm start or gentler encounter curriculum is a reasonable
   candidate. Neither was implemented or used in this pilot.
5. Keep both fixed suites and inspect each family. The 80% pedestrian gate is unmet, while
   the Stage 1 regression gate remains satisfied. Full live-Grove numerical dynamics parity
   and final-test generalization remain unverified.
