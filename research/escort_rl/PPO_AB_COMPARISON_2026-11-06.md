# Fixed-RNG A/B results — November 6, 2026

**B performed substantially worse than A at 20k in this controlled, single-seed experiment.**
Do not select B for a larger run on the present evidence. Neither policy passes Stage 1.
The RNG fix worked, and B's progress ledger matches its declared equation. This is a
learning-performance result, rather than evidence of another restart or accounting bug.
No code, reward, hyperparameter, checkpoint or old experiment was changed during analysis;
no additional training or validation rollouts were started.

Runs:
- A: `runs/rng_fixed_A_20261106_021910/` — fixed RNG, legacy_v1 reward.
- B: `runs/rng_fixed_B_20261106_021910/` — fixed RNG, formation_progress_v2 reward.
- New computed evidence: `runs/ab_analysis_20261106__s4pbpy4/summary.json` and `ab_comparison.png`.

## Held-out performance

Both were trained from scratch for exactly 20k transitions, using the same root seed,
workers, PPO architecture/hyperparameters and fixed 60-case validation suite.

| Metric | A, 10k | B, 10k | A, 20k | B, 20k |
|---|---:|---:|---:|---:|
| Success | 0/60 | 0/60 | **14/60 (23.3%)** | **1/60 (1.7%)** |
| Mean formation error, per-episode mean | .760 m | .736 m | **.640 m** | **1.014 m** |
| Mean actual robot advance | 2.163 m | .726 m | 3.440 m | 5.878 m |

At 20k, A succeeded on **5/20 straight, 5/20 gentle-turn and 4/20 stop/start**. B succeeded
on **0/20, 1/20 and 0/20** respectively. On paired cases, 14 were A-only successes and
one was B-only; none succeeded under both policies. The gap is 13 successes, or 21.7
percentage points, on this fixed suite.

Final failures:
- A: 36 recovery timeouts, 10 client losses.
- B: 49 recovery timeouts, nine client losses, one out-of-bounds.
- Neither validation suite reported a fall, supervisor failure or physics failure.

The old faulty-resume 20k model achieved 35%, but it saw different scenario streams; it
is not a fair estimate of what fixed RNG alone should reproduce. The fresh A 10k result
matches the original pilot's validation metrics, a useful reproducibility check.

## The new comparison is internally consistent

Read-only integrity checks confirmed:

- Initial actor/critic parameter tensors were **identical** between A/B.
- Source hashes, principal dependency versions and seed-stream definitions match.
- The **only configuration difference is reward_version**.
- Each reached 20k, 100 optimization epochs and 800 Adam steps.
- A sampled **167 scenarios / 167 unique seeds and signatures**.
- B sampled **170 / 170 unique seeds and signatures**.
- Actual first resumed seeds match the saved-stream continuations for both workers:

| Arm | Worker 0 | Worker 1 |
|---|---:|---:|
| A | 639056782823807736 | 3189754318648236116 |
| B | 3669430088568332896 | 4446982675518719192 |

A sampled 80 straight / 47 turns / 40 stops; B sampled 81 / 47 / 42. These are consistent
with the configured mix at this sample size. Different realized episode counts are
expected once policies/outcomes differ; the initial streams, not every transition, are
matched. No repeated startup sequence appeared.

The additional B scenario counts are not an uncontrolled configuration difference: policies
cause different termination times and therefore consume the same initial streams at
different rates. Both optimization budgets and validation instances are fixed.

## B still overshoots and drifts sideways

Signed errors are in the client heading frame relative to desired offset [-1.5,0].
Positive longitudinal error means too far forward relative to that target.

| Final 20k diagnostic | A | B |
|---|---:|---:|
| Terminal signed longitudinal error mean | -.542 m | **+1.826 m** |
| Terminal signed lateral error mean | -.812 m | **-1.459 m** |
| Terminal formation error norm mean | 1.205 m | **2.657 m** |
| More than .75 m ahead of target | 0/60 | **49/60** |
| More than .75 m behind target | 17/60 | 5/60 |
| Ahead of the client itself | 0/60 | **34/60** |
| Actual advance / route length, mean | .861 | **1.458** |

B's mean robot longitudinal position was .326 m ahead of the client, rather than 1.5 m
behind. A's mean was 2.042 m behind the client. A's remaining problems are therefore
lag/lateral alignment and recovery; B has both excessive forward travel and lateral drift.
Whole-episode formation averages understate the larger endpoint errors, especially in B.

After client arrival, time-step pooled mean commands and measured body velocities were:

| Arm | Applied `(vx,vy,angular_z)` | Measured `(vx,vy,angular_z)` |
|---|---|---|
| A | (.087, .165, -.045) | (.039, .053, -.045) |
| B | (.110, -.269, .010) | (.099, -.181, .011) |

Units are m/s, m/s and rad/s. Both continue requesting movement; B's sideways request is
particularly large. This supports a high-level control failure rather than blaming gait
inertia for sustained post-stop translation. The preceding audit's zero-command probes
stopped the same gait in .6–.8 s; no new stop probes were needed for this analysis.

### Success is not yet stop-and-hold

Under the current task definition, success is endpoint formation recovery plus progress
conditions; it does not require a stationary dwell. Of A's 14 successes, **13 still had
planar speed >.03 m/s at termination**, with mean terminal speed .162 m/s. B's one success
had .173 m/s. Those are valid successes under the existing definition; they should not be
reinterpreted as reliable stopping. No success criterion was changed retrospectively.
Stop-and-hold should remain a separate diagnostic before declaring robust escort behavior.

## The candidate bookkeeping worked, but that did not make PPO learn better

I reconstructed every 20k validation transition's progress increment from its logged raw
and desired-point progress and highwater state:
- A maximum discrepancy: **0**.
- B maximum discrepancy: **4.44e-16 m**.

There were **no cases** in which B achieved the endpoint geometric/raw-progress conditions
but was denied success solely by its accompanied-credit threshold. Only one B case met
those geometry/raw-progress conditions at all. Its worse score is not simply a stricter
bookkeeping-based completion label.

A's mean actual advance was 3.440 m with 3.399 m credited. B's actual advance was 5.878 m
with 3.413 m credited. B's extra travel was therefore largely **uncredited**, as intended
by the candidate. Credited advance and reward are not identical definitions across arms;
actual motion, formation and success provide the common comparison.

Mean final validation reward components:

| Component | A | B |
|---|---:|---:|
| Formation | -18.548 | -25.874 |
| Progress | +13.597 | +13.652 |
| Command change | -.00242 | -.00208 |
| Torso stability | -.000082 | -.000085 |
| Terminal success | +4.667 | +.333 |
| Terminal failure | -76.667 | -98.333 |

B's worse return accompanies worse geometry and failures; it is not merely the absence
of the old extra gap-closing bonus. Returns across different reward definitions should
not themselves be used as the selection criterion.

The previous synthetic audit remains valid: v2 removed a demonstrated local incentive
conflict. That does **not** prove it improves PPO's learning speed or the final policy.
A reward can be better aligned in a counterexample but harder to optimize or produce a
worse finite-budget trajectory. This experiment rejects adopting B on current performance;
it does not establish that v2 is universally inferior or identify one proven causal
mechanism for the difference.

## Shared learning issues and limitations

Exploration remains broad: sampled three-axis action vectors required clipping on at
least one axis **69.92% in A and 69.47% in B**. Final normalized action standard deviations
were A (1.002,1.005,1.051), B (.989,1.003,.995). This is a shared challenge and cannot by
itself explain why B was worse, especially since B's standard deviations are slightly lower.
The smoothness term is not a stopping controller: a nearly constant bad command incurs
little command-change cost. Neither arm contains an explicit velocity-matching/stop-dwell
objective, so that behavior must be learned through formation/progress/outcomes.

Completed stochastic training episodes across both chunks:
- A: 10 successes, 136 client losses, 14 recovery timeouts, three supervisor failures.
- B: six successes, 140 client losses, 18 recovery timeouts, two supervisor failures.

Do not confuse those stochastic training successes with deterministic held-out results.
Neither arm is safety-qualified. Training/collection time was 317.91 s for A and 319.34 s
for B, excluding validation/plotting; compute throughput is not the limiting difference.

This is **one paired training seed and 20k transitions per arm**, not convergence evidence
or a confidence interval across training seeds. Changed rewards alter gradients and later
state distributions; matched initialization does not force equivalent later trajectories.
More training might change the ranking; it must be measured, not assumed. Full numerical
live-Grove dynamics parity remains unverified, as before.

## Recommendation

1. **Keep A as the current control; do not promote or adopt B.** Preserve all checkpoints.
   Neither reaches the 90% overall / 80% per-family Stage 1 gate.
2. If continuing the planned diagnostic comparison, finish it at **30k per arm unchanged**
   with one additional 10k chunk each, rather than jumping to 100k. This tests whether B
   is merely slower to learn; use each arm's own final checkpoint, exact same configuration
   and new output paths. No continuation was started during this analysis.
3. Inspect global checkpoint quality, endpoint signed errors and post-stop commands at
   30k. If B remains worse, stop that arm rather than repeatedly increasing its budget.
4. Before selecting a reward generally, replicate with another training seed. If both
   still fail stopping, test one separately versioned change at a time (for example,
   exploration scale or a stopping/relative-velocity objective), not simultaneous reward,
   noise and observation changes. The current analysis does not authorize or implement
   such a change. Keep gait, dynamics, caps and observation access fixed.

The important correction to the earlier expectation is explicit: **v2 passed its reward
counterfactual and physics-neutrality tests, but it has not passed the learning test.**
This comparison is exactly why the small A/B experiment preceded another long run.
