# Stage 1 A/B at 30k — November 6, 2026

Run: `runs/paired_30k_20261106_040735_cSnYrg/`. Both continuations completed exactly
20k -> 30k, with 150 optimization epochs and 1,200 Adam updates. No new training,
validation rollout, reward/configuration change or checkpoint modification was performed
for this analysis. It reads the completed traces and archives.

## Results on the fixed 60-case validation suite

| Metric | A at 20k | A at 30k | B at 20k | B at 30k |
|---|---:|---:|---:|---:|
| Overall success | 14/60 | **14/60** | 1/60 | **1/60** |
| Straight | 5/20 | 8/20 | 0/20 | 0/20 |
| Gentle turn | 5/20 | **1/20** | 1/20 | 1/20 |
| Stop/start | 4/20 | 5/20 | 0/20 | 0/20 |
| Formation mean, mean of episode means | .640 m | **.776 m** | 1.014 m | **1.507 m** |
| Terminal error norm mean | 1.205 m | **1.819 m** | 2.657 m | **3.826 m** |
| Ended ahead of client itself | 0/60 | **39/60** | 34/60 | **59/60** |
| Actual robot advance mean | 3.440 m | 5.769 m | 5.878 m | 7.447 m |
| Credited advance mean (different definitions) | 3.399 m | 5.091 m | 3.413 m | 2.662 m |

A remains stronger, but its unchanged overall success hides considerable case churn:
only **3 of its 14 prior successes remained successes**. It lost 11 and gained 11.
Straight and stop/start scores improved, while turns regressed. This is not stable
mastery across families. B's previous one success was lost and replaced by a different case.

A failures: **45 recovery timeout, one out-of-bounds**. B failures: **44 recovery timeout,
15 client loss**. No falls, supervisor failures or physics failures occurred in either
30k deterministic validation suite. During the additional stochastic training, A recorded
nine successes, 41 client losses, 13 recovery timeouts and two supervisor failures;
B recorded three successes, 40 client losses, 17 recovery timeouts and one bounds failure.
Those training successes do not replace held-out results.

## Resume/infrastructure checks

Both initial validation summaries exactly equal their respective saved 20k final summaries.
For both workers in both arms, the first resumed scenario seed exactly equals the expected
continuation computed from the saved 20k worker RNG state. A has **234/234 unique scenario
seeds** across all three chunks; B has **233/233**. Thus the previous repeated-stream defect
is not responsible for these results. Throughput remained approximately **64.86/s A** and
**64.79/s B** during this 10k continuation, including optimizer/reset/logging/checkpoint costs
but excluding validation. This is a learning/behavior limitation, not a collection failure.

## A's overshoot returned despite slightly higher reward

At 20k A's mean signed longitudinal formation error at termination was -.542 m relative
to the desired following point; at 30k it became **+1.726 m**. Since desired offset is
-1.5 m, that is a mean robot position **.226 m ahead of the client**. Signed lateral error
mean changed -.812 -> +.381 m. Some lateral bias changed sign, but endpoint norm worsened.

A's mean post-arrival applied command was **(.156, -.138, .0074)** in m/s, m/s, rad/s.
It continues to ask for translation after the client stops. Thirteen of its 14 successful
cases still had planar speed >.03 m/s at termination. Current success criteria do not
require a stationary dwell; the score is not proof of stop-and-hold control.

A's mean validation return improved **-76.95 -> -75.09**, while mean formation quality
worsened. Progress reward increased **13.60 -> 20.37**, formation cost worsened
**-18.55 -> -23.45**, and average terminal success/failure components were unchanged.
This is concrete evidence that reward improvement alone is an unsafe checkpoint-selection
criterion here. It is consistent with the previously identified progress/formation
tradeoff, but does not prove a single causal explanation for policy behavior.

B worsened further: mean terminal signed longitudinal error **+3.392 m**, lateral
**-1.747 m**; its actual advance increased while credited advance decreased. The candidate
is not paying for all its extra travel, but that accounting alone has not taught control.
There is no current performance case for extending B unchanged to a large budget.

A's final normalized Gaussian action std was approximately **(.988,1.007,1.064)**;
B's **(.972,1.009,1.002)**. Exploration is still broad. That is a shared possible
learning difficulty, not proof of the reason for A/B's different behavior. No noise,
reward weights or observations were changed after inspecting these results.

## Can A continue to 100k?

**Technically yes; a blind unattended 70k extension is not presently supported by the
validation trend.** The runner, RNG restoration and optimizer continuation are functioning.
A is the better arm to investigate, but it has not improved overall success and its
stopping/turn behavior is unstable. The old full-run acceptance conditions of repeatable
behavior improvement have not passed. Stage 2 promotion remains out of scope.

My recommendation is to retain A as control and use a **gated continuation**, not assume
100k will solve this:

1. Preserve both the 20k and 30k A archives. The 30k run selected `best_validation=initial`,
   so the 20k policy remains preferred by the declared success/safety/formation ordering.
2. If investigating learning-budget effects, continue A by only one new 10k chunk to 40k,
   with the same settings, then review family success, signed endpoint error, post-stop
   commands and supervisor failures before authorizing remaining chunks. This is an
   experiment, not a promise of improvement. No continuation was started here.
3. Do not select solely on reward. If success degrades or overshoot/critical failures grow,
   stop that unchanged arm and design one controlled stopping/exploration experiment rather
   than raising the budget repeatedly. Keep the gait/physics/caps fixed.
4. Before strong reward-selection claims, replicate an additional training seed. These
   are single-seed, finite-budget learning trajectories, not convergence evidence.

To reach **100k total from A's 30k final checkpoint would require 70k additional
transitions**, not 100k new ones. From a 20k checkpoint it would require 80k. Restarting
unchanged from the same 20k checkpoint and RNG state is expected to reproduce the same
30k trajectory; it is not a mechanism for avoiding this regression.

Neither policy passes the Stage 1 90% overall / 80% per-family gate. Full numerical live
Grove dynamics parity remains unverified. Scripted noncolliding clients do not establish
human safety or real perception performance. No final-test cases, pedestrians or new
curriculum stages were evaluated.
