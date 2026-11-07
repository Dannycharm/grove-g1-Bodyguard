# Tracking/hold objective C: 20k results and next diagnostic

**C did not improve PPO task completion at 20k. The same actor architecture can nevertheless
solve the complete strict task through supervised imitation.** Do not launch another blind
100k continuation or introduce reward D on this evidence. The next useful experiment is
whether PPO can preserve an already-working mean controller after explicit supervised
initialization and value-function preparation.

Source run: `runs/tracking_stop_v3_20261106_171411/`.
This analysis read the completed run and added **one evaluation-only 60-episode check**:
`reference_imitation_under_C/`. No policy/critic updates, code/config changes or live
simulation interactions occurred. All archives remain unchanged. Computed data/figure:
`analysis/summary.json`, `analysis/strict_C_comparison.png`.

## Same-criterion comparison

Every row uses the exact same fixed sixty scenarios and the released tracking_stop_v3
reward/completion logic: endpoint formation/progress plus a one-second bounded-motion
window, including native speed/yaw ceilings. The imitation model is supervised, not a
PPO-from-scratch policy; this is a feasibility/capacity comparison, not equal-supervision
or sample-efficiency evidence.

| Controller | Overall | Straight | Gentle turn | Stop/start | Mean formation error | Mean terminal error norm |
|---|---:|---:|---:|---:|---:|---:|
| C at 10k | 0/60 | 0/20 | 0/20 | 0/20 | .727 m | 2.013 m |
| C at 20k | **1/60** | 0/20 | 1/20 | 0/20 | **.798 m** | **1.255 m** |
| Existing A20k, evaluated under C | **3/60** | 0/20 | 3/20 | 0/20 | **.643 m** | **1.220 m** |
| Diagnostic imitation actor, evaluated under C | **60/60** | 20/20 | 20/20 | 20/20 | **.0591 m** | **.0201 m** |

A's old endpoint-only 14/60 score is not the right comparator for the stricter task.
Under common criteria it scores 3/60, still above C's 1/60. This remains one training
seed at a finite budget, not proof that C can never learn or that A's reward is globally
superior. Both learned PPO controllers are far below the promotion gate.

C20k failures: **52 recovery timeout, seven client loss**. A reference: 47 recovery timeout,
10 client loss. Neither deterministic suite had falls, supervisor or physics failures.
The imitation reference had no failures of any kind.

## What C learned, and what it did not

C improved from mean actual robot advance .420 m at 10k to **3.928 m at 20k**. Its mean
terminal signed longitudinal formation error was **-.0466 m**, versus A's -.4734 m.
It did not show the earlier B/100k pattern of routinely passing in front of the client:
**0/60 C episodes ended ahead of the client itself**.

However, C's mean terminal signed lateral error was **-1.0496 m**. Correct average
following distance did not translate into correct planar formation. Whole-episode
formation mean .798 m also hides the larger terminal norm 1.255 m.

**12/60 C cases reached the endpoint geometry/raw-progress qualification at some point,
but only one completed the hold.** The recorded credit thresholds were not an all-time
block in those geometrically qualified cases. The other 48 cases never reached that
qualification. Thus weak score is not exclusively an artifact of requiring a hold: most
cases still have a formation problem, while most geometrically qualified cases fail to settle.
A reached geometry qualification in 14 cases and held successfully in three.

Mean post-client-arrival commands and measured body velocities:

| Controller | Applied `(vx,vy,angular_z)` | Measured `(vx,vy,angular_z)` |
|---|---|---|
| C20k | (.158,.040,-.0115) | (.091,.020,-.0126) |
| A20k under C | (.089,.170,-.0462) | (.043,.059,-.0469) |
| Imitation under C | (.0325,-.0030,.0026) | (-.0025,-.00015,-.000084) |

Units: m/s, m/s, rad/s. C continues requesting meaningful translation after the client
stops. The imitation actor's low residual commands allow the unchanged gait to settle.
Together with prior zero-command probes, this supports a high-level learning/control
failure rather than an inability of the gait to stop.

## Infrastructure and optimizer audit

- Both chunks completed: **20,000 transitions, 100 optimization epochs, 800 Adam steps**.
- Active executed counts: **2,000,000 physics steps, 200,000 gait calls**, plus warmups/evaluation.
- Training took **321.20 s**, approximately **62.27 transitions/s**, excluding validation/plotting.
- Source hashes, configuration, dependencies, resolved reward details and reference manifest
  match the current released implementation. No accidental objective drift was found.
- **180 scenario resets / 180 unique seeds**: 84 straight, 52 turns, 44 stops.
- Both first resumed worker seeds match independently advanced saved RNG states.
- The resumed initial validation summary exactly equals the preceding final summary.

The confirmed old RNG defect is not recurring here.

Final C PPO diagnostics: explained variance **-.00093**, value loss **217.73**, entropy
loss **-4.2485**, approximate KL **.00926**, objective clipping **.07349**. Gaussian action
standard deviations were **(.9695,1.0208,1.0023)**; **69.16%** of sampled three-axis vectors
crossed a Box bound. Objective clipping and action-bound clipping are different metrics.

Across **176 completed stochastic training episodes**, there were **zero successes**:
152 client losses, 20 recovery timeouts and four supervisor failures. No fall/physics-failure
reason codes were recorded. Dense costs did teach substantial forward progress, but the
policy did not obtain successful full-task training trajectories at this budget.

Near-zero explained variance and no successful training episodes are consistent with
weak return/value learning and difficult discovery under broad action noise. They do not
prove that the critic network is too small, that PPO is broken, or that reward alone is
the cause. Different reward scales also make raw value-loss comparisons across A/B/C
inappropriate. More samples might help; the current evidence does not justify assuming
that a blind larger budget will resolve the behavior.

## Why the imitation result matters

The saved diagnostic imitation actor has the **same two 128-unit actor layers and forty
causal inputs**. Its critic/log_std were not trained by imitation. The archive has zero
PPO learning timesteps; the generic evaluation CLI prints `ppo` for loaded SB3 archives,
but this actor was trained with supervised teacher labels, not PPO returns.

It achieved all **60 strict-hold successes**, with 1.0 s completed holds and mean terminal
formation error .0201 m. This extends the earlier twelve-case strict development check
to the full fixed suite. No future route information entered the actor.

This rules against actor representational capacity or broad task/gait infeasibility as
the immediate practical bottleneck on these scenarios. It does not establish critic
capacity or comparable learning difficulty. The imitation fit used 12,000 supervised
updates with expert labels, compared with 800 PPO Adam steps here; supervision and state
coverage differ. It also does not prove the C reward is optimally aligned or that it will
train well from scratch.

We now have a working neural controller, but **it is an imitation-trained controller**.
The research question should next be whether PPO can retain and improve that behavior,
not whether another hand-changed reward or wider actor can rescue cold-start discovery.

## Recommended next milestone

**No reward D, no larger actor, no automatic Stage 2 or 100k run.** Preserve all current
from-scratch controls and the imitation-only baseline.

1. Evaluate the frozen imitation mean with controlled stochastic noise on development
   scenarios to choose a conservative exploration level around an already-good policy.
   This differs from the failed lower-noise-from-zero test: the useful mean controller is
   now present. Do not assume the same noise value will work without checking.
2. Implement an explicit **actor initialization** path, validating tensor/observation/
   action/gait contracts and recording source archive/data hashes. It is **not --resume**:
   the imitation archive has no trained PPO critic/optimizer/RNG sidecars. Do not weaken
   strict resume checks or pretend previous PPO state exists.
3. With the initialized actor frozen, collect training-only full-G1 rollouts and fit the
   critic to actual C returns. Check held-out training-episode value fit against a constant
   baseline. Do not supply route progress or future state to the critic; keep the causal
   actor/critic observation boundary. Perfect value prediction is not required under
   partial observability.
4. Run a **short 5k PPO preservation test**, optionally extending to 10k only if held-out
   success and formation remain strong. Use separate fresh experiment outputs and retain
   the frozen imitation-only comparator. Monitor actor action drift, variance, critic fit,
   KL, formation, stopping and all supervisor failures.
5. If PPO rapidly destroys the initialized controller, investigate updates/critic/noise
   before collecting more data. If it preserves competence, this is a viable explicitly
   labeled imitation-pretrained PPO approach. Establish two passing PPO validation rounds
   before considering curriculum promotion.

A practical preservation gate should retain >=90% strict overall and >=80% per family,
no critical validation failures, and formation/stopping close to the frozen imitation
reference. These are prospective checks, not passed results of a PPO warm-start run.
The initialization/noise/value preparation must be versioned as a changed experimental
condition; do not claim resulting competence was discovered by PPO from scratch.
Keep BC-only in the later scientific comparisons to attribute improvements correctly.

Only the imitation evaluation was executed during this analysis; the proposed warm-start
trainer/critic phase and PPO experiment have **not** been implemented or launched here.
No existing checkpoint was overwritten. Full numerical live-Grove dynamics parity remains
unverified, and flat-ground scripted-client results are not human-safety or perception evidence.
