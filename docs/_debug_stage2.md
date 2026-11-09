# Debugging Stage 2: architecture, training history and current failure

**Project:** ECE 598JK Humanoid Escort RL
**Status date:** November 8, 2026
**Purpose:** A descriptive overview for understanding and debugging the current system.

This document describes the **one-pedestrian Stage 2** work. Stage 3, intended for
more complex/two-pedestrian encounters, has not been implemented or trained. The
phrase “methods tried for Stage 3” in the request is interpreted here as the current
Stage 2 debugging sequence. Nothing below is evidence of Stage 3 readiness.

**Update after this overview:** the frozen-base residual diagnostic has now been
implemented and evaluated. Sections 1–9 describe the earlier whole-MLP sequence;
see Section 11 for the new architecture, measured gains and remaining failures.

## 1. Where we are now

We have a physically simulated G1 that can follow a client successfully in Stage 1.
We also have a hand-designed, causal Stage 2 teacher that can demonstrate successful
pedestrian avoidance. However, our learned high-level MLP has **not successfully
acquired that avoidance behavior**. Recent attempts to teach it have also degraded
some of its previously reliable following/stopping behavior.

The latest aggregated-imitation candidate:

- Completes **0/20 crossings, 0/20 diagonals and 20/20 head-on** validation cases.
- Completes **47/60 Stage 1** cases; the other thirteen fail endpoint recovery/stop hold.
- Has forty geometric pedestrian intrusions in the pedestrian suite.
- Failed the continuation gates, so **no approved `prepared.zip` was saved**.
- Is an actor-only diagnostic artifact, not a policy approved for PPO continuation.

The successful Stage 1 weights remain intact. We have not changed the frozen gait,
physics, action limits or observation schema to manufacture better results. Full
numerical parity with the live Grove-G1 ROS/DDS stack remains unverified.

## 2. The architecture: two learned controllers at different levels

The high-level escort network does not walk the robot by commanding its joints.
It decides how the robot should move, and an existing walking controller executes
that request through full MuJoCo physics.

```text
Seeded client/pedestrian scenarios
             │ current simulated states only
             ▼
40-dimensional causal observation
             │
             ▼
High-level escort actor MLP, 5 Hz
             │ normalized [forward, lateral, turn] action
             ▼
Existing command limits, timeout and safety/controller behavior
             │ bounded body-frame velocity command
             ▼
Frozen NVIDIA WBC-AGILE ONNX gait, nominally 50 Hz
             │ joint targets and gains, feedback/history updates
             ▼
Reference-compatible PD, other-joint holds and actuator limits
             │ joint control
             ▼
Full G1 MuJoCo dynamics and contacts, 500 Hz
             │
             └────────── next current-state observation
```

The upper network is the one we train. The lower network is frozen. Teaching the
upper policy to yield does not retrain balance or replace the walking policy.

### 2.1 The frozen lower-level gait

The unchanged checkpoint is `unitree_g1_velocity_e2e.onnx`, with SHA-256:

```text
8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b
```

Its embedded neural core is `255 → 256 → 256 → 128 → 14`, with ELU hidden
activations. It processes history through explicit feedback tensors; it is not a
GRU/LSTM. Its interface has twelve inputs and ten outputs. It observes 29 body
joints and produces fourteen learned joint-position actions, with remaining joints
handled by reference holds. The graph includes its own input processing, action
scaling and gain outputs. We do not normalize or scale those a second time.

Each isolated environment owns its MuJoCo model/data, gait feedback, controller,
commands, clock and scenario state. Reset restores these states and performs startup
stabilization with the client clock paused. The robot may be placed at reset; its
base is **never directly moved during rollout**. Scripted people use mocap proxies.

An ordinary high-level action lasts 0.2 simulated seconds: **100 physics steps and
10 gait evaluations**, verified using counters. The physics timestep is 0.002 s.
This deterministic isolated schedule does not prove equivalence to asynchronous
ROS/DDS timing in the original live system.

### 2.2 Why conservative client speeds matter

The software command caps are ±0.6 m/s forward/lateral and ±0.9 rad/s turning.
Those are command limits, not measured performance guarantees. Sustained forward
motion measured approximately **0.407 m/s at the +0.6 m/s command cap**. Small
commands around 0.1 m/s produce little sustained translation.

Stage 1 client trajectories therefore use conservative **0.15–0.25 m/s peak speed
bounds**, not 0.5 m/s clients. Smooth trajectories accelerate from and decelerate to
rest, so a sampled peak-speed bound is not the route's average speed. Neither the
speed limit nor the gait was raised to improve learning.

## 3. Our exact high-level MLP

There are separate actor and critic MLPs inside Stable-Baselines3 `MlpPolicy`.
They receive the same observation but have separate hidden-layer parameters.

| Network | Structure | Purpose | Parameters |
|---|---|---|---:|
| Actor mean | 40 → 128 Tanh → 128 Tanh → 3 linear | Mean normalized velocity action | 22,147 |
| Critic | 40 → 128 Tanh → 128 Tanh → 1 linear | Expected discounted return | 21,889 |
| Gaussian log standard deviation | Three learned scalars | PPO exploration noise | 3 |
| Total | Actor, critic and log standard deviations | Saved SB3 policy | **44,039** |

These sizes and activations were verified by loading the selected Stage 1 checkpoint.
The MLP is feed-forward: there is no recurrent state or observation-frame stack.
Previous applied action is included as an observation, but that is not an LSTM.
The gait's separate history must not be confused with high-level MLP memory.

During PPO, the actor outputs a Gaussian mean. Three learned, state-independent
standard deviations determine sampling noise. A sampled action is clipped to the
normalized action Box. The output head itself is **linear, not a Tanh-squashed
Gaussian**. Deterministic evaluation uses the mean with the action-space handling.

For an applied normalized action `a`, the command envelope is:

```text
vx       = 0.6 × clip(a[0], -1, 1) m/s
vy       = 0.6 × clip(a[1], -1, 1) m/s
yaw_rate = 0.9 × clip(a[2], -1, 1) rad/s
```

The successful guarded Stage 1 policy has standard deviations near **0.05** on each
normalized axis. Before clipping, that corresponds to command noise standard
 deviations of about 0.03 m/s, 0.03 m/s and 0.045 rad/s. A small random perturbation
is not the same as discovering a coordinated, sustained evasive maneuver.

The critic does not select commands. It helps PPO estimate whether sampled actions
were better or worse than expected. Improving the critic alone does not directly
change what the deterministic actor does.

## 4. The 40 observation inputs

Values are `float32`, statically scaled and clipped to [-1, 1]. We do **not** add an
SB3 running observation normalizer. Indices below are zero-based and inclusive.

| Indices | Count | Meaning | Scaling |
|---|---:|---|---|
| 0–1 | 2 | Current client-relative planar position | Divide metres by 16 |
| 2–3 | 2 | Current client-minus-robot planar velocity | Divide m/s by 2 |
| 4–5 | 2 | Sine and cosine of client heading minus robot heading | Already bounded |
| 6–7 | 2 | Desired offset in client heading coordinates, initially [-1.5, 0] m | Divide metres by 3 |
| 8–9 | 2 | Client valid/fresh flags | 0 or 1; current proxy sets both from validity |
| 10–12 | 3 | Pelvis-origin forward/lateral velocity and body angular z | Divide by [2, 2, 4] |
| 13–15 | 3 | Torso projected gravity | Direction components |
| 16–18 | 3 | Torso angular velocity | Divide rad/s by 10 |
| 19–21 | 3 | Previous applied normalized action | Already in action coordinates |
| 22–27 | 6 | Pedestrian slot 1 | Described below |
| 28–33 | 6 | Pedestrian slot 2 | Empty in Stage 2 |
| 34–39 | 6 | Pedestrian slot 3 | Empty in Stage 2 |

Each pedestrian slot contains:

```text
[relative_x / 16,
 relative_y / 16,
 relative_velocity_x / 2,
 relative_velocity_y / 2,
 pedestrian_radius_in_metres / 1 metre,
 validity_mask]
```

In Stage 1 all eighteen pedestrian-slot values are zero. Stage 2 activates only
slot 1, indices 22–27. Empty slots remain zero and invalid; they are not fake people.
Three slots in the vector do not mean three-pedestrian curriculum support exists.

### 4.1 Frame and measurement details

Client/pedestrian planar relative positions and velocity differences are rotated
into robot **heading/yaw axes**. Relative velocity is the difference of physical
world velocities expressed in those axes; it is not the rotating-coordinate
position derivative with an extra angular cross-product term.

Robot motion uses the corrected **pelvis-origin** measurement rather than the
pelvis inertial COM. Its linear velocity is expressed in the full pelvis axes.
Observation index 12 is pelvis **body angular z**, not the exact derivative of
world Euler yaw when the robot tilts. Exact Euler yaw rate is available separately
for evaluation, including stop checks. These small convention distinctions matter
when diagnosing turns; the gait's own input processing was not altered.

### 4.2 What the policy cannot see

The actor and initial critic do not receive scenario-family labels, future human
waypoints, route progress/completion labels, evaluator metadata or resolved future
trajectories. Current client/pedestrian states are simulator-pose proxies, not camera
perception. Family labels can balance a supervised batch without becoming network
features. Reward/evaluation may use route geometry privately.

The teacher computes predicted closest approach from current relative positions
and velocities. The MLP does not receive a precomputed time-to-collision or risk
feature: it must infer a useful response from the existing vector.

## 5. How Stage 1 became successful

It was **not** a successful PPO-from-scratch result. The working method was
**imitation-pretrained PPO with guarded fine-tuning**.

### 5.1 Initial PPO and RNG failures

An early 100k PPO run peaked at 21/60 validation successes at 20k, then ended at
0/60. A confirmed restart bug repeatedly reset the scenario streams: 471 resets
across the added 90k transitions contained only 65 unique seeds.

SB3 model loading queued vector-environment seeds. Restoring our own worker RNG
states did not cancel those pending seeds; the next vector reset overwrote them.
The repair clears pending seeds/options after model setup, then restores each
worker's saved RNG stream and the model-side RNG states. Resume tests now check
expected future scenario sequences, not only policy weights/optimizer state.
Physical episodes still restart on resume; partial physics/rollout state is not
restored, so bit-identical uninterrupted learning is not promised.

Fixing RNG improved experimental validity but did not by itself solve learning.

### 5.2 Reward and exploration experiments did not solve cold-start PPO

With fixed RNG, control A reached 14/60 successes at both 20k and 30k. An alternative
formation-progress reward B reached 1/60. A's unchanged success count concealed case
churn and renewed overshooting; higher return did not consistently mean better
formation or stopping.

Reducing initial Gaussian sigma from 1 to approximately 0.368 reduced action
clipping, but achieved 0/60 validation success at 20k. The policy barely translated.
Lower noise alone was not enough.

A more explicit tracking/stopping objective C, `tracking_stop_v3`, achieved only
1/60 from scratch at 20k. However, the imitation actor achieved 60/60 under the SAME
strict criteria. Thus the task was feasible and the MLP could represent Stage 1,
but changing the reward alone had not taught PPO the behavior at that budget.

### 5.3 What the successful objective means

The desired position is 1.5 m behind the client, in the client's heading frame.
Current reward combines robust formation cost, interval velocity-tracking cost,
qualified new route progress, command-change/torso penalties and terminal outcomes.
Progress credit uses nondecreasing ledgers and a cap at the desired escort point;
client movement alone and repeatedly covering old ground do not earn new progress.

The robust formation/velocity costs discourage large errors without an unbounded
inverse-distance reward. Success is not merely “client arrived while G1 is nearby.”
The robot must achieve sufficient accompanied progress, restore endpoint formation
within 0.75 m and satisfy a **one-second stop hold** within the five-second recovery
window. The hold checks net speed ≤0.05 m/s, instantaneous planar speed ≤0.15 m/s,
excursion ≤0.10 m and yaw rate ≤0.1 rad/s. Client loss uses a 3 m distance limit with
three seconds of continuous grace.

These checks exposed policies that reached the right area but kept moving. They
are experimental task definitions, not a published optimal reward or a convergence
guarantee. Stage 2 retains this objective and adds geometric pedestrian clearance.

### 5.4 Supervised imitation supplied a feasible feedback controller

The original causal diagnostic follower computes commands using current relative
position, velocity and heading, with a fixed dead-zone compensation for the gait.
It has no future-route access and does not directly control joints or the base.
We trained the SAME 128×128 actor to match its normalized commands using MSE.

The successful capacity/imitation experiment used:

1. Eighteen teacher-controlled full-G1 episodes.
2. Two additional rounds of twelve learner-controlled episodes each, with teacher
   labels on the states the learner visited: **DAgger-style aggregation was already
   part of the successful Stage 1 procedure**.
3. Three successive stages of 4,000 supervised updates: **12,000 updates total**,
   learning rate 3e-4, batch up to 256, actor parameters only.
4. 8,096 causal samples from 42 episodes; 32 whole scenarios for fitting and ten
   for holdout. Final holdout MSE was about 0.00210.

The resulting actor achieved 60/60 held-out Stage 1 successes, and later 60/60 under
strict objective C. This demonstrated Stage 1 actor capacity, not PPO discovery or
capacity for every future pedestrian scenario. The diagnostic teacher is not the
completed final Reactive Baseline A for our research comparison.

### 5.5 Noise calibration and critic preparation made PPO initialization explicit

An actor that succeeds deterministically can still fail if large exploration noise
is added. On twelve matched development cases, sigma 1.0 and 0.37 scored 0/12,
0.10 scored 9/12, and 0.05 and 0.02 scored 12/12. We selected **0.05**, the largest
passing tested value. This was measured calibration, not an assumed magic constant.

A fresh PPO model imported only the successful actor. Its critic was fitted using
24 complete training-only stochastic episodes, 4,467 additional transitions and
6,000 critic-only updates. Actor and variance stayed unchanged. Values were learned
from the current stochastic actor's actual returns, not a different teacher's
returns. Whole-episode holdout checks prevented treating the fitting loss as proof.

### 5.6 PPO needed protection against destroying the imitation behavior

The first unguarded warm start still collapsed: a 60/60 initial actor became 0/60
after only 5k PPO transitions. The first optimizer block had approximate KL around
295, showing excessive policy movement relative to sigma 0.05. PPO's clipped
objective was not a hard guarantee that the policy would change only slightly.

The successful guarded configuration changed learning rate **3e-4 → 1e-5** and added
**`target_kl=0.01`**. These two guards were tested together, not independently.
The working configuration otherwise retained gamma 0.99, GAE lambda 0.95, clip range
0.2, 1,024 rollout steps per environment, batch size 256, ten nominal epochs,
entropy coefficient 0.001, value coefficient 0.5 and gradient norm cap 0.5.
Actual successful runs used two independent workers and PPO seed 42.

| Stage 1 model | Strict validation | Mean formation error |
|---|---:|---:|
| Prepared imitation initialization | 60/60 | 0.05906 m |
| Guarded PPO, 5k cumulative transitions | 60/60 | 0.05631 m |
| Guarded PPO, 10k cumulative transitions | 60/60 | 0.05771 m |

Both PPO rounds performed real optimizer updates and passed preservation checks.
The 5k model remains selected because success tied and its mean formation was
slightly better. This is evidence of **preserving imitation competence under PPO**,
not a major learned improvement or proof of robustness across training seeds.

The 5k/10k counts are PPO transitions only. They exclude imitation updates,
demonstration interactions, noise sweeps and critic-preparation samples. The old
failed 100k run is a different branch; the successful policy is not “that same
100k policy after another 10k.”

## 6. Stage 2: task and transition from Stage 1

Stage 2 introduces ONE scripted pedestrian: crossing, head-on lane-pass and diagonal
encounters, retaining client-only replay. Configured PPO scenario probabilities:

| Family | Probability |
|---|---:|
| Straight | 15% |
| Gentle turns | 12.5% |
| Stops/starts | 12.5% |
| Crossing | 35% |
| Head-on | 12.5% |
| Diagonal | 12.5% |

The transition starts from the selected successful Stage 1 checkpoint. Only the
newly active first-layer pedestrian input columns 22–27 in actor/critic and their
Adam moments were zero-initialized. Other learned weights, optimizer state and RNG
streams were preserved. With no pedestrian, original Stage 1 outputs remain exact
at this transition. We did not initialize an entirely new MLP.

The pedestrian is a scripted, nonreactive, noncolliding proxy. Intrusion is measured
with a swept pelvis footprint disc of radius 0.30 m against a pedestrian disc of
radius 0.25 m. This is **geometric overlap, not a MuJoCo person-contact event**.
A 0.30 m preferred surface margin adds shaping; negative surface clearance triggers
an intrusion failure with the existing terminal failure penalty. Actual robot/ground
contacts and full-body dynamics still run normally.

Fixed validation consists of twenty cases per pedestrian family, plus the SAME
original sixty Stage 1 cases. Stage 2 gates require at least 80% overall/per-encounter
success, no geometric intrusions/critical failures, and Stage 1 preservation:
≥95% overall, ≥90% per old family, limited formation regression. Two passing rounds
are required for promotion. Final-test seeds/routes remain untouched.

## 7. All Stage 2 approaches tried so far

The following are different branches/diagnostics, not one uninterrupted policy.

### 7.1 Direct guarded PPO continuation

The prepared Stage 1-derived policy scored 20/60 pedestrian validation: 20/20
head-on, 0/20 crossing, 0/20 diagonal. After 5k new Stage 2 PPO transitions it still
scored 20/60 with forty intrusions; Stage 1 stayed 60/60. Pedestrian input weights
changed, so this was not simply an accidentally frozen actor, but useful pedestrian
sensitivity remained weak. Low exploration may hinder discovering sustained early
yielding, but that alone was not proven the complete cause.

### 7.2 Causal-controller feasibility and reward audit

We tested follow-only, stop-only, yielding and sidestepping controllers, using the
same causal observations, commands and full G1 dynamics. Across the fixed sixty
pedestrian cases: follow scored 20/60, yield 30/60, sidestep 27/60, stop 0/60.
Yielding solved 20/20 crossings but only 3/20 diagonals.

Across 240 matched rollouts, all 23 successful yield-vs-collision comparisons and
seven successful sidestep-vs-collision comparisons favored avoidance under both
undiscounted and discounted return. However, early avoidance initially lost reward;
it began about 2.8 s before close-clearance shaping activated. This points to delayed
credit/behavior discovery, not a proof that the reward is globally ideal. It did
not justify blindly replacing the reward.

### 7.3 Critic-only preparation

Keeping the actor and variance bitwise unchanged, we collected 72 current-policy
stochastic episodes: 10,680 additional transitions, then 6,000 value-only updates.
Holdout value MSE improved **1508.64 → 219.62**; explained variance improved
approximately **0.019 → 0.826**. Impending-failure states changed from positive value
predictions to negative ones, although failure severity was still underestimated.
Head-on value predictions regressed substantially, so calibration was not uniform.

This changed the critic, not the actor. It could not itself teach avoidance actions.

### 7.4 Another bounded PPO run after critic preparation

After 5k further Stage 2 PPO transitions, pedestrian success remained 20/60:
0 crossing, 20 head-on, 0 diagonal. Stage 1 stayed 60/60. Rollout explained variance
stayed around 0.93, so the better critic did not immediately collapse. Nevertheless,
all completed crossing/diagonal training episodes still failed. Better failure
prediction alone did not produce a successful maneuver.

This branch reached 15k cumulative PPO transitions: 5k Stage 1 plus 10k Stage 2.
Later supervised branches below restart from the selected Stage 1-derived parent,
not from this failed 15k branch.

### 7.5 Initial stop/yield teacher

We built gated teacher collection and actor fine-tuning. The first teacher stopped
forward/lateral motion on predicted risk. In an eighteen-case development smoke
test it succeeded on all fifteen non-diagonal cases but **0/3 diagonals**. A diagonal
pedestrian could continue into a stopped robot, or the robot could wait long enough
to lose the client. The readiness gate correctly blocked fitting from that batch.

### 7.6 Improved velocity-candidate teacher

The improved teacher considers 49 planar velocity-grid candidates plus nominal
formation velocity. It predicts pedestrian motion from CURRENT observed velocity,
over a four-second horizon, with an assumed 0.4 s robot response lag. It scores
predicted clearance deficit, deviation from formation velocity and velocity changes.
It can brake, retreat or sidestep, then recover formation. Its predictions are
controller approximations, not access to scripted futures or a substitute simulator.
The same controller handles all families; without a pedestrian it returns the
original Stage 1 follower.

Results: the three formerly failed diagonals became 3/3 success; a separate
training-partition batch achieved **60/60**, ten per all six families; fixed
pedestrian validation achieved **60/60**, twenty per encounter type. The minimum
validation surface clearance was only 0.06089 m. This is demonstrated simulator
feasibility, not formal safety or a learned-policy result.

### 7.7 First Stage 2 behavior-cloning fit

We collected **10,878** teacher observations/actions and ran 3,000 actor-only
supervised updates from the Stage 1-derived parent. Critic and variance stayed
unchanged; stale actor Adam moments were explicitly reset.

Ordinary family-balanced MSE fit produced pedestrian success **20/60**: no crossings
or diagonals learned. Stage 1 remained 60/60, mean formation 0.08030 m. No prepared
PPO archive was saved because the gates failed.

Only **698/10,878 labels, 6.4%**, substantially differed from ordinary following
(normalized action L2 difference >0.15). A low average error could hide failure on
rare, decisive avoidance actions. On failed student states the teacher requested
retreat/lateral motion, but the actor kept commanding forward motion.

### 7.8 Intervention-balanced imitation v3

The v3 sampler keeps 50% Stage 1 replay. Within each pedestrian-family batch, half
the labels are substantial interventions and half ordinary following. This raises
the overall intervention share to about 25% of a batch. Whole-episode holdout is
preserved, and errors are reported by family and intervention status.

After another separate 3,000-update fit from the SAME Stage 1 parent, teacher-state
intervention errors fell sharply. But closed-loop validation became **14/60**:
0/20 crossing, 0/20 diagonal, 14/20 head-on, with 41 intrusions and five recovery
timeouts. Stage 1 remained 60/60 but mean formation worsened to 0.17805 m, failing
its preservation gate. Diagnostic actor parameters were saved; no approved ZIP.
Three identical reruns had identical actor hashes. They are reproducibility checks,
not independent training-seed replications or extra learning on one model.

### 7.9 Student-only DAgger collection

Using that diagnostic student, we collected eighteen NEW training-partition
episodes, three per family, while querying the causal teacher on each visited state.
**All 2,770 executed actions belonged to the student; zero teacher overrides.**
The data retained teacher command labels even when the student failed. These are
not successful teacher trajectories or teacher-return targets for the critic.

Twelve episodes succeeded; all six crossing/diagonal episodes intruded. The batch
contained 171 substantial intervention labels. Combined data then held **13,648
rows and 869 interventions**. Integrity/partition/scheduling checks passed.
This collection changed no weights and demonstrated no new learned success.

### 7.10 Latest fit on the aggregated dataset

We refit from the same Stage 1-derived parent for 3,000 supervised updates, using
the original demonstrations plus the student-state labels. This was not another
3,000 updates continued on the weighted student's optimizer. Selected update: 2,900.

| Fixed validation | Weighted-only | Aggregated fit |
|---|---:|---:|
| Crossing | 0/20 | **0/20** |
| Diagonal | 0/20 | **0/20** |
| Head-on | 14/20 | **20/20** |
| Pedestrian total | 14/60 | **20/60** |
| Geometric intrusions | 41 | **40** |
| Stage 1 | 60/60 | **47/60** |
| Stage 1 mean formation error | 0.17805 m | **0.10634 m** |

The lower mean formation error does NOT mean preservation recovered. Thirteen
Stage 1 episodes reached acceptable endpoint geometry AND accompanied progress,
but could not complete the one-second stop hold. Mean instantaneous terminal planar
speed in those failures was approximately 0.357 m/s, above the 0.15 m/s ceiling.
All thirteen ended in recovery timeout. There were no recorded critical failures
in these fixed-suite outcome counts.

On the SAME ORIGINAL teacher holdout, balanced imitation MSE worsened
**0.01723 → 0.02121**. Crossing-intervention MSE worsened 0.04589 → 0.05331 and
diagonal-intervention MSE 0.07268 → 0.09604. Thus adding one student-state batch
has not improved even the earlier teacher-state approximation at this budget.
Comparing global scores from different combined holdout sets would have been unfair;
this comparison uses identical original rows.

The latest candidate is diagnostic only. The original Stage 1 parent is still our
successful forward-use policy. No Stage 2 model has met the promotion gate.

## 8. What the current problem actually is

### Established observations

1. The gait/full-G1 runner can execute successful Stage 1 and teacher avoidance.
2. The MLP represents a successful Stage 1 controller; Stage 2 capacity is unproven.
3. Better value estimates did not by themselves produce avoidance.
4. A successful Stage 2 teacher has not transferred successfully into the actor.
5. Lower imitation error on teacher states has not guaranteed closed-loop success.
6. Shared actor updates can damage previously mastered stopping, even with replay.
7. Current recent failures are not explained by the old resume RNG defect.

### Plausible explanations that still need tests

- **Covariate shift:** the student deviates and reaches states unlike successful
  teacher rollouts. Offline teacher queries show much larger errors there. One
  aggregated batch has not yet fixed this.
- **Conflicting/ambiguous action labels:** a teacher may choose different passing
  sides or depend on previous decisions. MSE averages target actions; an average
  between two valid evasive options can be an unsafe straight-ahead command.
  This has not yet been demonstrated by a label-consistency audit.
- **Insufficient optimization or curriculum:** Stage 1 used three progressive
  4k-update stages and two learner-data rounds. Stage 2 has used separate 3k fits,
  one aggregation round, a more complex teacher and a different balanced objective.
  These are not equal-budget/equal-procedure comparisons.
- **Interference/forgetting:** the same hidden weights support following and
  avoidance. Replay is a soft training constraint, not mathematical protection of
  the Stage 1 action function or its local feedback stability.
- **Representation/observability:** small statically scaled pedestrian features,
  nonlinear risk calculation and teacher history may make the target difficult.
  We have not proved that adding recurrence, engineered features or width is needed.

It is therefore premature to say “the MLP is too small” or “the reward is wrong.”
The current supervised updates use action-label MSE, not task reward gradients;
changing the reward would not directly fix this imitation stage. Likewise, the
teacher's success does not mean its behavior is easy for this student to approximate.

## 9. Recommended next diagnostic, not an implemented solution

Pause broad fitting and PPO. Before spending another large budget:

1. Audit teacher-label consistency, passing-side decisions and dependence on causal
   history. Check whether similar actor inputs receive materially conflicting targets.
2. Try bounded overfitting of ONE crossing and ONE diagonal DEVELOPMENT encounter,
   and execute those exact cases in full G1 dynamics. Same-case success would test
   representability/optimization; it would not demonstrate generalization or promotion.
3. Explicitly protect the working Stage 1 function. Candidate approaches include
   anchoring no-pedestrian outputs to the successful parent or an opt-in architecture
   with a frozen Stage 1 actor plus a pedestrian-masked residual. A residual could be
   exactly zero when the pedestrian mask is absent, guaranteeing preservation of the
   original mean actions in that condition. It would still need safe pedestrian
   behavior, new provenance/transition tests and stochastic/PPO validation.
4. Only after a student works in closed loop, consider current-policy critic
   preparation and bounded guarded PPO. Do not fit the critic to returns from a
   different teacher policy, bypass gates or select using final-test episodes.

Those were recommendations at the earlier review. The history audit, frozen-base
residual and two-case overfit diagnostic have since been implemented; see Section 11.

## 10. Artifacts and reading map

The authoritative editable weight registry is
[MODEL_WEIGHTS.md](bodyguard/MODEL_WEIGHTS.md). Preserve manifests and RNG sidecars;
a registry entry does not back up binary weights.

| Artifact | Location relative to repository root | Meaning |
|---|---|---|
| Selected Stage 1 model | `research/escort_rl/runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip` | Working default parent, 5k PPO |
| Stage 1 continuation | `research/escort_rl/runs/imitation_guarded_resume_20261106_192731_IgG5HN/continuation_10k/final.zip` | Valid 10k continuation; not selected over 5k |
| Original imitation actor | `research/escort_rl/runs/capacity_exploration_20261106/imitation/diagnostic_imitation_actor.zip` | BC-only comparator, not a resumable PPO archive |
| Stage 2 transition | `research/escort_rl/runs/stage2_readiness_20261107/transition_ready/initial.zip` | Stage 1-derived parent, newly activated input columns |
| Latest failed aggregate | `research/escort_rl/runs/stage2_dagger_fit_XRTTA7/aggregated/candidate_actor.npz` | Diagnostic actor parameters; requires `candidate.json`; not PPO resume |

Implementation: [observation contract](../research/escort_rl/escort_rl/observations.py),
[Stage 2 environment](../research/escort_rl/escort_rl/stage2/env.py),
[guarded PPO config](../research/escort_rl/configs/ppo_stage1_imitation_guarded_v1.yaml),
[Stage 2 config](../research/escort_rl/stage2_config/stage2.yaml),
[teacher](../research/escort_rl/diagnostics/stage2_causal_teacher.py), and
[distillation v3](../research/escort_rl/diagnostics/stage2_distillation.py).

Measured evidence, in approximate chronological order:

- [Initial 100k failure and RNG evidence](../research/escort_rl/PPO_100K_ANALYSIS_2026-11-06.md)
- [RNG fix and reward audit](../research/escort_rl/PPO_RESUME_REWARD_AUDIT_2026-11-06.md)
- [A/B at 30k](../research/escort_rl/PPO_30K_COMPARISON_2026-11-06.md)
- [Capacity/imitation and exploration tests](../research/escort_rl/CAPACITY_EXPLORATION_2026-11-06.md)
- [Objective C cold-start comparison](../research/escort_rl/PPO_TRACKING_V3_RESULTS_2026-11-06.md)
- [Imitation initialization and unguarded collapse](../research/escort_rl/IMITATION_PPO_2026-11-06.md)
- [Guarded 5k](../research/escort_rl/PPO_IMITATION_GUARDED_RESULTS_2026-11-06.md)
- [Guarded 10k and RNG continuation proof](../research/escort_rl/PPO_IMITATION_GUARDED_10K_RESULTS_2026-11-06.md)
- [Stage 2 initial pilot](../research/escort_rl/STAGE2_PILOT_REVIEW_2026-11-07.md)
- [Stage 2 reward audit](../research/escort_rl/STAGE2_REWARD_AUDIT_2026-11-07.md)
- [Critic preparation](../research/escort_rl/STAGE2_CRITIC_RESULTS_2026-11-07.md)
- [PPO after critic preparation](../research/escort_rl/STAGE2_AFTER_CRITIC_REVIEW_2026-11-08.md)
- [Improved teacher](../research/escort_rl/STAGE2_TEACHER_V2_RESULTS_2026-11-08.md)
- [Initial student fit](../research/escort_rl/STAGE2_TEACHER_FIT_REVIEW_2026-11-08.md)
- [Weighted fit](../research/escort_rl/STAGE2_DISTILLATION_V3_REVIEW_2026-11-08.md)
- [Student-state collection](../research/escort_rl/STAGE2_DAGGER_COLLECTION_REVIEW_2026-11-08.md)
- [Latest aggregated-fit failure](../research/escort_rl/STAGE2_AGGREGATED_FIT_REVIEW_2026-11-08.md)

Latest implementation regression evidence: **150 tests passed**, zero failures/skips.
Those tests establish software contracts; they do not turn failed Stage 2 validation
into learned mastery. This documentation task ran no new policy fitting, rollout,
ROS/DDS operation or final-test evaluation.

## 11. Frozen-base residual update: first validated learning progress

The explicitly authorized residual experiment now preserves the selected Stage 1
actor completely and learns only corrections. Its base always sees zeroed pedestrian
slots. The applied residual is exactly zero when no pedestrian is valid. A new
51-input branch uses current observation-derived risk features plus detached base
actions, while the external observation remains 40 values. See the
[architecture guide](../research/escort_rl/STAGE2_RESIDUAL_V1.md).

Teacher audit showed sensitivity to hidden remembered velocity. Reconstructing it
from the observed previous command reproduced all but one of 5,394 recorded labels
within the audit threshold. This explicitly versioned observable teacher still
passed 60/60 pedestrian validation. It is not a silently changed expert.

One crossing and one diagonal development case both failed with zero residual,
then both succeeded after 4k BC updates and after successive DAgger rounds at 8k
and 12k cumulative supervised updates. The final residual's fixed validation is
crossing **12/20**, diagonal **17/20**, head-on **9/20**, total **38/60** with
22 geometric intrusions. Stage 1 is **60/60**, with all **11,892 observation/action
pairs exactly equal** to the selected base's original traces. This removes the
previous no-pedestrian forgetting failure for deterministic behavior, but does not
preserve head-on behavior because the residual is active when a pedestrian is valid.

The experiment changes architecture, derived features, teacher history handling and
progressive fitting budget together; it is not an isolated causal ablation. It is
promising imitation evidence, NOT Stage 2 promotion or readiness for PPO. Next is
broader training-only approach/head-on coverage with the base still frozen; do not
train on fixed validation failures. Full report and artifact hashes:
[residual results](../research/escort_rl/STAGE2_RESIDUAL_RESULTS_2026-11-08.md).

## 12. November 9 update: broader frozen-residual continuation

V2 retained the two development anchors and added one training-only case per approach
side for crossing, head-on and diagonal. The original v1 inference/teacher code was
unchanged. Its network weights, Adam moments and Torch/local batch RNG were restored;
8,000 new BC updates plus one student-only DAgger batch reached **20,000 cumulative
residual supervised updates**, zero residual PPO updates. All eight development
cases passed both post-fit rounds.

Fixed-suite results are crossing **12/20**, head-on **19/20**, diagonal **14/20**,
total **45/60 (75%)**, with fifteen geometric intrusions. Stage 1 remains **60/60**
and all **11,892 observation/action pairs are still exactly preserved**. Head-on
improved sharply, but diagonal regressed from v1's 17/20. Crossing gained two cases
and lost two, hiding churn behind an unchanged aggregate. Positive generator-side
crossings score 4/12 versus 8/8 on the negative side.

No gate was relaxed and no PPO/promotion follows. Next proposed work is more
training-only crossing/diagonal variation with head-on retention. Do not copy fixed
validation cases into demonstrations or treat eight development cases as robust
mastery. See [coverage v2 report](../research/escort_rl/STAGE2_RESIDUAL_COVERAGE_RESULTS_2026-11-09.md).

## 13. Targeted variation v3: 95% success, safety gate still fails

The next phase added predefined training-only crossing speed/timing and diagonal
side/angle strata, retaining all eight v2 cases and both head-on examples. It kept
the same per-phase cap of 8,000 residual-only updates, restoring v2 weights/Adam/RNG
and reaching **28,000 cumulative BC updates**, zero residual PPO. Architecture,
features, teacher, loss, gait, rewards and validation gates stayed fixed. The data
set grew to 10,285 labeled states from sixteen development cases; the student
progressed 12/16 -> 14/16 -> 16/16 across the planned BC/DAgger rounds.

On the original fixed suites: **20/20 crossing, 20/20 head-on, 17/20 diagonal**, total
**57/60 (95%)**, versus v2's 45/60. Stage 1 remains 60/60 with all 11,892 original
observation/action pairs exactly preserved. Only one geometric intrusion remains,
plus two client losses, all positive-side oblique diagonals. The numerical success
thresholds now pass, but the ZERO-INTRUSION gate does not; there is no Stage 2
promotion or authorization for PPO.

This supports targeted coverage rather than a blind longer same-case run, but the
absence of a matched same-parent/no-new-data control limits causal attribution.
Next proposed work is new training-only slow/oblique positive-side diagonal
passing/recovery examples while retaining the current successful families. Do not
copy validation trajectories, weaken client-loss/clearance rules or treat a reused
validation-suite score as final-test generalization. Full report:
[variation v3 results](../research/escort_rl/STAGE2_RESIDUAL_VARIATION_RESULTS_2026-11-09.md).

## November 9: bounded residual PPO experiment is now available

Residual variation v3 reached 57/60 pedestrian successes and retained 60/60 Stage 1; one intrusion still prevents promotion. Perfect imitation is not required to investigate whether PPO can improve this residual. The user authorized a conservative, maximum 5k-transition research pilot, with the exact Stage 1 actor frozen. The adapter imports the existing residual, adds a new critic and fresh PPO optimizer, and uses masked latent Gaussian exploration. No-pedestrian behavior remains exact. Critic calibration and post-training paired validation are mandatory. This is implemented tooling, not demonstrated PPO improvement. See [the run guide](../research/escort_rl/RESIDUAL_PPO_PILOT_V1.md) for equations, budgets, provenance, verification and commands.
