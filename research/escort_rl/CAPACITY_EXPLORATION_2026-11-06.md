# Actor capacity, exploration ablation, and opt-in tracking objective

**Completed tests 2 and 3. Prepared, but did not launch, the full objective-C experiment.**
The same 128×128 actor achieved 60/60 Stage 1 successes through diagnostic imitation.
Lower initial PPO exploration alone achieved 0/60 at 20k. This supports keeping the
actor architecture and testing a more explicit tracking/stopping objective; it does not
prove reward design is the only remaining problem or establish critic capacity.

Evidence root: `runs/capacity_exploration_20261106/`. All new outputs are separate from old
experiments. Original AGILE, physics, base environment, observations, generator, Stage 1
config and original reward files remain unchanged. No ROS/DDS/live simulator, Docker,
system installation, commit or push was used.

## 1. Capacity diagnostic: supervised imitation, not PPO

Implemented `capacity_diagnostic.py` and `tools/test_actor_capacity.py`. The teacher is the
existing causal diagnostic follower, invoked only with the actor's current 40-vector and
static configuration. No future route, family, evaluator fields or motor targets enter
supervised features or labels. This is not the final Reactive Baseline A and not a PPO result.

Protocol fixed before running:
- Same SB3 MlpPolicy actor: two 128-unit Tanh layers; 22,147 trained actor parameters,
  44,039 parameters across the saved full actor/critic policy.
- 18 teacher-controlled training episodes, followed by two rounds of 12 learner-controlled
  episodes with teacher labels on learner-visited observations. Every trajectory executes
  the original full-G1 physics/gait; no synthetic/kinematic rollouts were used.
- Independent model seed 314 and scenario stream seed 91001, train-only 0-modulo-4 scenario
  namespace, excluding reserved final-test parameter combinations.
- Three fixed stages of 4,000 supervised Adam updates each, learning rate 3e-4, batch up to
  256. Only actor parameters are optimized. No extra observation normalizer or feature scaler.
- Whole scenarios, selected by `(episode_seed//4)%5==0`, form the supervised holdout.
  The fixed 60-case RL validation suite is not used for gradients or selecting fit duration.

Actual data: **8,096 causal observations/actions from 42 unique episodes**. All 42 episodes
completed successfully. There were 32 supervised training scenarios / 6,023 rows and
10 holdout scenarios / 2,073 rows. Final normalized action MSE:

| Statistic | Result |
|---|---:|
| Train MSE | .001954 |
| Whole-scenario holdout MSE | .002099 |
| Holdout RMSE `(vx,vy,yaw)` | (.04284,.06678,.001327) |
| Physical command RMSE after caps | approximately (.0257 m/s,.0401 m/s,.00119 rad/s) |

The critic and Gaussian log standard deviations were verified **unchanged**. Saved
`diagnostic_imitation_actor.zip` has zero PPO timesteps/optimization epochs and no PPO
resume sidecars. A fresh process reloaded it and reproduced stored deterministic actions
exactly. This is deliberately not a PPO warm start or an RL-trained checkpoint.

**Closed-loop full-G1 held-out validation: 60/60 success**, 20/20 in all three families;
mean formation error **.06011 m**. The actor also later passed all twelve development
cases under the new stricter windowed hold definition (section 4).

Interpretation: a feasible Stage 1 feedback controller can be represented and executed
by this actor architecture. Widening it is not the first supported remedy. This does not
show that PPO can discover it efficiently: imitation received dense expert action labels
and 12,000 supervised updates, unlike reward-only PPO. It does not test whether the critic
can approximate long-horizon returns, or prove capacity for later pedestrian stages.

## 2. Exploration ablation: only initial log_std changed

Created `configs/ppo_stage1_low_exploration_v1.yaml`, copied from the fixed-RNG original
reward control. The sole hyperparameter change was `policy_kwargs.log_std_init=-1.0`,
initial sigma .367879 instead of 1. Same root seed 42, two workers, original reward,
40-vector, 128×128 networks, action envelope and all PPO defaults. Fresh initialization
parameter audit found **only log_std differed** from A; actor and critic weight tensors
were identical before learning.

Ran two exact 10k chunks, **20k actual PPO transitions**, 100 optimization epochs / 800
Adam steps. No larger experiment was launched. The fixed 60-case suite was evaluated
before/after each chunk as in control A.

| Held-out result | Control A, 20k | Lower exploration, 20k |
|---|---:|---:|
| Overall success | 14/60 | **0/60** |
| Straight / turn / stop-start | 5/20,5/20,4/20 | 0/20,0/20,0/20 |
| Failure breakdown | 36 recovery,10 client loss | **60 client loss** |
| Mean formation error | .64023 m | .73905 m |
| Mean actual robot advance | 3.440 m | **.242 m** |
| Sampled action-vector clipping, full training | 69.92% | **2.57%** |

The lower-noise 10k checkpoint was also 0/60. Its final per-axis standard deviations were
(.3673,.3708,.3679). Mean absolute deterministic commands were (.0353,.0457,.0822) in
m/s,m/s,rad/s, and mean measured body velocity (.0080,-.0070,.0594). The learned mean
controller barely translated; this is consistent with the previously measured low-speed
response limitation, but does not prove that dead zone alone caused the failed learning.

Stochastic training: 124 client losses, 14 recovery timeouts, two successes, no supervisor
or fall/physics-failure reason codes in 140 completed episodes. There were **144 unique
sampled seeds**, no repeated startup sequence. Actual first resumed seeds matched saved
worker continuations. Training time excluding evaluation was **324.35 s**, approximately
61.66 transitions/s, including optimizer/reset/logging/checkpoint costs.

Conclusion: lower sigma reduced saturation and supervisor failures but **did not improve
completion in this one-seed, 20k test**. It is not adopted as the new default. This is one
noise value, not a sweep or proof that exploration is irrelevant. No additional reward,
action filter, normalization or network change was mixed into this ablation.

## 3. Why an objective experiment is warranted

The actor-capacity test succeeded, whereas a large reduction in exploration did not solve
PPO. Existing A still learns motion without reliable formation recovery/stopping; original
progress reward can improve while physical formation worsens. These results justify a
versioned objective test, not a claim that one root cause has been proved. Value learning,
credit assignment, state aliasing, sample budget and other exploration choices remain
possible contributors.

The new **tracking_stop_v3** is an experimental control objective, not a reproduction of
an established humanoid-escort benchmark. Its structure follows position/relative-motion
tracking with robust bounded costs and operational stopping criteria. It is not potential-
based shaping and carries no policy-invariance or convergence guarantee. Thresholds and
weights are recorded, not claimed optimal or transferred uncritically from another robot.

The new PPO config keeps **A's original exploration** (log_std_init default 0), since the
tested lower setting performed worse. Architecture, learning rate, rollout/batch sizes,
gamma, entropy, timing, caps, physics and observations are unchanged.

## 4. Objective C equations and completion

All changes are opt-in. Original `rewards.py` and `env.py` remain byte-for-byte unchanged.
C uses an evaluator/environment adapter registered through the existing reward factory.
Original `legacy_v1` and `formation_progress_v2` remain available.

Desired world escort point:

```
p*_t = client_xy_t + R(client_heading_t) * [-1.5, 0]
e_p  = robot_xy_t - p*_t
vbar_robot = (robot_xy_t - robot_xy_previous) / dt
vbar_target = (p*_t - p*_previous) / dt
e_v = vbar_robot - vbar_target
rho(z) = z² if z <= 1, otherwise 2z - 1
```

Positions/velocities refer to the **pelvis origin in native MuJoCo world XY**. Target
velocity is the motion of the following point, including rotation of the formation offset
in turns. It uses only current/past poses, not future waypoints or desired future motion.
The first interval is initialized from the actual post-warmup pose; warmup earns no reward.
Averaging over the .2 s action interval avoids treating every gait-cycle speed pulse as
tracking error. This changes reward measurement only, not gait inputs or observations.

Reward components:

```
formation = -dt * formation_weight * min(10, rho(norm(e_p)/.75m))
velocity_tracking = -dt * 1.0 * min(4, rho(norm(e_v)/.25m/s))
```

The .75 m position scale matches the existing final formation tolerance; .25 m/s is the
Stage 1 configured peak-client-speed scale. Caps bound outlier costs. Near zero, position
cost has the same quadratic scale as the old exponential's local approximation; outside
that region it retains an error signal instead of rapidly saturating. Its reward scale
therefore differs at large errors; compare physical outcomes, not raw loss/return magnitudes.

Progress uses the v2 **desired-point/raw-motion dual highwater ledger**, preserving
anti-standing, anti-racing and no-repeat-credit safeguards. Command-change, torso-stability,
success (+20) and failure (-100) weights remain unchanged. No dense teacher action target
or new actor/critic information is supplied to PPO.

### Sampled stop-and-hold completion

After client arrival, C requires the existing <=.75 m formation and minimum raw/credited
advance conditions throughout a **one-second sampled window** (six 5Hz sample times), plus:

- Net pelvis displacement divided by window duration <=**.05 m/s**.
- Maximum sampled excursion from the first window position <=**.10 m**.
- Every window sample has instantaneous pelvis planar speed <=**.15 m/s**.
- Every window sample has absolute world Euler yaw rate <=**.10 rad/s**.

The five-second final recovery window and all existing safety/client-loss/bounds/time-limit
handling remain. The window rejects continued travel while permitting bounded gait
micromotion. It is an operational near-stop, not mathematically zero joint/body motion.

A first draft using instantaneous .05 m/s throughout rejected a follower case despite
~4 cm formation error, because .057 m/s pelvis pulses reset the hold. A short-interval
version similarly rejected small correction loops. The final definition uses the full
window's net motion and bounded excursion, not simply a raised instantaneous threshold.
Draft artifacts are retained and excluded by `EXCLUDED_OBJECTIVE_DRAFTS.json`. An intermediate
commentary mistakenly stated 12/12 follower success before checking the short-interval
summary; that result was 8/12. Only the final windowed evidence below is released.

Final full-G1 development checks (four train-namespace cases per family, separate from
the fixed 60-case suite and from capacity-fit episodes):

| Controller | New windowed task result |
|---|---:|
| Existing causal diagnostic follower | **12/12 success** |
| Diagnostic imitation actor | **12/12 success** |
| Zero command, one case/family | **0/3 success**, three client losses |

These establish feasibility on the tested development cases, not learned PPO superiority.
No full C learning experiment was executed. An eight-transition automated optimizer smoke
verified C's PPO integration, finite stepping and provenance; it is not a learning result.

## 5. Tests, files and provenance

Final regression suite: **102 passed, zero failed/skipped**, 74.61 s; fourteen non-failing
third-party Matplotlib/PyParsing deprecation warnings. `pip check`, source compilation and
`git diff --check` passed. The tests cover:
- Supervised episode-level splits and actor-only updates; frozen critic/log_std.
- Only-log_std exploration change.
- Original 45 environment/gait tests, prior PPO/RNG/reward tests and 28 generator/adapter tests.
- Correct moving/turning reference velocity, recovery versus standing, no flyby completion,
  exact one-second window, micromotion versus net travel, yaw/supervisor priority.
- Identical observations and qpos under matching actions across original/C environments,
  Gymnasium checker, 100 physics steps / ten gait calls, and actual C optimizer smoke.

New files: `escort_rl/capacity_diagnostic.py`, `escort_rl/tracking_objective.py`,
`tools/test_actor_capacity.py`, `tests/test_capacity_diagnostic.py`,
`tests/test_tracking_objective.py`, `configs/ppo_stage1_low_exploration_v1.yaml`,
`configs/tracking_stop_v3.yaml`, `configs/ppo_stage1_tracking_stop_v3.yaml`, this report.
Modified: `reward_variants.py` registers C; `ppo_training.py` records resolved objective
settings in provenance; README/PPO guide link this milestone. No base gait/controller,
runner, observation, scenario or original reward modification.

`unchanged_core.json` and release provenance record protected hashes. AGILE remains
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
Existing datasets, original archives, bundles and user modifications were preserved.
No dependencies were installed or upgraded.

Because source/config files were added, strict provenance will reject direct training
resumption of old-source checkpoints under this tree. No hash bypass/migration was added.
The C command deliberately starts fresh; the diagnostic imitation archive is not silently
used to initialize PPO. Reading old actors for evaluation is a separate allowed operation.
Do not edit source or configs between the two chunks.

## 6. Command to try objective C — not executed

This launches **20k fresh PPO transitions**, original 128×128 MLP/noise, in two 10k chunks.
It automatically saves/checkpoints and runs the fixed 60-case initial/final validations.
It does not launch ROS/Docker, pedestrians, Stage 2 or a 100k budget.

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export ESCORT_PY="$PWD/research/escort_rl/.venv/bin/python"
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios"
ESCORT_C_OUT="research/escort_rl/runs/tracking_stop_v3_$(date +%Y%m%d_%H%M%S)"

"$ESCORT_PY" research/escort_rl/tools/train_stage1_ppo.py \
  --config research/escort_rl/configs/ppo_stage1_tracking_stop_v3.yaml \
  --seed 42 --workers 2 --timesteps 10000 --chunks 2 \
  --output "$ESCORT_C_OUT"

# Same-criterion evaluation of the existing A20k actor, without training it.
"$ESCORT_PY" research/escort_rl/tools/evaluate_stage1_ppo.py \
  --config research/escort_rl/configs/ppo_stage1_tracking_stop_v3.yaml \
  --checkpoint research/escort_rl/runs/rng_fixed_A_20261106_021910/chunk_02/final.zip \
  --output "$ESCORT_C_OUT/reference_A20k_under_C"

echo "C results: $ESCORT_C_OUT"
```

C **changes completion criteria**, so its success count is not directly comparable to
A's old 14/60 endpoint score. The second command provides a common-criterion baseline.
Also compare mean/p95 formation, endpoint geometry, relative velocity, stop-window drift,
falls/supervisor failures and actual robot progress, not only reward. All C diagnostics
are in per-step evaluator JSONL; actor inputs remain the unchanged 40-vector.

The same learning budget/seed and original PPO settings make a useful first diagnostic
comparison, not a multi-seed conclusion. Do not promote curriculum or extend to 100k just
because optimizer loss decreases. C's learning effectiveness is untested until this run.
Full numerical live-Grove dynamics parity remains unverified, and scripted noncolliding
clients do not establish real-person safety or perception performance.
