# Full-G1 Gymnasium escort environment — Stage 1

**Implemented and verified; no PPO training was implemented or started.** The environment owns
one isolated full-G1 MuJoCo instance and drives the unchanged frozen AGILE policy through
body-frame velocity commands. No ROS/DDS, Nav2, TF publisher, live simulator or hardware is used.

## 1. Inspection and reuse decision

The existing standalone `research/escort_scenarios/escort_scenarios/generator.py` imports and
runs directly in the host Python 3.12 venv. It depends on PyYAML and the standard library,
requires an explicit seed/frame/bounds/limits, and already supports **zero pedestrians** with
one stable `client` ID. Its `Scenario.pose_at(id, elapsed_seconds)` returns analytic position,
velocity, acceleration, heading and angular velocity; each quintic segment starts/ends at rest.

Its existing turns are stop–rotate–go, rather than smooth moving turns. Therefore Stage 1 uses
a **thin scalar-progress adapter**: the unchanged generator produces 1D route distance s(t),
and the wrapper maps it to a straight line or constant-curvature circular arc. A gentle arc
has tangent heading theta(s)=theta0+k*s, speed |s_dot|, angular rate k*s_dot and acceleration
`s_ddot*tangent + k*s_dot²*normal`. Stops use existing waypoint dwell segments. No competing
general-purpose generator, mailbox, per-episode ROS bundle, or original-generator edit was made.

The runner API remains `reset()`, `command(vx,vy,wz)`, `step_physics()`, `advance(duration)` and
`measure()`. Added optional reset-only planar spawn placement and a per-instance actor driver.
Default behavior remains compatible with existing runner tests. The corrected pelvis-origin
velocity contract is preserved; frozen gait normalization/scaling was not changed.

Gymnasium was absent. After explicit user approval, installed **gymnasium==1.2.2** into the
existing venv, with its required cloudpickle 3.1.2 and farama-notifications 0.0.6 dependencies.
Existing NumPy 2.2.6, MuJoCo 3.3.6 and ONNX Runtime 1.20.1 pins were unchanged. PyYAML 6.0.3,
pytest 8.3.5 and matplotlib 3.10.1 were already available. Stable-Baselines3 was absent and
was not installed. Actual versions are recorded in `runs/stage1_environment_20261105/environment_versions.json`.

## 2. Files created or modified

All source changes are confined to `research/escort_rl/`:

| File | Responsibility |
|---|---|
| `configs/stage1.yaml` | Versioned Stage 1 probabilities, geometry/motion/spawn ranges, static observation scales, reward and outcome parameters |
| `escort_rl/stage1_scenarios.py` | Seeded adapter around existing generator; line/arc transforms, continuous geometry/motion validation, stable client identity and evaluator metadata |
| `escort_rl/observations.py` | Pure causal 40-float actor vector with named slices, explicit normalization and empty pedestrian masks |
| `escort_rl/rewards.py` | Bounded formation cost, new accompanied robot progress, penalties, completion guard and reason-coded outcomes |
| `escort_rl/env.py` | `G1EscortEnv`, Gym RNG/reset/step ownership, paused warmup, command mapping, client mocap updates and diagnostics |
| `escort_rl/mujoco_runner.py` (modified) | Optional actor names/driver and reset-only xy/yaw placement with consistent private startup-weld target; no gait/control/scheduling changes |
| `tools/stage1_scenarios.py` | Export resolved evaluator trajectories/metadata, validate and plot |
| `tools/verify_stage1_env.py` | Official checker, causal diagnostic follower, zero/random policies, seed suite, logs/plots, reproducibility and measured throughput |
| `tests/test_stage1_scenarios.py` | Determinism, variation, all families, client-only, continuity/limits, dwell, distribution and rejection |
| `tests/test_stage1_rewards.py` | Progress/farming/standing/racing guards, formation, loss/recovery/success/failure/truncation |
| `tests/test_stage1_env.py` | Official checker, observations, true counters, complete reset/independence, causal boundary, zero-agent failure, error channels |
| `pyproject.toml` (modified) | Optional pinned Stage 1 dependencies and test import path for the existing scenario package |
| `README.md`, this report | Current status, schemas, evidence and reproduction instructions |

Original Grove controllers, simulator source/assets, AGILE ONNX, reference manifest and
existing scenario generator remain unchanged. The checkpoint hash remains
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
Existing scenario bundles, prior experiments and pre-existing user modifications were preserved.

## 3. Supported scenarios and geometry

Default effective probabilities: **straight 50%, gentle_turn 25%, stop_start 25%**.
A deterministic 1,000-seed sampler check produced 484/267/249 respectively. Forced-family
reset options are available for testing and do not add privileged features to observations.

| Parameter | Configured range/default |
|---|---|
| Client count / pedestrians | One stable `client` / zero pedestrians |
| Route length | 3–5 m |
| Sampled peak speed limit | 0.15–0.25 m/s |
| Client initial x/y | Independently sampled in [-0.5,0.5] m |
| Route heading | [-pi,pi] rad |
| Gentle-turn curvature magnitude | 0.06–0.16 m^-1, randomized sign |
| Combined acceleration limit | 0.16 m/s² |
| Turn-rate limit | 0.06 rad/s |
| Stop location | 35–65% of route length |
| Stop duration | 1–3 s |
| Preferred robot offset | [-1.5,0] m in client heading frame |
| Initial client-frame formation perturbation | +/-0.12 m on each axis |
| Initial robot-heading perturbation | +/-0.12 rad |
| Validated Stage 1 arena | Native world x/y [-8,8] m, 0.5 m geometry margin |

**Speed is a sampled peak bound, not a constant cruise or mean speed.** Existing quintic
profiles accelerate from rest and decelerate to rest; mean route speed is recorded explicitly.
The stop/start family additionally decelerates to an interior dwell and resumes. This makes
the initial tasks conservative; it does not establish following a continuously cruising
0.25 m/s client under all conditions.

Only the existing **flat plane** scene is accepted. Its frozen source contains no nonrobot
obstacles; this is not an arbitrary-navigation-world free-space planner. Line/arc coordinate
extrema, the preferred escort path and actual reset spawn are checked continuously against
bounds/margins. Speed, tangential/centripetal acceleration and turn bounds are analytic;
dense sampled tests independently check them. Invalid ranges/probabilities/scales, too-short
routes, impossible geometry, increased caps or unsupported scenes fail with explicit errors.

Metadata records outer family, seed, resolved inner-generator configuration, route/spawn/
speed/stop/curvature parameters, generator/adapter versions and source hashes, Python/NumPy
versions, config hash, duration, mean speed, stable identity and an evaluator-only scenario ID.
The stateless trajectory is queried at the current active simulation time; stop headings
remain the last tangent heading, including at zero velocity.

## 4. Reset and physics ownership

`reset(seed=None, options=None) -> (observation, info)` calls Gymnasium seeding, then samples
an independent episode seed and family from the configured probabilities. Supported test
options are `family` and optional explicit `episode_seed`. The ordinary Gym seed and resolved
episode seed are distinct and logged; exact explicit scenario replay can use
`reset(seed=42, options={'family':'straight','episode_seed':42})`.

Each reset:

1. Samples and validates a client-only episode.
2. Restores private MuJoCo data, controller holds, safety/command/phase state and gait history.
3. Places the free base **only inside reset**, preserving the original height/upright state;
   updates only the private startup-weld reference to match the sampled xy/yaw. The default
   weld is restored before every reset. Original on-disk assets and physical parameters do
   not change.
4. Initializes only the `escort_client` mocap slot.
5. Executes 2.0 s / 1,000 physics steps of zero-command stabilization while the client route
   and active episode clock remain at zero. Warmup earns no reward and no progress.
6. Requires weld release, no fall/emergency, height >=0.65 m and upright cosine >=0.95;
   otherwise reset fails explicitly without silently resampling a successful episode.
7. Starts active time and initializes reward high-water/loss/previous-action state from the
   actual post-warmup physical state.

Warmup naturally fills **new** gait histories after they are cleared; the active episode does
not inherit old histories. With the established phase, warmup executes 99 gait calls. During
rollout, G1 movement results only from the existing ONNX, holds/safety/PD and MuJoCo contacts.
The client driver writes only its own noncolliding mocap pose, before/after integration for
consistent current-time sampling. It does not respond to G1 actions. Robot mocap/root writes
are absent from `step()`.

## 5. Actions and timing

Action: `Box(-1,1,shape=(3,),dtype=float32)`, mapped to body commands with configured caps
[0.6 m/s, 0.6 m/s, 0.9 rad/s]. Finite out-of-space actions are clipped and logged; malformed
or nonfinite actions are rejected before physics. No acceleration filter was introduced.
Raw normalized actions, clipped actions, raw/mapped commands, the explicitly unfiltered
command and the actual float32 applied command are separate diagnostics.

Each ordinary `step()` holds one command for **0.2 s** with **100 actual physics steps** at
0.002 s and **10 actual gait calls** after warmup. The existing timeout/saturation/PD/safety
semantics remain intact. All 3,898 recorded ordinary steps were audited for these counters.
An irrecoverable error reports actual completed counts rather than pretending all steps ran.

## 6. Causal observation schema

Observation: **40 float32 values**, fixed shape, finite, `Box(-1,1)` after explicit static
scales and clipping. Saturation counts are logged. No learned/running normalizer is added;
the frozen gait's embedded normalization remains separate.

| Slice | Features | Static scale |
|---|---|---|
| 0:2 | Planar client-minus-robot position in robot-heading axes | 16 m |
| 2:4 | Planar (client world velocity minus robot pelvis-origin world velocity), rotated into robot-heading axes | 2 m/s |
| 4:6 | sin/cos(client heading minus robot heading) | Unit |
| 6:8 | Desired offset in client heading axes | 3 m |
| 8:10 | Client validity/freshness | 0/1, exact synchronous pose proxy in Stage 1 |
| 10:13 | Pelvis-origin velocity vx/vy and body angular z in actual pelvis axes | [2 m/s,2 m/s,4 rad/s] |
| 13:16 | Torso projected world unit gravity | Unit |
| 16:19 | Measured torso angular velocity in torso axes | 10 rad/s |
| 19:22 | Previous applied high-level command normalized by command caps | Unit |
| 22:40 | Three slots, each [relative x,y,vx,vy,radius,valid] | All zeros/invalid in Stage 1 |

“Robot-heading axes” is the planar yaw frame; actual pelvis/torso axes include tilt. Relative
velocity is the rotated velocity **difference**, not the derivative of rotating-frame relative
position (it does not include the -omega-cross-r transport term). These conventions are
explicit rather than mixing them silently. All robot linear measurements refer to the pelvis
origin; torso angular velocity is invariant to rigid-body point.

The pure builder receives only the current client state, current robot/torso measurements,
previous applied command and static configuration. It receives **no scenario, route, family,
endpoint, progress or future-waypoint object**. Tests alter future trajectory metadata/tail
without changing the current state and confirm the actor vector is unchanged.

Future-route exports and `scenario_metadata()` are evaluator-only. `info['evaluation']`
contains reward/evaluation diagnostics, not actor features. Future actor/critic/normalizer
integration must consume the observation vector only. This is an API/data boundary, not a
claim that arbitrary Python code cannot inspect a private environment object. Client data
are headless simulator-pose proxies, not real camera tracking or learned social response.

## 7. Rewards and outcomes

Client-frame formation error follows the requested definition:
`e = R(client_heading).T @ (robot_xy - client_xy) - [-1.5,0]`.

Default components, individually logged:

- Formation: `dt * (exp(-(norm(e)/0.75)^2)-1)`, bounded in [-dt,0]. Perfect standing formation
  is not a positive living-reward source.
- Progress: 4 times **new robot route progress**, only while formation error <=2 m, client
  distance <=3 m, robot remains behind the client and the client is valid. Credit is capped
  by current client progress. The raw robot high-water mark advances even when ineligible,
  so racing ahead cannot bank credit for later and repeated traversal cannot be rewarded.
- Command change: -0.02 times squared change of normalized applied command.
- Torso stability: -0.02 * dt * nonnegative torso tilt measure.
- Terminal success: +20 once.
- Terminal failure: -100; explicitly penalizes task failure rather than rewarding an early exit.

Success requires client arrival, formation error <=0.75 m, and both actual robot advance and
**credited accompanied advance** >=60% of route length. Thus a stationary G1 cannot succeed,
and racing ahead then waiting in final formation cannot replace escorting the route. Routes
are at least 3 m. Late recovery is allowed for 5 s after client arrival.

`terminated=True` reasons: `success`, `fall`, `supervisor_failure`, `client_lost` (distance >3 m
or invalid client continuously for 3 s), `recovery_timeout`, `out_of_bounds`, `physics_failure`.
`truncated=True`: `time_limit` at the configured 100 active seconds. Failure/success takes
priority over simultaneous time limit. Terminal calls cannot be repeated without reset.

Falls are checked at each executed physics substep. Clock rollback/engine auto-reset and
irrecoverable RuntimeError/FloatingPointError/native MuJoCo FatalError end explicitly with a
failure penalty and finite last causal observation with freshness invalidated. Native-fatal
and fall branches were tested with labeled injected fixtures, not claimed as real recorded
failures. Startup failures raise before an episode begins. No silent early cutoff improves a
recorded return.

## 8. Verification actually executed

- **45 escort_rl tests passed, zero skipped**, including all prior frozen-gait/runtime tests.
- **28 existing generator/adapter Python tests passed**, confirming unchanged source compatibility.
- Gymnasium's official `gymnasium.utils.env_checker.check_env` passed in unit tests and CLI.
  Rendering checks are skipped because this environment is intentionally headless.
  [Official checker API](https://gymnasium.farama.org/api/utils/#environment-checking).
- Stable-Baselines3 checker: **not run because SB3 is absent**; no PPO/framework install was made.
- Scenario validation: all families, deterministic state/metadata, different seeds, zero
  pedestrians, continuous bounds/speed/acceleration/tangent/heading, true dwell, probabilities
  and invalid-input rejection.
- Environment validation: spaces/dtypes/finiteness, warmup pause/release, actual counts/caps,
  complete integration-state reset and deterministic traces, independent instances, actor
  action-independence, no future features, zero-agent noncompletion, success/failure/loss/
  recovery/timeout, reward farming guards and explicit failure channels.
- Python compilation and `git diff --check` passed. Original reference hashes still verify.

The required **25-episode full-G1 suite** used five forced episodes per family plus ten mixed.
It deliberately includes zero/random policies; these are not hand-selected successful runs.

| Episode group | Zero | Random | Diagnostic follower | Outcomes |
|---|---:|---:|---:|---|
| Straight forced | 1 | 1 | 3 | 3 success, 2 client loss |
| Gentle-turn forced | 1 | 1 | 3 | 3 success, 2 client loss |
| Stop/start forced | 1 | 1 | 3 | 3 success, 2 client loss |
| Mixed | 1 | 1 | 8 | 8 success, 2 client loss |
| **Total** | **4** | **4** | **17** | **17 success, 8 client loss** |

Actual sampled family totals were 9 straight, 8 gentle-turn and 8 stop/start. All **17
follower episodes succeeded** (6/6/5 by actual family); every zero/random episode ended in
client loss and none falsely completed. No reset stabilization, supervisor, fall or
irrecoverable-physics failures occurred. The failed zero/random episodes are controller/task
failures, not evidence that their scenario geometry is physically infeasible.

The diagnostic follower uses **only the same actor observation and command bounds**, with
simple current-state proportional tracking/feedforward and explicit fixed dead-zone
compensation. It does not read future route/evaluator information, is not PPO and is **not
Reactive Baseline A**. Seventeen development episodes do not establish generalization or a
trained policy's performance.

Follower per-episode mean formation error ranged **0.0509–0.0660 m**; per-episode p95 ranged
**0.0870–0.1226 m**. Minimum recorded pelvis height across all policies was **0.7064 m** and
minimum upright cosine **0.99655**.

All 3,898 transition logs were audited: finite bounded observations, zero pedestrian slots,
100 physics/10 gait calls per ordinary step, correct reward sums and valid terminal conditions.
Final guard refinements reject credit beyond 3 m and require accompanied completion progress;
all recorded successes already satisfy those guards and no recorded reward was affected.
The 25 traces therefore retain their rewards/outcomes; extra synthetic tests cover the newly
explicit edge cases. See `completion_guard_audit.json`.

## 9. Artifacts and plots

Fresh experiment root: `runs/stage1_environment_20261105/`.

- `scenario_*_42/`: resolved evaluator CSV, metadata, validation, XY and motion plots for each family.
- `straight_smoke/`: first successful physical follower rollout plus exact trace reproduction.
- `acceptance_suite/manifest.json`, `summary.json`: resolved configuration, frozen runtime
  provenance, checker status, all episode outcomes and measured performance.
- Per episode: `scenario_evaluator_metadata.json`, **separate** `observations.jsonl` and
  `evaluation.jsonl`, summary and diagnostics.
- `acceptance_suite/plots_v2/`: reviewed plots with XY, longitudinal/lateral/norm formation
  error against zero target/final tolerance, commands versus measured velocity and all reward
  components. These supplement rather than overwrite original figures.
- `log_audit.json`, `completion_guard_audit.json`: complete log checks.
- `tests_release.txt`, `existing_scenario_tests.txt`, versions/schema/source-hash files:
  exact validation/dependency/provenance records.

Representative plots:
[gentle-turn follower](runs/stage1_environment_20261105/acceptance_suite/plots_v2/episode_007_gentle_turn_follower_112/diagnostics.png),
[stop/start follower](runs/stage1_environment_20261105/acceptance_suite/plots_v2/episode_012_stop_start_follower_122/diagnostics.png),
[zero-action failure](runs/stage1_environment_20261105/acceptance_suite/plots_v2/episode_000_straight_zero_100/diagnostics.png).

## 10. Measured environment throughput

One real environment, 25 resets/episodes, **3,898 active transitions**, 779.6 active simulated
seconds plus 50.0 warmup seconds. Actual counts: **414,800 physics steps, 41,455 gait calls**.
The count differs from a simplistic 50 Hz multiplication because cold startup starts
inference at 20 ms; warmup executes 99 calls and each ordinary action executes ten.

| Metric | Measured |
|---|---:|
| Total timed rollout/reset/log time | 115.592 s |
| High-level transitions / wall second | **33.72** |
| Physics steps / wall second | 3,588.49 |
| Gait calls / wall second | 358.63 |
| Simulated seconds / wall second, including warmup | 7.177 |
| Reset p50 / p95 | 0.2712 / 0.2770 s |
| Step-call p50 / p95 | 0.02769 / 0.02973 s |

Includes scenario sampling/current-time mocap updates, observations, rewards, outcomes,
stabilization reset and flushed per-step JSONL. Excludes model loading, checkers and plotting.
Reset latency includes **two simulated seconds of stabilization**, unlike the earlier pure
runner state-reset microbenchmark. Two-worker environment throughput was not measured; the
prior 85.84 transitions/s bare-runner result must not be relabeled as this environment's rate.
No optimizer/training timing or GPU speedup is claimed.

## 11. Limitations and stop boundary

Full numerical trajectory parity with the asynchronous live Grove stack remains unverified.
The same frozen checkpoint, full dynamics/contacts, tested reset semantics and corrected
velocity point are preserved; reference response agreement and earlier tensor parity remain
useful evidence. Scheduler event ordering/freshness and delayed live emergency switching are
still declared approximations, not silently resolved by the Gymnasium wrapper.

Stage 1 is an obstacle-free bounded flat scene with a nonresponsive, noncolliding client pose
proxy. It is not navigation-world/camera-perception validation, physical human-contact safety,
Stage 2–4 curriculum, an evaluated final baseline, or real-person escort performance. The
current clients have conservative quintic peak-bound profiles; broad disturbances, alternative
gaits and higher client speeds remain untested. Verification seeds are development checks,
not final-test seeds or checkpoint-selection evidence.

**This milestone stops at the Stage 1 environment and verification. No PPO training,
optimizer, curriculum promotion, checkpoint training or live integration was implemented.**

## 12. Exact offline commands

From the host, using the existing approved venv and both source packages:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export ESCORT_PY="$PWD/research/escort_rl/.venv/bin/python"
export PYTHONPATH="$PWD/research/escort_rl:$PWD/research/escort_scenarios"
```

Generate/validate/export a new scenario (change family/seed/output as desired):

```bash
"$ESCORT_PY" research/escort_rl/tools/stage1_scenarios.py \
  --config research/escort_rl/configs/stage1.yaml \
  --family gentle_turn --seed 42 \
  --output research/escort_rl/runs/stage1_scenario_manual_001
```

One physical diagnostic follower episode with trace reproducibility:

```bash
"$ESCORT_PY" research/escort_rl/tools/verify_stage1_env.py \
  --config research/escort_rl/configs/stage1.yaml \
  --family straight --seeds 42 --policy follower --reproducibility \
  --output research/escort_rl/runs/stage1_follower_manual_001
```

Other control modes/families (new output paths):

```bash
"$ESCORT_PY" research/escort_rl/tools/verify_stage1_env.py \
  --family stop_start --seeds 123 --policy zero \
  --output research/escort_rl/runs/stage1_zero_manual_001
"$ESCORT_PY" research/escort_rl/tools/verify_stage1_env.py \
  --family mixed --seeds 123 124 --policy random \
  --output research/escort_rl/runs/stage1_random_manual_001
```

Full required 25-episode suite and all tests:

```bash
"$ESCORT_PY" research/escort_rl/tools/verify_stage1_env.py \
  --config research/escort_rl/configs/stage1.yaml \
  --suite --reproducibility --seeds 101 \
  --output research/escort_rl/runs/stage1_acceptance_manual_001
"$ESCORT_PY" -m pytest research/escort_rl/tests -q
"$ESCORT_PY" -m pytest research/escort_scenarios/tests/test_generator.py \
  research/escort_scenarios/tests/test_mujoco_adapter.py -q
```

Every tool refuses an existing output directory. No Docker/ROS launch, new rosbag, dependency
installation or training command is needed to repeat these offline checks.
