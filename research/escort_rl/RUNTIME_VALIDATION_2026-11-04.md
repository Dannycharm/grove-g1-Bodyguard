# Runtime validation — 2026-11-04

**Recommendation: FIX FIRST.** The existing isolated runner now executes stable, deterministic
full-G1 dynamics at useful throughput. It is a viable backend candidate, but it is not yet a
validated reference-compatible backend for the proposed 0.40–0.50 m/s escort task. Forward speed
saturates around 0.41 m/s under the unchanged 0.60 m/s command cap, low-speed translation has a
marked dead zone, velocity measurement conventions need correction, and C++/live trajectory
parity remains unestablished. This evidence does not justify switching the physics backend.
Do not start Gymnasium/PPO implementation as though these gates had passed.

## 1. Scope, environment and evidence

Used the existing `research/escort_rl/.venv` (Python 3.12.3). Installed versions exactly match:
NumPy 2.2.6, MuJoCo 3.3.6, ONNX Runtime 1.20.1, pytest 8.3.5, matplotlib 3.10.1. No dependency
was installed, downloaded or upgraded. No live ROS/DDS process was launched, commanded, reset
or reconfigured. No original Grove controller, simulator, checkpoint or model asset changed.
No Gymnasium, PPO, reward or curriculum implementation was added.

Machine: AMD Ryzen 9 7900X, 12 physical / 24 logical CPUs. CPU affinity was not pinned; this
is measured local-machine performance, not an isolated-host capacity guarantee.

Fresh evidence directory: `runs/runtime_validation_20261104/` (ignored experiment data; existing
runs and datasets were preserved). Key files:

- `environment.json`, `manifest.json`, `contract_and_startup.json`: environment, exact frozen
  reference hashes, ONNX signature, compiled model and actuator mapping/gain/startup checks.
- `tests.txt`: final test output.
- `smoke.jsonl`: independent 30 s stand/walk/stop trace.
- `commands/`: all 22 committed trials, manifest/config, raw JSONL, summaries and 25 plots.
- `long_holds/`: 24 supplemental 20 s forward holds, summaries and direct-stop measurements.
- `parity_inputs.txt`, `parity_python*.txt`, `python_repeatability.json`: open-loop Python
  fixture and repeated ten-output results, **not** a C++ comparison.
- `benchmarks/`: three trials each of one and two workers, manifests and flushed JSONL;
  `process_independence.json` checks byte-identical trajectory logs across trials/worker counts.
- `throughput_summary.json`, `command_table.md`, `profile/`: derived performance/command
  tables and instrumented CPU profile.
- `velocity_origin_check.json`, `velocity_frame_check.json`: independent sensor/native velocity
  checks, including a diagnosed COM/origin and inertial/body-axis distinction.

The historical implementation-stage `VALIDATION.md` is preserved; its dependency blockers
have been superseded by this report.

## 2. Tests actually executed

The original suite ran unchanged first: **14 passed, zero skipped**, including both previously
unexecuted runtime functions. The earlier three skips represented two dependency checks and
one skipped module; with dependencies present that module collects two tests.

Added three focused runtime tests and reran everything: **17 passed, zero skipped, zero
failures** (4.44 s). No test failure required a gait, physics, gain or scheduler change.

Executed coverage:

- Exact checkpoint/reference-file identity, 12 input / 10 output names, shapes and float32
  types, 29 observation joints, 14 action joints and 269 carried feedback floats.
- Actual ONNX inference, history updates/reset and independent policy state.
- Caps/timeout, blend/decimation, emergency latch, hold gains and PD equations.
- Compiled actuator-to-joint mappings, all 29 original control ranges, unit gearing and
  graph-produced gains (Kp exact; Kd within 1e-7 float32 conversion tolerance).
- Cold reset of complete MuJoCo `mjSTATE_INTEGRATION` state, including warm start, applied
  forces, controls, equality/mocap state, plus gait histories and controller/command/schedule
  state; repeat trajectories and all ten final feedback/output tensors match bitwise.
- Independent model/data and interleaving of two different command streams.
- First inference/release phase, ten seconds of stable standing and actual contacts.
- Measurement dwell logic and parity-comparator discrepancy/alignment rejection.

The separate 30 s smoke trial used 10 s standing, 15 s at vx=0.35, then 5 s stop. There were
no sampled falls or emergency latches. Additional full-suite and extended-trial evidence is
below. These are deterministic flat-ground simulation tests, not hardware safety evidence.

## 3. Frozen contract, physics and startup

Checkpoint SHA-256 reverified:
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
The manifest verifies original code, model, scene and mesh bytes. This establishes checkout
identity, not bytes or parameter overrides loaded by an existing live container.

Compiled model: nq=50, nv=49, 50 bodies, 107 geoms and 29 body actuators; original Dex3 joints remain separately held through
`qfrc_applied`. Actual compiled timestep is **0.002 s**, Newton solver (2), Euler integrator (0), 100 iterations, tolerance 1e-8,
gravity [0, 0, -9.81] m/s². Full compiled solver/integrator,
gravity, damping, armature, friction, gearing, limits and mapping arrays are recorded in
`contract_and_startup.json`; no engine option or model parameter was changed.

Measured first inference is applied before integration at **0.020 s**, visible in the next
state at 0.022 s. The startup weld is released before integration at **0.504 s**, after 500
driven bridge events, visible in the next state at 0.506 s. Observed physical contact counts
range from zero while welded/airborne to eight. Startup joint positions change under motor
control; rollouts never write robot base qpos/qvel or use robot mocap motion.

The deterministic scheduler remains the existing approximation: 5 ms controller and 1 ms
bridge events are rounded up to 2 ms physics boundaries. Controller lateness is 0/1 ms;
multiple bridge events can sample the same state. Body controllers activate together and
`mj_forward` refreshes sensors after each integration. These choices are measured internally,
not established as equivalent to asynchronous ROS/DDS timing.

## 4. Command-response results

Executed the **unchanged 22-case suite**: 594 simulated seconds / 297,000 physics steps,
29,700 recorded 50 Hz samples. Each case cold-resets, stabilizes/stands, holds its command for
8 s, halves it for 2 s, stops for 5 s, makes a small correction and stops again. No trial had
a detected fall or emergency latch. Across the suite, minimum recorded pelvis height was
**0.7001 m**, minimum upright cosine **0.99486**, and contact count reached eight.

The original logger's `body_velocity` is the pelvis **inertial COM velocity expressed in its
inertial axes**. These measured values are reported transparently here; see the important
origin/frame finding below. Mean forward COM versus pelvis-origin speed was checked
independently and does not explain the sustained speed deficit.

Selected original-suite steady values (mean final 2 s before deceleration):

| Applied vx (m/s) | Achieved vx (m/s) | First 50% crossing (s) |
|---:|---:|---:|
| 0.00 | 0.0002 | n/a |
| 0.10 | 0.0019 | 0.16 |
| 0.20 | 0.1595 | 0.22 |
| 0.35 | 0.2840 | 0.30 |
| 0.50 | 0.3913 | 0.38 |
| 0.60 | 0.4045 | 0.66 |
| -0.10 | 0.0001 | 0.14 |
| -0.35 | -0.2649 | 0.22 |
| -0.60 | -0.3870 | 0.32 |

The first-crossing definition is **not settling latency**: the 0.10 command produces a
transient then nearly stops despite continuous command refresh. Calling its 0.16 s crossing
“responsive low-speed walking” would be misleading.

| Command | Achieved main-axis mean | Relevant cross-axis mean |
|---|---:|---:|
| vy=+0.10 / -0.10 m/s | +0.0014 / -0.0012 m/s | near-stationary |
| vy=+0.35 / -0.35 m/s | +0.2451 / -0.2441 m/s | vx about 0.016–0.018 m/s |
| vy=+0.60 / -0.60 m/s | +0.5224 / -0.5174 m/s | vx about 0.023–0.026 m/s |
| wz=0.10 rad/s | 0.1049 rad/s | vy=-0.0143 m/s |
| wz=0.30 rad/s | 0.2755 rad/s | vy=-0.0022 m/s |
| wz=0.60 rad/s | 0.6229 rad/s | vy=-0.0344 m/s |
| wz=0.90 rad/s | 0.9259 rad/s | vy=-0.0593 m/s |
| wz=-0.60 rad/s | -0.6176 rad/s | vy=+0.0305 m/s |
| vx=0.35, wz=0.30 | vx=0.2828; wz=0.3249 | vy=-0.0265 m/s |
| vx=0.50, wz=0.60 | vx=0.3760; wz=0.6727 | vy=-0.0515 m/s |

Body angular z approximates Euler yaw rate while nearly upright; it is not asserted identical
for arbitrary tilt. Pure-yaw first crossings were 0.02 s at the suite's sampling resolution;
this is not evidence of zero physical/communication latency.

All cases and cross-axis values are in `command_table.md`. Commanded-versus-achieved plots:
[forward](runs/runtime_validation_20261104/commands/vx_curve.png),
[lateral](runs/runtime_validation_20261104/commands/vy_curve.png),
[yaw](runs/runtime_validation_20261104/commands/yaw_rate_curve.png).
The per-case `case_00.png` through `case_21.png` show requested/applied/measured time series.

### Longer holds and direct stops

Supplemented the suite without changing it: commands 0.10, 0.125, 0.15, 0.175, 0.40, 0.45,
0.50 and 0.60 m/s, each following 2/4/6 s standing, then 20 s hold and 5 s direct stop. All
24 trials remained stable (696 simulated seconds). The final **10 s** means were:

| Requested vx | Mean achieved vx | Range of three trial means |
|---:|---:|---:|
| 0.100 | 0.00069 | 0.00064–0.00073 |
| 0.125 | 0.04382 | 0.04310–0.04490 |
| 0.150 | 0.07257 | 0.06945–0.07528 |
| 0.175 | 0.11275 | 0.09321–0.12554 |
| 0.400 | 0.34062 | 0.33935–0.34213 |
| 0.450 | 0.36967 | 0.36875–0.37026 |
| 0.500 | 0.39314 | 0.39284–0.39336 |
| 0.600 | 0.40722 | 0.40630–0.40809 |

These trials vary stationary history; they are not independent policy training seeds or a
broad robustness evaluation. One-second moving-average vx at the 0.60 cap ranged from
**0.3948 to 0.4191 m/s**, showing little recovery headroom even for a 0.40 m/s client.
The transition between 0.10 and 0.125 commands shows strong low-speed attenuation/dead-zone
behavior; an exact breakpoint is not identified by these discrete samples.

The original suite's half-speed-then-zero stops from forward commands 0.20–0.60 settled in
0.56–0.76 s and 0.029–0.100 m. Those are **not full-speed direct-stop distances**. In the
supplemental direct stops, after sustained commands:

| Prior command | Stop time range (s) | Planar stopping-path range (m) |
|---:|---:|---:|
| 0.40 | 0.60–0.66 | 0.100–0.118 |
| 0.45 | 0.52–0.88 | 0.115–0.139 |
| 0.50 | 0.64–0.76 | 0.096–0.108 |
| 0.60 | 0.50–0.76 | 0.096–0.116 |

Stopping means planar speed and body angular z both below 0.05 for a continuous 0.5 s. Time
and path end at the **start** of that dwell, so confirmation occurs 0.5 s later. A low forward
command can still show a nontrivial measured stop time/path due to gait/yaw transients;
zero steady forward speed is not proof of motionless instantaneous state.

**Task implication:** software caps remain ±0.60 m/s x/y and ±0.90 rad/s yaw. These are not
achieved-speed bounds. The gait has not demonstrated reliable support across client speeds
0.40–0.50 m/s: 0.40 is marginal at maximum command, and 0.45–0.50 exceeds measured sustained
forward capability in this runner. There is no validated catch-up margin. Do not raise caps
or infer a higher attainable speed from export metadata.

### Velocity-convention finding

`measure()` combines pelvis-origin position with `mj_objectVelocity(... mjOBJ_BODY, local=1)`.
MuJoCo's body velocity uses the body inertial COM and inertial orientation; here the COM is
[0, 0, -0.07605] m relative to the pelvis, with a small inertial/body rotation. An independent
check against the IMU site at the pelvis origin found:

- `frame_vel` rotated to pelvis axes matches native IMU-site velocity within 1.11e-16.
- Transforming COM world velocity using `v_site = v_COM + omega × (p_site - p_COM)` matches
  the native site velocity within 1.11e-16; rotating by `ximat` reproduces the current local
  body logger within the same tolerance.
- In the diagnostic walking window, current logged vx averaged 0.41926 m/s versus pelvis-site
  vx 0.41884 m/s. Instantaneous differences can reach about 0.0555 m/s in the longer trials;
  a 20 ms finite-difference comparison had up to 0.0211 m/s RMS (including averaging error).

This is a measurement-contract issue, not an ONNX-input or physics instability. The gait
already consumes `imu_quat`/`imu_gyro` as in the reference. No runner/controller semantics were
changed during validation. Before freezing PPO observations/parity, explicitly select and
log pelvis-origin/body-axis velocity, or consistently use a documented COM convention in
both systems. Do not compare mixed origins against a 0.05 m/s trajectory RMS gate.
MuJoCo source/API references: [3.3.6 API](https://mujoco.readthedocs.io/en/3.3.6/APIreference/APIfunctions.html#mj-objectvelocity)
and [3.3.6 source](https://github.com/google-deepmind/mujoco/tree/3.3.6/src/engine).

## 5. Parity, reset and independence

The 1,000-observation synthetic fixture executed twice through Python ORT, including resets
at ticks 0, 1, 257 and 800. **All ten output tensors matched exactly, maximum absolute
difference 0.0.** Closed-loop reset tests also compare complete integration state and all
feedback outputs exactly. This is Python reproducibility, not Python/C++ parity.

The original C++ `AgilePolicy` source, tensor names/order and CPU/thread configuration were
inspected. Matching `libonnxruntime.so.1.20.1` exists in the Python wheel, but the C++ headers
were not found in the searched workspace/home/opt/system include paths. An actual probe
syntax compilation failed immediately with:

```text
fatal error: onnxruntime_cxx_api.h: No such file or directory
```

Thus **no C++ tensor discrepancy has been measured**. The existing probe and fixture remain
ready; full ten-output inference parity is pending compatible development headers. Controller
coverage currently consists of Python fixture tests/source comparison and graph gain checks,
not execution of the complete C++ safety/controller chain against matching fixtures.

No suitable repeated Grove reference trajectories were found. Existing escort observation
logs contain proxy/pelvis observations but not matching initial states, commands, full joint
state/targets/gains and event timing. They cannot establish dynamics parity. The audit's
proposed tolerances (tensor 1e-5 absolute/relative; joint RMS 0.02 rad; planar velocity RMS
0.05 m/s; 10 s endpoint/yaw 0.10 m/rad; no additional falls) remain **unpassed**.

Reset/state independence evidence is positive within this runner:

- Same command/reset sequences reproduce complete integration state and feedback bitwise.
- Interleaved environments have independent model/data and histories.
- The entire 500-transition worker-0 JSONL is byte-identical across all three one-worker trials
  and all three two-worker trials. Worker 1, which receives a different lateral command, also
  matches its own logs across trials and differs from worker 0. This verifies process/state
  independence for those deterministic streams, not cross-platform bitwise portability.

Remaining semantic differences include the synchronous activation/scheduler/sensor freshness
choices and the emergency path: the runner holds last safe targets at 100/2 gains; the live
reference can later switch to a measured-pose freeze at 10/1 via ROS timer/service. Stable
trials did not exercise that post-emergency switch. It needs an explicit acceptance decision
or reference-compatible implementation before safety-equivalence claims.

## 6. Throughput and collection budgets

Three fresh sequential trials per worker count; each worker collects 500 transitions at
5 Hz (100 simulated seconds, 50,000 actual physics steps), with cold reset every 50 transitions.
Each executes **4,990 actual gait calls**, not an assumed 5,000: inference starts at 20 ms
and the final boundary inference is outside each cold 10 s rollout. Aggregate evidence is
450,000 physics steps, 44,910 gait calls and 4,500 transitions across nine worker executions.

Included: real G1 contacts, three scripted noncolliding mocap proxies (client plus two
pedestrians), measurements, per-transition flushed JSONL and cold resets. Actor updates are
synthetic benchmark motion, not complete scenario generation, perception, human reactions or
PPO observation/reward work. No early physics skipping occurred. No sampled worker state
reported a fall/emergency. Load/1 s warmup is outside steady timing; end-to-end rate is also
reported. Workers are not barrier-synchronized, so the existing aggregate steady rate uses
sum of transitions divided by the longest worker interval; end-to-end rates are conservative
for these short jobs.

| Metric | 1 worker | 2 workers aggregate |
|---|---:|---:|
| Transitions/s, three trials | 43.921, 42.750, 42.716 | 85.781, 86.160, 85.564 |
| Mean ± sample SD transitions/s | 43.129 ± 0.686 | 85.835 ± 0.302 |
| Physics steps/s, mean | 4,312.9 | 8,583.5 |
| Gait calls/s, mean | 430.4 | 856.6 |
| Simulated seconds / wall second | 8.626 | 17.167 |
| End-to-end transitions/s, mean | 40.515 | 79.826 |
| Reset p50 / p95 (ms) | 0.136 / 0.147 | 0.133 / 0.146 |
| Transition latency, mean worker p50 / p95 (ms) | 23.159 / 24.150 | 23.151 / 23.948 |
| CPU use | about 100% of one logical CPU | about 100% per worker |
| Peak RSS per worker | about 573–574 MiB | about 573–574 MiB each |

Physics-step p50 was approximately 211–217 microseconds; p95 approximately 354–362
microseconds across workers. Reported mean worker quantiles are not pooled latency quantiles.
Summed RSS for two workers is about 1.12 GiB; it double-counts shared pages and is not PSS.
No worker counts beyond two were tested.

Collection-only estimates using the mean steady rates:

| Transitions | 1 worker | 2 workers |
|---:|---:|---:|
| 100,000 | 0.644 h (38.6 min) | 0.324 h (19.4 min) |
| 500,000 | 3.220 h | 1.618 h |
| 1,000,000 | 6.441 h | 3.236 h |

For 1M transitions, the observed trial-rate range implies 6.324–6.503 h (one worker) or
3.224–3.246 h (two). These are timing-repeatability ranges, not statistical confidence
intervals. Neural optimization, validation/evaluation, future escort observation/reward work,
policy exchange and deployment supervision are excluded. Extrapolation assumes comparable
contact complexity and logging. Throughput during a future PPO run must be remeasured.

**Practical choice among tested counts: two workers**, about 1.99× the single-worker rate.
The diagnostic 100-transition profile spent about 49.3% of instrumented wall time in native
`mj_step` plus the explicitly repeated `mj_forward`; Python control/holding overhead also
matters (scalar finger `np.clip` calls alone about 19.8% cumulative). Frozen-gait inference
including its Python wrapper was about 4.7%; JSON encoding plus flush was under 0.3% on this
filesystem. CPU physics/control, rather than ONNX or logging, dominates this workload.
Profiling adds overhead and is not used for throughput estimates. No optimization or removal
of reference behavior was performed.

## 7. Ranked blockers and decision

1. **Task capability:** measured forward saturation gives no reliable headroom for
   0.40 m/s following and insufficient sustained speed for 0.45–0.50 m/s. Translation near
   0.10 m/s is almost absent; intermediate low commands are nonlinear/history-dependent.
   Confirm these response limitations in the unchanged Grove reference before blaming the
   checkpoint rather than the port. Keep caps unchanged.
2. **Reference parity:** C++ dev headers are missing; the ten-output C++ comparison and full
   controller fixture execution remain open. No matching repeated reference trajectories
   establish dynamics parity or acceptable asynchronous scheduling deviation.
3. **Measurement contract:** fix/document velocity origin and orientation, consistently align
   reference frames/timestamps, and retain explicit transient-versus-sustained latency and
   direct-versus-decelerated stop definitions before freezing observations/evaluation.
4. **Failure/reset coverage:** deterministic cold resets and stable flat-ground trials pass,
   but post-emergency switching equivalence, navigation contacts/obstacles and perturbation
   recovery are not validated. Synthetic proxies do not establish pedestrian collision safety.
5. **Performance scope:** throughput is adequate for a small pilot on this machine, but
   scalability beyond two workers and actual PPO integration costs remain unmeasured.

**FIX FIRST** means retain this isolated full-MuJoCo approach and frozen AGILE while closing
those gaps. Stable physics and useful collection rates support continuing validation. They
do not approve the backend for the stated full-speed escort task. A gait-policy comparison
may become justified if the existing reference confirms the same speed/dead-zone limits;
there is currently no evidence that switching the physics backend solves them. No fallback
policy has been executed or compared in this milestone.

## 8. Exact reproduction and remaining manual commands

From repository root, without installation or live-stack access:

```bash
export PYTHONPATH="$PWD/research/escort_rl"
export PY="$PWD/research/escort_rl/.venv/bin/python"
"$PY" -m pytest research/escort_rl/tests -v
"$PY" research/escort_rl/tools/run.py commands --output research/escort_rl/runs/validation_commands_002
"$PY" research/escort_rl/tools/validation_benchmarks.py --output research/escort_rl/runs/validation_benchmarks_002
"$PY" research/escort_rl/tools/validate_long_holds.py --output research/escort_rl/runs/validation_long_holds_002
"$PY" research/escort_rl/tools/profile_rollout.py --output research/escort_rl/runs/validation_profile_002
```

Use new paths; all output tools refuse existing directories. The suite, supplemental protocol
and analysis are separate so the original walking-policy comparison suite stays unchanged.

### Proposed C++ dependency acquisition — approval required; NOT executed

Missing development headers include `onnxruntime_cxx_api.h` and its matching C API/inline/float
support headers. Use the complete Linux x64 **CPU 1.20.1** distribution, matching the Python
runtime and reference build. The official versioned asset was verified on the
[1.20.1 release asset list](https://github.com/microsoft/onnxruntime/releases/expanded_assets/v1.20.1).
No system installation or environment upgrade is required. Proposed local commands:

```bash
set -e
mkdir -p research/escort_rl/.deps
mkdir research/escort_rl/.deps/ort_1_20_1
curl --fail --location \
  https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
  --output research/escort_rl/.deps/ort_1_20_1/runtime.tgz
tar -xzf research/escort_rl/.deps/ort_1_20_1/runtime.tgz -C research/escort_rl/.deps/ort_1_20_1
export ORT_ROOT="$PWD/research/escort_rl/.deps/ort_1_20_1/onnxruntime-linux-x64-1.20.1"
```

After approval/acquisition, compile and compare in a new directory:

```bash
mkdir research/escort_rl/runs/parity_runtime_002
"$PY" research/escort_rl/tools/parity.py fixtures unused research/escort_rl/runs/parity_runtime_002/inputs.txt
c++ -std=c++20 -O2 -I"$ORT_ROOT/include" -Iworkspace/src/g1_controllers/include \
  research/escort_rl/tools/cpp_policy_probe.cpp workspace/src/g1_controllers/src/agile_policy.cpp \
  -L"$ORT_ROOT/lib" -Wl,-rpath,"$ORT_ROOT/lib" -lonnxruntime \
  -o research/escort_rl/runs/parity_runtime_002/cpp_probe
research/escort_rl/runs/parity_runtime_002/cpp_probe \
  workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx \
  research/escort_rl/runs/parity_runtime_002/inputs.txt research/escort_rl/runs/parity_runtime_002/cpp.txt
"$PY" research/escort_rl/tools/parity.py python research/escort_rl/runs/parity_runtime_002/inputs.txt research/escort_rl/runs/parity_runtime_002/python.txt
"$PY" research/escort_rl/tools/parity.py tensor research/escort_rl/runs/parity_runtime_002/python.txt research/escort_rl/runs/parity_runtime_002/cpp.txt
```

### Operator-controlled reference session — NOT executed

Only in an explicitly operator-approved, already managed reference session, verify configured
rates/path/ownership with read-only commands:

```bash
source /opt/ros/jazzy/setup.bash
source /root/workspace/install/setup.bash
ros2 node list
ros2 topic list -t
ros2 param get /controller_manager update_rate
ros2 param dump /agile_controller
ros2 param dump /locomotion_safety_controller
sha256sum /root/workspace/install/g1_controllers/share/g1_controllers/policy/unitree_g1_velocity_e2e.onnx
```

Resolve any nonempty `model_path` override and verify that file instead. Use the existing
operator-owned, collision-monitored command source for the committed suite; do not launch a
second publisher or reset the running robot. An approved read-only bag can preserve available
observations (verify listed topics first and choose a new path):

```bash
timeout --signal=INT 30s ros2 bag record \
  --output /root/data/escort_logs/runtime_reference_observations_001 \
  /joint_states /cmd_vel /g1_sensor_relay/actor_ground_truth
```

This bag is **insufficient on its own** for full parity: a separately approved reference
recorder/exporter must capture physics-time joint/base states, correct quaternion/velocity
frames, targets/gains, startup/reset state, commands and timing under matched initial
conditions. No such complete live recorder was found or injected here; there is no truthful
one-command full-reference capture to claim. Measure repeated-reference variability first.
After that export is available and explicitly aligned:

```bash
"$PY" research/escort_rl/tools/parity.py trajectory isolated_aligned.jsonl reference_aligned.jsonl
```

Do not infer dynamics parity merely from identical neural outputs or unrelated old proxy logs.
Stop here; Gymnasium/PPO and any policy switch require a separate milestone.

## Files changed for this validation milestone

- Modified `tests/test_runtime.py`: three additional real-runtime contract/reset/startup tests.
- Added `tools/validation_benchmarks.py`, `tools/validate_long_holds.py`,
  `tools/profile_rollout.py`, `tools/analyze_runtime_validation.py`: reproducible validation
  protocols, profiling and persisted-evidence analysis.
- Added this report; updated README status and linked the preserved historical VALIDATION report.
- Added `.deps/` to the research-local ignore list for the proposed, still-uninstalled native
  development package. Fresh experiment artifacts remain under the already-ignored `runs/`.

No runner, frozen-gait, controller-reference, command-suite, reference-manifest or original
Grove source/asset semantics were modified. All pre-existing changes outside this package
were left intact. Python compilation and `git diff --check` also passed.
