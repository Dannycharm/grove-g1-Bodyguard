# Pelvis velocity correction and Grove reference parity

**Decision: retain AGILE while completing moving-reference validation; full-dynamics parity
and PPO readiness remain unestablished.** The measurement mismatch is corrected, original
C++ inference now passes the declared tensor tolerances, and the supplied bag has been
exported offline. That bag contains only stationary behavior and cannot answer whether
Grove reproduces the isolated runner's forward-speed saturation.

## Answers to the milestone questions

| Question | Evidence-based answer |
|---|---|
| Does original Grove AGILE reproduce the speed limitation? | **Unknown.** The supplied bag has zero command messages and negligible measured movement. No matched forward/lateral/turn/stop reference trial exists in the available data. |
| Does the velocity correction materially affect results? | **No for sustained forward speed.** At the 0.60 command cap the corrected long-hold mean is 0.40648 m/s versus legacy 0.40722 m/s; the difference is -0.00073 m/s. Point/axis differences still matter for instantaneous tracking and parity. |
| Are Python/C++ gait outputs equivalent? | **Yes on the tested observations:** all ten arrays match exactly over 1,000 synthetic and 1,698 actual-runner observations, including reset cases; declared atol/rtol remain 1e-5. This is inference parity, not whole-controller/dynamics parity. |
| Are isolated/reference dynamics sufficiently compatible for training? | **Not established.** Missing matched moving trials, exact execution/safety signals, clock alignment, initial-state evidence and scene identity prevent acceptance. |
| Feasible Stage 1 client speeds? | **Provisional design range 0.15–0.25 m/s**, client-only and with stops, after the remaining reference gates. This leaves measured catch-up margin below about 0.41 m/s. It is not trained-task success evidence or approval to implement PPO now. Do not use 0.40–0.50 m/s as an established capability. |
| Retain AGILE or compare another policy? | Retain it through reference confirmation. If matched Grove trials confirm this saturation/dead-zone behavior and 0.45–0.50 m/s clients remain required, compare another frozen pretrained G1 gait in a separately scoped experiment. No alternative was run here, and changing physics backend is not justified by this evidence. |
| Acceptable approximations versus blockers? | A causal, explicitly labeled headless state velocity and separately logged Euler yaw rate are defined experimental conditions. Noncolliding scripted proxies remain geometry-only actors. Scheduling/sensor-age, safety-switch, reset/start and unmatched-scene differences are **not yet accepted**; they require measured reference variability and an explicit decision. Missing commands/targets/gains cannot be filled with zeros or guessed values. |

## 1. Changes and velocity contract

The canonical measurement now uses the existing **pelvis-origin `imu` site**, with linear
velocity expressed in **pelvis/body axes** (`site_xmat`/`xmat`, not inertial `ximat`). This is
source-verified against Grove's `sensor_publisher.cc::baseStateLoop()` and relay
`publishBaseState()`: `frame_vel` world velocity is rotated to the IMU site's body axes;
`imu_gyro` supplies body angular velocity. The G1 IMU site is at [0,0,0] with identity relative
orientation, and the runner rejects a different mount instead of silently changing the point.
The existing `/g1_sensor_relay/base_state` interface can therefore supply the preferred twist.

Schema v2 records include:

| Field | Measurement point / axes / units |
|---|---|
| `body_velocity` | Compatibility vector **[pelvis-origin vx, vy, body angular z]**, pelvis axes, m/s and rad/s |
| `linear_velocity_pelvis`, `linear_velocity_world` | Same pelvis origin, full 3D vector in pelvis / native MuJoCo world axes, m/s |
| `angular_velocity_pelvis`, `angular_velocity_world` | Angular velocity, pelvis / world axes, rad/s; independent of rigid-body point |
| `euler_yaw_rate_world` | Derivative of world ZYX yaw, rad/s; separately valid/invalid near the yaw singularity |
| `com_linear_velocity_world`, `com_linear_velocity_pelvis`, `com_linear_velocity_inertial` | Original inertial COM point, explicitly distinguished world / pelvis / inertia axes, m/s |
| `legacy_com_velocity` | Preserved old [COM vx, vy, inertial-axis angular z] vector for comparisons |

Rows and run manifests declare `measurement_schema_version=2` and
`velocity_convention=pelvis_origin_in_pelvis_axes_v2`. Legacy vectors carry
`pelvis_inertial_com_in_inertial_axes_v1`. World coordinates are native `mujoco_world`,
quaternions are wxyz, and provenance is isolated headless MuJoCo state, not camera perception.
Additional COM position/point metadata is logged. Future actor observations should select
only the intended causal fields; no future actor routes or evaluator-only labels were added.

Rigid-body verification uses
`v_origin_world = v_COM_world + omega_world × (p_origin_world - p_COM_world)`.
Axes are rotated independently with pelvis `R.T` or inertial `I.T`. Body angular z is not
silently renamed to Euler yaw rate; the latter is differentiated from the orientation
projection and marked invalid when that projection is singular.

**Preserved:** gait inputs, checkpoint, gains, command caps, scheduling, physics, contacts,
startup/release and reset. No original controller, simulator or model asset was modified.
The frozen checkpoint is still SHA-256
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.

## 2. Offline tests and behavior preservation

Final test result: **27 passed, zero skipped**. New checks cover:

- Analytic COM/origin point transforms under simultaneous translation and turning.
- World/pelvis/inertial-axis rotations and native MuJoCo site/body API agreement.
- Actual full-G1 turning measurements against `frame_vel` and `imu_gyro`.
- Measurement calls leave the complete MuJoCo integration state unchanged.
- Euler yaw derivative versus body angular z, including a tilted pose and singularity.
- Explicit, distinct steady-hold and full-zero-command windows in response metrics.
- Exact float32 round-trip for recorded inference fixtures.
- Recorder name-based joint mapping, missing/nonfinite data, evaluator-only actor filtering,
  missing-command semantics, clock rollback/epoch boundaries and lack of ROS graph/command APIs.
- Trajectory comparison rejects missing/nonfinite required arrays and missing/mixed conventions.

The previous 17 runtime/contract/reset/independence tests continue to pass. Python compilation
and whitespace checks also pass. An intermediate test asserted which of two simultaneously
invalid fixture conditions should fail first; that test fixture was corrected to isolate the
convention error. No physics/control change was used to pass a test.

The entire unchanged 22-case command suite was rerun. **All 29,700 samples** of time, measured
joint q/dq, pelvis pose, applied command, targets, gains, contact count and failure status
match the prior rollout **bitwise**. Preserved legacy velocities agree with the prior logger
within 1e-14. Thus the changed API adds read-only measurements without changing the executed
full-G1 dynamics. Cold-reset and interleaved-worker determinism still pass.

## 3. Corrected command response

Fresh evidence is in `runs/velocity_parity_20261105/`; no prior experiment was overwritten.
The 22-case suite contains 594 simulated seconds / 297,000 physics steps, followed by 15
supplemental trials (435 simulated seconds) using the existing long-hold utility: commands
0.10, 0.40, 0.45, 0.50 and 0.60 m/s, each after 2/4/6 s stand, 20 s hold and 5 s stop.
**No detected falls or emergency latches** occurred in either set.

Corrected original-suite means (final 2 s of the full-command hold):

| Requested command | Corrected achieved main-axis velocity |
|---|---:|
| vx=0.10 | 0.00200 m/s |
| vx=0.20 | 0.15840 m/s |
| vx=0.35 | 0.28363 m/s |
| vx=0.50 | 0.39095 m/s |
| vx=0.60 | 0.40428 m/s |
| vx=-0.35 / -0.60 | -0.26464 / -0.38654 m/s |
| vy=+0.35 / -0.35 | +0.24618 / -0.24450 m/s |
| vy=+0.60 / -0.60 | +0.52177 / -0.51699 m/s |
| body angular z command +0.10 / +0.30 | +0.10491 / +0.27551 rad/s |
| body angular z command +0.60 / +0.90 | +0.62291 / +0.92588 rad/s |
| vx=0.35, angular z=0.30 | vx=0.28299 m/s; angular z=0.32493 rad/s |
| vx=0.50, angular z=0.60 | vx=0.37430 m/s; angular z=0.67272 rad/s |

The largest change in any steady linear component across these cases is 0.00191 m/s. This is
small for the sustained-speed conclusion, but not grounds to mix points/axes in instantaneous
trajectory comparisons.

Long-hold **final 10 s** means across the three stationary histories:

| Command vx | Corrected mean | Trial-mean range | Legacy COM mean |
|---:|---:|---:|---:|
| 0.10 | 0.000723 | 0.000673–0.000769 | 0.000691 |
| 0.40 | 0.339865 | 0.338625–0.341325 | 0.340619 |
| 0.45 | 0.368947 | 0.368006–0.369441 | 0.369671 |
| 0.50 | 0.392428 | 0.392245–0.392627 | 0.393136 |
| 0.60 | **0.406485** | **0.405651–0.407313** | 0.407217 |

The low-speed translation attenuation remains. A transient first 50% crossing is not sustained
response or settling latency. The corrected maximum observed sustained forward speed remains
approximately **0.41 m/s under the software 0.60 m/s cap**. No higher-cap experiment was made;
this is not a claim about an unconstrained maximum possible gait speed.

Corrected half-speed-then-zero stopping results for forward commands 0.20, 0.35, 0.50 and 0.60
were respectively 0.60/0.80/0.56/0.72 s and 0.0297/0.0752/0.0507/0.1011 m. The threshold is
pelvis-origin planar speed and body angular z each below 0.05 for 0.5 s; the reported stopping
time ends at the **start** of the dwell, with confirmation 0.5 s later. These are not direct
stops from full speed because the suite includes a 2 s half-command phase.

Metrics now take an explicit zero-command time separately from the steady-hold end. This
removes ambiguity when comparing legacy/corrected stop metrics across the deceleration phase.
Authoritative paired metrics are in `measurement_comparison.json`; the initial interim
`commands/summary.json` was retained, not overwritten, and its nested legacy stop window was
superseded by that paired recomputation. The primary corrected stops use the proper window.

[Corrected versus legacy curves](runs/velocity_parity_20261105/measurement_comparison.png)
and existing per-case time series are available. The third curve is **body angular z**, not
Euler yaw rate; separate Euler-rate means are logged in the paired metrics.

## 4. Original C++ inference parity

Matching native **ONNX Runtime 1.20.1** headers and libraries were already present in
`.deps/ort_1_20_1/onnxruntime-linux-x64-1.20.1/`. No installation/download occurred. The existing
probe compiled with the **unchanged original** `agile_policy.cpp` and header; `ldd` verified
it resolves the local 1.20.1 library. Library/header/checkpoint/probe/source hashes and fixture
reset indices are recorded in `parity_provenance.json`.

Executed comparisons:

| Input evidence | Observations / resets | All 10 outputs |
|---|---|---|
| Existing seeded synthetic fixture | 1,000; resets 0, 1, 257, 800 | max absolute difference **0.0**, all pass |
| Actual runner-consumed float32 observations, standing/forward/lateral/turn/combined/stop | 1,698; resets 0 and 849 | max absolute difference **0.0**, all pass |

This includes absolute joint-position targets, Kp, Kd, previous raw actions and all six history
outputs. Original tolerances stay atol=rtol=1e-5. Reset/history behavior is compared through
actual output arrays, not only names/shapes.

A useful failed intermediate check was retained: the first actual-observation exporter wrote
raw float64 sensor values with nine decimal digits. Re-parsing those values changed **1,254
float32 input entries**, with maximum difference 4.77e-7; feedback amplified this to action
position differences up to 2.58e-5. Python and C++ reading that same old text still matched
exactly, proving the mismatch came from export precision. The fix casts to the actual consumed
float32 **before** nine-digit serialization. Actual live-runner outputs before/after this
export fix are bitwise identical. The corrected comparison passes exactly without changing
policy inference, checkpoint or tolerances. See `fixture_precision_diagnosis.json`.

Neural-output parity is now established for these tested inputs. The complete ROS safety/hold/
SDK/controller chain and full trajectories are **not** established as equivalent by this result.

## 5. Supplied reference bag and implemented exporter

The existing bag was available at
`data/escort_logs/runtime_reference_20261105_021529/`. It was opened read-only using existing
`rosbag2_py` and generated `g1_msgs`; no ROS graph/node was initialized. The host's installed
message overlay was stale, so the already-built host `rosidl_generator_py` module/library was
selected only in that export process. No rebuild or installation was performed.

Actual decoded evidence agrees with the supplied bag metadata:

- 5,557 JointState samples, all with valid named state for the 29 body joints.
- 280 valid pelvis snapshots, one epoch, 27.900 s native-time span.
- **Zero `/cmd_vel` messages** and no recorded base-state/inferring/safety channels.
- Backward native-time pelvis planar speed: mean 8.13e-8 m/s, max 2.67e-7 m/s.
- Largest body-joint position range: 2.98e-7 rad; largest joint speed: 1.42e-6 rad/s.
- Relay-stamp minus native-time offset varied by about **0.407 ms** over this static recording.

This is a stationary partial reference, not a forward-speed or stopping trial. An absent
command stream is **unknown**, not fabricated zero commands. The bag does not prove that
AGILE was actively inferring, which exact checkpoint was loaded, the applied target/gains,
or the safety/weld state. The supplied navigation/FAST-LIO launch arguments are useful
operator context, not proof of complete matched initial conditions or source identity.

The new exporter writes:

- `reference_events.jsonl`: separate normalized asynchronous robot-only streams, preserving
  recorder/header/native timestamps and ROS-to-wxyz quaternion conversion.
- `pelvis_backward_estimates.jsonl`: explicitly estimated, causal pose differences at native
  capture times, not a replacement for native twist.
- `coverage.json`: actual counts, validity, missing signals and no-full-parity flag.
- `provenance.json`: before/after input file hashes, exporter/normalizer hashes and evaluator-only
  labels. Bag bytes and metadata were unchanged.

Actor rosters/poses and sensor alignment transforms are dropped rather than passed toward any
future actor observation. No commands, clients, reset methods, live subscriptions, publishers
or controller switches exist in the exporter. Static capability tests and the executed
read-only export verify the intended interface boundary. The final export is
`runs/velocity_parity_20261105/reference_export_final/`.

## 6. Full-dynamics protocol and remaining operator work

[REFERENCE_COLLECTION.md](REFERENCE_COLLECTION.md) contains the exact existing-container
verification, fresh-directory/QoS recording and offline export commands, plus the complete
matched-trial protocol. No live command was executed by the agent.

The existing interfaces cover measured joints, pelvis pose and (with `/base_state`) the
correct velocity point/axes. They do **not** expose exact executed high-level commands,
final locomotion motor targets/gains, internal emergency/blend state, weld state or a shared
native timestamp for every controller/state sample. A read-only subscriber to existing
Unitree DDS `rt/lowcmd`/`rt/lowstate` is the smallest optional next observation layer for
motor commands, but would still not expose all internal/physics timing. That observer and
any further bounded native event trace are **proposals only**, not injected instrumentation.

Before accepting dynamics compatibility:

1. Match checkpoint, complete robot/scene identity and compiled engine/controller options.
   The current isolated command suite uses **flat** ground; the supplied reference is an
   actor-augmented **navigation** bundle. The runner's current selector does not load that
   arbitrary bundle. Exact scene equivalence has not been demonstrated.
2. Establish equivalent initial pose/velocity, joint/hold/gait histories, control phases and
   startup/release or a documented settled-start condition under operator control.
3. Execute the same representative forward/lateral/turn/stop sequences through the existing
   exclusive, collision-monitored command owner. Record raw/received/executed commands and
   overrides. No new publisher/replay utility is supplied.
4. Obtain at least five repeated reference runs per representative condition to characterize
   asynchronous variability and failures. Preserve all runs and no cherry-picking.
5. Align prospectively by verified command/control/native-time events, with uncertainty;
   do not fit response-dependent shifts, assume fixed relay offset or extrapolate through loss.
6. Compare common pelvis-origin/body-axis velocities, separate body angular z/Euler yaw,
   joints, targets/gains, base motion, stops and failures. Keep declared numerical tolerances
   unless an explicitly documented acceptance decision changes them.

The trajectory comparator rejects incomplete/nonfinite arrays and incompatible velocity
conventions. It reports additional angular/base errors and explicitly returns
`full_dynamics_parity_established=false`; satisfying its numeric fixture gate alone cannot
replace provenance/initial-condition/clock/repeatability review. The supplied asynchronous
export cannot be silently converted into a complete aligned trajectory with invented targets.

**Still requiring an operator-controlled run:** loaded identity/startup verification; moving
matched commands; native velocity-topic capture; final target/gain/executed-command and safety
coverage; repeated-reference variability; and the eventual aligned full-dynamics comparison.
The static bag and successful neural parity do not answer these outstanding questions.

## 7. Files, tests and reproduction

Modified only this research package:

- `escort_rl/mujoco_runner.py`: cached/validated existing pelvis-origin IMU site; corrected
  read-only measurement fields/schema/manifest. Rollout/reset/inference methods unchanged.
- `escort_rl/measurements.py`: rigid-point/frame velocity extraction, separate Euler yaw,
  preserved legacy velocity, quaternion utility and explicit metric windows/convention checks.
- `escort_rl/parity.py`, `tools/parity.py`: strict trajectory input conventions/availability;
  actual-runner fixture recording with correct float32 serialization; original tensor comparison reused.
- `tools/run.py`, `tools/validate_long_holds.py`: reuse existing protocols, paired legacy metrics,
  explicit stop timing and selectable supplemental commands. Original suite/config unchanged.
- Added `escort_rl/reference_recording.py`, `tools/export_reference_bag.py` and
  `configs/reference_record_qos.yaml`: offline recorder/exporter and operator bag QoS.
- Added velocity/reference-recording tests and extended existing contract tests; README and
  this report/manual collection guide document current behavior and evidence.

No original Grove source/asset, frozen gait checkpoint, command-suite values or reference
manifest changed. No PPO/Gymnasium/reward/curriculum or new locomotion policy was added.
No dependencies, Docker lifecycle actions, commits, pushes, live simulator actions or live
ROS commands occurred.

Reproduce offline with fresh paths (existing `.venv`, no install):

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
export PYTHONPATH="$PWD/research/escort_rl"
export PY="$PWD/research/escort_rl/.venv/bin/python"
"$PY" -m pytest research/escort_rl/tests -q
"$PY" research/escort_rl/tools/run.py commands --output research/escort_rl/runs/pelvis_commands_002
"$PY" research/escort_rl/tools/validate_long_holds.py --commands 0.1 0.4 0.45 0.5 0.6 \
  --output research/escort_rl/runs/pelvis_long_holds_002
```

Original C++ parity, using the existing native distribution:

```bash
export ORT_ROOT="$PWD/research/escort_rl/.deps/ort_1_20_1/onnxruntime-linux-x64-1.20.1"
mkdir research/escort_rl/runs/pelvis_parity_002
c++ -std=c++20 -O2 -I"$ORT_ROOT/include" -Iworkspace/src/g1_controllers/include \
  research/escort_rl/tools/cpp_policy_probe.cpp workspace/src/g1_controllers/src/agile_policy.cpp \
  -L"$ORT_ROOT/lib" -Wl,-rpath,"$ORT_ROOT/lib" -lonnxruntime \
  -o research/escort_rl/runs/pelvis_parity_002/cpp_probe
"$PY" research/escort_rl/tools/parity.py fixtures unused research/escort_rl/runs/pelvis_parity_002/inputs.txt
research/escort_rl/runs/pelvis_parity_002/cpp_probe \
  workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx \
  research/escort_rl/runs/pelvis_parity_002/inputs.txt research/escort_rl/runs/pelvis_parity_002/cpp.txt
"$PY" research/escort_rl/tools/parity.py python research/escort_rl/runs/pelvis_parity_002/inputs.txt research/escort_rl/runs/pelvis_parity_002/python.txt
"$PY" research/escort_rl/tools/parity.py tensor research/escort_rl/runs/pelvis_parity_002/python.txt research/escort_rl/runs/pelvis_parity_002/cpp.txt
"$PY" research/escort_rl/tools/parity.py record-runner unused research/escort_rl/runs/pelvis_parity_002/runner_inputs.txt
research/escort_rl/runs/pelvis_parity_002/cpp_probe \
  workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx \
  research/escort_rl/runs/pelvis_parity_002/runner_inputs.txt research/escort_rl/runs/pelvis_parity_002/runner_cpp.txt
"$PY" research/escort_rl/tools/parity.py tensor \
  research/escort_rl/runs/pelvis_parity_002/runner_inputs.txt.actual_outputs.txt \
  research/escort_rl/runs/pelvis_parity_002/runner_cpp.txt
```

No missing dependency needs installation for the executed tasks. If the native distribution
is unavailable in another environment, report that before acquiring it; the pinned version
must remain 1.20.1. Historical local-acquisition instructions remain in the previous runtime
report but were not executed in this milestone.

Prior throughput is historical: the extra labeled measurement work was not rebenchmarked in
this milestone. Do not present 85.84 transitions/s as a fresh measurement of the new schema.
Stop here; moving-reference confirmation and an explicit compatibility decision precede the
future PPO-environment milestone.
