# Operator-controlled Grove reference collection

Current status: the requested three round trips have been captured and analyzed. See
[MOVING_REFERENCE_ANALYSIS_2026-11-05.md](MOVING_REFERENCE_ANALYSIS_2026-11-05.md).
No repeat of this standard recording is requested. The formal matched-trial requirements
below describe later acceptance work, not a request to redo the completed operator procedure.

This protocol uses existing interfaces. The only new executable is an **offline bag exporter**;
it cannot create ROS nodes, publishers, subscriptions, service clients, reset robots or change
controller timing. Do not launch, stop or reset the current stack to run these verification
commands. Command execution remains with the operator's existing approved command owner.

## Existing signal coverage, source verified

| Existing interface | Useful signal | Limitation |
|---|---|---|
| `/joint_states` (`sensor_msgs/msg/JointState`) | Named measured joint q/dq/effort; 29 body joints selected by name | Header is ROS/controller time, state was sampled earlier through DDS; extra fingers must not shift indices |
| `/g1_sensor_relay/base_state` (`nav_msgs/msg/Odometry`) | Pelvis IMU-site pose, body-axis linear velocity and gyro angular velocity; source loop nominally 200 Hz | Header is mapped relay capture time; no explicit native simulation time; header `odom` default does not itself establish a world transform |
| `/g1_sensor_relay/actor_ground_truth` (`g1_msgs/msg/SimulatorActorSnapshot`) | Pelvis-origin pose, native simulation time and epoch | About 10 Hz in supplied bag; no twist; actor fields are evaluator-only and dropped by exporter |
| `/cmd_vel` (`geometry_msgs/msg/Twist`) | Published body-command request | No header, receipt/timeout/clipping/inference timing or execution guarantee |
| `/agile_controller/inferring` (`std_msgs/msg/Bool`) | Transient-local flag set after first successful inference | Not a per-tick heartbeat and not emergency status |
| `/locomotion_safety_controller/transition_event` | Lifecycle transitions | Not the internal emergency latch or exact blend/integrator state |
| `/diagnostics` | Existing hardware temperatures | Not full locomotion safety status |

The base-state sampler reads `frame_pos`, `frame_vel`, `imu_quat`, `imu_gyro` on the existing
`imu` site and rotates world linear velocity with `site_xmat.T`. The G1 site is located at the
pelvis origin and has identity orientation relative to it. This exactly defines the corrected
runner convention. The policy uses the same IMU quaternion/gyro; no gait input is changed.

`stampFor()` estimates a varying arrival/capture offset; it is not a fixed affine clock
calibration. Do not equate a JointState stamp, a bag timestamp, a relay-mapped stamp and native
simulation time. The exporter preserves these separately and does not auto-align them.

## Manual verification inside the existing container

The compose configuration mounts the repository at `/root/.host-repo`. These commands read
state only. First verify that each candidate topic actually has a compatible publisher:

```bash
source /opt/ros/jazzy/setup.bash
source /root/workspace/install/setup.bash
export ESCORT_REPO=/root/.host-repo
ros2 topic list -t
ros2 control list_controllers
ros2 topic info /joint_states --verbose
ros2 topic info /cmd_vel --verbose
ros2 topic info /g1_sensor_relay/base_state --verbose
ros2 topic info /g1_sensor_relay/actor_ground_truth --verbose
ros2 topic info /agile_controller/inferring --verbose
ros2 topic info /locomotion_safety_controller/transition_event --verbose
ros2 param get /controller_manager update_rate
ros2 param dump /agile_controller
ros2 param dump /locomotion_safety_controller
ros2 param get /g1_sensor_relay base_state_frame_id
ros2 param get /g1_sensor_relay base_state_odom_frame
```

Inspect actual values and publisher ownership. In particular, `nav:=true` is not evidence of
an active velocity stream or of the correct monitored command owner. The supplied bag had
zero command messages. Do not start a second velocity source, disable collision monitoring,
or treat an absent command stream as recorded zero commands.

Sample base-state and the latched inference flag without altering any controller:

```bash
timeout --signal=INT 5s ros2 topic echo /g1_sensor_relay/base_state \
  nav_msgs/msg/Odometry --once --qos-reliability reliable
timeout --signal=INT 5s ros2 topic echo /agile_controller/inferring \
  std_msgs/msg/Bool --once --qos-reliability reliable --qos-durability transient_local
```

Do not substitute `/odom` produced by FAST-LIO for simulator truth. Resolve the configured
base-state frame labels and use snapshot `mujoco_world` pose to establish the actual raw
coordinate convention. ROS quaternions are xyzw fields; exported arrays are explicitly wxyz.

## Save a new evidence directory and record

Choose a fresh directory; these commands do not overwrite the existing 20261105 bag:

```bash
export REF_EVIDENCE="/root/data/escort_logs/reference_velocity_$(date -u +%Y%m%dT%H%M%S_%N)"
mkdir "$REF_EVIDENCE"
ros2 param dump /agile_controller > "$REF_EVIDENCE/agile_parameters.yaml"
ros2 param dump /locomotion_safety_controller > "$REF_EVIDENCE/safety_parameters.yaml"
ros2 param dump /g1_sensor_relay > "$REF_EVIDENCE/relay_parameters.yaml"
ros2 control list_controllers > "$REF_EVIDENCE/controllers.txt"
ros2 topic info /cmd_vel --verbose > "$REF_EVIDENCE/command_publishers.txt"
ros2 topic info /g1_sensor_relay/base_state --verbose > "$REF_EVIDENCE/base_state_publishers.txt"
sha256sum "$(ros2 pkg prefix --share g1_controllers)/policy/unitree_g1_velocity_e2e.onnx" \
  > "$REF_EVIDENCE/package_checkpoint.sha256"
```

If `model_path` is nonempty, also hash that exact override file; package-share bytes are then
not evidence of the selected checkpoint. For the supplied operator launch, archive source
bundle identities as well:

```bash
export REF_BUNDLE=/root/data/escort_navigation_observations_seed42
find "$REF_BUNDLE" -maxdepth 1 -type f \
  \( -name '*.xml' -o -name '*.yaml' -o -name '*.json' \) \
  -exec sha256sum {} + > "$REF_EVIDENCE/bundle_files.sha256"
```

These are on-disk identities. A complete matched experiment also needs the operator's
actual launch arguments/model path, engine version and compiled options, and startup evidence
that these were the bytes loaded. Do not label current on-disk hashes as in-memory proof.

After confirming the listed endpoints/QoS, record all available read-only channels:

```bash
timeout --signal=INT 35s ros2 bag record \
  --qos-profile-overrides-path "$ESCORT_REPO/research/escort_rl/configs/reference_record_qos.yaml" \
  --output "$REF_EVIDENCE/bag" \
  --topics /joint_states /cmd_vel /g1_sensor_relay/base_state \
  /g1_sensor_relay/actor_ground_truth /agile_controller/inferring \
  /locomotion_safety_controller/transition_event /diagnostics
ros2 bag info "$REF_EVIDENCE/bag" > "$REF_EVIDENCE/bag_info.txt"
cat "$REF_EVIDENCE/bag_info.txt"
```

`timeout` can return status 124 after the expected SIGINT; inspect the recorder's final flush
and bag metadata. Check discovery has subscribed before the operator starts an authorized
trial. A 35 s recorder window leaves room for a 27 s suite case and discovery, but does not
itself capture cold startup on an already running simulator. Capture startup only during a
separate operator-managed session; no launch/reset commands are issued by this package.

The QoS file mirrors source-verified defaults and the supplied reliable/transient-local
JointState endpoint. If installed publishers differ, record their actual QoS first and make
a versioned capture configuration; do not change publisher/controller settings. Inspect
message counts and losses after every run. A listed topic with count zero remains missing.

## Offline export (no ROS graph access)

With the installed generated `g1_msgs` overlay sourced:

```bash
PYTHONPATH="$ESCORT_REPO/research/escort_rl:$PYTHONPATH" \
  python3 "$ESCORT_REPO/research/escort_rl/tools/export_reference_bag.py" \
  --bag "$REF_EVIDENCE/bag" --output "$REF_EVIDENCE/export"
```

The tool opens only `SequentialReader`, uses generated message deserialization, selects an
allowlist, normalizes body joint names, drops actor poses, computes clearly labeled backward
pose derivatives and hashes the bag before/after. It writes separate asynchronous streams,
a signal-coverage report and evaluator-only provenance. It does not fabricate targets,
gains, commands, safety states or exact physics timestamps. Outputs are **not RL observations**.
If generated message types are unavailable, report the overlay problem; do not rebuild the
working stack or install dependencies automatically.

The supplied host bag was exported using the already-built host message module (installed
host overlay was stale), without changing either overlay:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
PYTHONPATH="$PWD/research/escort_rl:$PWD/workspace/build/g1_msgs/rosidl_generator_py:$PYTHONPATH" \
LD_LIBRARY_PATH="$PWD/workspace/build/g1_msgs:$LD_LIBRARY_PATH" \
/usr/bin/python3 research/escort_rl/tools/export_reference_bag.py \
  --bag data/escort_logs/runtime_reference_20261105_021529 \
  --output research/escort_rl/runs/reference_export_manual_002
```

## Matched experiment and acceptance protocol

1. **Identity:** verify exact checkpoint; robot joints/inertias/geoms/actuators/meshes, engine,
   timestep and solver; startup weld/hold behavior and all controller parameters. The supplied
   live run uses the navigation actor bundle; previous isolated command tests use the flat
   scene. Those scene identities are not interchangeable. Match the complete bundle, or
   arrange a separately approved flat reference session. The current runner's scene selector
   does not load an arbitrary actor bundle; do not claim exact matching just from joint names.
2. **Initial conditions:** record robot pose/qvel, all joint q/dq, holds, gait/reset history,
   commands and control phases. Match cold startup/release in a fresh operator-controlled
   trial, or explicitly document a settled-start condition and measure its repeatability.
   No automatic live reset is permitted. Do not align away a formation/base error by fitting
   a final trajectory transform.
3. **Command protocol:** reuse `configs/command_suite.json`, including stand/8 s hold/2 s
   half-command/5 s zero/small correction/final zero. Representative cases: forward 0.20
   and 0.60; lateral 0.35; yaw 0.60; combined vx=0.35/wz=0.30; 0.10 dead-zone and zero controls.
   Use the **existing approved, exclusive, collision-monitored command owner**. No publisher
   or command-replay utility is supplied here. If that owner cannot execute/verify this
   sequence, the matched-command requirement remains open. Keep raw/monitored/received and
   actually executed commands distinct and log any override.
4. **Repeated reference:** collect at least five runs per representative condition before
   acceptance; report each run, variability and failures. Do not select the closest run or
   tune on a final-test scenario. Current stationary data are useful only as a partial sanity
   check, not as a substitute for movement trials.
5. **Timing:** use a shared verified event/native-time trace for joint/targets/gains/base.
   Retain controller sampling time, inference event, DDS receipt and physics-boundary timing.
   Bag arrival and `stampFor()` are insufficient for exact joint-phase alignment. Record
   alignment uncertainty; do not use velocity-response-dependent shifts or dynamic time warping
   to erase real latency. Align commands/events prospectively and resample explicitly only
   within logged validity intervals; do not extrapolate across losses or resets.
6. **Coordinates:** use pelvis-origin linear velocity in pelvis axes, body gyro components,
   and separately labeled world ZYX yaw derivative. Compare native world poses with a verified
   fixed world relation. Convert ROS xyzw to wxyz explicitly. COM and origin measurements
   require the rigid-body point transform; legacy records cannot silently enter v2 comparison.
7. **Comparison:** retain declared tensor atol/rtol=1e-5; identical-input controller fixtures
   atol=1e-5; provisional 10 s dynamics joint RMS <=0.02 rad, planar-velocity RMS <=0.05 m/s,
   endpoint/yaw difference <=0.10 m/rad and no additional falls/emergency failures. Report
   target/gain, angular response, base position/orientation and stopping distributions too.
   Establish reference variability first; any changed tolerance needs a documented decision.

The existing trajectory comparator now rejects missing/nonfinite required arrays and missing
or mixed velocity conventions. It reports numerical discrepancies but explicitly does not
establish full dynamics parity by itself. After a complete external recorder, verified
manifest and explicit alignment are available:

```bash
PYTHONPATH="$ESCORT_REPO/research/escort_rl:$PYTHONPATH" \
  python3 "$ESCORT_REPO/research/escort_rl/tools/parity.py" \
  trajectory isolated_aligned_v2.jsonl reference_aligned_v2.jsonl
```

A normalized asynchronous export is not automatically an aligned trajectory file.

## Smallest optional instrumentation when topics are insufficient — proposal only

No existing ROS interface exposes the 29 motor targets/Kp/Kd/feedforward after safety, the
exact gait-consumed command/inference phase, safety latch/blend or weld-release event.
`/right_hand_controller/controller_state` is an arm/hand trajectory-controller state; it is
not a locomotion-target/gain trace.

The smallest non-controller-invasive addition would be a separate **subscribe-only Unitree
DDS observer** for existing `rt/lowcmd`/`rt/lowstate`: timestamp receipt, decode SDK joint order,
and export motor targets/gains plus measured state. It must not instantiate a LowCmd
publisher, service client, robot object with motion methods, or command/reset channel. This
would expose final motor commands but still not exact policy inference times or physics-time
safety/weld state. DDS topics are not ROS topic aliases. Such an observer is **not implemented
or launched** in this milestone; the offline ROS bag exporter is the implemented recorder path.

For full numerical dynamics parity, a further optional read-only simulator event trace would
need physics time, compiled identity, qpos/qvel, actually used controls, equality state and
sampling/control event markers. A bounded snapshot queue drained by a separate writer would
avoid blocking the control path; record drops and benchmark timing perturbation. Exact
controller-internal latch/blend/command-consumption state remains unavailable without a
separately approved diagnostic hook. No hook, simulator modification or rebuild was made.
Until these requirements are available or transparently waived for a weaker statistical
comparison, full-dynamics/controller-timing parity remains a blocker.
