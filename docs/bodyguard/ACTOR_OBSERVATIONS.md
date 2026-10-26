# Week 1 actor observations and reproducible trajectory logs

## Scope and provenance

`escort_observation_bridge` converts **current simulator poses** to causal actor observations.
This is explicitly `simulation_pose_proxy`, **not perception**, a Kalman filter, prediction,
or escort control. It never reads future playback segments, publishes velocity commands,
changes TF/time settings, or starts/stops/resets the simulator/controllers.

Stable actor IDs/roles come from `observation_roster.yaml`. This file contains only IDs, roles,
body names, seed, and scenario ID; it has no waypoints or future trajectories. New actor bundles
include `sensors.yaml`, copied from the original sensor config with actor bodies and `pelvis`
added to the existing `object_bodies` list. Existing bundles must be prepared into a **new**
directory; do not rewrite or reconfigure a running scene. See [adapter operation](MUJOCO_PEDESTRIAN_ADAPTER.md).

The existing simulator object-pose packet is reused unchanged. The relay pairs it with the
LiDAR packet by **identical native simulation capture time**. The existing camera-frame
`object_poses` topics and other sensor outputs are unchanged. A separate relay snapshot retains
native MuJoCo coordinates and both same-capture alignment anchors; it does not use the existing
camera conversion's arrival stamp/latest TF. No simulator fork rebuild is needed for this component.

## Interfaces

| Topic | ROS type | Source / semantics |
| --- | --- | --- |
| `/g1_sensor_relay/actor_ground_truth` | `g1_msgs/msg/SimulatorActorSnapshot` | Evaluator-only current truth, source simulation time/epoch, capture-mapped ROS stamp, LiDAR and pelvis poses; raw `mujoco_world` coordinates |
| `/escort/tracks` | `g1_msgs/msg/ActorStateArray` | Causal proxy observations; all roster actors included, even invalid/missing ones; existing `odom` by default, optionally existing `map` |
| `/escort/client_pose` | `geometry_msgs/msg/PoseStamped` | Convenience output only for valid client positions; same capture stamp/frame as tracks; no invalidation field, so consumers must also check `/escort/tracks` and freshness |
| `/escort/replay/tracks`, `/escort/replay/client_pose` | Same respective observation types | Recorded observation replay; intentionally separate from live publishers |

The shared `g1_msgs` interface package holds the new schemas; there is no second competing
interface package. `ActorState` includes stable ID, role, pose, heading, linear velocity,
position validity, velocity validity, source, and invalid reason. Pose/linear velocity are in
array `header.frame_id`; heading is yaw about that frame's +z axis. Only linear velocity is
estimated; Twist angular fields remain zero and are not an angular-rate estimate.
First samples, gaps, and resets have `velocity_valid=false`, even when position is valid.
The array includes a bridge epoch and `source_time_s` for derivative/episode interpretation.
Invalid poses/velocities are zero placeholders, not measurements; always inspect validity.

## Coordinate verification and time

There is no assumption that MuJoCo world equals ROS `map` or `odom`, and no new TF root is
broadcast. The descriptive raw `mujoco_world` label is not inserted into the TF tree.
For each paired snapshot, the bridge looks up **capture-time** `target → mid360_link` and
`target → pelvis` using the existing TF tree. Target is restricted to existing `odom` or `map`.

```text
target_T_world = target_T_sensor × inverse(world_T_sensor)
predicted_target_T_pelvis = target_T_world × world_T_pelvis
```

The independent captured pelvis pose must agree with TF within 0.15 m and 0.2 rad by default.
Otherwise all observations from that snapshot are invalid. Missing TF waits at most 0.3 s
in a bounded queue, then fails closed; there is no fallback to latest TF or identity.
Full quaternion/3D composition is tested, not only planar offsets. Alignment residuals and the
resolved transform are logged only under `evaluation/`. This checks frame/extrinsic consistency;
it does **not** independently establish AMCL/FAST-LIO accuracy, calibration, or a static global
map alignment. Live residuals/coordinate correctness remain **unverified until operator checks**.

The repository's Nav2/localization configuration uses `use_sim_time: false` and explicitly
notes that this track has no `/clock`. The relay's existing `stampFor()` maps native capture time
to its ROS wall clock. New snapshots use that same LiDAR capture stamp. The bridge preserves
it, and refuses `use_sim_time:=true` for this source; it changes no global ROS time settings.
Linear velocities are backward differences in transformed coordinates divided by **native
simulation seconds**, not message-arrival intervals or a drifting host timer. Dataset metadata
records these semantics. Real sensor integration will need its own source/clock contract.

## Missing actors and resets

- Missing/invalid actor poses produce invalid roster entries immediately and clear their velocity history.
- More than 0.7 s without processed input emits one invalid timeout array and clears history.
  Timeout stamps are explicitly `ros_wall_timeout`; measurement stamps remain capture stamps.
- Source simulation-time rollback or source-epoch/ROS-stamp regression starts a new bridge epoch.
- Physically implausible position jumps split all derivative histories and advance the epoch.
  Default jump test is `distance > 3 m/s × dt + 0.1 m` for contiguous samples.
- Duplicate capture times are invalid; gaps over 0.5 s do not yield velocity estimates.
- No reset command is sent anywhere. A scenario reset that produces no observable jump is not
  inferable from poses alone: use a new dataset/episode for operator resets/replays rather than
  treating a continuous log as a fresh experiment. Shutdown/restart also uses a new dataset path.

Map corrections or TF changes can resemble jumps, so `odom` is the preferred derivative frame.
The jump check is a consistency guard, not a tracker/filter or a safety controller.

## Build (manual container maintenance; no nodes launched by build)

From the existing Jazzy development container, with its normal environment:

```bash
source /opt/ros/jazzy/setup.bash
cd /root/workspace
colcon build --packages-select g1_msgs g1_sensor_relay escort_observation_bridge g1_bringup \
  --symlink-install
source install/setup.bash
```

The local isolated build successfully generated `g1_msgs` and built the Python package.
The relay build was blocked locally by missing `vision_msgs` development files and inaccessible
stale workspace install paths. Rebuild/verify it in the normal development container before
using the new snapshot. No Docker commands, package installs, or live relay restarts were run.
A running relay cannot acquire a newly compiled publisher automatically; arrange a separate
operator-approved session. Do not restart the current stack just to test this component.

Prepare a fresh bundle as in the adapter guide, changing `--output` to, for example,
`/root/data/escort_navigation_observations_seed42`. The new files are:
`observation_roster.yaml` and `sensors.yaml`. When the operator next launches that bundle,
the existing simulation launch selects its sensor config; no baseline config is edited.

## Manual live verification (not performed by the agent)

Inside an operator-managed session using the freshly prepared bundle and rebuilt relay:

```bash
source /opt/ros/jazzy/setup.bash
source /root/workspace/install/setup.bash
ros2 interface show g1_msgs/msg/SimulatorActorSnapshot
ros2 interface show g1_msgs/msg/ActorStateArray
timeout --kill-after=2s 5s ros2 topic info /g1_sensor_relay/actor_ground_truth --verbose
timeout --kill-after=2s 5s ros2 topic echo /g1_sensor_relay/actor_ground_truth \
  g1_msgs/msg/SimulatorActorSnapshot --once --qos-reliability best_effort
```

Check that `pelvis_valid` is true, body names match the roster, source times are sensible,
and the raw frame is `mujoco_world`. Then **explicitly** run only the observation/logger node;
it does not move actors or G1:

```bash
ros2 run escort_observation_bridge actor_bridge --ros-args \
  -p allow_simulation_proxy:=true \
  -p roster_file:=/root/data/escort_navigation_observations_seed42/observation_roster.yaml \
  -p evaluation_config_file:=/root/data/escort_navigation_observations_seed42/evaluation_truth/scenario.yaml \
  -p dataset_dir:=/root/data/escort_logs/seed42_run001 \
  -p target_frame:=odom
```

Use a new/empty dataset path. Default proxy mode is **off** until explicitly acknowledged.
Do not add this node automatically to the baseline launch. In a second shell:

```bash
timeout --kill-after=2s 5s ros2 topic echo /escort/tracks g1_msgs/msg/ActorStateArray --once
timeout --kill-after=2s 5s ros2 topic echo /escort/client_pose geometry_msgs/msg/PoseStamped --once
timeout --kill-after=2s 5s ros2 run tf2_ros tf2_echo odom mid360_link
timeout --kill-after=2s 5s ros2 run tf2_ros tf2_echo odom pelvis
```

For precise alignment use the logged **capture-time** residuals, not the latest `tf2_echo`
outputs. Inspect `evaluation/current_ground_truth.jsonl`: `alignment.verified` must be true,
residuals below configured limits, and projected known actor locations consistent with the
existing TF/RViz scene. Check client PoseStamped and track-array stamps/frames match exactly.
Only after the operator independently authorizes actor playback should moving-frame velocities,
IDs, disappearance, timeout, and epoch behavior be inspected. Do not reset G1 for this check.
No manual navigation/motor/velocity commands are needed or supplied here.

## Dataset layout, reproducibility, and replay

```text
manifest.json                         scenario ID, seed, configuration, timestamp semantics, code hash
observations/tracks.jsonl             causal records only; policy/training input stream
evaluation/current_ground_truth.jsonl current evaluator truth plus alignment evidence
evaluation/configuration.json        optional archived full scenario configuration (may include future paths)
```

Files are append-only canonical JSONL, flushed per record; existing datasets are never overwritten.
Code version fingerprints the actual Python sources, including uncommitted changes. The scenario
ID hashes resolved scenario config, and the seed/config/bridge limits are recorded. Identical
ordered input/timeout records and metadata produce byte-identical logs. Independent live runs
will have different ROS wall stamps/arrival schedules; do not claim identical live files across runs.
No host “recorded now” timestamp is invented to replace capture time. Logs are crash-flushed,
not transactional/fsync guarantees; partial final lines after a crash need explicit recovery.

Policy consumers receive only current observations/history, never raw snapshots, alignment
anchors/transforms, future scenario paths, or evaluator configuration. The strict observation
schema rejects extra fields, including nested future fields. This is a software/data boundary,
not an access-control sandbox: isolate evaluator files/topics from policy processes operationally.
Proxy data remains truth-derived even after conversion; it cannot validate perception quality.

Replay accepts **only** `observations/tracks.jsonl` and streams it in file order without reading
future evaluator labels. It preserves original capture stamps, frames, velocities, and epochs;
its wall timer controls delivery rate only. It does not re-stamp into a live experiment or use TF:

```bash
ros2 run escort_observation_bridge actor_bridge --ros-args \
  -p replay_file:=/root/data/escort_logs/seed42_run001/observations/tracks.jsonl \
  -p replay_period_s:=0.1
timeout --kill-after=2s 5s ros2 topic echo /escort/replay/tracks --once
```

Do not feed old-stamped replay into a live robot/controller. The complete generator CSV is
**evaluator truth**, not a policy replay input. Offline tests use no ROS graph:

```bash
cd /root/.host-repo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=workspace/src/escort_observation_bridge \
  python3 -m unittest discover -s workspace/src/escort_observation_bridge/test -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/escort_scenarios \
  python3 -m unittest discover -s research/escort_scenarios/tests -v
git diff --check
```

Generated-message conversion tests require the built `g1_msgs` overlay sourced; otherwise those
three tests report SKIP explicitly. They construct message objects only and never initialize
rclpy/DDS. Unit tests and successful interface compilation are not live ROS/TF verification.
