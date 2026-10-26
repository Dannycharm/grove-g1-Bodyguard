# MuJoCo pedestrian adapter — manual operation only

## Design and evidence

The existing fork has no named-body runtime pose command interface. Its SDK bridge accepts
motor commands, not actor poses; changing that bridge would violate controller ownership.
`main.cc` owns process-local `mjModel* m` and `mjData* d`. `PhysicsLoop` holds `sim.mtx` around
stepping/paused forward updates. `sensor_publisher.cc` copies live geometry transforms while
holding that same mutex, then raycasts/renders its own snapshots outside the lock.

The smallest extension is therefore an opt-in `pedestrian_playback.{h,cc}` component with
one outer-loop hook for commands and one hook after each of the two `mj_step` call sites.
All pose writes occur on the physics thread under `sim.mtx`. No actor worker thread,
DDS channel, ROS publisher, motor interface, qpos/qvel/ctrl edit, or second simulator is added.
The existing SDK bridge, robot model topology, balance policy, controllers, navigation velocity
path, collision monitor, sensor relay, and DDS configuration are preserved.

When enabled, playback updates only named `escort_<actor_id>` mocap slots and calls
`mj_kinematics` to refresh transforms before the next snapshot. Time advances by differences
in `mjData.time`, independent of wall-clock pacing. Global simulation pause freezes scenario
progress; the scenario's own pause holds pedestrians while G1 simulation continues. A model
change or backward simulation-time jump disables the adapter; it never resets G1 or silently
starts a new episode. Existing viewer reload/reset behavior is unsafe for the running G1 stack;
do not use Backspace, Reload, or simulation Reset as scenario controls.

Sources inspected: `workspace/vendor/unitree_mujoco/GROVE.md`, its `simulate/src/main.cc`,
`param.h`, `sensor_publisher.cc`, and `simulate/CMakeLists.txt`; Grove-G1's
`g1_bringup/launch/{sim,bringup}.launch.py`, navigation MJCF/walking overlay,
`g1_navigation/maps/facility.yaml`, and sensor-relay config/source. No live graph was queried.

## Scene preparation (offline; does not launch)

Run **inside the existing development container**, using the repository's standard mounts.
Use a fresh output directory. Do not overwrite an active bundle. The explicit transform below
maps the scenario frame to MuJoCo world; it says nothing about the ROS map transform.

```bash
cd /root/.host-repo
export PYTHONPATH=/root/.host-repo/research/escort_scenarios
export PYTHONDONTWRITEBYTECODE=1
python3 -m escort_scenarios.mujoco_adapter prepare \
  research/escort_scenarios/configs/mujoco_navigation.yaml \
  --output /root/data/escort_navigation_seed42 \
  --navigation-scene /root/workspace/src/g1_bringup/mjcf/g1_navigation_scene.xml \
  --robot-model /opt/unitree_robotics/unitree_mujoco/unitree_robots/g1/g1_29dof.xml \
  --map-yaml /root/workspace/src/g1_navigation/maps/facility.yaml
```

The adapter config points to `navigation_actors.yaml`, which uses the existing generator's
actor IDs, roles, seed, waypoints, limits, and deterministic interpolation. Edit or copy those
configs for new experiments; unknown adapter keys and invalid coordinates are rejected.
This example places paths within the central intersection, not the offline generator's much
larger generic crossing paths (which intersect facility partitions).

The bundle contains:

- `robot.xml`: a copy of the original model, with only its mesh directory resolved absolutely.
- `navigation.xml`: a copy of the existing navigation scene pointing to that model copy.
- `scene.xml`: overlay with the original `pelvis_startup_hold` walking weld and separate actors.
- `playback.yaml`: resolved exact quintic segments in MuJoCo world coordinates.
- `adapter.yaml`, `metadata.json`, `evaluation_truth/`: configuration, provenance, hashes, and truth.

The original scene and map are never edited. Continue using the matching committed facility
map. Metadata records the explicit scenario-to-MuJoCo transform and leaves the ROS map transform
unknown. Future robot observations/predictions/planners must not read playback segments or
`evaluation_truth`: these include complete future simulator truth. Existing sensor topics remain unchanged;
new bundles also contain a separate `sensors.yaml` listing the actor bodies and pelvis for
its existing object-pose sampler, plus a path-free `observation_roster.yaml`. This does not change
the baseline sensor config or expose policy tracks automatically. See [actor observations](ACTOR_OBSERVATIONS.md).

Clearance checks cover the actor footprint against rectangular scene bounds, continuous line
segments against expanded conservative primitive envelopes in the navigation scene, pairwise
initial spawn clearance, and a configured reserved initial G1 footprint. Unsupported scene
geometry/nested bodies fail closed. Runtime bind/reset/replay/start-at-zero also check the
current G1 pelvis XY with the configured clearance radius. These checks do not prove that G1
will remain clear during playback, or that actors avoid each other after spawning. The bundled
scene is trusted offline input; do not hand-edit `playback.yaml` to bypass preparation checks.

## Geometry and sensor limitations

Actors are upright capsule/head primitives, not articulated human models. Stable body names
are `escort_client`, `escort_pedestrian_1`, `escort_pedestrian_2`; zero to two pedestrians are
supported. Their geoms are in group 2, the existing LiDAR ray group and one of the RGB-D
rendered groups. They use `contype=0`, `conaffinity=0`: **visual-only, no physical contact**.
They can be perceived as obstacles if sensors actually see them, but cannot physically stop G1.
The C++ binder rejects accidentally collidable actor geometry. Collidable pedestrians are not
implemented because kinematic contacts could disturb the walking policy.

Group compatibility is source evidence only. Actual RGB-D visibility/occlusion, LiDAR returns,
TF projection, RViz display, dynamic costmap behavior, timing cost, and collision-monitor
responses are **not runtime verified**. Do not claim sensor/collision integration from tests.

## Simulator rebuild (operator, not executed in this task)

The host lacks the MuJoCo SDK/library. The full simulator build cannot run here. The development
image has the dependencies and copies the fork into `/opt/unitree_robotics/unitree_mujoco`;
building only the host-mounted fork does not update that executable. For a scheduled maintenance
session with no running simulator, apply the isolated patch to that installed source and rebuild:

```bash
cd /root/.host-repo
patch --dry-run --forward -p1 -d /opt/unitree_robotics/unitree_mujoco \
  < research/escort_scenarios/patches/unitree_mujoco-pedestrians.patch
patch --forward -p1 -d /opt/unitree_robotics/unitree_mujoco \
  < research/escort_scenarios/patches/unitree_mujoco-pedestrians.patch
cmake -S /opt/unitree_robotics/unitree_mujoco/simulate \
  -B /opt/unitree_robotics/unitree_mujoco/simulate/build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/unitree_robotics/lib/cmake
cmake --build /opt/unitree_robotics/unitree_mujoco/simulate/build \
  --target unitree_mujoco --parallel 2
source /opt/ros/jazzy/setup.bash
cd /root/workspace
colcon build --packages-select g1_bringup --symlink-install
source install/setup.bash
```

Do not apply twice or overwrite unrelated installed-source changes; the dry run must succeed.
No Docker invocation is needed. A later image rebuild also picks up the modified fork via the
existing Dockerfile, but no image/container operations were performed here.

Rollback: leave both pedestrian environment variables unset for baseline launches. That restores
the original scene-staging path and makes playback a no-op. To remove the source extension during
operator-approved maintenance, reverse the exact patch with `patch --dry-run -R -p1` first, then
`patch -R -p1` against the same installed fork, and rebuild using the commands above. The checked-in
baseline world assets were never changed. Do not use git reset/clean or replace a running binary.
The fork is a submodule: preserve its new source files and patch when transferring this work;
no commits, submodule-pointer updates, or pushes were made by the agent.

## Exact opt-in launch (operator, not executed)

Only launch after the rebuilt executable is installed and the operator has arranged that no
other simulator/controller stack is running. Keep the existing sourced Jazzy DDS environment;
do not override its domain/RMW/NIC settings. This command starts the normal G1 walking/Nav2 stack
with the separate actor scene, **but leaves scenario playback ready and motionless**:

```bash
source /opt/ros/jazzy/setup.bash
source /root/workspace/install/setup.bash
GROVE_G1_PEDESTRIAN_BUNDLE=/root/data/escort_navigation_seed42 \
  ros2 launch g1_bringup bringup.launch.py \
  world:=navigation mode:=localization nav:=true sensors:=true \
  odometry:=fast_lio headless:=false rviz:=true
```

The opt-in launch requires navigation world, sensors enabled, and unpinned walking configuration.
It selects `scene.xml` by absolute path and passes `playback.yaml` to the child simulator.
Without `GROVE_G1_PEDESTRIAN_BUNDLE`, the baseline staging path remains unchanged.
Never set `GROVE_G1_PEDESTRIAN_SCENARIO` separately through the ROS launch.

## Controlled scenario lifecycle (explicit actor-motion commands)

These commands are for the operator **after launch and inspection**, not automatic verification.
One bundle has one atomic command mailbox; designate one operator/writer and wait for each
acknowledgement before submitting the next command. Sequence numbers must strictly increase.
No command resets `mjData.time`, the G1 state, controllers, odometry, or navigation.

```bash
cd /root/.host-repo
export PYTHONPATH=/root/.host-repo/research/escort_scenarios
python3 -m escort_scenarios.mujoco_adapter command \
  --bundle /root/data/escort_navigation_seed42 --sequence 1 --action start
cat /root/data/escort_navigation_seed42/status.yaml
# Later, explicitly freeze actor progress:
python3 -m escort_scenarios.mujoco_adapter command \
  --bundle /root/data/escort_navigation_seed42 --sequence 2 --action pause
cat /root/data/escort_navigation_seed42/status.yaml
# Only after pause acknowledgement and visual clearance inspection:
python3 -m escort_scenarios.mujoco_adapter command \
  --bundle /root/data/escort_navigation_seed42 --sequence 3 --action reset
# Or --action replay to reset actor time and start in one explicit operation.
```

`start` resumes ready/paused playback; `pause` holds current actor poses; `reset` returns actors
to their initial positions and state `ready`; `replay` resets actors and starts. Reset/replay
are rejected while playing or when initial actor positions overlap the current G1 footprint.
At completion the scenario stays `finished`; it never loops automatically. A reset/replay
teleports actors and invalidates continuity of any previous observation episode: future loggers
must start a distinct episode. Do not reset actors during an active navigation experiment.

`status.yaml` acknowledges the consumed sequence, state, scenario time, and error at transitions
(not a continuous time telemetry stream). Invalid commands are reported without resetting G1.
On a new simulator process, an existing mailbox is ignored so old commands cannot start motion.
A sequence acknowledgement with an error is a rejection, not success. Filesystem polling/parsing
is on the physics thread under its mutex; commands are small and infrequent, but latency and
real-time overhead require manual measurement. Never use a slow/network filesystem for bundles.

## Offline checks and manual sensor/RViz verification

```bash
cd /root/.host-repo
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/escort_scenarios \
  python3 -m unittest discover -s research/escort_scenarios/tests -v
```

The C++ test builds actual playback code with `g++ -Wall -Wextra -Werror -lyaml-cpp` and a
**test-only MuJoCo API double**. It validates state/discovery/pose logic, not real MuJoCo
ABI, scene compilation, contacts, rendering, or LiDAR. The remaining generator/bundle tests
run offline; no simulation is launched and no mailbox is written to an active bundle.

Inside the operator-launched existing container session, the following bounded read-only commands
inspect interfaces. Run them before and after an explicitly approved scenario start:

```bash
timeout --kill-after=2s 5s ros2 topic info /camera/color/image_raw --verbose
timeout --kill-after=2s 5s ros2 topic echo /camera/color/image_raw \
  sensor_msgs/msg/Image --once --field header --qos-reliability best_effort
timeout --kill-after=2s 5s ros2 topic echo /camera/aligned_depth_to_color/image_raw \
  sensor_msgs/msg/Image --once --field header --qos-reliability best_effort
timeout --kill-after=2s 8s ros2 topic hz /camera/aligned_depth_to_color/image_raw
timeout --kill-after=2s 5s ros2 topic info /livox/lidar --verbose
timeout --kill-after=2s 5s ros2 topic echo /livox/lidar \
  sensor_msgs/msg/PointCloud2 --once --field header --qos-reliability best_effort
timeout --kill-after=2s 8s ros2 topic hz /livox/lidar
timeout --kill-after=2s 5s ros2 run tf2_ros tf2_echo map camera_depth_optical_frame
timeout --kill-after=2s 5s ros2 run tf2_ros tf2_echo map mid360_link
timeout --kill-after=2s 5s ros2 lifecycle get /collision_monitor
timeout --kill-after=2s 5s ros2 topic info /cmd_vel --verbose
```

The launch opens RViz; use that instance rather than opening another by default. If none is
open and the operator chooses to open only a viewer:

```bash
ros2 run rviz2 rviz2 -d /root/workspace/src/g1_navigation/config/g1_navigation.rviz
```

In RViz set Fixed Frame to `map`; add an Image display for `/camera/color/image_raw`,
a DepthCloud for `/camera/aligned_depth_to_color/image_raw` and its CameraInfo topic,
and PointCloud2 for `/livox/lidar` with Best Effort reliability. Confirm actual actor silhouettes,
depth surfaces and spatially corresponding LiDAR returns during playback, not merely topic names
or nonzero rates. Verify sensor-frame alignment through the real TF chain; no actor TF/track
publishers are added. Inspect Nav2 dynamic obstacles and collision-monitor behavior separately
without adding direct velocity writers. Physical contact is intentionally absent and cannot be
validated as working. No navigation goals or motor commands are supplied by these instructions.
