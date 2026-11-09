# Isolated frozen G1 runner and Stage 1 escort environment

**Current Stage 2 result (2026-11-09):** frozen-base residual imitation followed by 5k guarded residual PPO passed two full fixed-suite paired rounds: 60/60 pedestrian and 60/60 Stage 1, zero geometric intrusions or critical failures. This is repeatability for one selected model seed, not independent new-route or multi-seed generalization. See [measured results](RESIDUAL_PPO_5K_RESULTS_2026-11-09.md), [pilot protocol](RESIDUAL_PPO_PILOT_V1.md), [archive scope](STAGE2_GIT_ARCHIVE.md), and [the explanatory guide](../../docs/bodyguard/STAGE2_RL_TRAINING_EXPLAINED.md). No additional training or Stage 3 is automatically authorized.

**Latest diagnostic results:** the unchanged 128×128 actor achieved 60/60 held-out
successes through diagnostic imitation; lower initial PPO noise alone achieved 0/60
at 20k. See [CAPACITY_EXPLORATION_2026-11-06.md](CAPACITY_EXPLORATION_2026-11-06.md)
for evidence, 102 passing tests, and the prepared opt-in `tracking_stop_v3` experiment.
No full v3 PPO experiment has been launched. V3 changes reward and completion criteria;
evaluate old policies under the same criteria before comparing success counts.

**Current: restart RNG correction and reward audit complete.**
[PPO_RESUME_REWARD_AUDIT_2026-11-06.md](PPO_RESUME_REWARD_AUDIT_2026-11-06.md)
documents the tested SB3 2.7.0 restart protocol, opt-in `formation_progress_v2`,
92 passing tests, checkpoint/stop diagnostics and the two prepared fresh 20k comparisons.
No follow-up experiment has been launched. Legacy checkpoints remain read-only and fail
the new strict restart-protocol check; use the fresh comparison commands in that report.

**Stage 1 Gymnasium environment is implemented and validated.** See
[STAGE1_ENVIRONMENT.md](STAGE1_ENVIRONMENT.md) for the scenario/observation contracts,
reward and outcome rules, changed files, 25-episode results, throughput and exact commands.
The environment reuses the standalone scenario generator, owns private full-G1 physics,
and implements straight, gentle-turn and stop/start client-only scenarios.
Stage 1 PPO infrastructure and a limited 10,000-transition pilot are now complete; see
[PPO_STAGE1.md](PPO_STAGE1.md) for usage and [PPO_PILOT_2026-11-05.md](PPO_PILOT_2026-11-05.md)
for measured results. The pilot did not pass the escort validation gate. No longer training
run or curriculum promotion was performed. Full live-reference dynamics parity remains
unverified. The existing `.venv` now also contains user-approved `gymnasium==1.2.2`;
The same venv also contains user-approved CPU PyTorch 2.7.1, SB3 2.7.0 and TensorBoard 2.20.0.

Status: pelvis-origin/body-axis velocity is corrected and Python/original-C++ inference
parity passes on the tested arrays. The moving Grove recording now supports the same roughly
0.40–0.41 m/s response at a 0.60 m/s request; see
[MOVING_REFERENCE_ANALYSIS_2026-11-05.md](MOVING_REFERENCE_ANALYSIS_2026-11-05.md).
**Exact moving-reference dynamics parity remains open.** No additional equivalent operator
bag is requested for this milestone. Prior gait-validation reports and the
[reference collection guide](REFERENCE_COLLECTION.md) are retained for provenance.
Only this research directory was added. Original controllers, checkpoint, assets and live
simulator are untouched. Imports never initialize ROS, DDS, a viewer or a command publisher.

## Files

- `configs/gait_reference.json`: v1 reference signature, joint tables, gains, timing, caps,
  tolerances, actuator XML attributes/defaults and SHA-256 for checkpoint, meshes, models,
  scenes and relevant reference sources. It refuses changed files at runtime.
- `configs/command_suite.json`: fixed policy-independent command-response protocol.
- `escort_rl/frozen_gait.py`: CPU ORT, one intra/inter-op thread, exact 12/10 signature,
  float32 inputs and all seven feedback tensors; no second scaling/normalization.
- `escort_rl/controller_reference.py`: command caps/timeout, 200 Hz safety, decimated
  inference, captured holds, float32 LowCmd targets/gains and simulator PD equation.
- `escort_rl/mujoco_runner.py`: private model/data and temporary scene staging, actual
  motor torques/contacts, startup weld release and cold reset. No root writes in stepping.
- `escort_rl/measurements.py`, `parity.py`: explicit numerical measurement/comparison rules.
- `tools/run.py`: JSONL command trials/plots and one/multiple-process throughput collection.
- `tools/parity.py`, `cpp_policy_probe.cpp`: recorded-input Python/C++ ten-output comparison
  and time-aligned full-trajectory comparison. The C++ test-only access macro exposes existing
  buffers; it does not edit or replace `AgilePolicy`.
- `tests/`: portable contract/controller tests and dependency-gated inference/dynamics tests.
- `tools/export_reference_bag.py`, `escort_rl/reference_recording.py`: read-only offline
  rosbag export, explicit missing-signal coverage and evaluator-only robot-state records.
- `VELOCITY_AND_REFERENCE_PARITY.md`: latest correction, array parity, supplied bag analysis
  and decision. `VALIDATION.md` and the prior runtime report are historical evidence.

## Historical dependency setup — now present in the existing `.venv`

At the implementation milestone, both inspected host interpreters lacked `mujoco` and
`onnxruntime`. The existing `.venv` now contains every pinned runtime/test/plot dependency;
no installation was performed during runtime validation. The historical setup is retained below.
`onnx` was also absent
but not required by this implementation. No installations were made. C++ ORT headers/library
were not found in the checked workspace or standard include locations.
Use Python 3.12 rather than the default Conda Python 3.14 for the pinned ORT wheel.
These commands create a new local environment and require operator authorization to install:

```bash
cd /home/praise/ECE598_Bodyguard/grove-g1
/usr/bin/python3 -m venv research/escort_rl/.venv
research/escort_rl/.venv/bin/python -m pip install numpy==2.2.6 mujoco==3.3.6 onnxruntime==1.20.1 pytest==8.3.5 matplotlib==3.10.1
```

If `venv`/ensurepip is unavailable, stop and report it; do not install system packages.
The package is intended to run from this repository with `PYTHONPATH`; it resolves frozen
checkout assets by repository path. It is not a relocatable wheel containing robot assets.

## Offline validation commands after dependencies are authorized

Run from repository root. Use a new output path every time; tools refuse existing paths.
These commands launch only isolated offline physics, not training or a live simulator.

```bash
export PYTHONPATH="$PWD/research/escort_rl"
export PY="$PWD/research/escort_rl/.venv/bin/python"
"$PY" -m pytest research/escort_rl/tests -q
"$PY" research/escort_rl/tools/run.py commands --output research/escort_rl/runs/commands_001
"$PY" research/escort_rl/tools/run.py throughput --workers 1 --transitions 500 --output research/escort_rl/runs/throughput_1_001
"$PY" research/escort_rl/tools/run.py throughput --workers 2 --transitions 500 --output research/escort_rl/runs/throughput_2_001
```

Each command trial resets cold, stabilizes 2 s, stands 2 s, holds 8 s, halves the command
for 2 s, stops 5 s, makes a small correction for 3 s, then stops 5 s. Commands refresh at
5 Hz, metrics sample at 50 Hz. All raw requested/applied commands, velocities, targets,
gains, torques, attitude/height and contact count are logged. No command acceleration filter
exists in the reference; applied commands are not silently rate limited. Trials continue full
physics after a fall/emergency, retain failure flags and mark capability trials invalid.
The runner holds the safety fallback after a divergence; it does not emulate ROS switching.

Tracking is the mean of the final 2 s of full-command hold. Latency is first simultaneous
50% response on commanded axes (no claim of settling time). Stopping requires planar speed
and body angular z below 0.05 for 0.5 s; distance is planar path length from the full-zero
command to the start of that dwell. Null is a censored/unobserved response, not zero.
Roll/pitch affects body angular z versus Euler yaw rate; failed/tilted trials cannot establish
command capability. Inspect curves with failure flags and time series, not as aggregate proof.

Throughput includes full physics, synthetic noncolliding actor updates, measurements,
flushed JSONL and periodic cold resets. No dynamics steps are skipped after failure. Load/warmup
is outside steady timing; end-to-end wall time includes it. Reports count actual physics and
inferences, CPU process time/percentage, peak RSS, per-physics/per-transition p50/p95 latency,
reset samples and collection estimates N / measured aggregate transition rate. Optimizer time
is unknown and excluded. Repeat several trials and worker counts before selecting a budget.

## Python/C++ inference parity

The existing C++ source must be compiled against ORT 1.20.1 in an approved development
environment. `ORT_ROOT` below must point to an already installed matching distribution;
no download/install is performed by these commands. These commands do not access ROS/DDS.

```bash
mkdir -p research/escort_rl/runs/parity_001
"$PY" research/escort_rl/tools/parity.py fixtures unused research/escort_rl/runs/parity_001/inputs.txt
c++ -std=c++20 -O2 -I"$ORT_ROOT/include" -Iworkspace/src/g1_controllers/include \
  research/escort_rl/tools/cpp_policy_probe.cpp workspace/src/g1_controllers/src/agile_policy.cpp \
  -L"$ORT_ROOT/lib" -Wl,-rpath,"$ORT_ROOT/lib" -lonnxruntime \
  -o research/escort_rl/runs/parity_001/cpp_probe
research/escort_rl/runs/parity_001/cpp_probe \
  workspace/src/g1_controllers/policy/unitree_g1_velocity_e2e.onnx \
  research/escort_rl/runs/parity_001/inputs.txt research/escort_rl/runs/parity_001/cpp.txt
"$PY" research/escort_rl/tools/parity.py python research/escort_rl/runs/parity_001/inputs.txt research/escort_rl/runs/parity_001/python.txt
"$PY" research/escort_rl/tools/parity.py tensor research/escort_rl/runs/parity_001/python.txt research/escort_rl/runs/parity_001/cpp.txt
```

The fixture has 1,000 observations and resets at ticks 0, 1, 257 and 800. It is synthetic
open-loop evidence only. Use recorded causal observations in the same row schema as well:
reset flag, quaternion(4), gyro(3), command(3), positions(29), velocities(29).
All ten outputs are compared at atol=rtol=1e-5, including all history values.
The existing C++ probe has now been compiled and executed with ORT 1.20.1. All ten arrays
matched exactly for the tested synthetic and actual-runner inputs; see the latest report.
This does not establish controller-chain or full-dynamics parity.


## Velocity schema v2

`body_velocity` now means [pelvis-origin vx, vy, body angular z] in pelvis axes, matching the
existing Grove IMU-site base-state sampler. Full linear/angular world/body vectors, separate
Euler yaw rate and preserved `legacy_com_velocity` are logged with explicit conventions.
Legacy unlabeled/COM logs must not be silently compared to v2 trajectories. Measurement calls
are read-only and do not change gait inputs or rollout/reset state. For decelerated trials,
response metrics distinguish the full-command hold end from the subsequent zero-command time.

## Scheduling and manual reference integration

The reference controller manager runs at 200 Hz, gait at 50 Hz, and SDK/Dex3 callbacks at
1 kHz, independently of physics. The runner keeps the engine timestep at its compiled
2 ms value and rejects a different engine version/timestep. Events are rounded up to the
next physics boundary: controller lateness alternates 1 ms/0 ms; multiple bridge events at
the same boundary see the same state. Controller precedes bridge on equal due times. First
controller is at 5 ms, first inference at 20 ms. All body controllers activate together in
the runner; startup PD is 200/5 before that. The startup weld is released after 500 bridge
events with all motors driven, at the following physics boundary (~0.504 s in this schedule).
Sensors are refreshed by `mj_forward` after each physics step, an explicit extra cost and
sensor-freshness choice. These choices are deterministic approximations, not measured parity
with wall-clock scheduling or DDS latency.

The default is the existing flat scene wrapped by the existing walk overlay. `--scene navigation`
selects the original navigation world. The original mesh geometry, actuator saturation, gearing,
damping, armature, friction and contacts remain loaded. No navigation assets are edited.
Three optional benchmark proxies mean one client plus two pedestrians; they are synthetic
mocap objects with collision disabled. Contact count is a physical engine count, not a
pedestrian-overlap metric. This milestone has no perception or social-response evidence.

Manual integration remains an operator-controlled experiment. Before any new live run, verify
loaded checkpoint/parameters, exclusive command ownership, collision-monitor chain and actual
compiled scene/solver. Measure repeatability of the existing reference first. No live launch,
reset, controller switch, publisher or recorder was executed. Read-only inventory commands
for an operator-approved session are:

```bash
ros2 node list
ros2 topic list -t
ros2 param dump /agile_controller
ros2 param dump /locomotion_safety_controller
```

Use the operator's existing authorized command owner to execute the committed command suite;
do not publish a second `/cmd_vel`. An approved reference recorder still needs to export
physics-time joint/base state and controller targets/gains in the JSONL schema emitted by
`measure()`, plus checkpoint/model identity and event timing. A bag alone cannot guarantee
all this state. No recorder is injected into the live stack by this package.

After explicit time/frame alignment (no automatic resampling):

```bash
"$PY" research/escort_rl/tools/parity.py trajectory isolated.jsonl reference.jsonl
```

Review controller targets/gains and reset startup traces separately as well. The comparator
reports provisional joint/velocity RMS, endpoint/yaw and failure counts; its pass flag is
not scientific acceptance without a repeated, provenance-verified 10 s reference. Original
audited tolerances are preserved. Controller arithmetic tests are not C++ controller parity.
