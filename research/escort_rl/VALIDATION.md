# Frozen gait implementation-stage evidence — 2026-11-04

Historical report: runtime dependencies are now available. The subsequent
[RUNTIME_VALIDATION_2026-11-04.md](RUNTIME_VALIDATION_2026-11-04.md) supersedes runtime status
and measurements below. This implementation-stage record is retained for provenance.

Decision: retain AGILE provisionally. **Do not proceed to PPO environment implementation on
this technical evidence yet.** Runtime dependencies, inference/controller parity, stable full
physics, live reference repeatability and measured throughput remain open gates.

## Work and identity

Created only `research/escort_rl/`: source modules, pinned package metadata, reference manifest,
fixed command suite, tools, focused tests and these documents (file inventory in README).
The pre-existing modified `AGENTS.md`, untracked document and baseline package were preserved.
No original controller/weight/asset edits, installs, Docker lifecycle actions, simulator
interaction, ROS publishing, training, commits or pushes occurred.

Read the supplied previous audit and repository AGENTS.md, vendored GROVE.md, gait/controller/
safety/freeze sources, launch scene staging, startup/bridge/Dex3 sources, scenario adapter and
observation documentation. No additional nested AGENTS.md was found by the repository search.

Checkout checkpoint SHA-256 was reverified:
`8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b`.
This establishes checkout identity only; the bytes/overrides loaded by a running container
remain unverified. The manifest pins 29 observation joints, 14 action joints, 12 inputs,
10 outputs, 269 feedback floats and reference code/assets. Normalization/action details
from the supplied graph audit are identified as such, not newly executed graph evidence.

## Tests actually run

From repository root:

```bash
PYTHONPATH=research/escort_rl python3 -m pytest research/escort_rl/tests -q
python3 -m compileall -q research/escort_rl/escort_rl research/escort_rl/tools
git diff --check
```

Latest result: **10 passed, 3 skipped**. Portable tests establish frozen-file identity/signature
metadata, finite input validation, caps/timeout, reset clearing, blend/inference phase,
emergency latching, PD/holds, deterministic scheduler counts/jitter, seven-feedback Python
wiring, all-ten-output comparison coverage, suite coverage stopping-dwell calculations and trajectory/controller discrepancy detection.
The feedback test uses a fake session solely to exercise buffer wiring. It is not ONNX or
physics evidence. Python syntax compilation and whitespace checks passed.

Skipped: two dependency checks and the complete runtime test module. That module contains
actual inference reset/history isolation and full-MuJoCo reset/interleaving/startup checks;
none ran. No runtime stability, contact, reset or worker-independence claim follows from the
portable tests. The C++ probe was not compiled; C++ ORT development files were not found in
the inspected locations.

A 1,000-row deterministic inference fixture was generated successfully to exercise the
fixture CLI; the generated temporary artifact was removed. It can be reproduced with README
commands and was not sent through either policy.

## Actual dynamics / speed / throughput results

The offline benchmark was attempted with:

```bash
PYTHONPATH=research/escort_rl python3 research/escort_rl/tools/run.py throughput \
  --output research/escort_rl/runs/dependency_probe_20261104 --transitions 5
```

It failed at `import mujoco` with `ModuleNotFoundError` before any physics or gait inference.
The tool saved `failure.json` in that fresh ignored run directory. Both inspected Python
interpreters lack MuJoCo and ONNX Runtime. README contains exact proposed isolated install
commands; none were executed.

| Required evidence | Measured result |
|---|---|
| Python vs C++ all ten tensors and feedback | Not run |
| Python vs C++ controller targets/gains | Not run; Python equation tests only |
| MuJoCo compiled timestep/solver/actuators | Not loaded; 2 ms expected from source/default audit |
| Stable startup, release, contacts and motion | Not run |
| Dynamics trajectory parity vs reference | No recorded reference available |
| Achieved vx/vy/yaw and command tracking | Not measured |
| Dead zone, latency, deceleration, stopping | Not measured |
| Commanded-versus-achieved/time-series plots | Tool provided; no measured plots generated |
| Physics/inference/5 Hz transitions per second | Not measured |
| Reset latency, CPU/RAM and latency distribution | Not measured |
| Determinism and parallel independence | Runtime tests prepared; not executed |
| Collection time for 100k/500k/1M transitions | Cannot estimate without measured rate |

Caps remain **±0.6 m/s x/y and ±0.9 rad/s yaw**. A software cap is not measured capability.
No assertion about maximum attainable velocity, low-speed dead zone, stop reliability or
ability to follow a 0.45–0.50 m/s client is justified yet.

## Parity limitations requiring attention

- Deterministic 5 ms controller / 1 ms bridge events round up to a 2 ms physics boundary.
  Controller delay is 0 or 1 ms; 1 kHz bridge callbacks can share a physics state. No claim
  that this reproduces the existing asynchronous communication latency or sensor age.
- Runner activates body holds/safety together and reproduces preactivation 200/5 PD, safety
  blend, 500-driven-bridge-tick weld release and uncommanded finger hold. Reference activation
  races and exact release phase must be measured. No longer startup interval is hidden in reset.
- `mj_forward` refreshes sensors after every integration step. This is a deliberate synchronous
  freshness/cost choice requiring parity validation, not an assumed wall-clock equivalent.
- After divergence the runner keeps last integrated target on fallback 100/2 gains and flags
  failure. The reference can later switch to a measured-pose freeze controller at 10/1 through
  a 100 ms ROS timer/service. That asynchronous emergency transition is not reproduced.
- Flat-scene tests isolate gait response; navigation-world/contact comparisons remain separate.
  Synthetic actor updates cover bookkeeping cost, not the complete scenario generator or social
  interaction. Proxies are noncolliding and cannot establish physical human-contact safety.
- The open-loop harness compares all output tensors and resets. The trajectory harness compares
  time-aligned joint state, target/gains, velocity, endpoint/yaw and failure flags. It requires
  external verified reference JSONL; a live recorder is not injected automatically. The
  proposed numerical tolerances are unchanged and have **not passed**.

## Gait choice recommendation

Grove AGILE retains the existing 29-observation/14-action controller and unchanged checkpoint,
which minimizes migration work. The official Unitree RL Gym G1 deployment is a potential
fallback, but its published configuration uses a TorchScript `motion.pt`, 12 actions/47
observations, different gains/scales and its own model; it is not a compatible replacement.
The official configuration has a 2 ms simulation interval and decimation 10 (50 Hz policy).
These are configuration facts, not measured speed or responsiveness on Grove.
Sources checked: [official G1 configuration](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_mujoco/configs/g1.yaml)
and [official deployment implementation](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/deploy/deploy_mujoco/deploy_mujoco.py).

Retain AGILE until the exact suite can measure stable achieved speed near 0.5 m/s, small
corrections and stopping. No observed limitation currently justifies switching policies,
because no command-response measurement ran. If stable achieved speed is below client speed,
latency prevents recovery, or low-speed control/stopping fails reproducibly, evaluate the
fallback on the same full-G1 scenario/limits/supervision protocol under a separately scoped
port. A larger advertised command range alone is not grounds to switch or raise caps.

## Remaining manual gates and next milestone

1. Authorize/install only isolated pinned runtime dependencies, then execute runtime tests,
   inference probe and short difficult turn/stop smoke trial before the complete suite.
2. Confirm actual loaded reference checkpoint, model/engine parameters and command chain in an
   operator-approved session; collect repeated physics-time reference observations/trajectories
   with an approved recorder and exclusive collision-monitored command owner.
3. Execute ten-output parity and controller/reset fixture comparison, then matched trajectories.
   Review scheduling, sensor freshness and emergency differences; measure reference variability
   before accepting/revising dynamics tolerances.
4. Execute full command suite, inspect failure-qualified curves/time series, and decide whether
   0.45–0.50 m/s client tracking has measured headroom and acceptable latency/stopping.
5. Measure one then two workers with contacts/actor updates/reset/logging; verify independence
   and determinism, and estimate collection budgets from actual full-physics step counts.

Only after these gates pass should a separate Gymnasium/PPO-environment milestone be approved.
This package stops at locomotion-runner preparation and feasible offline tests.
