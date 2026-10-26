# Offline escort scenarios — Phase 1B, Task 1

Deterministic evaluation-ground-truth trajectories for Predictive Social Navigation.
Python 3.10+ and PyYAML are required; plotting additionally uses Matplotlib. These libraries
are already available in the development environment used for this task. No ROS 2, MuJoCo,
PyTorch, network service, wall-clock timing, or simulation launch is required.

## Reproduce from the repository root

No installation is necessary. Choose a **new or empty** output directory; existing exports
are never overwritten. The committed crossing example was generated with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/escort_scenarios \
  python3 -m escort_scenarios \
  research/escort_scenarios/configs/perpendicular_crossing.yaml \
  --seed 42 --output data/escort_scenarios/examples/crossing_seed42 --plot
```

For another run, change `--output`, for example to `/tmp/escort-crossing-seed42`.
`--seed` overrides the YAML seed and is recorded in the exported configuration.
Omit `--plot` to use only Python and PyYAML. Installing the isolated project with its
`pyproject.toml` is optional; `escort-scenarios` then exposes the same CLI.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research/escort_scenarios \
  python3 -m unittest discover -s research/escort_scenarios/tests -v
```

## Scenarios and configuration

Four YAML examples provide the families `straight`, `perpendicular_crossing`, `head_on`,
and `client_turn_slowdown_stop`. Each has exactly one designated client and zero to two
pedestrians, selected by `pedestrian_count`. Default stable IDs are `client`, `pedestrian_1`,
and `pedestrian_2`. The turn scenario includes a 90-degree turn, a slower leg, and a
three-second dwell. All actors start at scenario time zero; final poses hold indefinitely.
Crossing/head-on describe geometric encounter families, not a guarantee of simultaneous
collision or minimum separation. Optional jitter changes encounter alignment.

Required schema fields: `schema_version: 1`, `family`, explicit integer `seed` in
`[0, 2**64)`, explicit `frame_id`, rectangular `bounds.x/y: [min, max]`, and positive
`limits.max_speed_mps`, `max_acceleration_mps2`, `max_turn_rate_rps`.
Defaults: `sample_dt_s: 0.1`, `time_origin_s: 0`, `pedestrian_count: 1`, no randomization.
Unknown configuration keys, duplicate YAML keys, invalid roles/IDs, nonfinite values,
out-of-bounds waypoints, and invalid limits are rejected.

`randomization.position_jitter_m` is a nonnegative uniform **per-actor translation** applied
to every waypoint; `randomization.speed_scale: [min, max]` samples a per-actor speed multiplier
in `(0, 1]`. Both original and randomized waypoint locations must lie within the bounds;
invalid draws fail explicitly rather than clipping or silently resampling. A seeded local
PRNG stream derived from each actor ID avoids global random state and actor-order effects.
Reproducibility is supported for the same generator/Python version; metadata records versions
and resolved values so experiments need not rely on PRNG compatibility across versions.
Different seeds affect enabled randomization; disabling randomization intentionally removes
seed-dependent motion differences.

Optional `actors` replaces the family's built-in paths while retaining its experiment label.
The list must still match `pedestrian_count` and contain exactly one client. For example:

```yaml
pedestrian_count: 0
actors:
  - id: client
    role: client
    initial_yaw: 0.0
    z_m: 0.0
    waypoints:
      - xy: [-4.0, -2.0]
      - xy: [0.0, -2.0]
      - xy: [0.0, 0.0]
        speed_scale: 0.4
        dwell_s: 3.0
      - xy: [0.0, 3.0]
```

A waypoint may also be a simple `[x, y]` list. `speed_scale` applies to the **incoming**
translation and is a cap multiplier in `(0, 1]`; `dwell_s` is a nonnegative hold after arrival.
The first waypoint's dwell precedes movement; its incoming speed multiplier has no effect.
Repeated waypoints are valid holds. `z_m` is a constant adapter reference height, not inferred
from a pedestrian mesh or map. The abstract frame uses metres, z up, planar yaw in radians
about +z, and planar velocity in that same frame. Initial yaw defaults to the first leg's
heading; it can be overridden. Yaw stays continuous/unwrapped through turns.

## Interpolation and guaranteed constraints

Every straight leg uses analytic quintic smoothstep `s(u) = 10u³ − 15u⁴ + 6u⁵`.
For length `L`, choose duration at least both `1.875 L / speed_cap` and
`sqrt((10/sqrt(3)) L / acceleration_limit)`. This bounds vector speed and acceleration
at **all** simulation times, not just CSV samples. Motion starts/ends with zero velocity
and acceleration. Before a new heading, the stationary actor rotates along the shortest
angular arc using the same quintic profile, with duration at least
`1.875 abs(delta_yaw) / turn_rate_limit`. Yaw rate is bounded continuously; angular
acceleration and jerk are not independently constrained.

This deliberately conservative stop/turn/go model stops at every waypoint. It does not
implement smooth moving corners or constant-speed cruise; “straight walking” describes
path geometry. Slower caps can extend run duration. There is no implicit wall-clock
integration or numerical controller. Interpolation is convex between bounded endpoints,
so positions remain within the axis-aligned bounds. Bounds do not represent walls,
walkable polygons, dynamic collisions, actor radii, or the G1 footprint. Head-on paths
may intersect; these are offline scenarios, not permission to execute motion.
Exports include the exact final time as well as the regular sample grid; relative times
are strictly increasing per actor (shared timestamps across actors are expected).
Unrepresentable timestamps and exports exceeding one million sample intervals are rejected.

## Query API and future adapter boundary

```python
from escort_scenarios import generate, load_config

truth = generate(load_config("research/escort_scenarios/configs/perpendicular_crossing.yaml"))
state = truth.pose_at("client", simulation_time_s=2.25)
# state: actor_id, role, frame_id, time_s, timestamp_s, x/y/z, yaw,
# vx/vy/vz, ax/ay, yaw_rate, valid, provenance
```

`simulation_time_s` is relative to this scenario, independent of call order and host time.
`timestamp_s = time_origin_s + simulation_time_s`; it is not a ROS clock reading or UNIX time
unless an adapter explicitly assigns that meaning. Negative/nonfinite times and unknown IDs
are rejected. `valid=True` describes defined analytic truth, not sensor detection validity.

A future MuJoCo mocap adapter can query states and convert yaw to a quaternion, but must
explicitly configure the scenario-to-MuJoCo transform, reference height, and simulation time
origin. No adapter is implemented. **Do not assume MuJoCo world and ROS map coordinates match.**
Evaluation must likewise supply a verified map transform rather than relabeling frames.

## Export/provenance boundary

- `ground_truth_trajectories.csv`: actor states, time-major ordering; strictly increasing times
  per actor; positions/derivatives, IDs, roles, frame, validity, and explicit ground-truth provenance.
- `scenario.yaml`: effective source configuration including seed override.
- `resolved_scenario.yaml`: defaults, actor paths, random offsets/speed factors, and constraints.
  This is a resolved record, not the input schema for `generate`.
- `resolved_trajectories.json`: exact analytic segment endpoints, times, headings, and actor identity
  for a future replay/adapter implementation; CSV alone is only a sampled representation.
- `metadata.json`: seed, versions, units/time basis, frame, config hash, and `policy_input: false`.
- `xy.png` with `--plot`: spatial paths; circles are starts, crosses are endpoints. This is not
  evidence of timing synchronization or collision-free motion.

These exports contain **complete future evaluation truth**. Keep their directory and the
`Scenario` object inaccessible to prediction/planning/policy modules. A future causal observation
layer must supply only present/past observations, with explicit track IDs, frames, timestamps,
and validity; it must never forward complete paths or future simulator trajectories. No observation,
predictor, planner, logger for robot runtime, ROS publisher, or simulator binding is added here.

## Verification scope

Unit tests cover all four families, deterministic exports, seed variation, stable identities,
custom paths, monotonic timestamps, analytic derivative extrema, continuity, endpoints/holds,
interpolation derivatives, bounds, invalid YAML/config/query rejection, and CLI behavior.
Manual offline runtime verification is the reproduction command above; inspect exported
metadata/CSV and the crossing plot. This does not validate MuJoCo replay, ROS transforms,
Nav2 integration, perception, or safe robot navigation; those remain future, separately
approved tasks. Existing simulation scenes, controllers, sensor relay, and Nav2 are untouched.
