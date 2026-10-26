"""Analytic stop/turn/go trajectories, independent of ROS and simulator bindings.

Quintic smoothstep extrema: max ds/du=15/8, max |d²s/du²|=10/sqrt(3).
Each translation and rotation starts and ends at rest; all positions are convex
combinations of bounded waypoints. Bounds and derivative limits hold continuously.
"""

import copy
import csv
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

FAMILIES = ("straight", "perpendicular_crossing", "head_on", "client_turn_slowdown_stop")
VERSION = "0.1.0"


def number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"{name} must be a mapping with keys {sorted(allowed)}")


def vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} requires {size} numbers")
    return [number(v, name) for v in value]


def load_config(path):
    # Reject duplicate keys rather than silently allowing YAML to overwrite settings.
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"Non-string or duplicate YAML key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    with Path(path).open() as stream:
        return yaml.load(stream, Loader=UniqueLoader)


def smooth(u):
    return u**3 * (10 + u * (-15 + 6 * u))


def ds(u):
    return 30 * u**2 * (1 - u) ** 2


def dds(u):
    return 60 * u * (1 - u) * (1 - 2 * u)


def angle_delta(start, end):
    return (end - start + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class ActorState:
    actor_id: str
    role: str
    frame_id: str
    time_s: float
    timestamp_s: float
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    yaw_rate: float
    valid: bool = True
    provenance: str = "simulator_ground_truth"


@dataclass(frozen=True)
class Segment:
    start: float
    duration: float
    xy0: tuple
    xy1: tuple
    yaw0: float
    yaw1: float


@dataclass(frozen=True)
class ActorTrajectory:
    actor_id: str
    role: str
    frame_id: str
    time_origin_s: float
    z: float
    initial_xy: tuple
    initial_yaw: float
    segments: tuple
    duration_s: float

    def at(self, time_s):
        """Query relative simulation seconds; hold final pose after completion.

        Yaw is continuous/unwrapped radians. Velocity and acceleration are analytic,
        not finite differences. Negative/nonfinite times are rejected.
        """
        t = number(time_s, "time_s", nonnegative=True)
        xy, yaw = self.initial_xy, self.initial_yaw
        vx = vy = ax = ay = yaw_rate = 0.0
        for segment in self.segments:
            if t >= segment.start + segment.duration:
                xy, yaw = segment.xy1, segment.yaw1
                continue
            if t < segment.start:
                break
            u = (t - segment.start) / segment.duration
            dx = segment.xy1[0] - segment.xy0[0]
            dy = segment.xy1[1] - segment.xy0[1]
            xy = (segment.xy0[0] + dx * smooth(u), segment.xy0[1] + dy * smooth(u))
            yaw = segment.yaw0 + (segment.yaw1 - segment.yaw0) * smooth(u)
            vx, vy = dx * ds(u) / segment.duration, dy * ds(u) / segment.duration
            ax, ay = dx * dds(u) / segment.duration**2, dy * dds(u) / segment.duration**2
            yaw_rate = (segment.yaw1 - segment.yaw0) * ds(u) / segment.duration
            break
        return ActorState(
            self.actor_id, self.role, self.frame_id, t, self.time_origin_s + t,
            xy[0], xy[1], self.z, yaw, vx, vy, 0.0, ax, ay, yaw_rate,
        )


@dataclass(frozen=True)
class Scenario:
    config: dict
    resolved_config: dict
    actors: tuple
    sample_dt_s: float

    @property
    def duration_s(self):
        return max(actor.duration_s for actor in self.actors)

    def pose_at(self, actor_id, simulation_time_s):
        """Evaluation/adapter API, not a policy observation interface."""
        for actor in self.actors:
            if actor.actor_id == actor_id:
                return actor.at(simulation_time_s)
        raise KeyError(actor_id)

    def sample_times(self):
        count = math.floor(self.duration_s / self.sample_dt_s)
        times = [i * self.sample_dt_s for i in range(count + 1)]
        if times[-1] < self.duration_s:
            times.append(self.duration_s)
        else:
            times[-1] = self.duration_s
        origin = self.resolved_config["time_origin_s"]
        if any(origin + a >= origin + b for a, b in zip(times, times[1:])):
            raise ValueError("Timestamp precision cannot represent distinct sample times")
        return times

    def export(self, directory):
        """Export evaluation truth, configs, and reproducibility metadata.

        Refuse nonempty destinations to preserve previous experimental assets.
        """
        target = Path(directory)
        if target.exists() and any(target.iterdir()):
            raise ValueError(f"Output directory is not empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        rows = [asdict(actor.at(t)) for t in self.sample_times() for actor in self.actors]
        with (target / "ground_truth_trajectories.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        for name, value in (("scenario.yaml", self.config), ("resolved_scenario.yaml", self.resolved_config)):
            (target / name).write_text(yaml.safe_dump(value, sort_keys=True))
        # Segments support exact analytic replay, independent of the CSV sampling rate.
        payload = [asdict(actor) for actor in self.actors]
        (target / "resolved_trajectories.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        digest = hashlib.sha256(
            json.dumps(self.resolved_config, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        metadata = {
            "schema_version": 1, "generator_version": VERSION,
            "python_version": sys.version.split()[0], "pyyaml_version": yaml.__version__,
            "seed": self.resolved_config["seed"], "family": self.resolved_config["family"],
            "frame_id": self.resolved_config["frame_id"],
            "coordinate_transform_to_mujoco_or_ros_map": None,
            "time_basis": "relative simulation seconds + explicit time_origin_s",
            "units": "metres, seconds, radians", "provenance": "simulator_ground_truth",
            "policy_input": False, "duration_s": self.duration_s,
            "interpolation": "quintic_stop_turn_go_v1", "resolved_config_sha256": digest,
            "actor_ids": [a.actor_id for a in self.actors],
            "safety_scope": "kinematic limits and axis-aligned bounds only; no obstacle/collision checks",
        }
        (target / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n")


def templates(family, pedestrians):
    client = [[-4, 0], [4, 0]]
    others = [[[4, 1.5], [-4, 1.5]], [[-3, -2], [3, -2]]]
    if family == "straight":
        others = [[[-4, 1.5], [4, 1.5]], [[-4, -1.5], [4, -1.5]]]
    elif family == "perpendicular_crossing":
        others = [[[0, -4], [0, 4]], [[2, 4], [2, -4]]]
    elif family == "head_on":
        others = [[[4, 0], [-4, 0]], [[4, 1], [-4, 1]]]
    elif family == "client_turn_slowdown_stop":
        client = [
            {"xy": [-4, -2]}, {"xy": [0, -2]},
            {"xy": [0, 0], "speed_scale": 0.4, "dwell_s": 3},
            {"xy": [0, 3], "speed_scale": 0.6},
        ]
    return [
        {"id": "client", "role": "client", "waypoints": client},
        *[{"id": f"pedestrian_{i + 1}", "role": "pedestrian", "waypoints": others[i]}
          for i in range(pedestrians)],
    ]


def generate(config):
    """Resolve an explicit seed into continuously bounded ground-truth paths."""
    keys(config, {"schema_version", "family", "seed", "frame_id", "time_origin_s",
                  "sample_dt_s", "bounds", "limits", "randomization", "pedestrian_count",
                  "actors"}, "scenario")
    config = copy.deepcopy(config)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    family = config.get("family")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("An explicit integer seed in [0, 2**64) is required")
    frame = config.get("frame_id")
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError("An explicit nonempty frame_id is required")
    origin = number(config.get("time_origin_s", 0), "time_origin_s", nonnegative=True)
    dt = number(config.get("sample_dt_s", 0.1), "sample_dt_s", positive=True)
    bounds = config.get("bounds")
    keys(bounds, {"x", "y"}, "bounds")
    bounds = {axis: vector(bounds.get(axis), 2, f"bounds.{axis}") for axis in ("x", "y")}
    if any(lo >= hi for lo, hi in bounds.values()):
        raise ValueError("Bounds require min < max")
    limits = config.get("limits")
    keys(limits, {"max_speed_mps", "max_acceleration_mps2", "max_turn_rate_rps"}, "limits")
    limits = {name: number(limits.get(name), name, positive=True) for name in (
        "max_speed_mps", "max_acceleration_mps2", "max_turn_rate_rps")}
    randomization = config.get("randomization", {})
    keys(randomization, {"position_jitter_m", "speed_scale"}, "randomization")
    jitter = number(randomization.get("position_jitter_m", 0), "position_jitter_m", nonnegative=True)
    speed_range = vector(randomization.get("speed_scale", [1, 1]), 2, "random speed_scale")
    if not 0 < speed_range[0] <= speed_range[1] <= 1:
        raise ValueError("Random speed_scale must satisfy 0 < min <= max <= 1")
    count = config.get("pedestrian_count", 1)
    if type(count) is not int or not 0 <= count <= 2:
        raise ValueError("pedestrian_count must be 0, 1 or 2")
    actors = config.get("actors", templates(family, count))
    if not isinstance(actors, list) or not 1 <= len(actors) <= 3:
        raise ValueError("Require one client and up to two pedestrians")
    ids, roles, trajectories, resolved_actors = set(), [], [], []
    for actor in actors:
        keys(actor, {"id", "role", "waypoints", "initial_yaw", "z_m"}, "actor")
        actor_id, role = actor.get("id"), actor.get("role")
        if not isinstance(actor_id, str) or not actor_id.strip() or actor_id in ids:
            raise ValueError("Actor IDs must be unique nonempty strings")
        if role not in ("client", "pedestrian"):
            raise ValueError("Actor role must be client or pedestrian")
        ids.add(actor_id)
        roles.append(role)
        # Per-ID streams preserve random draws if actor order changes.
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{actor_id}".encode()).digest(), "big"))
        offset = (rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter))
        speed_scale = rng.uniform(*speed_range)
        points = actor.get("waypoints")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("Each actor needs at least two waypoints")
        resolved_points = []
        for point in points:
            point = point if isinstance(point, dict) else {"xy": point}
            keys(point, {"xy", "dwell_s", "speed_scale"}, "waypoint")
            original = vector(point.get("xy"), 2, "waypoint.xy")
            xy = [original[i] + offset[i] for i in range(2)]
            for candidate in (original, xy):
                if any(not bounds[axis][0] <= candidate[i] <= bounds[axis][1]
                       for i, axis in enumerate(("x", "y"))):
                    raise ValueError(f"Waypoint outside bounds for {actor_id}: {candidate}")
            scale = number(point.get("speed_scale", 1), "waypoint.speed_scale", positive=True)
            if scale > 1:
                raise ValueError("Waypoint speed_scale must be <= 1")
            resolved_points.append({"xy": xy, "dwell_s": number(point.get("dwell_s", 0),
                                    "dwell_s", nonnegative=True), "speed_scale": scale})
        xy = tuple(resolved_points[0]["xy"])
        initial_xy = xy
        first_delta = [resolved_points[1]["xy"][i] - xy[i] for i in range(2)]
        yaw = number(actor.get("initial_yaw", math.atan2(first_delta[1], first_delta[0])), "initial_yaw")
        initial_yaw = yaw
        z = number(actor.get("z_m", 0), "z_m")
        segments, clock = [], 0.0

        def add(duration, end_xy, end_yaw):
            nonlocal clock, xy, yaw
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("Segment duration must be finite and nonnegative")
            if duration > 0:
                if not math.isfinite(clock + duration) or clock + duration <= clock:
                    raise ValueError("Segment duration cannot be represented at this scenario time")
                segments.append(Segment(clock, duration, xy, tuple(end_xy), yaw, end_yaw))
                clock += duration
            xy, yaw = tuple(end_xy), end_yaw

        add(resolved_points[0]["dwell_s"], xy, yaw)
        for point in resolved_points[1:]:
            end = point["xy"]
            distance = math.hypot(end[0] - xy[0], end[1] - xy[1])
            if distance > 0:
                heading = yaw + angle_delta(yaw, math.atan2(end[1] - xy[1], end[0] - xy[0]))
                add(1.875 * abs(heading - yaw) / limits["max_turn_rate_rps"], xy, heading)
                speed = limits["max_speed_mps"] * speed_scale * point["speed_scale"]
                if speed == 0:
                    raise ValueError("Speed limit underflows numeric precision")
                duration = max(
                    1.875 * distance / speed,
                    math.sqrt((10 / math.sqrt(3)) * distance / limits["max_acceleration_mps2"]),
                )
                add(duration, end, yaw)
            add(point["dwell_s"], xy, yaw)
        trajectories.append(ActorTrajectory(actor_id, role, frame, origin, z, initial_xy,
                                            initial_yaw, tuple(segments), clock))
        resolved_actors.append({"id": actor_id, "role": role, "z_m": z,
                                "initial_yaw": initial_yaw, "speed_scale": speed_scale,
                                "position_offset_m": list(offset), "waypoints": resolved_points})
    if roles.count("client") != 1 or roles.count("pedestrian") != count:
        raise ValueError("Require exactly one client and pedestrian_count matching pedestrian roles")
    if max(a.duration_s for a in trajectories) == 0:
        raise ValueError("Scenario must have positive duration")
    resolved = {"schema_version": 1, "family": family, "seed": seed, "frame_id": frame,
                "time_origin_s": origin, "sample_dt_s": dt, "bounds": bounds,
                "limits": limits, "pedestrian_count": count, "actors": resolved_actors}
    # Prevent timestamp precision loss, duplicate CSV times, or unbounded export allocation.
    duration = max(a.duration_s for a in trajectories)
    if not math.isfinite(duration) or duration / dt > 1_000_000:
        raise ValueError("Scenario exceeds one million sample intervals")
    if origin + dt == origin or origin + duration == origin:
        raise ValueError("time_origin_s cannot represent this scenario's timestamps")
    return Scenario(config, resolved, tuple(trajectories), dt)
