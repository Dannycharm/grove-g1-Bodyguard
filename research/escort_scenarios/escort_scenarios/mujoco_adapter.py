"""Prepare isolated visual-only MuJoCo bundles; explicit operator lifecycle commands.

No imports of ROS/MuJoCo, no process launching, and no writes to baseline assets.
"""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import yaml

from .generator import generate, keys, load_config, number, vector


def intersects(a, b, box, margin):
    """Continuous segment versus expanded XY AABB, including tangency."""
    low, high = 0.0, 1.0
    for i in range(2):
        lo, hi = box[i][0] - margin, box[i][1] + margin
        delta = b[i] - a[i]
        if abs(delta) < 1e-15:
            if not lo <= a[i] <= hi:
                return False
        else:
            enter, leave = sorted(((lo - a[i]) / delta, (hi - a[i]) / delta))
            low, high = max(low, enter), min(high, leave)
            if low > high:
                return False
    return True


def obstacles(scene):
    """Conservative envelopes for the existing navigation primitives.

    Rotated root-body geoms use a sphere envelope around their body origin. Unknown
    shapes/nested bodies fail closed rather than pretending clearance was checked.
    Robot include is excluded here; reserved spawn clearance is validated separately.
    """
    world = ET.parse(scene).getroot().find("worldbody")
    if world is None:
        raise ValueError("Navigation worldbody missing")
    result = []
    for parent in [world, *world.findall("body")]:
        if parent is not world and parent.findall("body"):
            raise ValueError("Nested scene bodies need explicit clearance support")
        origin = vector([float(v) for v in parent.get("pos", "0 0 0").split()], 3, "body pos")
        for geom in parent.findall("geom"):
            kind = geom.get("type", "sphere")
            if kind == "plane":
                continue
            size = [float(v) for v in geom.get("size", "").split()]
            pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
            if len(pos) != 3 or kind not in ("box", "cylinder", "sphere"):
                raise ValueError(f"Unsupported clearance geom: {kind}")
            if kind == "box" and len(size) == 3:
                extent = size
            elif kind == "cylinder" and len(size) == 2:
                extent = [size[0], size[0], size[1]]
            elif kind == "sphere" and len(size) == 1:
                extent = size * 3
            else:
                raise ValueError("Invalid scene primitive size")
            if any(k in parent.attrib or k in geom.attrib for k in ("euler", "quat", "axisangle", "xyaxes", "zaxis")):
                radius = math.sqrt(sum(v*v for v in pos)) + math.sqrt(sum(v*v for v in extent))
                center, extent = origin, [radius] * 3
            else:
                center = [origin[i] + pos[i] for i in range(3)]
            result.append((geom.get("name", parent.get("name", "unnamed")),
                           [[center[i] - extent[i], center[i] + extent[i]] for i in range(2)]))
    return result


def prepare(config_path, output, navigation_scene, robot_model, map_yaml):
    cfg = load_config(config_path)
    keys(cfg, {"scenario", "scenario_to_mujoco", "scene_bounds", "actor_radius_m",
               "clearance_m", "g1_spawn_xy", "g1_clearance_radius_m"}, "adapter")
    if not isinstance(cfg.get("scenario"), str):
        raise ValueError("scenario must name a generator YAML file")
    scenario_path = (Path(config_path).resolve().parent / cfg["scenario"]).resolve()
    scenario = generate(load_config(scenario_path))
    transform = vector(cfg.get("scenario_to_mujoco"), 3, "scenario_to_mujoco [x,y,yaw]")
    keys(cfg.get("scene_bounds"), {"x", "y"}, "scene_bounds")
    bounds = {axis: vector(cfg["scene_bounds"].get(axis), 2, axis) for axis in ("x", "y")}
    if any(lo >= hi for lo, hi in bounds.values()):
        raise ValueError("scene bounds require min < max")
    radius = number(cfg.get("actor_radius_m"), "actor_radius_m", positive=True)
    if radius > 0.5:
        raise ValueError("Actor radius must be <= 0.5 m")
    margin = number(cfg.get("clearance_m", 0.1), "clearance_m", nonnegative=True)
    robot_radius = number(cfg.get("g1_clearance_radius_m"), "g1_clearance_radius_m", positive=True)
    robot_spawn = vector(cfg.get("g1_spawn_xy"), 2, "g1_spawn_xy")
    blocks = obstacles(navigation_scene)
    c, s = math.cos(transform[2]), math.sin(transform[2])

    def xy(point):
        return [transform[0] + c*point[0] - s*point[1],
                transform[1] + s*point[0] + c*point[1]]

    actors = []
    for actor in scenario.actors:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", actor.actor_id):
            raise ValueError("Adapter actor IDs must match [A-Za-z][A-Za-z0-9_]{0,23}")
        if actor.z != 0:
            raise ValueError("This ground-level primitive adapter requires z_m=0")
        points = [xy(actor.initial_xy)] + [xy(seg.xy1) for seg in actor.segments]
        for point in points:
            if any(not bounds[axis][0] + radius <= point[i] <= bounds[axis][1] - radius
                   for i, axis in enumerate(("x", "y"))):
                raise ValueError(f"Actor footprint outside scene bounds: {actor.actor_id}")
        for a, b in zip(points, points[1:]):
            for name, box in blocks:
                if intersects(a, b, box, radius + margin):
                    raise ValueError(f"Path of {actor.actor_id} intersects conservative obstacle {name}")
        start = points[0]
        if math.dist(start, robot_spawn) <= radius + robot_radius + margin:
            raise ValueError("Actor spawn intersects reserved G1 footprint")
        if any(math.dist(start, other["initial_xy"]) <= 2*radius + margin for other in actors):
            raise ValueError("Actor spawn footprints overlap")
        segments = []
        for seg in actor.segments:
            data = asdict(seg)
            data.update(xy0=xy(seg.xy0), xy1=xy(seg.xy1),
                        yaw0=seg.yaw0 + transform[2], yaw1=seg.yaw1 + transform[2])
            segments.append(data)
        actors.append({"id": actor.actor_id, "role": actor.role,
                       "body": f"escort_{actor.actor_id}", "radius": radius,
                       "initial_xy": start, "initial_yaw": actor.initial_yaw + transform[2],
                       "segments": segments})
    target = Path(output).resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError("Bundle output must be new or empty")
    target.mkdir(parents=True, exist_ok=True)
    # Copy the model and resolve mesh paths; do not mutate robot topology or motor data.
    robot_model = Path(robot_model).resolve()
    robot = ET.parse(robot_model).getroot()
    compiler = robot.find("compiler")
    if compiler is None:
        raise ValueError("Robot compiler/mesh directory missing")
    compiler.set("meshdir", str((robot_model.parent / compiler.get("meshdir", ".")).resolve()))
    if robot.findall("include"):
        raise ValueError("Robot model includes need explicit resolution support")
    ET.ElementTree(robot).write(target / "robot.xml", encoding="unicode")
    scene = ET.parse(navigation_scene).getroot()
    includes = scene.findall("include")
    if len(includes) != 1 or includes[0].get("file") != "g1_29dof.xml":
        raise ValueError("Expected original navigation scene's single G1 include")
    includes[0].set("file", "robot.xml")
    ET.ElementTree(scene).write(target / "navigation.xml", encoding="unicode")
    overlay = ET.Element("mujoco", model="escort_navigation")
    ET.SubElement(overlay, "include", file="navigation.xml")
    eq = ET.SubElement(overlay, "equality")
    ET.SubElement(eq, "weld", name="pelvis_startup_hold", body1="pelvis")
    world = ET.SubElement(overlay, "worldbody")
    for actor in actors:
        heading = actor["initial_yaw"]
        body = ET.SubElement(world, "body", name=actor["body"], mocap="true",
                             pos=f'{actor["initial_xy"][0]} {actor["initial_xy"][1]} 0',
                             quat=f"{math.cos(heading/2)} 0 0 {math.sin(heading/2)}")
        color = "0.1 0.7 0.95 1" if actor["role"] == "client" else "0.95 0.5 0.1 1"
        ET.SubElement(body, "geom", name=actor["body"]+"_torso", type="capsule",
                      fromto="0 0 0.24 0 0 1.35", size=str(radius), rgba=color,
                      group="2", contype="0", conaffinity="0")
        ET.SubElement(body, "geom", name=actor["body"]+"_head", type="sphere",
                      pos="0 0 1.64", size=str(min(radius, 0.17)), rgba=color,
                      group="2", contype="0", conaffinity="0")
    ET.indent(overlay)
    ET.ElementTree(overlay).write(target / "scene.xml", encoding="unicode")
    playback = {"schema_version": 1, "bounds": bounds, "duration": scenario.duration_s,
                "robot_clearance": robot_radius + margin, "actors": actors}
    (target / "playback.yaml").write_text(yaml.safe_dump(playback, sort_keys=True))
    (target / "adapter.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    # Reuse the existing sampler's object-pose wire format. Separate sensor config;
    # never modify the baseline file or any already-running sampler.
    sensor_source = Path(navigation_scene).resolve().parent.parent / "config/sim_sensors.yaml"
    sensor_config = load_config(sensor_source)
    sensor_config["object_bodies"] = list(dict.fromkeys(
        sensor_config.get("object_bodies", []) + [a["body"] for a in actors] + ["pelvis"]))
    (target / "sensors.yaml").write_text(yaml.safe_dump(sensor_config, sort_keys=True))
    scenario_id = hashlib.sha256(json.dumps(scenario.resolved_config, sort_keys=True).encode()).hexdigest()[:16]
    roster = {"scenario_id": scenario_id, "seed": scenario.resolved_config["seed"],
              "actors": [{"id": a["id"], "role": a["role"], "body": a["body"]} for a in actors]}
    (target / "observation_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=True))
    scenario.export(target / "evaluation_truth")
    metadata = {"collision_mode": "visual_only", "geom_group": 2,
                "frame": "mujoco_world", "scenario_frame": scenario.resolved_config["frame_id"],
                "scenario_to_mujoco": transform, "ros_map_transform": None,
                "baseline_files_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                          for p in (navigation_scene, robot_model, map_yaml)},
                "map_yaml": str(Path(map_yaml).resolve()),
                "clearance_scope": "continuous static primitive envelopes and reserved initial G1 footprint; not dynamic collision validation"}
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    return playback


def command(bundle, action, sequence):
    """Explicit operator mutation; never called by prepare or startup."""
    if action not in ("start", "pause", "reset", "replay") or type(sequence) is not int or sequence <= 0:
        raise ValueError("Valid action and positive integer sequence required")
    root = Path(bundle)
    if not (root / "playback.yaml").is_file():
        raise ValueError("Prepared bundle required")
    target = root / "control.yaml"
    if target.exists() and load_config(target)["sequence"] >= sequence:
        raise ValueError("Command sequence must increase")
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix="control-", delete=False) as stream:
        yaml.safe_dump({"sequence": sequence, "action": action}, stream)
        temporary = stream.name
    os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("config")
    prep.add_argument("--output", required=True)
    prep.add_argument("--navigation-scene", required=True)
    prep.add_argument("--robot-model", required=True)
    prep.add_argument("--map-yaml", required=True)
    cmd = sub.add_parser("command")
    cmd.add_argument("--bundle", required=True)
    cmd.add_argument("--action", choices=("start", "pause", "reset", "replay"), required=True)
    cmd.add_argument("--sequence", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "prepare":
            prepare(args.config, args.output, args.navigation_scene, args.robot_model, args.map_yaml)
            print("Prepared visual-only bundle; no simulator launched or lifecycle command sent.")
        else:
            command(args.bundle, args.action, args.sequence)
            print("Command queued; inspect status.yaml for acknowledgement/rejection.")
    except (ValueError, OSError, ET.ParseError, yaml.YAMLError) as error:
        parser.exit(2, f"Adapter error: {error}\n")


if __name__ == "__main__":
    main()
