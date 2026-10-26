"""Offline behavioral and continuous-bound tests; no ROS or MuJoCo."""

import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from escort_scenarios import generate, load_config
from escort_scenarios.generator import FAMILIES

ROOT = Path(__file__).resolve().parents[1]


def config(family="perpendicular_crossing"):
    return load_config(ROOT / "configs" / f"{family}.yaml")


class GeneratorTests(unittest.TestCase):
    def test_same_seed_identical_resolved_trajectories_and_exports(self):
        first, second = generate(config()), generate(config())
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            first.export(a)
            second.export(b)
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes())

    def test_seed_changes_random_parameters(self):
        original = config()
        changed = copy.deepcopy(original)
        changed["seed"] += 1
        self.assertNotEqual(generate(original).resolved_config, generate(changed).resolved_config)
        self.assertNotEqual(generate(original).pose_at("client", 1), generate(changed).pose_at("client", 1))

    def test_generate_does_not_mutate_input(self):
        original = config()
        before = copy.deepcopy(original)
        generate(original)
        self.assertEqual(original, before)

    def test_actor_ids_and_roles_are_stable(self):
        for family in FAMILIES:
            for seed in range(3):
                cfg = config(family)
                cfg["seed"] = seed
                scenario = generate(cfg)
                for t in (0, 0.123, scenario.duration_s, scenario.duration_s + 10):
                    self.assertEqual([(a.at(t).actor_id, a.at(t).role) for a in scenario.actors],
                                     [("client", "client"), ("pedestrian_1", "pedestrian"),
                                      ("pedestrian_2", "pedestrian")])

    def test_zero_one_two_pedestrians(self):
        for n in range(3):
            cfg = config()
            cfg["pedestrian_count"] = n
            self.assertEqual(len(generate(cfg).actors), n + 1)

    def test_timestamps_and_endpoints_in_csv(self):
        cfg = config()
        cfg["time_origin_s"] = 1000
        scenario = generate(cfg)
        with tempfile.TemporaryDirectory() as directory:
            scenario.export(directory)
            with (Path(directory) / "ground_truth_trajectories.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            for actor in scenario.actors:
                samples = [r for r in rows if r["actor_id"] == actor.actor_id]
                times = [float(r["time_s"]) for r in samples]
                self.assertEqual(times[0], 0)
                self.assertEqual(times[-1], scenario.duration_s)
                self.assertTrue(all(a < b for a, b in zip(times, times[1:])))
                for row in samples:
                    self.assertAlmostEqual(float(row["timestamp_s"]), 1000 + float(row["time_s"]))
                    self.assertEqual(row["provenance"], "simulator_ground_truth")

    def test_continuous_extrema_constraints_bounds_and_boundary_continuity(self):
        # Include analytic derivative extrema, not just the exported sample grid.
        extrema = (0, (3 - math.sqrt(3)) / 6, 0.5, (3 + math.sqrt(3)) / 6, 1)
        for family in FAMILIES:
            for seed in (0, 42, 777):
                cfg = config(family)
                cfg["seed"] = seed
                scenario = generate(cfg)
                for actor in scenario.actors:
                    times = scenario.sample_times()
                    for segment in actor.segments:
                        times += [segment.start + u * segment.duration for u in extrema]
                        end = segment.start + segment.duration
                        left, right = actor.at(max(0, end - 1e-7)), actor.at(end + 1e-7)
                        self.assertLess(math.hypot(left.x - right.x, left.y - right.y), 1e-5)
                        self.assertLess(abs(left.yaw - right.yaw), 1e-5)
                    for time in times:
                        state = actor.at(time)
                        self.assertLessEqual(math.hypot(state.vx, state.vy), cfg["limits"]["max_speed_mps"] + 1e-9)
                        self.assertLessEqual(math.hypot(state.ax, state.ay), cfg["limits"]["max_acceleration_mps2"] + 1e-9)
                        self.assertLessEqual(abs(state.yaw_rate), cfg["limits"]["max_turn_rate_rps"] + 1e-9)
                        self.assertTrue(cfg["bounds"]["x"][0] <= state.x <= cfg["bounds"]["x"][1])
                        self.assertTrue(cfg["bounds"]["y"][0] <= state.y <= cfg["bounds"]["y"][1])

    def test_interpolation_velocity_and_acceleration_match_position(self):
        scenario = generate(config("client_turn_slowdown_stop"))
        for actor in scenario.actors:
            for segment in actor.segments:
                for u in (0.2, 0.5, 0.8):
                    t = segment.start + u * segment.duration
                    h = 1e-4
                    left, mid, right = actor.at(t - h), actor.at(t), actor.at(t + h)
                    self.assertAlmostEqual((right.x - left.x) / (2 * h), mid.vx, places=6)
                    self.assertAlmostEqual((right.y - left.y) / (2 * h), mid.vy, places=6)
                    self.assertAlmostEqual((right.vx - left.vx) / (2 * h), mid.ax, places=6)
                    self.assertAlmostEqual((right.vy - left.vy) / (2 * h), mid.ay, places=6)
                    self.assertAlmostEqual((right.yaw - left.yaw) / (2 * h), mid.yaw_rate, places=6)

    def test_midpoint_endpoints_and_final_hold(self):
        cfg = config("straight")
        cfg.update(pedestrian_count=0, randomization={})
        actor = generate(cfg).actors[0]
        self.assertEqual((actor.at(0).x, actor.at(0).y), (-4, 0))
        self.assertAlmostEqual(actor.at(actor.duration_s / 2).x, 0)
        for time in (actor.duration_s, actor.duration_s + 100):
            state = actor.at(time)
            self.assertEqual((state.x, state.y), (4, 0))
            self.assertEqual((state.vx, state.vy, state.ax, state.ay, state.yaw_rate), (0, 0, 0, 0, 0))

    def test_client_turn_slowdown_and_stop(self):
        scenario = generate(config("client_turn_slowdown_stop"))
        actor = scenario.actors[0]
        translations = [s for s in actor.segments if s.xy0 != s.xy1]
        rotations = [s for s in actor.segments if s.yaw0 != s.yaw1]
        holds = [s for s in actor.segments if s.xy0 == s.xy1 and s.yaw0 == s.yaw1]
        self.assertTrue(rotations)
        self.assertTrue(any(s.duration == 3 for s in holds))
        peaks = [math.hypot(actor.at(s.start + s.duration / 2).vx,
                            actor.at(s.start + s.duration / 2).vy) for s in translations]
        self.assertLess(peaks[1], peaks[0])

    def test_family_geometry(self):
        for family in FAMILIES:
            cfg = config(family)
            cfg["randomization"] = {}
            scenario = generate(cfg)
            client, pedestrian = scenario.actors[:2]
            if family == "perpendicular_crossing":
                self.assertEqual(client.initial_xy[1], 0)
                self.assertEqual(pedestrian.initial_xy[0], 0)
            elif family == "head_on":
                self.assertEqual(client.initial_xy, (pedestrian.at(pedestrian.duration_s).x, pedestrian.at(pedestrian.duration_s).y))
                self.assertEqual((pedestrian.at(pedestrian.duration_s).x, pedestrian.at(pedestrian.duration_s).y), (-4, 0))
            elif family == "straight":
                self.assertAlmostEqual(client.initial_yaw, pedestrian.initial_yaw)

    def test_custom_actor_paths_and_order_independence(self):
        cfg = config()
        cfg["pedestrian_count"] = 1
        cfg["actors"] = [
            {"id": "escort_client", "role": "client", "initial_yaw": math.pi - 0.01,
             "z_m": 0.9, "waypoints": [[1, 0], [-1, -0.01]]},
            {"id": "walker", "role": "pedestrian", "waypoints": [[0, -1], [0, 1]]},
        ]
        scenario = generate(cfg)
        reversed_cfg = copy.deepcopy(cfg)
        reversed_cfg["actors"].reverse()
        other = generate(reversed_cfg)
        self.assertEqual(scenario.pose_at("walker", 1), other.pose_at("walker", 1))
        self.assertEqual(scenario.pose_at("escort_client", 0).z, 0.9)
        # Shortest yaw arc crosses pi without a discontinuity/wrap jump.
        actor = scenario.actors[0]
        rotations = [s for s in actor.segments if s.yaw0 != s.yaw1]
        self.assertLess(abs(rotations[0].yaw1 - rotations[0].yaw0), 0.1)

    def test_invalid_configs(self):
        mutations = [
            {"seed": None}, {"seed": True}, {"seed": -1}, {"schema_version": True},
            {"family": "unknown"}, {"frame_id": ""}, {"sample_dt_s": 0},
            {"sample_dt_s": float("nan")}, {"pedestrian_count": 3},
            {"pedestrian_count": True}, {"bounds": {"x": [1, -1], "y": [-6, 6]}},
            {"bounds": {"x": [-1, 1], "y": [-1, 1]}},
            {"randomization": {"speed_scale": [0, 1]}},
            {"randomization": {"position_jitter_m": -1}},
            {"randomization": {"position_jitter_m": 1000}},
            {"limits": {"max_speed_mps": -1}}, {"typo": 5}, {"actors": []},
            {"sample_dt_s": 1e-12}, {"time_origin_s": 1e30},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                cfg = config()
                cfg.update(mutation)
                generate(cfg)
        for actors in (
            [{"id": "x", "role": "pedestrian", "waypoints": [[0, 0], [1, 0]]}],
            [{"id": "x", "role": "client", "waypoints": [[0, 0]]}],
            [{"id": "x", "role": "client", "waypoints": [[0, 0], {"xy": [1, 0], "speed_scale": 2}]}],
            [{"id": "x", "role": "client", "waypoints": [[0, 0], [1, 0]]}] * 2,
        ):
            with self.subTest(actors=actors), self.assertRaises(ValueError):
                generate(dict(config(), actors=actors, pedestrian_count=0))

    def test_tiny_duration_keeps_zero_and_exact_endpoint(self):
        cfg = config("straight")
        cfg.update(pedestrian_count=0, randomization={}, actors=[{
            "id": "client", "role": "client", "waypoints": [[0, 0], [1e-30, 0]]}])
        scenario = generate(cfg)
        self.assertEqual(scenario.sample_times(), [0.0, scenario.duration_s])

    def test_repeated_waypoints_hold_without_jumps(self):
        cfg = config("straight")
        cfg.update(pedestrian_count=0, randomization={}, actors=[{
            "id": "client", "role": "client", "waypoints": [
                {"xy": [0, 0], "dwell_s": 2}, {"xy": [0, 0], "dwell_s": 1}, [1, 0]]}])
        actor = generate(cfg).actors[0]
        for t in (0, 1, 2, 2.5, 3):
            self.assertEqual((actor.at(t).x, actor.at(t).vx), (0, 0))
        self.assertEqual(actor.at(actor.duration_s).x, 1)

    def test_query_order_does_not_change_states(self):
        scenario = generate(config())
        expected = [scenario.pose_at("client", t) for t in (0, 1, 5)]
        for index in (2, 0, 1, 2, 1):
            self.assertEqual(scenario.pose_at("client", (0, 1, 5)[index]), expected[index])

    def test_invalid_queries(self):
        scenario = generate(config())
        for time in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                scenario.pose_at("client", time)
        with self.assertRaises(KeyError):
            scenario.pose_at("missing", 0)

    def test_yaml_duplicate_keys_and_unsafe_tags_rejected(self):
        import yaml

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("seed: 1\nseed: 2\n")
            with self.assertRaises(ValueError):
                load_config(path)
            path.write_text("!!python/object:foo {}")
            with self.assertRaises(yaml.YAMLError):
                load_config(path)

    def test_export_truth_metadata_and_no_overwrite(self):
        scenario = generate(config())
        with tempfile.TemporaryDirectory() as directory:
            scenario.export(directory)
            self.assertNotIn(b"\r", (Path(directory) / "ground_truth_trajectories.csv").read_bytes())
            meta = json.loads((Path(directory) / "metadata.json").read_text())
            self.assertFalse(meta["policy_input"])
            self.assertIsNone(meta["coordinate_transform_to_mujoco_or_ros_map"])
            self.assertEqual(meta["seed"], 42)
            segments = json.loads((Path(directory) / "resolved_trajectories.json").read_text())
            self.assertEqual(segments[0]["actor_id"], "client")
            with self.assertRaises(ValueError):
                scenario.export(directory)

    def test_cli_export_and_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-m", "escort_scenarios", str(ROOT / "configs/straight.yaml"),
                       "--output", str(Path(directory) / "run"), "--seed", "7"]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(directory) / "run/metadata.json").read_text())["seed"], 7)
            command[-1] = "-1"
            self.assertEqual(subprocess.run(command, env=env, capture_output=True, timeout=20).returncode, 2)


if __name__ == "__main__":
    unittest.main()
