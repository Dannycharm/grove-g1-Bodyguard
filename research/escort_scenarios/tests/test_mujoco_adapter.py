"""Offline bundle and playback tests; C++ uses a clearly labeled MuJoCo API double."""

import ast
import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import yaml

from escort_scenarios.generator import generate, load_config
from escort_scenarios.mujoco_adapter import command, intersects, prepare

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "research/escort_scenarios"
BRINGUP = ROOT / "workspace/src/g1_bringup"
SIM = ROOT / "workspace/vendor/unitree_mujoco/simulate"
NAV = BRINGUP / "mjcf/g1_navigation_scene.xml"
ROBOT = SIM.parent / "unitree_robots/g1/g1_29dof.xml"
MAP = ROOT / "workspace/src/g1_navigation/maps/facility.yaml"


def bundle(output, cfg=None):
    return prepare(cfg or PACKAGE / "configs/mujoco_navigation.yaml", output, NAV, ROBOT, MAP)


class AdapterTests(unittest.TestCase):
    def test_bundle_actor_discovery_visual_geometry_and_baseline_preservation(self):
        originals = {p: p.read_bytes() for p in (NAV, ROBOT, MAP)}
        with tempfile.TemporaryDirectory() as tmp:
            data = bundle(tmp)
            overlay = ET.parse(Path(tmp) / "scene.xml").getroot()
            actors = overlay.findall("worldbody/body")
            self.assertEqual([a.get("name") for a in actors],
                             ["escort_client", "escort_pedestrian_1", "escort_pedestrian_2"])
            self.assertEqual(len(data["actors"]), 3)
            for actor in actors:
                self.assertEqual(actor.get("mocap"), "true")
                self.assertIsNone(actor.find("joint"))
                for geom in actor.findall("geom"):
                    self.assertEqual((geom.get("contype"),geom.get("conaffinity"),geom.get("group")),("0","0","2"))
            self.assertIsNotNone(overlay.find("equality/weld[@name='pelvis_startup_hold']"))
            self.assertFalse((Path(tmp) / "control.yaml").exists())
            sampled=load_config(Path(tmp)/"sensors.yaml")["object_bodies"]
            self.assertIn("pelvis",sampled)
            self.assertTrue(all(a["body"] in sampled for a in data["actors"]))
            roster=load_config(Path(tmp)/"observation_roster.yaml")
            self.assertEqual(set(roster),{"scenario_id","seed","actors"})
            self.assertEqual(roster["seed"],42)
            self.assertTrue(all(set(actor)=={"id","role","body"} for actor in roster["actors"]))
            metadata=json.loads((Path(tmp)/"metadata.json").read_text())
            self.assertIsNone(metadata["ros_map_transform"])
            self.assertEqual(metadata["collision_mode"],"visual_only")
            copied=ET.parse(Path(tmp)/"robot.xml").getroot()
            original=ET.parse(ROBOT).getroot()
            original.find("compiler").set("meshdir",copied.find("compiler").get("meshdir"))
            self.assertEqual(ET.tostring(copied), ET.tostring(original))
        self.assertEqual(originals,{p:p.read_bytes() for p in originals})

    def test_deterministic_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            bundle(a); bundle(b)
            for p in a.rglob("*"):
                if p.is_file(): self.assertEqual(p.read_bytes(),(b/p.relative_to(a)).read_bytes())

    def test_continuous_obstacle_intersection(self):
        self.assertTrue(intersects([-2,0],[2,0],[[-0.1,0.1],[-1,1]],0))
        self.assertFalse(intersects([-2,2],[2,2],[[-0.1,0.1],[-1,1]],0))

    def test_invalid_config_bounds_transform_and_spawn(self):
        original=load_config(PACKAGE/"configs/mujoco_navigation.yaml")
        original["scenario"]=str(PACKAGE/"configs/navigation_actors.yaml")
        mutations=[{"scenario_to_mujoco":[0,0]}, {"scenario_to_mujoco":[0,5,0]},
                   {"scene_bounds":{"x":[1,-1],"y":[-2,2]}},
                   {"actor_radius_m":-1},{"g1_spawn_xy":[-1.4,0.8]},
                   {"scenario_to_mujoco":[0,0,float("nan")]}, {"typo":True}]
        with tempfile.TemporaryDirectory() as tmp:
            for i,mutation in enumerate(mutations):
                cfg=copy.deepcopy(original); cfg.update(mutation)
                path=Path(tmp)/f"bad{i}.yaml"; path.write_text(yaml.safe_dump(cfg))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    bundle(Path(tmp)/f"out{i}",path)

    def test_actor_counts_and_explicit_coordinate_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            for count in (0,1,2):
                scenario=load_config(PACKAGE/"configs/navigation_actors.yaml")
                scenario["actors"]=scenario["actors"][:count+1]
                scenario["pedestrian_count"]=count
                scenario_path=Path(tmp)/f"scenario{count}.yaml"
                scenario_path.write_text(yaml.safe_dump(scenario))
                cfg=load_config(PACKAGE/"configs/mujoco_navigation.yaml")
                cfg["scenario"]=str(scenario_path)
                cfg["scenario_to_mujoco"]=[0.05,0,0]
                cfg_path=Path(tmp)/f"adapter{count}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg))
                data=bundle(Path(tmp)/f"bundle{count}",cfg_path)
                self.assertEqual(len(data["actors"]),count+1)
                self.assertAlmostEqual(data["actors"][0]["initial_xy"][0],-1.35)

    def test_command_is_explicit_atomic_and_monotonic(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle(tmp)
            command(tmp,"start",1)
            self.assertEqual(load_config(Path(tmp)/"control.yaml"),{"sequence":1,"action":"start"})
            with self.assertRaises(ValueError): command(tmp,"reset",1)
            command(tmp,"pause",2)
            self.assertEqual(load_config(Path(tmp)/"control.yaml")["action"],"pause")
            with self.assertRaises(ValueError): command(tmp,"invalid",3)

    def test_launch_bundle_does_not_stage_or_remove_original_scenes(self):
        source=ast.parse((BRINGUP/"launch/sim.launch.py").read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="_stage_scene")
        namespace={"os":os}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"sim.launch.py","exec"),namespace)
        with tempfile.TemporaryDirectory() as tmp:
            bundle(tmp)
            with patch.dict(os.environ,{"GROVE_G1_PEDESTRIAN_BUNDLE":tmp}):
                self.assertEqual(namespace["_stage_scene"]("navigation",True,False),[])
                with self.assertRaises(RuntimeError): namespace["_stage_scene"]("navigation",True,True)
                with self.assertRaises(RuntimeError): namespace["_stage_scene"]("apartment",True,False)

    @unittest.skipUnless(shutil.which("g++"),"g++ unavailable")
    def test_actual_cpp_playback_with_mujoco_api_double(self):
        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/"bundle"; bundle(data)
            scenario=generate(load_config(PACKAGE/"configs/navigation_actors.yaml"))
            with Path(str(data/"playback.yaml")+".golden").open("w") as stream:
                for t in (0,0.25,1,2,4,7,10):
                    state=scenario.pose_at("client",t)
                    import math
                    stream.write(f"{t} {state.x:.17g} {state.y:.17g} {math.cos(state.yaw/2):.17g} {math.sin(state.yaw/2):.17g}\n")
            binary=Path(tmp)/"playback_test"
            build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror",
                "-I"+str(PACKAGE/"tests/mujoco_stub"),"-I"+str(SIM/"src"),
                str(SIM/"src/pedestrian_playback.cc"),str(PACKAGE/"tests/test_playback.cc"),
                "-lyaml-cpp","-o",str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(build.returncode,0,build.stderr)
            result=subprocess.run([str(binary),str(data/"playback.yaml")],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn("MuJoCo API double only",result.stdout)


if __name__ == "__main__": unittest.main()
