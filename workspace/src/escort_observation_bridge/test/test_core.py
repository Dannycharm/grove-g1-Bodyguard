"""Offline unit tests; neither ROS nor MuJoCo imported."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from escort_observation_bridge.core import IDENTITY, ObservationBridge, compose, inverse, verified_alignment, yaw
from escort_observation_bridge.dataset import DatasetWriter, canonical, replay_observations

ROSTER=[{"id":"client","role":"client","body":"escort_client"},
        {"id":"pedestrian_1","role":"pedestrian","body":"escort_pedestrian_1"}]


def p(x=0,y=0,z=0,heading=0):
    return {"position":[x,y,z],"quaternion":[0,0,math.sin(heading/2),math.cos(heading/2)]}


def raw(t=0,x=1,epoch=0):
    return {"stamp_ns":1_000_000_000+round(t*1e9),"epoch":epoch,"simulation_time_s":t,
            "frame_id":"mujoco_world","objects":{"escort_client":p(x),"escort_pedestrian_1":p(0,2)}}


class CoreTests(unittest.TestCase):
    def test_nonidentity_alignment_and_independent_anchor(self):
        target_world=p(4,-2,0.3,math.pi/2)
        world_sensor=p(1,0,1.2,0.4); world_pelvis=p(0.4,0.2,0.7,0.1)
        calculated,residual=verified_alignment(world_sensor,compose(target_world,world_sensor),
            world_pelvis,compose(target_world,world_pelvis))
        projected=compose(calculated,p(2,1))
        for a,b in zip(projected["position"],[3,0,0.3]): self.assertAlmostEqual(a,b)
        self.assertAlmostEqual(yaw(projected),math.pi/2)
        self.assertAlmostEqual(residual["position_error_m"],0)
        neutral=compose(calculated,inverse(calculated))
        for v in neutral["position"]: self.assertAlmostEqual(v,0)

    def test_anchor_mismatch_rejected(self):
        with self.assertRaises(ValueError): verified_alignment(p(),p(),p(),p(1))
        with self.assertRaises(ValueError): verified_alignment(p(),p(),p(),p(heading=0.8))

    def test_full_3d_rotation(self):
        angle=0.4
        transform={"position":[1,2,3],"quaternion":[math.sin(angle/2),0,0,math.cos(angle/2)]}
        restored=compose(inverse(transform),compose(transform,p(3,4,5)))
        for a,b in zip(restored["position"],[3,4,5]): self.assertAlmostEqual(a,b)

    def test_stable_ids_roles_and_exact_capture_timestamp(self):
        bridge=ObservationBridge(list(reversed(ROSTER)))
        a=bridge.process(raw(),p(3,2,heading=math.pi/2))
        b=bridge.process(raw(0.1),p(3,2,heading=math.pi/2))
        self.assertEqual([(x["actor_id"],x["role"]) for x in a["actors"]],[(x["actor_id"],x["role"]) for x in b["actors"]])
        self.assertEqual(a["stamp_ns"],1_000_000_000)
        self.assertEqual(a["frame_id"],"odom")
        self.assertAlmostEqual(a["actors"][0]["pose"]["position"][0],3)
        self.assertAlmostEqual(a["actors"][0]["pose"]["position"][1],3)
        self.assertEqual(a["actors"][0]["observation_source"],"simulation_pose_proxy")

    def test_velocity_estimate_uses_sim_time_not_arrival_or_ros_offset(self):
        bridge=ObservationBridge(ROSTER)
        first=bridge.process(raw(),IDENTITY,arrival_ns=5_000_000_000)
        sample=raw(0.1,1.2); sample["stamp_ns"]+=50_000_000
        second=bridge.process(sample,IDENTITY,arrival_ns=9_000_000_000)
        self.assertFalse(first["actors"][0]["velocity_valid"])
        self.assertTrue(second["actors"][0]["velocity_valid"])
        self.assertAlmostEqual(second["actors"][0]["velocity"][0],2)
        self.assertEqual(second["stamp_ns"],sample["stamp_ns"])

    def test_missing_actor_and_reappearance(self):
        bridge=ObservationBridge(ROSTER); bridge.process(raw(),IDENTITY)
        sample=raw(0.1); del sample["objects"]["escort_client"]
        missing=bridge.process(sample,IDENTITY)
        self.assertFalse(missing["actors"][0]["valid"])
        returned=bridge.process(raw(0.2),IDENTITY)
        self.assertFalse(returned["actors"][0]["velocity_valid"])
        self.assertTrue(returned["actors"][0]["valid"])

    def test_tf_failure_wrong_frame_and_invalid_pose_fail_closed(self):
        for mutation in ("tf","frame","pose"):
            bridge=ObservationBridge(ROSTER)
            sample=raw(); transform=IDENTITY
            if mutation=="tf": transform=None
            if mutation=="frame": sample["frame_id"]="fake_map"
            if mutation=="pose": sample["objects"]["escort_client"]["quaternion"]=[0,0,0,0]
            result=bridge.process(sample,transform,"TF unavailable" if transform is None else "")
            self.assertFalse(result["actors"][0]["valid"])

    def test_reset_epoch_source_time_and_ros_clock_regression(self):
        for change in ("source_epoch","simulation_time","ros_time"):
            bridge=ObservationBridge(ROSTER); bridge.process(raw(1),IDENTITY)
            sample=raw(1.1,1.01)
            if change=="source_epoch": sample["epoch"]=1
            if change=="simulation_time": sample["simulation_time_s"]=0
            if change=="ros_time": sample["stamp_ns"]=1
            result=bridge.process(sample,IDENTITY)
            self.assertEqual(result["epoch"],1)
            self.assertFalse(result["actors"][0]["velocity_valid"])

    def test_teleport_splits_all_derivative_histories(self):
        bridge=ObservationBridge(ROSTER)
        bridge.process(raw(),IDENTITY)
        result=bridge.process(raw(0.1,5),IDENTITY)
        self.assertEqual(result["epoch"],1)
        self.assertTrue(all(not a["velocity_valid"] for a in result["actors"]))

    def test_duplicate_capture_invalid_and_gap_derivative_invalid(self):
        bridge=ObservationBridge(ROSTER)
        bridge.process(raw(),IDENTITY)
        self.assertTrue(all(not a["valid"] for a in bridge.process(raw(),IDENTITY)["actors"]))
        self.assertFalse(bridge.process(raw(1),IDENTITY)["actors"][0]["velocity_valid"])

    def test_disappearance_timeout_once_and_recovery(self):
        bridge=ObservationBridge(ROSTER)
        bridge.process(raw(),IDENTITY,arrival_ns=1_000_000_000)
        self.assertIsNone(bridge.expire(1_500_000_000))
        stale=bridge.expire(2_000_000_000)
        self.assertTrue(all(not actor["valid"] for actor in stale["actors"]))
        self.assertIsNone(bridge.expire(3_000_000_000))
        recovered=bridge.process(raw(1),IDENTITY,arrival_ns=3_000_000_000)
        self.assertFalse(recovered["actors"][0]["velocity_valid"])

    def test_invalid_roster_and_timestamp(self):
        with self.assertRaises(ValueError): ObservationBridge(ROSTER,target_frame="another_map")
        with self.assertRaises(ValueError): ObservationBridge(ROSTER*2)
        with self.assertRaises(ValueError): ObservationBridge(ROSTER,max_gap_s=-1)
        for key,value in (("stamp_ns",-1),("simulation_time_s",float("nan")),("epoch",True)):
            sample=raw(); sample[key]=value
            with self.assertRaises(ValueError): ObservationBridge(ROSTER).process(sample,IDENTITY)


class DatasetTests(unittest.TestCase):
    def metadata(self):
        return {"scenario_id":"scenario42","seed":42,"configuration":{"roster":ROSTER},
                "timestamp_source":"test_capture","code_version":"test-code-sha"}

    def test_byte_repeatable_logs_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a","b"):
                bridge=ObservationBridge(ROSTER)
                writer=DatasetWriter(Path(tmp)/name,self.metadata())
                for t in (0,0.1,0.2):
                    source=raw(t,1+t)
                    writer.record(bridge.process(source,IDENTITY),source)
                writer.close()
            a=Path(tmp)/"a"; b=Path(tmp)/"b"
            for path in a.rglob("*"):
                if path.is_file(): self.assertEqual(path.read_bytes(),(b/path.relative_to(a)).read_bytes())
            records=list(replay_observations(a/"observations/tracks.jsonl"))
            self.assertEqual(len(records),3)
            self.assertEqual("".join(canonical(v) for v in records),(a/"observations/tracks.jsonl").read_text())

    def test_training_inputs_do_not_contain_future_truth_or_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=DatasetWriter(tmp,self.metadata(),{"future_waypoints":[[100,200]]})
            observation=ObservationBridge(ROSTER).process(raw(),IDENTITY)
            writer.record(observation,{"future_evaluation_only":[[100,200]]},{"world_transform":"evaluator_only"})
            writer.close()
            policy=(Path(tmp)/"observations/tracks.jsonl").read_text()
            self.assertNotIn("future",policy); self.assertNotIn("alignment",policy)
            self.assertNotIn("sensor_pose",policy)
            with self.assertRaises(ValueError): list(replay_observations(Path(tmp)/"evaluation/current_ground_truth.jsonl"))

    def test_schema_rejects_future_fields_nested_at_any_level(self):
        for level in ("top","actor","pose","velocity"):
            with tempfile.TemporaryDirectory() as tmp:
                writer=DatasetWriter(tmp,self.metadata())
                observation=ObservationBridge(ROSTER).process(raw(),IDENTITY)
                if level=="top": observation["future"]=[1]
                if level=="actor": observation["actors"][0]["future"]=[1]
                if level=="pose": observation["actors"][0]["pose"]=copy.deepcopy(observation["actors"][0]["pose"]); observation["actors"][0]["pose"]["future"]=[1]
                if level=="velocity": observation["actors"][0]["velocity"]={"future":[1]}
                with self.assertRaises(ValueError): writer.record(observation)
                writer.close()

    def test_overwrite_and_incomplete_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=DatasetWriter(tmp,self.metadata()); writer.close()
            with self.assertRaises(ValueError): DatasetWriter(tmp,self.metadata())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): DatasetWriter(tmp,{"seed":42})


if __name__=="__main__": unittest.main()
