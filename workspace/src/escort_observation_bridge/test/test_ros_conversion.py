"""Generated-message conversion tests only. Never initialize rclpy or DDS."""

import unittest
from types import SimpleNamespace

try:
    from geometry_msgs.msg import Pose
    from g1_msgs.msg import SimulatorActorSnapshot
    from escort_observation_bridge.node import ActorBridgeNode, snapshot_dict
    from escort_observation_bridge.core import IDENTITY, ObservationBridge
    REASON=""
except ImportError as error:
    REASON=str(error)


@unittest.skipIf(bool(REASON),"Generated ROS messages/API unavailable: "+REASON)
class MessageConversionTests(unittest.TestCase):
    def source(self):
        source=SimulatorActorSnapshot()
        source.header.frame_id="mujoco_world"
        source.header.stamp.sec=7
        source.header.stamp.nanosec=123
        source.simulation_time_s=1.0
        source.body_names=["escort_client"]
        p=Pose(); p.position.x=2.; p.orientation.w=1.
        source.body_poses=[p]
        return source

    def test_generated_messages_and_client_pose_share_capture_stamp(self):
        source=self.source()
        bridge=ObservationBridge([{"id":"client","body":"escort_client","role":"client"}])
        observation=bridge.process(snapshot_dict(source),IDENTITY)
        tracks=[]; client=[]
        node=SimpleNamespace(tracks=SimpleNamespace(publish=tracks.append),client=SimpleNamespace(publish=client.append))
        ActorBridgeNode.publish(node,observation)
        self.assertEqual(tracks[0].header.stamp,source.header.stamp)
        self.assertEqual(client[0].header.stamp,source.header.stamp)
        self.assertEqual(tracks[0].actors[0].pose.position.x,2)
        self.assertEqual(tracks[0].actors[0].role,"client")
        self.assertEqual(tracks[0].header.frame_id,"odom")

    def test_missing_client_has_invalid_track_and_no_pose(self):
        source=self.source(); source.body_names=[]; source.body_poses=[]
        bridge=ObservationBridge([{"id":"client","body":"escort_client","role":"client"}])
        tracks=[]; client=[]
        node=SimpleNamespace(tracks=SimpleNamespace(publish=tracks.append),client=SimpleNamespace(publish=client.append))
        ActorBridgeNode.publish(node,bridge.process(snapshot_dict(source),IDENTITY))
        self.assertFalse(tracks[0].actors[0].valid)
        self.assertEqual(client,[])

    def test_mismatched_or_duplicate_raw_identity_arrays_rejected(self):
        source=self.source(); source.body_names.append("escort_client")
        with self.assertRaises(ValueError): snapshot_dict(source)


if __name__=="__main__": unittest.main()
