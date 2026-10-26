"""ROS adapter only: no simulator/controller lifecycle or motion commands."""

from collections import deque
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from g1_msgs.msg import ActorState, ActorStateArray, SimulatorActorSnapshot
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .core import ObservationBridge, verified_alignment
from .dataset import DatasetWriter, code_version, replay_observations


def pose_dict(pose):
    return {"position":[pose.position.x,pose.position.y,pose.position.z],
            "quaternion":[pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]}


def transform_dict(transform):
    return {"position":[transform.translation.x,transform.translation.y,transform.translation.z],
            "quaternion":[transform.rotation.x,transform.rotation.y,transform.rotation.z,transform.rotation.w]}


def set_pose(output,value):
    output.position.x,output.position.y,output.position.z=map(float,value["position"])
    output.orientation.x,output.orientation.y,output.orientation.z,output.orientation.w=map(float,value["quaternion"])


def snapshot_dict(message):
    names=list(message.body_names)
    if len(names)!=len(message.body_poses) or len(names)!=len(set(names)):
        raise ValueError("Duplicate body IDs in raw snapshot")
    return {"stamp_ns":message.header.stamp.sec*10**9+message.header.stamp.nanosec,
            "frame_id":message.header.frame_id,"epoch":message.epoch,
            "simulation_time_s":message.simulation_time_s,
            "sensor_frame_id":message.sensor_frame_id,"sensor_pose":pose_dict(message.sensor_pose),
            "pelvis_valid":message.pelvis_valid,"pelvis_pose":pose_dict(message.pelvis_pose),
            "objects":{name:pose_dict(pose) for name,pose in zip(names,message.body_poses)}}


class ActorBridgeNode(Node):
    def __init__(self):
        super().__init__("escort_actor_bridge")
        defaults={"roster_file":"","dataset_dir":"","evaluation_config_file":"",
                  "allow_simulation_proxy":False,"target_frame":"odom",
                  "raw_topic":"/g1_sensor_relay/actor_ground_truth",
                  "alignment_position_tolerance_m":0.15,"alignment_angle_tolerance_rad":0.2,
                  "max_gap_s":0.5,"max_speed_mps":3.0,"stale_after_s":0.7,
                  "tf_wait_s":0.3,"replay_file":"","replay_period_s":0.1}
        for key,value in defaults.items(): self.declare_parameter(key,value)
        self.writer=None; self.pending=deque(); self.replay=None
        if self.get_parameter("use_sim_time").value:
            raise ValueError("This source uses relay wall timestamps; do not enable use_sim_time")
        if self.p("replay_file"):
            if self.p("replay_period_s")<=0: raise ValueError("Replay period must be positive")
            self.replay=iter(replay_observations(self.p("replay_file")))
            prefix="/escort/replay"
        else:
            if not self.p("allow_simulation_proxy"):
                raise ValueError("Explicit allow_simulation_proxy:=true required; this is NOT perception")
            with Path(self.p("roster_file")).open() as stream: roster=yaml.safe_load(stream)
            self.core=ObservationBridge(roster["actors"],self.p("target_frame"),
                self.p("max_gap_s"),self.p("max_speed_mps"),stale_after_s=self.p("stale_after_s"))
            self.core.last_arrival=self.get_clock().now().nanoseconds
            if self.p("tf_wait_s")<=0: raise ValueError("TF wait must be positive")
            # Only the logger archives this evaluator config. It is never passed to core/outputs.
            evaluation=None
            if self.p("evaluation_config_file"):
                with Path(self.p("evaluation_config_file")).open() as stream: evaluation=yaml.safe_load(stream)
            metadata={"scenario_id":roster["scenario_id"],"seed":roster["seed"],
                      "configuration":{"roster":roster,"bridge_parameters":{
                          key:self.p(key) for key in defaults if key not in ("dataset_dir","evaluation_config_file")}},
                      "timestamp_source":"relay capture mapped ROS wall; derivatives per simulation second",
                      "code_version":code_version()}
            if not self.p("dataset_dir"): raise ValueError("New dataset_dir required")
            self.writer=DatasetWriter(self.p("dataset_dir"),metadata,evaluation)
            self.tf=Buffer(); self.listener=TransformListener(self.tf,self)
            self.subscription=self.create_subscription(SimulatorActorSnapshot,self.p("raw_topic"),self.receive,qos_profile_sensor_data)
            prefix="/escort"
        self.tracks=self.create_publisher(ActorStateArray,prefix+"/tracks",10)
        self.client=self.create_publisher(PoseStamped,prefix+"/client_pose",10)
        self.timer=self.create_timer(self.p("replay_period_s") if self.replay is not None else 0.02,self.tick)

    def p(self,key): return self.get_parameter(key).value

    def receive(self,message):
        try:
            snapshot=snapshot_dict(message)
            arrival=self.get_clock().now().nanoseconds
            if len(self.pending)>=20:
                raw,received=self.pending.popleft()
                self.process(raw,received,None,"TF queue overflow")
            self.pending.append((snapshot,arrival))
        except (ValueError,TypeError,KeyError) as error:
            self.get_logger().error(f"Invalid raw snapshot: {error}")

    def tick(self):
        if self.replay is not None:
            try: self.publish(next(self.replay))
            except StopIteration: self.timer.cancel()
            return
        now=self.get_clock().now().nanoseconds
        while self.pending:
            raw,received=self.pending[0]
            alignment=None; error=""; diagnostics=None
            try:
                if not raw["pelvis_valid"]: raise ValueError("Independent pelvis anchor missing")
                stamp=Time(nanoseconds=raw["stamp_ns"],clock_type=self.get_clock().clock_type)
                sensor=self.tf.lookup_transform(self.core.frame,raw["sensor_frame_id"],stamp)
                pelvis=self.tf.lookup_transform(self.core.frame,"pelvis",stamp)
                alignment,diagnostics=verified_alignment(raw["sensor_pose"],transform_dict(sensor.transform),
                    raw["pelvis_pose"],transform_dict(pelvis.transform),
                    self.p("alignment_position_tolerance_m"),self.p("alignment_angle_tolerance_rad"))
            except TransformException as exception:
                if now-received<self.p("tf_wait_s")*1e9: break
                error=f"Capture-time TF unavailable: {exception}"
            except ValueError as exception:
                error=str(exception)
            self.pending.popleft()
            self.process(raw,received,alignment,error,diagnostics)
        expired=self.core.expire(now)
        if expired:
            self.writer.record(expired); self.publish(expired)

    def process(self,raw,received,alignment,error="",diagnostics=None):
        try:
            observation=self.core.process(raw,alignment,error,received)
            self.writer.record(observation,raw,{"verified":alignment is not None,"error":error,
                "target_T_world":alignment,"residual":diagnostics})
            self.publish(observation)
            if error: self.get_logger().warning(error)
        except (ValueError,TypeError,KeyError) as exception:
            self.get_logger().error(f"Rejected snapshot: {exception}")

    def publish(self,value):
        message=ActorStateArray()
        message.header.stamp.sec=value["stamp_ns"]//10**9
        message.header.stamp.nanosec=value["stamp_ns"]%10**9
        message.header.frame_id=value["frame_id"]
        message.epoch=value["epoch"]; message.source_time_s=value["source_time_s"]
        message.timestamp_source=value["timestamp_source"]
        for item in value["actors"]:
            actor=ActorState()
            for key in ("actor_id","role","heading","valid","velocity_valid","observation_source","invalid_reason"):
                setattr(actor,key,item[key])
            set_pose(actor.pose,item["pose"])
            actor.velocity.linear.x,actor.velocity.linear.y,actor.velocity.linear.z=item["velocity"]
            message.actors.append(actor)
            if actor.role=="client" and actor.valid:
                client=PoseStamped(); client.header=message.header; client.pose=actor.pose
                self.client.publish(client)
        self.tracks.publish(message)

    def destroy_node(self):
        if self.writer: self.writer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node=None
    try:
        node=ActorBridgeNode(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node: node.destroy_node()
        rclpy.try_shutdown()
