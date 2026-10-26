"""ROS-independent observation logic. Inputs are CURRENT snapshots, never trajectories."""

import math


IDENTITY = {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]}
SOURCE = "simulation_pose_proxy"


def finite(values):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)


def pose(value):
    p, q = list(value["position"]), list(value["quaternion"])
    if len(p) != 3 or len(q) != 4 or not finite(p + q):
        raise ValueError("Invalid pose")
    norm = math.sqrt(sum(v*v for v in q))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid quaternion")
    return {"position": p, "quaternion": [v/norm for v in q]}


def multiply(a, b):
    x,y,z,w = a
    X,Y,Z,W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def rotate(q, p):
    return multiply(multiply(q, [*p, 0]), [-q[0],-q[1],-q[2],q[3]])[:3]


def compose(a, b):
    a,b=pose(a),pose(b)
    rotated=rotate(a["quaternion"],b["position"])
    return pose({"position": [x+y for x,y in zip(a["position"],rotated)],
                 "quaternion": multiply(a["quaternion"],b["quaternion"])})


def inverse(a):
    a=pose(a)
    q=[-a["quaternion"][0],-a["quaternion"][1],-a["quaternion"][2],a["quaternion"][3]]
    return {"position": rotate(q,[-v for v in a["position"]]),"quaternion":q}


def yaw(p):
    x,y,z,w=pose(p)["quaternion"]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def verified_alignment(world_sensor, target_sensor, world_pelvis, target_pelvis,
                       position_tolerance=0.15, angle_tolerance=0.2):
    """target_T_world = target_T_sensor * inverse(world_T_sensor).

    Independent same-capture pelvis anchor checks extrinsics/frame consistency.
    This is NOT proof of localization accuracy or a new TF broadcaster.
    """
    if not finite([position_tolerance,angle_tolerance]) or min(position_tolerance,angle_tolerance)<=0:
        raise ValueError("Positive alignment tolerances required")
    transform=compose(target_sensor,inverse(world_sensor))
    predicted=compose(transform,world_pelvis)
    actual=pose(target_pelvis)
    distance=math.dist(predicted["position"],actual["position"])
    dot=abs(sum(a*b for a,b in zip(predicted["quaternion"],actual["quaternion"])))
    angle=2*math.acos(min(1.0,dot))
    if distance>position_tolerance or angle>angle_tolerance:
        raise ValueError(f"Anchor mismatch: position={distance:.6f}m rotation={angle:.6f}rad")
    return transform,{"position_error_m":distance,"rotation_error_rad":angle}


class ObservationBridge:
    def __init__(self, roster, target_frame="odom", max_gap_s=0.5, max_speed_mps=3.0,
                 jump_margin_m=0.1, stale_after_s=0.7):
        if target_frame not in ("map","odom"):
            raise ValueError("Use an existing map or odom target frame")
        if not finite([max_gap_s,max_speed_mps,jump_margin_m,stale_after_s]) or min(max_gap_s,max_speed_mps,stale_after_s)<=0 or jump_margin_m<0:
            raise ValueError("Invalid bridge limits")
        self.roster=sorted(roster,key=lambda a:a["id"])
        ids=[a["id"] for a in self.roster]; bodies=[a["body"] for a in self.roster]
        if not 1<=len(ids)<=3 or len(set(ids))!=len(ids) or len(set(bodies))!=len(bodies):
            raise ValueError("Invalid/duplicate actor roster")
        if sum(a["role"]=="client" for a in self.roster)!=1 or any(a["role"] not in ("client","pedestrian") for a in self.roster):
            raise ValueError("Exactly one client and up to two pedestrians required")
        if any(not isinstance(x,str) or not x for x in ids+bodies):
            raise ValueError("Nonempty actor IDs/body names required")
        self.frame=target_frame
        self.max_gap=max_gap_s; self.max_speed=max_speed_mps; self.jump_margin=jump_margin_m
        self.stale_ns=int(stale_after_s*1e9)
        self.epoch=0; self.source_epoch=None; self.last_time=None; self.last_stamp=None
        self.history={}; self.last_arrival=None; self.expired=False

    def _invalid(self, actor, reason):
        return {"actor_id":actor["id"],"role":actor["role"],"pose":IDENTITY,
                "velocity":[0.0,0.0,0.0],"heading":0.0,"valid":False,
                "velocity_valid":False,"observation_source":SOURCE,"invalid_reason":reason}

    def process(self, snapshot, transform=None, alignment_error="", arrival_ns=None):
        stamp=snapshot["stamp_ns"]; time=snapshot["simulation_time_s"]; source_epoch=snapshot["epoch"]
        if type(stamp) is not int or stamp<0 or type(source_epoch) is not int or source_epoch<0 or not finite([time]) or time<0:
            raise ValueError("Invalid source timestamp/epoch")
        if snapshot.get("frame_id")!="mujoco_world":
            transform=None; alignment_error="Unexpected raw coordinate frame"
        reset=(self.source_epoch is not None and source_epoch!=self.source_epoch) or (
            self.last_time is not None and (time<self.last_time or stamp<self.last_stamp))
        if reset:
            self.epoch+=1; self.history.clear()
        duplicate=not reset and self.last_time is not None and (time==self.last_time or stamp==self.last_stamp)
        self.source_epoch=source_epoch
        if duplicate:
            transform=None; alignment_error="Duplicate capture time"
        transformed={}
        if transform is not None:
            for actor in self.roster:
                raw=snapshot["objects"].get(actor["body"])
                if raw is not None:
                    try: transformed[actor["id"]]=compose(transform,raw)
                    except (ValueError,KeyError,TypeError): pass
            # A teleport/map correction splits the derivative episode for ALL actors.
            for actor_id,current in transformed.items():
                previous=self.history.get(actor_id)
                if previous and 0<time-previous[0]<=self.max_gap:
                    if math.dist(current["position"],previous[1]["position"])>self.max_speed*(time-previous[0])+self.jump_margin:
                        self.epoch+=1; self.history.clear(); break
        result=[]
        for actor in self.roster:
            actor_id=actor["id"]; current=transformed.get(actor_id)
            if current is None:
                self.history.pop(actor_id,None)
                result.append(self._invalid(actor,alignment_error or "Actor missing or invalid pose"))
                continue
            velocity=[0.0,0.0,0.0]; velocity_valid=False
            previous=self.history.get(actor_id)
            if previous and 0<time-previous[0]<=self.max_gap:
                velocity=[(a-b)/(time-previous[0]) for a,b in zip(current["position"],previous[1]["position"])]
                velocity_valid=True
            self.history[actor_id]=(time,current)
            result.append({"actor_id":actor_id,"role":actor["role"],"pose":current,
                           "velocity":velocity,"heading":yaw(current),"valid":True,
                           "velocity_valid":velocity_valid,"observation_source":SOURCE,
                           "invalid_reason":"" if velocity_valid else "No contiguous derivative history"})
        if not duplicate:
            self.last_time=time; self.last_stamp=stamp
        self.last_arrival=stamp if arrival_ns is None else arrival_ns
        self.expired=False
        return {"stamp_ns":stamp,"frame_id":self.frame,"epoch":self.epoch,
                "source_time_s":time,"timestamp_source":"relay_capture_mapped_ros_wall",
                "actors":result}

    def expire(self, now_ns):
        if self.last_arrival is None or self.expired or now_ns-self.last_arrival<=self.stale_ns:
            return None
        self.expired=True; self.history.clear()
        return {"stamp_ns":now_ns,"frame_id":self.frame,"epoch":self.epoch,
                "source_time_s":self.last_time if self.last_time is not None else 0.0,"timestamp_source":"ros_wall_timeout",
                "actors":[self._invalid(a,"Source disappeared/stale") for a in self.roster]}
