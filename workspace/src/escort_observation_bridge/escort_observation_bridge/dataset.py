"""Append-only deterministic JSONL with physically separate observation/truth files."""

import hashlib
import json
from pathlib import Path

from .core import SOURCE, finite, pose


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n"


def code_version():
    """Fingerprint actual Python sources, including uncommitted code."""
    digest=hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return "escort_observation_bridge/0.1.0/sha256:"+digest.hexdigest()


class DatasetWriter:
    def __init__(self,directory,metadata,evaluation_configuration=None):
        required={"scenario_id","seed","configuration","timestamp_source","code_version"}
        if required-set(metadata):
            raise ValueError("Missing reproducibility metadata")
        root=Path(directory)
        if root.exists() and any(root.iterdir()):
            raise ValueError("Dataset must be new or empty; existing runs are never overwritten")
        (root/"observations").mkdir(parents=True,exist_ok=True)
        (root/"evaluation").mkdir()
        manifest=dict(metadata,schema_version=1,provenance="simulation_pose_proxy_not_perception",
                      policy_input_path="observations/tracks.jsonl",
                      evaluator_only_path="evaluation/current_ground_truth.jsonl")
        (root/"manifest.json").write_text(canonical(manifest))
        if evaluation_configuration is not None:
            (root/"evaluation/configuration.json").write_text(canonical(evaluation_configuration))
        self.observations=(root/"observations/tracks.jsonl").open("x")
        self.truth=(root/"evaluation/current_ground_truth.jsonl").open("x")

    def record(self,observation,truth=None,alignment=None):
        validate_observation(observation)
        record=canonical(observation)
        evaluation=canonical({"snapshot":truth,"alignment":alignment}) if truth is not None else None
        self.observations.write(record); self.observations.flush()
        if evaluation is not None:
            self.truth.write(evaluation); self.truth.flush()

    def close(self):
        self.observations.close(); self.truth.close()


def replay_observations(path):
    path=Path(path)
    if path.name!="tracks.jsonl" or path.parent.name!="observations":
        raise ValueError("Replay accepts only observations/tracks.jsonl, never evaluator truth")
    with path.open() as stream:
        for line in stream:
            value=json.loads(line)
            validate_observation(value)
            yield value


def validate_observation(value):
    if set(value)!={"stamp_ns","frame_id","epoch","source_time_s","timestamp_source","actors"}:
        raise ValueError("Unexpected policy record fields (future/evaluator data forbidden)")
    if type(value["stamp_ns"]) is not int or value["stamp_ns"]<0 or type(value["epoch"]) is not int or value["epoch"]<0:
        raise ValueError("Invalid observation time/epoch")
    if value["frame_id"] not in ("map","odom") or not finite([value["source_time_s"]]):
        raise ValueError("Invalid policy frame/time")
    if not isinstance(value["actors"],list) or not 1<=len(value["actors"])<=3:
        raise ValueError("Invalid actor list")
    ids=set()
    for actor in value["actors"]:
        if set(actor)!={"actor_id","role","pose","velocity","heading","valid","velocity_valid",
                         "observation_source","invalid_reason"}:
            raise ValueError("Unexpected policy actor fields")
        if actor["actor_id"] in ids or actor["role"] not in ("client","pedestrian"):
            raise ValueError("Invalid actor identity")
        ids.add(actor["actor_id"])
        if set(actor["pose"])!={"position","quaternion"}:
            raise ValueError("Unexpected policy pose fields")
        pose(actor["pose"])
        if not isinstance(actor["velocity"],list) or len(actor["velocity"])!=3 or not finite(actor["velocity"]+[actor["heading"]]):
            raise ValueError("Invalid observed velocity/heading")
        if type(actor["valid"]) is not bool or type(actor["velocity_valid"]) is not bool or actor["observation_source"]!=SOURCE:
            raise ValueError("Invalid validity/provenance")
