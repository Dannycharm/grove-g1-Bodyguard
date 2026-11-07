"""Offline SequentialReader only. No Node, publisher, service, subprocess or ROS init."""
import argparse
import json
from pathlib import Path
from escort_rl.contract import sha256
import escort_rl.reference_recording as normalization_module
from escort_rl.reference_recording import TOPICS,normalize,summarize

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bag',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();bag=Path(a.bag).resolve();out=Path(a.output).resolve()
    if not (bag/'metadata.yaml').is_file():p.error('Expected an existing rosbag directory')
    # Dependencies come from the existing sourced ROS overlay, never installed by this tool.
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from rosidl_runtime_py.convert import message_to_ordereddict
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id=''),rosbag2_py.ConverterOptions('',''))
    inventory={t.name:t.type for t in reader.get_all_topics_and_types()}
    classes={}
    for name,kind in inventory.items():
        if name in TOPICS:
            if kind!=TOPICS[name]:raise ValueError(f'Unexpected topic type: {name}: {kind}')
            classes[name]=get_message(kind)
    before={str(f.relative_to(bag)):sha256(f) for f in sorted(bag.iterdir()) if f.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    events=[]
    with open(out/'reference_events.jsonl','x') as f:
        while reader.has_next():
            topic,data,stamp=reader.read_next()
            if topic not in classes:continue
            msg=message_to_ordereddict(deserialize_message(data,classes[topic]))
            event=normalize(topic,msg,stamp);events.append(event)
            f.write(json.dumps(event,allow_nan=False)+'\n')
    summary,estimates=summarize(events,inventory)
    after={str(f.relative_to(bag)):sha256(f) for f in sorted(bag.iterdir()) if f.is_file()}
    if before!=after:raise RuntimeError('Input bag changed while exporting')
    with open(out/'coverage.json','x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    with open(out/'pelvis_backward_estimates.jsonl','x') as f:
        for e in estimates:f.write(json.dumps(e,allow_nan=False)+'\n')
    with open(out/'provenance.json','x') as f:json.dump(dict(source_bag=str(bag),source_sha256=before,
        input_unchanged=True,evaluator_only=True,can_be_policy_input=False,
        exporter_sha256=sha256(__file__),normalizer_sha256=sha256(normalization_module.__file__),note='No alignment or loaded/runtime identity is inferred'),f,indent=2)
    print(json.dumps(summary,indent=2))
