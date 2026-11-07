import ast
from pathlib import Path
import numpy as np
import pytest
from escort_rl.contract import load_contract,PACKAGE
from escort_rl.reference_recording import normalize,summarize
from escort_rl.parity import trajectory_report


def header(stamp=0):return dict(stamp=dict(sec=stamp,nanosec=0),frame_id='mujoco_world')
def pose(x=0):return dict(position=dict(x=x,y=0.,z=.73),orientation=dict(w=1.,x=0.,y=0.,z=0.))
def snapshot(time,x=0,epoch=0):
    return dict(header=header(round(time)),simulation_time_s=time,epoch=epoch,pelvis_valid=True,
                pelvis_pose=pose(x),body_names=['privileged_actor'],body_poses=[pose(99)])


def test_joint_name_mapping_missing_and_nonfinite():
    names=load_contract()['observation_joints'][::-1]
    msg=dict(header=header(),name=names,position=list(range(29)),velocity=[0.]*29,effort=[])
    e=normalize('/joint_states',msg,123)
    assert e['joint_position']==list(range(29))[::-1] and e['all_body_joints_valid']
    assert e['joint_effort']==[None]*29 and e['header_stamp_ns']==0
    msg['velocity'][0]=float('nan')
    assert not normalize('/joint_states',msg,1)['all_body_joints_valid']
    msg['name']=[names[0]]*29
    with pytest.raises(ValueError):normalize('/joint_states',msg,1)


def test_truth_is_evaluator_only_and_no_missing_command_fabrication():
    a=normalize('/g1_sensor_relay/actor_ground_truth',snapshot(1),1)
    b=normalize('/g1_sensor_relay/actor_ground_truth',snapshot(1.1,.02),2)
    assert a['evaluator_only'] and 'body_poses' not in a and 'body_names' not in a
    report,estimates=summarize([a,b],{})
    assert not report['command_stream_observed'] and not report['missing_commands_are_zero']
    assert not report['full_dynamics_parity_ready']
    np.testing.assert_allclose(estimates[0]['linear_velocity_pelvis_estimate'],[.2,0,0],atol=1e-12)
    b['epoch']=1
    assert not summarize([a,b],{})[1]
    b['epoch']=0;b['simulation_time_s']=.5
    assert not summarize([a,b],{})[1]


def test_received_command_is_not_executed_command():
    msg=dict(linear=dict(x=.8,y=0,z=0),angular=dict(x=0,y=0,z=1.2))
    row=normalize('/cmd_vel',msg,99)
    assert row['requested']==[.8,0,1.2] and row['applied_command'] is None
    report,_=summarize([row],{})
    assert report['command_stream_observed'] and report['nonzero_command_messages']==1
    assert not report['full_dynamics_parity_ready']


def test_exporter_has_no_ros_graph_or_command_api():
    tree=ast.parse((PACKAGE/'tools/export_reference_bag.py').read_text())
    forbidden={'create_publisher','publish','create_client','call_async','init','Node',
               'create_subscription','step_physics','reset','Popen','run','system'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
            assert name not in forbidden


def test_trajectory_rejects_partial_or_mixed_conventions():
    row=dict(time=0.,velocity_convention='pelvis_origin_in_pelvis_axes_v2')
    with pytest.raises(ValueError,match='Missing/invalid'):trajectory_report([row],[row])
    for key,width in [('joint_position',29),('body_velocity',3),('position',3),('quaternion',4),('target',29),('kp',29),('kd',29)]:
        row[key]=[0.]*width
    with pytest.raises(ValueError,match='convention'):trajectory_report([row],[dict(row,velocity_convention='legacy')])
