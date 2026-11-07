import json
import numpy as np
import pytest
from escort_rl.contract import load_contract,ROOT,PACKAGE
from escort_rl.controller_reference import ControllerReference,Schedule
from escort_rl.parity import tensor_report,trajectory_report
from escort_rl.measurements import response_metrics

@pytest.fixture
def controller():
    c=load_contract(); r=ControllerReference(c['observation_joints'],c['action_joints'])
    r.reset(np.zeros(29)); return r

def test_reference_identity_and_signature():
    c=load_contract()
    assert c['checkpoint_sha256']=='8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b'
    assert len(c['inputs'])==12 and len(c['outputs'])==10
    assert len(set(c['observation_joints']))==29
    assert sum(np.prod(x['shape']) for x in c['inputs'][5:])==269
    assert sum(np.prod(x['shape']) for x in c['outputs'])==311
    assert len(c['assets']['actuators'])==29

def test_command_limits_timeout_and_reset(controller):
    controller.command([4,-4,3],0)
    np.testing.assert_allclose(controller.command_at(.5),[.6,-.6,.9])
    assert not controller.command_at(.500001).any()
    with pytest.raises(ValueError): controller.command([np.nan,0,0],1)
    controller.reset(np.ones(29))
    assert controller.command_stamp is None and controller.tick==0 and controller.raw is None
    assert not controller.command_at(0).any()

def test_blend_inference_and_emergency(controller):
    calls=[]
    def infer():
        calls.append(1);return np.ones(14),np.full(14,20.),np.full(14,.2)
    for _ in range(4): controller.update(np.zeros(29),infer)
    assert len(calls)==1
    np.testing.assert_allclose(controller.target[controller.action_indices],.4)
    np.testing.assert_allclose(controller.kp[controller.action_indices],20)
    previous=controller.target.copy()
    speed=np.zeros(29);speed[controller.action_indices[0]]=36
    controller.update(speed,infer)
    assert controller.emergency
    np.testing.assert_array_equal(controller.target,previous)
    np.testing.assert_allclose(controller.kp[controller.action_indices],100)
    controller.update(np.zeros(29),infer)
    assert controller.emergency

def test_joint_holds_and_pd(controller):
    controller.target[:]=.1
    expected=controller.kp*float(np.float32(.1))-controller.kd*.2
    np.testing.assert_allclose(controller.torque(np.zeros(29),np.full(29,.2)),expected)
    assert controller.kp[controller.names.index('waist_yaw_joint')]==10
    assert controller.kp[controller.names.index('left_wrist_yaw_joint')]==300

def test_schedule_counts_jitter_and_repeatability():
    s=Schedule(.002); results=[s.events() for _ in range(101)]
    controllers=[(i*.002,t) for i,events in enumerate(results) for k,t in events if k=='controller']
    assert len(controllers)==40
    assert max(now-due for now,due in controllers)<=.001000001
    assert [x[0] for x in results[10]]==['bridge','controller','bridge']
    s.reset(); assert results==[s.events() for _ in range(101)]

def test_parity_reports_all_outputs_and_rejects_bad_shape():
    a=np.zeros((10,311));b=a.copy();b[-1,-1]=.01
    report=tensor_report(a,b)
    assert len(report)==10 and not report['actions_history_out']['pass']
    with pytest.raises(ValueError): tensor_report(a,b[:1])
    b[:]=np.nan
    assert not any(v['pass'] for v in tensor_report(a,b).values())

def test_command_suite_coverage():
    suite=json.loads((PACKAGE/'configs/command_suite.json').read_text())
    assert all([x,0,0] in suite['commands'] for x in [0,.1,.2,.35,.5,.6])
    assert all([0,0,x] in suite['commands'] for x in [.1,.3,.6,.9])

@pytest.mark.parametrize('module',['mujoco','onnxruntime'])
def test_runtime_dependencies(module):
    pytest.importorskip(module)

def test_feedback_mapping_with_distinct_tensors():
    # Test Python buffer wiring only; this fake session is never dynamics evidence.
    from escort_rl.frozen_gait import FrozenGait
    g=FrozenGait.__new__(FrozenGait);g.contract=load_contract()
    class Session:
        def run(self,names,feed):
            self.feed={k:v.copy() for k,v in feed.items()}
            return [np.full(x['shape'],i,np.float32) for i,x in enumerate(g.contract['outputs'])]
    g.session=Session();g.reset()
    args=([1,0,0,0],[0,0,0],[0,0,0],np.zeros(29),np.zeros(29))
    g.run(*args);g.run(*args)
    for i,spec in enumerate(g.contract['inputs'][5:]):
        np.testing.assert_array_equal(g.session.feed[spec['name']],i+3)
    g.reset()
    assert all(not x.any() for x in g.state.values())

def test_stop_metrics_require_continuous_dwell():
    config={'latency_fraction':.5,'stop_threshold_m_s':.05,'stop_threshold_rad_s':.05,
            'stop_dwell_s':.5,'sample_period_s':.02,'stop_s':2}
    rows=[]
    for i in range(300):
        t=i*.02
        speed=.2 if t<3 else 0
        rows.append(dict(time=t,body_velocity=[speed,0,0],position=[min(t,3)*.2,0,1],fallen=False,emergency=False))
    m=response_metrics(rows,[.2,0,0],1,2,config)
    assert m['stopping_time_s']==1
    assert m['stopping_distance_m']==pytest.approx(.2)

def test_trajectory_controller_discrepancy_and_alignment():
    row=dict(velocity_convention='pelvis_origin_in_pelvis_axes_v2',time=10.,joint_position=[0.]*29,body_velocity=[0.,0.,0.],position=[0.,0.,.79],
             quaternion=[1.,0.,0.,0.],fallen=False,emergency=False,target=[0.]*29,kp=[100.]*29,kd=[2.]*29)
    assert trajectory_report([row],[row])['provisional_pass']
    changed=dict(row,kp=[101.]*29)
    report=trajectory_report([row],[changed])
    assert not report['provisional_pass'] and report['controller']['kp']['max_abs']==1
    with pytest.raises(ValueError): trajectory_report([row],[dict(row,time=9.)])

def test_recorded_observation_export_preserves_consumed_float32_bits():
    import io
    from escort_rl.parity import observation_fixture_row
    values=np.random.default_rng(598).normal(size=68)
    row=observation_fixture_row(True,values[:4],values[4:7],values[7:10],values[10:39],values[39:])
    buffer=io.StringIO();np.savetxt(buffer,row[None,:],fmt='%.9g');buffer.seek(0)
    loaded=np.loadtxt(buffer).astype(np.float32)
    np.testing.assert_array_equal(loaded,row)
    np.testing.assert_array_equal(loaded[1:],values.astype(np.float32))
