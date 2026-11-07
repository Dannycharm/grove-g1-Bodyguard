"""Opt-in tracking/stop objective: pure evaluator cases and real-physics invariants."""
from dataclasses import replace
import numpy as np
from escort_rl.tracking_objective import TrackingStopEvaluator,desired_point,settings
from escort_rl.reward_variants import make_environment
from escort_rl.stage1_scenarios import load_config,sample_episode


def fixture(initial=-1.5):
    config=load_config();client=replace(sample_episode(1,'straight').client_pose_at(0),x=0.,y=0.,yaw=0.)
    row={'linear_velocity_world':[0.,0.,0.],'euler_yaw_rate_world':0.}
    evaluator=TrackingStopEvaluator(config,initial,4.,lambda xy:float(xy[0]),lambda:row,[initial,0.],client)
    return evaluator,client,row


def step(e,client,t,robot,**flags):
    values=dict(elapsed=t,dt=.2,robot_xy=[robot,0.],client=client,robot_progress=robot,
        client_progress=client.x,client_arrived=None,action=np.zeros(3))
    values.update(flags);return e.update(**values)


def test_correct_tracking_and_turning_reference():
    e,c,row=fixture()
    reward,_,_,info=step(e,replace(c,x=.04),.2,-1.46)
    assert info['reward_components']['formation']==0
    assert abs(info['reward_components']['velocity_tracking'])<1e-20
    assert reward>0
    e,c,row=fixture();changed=replace(c,yaw=.02);point=desired_point(changed,[-1.5,0])
    _,_,_,info=e.update(elapsed=.2,dt=.2,robot_xy=point,client=changed,robot_progress=point[0],client_progress=0.,client_arrived=None,action=np.zeros(3))
    assert np.linalg.norm(info['desired_interval_velocity_world'])>.14
    assert info['relative_velocity_error_m_s']==0


def test_stationary_correct_overshoot_and_recovery():
    e,c,row=fixture();assert step(e,c,.2,-1.5)[0]==0
    e,c,row=fixture();overshoot=0
    for i in range(1,16):overshoot+=step(e,c,i*.2,-1.5+.04*i)[0]
    assert overshoot<0
    e,c,row=fixture(-2.);recovery=0
    for i in range(1,51):recovery+=step(e,c,i*.2,-2+.01*i)[0]
    held,c,row=fixture(-2.);holding=sum(step(held,c,i*.2,-2)[0] for i in range(1,51))
    assert recovery>holding


def test_no_flyby_completion_and_one_second_sampled_hold():
    e,c,row=fixture();arrived=replace(c,x=4.)
    for i in range(6):
        r,t,tr,info=step(e,arrived,10+i*.2,2.5,client_arrived=10.)
        assert t==(i==5)
    assert info['reason']=='success' and info['stop_hold_s']>=1-1e-9
    e,c,row=fixture();row['linear_velocity_world']=[.2,0.,0.]
    for i in range(26):r,t,tr,info=step(e,replace(c,x=4),10+i*.2,2.5,client_arrived=10.)
    assert t and info['reason']=='recovery_timeout' and info['reward_components']['terminal_success']==0


def test_hold_resets_and_safety_priority():
    e,c,row=fixture();arrived=replace(c,x=4.)
    step(e,arrived,10.,2.5,client_arrived=10.);step(e,arrived,10.2,2.5,client_arrived=10.)
    row['euler_yaw_rate_world']=.2
    _,_,_,info=step(e,arrived,10.4,2.5,client_arrived=10.);assert info['stop_hold_s']==0
    row['euler_yaw_rate_world']=0.
    _,term,_,info=step(e,arrived,10.6,2.5,client_arrived=10.,emergency=True)
    assert term and info['reason']=='supervisor_failure' and info['reward_components']['terminal_failure']==-100


def test_bounded_gait_micromotion_is_not_forward_travel():
    e,c,row=fixture();row['linear_velocity_world']=[.057,0.,0.]
    for i in range(8):
        _,term,_,info=step(e,replace(c,x=4.),10+i*.2,2.5+(.002 if i%2 else -.002),client_arrived=10.)
        if term:break
    assert info['reason']=='success'
    assert info['stop_window_speed_m_s']<.05<info['stop_native_speed_m_s']<.15


def test_windowed_hold_rejects_travel_but_allows_small_correction_loops():
    e,c,row=fixture();row['linear_velocity_world']=[.08,0.,0.]
    for i in range(6):
        _,term,_,info=step(e,replace(c,x=4.),10+i*.2,2.5+(.008 if i%2 else -.008),client_arrived=10.)
    assert term and info['reason']=='success'
    assert info['stop_interval_speed_m_s']>.05 and info['stop_window_excursion_m']<.10
    e,c,row=fixture()
    for i in range(26):
        _,term,_,info=step(e,replace(c,x=4.),10+i*.2,2.5+.02*i,client_arrived=10.)
    assert term and info['reason']=='recovery_timeout'


def test_tracking_env_preserves_observation_and_physics():
    from gymnasium.utils.env_checker import check_env
    a=make_environment();b=make_environment(reward_version='tracking_stop_v3')
    try:
        check_env(b,skip_render_check=True)
        oa,_=a.reset(seed=42);ob,_=b.reset(seed=42);assert np.array_equal(oa,ob)
        for _ in range(6):
            action=np.array([.15,0.,.03],np.float32)
            oa,_,_,_,ia=a.step(action);ob,_,_,_,ib=b.step(action)
            assert np.array_equal(oa,ob) and np.array_equal(a._runner.data.qpos,b._runner.data.qpos)
            assert ib['physics_steps']==100 and ib['gait_calls']==10
            assert ib['evaluation']['reward_version']=='tracking_stop_v3'
    finally:a.close();b.close()


def test_tracking_training_cli_backend_smoke(tmp_path):
    import json,yaml
    from escort_rl.ppo_training import run_training
    from escort_rl.contract import PACKAGE
    c=yaml.safe_load((PACKAGE/'configs/ppo_stage1_tracking_stop_v3.yaml').read_text())
    c['environment_config']=str(PACKAGE/'configs/stage1.yaml');c['split_config']=str(PACKAGE/'configs/stage1_splits.json')
    c['ppo'].update(n_steps=4,batch_size=4,n_epochs=1)
    config=tmp_path/'tracking.yaml';config.write_text(yaml.safe_dump(c))
    r=run_training(config,42,1,8,tmp_path/'smoke',validation=False)
    assert r['end_timesteps']==8 and r['active_physics_steps']==800
    manifest=json.loads((tmp_path/'smoke/manifest.json').read_text())
    assert manifest['reward_version']=='tracking_stop_v3'
    assert manifest['reward_details']['stop_hold_s']==1.
