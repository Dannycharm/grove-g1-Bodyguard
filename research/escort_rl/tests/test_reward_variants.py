"""Deterministic evidence for reward conflict/correction, not synthetic locomotion."""
from dataclasses import replace
import numpy as np
from escort_rl.reward_audit import audit_case,CASES
from escort_rl.reward_variants import FormationProgressEvaluator,make_environment
from escort_rl.stage1_scenarios import load_config,sample_episode


def prefix(result,seconds=3):
    return sum(r['reward'] for r in result['transitions'] if r['elapsed_s']<=seconds and not r.get('absorbing'))


def test_stationary_client_incentive_and_opt_in_correction():
    original=audit_case('stopped_crosses_front');candidate=audit_case('stopped_crosses_front','formation_progress_v2')
    assert prefix(original)>1.8 and prefix(candidate)<0
    assert audit_case('stopped_correct')['total_reward']==0
    assert audit_case('stopped_correct','formation_progress_v2')['total_reward']==0
    assert candidate['reward_components']['progress']==0
    for field in ('formation','command_change','torso_stability','terminal_success','terminal_failure'):
        assert original['reward_components'][field]==candidate['reward_components'][field]


def test_correct_following_recovery_and_anti_farming():
    for case in ('walking_correct','stopped_recovery','resumes_recovers'):
        old=audit_case(case);new=audit_case(case,'formation_progress_v2')
        assert np.isclose(old['total_reward'],new['total_reward'])
        assert new['reward_components']['progress']>0
    for case in ('robot_stationary','races_ahead_waits'):
        new=audit_case(case,'formation_progress_v2')
        assert new['total_reward']<0 and new['reward_components']['terminal_success']==0
    assert audit_case('oscillates','formation_progress_v2')['reward_components']['progress']==0
    # Same 25 s horizon including absorbing terminal states, finite separate components.
    for case in CASES:
        for version in ('legacy_v1','formation_progress_v2'):
            x=audit_case(case,version)
            assert len(x['transitions'])==125 and np.isfinite(x['total_reward'])
            assert np.isclose(x['total_reward'],sum(x['reward_components'].values()))


def test_small_ahead_offset_still_earns_moving_accompaniment():
    c=load_config();client=sample_episode(1,'straight').client_pose_at(0)
    e=FormationProgressEvaluator(c,-1.4,4,lambda xy:float(xy[0]))
    total=0
    for i in range(1,51):
        cp=i*.04;rp=cp-1.4
        _,_,_,info=e.update(elapsed=i*.2,dt=.2,robot_xy=[rp,0],client=replace(client,x=cp,y=0.,yaw=0.),
            robot_progress=rp,client_progress=cp,client_arrived=None,action=np.zeros(3))
        total+=info['new_robot_progress_m']
    assert total>1.8 # No per-step discontinuity for a harmless 0.1m ahead formation offset.


def test_candidate_geometry_and_original_dynamics_identical():
    from gymnasium.utils.env_checker import check_env
    c=load_config();a=make_environment(config=c);b=make_environment(config=c,reward_version='formation_progress_v2')
    try:
        check_env(b,skip_render_check=True)
        oa,_=a.reset(seed=42);ob,_=b.reset(seed=42)
        assert np.array_equal(oa,ob)
        rng=np.random.default_rng(42)
        for _ in range(8):
            action=rng.uniform(-.2,.2,3).astype(np.float32)
            oa,_,_,_,ia=a.step(action);ob,_,_,_,ib=b.step(action)
            assert np.array_equal(oa,ob)
            assert np.array_equal(a._runner.data.qpos,b._runner.data.qpos)
            assert np.array_equal(a._runner.data.qvel,b._runner.data.qvel)
            assert ia['physics_steps']==ib['physics_steps']==100
            assert ib['evaluation']['reward_version']=='formation_progress_v2'
            assert np.isfinite(ib['evaluation']['desired_formation_progress_m'])
        assert a._runner.contract['checkpoint_sha256']==b._runner.contract['checkpoint_sha256']
    finally:a.close();b.close()


def test_comparison_variants_share_initial_policy_and_scenario_stream(tmp_path):
    import torch
    from stable_baselines3 import PPO
    from escort_rl.ppo_training import read_config,make_vec,streams
    from escort_rl.sb3_restart import clear_pending_resets
    torch.set_num_threads(1);policies=[];observations=[];scenario_states=[]
    for version in ('legacy_v1','formation_progress_v2'):
        c=read_config();c['reward_version']=version;v=make_vec(c,1,42)
        try:
            model=PPO('MlpPolicy',v,seed=streams(42,1)['model'],device='cpu',**c['ppo'])
            clear_pending_resets(v)
            policies.append({k:t.clone() for k,t in model.policy.state_dict().items()})
            observations.append(v.reset());scenario_states.append(v.env_method('rng_state'))
        finally:v.close()
    assert all(torch.equal(policies[0][k],v) for k,v in policies[1].items())
    assert np.array_equal(*observations) and scenario_states[0]==scenario_states[1]
