"""Stage 2 geometry, causal slots, full dynamics replay and explicit stage transition."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import copy,json
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from escort_rl.contract import PACKAGE
from escort_rl.reward_variants import make_environment
from escort_rl.stage2.scenarios import load_config,sample_episode,FAMILIES,ENCOUNTERS,DEFAULT
from escort_rl.stage2.env import Stage2Env,swept_clearance
from escort_rl.stage2.evaluation import gate,cases


def test_scenarios_repeat_mix_geometry_and_motion():
    c=load_config();counts=Counter()
    for seed in range(80):
        ep=sample_episode(seed*4,config=c);m=ep.scenario_metadata();counts[m['family']]+=1
        assert m==sample_episode(seed*4,config=c).scenario_metadata()
        assert m['pedestrian_count']==int(m['family'] in ENCOUNTERS)
        if ep.pedestrian:
            times=np.arange(0,ep.pedestrian.duration_s+.05,.05)
            states=[ep.pedestrian_pose_at(float(t)) for t in times]
            assert max(np.hypot(s.vx,s.vy) for s in states)<=m['encounter']['peak_speed_m_s']+1e-8
            assert max(np.hypot(s.ax,s.ay) for s in states)<=c['pedestrian']['max_acceleration_m_s2']+1e-8
            assert all(s.actor_id=='pedestrian_1' for s in states)
            assert m['encounter']['client_pedestrian_continuous_clearance_lower_bound_m']>=c['pedestrian']['minimum_client_surface_clearance_m']
    assert set(counts)==set(FAMILIES)
    assert 15<=sum(counts[f] for f in FAMILIES[:3])<=50


def test_invalid_config_and_swept_geometry(tmp_path):
    import yaml
    c=yaml.safe_load(DEFAULT.read_text());c['families']['crossing']=.9;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='probabilities'):load_config(p)
    assert np.isclose(swept_clearance([1,0],[-1,0],.55),-.55)
    assert np.isclose(swept_clearance([1,1],[-1,1],.55),.45)


@pytest.mark.parametrize('family',['straight','gentle_turn','stop_start'])
def test_stage1_replay_is_exact_full_dynamics(family):
    a=make_environment(reward_version='tracking_stop_v3');b=Stage2Env()
    try:
        oa,_=a.reset(options=dict(family=family,episode_seed=320000));ob,_=b.reset(options=dict(family=family,episode_seed=320000))
        assert np.array_equal(oa,ob)
        for _ in range(3):
            ra=a.step(np.array([.25,.01,.02],np.float32));rb=b.step(np.array([.25,.01,.02],np.float32))
            assert np.array_equal(ra[0],rb[0]) and ra[1:4]==rb[1:4]
            assert np.array_equal(a._runner.data.qpos,b._runner.data.qpos) and np.array_equal(a._runner.data.qvel,b._runner.data.qvel)
            assert rb[-1]['physics_steps']==100 and rb[-1]['gait_calls']==10
            assert not rb[0][22:].any() and rb[-1]['minimum_geometric_surface_clearance_m'] is None
    finally:a.close();b.close()


def test_causal_pedestrian_slots_reset_and_intrusion():
    e=Stage2Env()
    try:
        a,_=e.reset(seed=42,options=dict(family='crossing',episode_seed=320000));trace=[]
        for _ in range(3):trace.append(e.step(np.zeros(3))[0])
        b,_=e.reset(seed=42,options=dict(family='crossing',episode_seed=320000));assert np.array_equal(a,b)
        assert b.dtype==np.float32 and b.shape==(40,) and e.observation_space.contains(b)
        assert b[27]==1 and not b[28:].any()
        for old in trace:assert np.array_equal(old,e.step(np.zeros(3))[0])
        state=e._stage2_episode.pedestrian_pose_at(0);xy=e._runner.measure()['position'][:2]
        e._stage2_episode.pedestrian_pose_at=lambda t:replace(state,x=xy[0],y=xy[1],vx=0.,vy=0.)
        _,r,term,trunc,info=e.step(np.zeros(3));assert term and not trunc and info['reason']=='pedestrian_intrusion'
        assert info['pedestrian_intrusion'] and info['reward_components']['terminal_success']==0
        assert info['minimum_geometric_surface_clearance_m']<0 and np.isfinite(r)
        e.reset(options=dict(family='straight',episode_seed=320000));assert not e._intrusion and e._previous_relative is None
    finally:e.close()


def test_gym_checker_and_no_metadata_access_during_observe():
    e=Stage2Env()
    try:
        check_env(e,skip_render_check=True)
        e.reset(options=dict(family='diagonal',episode_seed=320000))
        e._stage2_episode.scenario_metadata=lambda:(_ for _ in ()).throw(AssertionError('privileged metadata accessed'))
        obs,*_=e._observe();assert np.isfinite(obs).all() and obs[27]==1
    finally:e.close()


def test_explicit_transition_preserves_optimizer_and_stage1_outputs(tmp_path):
    from escort_rl.stage2.training import transition,load_stage2
    parent=PACKAGE/'runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip'
    meta=transition(parent,DEFAULT,tmp_path/'transition');assert meta['stage1_probe_outputs_exact'] and meta['stage2_training_transitions']==0
    model,saved=load_stage2(tmp_path/'transition/initial.zip',DEFAULT)
    assert model.num_timesteps==5000 and model.gradient_steps==194 and model.policy.optimizer.state
    for subnet in ('policy_net','value_net'):
        layer=getattr(model.policy.mlp_extractor,subnet)[0];assert not torch.count_nonzero(layer.weight[:,22:28])
        assert not torch.count_nonzero(model.policy.optimizer.state[layer.weight]['exp_avg'][:,22:28])
    assert saved['parent_sha256'] and saved['workers']==2


def test_gate_requires_both_full_suites_and_each_family():
    c=load_config();parent={'formation_mean_m':.0563}
    ped=dict(full_suite=True,success_rate=.85,per_family={f:{'success_rate':.85} for f in ENCOUNTERS},reasons={},geometric_intrusions=0,formation_mean_m=.2)
    old=dict(full_suite=True,success_rate=1.,per_family={f:{'success_rate':1.} for f in FAMILIES[:3]},reasons={},geometric_intrusions=0,formation_mean_m=.06)
    assert gate(ped,old,c,parent)['round_pass']
    ped['per_family']['diagonal']['success_rate']=.5;assert not gate(ped,old,c,parent)['round_pass']
    ped['per_family']['diagonal']['success_rate']=.85;old['success_rate']=.9;assert not gate(ped,old,c,parent)['round_pass']


def test_two_stage2_instances_are_independent():
    a=Stage2Env();b=Stage2Env()
    try:
        a.reset(options=dict(family='crossing',episode_seed=320000));b.reset(options=dict(family='diagonal',episode_seed=320004))
        state=b._runner.data.qpos.copy();phase=b._runner.physics_steps;command=b._runner.controller.applied.copy()
        a.step(np.array([.2,0,0],np.float32))
        assert np.array_equal(b._runner.data.qpos,state) and b._runner.physics_steps==phase and np.array_equal(b._runner.controller.applied,command)
        assert a._stage2_episode is not b._stage2_episode and a._runner.gait is not b._runner.gait
    finally:a.close();b.close()
