import copy
from collections import Counter
import numpy as np
import pytest
from escort_rl.stage1_scenarios import load_config,sample_episode,FAMILIES


def test_existing_generator_reused_and_seeded():
    for family in FAMILIES:
        a=sample_episode(42,family);b=sample_episode(42,family)
        assert a.scenario_metadata()==b.scenario_metadata()
        assert len(a._progress.actors)==1 and a._progress.resolved_config['pedestrian_count']==0
        assert a.client_pose_at(3)==b.client_pose_at(3)
        assert a.client_pose_at(3).actor_id=='client'
        varied=sample_episode(43,family)
        assert a.scenario_metadata()['scenario_id']!=varied.scenario_metadata()['scenario_id']
        assert not np.allclose([a.client_pose_at(3).x,a.client_pose_at(3).y],[varied.client_pose_at(3).x,varied.client_pose_at(3).y])


def test_motion_continuity_limits_bounds_and_stop():
    c=load_config()
    for family in FAMILIES:
        for seed in range(8):
            e=sample_episode(seed,family);e.validate(c)
            t=np.r_[np.arange(0,e.duration_s,.02),e.duration_s]
            assert (np.diff(t)>0).all()
            states=[e.client_pose_at(x) for x in t]
            xy=np.array([[x.x,x.y] for x in states]);v=np.array([[x.vx,x.vy] for x in states])
            assert np.linalg.norm(v,axis=1).max()<=.25+1e-8
            assert max(np.hypot(x.ax,x.ay) for x in states)<=c['scenario']['max_acceleration_m_s2']+1e-8
            assert max(abs(x.yaw_rate) for x in states)<=c['scenario']['max_turn_rate_rad_s']+1e-8
            assert np.linalg.norm(np.diff(xy,axis=0),axis=1).max()<=.25*.02+1e-8
            assert np.max(np.abs(np.diff([x.yaw for x in states])))<=.06*.02+1e-8
            if family=='gentle_turn':
                moving=np.linalg.norm(v,axis=1)>1e-7
                np.testing.assert_allclose(np.cos([x.yaw for x in states])[moving],v[moving,0]/np.linalg.norm(v[moving],axis=1),atol=1e-9)
            if family=='stop_start':
                stopped=[x for x in states if np.hypot(x.vx,x.vy)<1e-9 and 1e-5<e.client_progress(x.time_s)<e.length-1e-5]
                assert len(stopped)*.02>=.98
                assert np.ptp([x.yaw for x in stopped])<1e-10


def test_probabilities_and_geometry_rejection():
    counts=Counter(sample_episode(seed).scenario_metadata()['family'] for seed in range(1000))
    for name,p in load_config()['families'].items():assert abs(counts[name]/1000-p)<.05
    c=load_config();c['families']['straight']=.8
    with pytest.raises(ValueError,match='sum to one'):sample_episode(0,config=c)
    c=load_config();c['scenario']['bounds_xy_m']=[[-.1,.1],[-.1,.1]]
    with pytest.raises(ValueError,match='No valid'):sample_episode(0,config=c)
    c=load_config();c['scenario']['route_length_m']=[.2,.5]
    with pytest.raises(ValueError,match='3 m'):sample_episode(0,config=c)
