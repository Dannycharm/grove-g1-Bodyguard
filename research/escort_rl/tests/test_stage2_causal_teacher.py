import importlib.util
from pathlib import Path
import numpy as np
import pytest
from escort_rl.stage2.env import Stage2Env

p=Path(__file__).resolve().parents[1]/'diagnostics/stage2_causal_teacher.py'
s=importlib.util.spec_from_file_location('stage2_causal_teacher',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

@pytest.fixture(scope='module')
def config():
    env=Stage2Env()
    try:yield env.config
    finally:env.close()

def observation(c):
    o=c['observation'];x=np.zeros(40,np.float32);x[0]=1.5/o['position_scale_m'];x[5]=1;x[6]=-1.5/o['offset_scale_m'];x[8:10]=1
    return x

def test_no_pedestrian_preserves_original_following(config):
    t=m.Teacher(config);x=observation(config);x[2]=.2/config['observation']['relative_velocity_scale_m_s']
    assert np.array_equal(t.act(x),t.follow(x,config))

def test_approaching_pedestrian_changes_command(config):
    x=observation(config);o=config['observation'];x[2]=.2/o['relative_velocity_scale_m_s'];x[22]=1.2/o['position_scale_m'];x[24]=-.2/o['relative_velocity_scale_m_s'];x[27]=1
    t=m.Teacher(config);a=t.act(x)
    assert a.shape==(3,) and a.dtype==np.float32 and np.isfinite(a).all() and np.abs(a).max()<=1
    assert not np.allclose(a,t.follow(x,config))
    assert t.active

def test_teacher_is_deterministic_and_instances_are_independent(config):
    a=m.Teacher(config);b=m.Teacher(config);x=observation(config);x[22:24]=[.06,.02];x[24]=-.2;x[27]=1
    for _ in range(5):assert np.array_equal(a.act(x),b.act(x))
    old=b.previous.copy();a.act(x*0);assert np.array_equal(old,b.previous)

def test_far_pedestrian_does_not_stop_forward_progress(config):
    x=observation(config);o=config['observation'];x[2]=.2/o['relative_velocity_scale_m_s'];x[22:24]=[.5,.5];x[27]=1
    a=m.Teacher(config).act(x);assert a[0]>0
