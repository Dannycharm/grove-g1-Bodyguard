import numpy as np
import pytest
from escort_rl.moving_reference import candidate_commands,continuous_duration,conditioned_speed


def test_command_matching_is_causal_and_expires():
    commands=np.array([[.6,0,0],[0,0,.9]])
    values,fresh,age=candidate_commands([.9,1.,1.5,1.50001,2.],[1.,2.],commands)
    np.testing.assert_array_equal(fresh,[False,True,True,False,True])
    np.testing.assert_array_equal(values,[[0,0,0],[.6,0,0],[.6,0,0],[0,0,0],[0,0,.9]])
    with pytest.raises(ValueError):candidate_commands([0],[],[])


def test_continuous_condition_resets_on_change_or_gap():
    values=continuous_duration([0,.01,.02,.1,.11,.12],[True,True,False,True,True,True])
    np.testing.assert_allclose(values,[0,.01,0,0,.01,.02])


def test_conditioned_speed_uses_time_weights_and_excludes_turns():
    t=np.arange(0,2,.01);v=np.zeros((len(t),3));v[:,0]=.4
    u=np.zeros((len(t),3));u[:,0]=.6;u[t>1.5,2]=.9;v[t>1.5,0]=.8
    report=conditioned_speed(t,v,u,np.ones(len(t),bool),.1,1.)
    assert report['mean_vx']==pytest.approx(.4)
    assert .49<=report['duration_s']<=.52
    report=conditioned_speed(t,v,u,np.zeros(len(t),bool),.1,1.)
    assert report['samples']==0 and report['mean_vx'] is None
