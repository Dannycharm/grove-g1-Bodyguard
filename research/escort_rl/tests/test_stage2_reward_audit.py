"""Diagnostic controllers obey the causal vector/action boundary, without training."""
import importlib.util
import numpy as np
from escort_rl.contract import PACKAGE
from escort_rl.stage1_scenarios import load_config

spec=importlib.util.spec_from_file_location('reward_audit',PACKAGE/'diagnostics/stage2_reward_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def observation():
    o=np.zeros(40,np.float32);o[0]=1.5/16;o[5]=1;o[6]=-1.5/3;o[8:10]=1;o[10]=.2/2;o[22]=.6/16;o[23]=.6/16;o[24]=-.2/2;o[25]=-.3/2;o[26]=.25;o[27]=1;return o


def test_all_controller_outputs_finite_and_bounded_without_mutating_observation():
    o=observation();before=o.copy();config=load_config()
    for strategy in ('follow','yield','sidestep','stop'):
        controller=module.Controller(strategy,config);action=controller.act(o)
        assert action.shape==(3,) and action.dtype==np.float32 and np.isfinite(action).all() and np.max(np.abs(action))<=1
        assert np.array_equal(o,before)
        if strategy in ('yield','sidestep'):assert controller.active and action[0]==0
        if strategy=='stop':assert not action.any()


def test_absent_pedestrian_uses_original_follower_and_stop_remains_zero():
    config=load_config();o=observation();o[22:]=0
    original=module.Controller('follow',config).act(o)
    for strategy in ('yield','sidestep'):assert np.array_equal(module.Controller(strategy,config).act(o),original)
    assert not module.Controller('stop',config).act(o).any()


def test_horizon_endpoint_inside_risk_radius_triggers_before_closest_approach():
    o=observation();o[22:24]=np.array([.7,-1.5])/16;o[24:26]=np.array([-.2,.16])/2
    c=module.Controller('yield',load_config());a=c.act(o)
    assert c.active and a[0]==0
