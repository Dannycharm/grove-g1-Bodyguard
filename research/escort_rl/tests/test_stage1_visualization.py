"""Viewer accesses only a separate render model; visualization cannot drive dynamics."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from escort_rl.contract import PACKAGE
from escort_rl.reward_variants import make_environment

spec=importlib.util.spec_from_file_location('stage1_player',PACKAGE/'visualization/stage1_viewer.py')
player=importlib.util.module_from_spec(spec);spec.loader.exec_module(player)


@pytest.mark.parametrize('family',['straight','gentle_turn','stop_start'])
def test_visual_copy_and_callback_preserve_dynamics(family):
    a=make_environment(reward_version='tracking_stop_v3');b=make_environment(reward_version='tracking_stop_v3')
    display=None
    try:
        oa,_=a.reset(options=dict(family=family,episode_seed=320000));ob,_=b.reset(options=dict(family=family,episode_seed=320000))
        display=player.Display(b,'none','human',640,400)
        assert display.model is not b._runner.model and display.data is not b._runner.data
        display.data.qpos[0]+=10 # Even a viewer-side perturbation cannot move the robot.
        assert not np.array_equal(display.data.qpos,b._runner.data.qpos)
        original=b._runner.actor_driver
        def callback(t,d):original(t,d);display.draw(max(0.,t-b._origin))
        b._runner.actor_driver=callback
        action=np.array([.25,.01,.02],np.float32)
        for _ in range(2):
            ra=a.step(action);rb=b.step(action)
            assert np.array_equal(ra[0],rb[0]) and ra[1:4]==rb[1:4]
            assert np.array_equal(a._runner.data.qpos,b._runner.data.qpos)
            assert np.array_equal(a._runner.data.qvel,b._runner.data.qvel)
            assert np.array_equal(a._runner.data.ctrl,b._runner.data.ctrl)
    finally:
        if display:display.close()
        a.close();b.close()


def test_human_draw_geometries_are_finite_and_stop_swing():
    a=player.human_segments(1,2,.5,0,0);b=player.human_segments(1,2,.5,.4,0)
    assert len(a)==11
    for one,two in zip(a,b):
        assert np.isfinite(np.r_[one[0],one[1],one[2],one[3]]).all()
        assert np.array_equal(one[0],two[0]) and np.array_equal(one[1],two[1])
