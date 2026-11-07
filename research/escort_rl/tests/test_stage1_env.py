import copy
from dataclasses import replace
import inspect
import numpy as np
import pytest
pytest.importorskip('gymnasium')
from gymnasium.utils.env_checker import check_env
from escort_rl.env import G1EscortEnv
from escort_rl.stage1_scenarios import load_config
from escort_rl.observations import build_observation,SIZE


def test_gymnasium_checker():
    env=G1EscortEnv()
    try:check_env(env,skip_render_check=True)
    finally:env.close()


def test_observation_spaces_paused_warmup_and_actual_schedule():
    env=G1EscortEnv()
    try:
        obs,info=env.reset(seed=42,options={'family':'gentle_turn'})
        assert obs.shape==(SIZE,) and obs.dtype==np.float32 and env.observation_space.contains(obs)
        assert not obs[22:].any() and np.isfinite(obs).all()
        assert info['elapsed_s']==0 and info['warmup_simulation_s']==pytest.approx(2.)
        assert info['startup_weld_released'] and not env._runner.data.eq_active[env._runner.weld]
        client=env._scenario.client_pose_at(0)
        np.testing.assert_allclose(env._runner.data.mocap_pos[env._client_slot,:2],[client.x,client.y],atol=1e-12)
        time=env._runner.data.time
        obs,reward,terminated,truncated,info=env.step(np.array([2.,-2.,3.],np.float32))
        assert env._runner.data.time-time==pytest.approx(.2,abs=1e-10)
        assert info['physics_steps']==100 and info['gait_calls']==10
        np.testing.assert_allclose(info['clipped_command'],[.6,-.6,.9])
        assert obs.dtype==np.float32 and env.observation_space.contains(obs) and np.isfinite(reward)
        with pytest.raises(ValueError):env.step([np.nan,0,0])
    finally:env.close()


def test_reset_reproduces_full_state_and_environment_timers():
    env=G1EscortEnv()
    try:
        def trace():
            obs,info=env.reset(seed=81,options={'family':'straight'})
            result=[obs.copy()]
            for i in range(6):
                obs,reward,term,trunc,info=env.step(np.array([.5,0,.05],np.float32))
                result.append(np.r_[obs,reward,term,trunc])
            return result,env._runner.data.qpos.copy()
        first,q=trace()
        spec=env._runner.mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(env._runner.mj.mj_stateSize(env._runner.model,spec))
        env._runner.mj.mj_getState(env._runner.model,env._runner.data,state,spec)
        second,q2=trace()
        state2=state.copy();env._runner.mj.mj_getState(env._runner.model,env._runner.data,state2,spec)
        np.testing.assert_array_equal(state,state2)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(q,q2)
        env.reset(seed=81,options={'family':'straight'})
        assert env._elapsed==0 and env._evaluator.loss_seconds==0
        assert not env._previous_applied.any() and not env._runner.controller.emergency
        assert env._runner.physics_steps==1000 and env._runner.gait.calls==99
    finally:env.close()


def test_independent_instances_and_client_action_independence():
    a,b=G1EscortEnv(),G1EscortEnv()
    try:
        a.reset(seed=17,options={'family':'stop_start'});b.reset(seed=17,options={'family':'stop_start'})
        for _ in range(5):
            a.step(np.array([.7,0,0],np.float32));b.step(np.zeros(3,np.float32))
            np.testing.assert_array_equal(a._runner.data.mocap_pos,b._runner.data.mocap_pos)
        assert not np.array_equal(a._runner.data.qpos,b._runner.data.qpos)
        before=b._runner.data.qpos.copy();a.reset(seed=18)
        np.testing.assert_array_equal(before,b._runner.data.qpos)
    finally:a.close();b.close()


def test_no_future_information_in_actor_builder():
    env=G1EscortEnv()
    try:
        obs,_=env.reset(seed=42,options={'family':'straight'})
        assert 'scenario' not in inspect.signature(build_observation).parameters
        actor=env._scenario._progress.actors[0]
        segment=actor.segments[0]
        future=replace(segment,xy1=(segment.xy1[0]+100,0))
        new_actor=replace(actor,segments=(future,))
        env._scenario._progress=replace(env._scenario._progress,actors=(new_actor,))
        env._scenario._metadata['family']='private_future_label'
        after,_,_,_,_=env._observe()
        np.testing.assert_array_equal(obs,after)
    finally:env.close()


def test_timeout_and_client_loss_are_distinct():
    c=load_config();c['environment']['time_limit_s']=.4
    env=G1EscortEnv(config=c)
    try:
        env.reset(seed=1);env.step(np.zeros(3,np.float32))
        _,_,term,trunc,info=env.step(np.zeros(3,np.float32))
        assert not term and trunc and info['reason']=='time_limit'
        with pytest.raises(RuntimeError):env.step(np.zeros(3,np.float32))
    finally:env.close()
    c=load_config();c['outcomes']['client_distance_limit_m']=.01;c['outcomes']['client_loss_grace_s']=.4
    env=G1EscortEnv(config=c)
    try:
        env.reset(seed=1);env.step(np.zeros(3,np.float32))
        _,_,term,trunc,info=env.step(np.zeros(3,np.float32))
        assert term and not trunc and info['reason']=='client_lost'
    finally:env.close()


def test_zero_actions_cannot_complete_and_robot_does_not_follow_kinematically():
    env=G1EscortEnv()
    try:
        env.reset(seed=31,options={'family':'straight'})
        initial=env._runner.data.xpos[env._runner.pelvis].copy()
        for _ in range(500):
            obs,reward,term,trunc,info=env.step(np.zeros(3,np.float32))
            if term or trunc:break
        assert term or trunc
        assert info['reason']!='success' and info['evaluation']['robot_advance_m']<.1
        assert np.linalg.norm(env._runner.data.xpos[env._runner.pelvis,:2]-initial[:2])<.1
    finally:env.close()


@pytest.mark.parametrize("failure_type", ["runtime", "mujoco_fatal"])
def test_explicit_physics_failure_and_fall_channels(failure_type):
    env=G1EscortEnv()
    try:
        env.reset(seed=3)
        original=env._runner.step_physics
        exception_type=RuntimeError if failure_type=='runtime' else env._runner.mj.FatalError
        def fail():raise exception_type('injected unit fixture')
        env._runner.step_physics=fail
        obs,reward,term,trunc,info=env.step(np.zeros(3,np.float32))
        assert term and not trunc and info['reason']=='physics_failure'
        assert info['physics_steps']==0 and not obs[8:10].any()
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        env._runner.step_physics=original;env.reset(seed=3)
        measure=env._runner.measure
        def fallen():
            row=measure();row['fallen']=True;return row
        env._runner.measure=fallen
        _,_,term,trunc,info=env.step(np.zeros(3,np.float32))
        assert term and not trunc and info['reason']=='fall'
    finally:env.close()
