import numpy as np
import pytest
pytest.importorskip('mujoco')
pytest.importorskip('onnxruntime')
from escort_rl.frozen_gait import FrozenGait
from escort_rl.mujoco_runner import MujocoRunner

def test_gait_reset_and_independent_feedback():
    a,b=FrozenGait(),FrozenGait()
    args=([1,0,0,0],[0,0,0],[.1,0,0],np.zeros(29),np.zeros(29))
    first=a.run(*args)
    for _ in range(20): a.run(*args)
    assert not any(x.any() for x in b.state.values())
    a.reset(); reset=a.run(*args); independent=b.run(*args)
    for x,y,z in zip(first,reset,independent):
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(y,z)

def test_full_dynamics_reset_and_interleaving():
    a,b=MujocoRunner(actors=3),MujocoRunner(actors=3)
    try:
        def trace(r):
            rows=[]
            for _ in range(15):
                r.command([.1,0,.1]);r.advance(.2);rows.append(r.data.qpos.copy())
            return rows
        expected=trace(a); a.reset()
        assert a.gait.calls==0 and a.controller.tick==0 and a.settle_ticks==0
        assert a.data.eq_active[a.weld]
        actual=[]
        for i in range(15):
            b.command([-.2,0,0]); b.advance(.2)
            a.command([.1,0,.1]);a.advance(.2);actual.append(a.data.qpos.copy())
        np.testing.assert_array_equal(expected,actual)
        assert not a.data.eq_active[a.weld]
        assert not np.shares_memory(a.data.qpos,b.data.qpos)
        assert a.physics_steps==1500
        # Motion must emerge from actuators and contacts, with root free.
        assert a.model.nu==29 and a.data.ncon>0
    finally:
        a.close();b.close()

def test_compiled_reference_mappings_and_graph_gains():
    r=MujocoRunner()
    try:
        for i, spec in enumerate(r.contract['assets']['actuators']):
            assert r.names[int(r.motor_obs[i])] == spec['joint']
            np.testing.assert_array_equal(r.model.actuator_ctrlrange[i],
                                          [float(x) for x in spec['ctrlrange'].split()])
            np.testing.assert_array_equal(r.model.actuator_gear[i],[1,0,0,0,0,0])
            assert r.model.actuator_ctrllimited[i]
        r.advance(.04)
        np.testing.assert_array_equal(r.gait.last_outputs[1].ravel(),r.contract['kp'])
        np.testing.assert_allclose(r.gait.last_outputs[2].ravel(),r.contract['kd'],atol=1e-7,rtol=0)
        for spec, state in zip(r.contract['inputs'][5:], r.gait.last_outputs[3:]):
            np.testing.assert_array_equal(r.gait.state[spec['name']],state)
    finally:
        r.close()


def test_complete_integration_and_feedback_reset():
    r=MujocoRunner(actors=3)
    try:
        def state():
            spec=r.mj.mjtState.mjSTATE_INTEGRATION
            result=np.empty(r.mj.mj_stateSize(r.model,spec))
            r.mj.mj_getState(r.model,r.data,result,spec)
            return result
        initial=state()
        def sequence():
            states=[]
            for cmd in ([0,0,0],[.35,0,.3],[0,.35,0],[0,0,0]):
                for _ in range(5):
                    r.command(cmd);r.advance(.2);states.append(state())
            return states,[v.copy() for v in r.gait.last_outputs]
        first,feedback=sequence()
        r.reset()
        np.testing.assert_array_equal(state(),initial)
        assert not r.pending_release and r.settle_ticks==0 and r.physics_steps==0
        assert r.schedule.step==0 and r.schedule.next_controller==5_000_000
        assert r.schedule.next_bridge==0 and not r.controller.emergency
        assert r.controller.blend==0 and r.controller.command_stamp is None
        assert all(not v.any() for v in r.gait.state.values())
        assert r.gait.last_outputs is None
        second,feedback2=sequence()
        np.testing.assert_array_equal(first,second)
        for x,y in zip(feedback,feedback2):np.testing.assert_array_equal(x,y)
    finally:
        r.close()


def test_startup_release_and_ten_second_stand():
    r=MujocoRunner()
    try:
        release=None;contact_peak=0
        for i in range(5000):
            r.step_physics()
            if not r.data.eq_active[r.weld] and release is None:
                release=r.data.time-r.model.opt.timestep
            contact_peak=max(contact_peak,r.data.ncon)
            if i%100==99:
                row=r.measure()
                assert not row['fallen'] and not row['emergency']
                if row['time']>1:
                    assert row['height']>.65 and row['upright']>.95
        assert release==pytest.approx(.504,abs=1e-12)
        assert contact_peak>0
        assert r.gait.calls==499
    finally:
        r.close()
