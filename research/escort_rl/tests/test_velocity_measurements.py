"""Analytic fixtures do not constitute locomotion evidence; G1 checks use actual dynamics."""
import numpy as np
import pytest
from escort_rl.measurements import euler_yaw_rate, robot_velocities, VELOCITY_CONVENTION


def test_yaw_rate_is_not_body_angular_z():
    # ZYX pose and a world angular velocity; finite-difference the rotation itself.
    mj=pytest.importorskip('mujoco')
    R=np.empty(9); q=np.empty(4)
    mj.mju_euler2Quat(q,np.array([.4,-.3,1.2]),'xyz')
    mj.mju_quat2Mat(R,q);R=R.reshape(3,3)
    omega=np.array([.2,-.3,.7]);body=R.T@omega
    dt=1e-7; delta=np.empty(4);rot=np.empty(9)
    mj.mju_axisAngle2Quat(delta,omega/np.linalg.norm(omega),np.linalg.norm(omega)*dt)
    mj.mju_quat2Mat(rot,delta);next_R=rot.reshape(3,3)@R
    fd=np.arctan2(np.sin(np.arctan2(next_R[1,0],next_R[0,0])-np.arctan2(R[1,0],R[0,0])),
                  np.cos(np.arctan2(next_R[1,0],next_R[0,0])-np.arctan2(R[1,0],R[0,0])))/dt
    assert euler_yaw_rate(R,omega)==pytest.approx(fd,abs=1e-7)
    assert abs(euler_yaw_rate(R,omega)-body[2])>.01
    assert euler_yaw_rate(np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),omega) is None


def test_analytic_turning_point_and_frame_transforms():
    mj=pytest.importorskip('mujoco')
    model=mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="pelvis" pos="1 2 3" euler="20 -15 60"><freejoint/>
      <inertial pos=".2 .1 -.07" mass="1" diaginertia=".1 .2 .3"/>
      <site name="imu" pos="0 0 0"/></body></worldbody></mujoco>''')
    data=mj.MjData(model)
    # Initial mathematical fixture only: no rollout or root-position motion substitute.
    data.qvel[:]=[.3,-.2,.1,.4,.2,.7];mj.mj_forward(model,data)
    p=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'pelvis');s=0
    result=robot_velocities(mj,model,data,p,s);R=data.xmat[p].reshape(3,3)
    expected=np.array(result['com_linear_velocity_world'])+np.cross(
        result['angular_velocity_world'],data.xpos[p]-data.xipos[p])
    np.testing.assert_allclose(result['linear_velocity_world'],expected,atol=1e-12)
    np.testing.assert_allclose(result['linear_velocity_pelvis'],R.T@expected,atol=1e-12)
    assert np.linalg.norm(expected-result['com_linear_velocity_world'])>.01
    native=np.empty(6);mj.mj_objectVelocity(model,data,mj.mjtObj.mjOBJ_SITE,s,native,1)
    np.testing.assert_allclose(result['linear_velocity_pelvis'],native[3:],atol=1e-12)
    legacy=np.empty(6);mj.mj_objectVelocity(model,data,mj.mjtObj.mjOBJ_BODY,p,legacy,1)
    np.testing.assert_allclose(result['legacy_com_velocity'],legacy[[3,4,2]],atol=1e-12)


def test_g1_measurement_is_read_only_and_matches_grove_site():
    pytest.importorskip('mujoco');pytest.importorskip('onnxruntime')
    from escort_rl.mujoco_runner import MujocoRunner
    r=MujocoRunner()
    try:
        differences=[]
        for i in range(250):
            r.command([.35,0,.3] if i>=50 else [0,0,0]);r.advance(.02)
            spec=r.mj.mjtState.mjSTATE_INTEGRATION
            before=np.empty(r.mj.mj_stateSize(r.model,spec));r.mj.mj_getState(r.model,r.data,before,spec)
            row=r.measure();after=before.copy();r.mj.mj_getState(r.model,r.data,after,spec)
            np.testing.assert_array_equal(before,after)
            assert row['velocity_convention']==VELOCITY_CONVENTION
            R=r.data.xmat[r.pelvis].reshape(3,3)
            np.testing.assert_allclose(row['linear_velocity_world'],r.sensor('frame_vel'),atol=1e-12)
            np.testing.assert_allclose(row['linear_velocity_pelvis'],R.T@r.sensor('frame_vel'),atol=1e-12)
            np.testing.assert_allclose(row['angular_velocity_pelvis'],r.sensor('imu_gyro'),atol=1e-12)
            expected=np.array(row['com_linear_velocity_world'])+np.cross(
                row['angular_velocity_world'],r.data.xpos[r.pelvis]-r.data.xipos[r.pelvis])
            np.testing.assert_allclose(row['linear_velocity_world'],expected,atol=1e-12)
            differences.append(np.linalg.norm(np.array(row['body_velocity'])-row['legacy_com_velocity']))
        assert max(differences)>.001
    finally:r.close()


def test_metrics_separate_hold_and_zero_command_times():
    from escort_rl.measurements import response_metrics
    rows=[]
    for i in range(100):
        t=i*.02;speed=.3 if t<.8 else (.1 if t<1.2 else 0.)
        rows.append(dict(time=t,body_velocity=[speed,0,0],position=[0,0,.7],fallen=False,emergency=False,
                         velocity_convention=VELOCITY_CONVENTION))
    cfg=dict(latency_fraction=.5,stop_threshold_m_s=.05,stop_threshold_rad_s=.05,
             stop_dwell_s=.1,sample_period_s=.02,stop_s=.5)
    result=response_metrics(rows,[.3,0,0],0,.8,cfg,stop_command_time=1.2)
    assert result['achieved'][0]==pytest.approx(.3)
    assert result['stopping_time_s']==pytest.approx(0.)
    assert result['hold_end_time']==.8 and result['stop_command_time']==1.2
