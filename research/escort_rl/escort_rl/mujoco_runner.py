"""Private full-dynamics MuJoCo instance. No callbacks, ROS, DDS or root writes in step."""
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .contract import ROOT, PACKAGE, load_contract, sha256
from .frozen_gait import FrozenGait
from .controller_reference import ControllerReference, Schedule
from .measurements import robot_velocities, VELOCITY_CONVENTION

class MujocoRunner:
    def __init__(self, scene='flat', actors=0, actor_names=None, actor_driver=None):
        import mujoco as mj
        self.mj = mj
        self.actor_driver = actor_driver
        if mj.__version__ != '3.3.6':
            raise RuntimeError(f'Reference engine is 3.3.6, found {mj.__version__}')
        if scene not in ('flat', 'navigation') or actors not in range(4):
            raise ValueError('scene flat/navigation; actors 0..3 (client plus <=2 pedestrians)')
        self.contract = c = load_contract()
        self.gait = FrozenGait()
        self._temp = tempfile.TemporaryDirectory(prefix='escort_rl_')
        directory = Path(self._temp.name)
        # Match launch staging in private files only. Mesh paths are the only robot XML edit.
        robot_path = ROOT / c['assets']['robot']
        robot = ET.parse(robot_path)
        robot.getroot().find('compiler').set('meshdir', str(robot_path.parent / 'meshes'))
        robot.write(directory / 'g1_29dof.xml')
        (directory / 'g1_walk_base.staged.xml').write_bytes(
            (ROOT / c['assets'][scene + '_scene']).read_bytes())
        overlay = ET.parse(ROOT / c['assets']['overlay'])
        world = ET.SubElement(overlay.getroot(), 'worldbody')
        actor_names = [f'benchmark_actor_{i}' for i in range(actors)] if actor_names is None else list(actor_names)
        if len(actor_names)!=actors or len(set(actor_names))!=actors:
            raise ValueError('One unique name per scripted mocap actor required')
        for i in range(actors):
            body = ET.SubElement(world, 'body', name=actor_names[i], mocap='true',
                                 pos=f'3 {i*1.5} 0')
            ET.SubElement(body, 'geom', type='capsule', fromto='0 0 .24 0 0 1.35',
                          size='.25', contype='0', conaffinity='0')
        overlay.write(directory / 'scene.xml')
        self.model = mj.MjModel.from_xml_path(str(directory / 'scene.xml'))
        self.data = mj.MjData(self.model)
        if abs(self.model.opt.timestep - .002) > 1e-12 or self.model.nu != 29:
            raise ValueError('Unexpected compiled timestep or actuator count')
        self.scene, self.actors = scene, actors
        self.names = c['observation_joints']
        self.joints = np.array([self._id(mj.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        actuator_joints = self.model.actuator_trnid[:, 0]
        self.motor_obs = np.array([list(self.joints).index(j) for j in actuator_joints])
        self.fingers = []
        for side in ('left','right'):
            for suffix, limit in zip(('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1'),
                                     c['holds']['finger_effort_limits']):
                j = self._id(mj.mjtObj.mjOBJ_JOINT, f'{side}_hand_{suffix}_joint')
                self.fingers.append((self.model.jnt_qposadr[j],self.model.jnt_dofadr[j],limit))
        self.pelvis = self._id(mj.mjtObj.mjOBJ_BODY, 'pelvis')
        self.imu_site = self._id(mj.mjtObj.mjOBJ_SITE, 'imu')
        if (self.model.site_bodyid[self.imu_site] != self.pelvis or
                not np.array_equal(self.model.site_pos[self.imu_site], [0, 0, 0]) or
                not np.array_equal(self.model.site_quat[self.imu_site], [1, 0, 0, 0])):
            raise ValueError('Measurement contract requires pelvis-origin, pelvis-aligned imu site')
        self.weld = self._id(mj.mjtObj.mjOBJ_EQUALITY, 'pelvis_startup_hold')
        self._reset_eq_data = self.model.eq_data.copy()
        self._free_joint = int(self.model.body_jntadr[self.pelvis])
        if self.model.jnt_type[self._free_joint] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected G1 pelvis free joint")
        self.controller = ControllerReference(self.names,c['action_joints'])
        self.schedule = Schedule(self.model.opt.timestep)
        self.reset()

    def _id(self, kind, name):
        result = self.mj.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f'Missing model element {name}')
        return result

    def sensor(self, name):
        i = self._id(self.mj.mjtObj.mjOBJ_SENSOR, name)
        a = self.model.sensor_adr[i]
        return self.data.sensordata[a:a+self.model.sensor_dim[i]].copy()

    def reset(self, *, base_xy=None, base_yaw=None):
        # Only this private reset may place the base. Masses/gains/timestep remain untouched.
        self.model.eq_data[:] = self._reset_eq_data
        self.mj.mj_resetData(self.model,self.data)
        if base_xy is not None or base_yaw is not None:
            if base_xy is None or base_yaw is None:
                raise ValueError('Provide both base_xy and base_yaw for a reset spawn')
            xy=np.asarray(base_xy,float)
            if xy.shape!=(2,) or not np.isfinite(xy).all() or not np.isfinite(base_yaw):
                raise ValueError('Finite planar reset pose required')
            if self.model.eq_obj1id[self.weld]!=self.pelvis or self.model.eq_obj2id[self.weld]!=0:
                raise ValueError('Spawn reset requires the reference pelvis-to-world startup weld')
            a=int(self.model.jnt_qposadr[self._free_joint])
            self.data.qpos[a:a+2]=xy
            q=np.array([np.cos(base_yaw/2),0.,0.,np.sin(base_yaw/2)])
            self.data.qpos[a+3:a+7]=q
            R=np.array([[np.cos(base_yaw),-np.sin(base_yaw),0.],
                        [np.sin(base_yaw),np.cos(base_yaw),0.],[0.,0.,1.]])
            self.model.eq_data[self.weld,3:6]=R.T@(-self.data.qpos[a:a+3])
            self.model.eq_data[self.weld,6:10]=q*np.array([1.,-1.,-1.,-1.])
        # Startup-pose equalities, exactly as ApplyStartupPose; only inside private reset.
        for e in range(self.model.neq):
            name = self.mj.mj_id2name(self.model,self.mj.mjtObj.mjOBJ_EQUALITY,e) or ''
            if name.startswith('startup_hold_') and self.model.eq_type[e] == self.mj.mjtEq.mjEQ_JOINT:
                j = self.model.eq_obj1id[e]
                a = self.model.jnt_qposadr[j]
                self.data.qpos[a] = self.model.qpos0[a] + self.model.eq_data[e,0]
                if j in self.joints:
                    self.data.eq_active[e] = 0
        self.mj.mj_forward(self.model,self.data)
        self.gait.reset()
        self.controller.reset(self.data.qpos[self.qadr])
        self.schedule.reset()
        self.finger_hold = [self.data.qpos[a] for a,_,_ in self.fingers]
        self.settle_ticks = 0
        self.pending_release = False
        self.physics_steps = 0
        self.torque = np.zeros(self.model.nu)
        self.data.ctrl[:] = 0
        self.last_command = np.zeros(3)

    def command(self, command):
        self.controller.command(command, self.data.time)

    def _infer(self, event_time):
        return self.gait.run(self.sensor('imu_quat'),self.sensor('imu_gyro'),
            self.controller.command_at(event_time),self.data.qpos[self.qadr],self.data.qvel[self.vadr])

    def step_physics(self):
        d, m, c = self.data, self.model, self.controller
        for kind, time in self.schedule.events():
            if kind == 'controller':
                c.update(d.qvel[self.vadr], lambda: self._infer(time))
            else:
                if c.tick == 0:
                    torque = 200 * (c.activation-d.qpos[self.qadr])-5*d.qvel[self.vadr]
                else:
                    torque = c.torque(d.qpos[self.qadr],d.qvel[self.vadr])
                    self.settle_ticks += 1
                    self.pending_release |= self.settle_ticks >= 500
                self.torque = torque[self.motor_obs]
                # Simulator writes raw torque; MuJoCo enforces ctrlrange itself.
                d.ctrl[:] = self.torque
                for hold,(a,v,limit) in zip(self.finger_hold,self.fingers):
                    d.qfrc_applied[v] = np.clip(1.5*(hold-d.qpos[a])-.2*d.qvel[v],-limit,limit)
        if self.pending_release:
            d.eq_active[self.weld] = 0
        # Optional noncolliding scripted proxy update; never touches robot qpos/qvel.
        if self.actor_driver is None:
            for i in range(self.actors):
                d.mocap_pos[i] = [3 + .5*np.sin(d.time + i), i*1.5, 0]
        else:
            self.actor_driver(float(d.time),d)
        self.mj.mj_step(m,d)
        if self.actor_driver is not None:
            self.actor_driver(float(d.time),d)
        self.mj.mj_forward(m,d)  # refresh sensors/derived poses at the new integration state
        self.physics_steps += 1
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise RuntimeError('Nonfinite dynamics')

    def advance(self, duration=.2):
        steps = round(duration / self.model.opt.timestep)
        if steps < 0 or abs(steps*self.model.opt.timestep-duration)>1e-10:
            raise ValueError('Duration must be a nonnegative integer physics interval')
        for _ in range(steps):
            self.step_physics()
        return self.measure()

    def measure(self):
        rotation = self.data.xmat[self.pelvis].reshape(3,3)
        velocities = robot_velocities(self.mj, self.model, self.data, self.pelvis, self.imu_site)
        return dict(time=float(self.data.time), requested=self.controller.requested.tolist(),
            applied=self.controller.applied.tolist(), **velocities,
            position=self.data.xpos[self.pelvis].tolist(),quaternion=self.data.xquat[self.pelvis].tolist(),
            upright=float(rotation[2,2]),height=float(self.data.xpos[self.pelvis,2]),
            fallen=bool(rotation[2,2]<.5 or self.data.xpos[self.pelvis,2]<.45),
            emergency=bool(self.controller.emergency),contacts=int(self.data.ncon),
            torque_rms=float(np.sqrt(np.mean(self.torque**2))),
            joint_position=self.data.qpos[self.qadr].tolist(),
            joint_velocity=self.data.qvel[self.vadr].tolist(),
            target=self.controller.target.tolist(),kp=self.controller.kp.tolist(),kd=self.controller.kd.tolist())

    def runtime_manifest(self):
        import onnxruntime
        sources={str(p.relative_to(PACKAGE)):sha256(p) for folder in ('escort_rl','tools','configs')
                 for p in sorted((PACKAGE/folder).rglob('*')) if p.suffix in ('.py','.cpp','.json')}
        return dict(engine=self.mj.__version__,onnxruntime=onnxruntime.__version__,numpy=np.__version__,
                    research_files=sources,scene=self.scene,actors=self.actors,
                    measurement_schema_version=2,velocity_convention=VELOCITY_CONVENTION,
                    timestep=self.model.opt.timestep,solver=int(self.model.opt.solver),
                    integrator=int(self.model.opt.integrator),iterations=int(self.model.opt.iterations),
                    tolerance=float(self.model.opt.tolerance),gravity=self.model.opt.gravity.tolist(),
                    reference=self.contract,scheduling_parity='unverified')

    def close(self):
        self._temp.cleanup()
