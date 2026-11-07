"""Isolated Stage 1 policy playback, with a separate render-only model/data copy.

No ROS/DDS imports. Never writes robot state in the dynamics instance. The client
human is decorative MjvGeom geometry, not a physical actor or a perception model.
Keep this tool outside training source globs so inference tooling does not invalidate
strict training-resume inventories.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np


def human_segments(x,y,yaw,t,speed):
    """Current-pose-only stylized figure; no scenario future or physics writes."""
    R=np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    def world(v):return R@np.asarray(v,float)+[x,y,0.]
    swing=.16*np.sin(2*np.pi*1.4*t)*min(1.,max(0.,speed)/.15)
    pieces=[]
    def limb(a,b,r,color):pieces.append((world(a),world(b),r,color))
    blue=[.12,.38,.78,1.];skin=[.82,.60,.43,1.];pants=[.12,.16,.22,1.]
    limb([0,0,.91],[0,0,1.39],.15,blue)
    limb([0,0,1.42],[0,0,1.64],.115,skin)
    for side in (-1,1):
        hip=[0,side*.10,.90];knee=[side*swing,side*.10,.49];foot=[-side*swing,side*.10,.08]
        limb(hip,knee,.065,pants);limb(knee,foot,.055,pants)
        shoulder=[0,side*.20,1.34];elbow=[-side*swing,side*.24,1.08];hand=[-side*swing*1.3,side*.23,.88]
        limb(shoulder,elbow,.045,blue);limb(elbow,hand,.04,skin)
    # Small forward nose makes current heading visually identifiable.
    limb([.06,0,1.60],[.15,0,1.60],.035,skin)
    return pieces


def add_human(mj,scene,client,t):
    for a,b,r,color in human_segments(client.x,client.y,client.yaw,t,np.hypot(client.vx,client.vy)):
        if scene.ngeom>=scene.maxgeom:raise RuntimeError('Render geometry capacity exceeded')
        geom=scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
        mj.mjv_connector(geom,mj.mjtGeom.mjGEOM_CAPSULE,r,a,b);scene.ngeom+=1


class Display:
    def __init__(self,env,mode,shape,width,height):
        import mujoco as mj
        self.mj=mj;self.env=env;self.mode=mode;self.shape=shape;self.family=env.scenario_metadata()["family"]
        # A viewer can perturb its copy, but cannot access or edit dynamics state.
        self.model=copy.copy(env._runner.model);self.data=mj.MjData(self.model)
        self.model.vis.global_.offwidth=max(width,self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight=max(height,self.model.vis.global_.offheight)
        if shape=='human':
            body=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,'escort_client')
            self.model.geom_rgba[self.model.geom_bodyid==body,3]=0.
        self.viewer=None;self.renderer=None;self.frames=[];self.frame_count=0
        self.camera=mj.MjvCamera();self.camera.distance=5.5;self.camera.azimuth=135;self.camera.elevation=-25
        self.copy_state()
        if mode=='viewer':
            import mujoco.viewer
            self.viewer=mujoco.viewer.launch_passive(self.model,self.data,show_left_ui=False,show_right_ui=False)
            self.viewer.cam.distance=5.5;self.viewer.cam.azimuth=135;self.viewer.cam.elevation=-25
        elif mode=='gif':self.renderer=mj.Renderer(self.model,height=height,width=width)

    def copy_state(self):
        src=self.env._runner.data;dst=self.data
        for name in ('qpos','qvel','act','ctrl','mocap_pos','mocap_quat','eq_active'):
            getattr(dst,name)[:]=getattr(src,name)
        dst.time=src.time;self.mj.mj_forward(self.model,dst)

    def draw(self,t):
        self.copy_state();client=self.env._scenario.client_pose_at(t)
        pos=np.array([client.x,client.y,.8]);robot=self.env._runner.data.qpos[:2]
        pos[:2]=(pos[:2]+robot)/2
        if self.viewer:
            if not self.viewer.is_running():raise ViewerClosed()
            with self.viewer.lock():
                self.viewer.cam.lookat[:]=pos;self.viewer.user_scn.ngeom=0
                if self.shape=='human':add_human(self.mj,self.viewer.user_scn,client,t)
            self.viewer.sync(state_only=True)
        if self.renderer:
            from PIL import Image,ImageDraw
            self.camera.lookat[:]=pos;self.renderer.update_scene(self.data,camera=self.camera)
            if self.shape=='human':add_human(self.mj,self.renderer.scene,client,t)
            image=Image.fromarray(self.renderer.render());d=ImageDraw.Draw(image)
            d.rectangle((0,0,image.width,24),fill='white');d.text((8,6),f'{self.family}   t={t:.2f}s   Frozen AGILE plus PPO   Human is visual only',fill='black')
            self.frames.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        self.frame_count+=1

    def close(self):
        if self.viewer:self.viewer.close()
        if self.renderer:self.renderer.close()


class ViewerClosed(Exception):pass


def checked_policy(checkpoint):
    from stable_baselines3 import PPO
    from escort_rl.contract import ROOT,load_contract,sha256
    path=Path(checkpoint).resolve();manifest=json.loads(path.with_suffix('.manifest.json').read_text());contract=load_contract()
    if manifest['frozen_gait_sha256']!=contract['checkpoint_sha256']:raise ValueError('Checkpoint gait contract differs')
    protected=('env.py','observations.py','mujoco_runner.py','frozen_gait.py','controller_reference.py','measurements.py','stage1_scenarios.py','rewards.py','reward_variants.py','tracking_objective.py')
    for name in protected:
        key='research/escort_rl/escort_rl/'+name
        if manifest['source_hashes'][key]!=sha256(ROOT/key):raise ValueError('Inference runtime changed: '+key)
    for name in ('stage1.yaml','tracking_stop_v3.yaml','gait_reference.json'):
        key='research/escort_rl/configs/'+name
        if manifest['source_hashes'][key]!=sha256(ROOT/key):raise ValueError('Inference config changed: '+key)
    if manifest['reward_version']!='tracking_stop_v3':raise ValueError('This player requires objective C')
    model=PPO.load(path,device='cpu')
    if model.observation_space.shape!=(40,) or model.action_space.shape!=(3,):raise ValueError('Unexpected policy spaces')
    return model,manifest,sha256(path)


def run(args):
    import torch
    from escort_rl.reward_variants import make_environment
    from escort_rl.stage1_scenarios import DEFAULT_CONFIG
    from escort_rl.ppo_evaluation import EpisodeMetrics
    torch.set_num_threads(1)
    model,manifest,digest=checked_policy(args.checkpoint)
    if args.seed<0 or args.seed%4!=0:raise ValueError('Playback uses development seeds divisible by four; final-test namespace is prohibited')
    if args.fps not in (5,10,20,25,50):raise ValueError('fps must divide the 500 Hz physics clock: 5,10,20,25,50')
    if args.speed<=0 or args.max_seconds is not None and args.max_seconds<=0:raise ValueError('Positive speed/duration required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    env=make_environment(DEFAULT_CONFIG,reward_version='tracking_stop_v3');display=None;metrics=EpisodeMetrics();status='not_started';last={}
    try:
        obs,reset_info=env.reset(options={'family':args.family,'episode_seed':args.seed})
        from escort_rl.ppo_evaluation import reserved
        split=json.loads(Path(manifest['config']['split_path']).read_text())
        if reserved(env.scenario_metadata(),split):raise ValueError('Scenario falls in reserved final-test combination; choose another development seed')
        (out/'scenario_evaluator_metadata.json').write_text(json.dumps(env.scenario_metadata(),indent=2))
        display=Display(env,args.mode,args.client_shape,args.width,args.height)
        original_driver=env._runner.actor_driver;origin=float(env._runner.data.time);next_frame=origin;begin=time.perf_counter()
        def drive(t,data):
            nonlocal next_frame
            original_driver(t,data)
            if t+1e-9>=next_frame:
                display.draw(max(0.,t-origin));next_frame+=1./args.fps
                if args.mode=='viewer':time.sleep(max(0.,(t-origin)/args.speed-(time.perf_counter()-begin)))
        env._runner.actor_driver=drive
        with (out/'trace.jsonl').open('x') as f:
            while True:
                action,_=model.predict(obs,deterministic=True);old_obs=obs.copy()
                obs,reward,term,trunc,info=env.step(action);last=info;metrics.add(reward,info)
                f.write(json.dumps(dict(observation=old_obs.tolist(),action=action.tolist(),reward=reward,terminated=term,truncated=trunc,info=info))+'\n')
                if term or trunc:status=info['reason'];break
                if args.max_seconds is not None and info['elapsed_s']>=args.max_seconds:status='preview_limit';break
    except ViewerClosed:status='viewer_closed'
    except KeyboardInterrupt:status='keyboard_interrupt'
    finally:
        if display:
            if display.frames:
                display.frames[0].save(out/'episode.gif',save_all=True,append_images=display.frames[1:],duration=round(1000/args.fps),loop=0,optimize=False)
            display.close()
        summary=dict(status=status,full_episode_completed=status not in ('not_started','preview_limit','viewer_closed','keyboard_interrupt'),
            checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=digest,checkpoint_ppo_timesteps=model.num_timesteps,
            family=args.family,episode_seed=args.seed,render_frames=display.frame_count if display else 0,
            render_mode=args.mode,client_appearance=args.client_shape,human_physics='none; decorative geometry only',
            last_elapsed_s=last.get('elapsed_s',0.),reset_info=locals().get('reset_info'),new_training_updates=0)
        if metrics.rows:summary['episode_metrics']=metrics.summary(args.family,args.seed)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));env.close()
    print(json.dumps(summary,indent=2));return summary


def main():
    from escort_rl.contract import PACKAGE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(PACKAGE/'runs/imitation_guarded_20261106_190724_LioFor/preservation_5k/final.zip'))
    p.add_argument('--family',choices=['straight','gentle_turn','stop_start'],required=True);p.add_argument('--seed',type=int,default=320000)
    p.add_argument('--mode',choices=['viewer','gif','none'],default='viewer');p.add_argument('--client-shape',choices=['human','capsule'],default='human')
    p.add_argument('--fps',type=int,default=25);p.add_argument('--speed',type=float,default=1.);p.add_argument('--max-seconds',type=float)
    p.add_argument('--width',type=int,default=800);p.add_argument('--height',type=int,default=500);p.add_argument('--output',required=True)
    a=p.parse_args();run(a)


if __name__=='__main__':main()
