"""Render six representative fixed-validation episodes with isolated full-G1 physics.
Clips use a separate display model; decorative humans have no physical contact.
"""
import argparse,json,os,sys,copy,hashlib
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parents[1]/'diagnostics'));sys.path.insert(0,str(HERE.parent))
from residual_ppo import checked_load,read_config,json_numpy
from residual_ppo_policy import ExecutedPolicy
from stage1_viewer import Display,human_segments
from escort_rl.stage2.env import Stage2Env
from escort_rl.contract import ROOT

CASES=[('crossing',3000001),('head_on',3000001),('diagonal',3000037),('straight',1000001),('gentle_turn',1000001),('stop_start',1000001)]
CHECKPOINT=ROOT/'research/escort_rl/runs/residual_ppo_recovered_9htyqV/pilot/final.zip'

class ClipDisplay(Display):
    def __init__(self,env):
        super().__init__(env,'gif','human',768,512)
        for name in ('escort_client','escort_pedestrian_1'):
            bid=self.mj.mj_name2id(self.model,self.mj.mjtObj.mjOBJ_BODY,name)
            self.model.geom_rgba[self.model.geom_bodyid==bid,3]=0.
        self.camera.azimuth=135;self.camera.elevation=-38;self.palette=None;self.stills={}
        try:self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        except OSError:self.font=ImageFont.load_default()
    def human(self,state,t,red=False):
        scene=self.renderer.scene
        for a,b,r,color in human_segments(state.x,state.y,state.yaw,t,np.hypot(state.vx,state.vy)):
            if red and color==[.12,.38,.78,1.]:color=[.85,.18,.12,1.]
            g=scene.geoms[scene.ngeom]
            self.mj.mjv_initGeom(g,self.mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color,np.float32))
            self.mj.mjv_connector(g,self.mj.mjtGeom.mjGEOM_CAPSULE,r,a,b);scene.ngeom+=1
    def draw(self,t):
        self.copy_state();client=self.env._stage2_episode.client_pose_at(t);ped=self.env._stage2_episode.pedestrian_pose_at(t)
        robot=self.env._runner.data.qpos[:2];points=[robot,np.array([client.x,client.y])]
        if ped is not None:points.append(np.array([ped.x,ped.y]))
        center=np.mean(points,axis=0);extent=max(np.linalg.norm(p-center) for p in points)
        self.camera.lookat[:]=[*center,.65];self.camera.distance=max(5.,2.5*extent+2.)
        self.renderer.update_scene(self.data,camera=self.camera);self.human(client,t)
        if ped is not None:self.human(ped,t,True)
        im=Image.fromarray(self.renderer.render());d=ImageDraw.Draw(im)
        d.rectangle((0,0,768,60),fill='#102033');d.text((12,7),f'{self.family.replace("_"," ").title()} | t={t:05.1f}s | playback 2.5x',font=self.font,fill='white')
        d.text((12,33),'G1: frozen Stage 1 + PPO residual | Client: blue | Pedestrian: red',font=self.font,fill='#d6e2f1')
        d.rectangle((0,486,768,512),fill='#102033');d.text((12,490),'Full MuJoCo dynamics. Humans are scripted, noncolliding visual proxies.',font=self.font,fill='white')
        if self.palette is None:self.palette=im.quantize(colors=128)
        self.frames.append(im.quantize(palette=self.palette));self.frame_count+=1
        if self.frame_count%100==0:self.stills[self.frame_count]=im.copy()

def run(output,checkpoint=CHECKPOINT,format="mp4"):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    model,manifest=checked_load(checkpoint,read_config());policy=ExecutedPolicy(model);results=[]
    for family,seed in CASES:
        env=Stage2Env();display=None;folder=out/family;folder.mkdir()
        try:
            obs,_=env.reset(options=dict(family=family,episode_seed=seed));display=ClipDisplay(env)
            old_driver=env._runner.actor_driver;origin=float(env._runner.data.time);next_frame=origin
            def drive(t,data):
                nonlocal next_frame
                old_driver(t,data)
                if t+1e-9>=next_frame:display.draw(max(0.,t-origin));next_frame+=.1
            env._runner.actor_driver=drive;physics=gait=0
            with (folder/'trace.jsonl').open('x') as trace:
                while True:
                    action,_=policy.predict(obs,deterministic=True);obs,r,term,trunc,info=env.step(action);physics+=info['physics_steps'];gait+=info['gait_calls']
                    trace.write(json.dumps(dict(action=action,reward=r,info=info),default=json_numpy)+'\n')
                    if term or trunc:break
            clip=folder/('episode.'+format)
            if format=='gif':display.frames[0].save(clip,save_all=True,append_images=display.frames[1:],duration=40,loop=0,optimize=False)
            else:
                from mp4_encoder import encode_mp4
                encode_mp4(display.frames,clip,fps=25)
            # Middle encounter image and final settling image for quick visual QA.
            for label,index in [('middle',len(display.frames)//2),('final',len(display.frames)-1)]:display.frames[index].convert('RGB').save(folder/f'{label}.png')
            result=dict(family=family,seed=seed,status=info['reason'],simulation_duration_s=info['elapsed_s'],frames=len(display.frames),playback_speed=2.5,active_physics_steps=physics,active_gait_calls=gait,minimum_surface_clearance_m=info.get('minimum_geometric_surface_clearance_m'),clip=str(clip),checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),residual_ppo_transitions=model.num_timesteps,new_training_updates=0,semantics='Full-G1 rollout; render-only human figures; no physical person contacts; fixed validation example, not an independent test')
            (folder/'summary.json').write_text(json.dumps(result,indent=2));results.append(result);print(json.dumps(result),flush=True)
        finally:
            if display:display.close()
            env.close()
    (out/'summary.json').write_text(json.dumps(results,indent=2));return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--checkpoint',default=str(CHECKPOINT));p.add_argument('--format',choices=['mp4','gif'],default='mp4');args=p.parse_args();run(args.output,args.checkpoint,args.format)
