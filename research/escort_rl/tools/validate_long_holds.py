"""Supplement suite with longer forward holds and independent velocity checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from escort_rl.mujoco_runner import MujocoRunner

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--commands',nargs='+',type=float,default=[.1,.125,.15,.175,.4,.45,.5,.6])
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=MujocoRunner(); results=[]
    with open(out/'manifest.json','x') as f:json.dump(r.runtime_manifest(),f,indent=2)
    for command in a.commands:
        for trial,stand_s in enumerate((2.,4.,6.),start=1):
            r.reset();rows=[];checks=[]
            for label,duration,cmd in [('stand',stand_s,[0,0,0]),('hold',20.,[command,0,0]),('stop',5.,[0,0,0])]:
                for i in range(round(duration/.02)):
                    if i%10==0:r.command(cmd)
                    before=r.data.xpos[r.pelvis].copy()
                    row=r.advance(.02);row['segment']=label;rows.append(row)
                    world=r.sensor('frame_vel')
                    rotation=r.data.xmat[r.pelvis].reshape(3,3)
                    body=rotation.T@world
                    if label=='hold' and i>=500:
                        measured=np.array(row['body_velocity'][:2])
                        finite_difference=rotation.T@((np.array(row['position'])-before)/.02)
                        checks.append([*(measured-body[:2]),*(measured-finite_difference[:2])])
            name=f'vx_{command:g}_trial_{trial}'
            with open(out/(name+'.jsonl'),'x') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
            hold=[row for row in rows if row['segment']=='hold' and row['time']>stand_s+10+1e-8]
            values=np.array([x['body_velocity'] for x in hold]);checks=np.array(checks)
            v=values[:,0];window=50
            rolling=np.convolve(v,np.ones(window)/window,'valid')
            results.append(dict(name=name,requested=command,stationary_s=stand_s,
                velocity_convention=hold[-1]['velocity_convention'],
                mean_body_velocity=values.mean(axis=0).tolist(),
                mean_legacy_com_velocity=np.mean([x['legacy_com_velocity'] for x in hold],axis=0).tolist(),vx_p05_p95=np.percentile(v,[5,95]).tolist(),
                vx_one_second_mean_min_max=[float(rolling.min()),float(rolling.max())],
                failure=any(x['fallen'] or x['emergency'] for x in rows),
                imu_site_vs_body_velocity_max_abs=float(np.abs(checks[:,:2]).max()),
                finite_difference_vs_body_velocity_rms=float(np.sqrt(np.mean(checks[:,2:]**2))),
                actual_applied=hold[-1]['applied']))
            print(name,results[-1]['mean_body_velocity'],results[-1]['failure'],flush=True)
    with open(out/'summary.json','x') as f:json.dump(results,f,indent=2)
    r.close()

if __name__=='__main__':main()
