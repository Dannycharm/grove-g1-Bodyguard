import argparse,json
from pathlib import Path
import numpy as np
from escort_rl.parity import tensor_report,trajectory_report,observation_fixture_row
from escort_rl.frozen_gait import FrozenGait
p=argparse.ArgumentParser(); p.add_argument('mode',choices=['fixtures','record-runner','python','tensor','trajectory'])
p.add_argument('input'); p.add_argument('output'); a=p.parse_args()
if a.mode=='fixtures':
    rng=np.random.default_rng(598); rows=[]
    for i in range(1000):
        q=rng.normal(0,.02,4); q[0]+=1; q/=np.linalg.norm(q)
        rows.append(np.r_[int(i in (0,1,257,800)),q,rng.normal(0,.1,3),rng.uniform(-.5,.5,3),
                          rng.normal(0,.1,29),rng.normal(0,.1,29)])
    with open(a.output,'x') as f: np.savetxt(f,rows,fmt='%.9g')
elif a.mode=='record-runner':
    from escort_rl.mujoco_runner import MujocoRunner
    r=MujocoRunner();rows=[];actual=[]
    original=r.gait.run
    def capture(quat,gyro,command,position,velocity):
        rows.append(observation_fixture_row(r.gait.calls==0,quat,gyro,command,position,velocity))
        result=original(quat,gyro,command,position,velocity)
        actual.append(np.concatenate([v.ravel() for v in r.gait.last_outputs]))
        return result
    r.gait.run=capture
    for episode in range(2):
        r.reset()
        for command,seconds in [([0,0,0],2),([.35,0,0],3),([0,.35,0],3),
                                ([0,0,.6],3),([.35,0,.3],3),([0,0,0],3)]:
            for i in range(round(seconds/.02)):
                if i%10==0:r.command(command)
                r.advance(.02)
    with open(a.output,'x') as f:np.savetxt(f,rows,fmt='%.9g')
    with open(a.output+'.actual_outputs.txt','x') as f:np.savetxt(f,actual,fmt='%.9g')
    with open(a.output+'.manifest.json','x') as f:json.dump(r.runtime_manifest(),f,indent=2)
    r.close()
elif a.mode=='python':
    g=FrozenGait(); rows=[]
    for row in np.loadtxt(a.input,ndmin=2):
        if row[0]: g.reset()
        g.run(row[1:5],row[5:8],row[8:11],row[11:40],row[40:69])
        rows.append(np.concatenate([v.ravel() for v in g.last_outputs]))
    with open(a.output,'x') as f: np.savetxt(f,rows,fmt='%.9g')
elif a.mode=='tensor':
    print(json.dumps(tensor_report(np.loadtxt(a.input,ndmin=2),np.loadtxt(a.output,ndmin=2)),indent=2))
else:
    def read(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
    print(json.dumps(trajectory_report(read(a.input),read(a.output)),indent=2))
