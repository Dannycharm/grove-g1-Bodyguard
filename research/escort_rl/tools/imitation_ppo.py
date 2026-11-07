"""Explicit offline imitation-pretrained PPO diagnostics. No ROS/DDS authority."""
import argparse
import json
from escort_rl.contract import PACKAGE
from escort_rl.imitation_ppo import noise_sweep,train


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['noise','train'])
    p.add_argument('--config',default=str(PACKAGE/'configs/ppo_stage1_imitation_warmstart_v1.yaml'))
    p.add_argument('--output',required=True);p.add_argument('--noise-report');p.add_argument('--resume')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--workers',type=int,choices=[1,2],default=2)
    p.add_argument('--timesteps',type=int,default=5000)
    a=p.parse_args()
    if a.mode=='noise':result=noise_sweep(a.config,a.output)
    else:result=train(a.config,a.output,a.noise_report,a.seed,a.workers,a.timesteps,a.resume)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
