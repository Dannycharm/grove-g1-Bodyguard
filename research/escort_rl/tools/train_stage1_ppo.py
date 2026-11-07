"""Explicit isolated PPO entry point. Never imports or launches ROS/DDS."""
import argparse
import json
from pathlib import Path
from escort_rl.ppo_training import DEFAULT_PPO_CONFIG,run_training,benchmark


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_PPO_CONFIG));p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,choices=[1,2],default=1);p.add_argument('--timesteps',type=int,default=10000)
    p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--benchmark',action='store_true')
    p.add_argument('--chunks',type=int,choices=[1,2,3],default=1,
        help='One ordinary run, or 2/3 fresh 10k chunks for a controlled 20k/30k comparison')
    a=p.parse_args()
    if a.chunks>1:
        if a.resume or a.benchmark or a.timesteps!=10000:
            p.error('Multi-chunk diagnostics require a fresh start, no benchmark, and 10000 steps/chunk')
        root=Path(a.output);root.mkdir(parents=True,exist_ok=False);results=[];resume=None
        for i in range(a.chunks):
            folder=root/f'chunk_{i+1:02d}'
            results.append(run_training(a.config,a.seed,a.workers,a.timesteps,folder,resume))
            resume=folder/'final.zip'
        result=dict(chunks=results,final_checkpoint=str(resume),aggregate_new_transitions=sum(r['new_transitions'] for r in results))
        (root/'comparison_result.json').write_text(json.dumps(result,indent=2))
    else:
        result=benchmark(a.config,a.output) if a.benchmark else run_training(a.config,a.seed,a.workers,a.timesteps,a.output,a.resume)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
