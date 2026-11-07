"""Held-out validation only. Final-test scenarios are deliberately not exposed."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from escort_rl.ppo_training import read_config,DEFAULT_PPO_CONFIG
from escort_rl.stage1_scenarios import load_config
from escort_rl.ppo_evaluation import validation_cases,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_PPO_CONFIG));p.add_argument('--checkpoint')
    p.add_argument('--policy',choices=['ppo','follower','zero'],default='ppo');p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1);c=read_config(a.config);env=load_config(c['environment_path'])
    if a.policy=='ppo' and not a.checkpoint:p.error('--checkpoint is required for PPO')
    model=PPO.load(a.checkpoint,device='cpu') if a.policy=='ppo' else None
    cases=validation_cases(env,json.loads(Path(c['split_path']).read_text()))
    print(json.dumps(evaluate(model,env,cases,a.output,kind=a.policy,reward_version=c['reward_version']),indent=2))


if __name__=='__main__':main()
