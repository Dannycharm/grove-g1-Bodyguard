"""Training diagnostics/checkpoints; evaluator fields are logging-only."""
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from .ppo_evaluation import EpisodeMetrics


class ResearchCallback(BaseCallback):
    def __init__(self,output,workers,checkpoint_interval,save_checkpoint):
        super().__init__();self.output=Path(output);self.metrics=[EpisodeMetrics() for _ in range(workers)]
        self.interval=checkpoint_interval;self.save_checkpoint=save_checkpoint;self.last_checkpoint=0
        self.episodes=[];self.start=time.perf_counter();self.steps=0;self.physics=0;self.gait=0
        self.file=None;self.trace=None
    def _on_training_start(self):
        if self.file is None:
            self.file=(self.output/'training_episodes.jsonl').open('x')
            self.trace=(self.output/'training_steps.jsonl').open('x')
    def _on_step(self):
        for i,info in enumerate(self.locals['infos']):
            self.metrics[i].add(self.locals['rewards'][i],info);self.steps+=1
            self.physics+=info.get('physics_steps',0);self.gait+=info.get('gait_calls',0)
            self.trace.write(json.dumps(dict(timestep=self.num_timesteps,worker=i,reward=float(self.locals['rewards'][i]),
                reason=info['reason'],reward_components=info.get('reward_components',{}),
                action_clipped=bool(np.any(np.abs(info.get('raw_action',[0,0,0]))>1)),
                action=self.locals['actions'][i].tolist(),observation_clipped=info.get('observation_saturated_features',0),
                physics_steps=info.get('physics_steps',0),gait_calls=info.get('gait_calls',0)))+'\n')
            if self.locals['dones'][i]:
                meta=info['scenario_diagnostics'];summary=self.metrics[i].summary(meta['family'],meta['episode_seed'])
                summary['timestep']=self.num_timesteps;summary['accompanied_fraction']=summary['accompanied_progress_m']/meta['route_length_m']
                self.episodes.append(summary);self.file.write(json.dumps(summary)+'\n');self.file.flush();self.metrics[i]=EpisodeMetrics()
        if self.num_timesteps-self.last_checkpoint>=self.interval:
            self.save_checkpoint(self.model,self.output/f'checkpoint_{self.num_timesteps:09d}')
            self.last_checkpoint=self.num_timesteps
        return True
    def _on_rollout_end(self):
        recent=self.episodes[-100:]
        if recent:
            self.logger.record('rollout/ep_rew_mean',np.mean([e['return_total'] for e in recent]))
            self.logger.record('rollout/ep_len_mean',np.mean([e['steps'] for e in recent]))
            self.logger.record('escort/success_rate',np.mean([e['success'] for e in recent]))
            self.logger.record('escort/formation_mean_m',np.mean([e['formation_mean_m'] for e in recent if e['formation_mean_m'] is not None]))
            for family in ('straight','gentle_turn','stop_start'):
                subset=[e for e in recent if e['family']==family]
                if subset:self.logger.record(f'escort/{family}_success',np.mean([e['success'] for e in subset]))
            for reason,count in Counter(e['reason'] for e in recent).items():self.logger.record('failures/'+reason,count)
            for key in recent[-1]['reward_components']:
                self.logger.record('reward_components/'+key,np.mean([e['reward_components'].get(key,0) for e in recent]))
            for key in ('formation_p95_m','accompanied_progress_m','robot_progress_m','duration_s','action_clipped_fraction','command_saturation_fraction','observation_clipped_features'):
                self.logger.record('escort/'+key,np.mean([e[key] for e in recent if e[key] is not None]))
        self.logger.record('runtime/collected_transitions',self.steps)
        self.trace.flush()
    def close(self):
        for stream in (self.file,self.trace):
            if stream:stream.close()
