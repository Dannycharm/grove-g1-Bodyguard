"""Opt-in Stage 2 tools. Commands never launch ROS/DDS or an existing simulator."""
import argparse,json
from pathlib import Path
from escort_rl.stage2.scenarios import DEFAULT,load_config,sample_episode,FAMILIES
from escort_rl.stage2.training import transition,load_stage2,train
from escort_rl.stage2.evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['scenario','transition','evaluate','train'])
    p.add_argument('--config',default=str(DEFAULT));p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--parent');p.add_argument('--family',choices=FAMILIES,default='crossing');p.add_argument('--seed',type=int,default=320000);p.add_argument('--suite',choices=['stage1','pedestrian'],default='pedestrian');p.add_argument('--episodes-per-encounter',type=int);p.add_argument('--timesteps',type=int,default=5000)
    a=p.parse_args()
    if a.episodes_per_encounter is not None and not 1<=a.episodes_per_encounter<=20:p.error("Use 1..20 evaluation episodes per encounter")
    if a.mode=='scenario':
        c=load_config(a.config);e=sample_episode(a.seed,a.family,c);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);result=e.scenario_metadata();(out/'metadata.json').write_text(json.dumps(result,indent=2))
    elif a.mode=='transition':
        if not a.parent:p.error('transition requires --parent')
        result=transition(a.parent,a.config,a.output)
    elif a.mode=='evaluate':
        if not a.checkpoint:p.error('evaluate requires --checkpoint')
        model,_=load_stage2(a.checkpoint,a.config);c=load_config(a.config);c['_config_path']=str(Path(a.config).resolve());result=evaluate(model,c,a.output,suite=a.suite,per_encounter=a.episodes_per_encounter)
    else:
        if not a.checkpoint:p.error('train requires --checkpoint for explicit Stage 2 continuation')
        result=train(a.checkpoint,a.config,a.output,a.timesteps)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
