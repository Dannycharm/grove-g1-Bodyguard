"""Reuse immutable actor-noise evidence for a NEW lr/KL-only initialization experiment.

Never loads/relabels a checkpoint or bypasses strict PPO resume provenance. Evaluated
actor, Gaussian law, noise gates, scenarios and environment must remain identical.
"""
import argparse
import copy
import json
from pathlib import Path
from escort_rl.contract import sha256
from escort_rl.ppo_training import read_config,provenance
from escort_rl.imitation_ppo import PROTOCOL,write,noise_pass


def prepare(config, source_report, output):
    c=read_config(config);p=Path(source_report);r=json.loads(p.read_text())
    oldmanifest=json.loads((p.parent/'manifest.json').read_text());current=provenance(c)
    if r['protocol']!=PROTOCOL or oldmanifest['dependencies']!=current['dependencies']:
        raise ValueError('Calibration runtime/protocol mismatch')
    a,b=copy.deepcopy(r['config']),copy.deepcopy(c)
    for cfg in (a,b):
        cfg['ppo'].pop('learning_rate',None);cfg['ppo'].pop('target_kl',None)
    if a!=b:raise ValueError('Calibration reuse allows ONLY learning-rate/target-KL changes')
    if c['ppo'].get('target_kl')!=.01 or c['ppo']['learning_rate']!=1e-5:
        raise ValueError('Require declared guarded settings: lr=1e-5, target_kl=.01')
    for path,digest in r['source_hashes'].items():
        if current['source_hashes'].get(path)!=digest:
            raise ValueError('Previously measured source changed: '+path)
    passing=[x['sigma'] for x in r['results'] if noise_pass(x,c['imitation'])]
    if not passing or r['selected_sigma']!=max(passing):raise ValueError('No valid measured calibration')
    r=copy.deepcopy(r);r.update(config=c,source_hashes=current['source_hashes'],
        calibration_reuse=dict(source_report=str(p.resolve()),source_report_sha256=sha256(p),
            rationale='Same frozen actor, noise law, scenarios, gates and environment; only subsequent PPO lr/KL differ. No new noise trials claimed.'))
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',current);write(out/'noise_report.json',r)
    return dict(selected_sigma=r['selected_sigma'],noise_report=str(out/'noise_report.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--source-report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.config,a.source_report,a.output),indent=2))


if __name__=='__main__':main()
