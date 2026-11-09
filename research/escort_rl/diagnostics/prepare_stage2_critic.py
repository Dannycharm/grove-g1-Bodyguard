"""Prepare ONLY the critic from current-policy stochastic Stage 2 training rollouts.

No teacher, actor fitting, new reward, PPO learn(), ROS or DDS. Kept outside the existing
training source globs so original checkpoint runtime provenance stays strict/unchanged.
The fresh-optimizer imitation fit helper cannot be used here: the trained PPO optimizer
must be preserved, except explicitly resetting stale value-only Adam states after fitting.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import random
import time
from collections import Counter
import numpy as np
import torch
from escort_rl.contract import ROOT,PACKAGE,sha256
from escort_rl.imitation_ppo import returns,CRITIC_PREFIXES
from escort_rl.ppo_training import checkpoint
from escort_rl.stage2.env import Stage2Env
from escort_rl.stage2.scenarios import DEFAULT,FAMILIES,ENCOUNTERS,load_config
from escort_rl.stage2.training import Sampler,make_vec,load_stage2
from escort_rl.sb3_restart import restore_checkpoint_rng

DEFAULT_PREP=PACKAGE/'diagnostics/stage2_critic_preparation_v1.json'
DEFAULT_MODEL=PACKAGE/'runs/stage2_pilot_YfC5S4/pilot_5k/final.zip'
PROTOCOL='stage2_critic_only_preparation_v1'
REPLAY=('straight','gentle_turn','stop_start')


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def read_preparation(path):
    c=json.loads(Path(path).read_text())
    if c['protocol']!=PROTOCOL:raise ValueError('Unknown critic preparation protocol')
    for k in ('scenario_seed','action_seed','fit_seed','split_seed'):
        if type(c[k]) is not int or c[k]<0:raise ValueError('Nonnegative seed required: '+k)
    for k in ('episodes','updates','batch_size','score_interval'):
        if type(c[k]) is not int or c[k]<1:raise ValueError('Positive integer required: '+k)
    for k in ('learning_rate','max_grad_norm','max_replay_mse_ratio','max_replay_mse_increase_absolute'):
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError('Invalid positive preparation setting: '+k)
    if not 0<c['holdout_fraction']<.5 or not 0<c['minimum_near_intrusion_negative_fraction']<=1:raise ValueError('Invalid holdout/sign gate')
    return c


def protected_weights(model):
    return {k:v.detach().clone() for k,v in model.policy.state_dict().items() if not k.startswith(CRITIC_PREFIXES)}


def optimizer_actor_state(model):
    return {k:copy.deepcopy(model.policy.optimizer.state.get(p,{})) for k,p in model.policy.named_parameters() if not k.startswith(CRITIC_PREFIXES)}


def digest_state(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def distribution(model,observations):
    mean=[];std=[]
    with torch.no_grad():
        for start in range(0,len(observations),1024):
            x=torch.as_tensor(observations[start:start+1024],dtype=torch.float32)
            d=model.policy.get_distribution(x).distribution
            mean.append(d.mean.cpu().numpy());std.append(d.stddev.cpu().numpy())
    return np.concatenate(mean),np.concatenate(std)


def collect(model,stage2_path,c,out):
    """Fresh independent preparation RNGs; only current stochastic actor drives G1."""
    config=load_config(stage2_path);split=json.loads((Path(config['stage1_config']).parent/'stage1_splits.json').read_text())
    env=Sampler(Stage2Env(stage2_path),split,c['scenario_seed'],str(out/'preparation_scenarios.jsonl'))
    action_rng=np.random.default_rng(c['action_seed']);xs=[];ys=[];ids=[];families=[];near=[];episodes=[]
    physics=gait=warm_physics=warm_gait=0;begin=time.perf_counter()
    try:
        for ep in range(c['episodes']):
            obs,reset=env.reset();warm_physics+=reset['warmup_physics_steps'];warm_gait+=reset['warmup_gait_calls']
            observations=[];rewards=[]
            while True:
                observations.append(obs.copy());mu,sigma=distribution(model,obs[None,:])
                # Same Gaussian law as SB3, independent reproducible prep stream.
                action=np.clip(mu[0]+sigma[0]*action_rng.standard_normal(3),-1,1).astype(np.float32)
                obs,reward,term,trunc,info=env.step(action);rewards.append(float(reward));physics+=info['physics_steps'];gait+=info['gait_calls']
                if term or trunc:break
            bootstrap=0.
            if trunc and not term:
                with torch.no_grad():bootstrap=float(model.policy.predict_values(torch.as_tensor(obs[None,:])).item())
            family=env.current['family'];reason=info['reason'];n=len(observations)
            xs.extend(observations);ys.extend(returns(rewards,model.gamma,bootstrap));ids.extend([ep]*n);families.extend([family]*n)
            near.extend([family in ('crossing','diagonal') and reason=='pedestrian_intrusion' and j>=n-10 for j in range(n)])
            episodes.append(dict(episode_id=ep,episode_seed=env.current['episode_seed'],family=family,reason=reason,steps=n,terminated=bool(term),truncated=bool(trunc),bootstrap=bootstrap,return_total=sum(rewards),scenario=env.current))
            write(out/'episodes.json',episodes)
            print(f'critic collection {ep+1}/{c["episodes"]} {family}: {reason}',flush=True)
            if reason in ('fall','supervisor_failure','physics_failure'):raise RuntimeError('Critical dynamics failure; no prepared model will be published')
    finally:
        env.close()
        if xs:
            np.savez_compressed(out/'rollouts.npz',observations=np.asarray(xs,np.float32),returns=np.asarray(ys,np.float32),episode_ids=np.asarray(ids,np.int32),families=np.asarray(families),near_intrusion=np.asarray(near,bool))
        write(out/'collection.json',dict(completed_episodes=len(episodes),transitions=len(xs),active_physics_steps=physics,active_gait_calls=gait,warmup_physics_steps=warm_physics,warmup_gait_calls=warm_gait,wall_s=time.perf_counter()-begin,scenario_rng_state=env.rng_state(),action_rng_state=action_rng.bit_generator.state,families=dict(Counter(e['family'] for e in episodes)),outcomes=dict(Counter(e['reason'] for e in episodes))))
    return dict(observations=np.asarray(xs,np.float32),returns=np.asarray(ys,np.float32),episode_ids=np.asarray(ids,np.int32),families=np.asarray(families),near_intrusion=np.asarray(near,bool))


def split_episodes(data,c):
    rng=np.random.default_rng(c['split_seed']);hold_ids=[]
    for family in FAMILIES:
        unique=np.unique(data['episode_ids'][data['families']==family])
        if len(unique)<3:raise ValueError(f'Need >=3 complete episodes of {family} for stratified episode holdout; increase --episodes, do not silently drop a family')
        rng.shuffle(unique);n=max(1,min(len(unique)-2,int(np.ceil(len(unique)*c['holdout_fraction']))));hold_ids.extend(unique[:n].tolist())
    hold=np.isin(data['episode_ids'],hold_ids)
    assert not set(data['episode_ids'][hold]) & set(data['episode_ids'][~hold])
    return ~hold,hold


def value_predictions(model,x):
    with torch.no_grad():return np.concatenate([model.policy.predict_values(torch.as_tensor(x[i:i+1024])).cpu().numpy().ravel() for i in range(0,len(x),1024)])


def scores(model,data,train,hold):
    x=data['observations'];y=data['returns'];pred=value_predictions(model,x)
    def group(mask):
        target=y[mask];v=pred[mask]
        if not len(v):return dict(samples=0,mse=None,mean_value=None,mean_return=None,negative_fraction=None)
        return dict(samples=len(v),mse=float(np.mean((v-target)**2)),mae=float(np.mean(np.abs(v-target))),mean_value=float(np.mean(v)),mean_return=float(np.mean(target)),negative_fraction=float(np.mean(v<0)))
    result=dict(train=group(train),holdout=group(hold),per_family={f:group(hold&(data['families']==f)) for f in FAMILIES},near_intrusion=group(hold&data['near_intrusion']))
    result['holdout']['explained_variance']=float(1-np.var(y[hold]-pred[hold])/max(np.var(y[hold]),1e-12))
    result['constant_holdout_mse']=float(np.mean((y[hold]-np.mean(y[train]))**2))
    return result


def gate(score,initial,c):
    reasons=[]
    if score['holdout']['mse']>=min(initial['holdout']['mse'],score['constant_holdout_mse']):reasons.append('overall_holdout_not_better_than_parent_and_constant')
    for family in ('crossing','diagonal'):
        if score['per_family'][family]['mse']>=initial['per_family'][family]['mse']:reasons.append(f'{family}_holdout_not_improved')
    for family in REPLAY:
        limit=max(initial['per_family'][family]['mse']*c['max_replay_mse_ratio'],initial['per_family'][family]['mse']+c['max_replay_mse_increase_absolute'])
        if score['per_family'][family]['mse']>limit:reasons.append(f'{family}_value_regression')
    if not score['near_intrusion']['samples']:reasons.append('no_heldout_near_intrusion_samples')
    elif score['near_intrusion']['negative_fraction']<c['minimum_near_intrusion_negative_fraction']:reasons.append('near_intrusion_predictions_not_negative')
    return reasons


def fit(model,data,c):
    x=data['observations'];y=data['returns']
    if x.ndim!=2 or x.shape[1]!=40 or x.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Require finite, unchanged 40-float causal observations and returns')
    train,hold=split_episodes(data,c);protected=protected_weights(model);actor_opt=digest_state(optimizer_actor_state(model));param_groups=copy.deepcopy(model.policy.optimizer.state_dict()['param_groups'])
    before_mean,before_std=distribution(model,x);initial=scores(model,data,train,hold)
    params=[p for k,p in model.policy.named_parameters() if k.startswith(CRITIC_PREFIXES)]
    if not params:raise ValueError('No compatible value-only parameters')
    flags={k:p.requires_grad for k,p in model.policy.named_parameters()}
    for k,p in model.policy.named_parameters():p.requires_grad_(k.startswith(CRITIC_PREFIXES))
    model.policy.zero_grad(set_to_none=True)
    opt=torch.optim.Adam(params,lr=c['learning_rate']);rng=np.random.default_rng(c['fit_seed']);index=np.flatnonzero(train)
    history=[dict(update=0,metrics=initial)];best=None;best_score=float('inf');best_update=None;best_any=None;best_any_score=float('inf');start=time.perf_counter()
    try:
        for update in range(1,c['updates']+1):
            batch=rng.choice(index,c['batch_size'],replace=True);values=model.policy.predict_values(torch.as_tensor(x[batch])).ravel();loss=torch.mean((values-torch.as_tensor(y[batch]))**2)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,c['max_grad_norm']);opt.step()
            if update%c['score_interval']==0 or update==c['updates']:
                score=scores(model,data,train,hold);reasons=gate(score,initial,c);history.append(dict(update=update,metrics=score,gate_reasons=reasons))
                state={k:v.detach().clone() for k,v in model.policy.state_dict().items() if k.startswith(CRITIC_PREFIXES)}
                if score['holdout']['mse']<best_any_score:best_any_score=score['holdout']['mse'];best_any=state
                if not reasons and score['holdout']['mse']<best_score:best_score=score['holdout']['mse'];best=state;best_update=update
                print(f'critic fit {update}/{c["updates"]}: holdout MSE={score["holdout"]["mse"]:.3f} gate={reasons or "pass"}',flush=True)
        model.policy.load_state_dict(best if best is not None else best_any,strict=False)
    finally:
        for k,p in model.policy.named_parameters():p.requires_grad_(flags[k])
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in protected.items()),'Protected actor/variance/feature tensors changed'
    after_mean,after_std=distribution(model,x)
    assert np.array_equal(before_mean,after_mean) and np.array_equal(before_std,after_std),'Actor distribution changed'
    assert actor_opt==digest_state(optimizer_actor_state(model)) and param_groups==model.policy.optimizer.state_dict()['param_groups'],'PPO actor optimizer state changed'
    final=scores(model,data,train,hold)
    report=dict(protocol=PROTOCOL,accepted=best is not None,selected_update=best_update,critic_optimizer_updates=c['updates'],wall_s=time.perf_counter()-start,initial=initial,final=final,final_gate_reasons=gate(final,initial,c),history=history,
        train_episode_ids=np.unique(data['episode_ids'][train]).tolist(),holdout_episode_ids=np.unique(data['episode_ids'][hold]).tolist(),train_samples=int(train.sum()),holdout_samples=int(hold.sum()),protected_tensor_hash_before=digest_state(protected),protected_tensor_hash_after=digest_state(protected_weights(model)),actor_distribution_bitwise_unchanged=True,actor_optimizer_state_unchanged=True,selection='Minimum holdout MSE among checkpoints meeting all declared gates; no fixed policy validation or final-test data used')
    return report


def reset_value_optimizer(model):
    removed=[]
    for name,param in model.policy.named_parameters():
        if name.startswith(CRITIC_PREFIXES):model.policy.optimizer.state.pop(param,None);removed.append(name)
    return removed


def register(out,parent,report,metadata):
    path=(out/'prepared.zip').relative_to(ROOT).as_posix();base=out.relative_to(ROOT).as_posix();parent=Path(parent).resolve().relative_to(ROOT).as_posix()
    block=f'''\n## Stage 2 critic only preparation {base}\n\n- **Weights:** [{path}](../../{path}).\n- **SHA-256:** `{sha256(out/'prepared.zip')}`.\n- **Parent:** `{parent}`.\n- **PPO transitions:** {metadata['stage2_start_timesteps']+metadata['stage2_training_transitions']:,} cumulative, {metadata['stage2_training_transitions']:,} Stage 2; unchanged by preparation.\n- **Actor and variance:** bitwise unchanged; actor optimizer and training RNG streams preserved.\n- **Critic:** separately fitted to current stochastic actor's training-only returns; value-only Adam state reset.\n- **Report:** [fit_report.json](../../{base}/fit_report.json).\n- **Manifest and RNG:** [prepared.manifest.json](../../{base}/prepared.manifest.json), [prepared.rng.pt](../../{base}/prepared.rng.pt).\n- **Status:** critic holdout gates passed; no claim of improved actor success or Stage 2 promotion. Local artifact until deliberately archived.\n\n'''
    registry=ROOT/'docs/bodyguard/MODEL_WEIGHTS.md'
    with registry.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(block);f.flush();fcntl.flock(f,fcntl.LOCK_UN)


def prepare(checkpoint_path,stage2_path,prep_path,output,episodes=None,updates=None):
    c=read_preparation(prep_path)
    if episodes is not None:c['episodes']=episodes
    if updates is not None:c['updates']=updates
    if type(c['episodes']) is not int or c['episodes']<18 or type(c['updates']) is not int or c['updates']<1:raise ValueError('Require >=18 episodes and positive critic update count')
    out=Path(output).resolve();out.relative_to(ROOT);parent=Path(checkpoint_path).resolve();parent_hash=sha256(parent)
    old=json.loads(parent.with_suffix('.manifest.json').read_text())
    if old.get('stage')!='stage2' or old.get('protocol')!='stage2_single_pedestrian_transition_v1':raise ValueError('Require a genuine Stage 2 checkpoint with manifest/RNG sidecar')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    vec=make_vec(str(Path(stage2_path).resolve()),old['workers'],old['root_seed'],out)
    try:
        model,metadata=load_stage2(parent,stage2_path,vec);saved=torch.load(parent.with_suffix('.rng.pt'),weights_only=False)
        initial_counters=(model.num_timesteps,model._n_updates,getattr(model,'gradient_steps',0));was_training=model.policy.training;model.policy.set_training_mode(False)
        frozen_source=Path(__file__).read_bytes();(out/'preparation_source.py').write_bytes(frozen_source);source_digest=hashlib.sha256(frozen_source).hexdigest()
        write(out/'preparation_manifest.json',dict(protocol=PROTOCOL,parent_checkpoint=str(parent),parent_sha256=parent_hash,preparation=c,stage2_config=str(Path(stage2_path).resolve()),source_sha256=source_digest,preparation_config_sha256=sha256(prep_path),training_partitions='Original training residue 0 and reserved-combination exclusion; fixed validation/final tests never sampled',action_law='Current policy unsquashed Gaussian mean/std with Box clipping; independent NumPy action stream',training_rng='Parent training streams held unchanged; preparation streams independent'))
        data=collect(model,stage2_path,c,out);report=fit(model,data,c);write(out/'fit_report.json',report)
        if not report['accepted']:raise RuntimeError('Critic holdout gates failed. Reports/data saved; NO prepared checkpoint. Review fit_report.json before changing settings.')
        actor_before=digest_state(optimizer_actor_state(model));reset=reset_value_optimizer(model);assert actor_before==digest_state(optimizer_actor_state(model))
        assert initial_counters==(model.num_timesteps,model._n_updates,getattr(model,'gradient_steps',0))
        model.policy.set_training_mode(was_training)
        restore_checkpoint_rng(model,vec,saved)
        metadata=copy.deepcopy(metadata);metadata['critic_preparation']=dict(protocol=PROTOCOL,parent_checkpoint=str(parent),parent_sha256=parent_hash,source_sha256=source_digest,settings=c,data_sha256=sha256(out/'rollouts.npz'),report_sha256=sha256(out/'fit_report.json'),collection=json.loads((out/'collection.json').read_text()),actor_unchanged=True,actor_optimizer_preserved=True,value_optimizer_reset=reset,critic_optimizer_updates=c['updates'],selected_critic_update=report['selected_update'],ppo_counters_unchanged=True,training_rng_unchanged=True,additional_sample_budget='Critic collection is extra full-G1 interaction, separate from PPO num_timesteps')
        if sha256(parent)!=parent_hash:raise RuntimeError('Original checkpoint changed; do not publish prepared model')
        checkpoint(model,out/'prepared',metadata)
        register(out,parent,report,metadata)
        result=dict(status='accepted',checkpoint=str(out/'prepared.zip'),ppo_timesteps=model.num_timesteps,stage2_ppo_transitions=metadata['stage2_training_transitions'],critic_collection_transitions=len(data['returns']),critic_optimizer_updates=c['updates'],actor_changed=False,ppo_learn_called=False,registry_updated=True,initial_holdout_mse=report['initial']['holdout']['mse'],final_holdout_mse=report['final']['holdout']['mse'])
        write(out/'result.json',result);return result
    except (KeyboardInterrupt,RuntimeError,ValueError) as exc:
        write(out/'not_completed.json',dict(status='not_completed',reason=str(exc),original_checkpoint_untouched=sha256(parent)==parent_hash));raise
    finally:vec.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',default=str(DEFAULT_MODEL));p.add_argument('--config',default=str(DEFAULT));p.add_argument('--preparation-config',default=str(DEFAULT_PREP));p.add_argument('--output',required=True);p.add_argument('--episodes',type=int);p.add_argument('--updates',type=int)
    a=p.parse_args();print(json.dumps(prepare(a.checkpoint,a.config,a.preparation_config,a.output,a.episodes,a.updates),indent=2))


if __name__=='__main__':main()
