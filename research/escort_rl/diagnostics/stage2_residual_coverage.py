"""Versioned v1 residual continuation with broader training-only encounter coverage.
Reuse unchanged residual inference, frozen base, teacher, dynamics and evaluation.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import yaml
from escort_rl.contract import ROOT,PACKAGE,sha256
from escort_rl.stage2.scenarios import DEFAULT,ENCOUNTERS,load_config,sample_episode
from escort_rl.ppo_evaluation import reserved
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stage2_residual as core

CONFIG=Path(__file__).with_name('stage2_residual_coverage_v2.yaml')
PROTOCOL='frozen_stage1_residual_coverage_v2'
VARIATION_PROTOCOL='frozen_stage1_residual_variation_v3'

def read_config(path=CONFIG):
    c=yaml.safe_load(Path(path).read_text())
    if c['protocol'] not in (PROTOCOL,VARIATION_PROTOCOL) or not all(c[k] is True for k in ('balance_families','balance_sides','balance_cases')):raise ValueError('Unknown coverage/balance protocol')
    for k in ('iterations','updates_per_iteration','batch_size','new_cases_per_family_per_side','family_seed_stride'):
        if type(c[k]) is not int or c[k]<1:raise ValueError('Invalid integer: '+k)
    if type(c['candidate_seed_start']) is not int or c['candidate_seed_start']<0 or any(not np.isfinite(c[k]) or c[k]<=0 for k in ('learning_rate','max_grad_norm')):raise ValueError('Invalid seed or optimizer setting')
    if c['candidate_seed_start']%4 or c['family_seed_stride']%4 or c['teacher_execution_probability']!=0:raise ValueError('Require training partition and student-only DAgger')
    if c['protocol']==VARIATION_PROTOCOL:
        cells=c.get('variation_cells',[])
        if not cells or len({cell['name'] for cell in cells})!=len(cells):raise ValueError('Distinct variation cells required')
        for cell in cells:
            if cell['family'] not in ('crossing','diagonal') or cell['side'] not in (-1,1):raise ValueError('Invalid variation family/side')
            for k in ('speed_m_s','encounter_fraction','angle_degrees'):
                if k in cell and (len(cell[k])!=2 or not np.isfinite(cell[k]).all() or cell[k][0]>=cell[k][1]):raise ValueError('Invalid variation range: '+k)
    return c

def resolve_case(family,seed,sc,split):
    if family not in ENCOUNTERS or type(seed) is not int or seed%4:raise ValueError('Encounter training partition required')
    meta=sample_episode(seed,family,sc).scenario_metadata()
    if reserved(meta,split):raise ValueError('Reserved final-test parameter combination')
    return dict(family=family,episode_seed=seed,side=int(meta['encounter']['side']),metadata=meta)

def select_cases(c):
    sc=load_config();split=json.loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text());out=[];seen=set();rejections=[]
    for case in c['retain_cases']:
        r=resolve_case(case['family'],case['episode_seed'],sc,split);key=(r['family'],r['episode_seed'])
        if key in seen:raise ValueError('Duplicate retained case')
        seen.add(key);r['selection']='retained_development_anchor';out.append(r)
    if c['protocol']==VARIATION_PROTOCOL:
        # Choose BEFORE seeing outcomes; no failed validation IDs/labels are used.
        for cell in c['variation_cells']:
            j=ENCOUNTERS.index(cell['family']);seed=c['candidate_seed_start']+j*c['family_seed_stride']
            for _ in range(10000):
                candidate=seed;seed+=4
                if (cell['family'],candidate) in seen:continue
                try:r=resolve_case(cell['family'],candidate,sc,split)
                except ValueError as exc:rejections.append(dict(family=cell['family'],episode_seed=candidate,reason=str(exc)));continue
                if not matches_cell(r,cell):continue
                seen.add((r['family'],candidate));r.update(selection='new_training_variation',variation_cell=cell['name']);out.append(r);break
            else:raise RuntimeError('Cannot resolve variation cell: '+cell['name'])
        return out,rejections
    for j,family in enumerate(ENCOUNTERS):
        count={-1:0,1:0};seed=c['candidate_seed_start']+j*c['family_seed_stride']
        for _ in range(10000):
            candidate=seed;seed+=4
            if (family,candidate) in seen:continue
            try:r=resolve_case(family,candidate,sc,split)
            except ValueError as exc:rejections.append(dict(family=family,episode_seed=candidate,reason=str(exc)));continue
            side=r['side']
            if count[side]>=c['new_cases_per_family_per_side']:continue
            count[side]+=1;seen.add((family,candidate));r['selection']='new_side_balanced_development';out.append(r)
            if min(count.values())>=c['new_cases_per_family_per_side']:break
        else:raise RuntimeError('Cannot resolve side coverage for '+family)
    return out,rejections

def matches_cell(case,cell):
    details=case['metadata']['encounter']
    values=dict(speed_m_s=details['peak_speed_m_s'],encounter_fraction=details['encounter_fraction'],angle_degrees=abs(float(np.rad2deg(details['approach_angle_rad']))))
    return case['family']==cell['family'] and case['side']==cell['side'] and all(cell[k][0]<=values[k]<cell[k][1] for k in values if k in cell)

def restore_parent(c):
    path=ROOT/c['parent_checkpoint']
    if sha256(path)!=c['parent_sha256']:raise ValueError('Coverage parent SHA mismatch')
    policy,manifest=core.load(path)  # strict v1 source/runtime/base/teacher checks unchanged
    saved=torch.load(path,weights_only=False)
    if saved.get('optimizer') is None or saved.get('numpy_rng') is None:raise ValueError('Parent lacks optimizer/RNG; refusing fresh initialization')
    opt=torch.optim.Adam(policy.net.parameters(),lr=c['learning_rate']);opt.load_state_dict(saved['optimizer'])
    if any(group['lr']!=c['learning_rate'] for group in opt.param_groups):raise ValueError('Parent/config learning-rate mismatch')
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    return policy,opt,rng,manifest

def seed_descriptor(rng):return hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()

def verified_parent_sources(parent_manifest):
    sources=parent_manifest.get('extra',{}).get('data')
    if not sources:raise ValueError('Parent data provenance missing')
    paths=[]
    for source in sources:
        path=Path(source['path']);path=path if path.is_absolute() else ROOT/path
        if sha256(path/'labels.npz')!=source['sha256']:raise ValueError('Parent label hash mismatch: '+str(path))
        if 'summary_sha256' in source and sha256(path/'summary.json')!=source['summary_sha256']:raise ValueError('Parent summary hash mismatch: '+str(path))
        paths.append(path)
    return paths

def data_arrays(paths,phase_weights):
    xx=[];yy=[];phases=[];families=[];sides=[];keys=[];provenance=[]
    sc=load_config();split=json.loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text())
    for path in paths:
        path=Path(path);summary=json.loads((path/'summary.json').read_text());rows=summary['episodes_detail']
        with np.load(path/'labels.npz',allow_pickle=False) as d:x=d['observations'].copy();y=d['teacher_actions'].copy();ids=d['episode_ids'].copy();p=d['phases'].copy()
        if x.dtype!=np.float32 or y.dtype!=np.float32 or x.shape!=(len(ids),40) or y.shape!=(len(ids),3) or p.shape!=(len(ids),) or not np.issubdtype(ids.dtype,np.integer) or not np.isfinite(x).all() or not np.isfinite(y).all() or np.max(np.abs(y),initial=0)>1:raise ValueError('Invalid residual labels')
        cases=[resolve_case(r['family'],int(r['episode_seed']),sc,split) for r in rows]
        for ep,phase in zip(ids,p):
            if not 0<=ep<len(rows) or str(phase) not in phase_weights:raise ValueError('Invalid episode/phase reference')
            r=cases[int(ep)];families.append(r['family']);sides.append(r['side']);keys.append((r['family'],r['episode_seed']))
        xx.append(x);yy.append(y);phases.extend(str(v) for v in p);provenance.append(dict(path=str(path.resolve()),sha256=sha256(path/'labels.npz'),summary_sha256=sha256(path/'summary.json')))
    return np.concatenate(xx),np.concatenate(yy),np.asarray(phases),np.asarray(families),np.asarray(sides),keys,provenance

def sampling_probabilities(phases,families,sides,keys,phase_weights):
    """Equal family, side and CASE mass; phase priorities within each case."""
    weights=np.zeros(len(phases),float)
    if set(families)!=set(ENCOUNTERS):raise ValueError('All encounter families required')
    for family in ENCOUNTERS:
        for side in (-1,1):
            mask=(families==family)&(sides==side);case_keys=sorted({keys[i] for i in np.flatnonzero(mask)})
            if not case_keys:raise ValueError('Missing approach side: '+family)
            for key in case_keys:
                ii=np.array([i for i in np.flatnonzero(mask) if keys[i]==key]);w=np.array([phase_weights[str(phases[i])] for i in ii],float)
                if not np.isfinite(w).all() or np.min(w)<=0:raise ValueError('Invalid positive phase weight')
                weights[ii]=w/w.sum()/len(case_keys)/6.
    if not np.isclose(weights.sum(),1):raise RuntimeError('Invalid sampling mass')
    return weights

def fit_iteration(policy,optimizer,rng,paths,c):
    x,y,ph,fs,ss,keys,sources=data_arrays(paths,policy.c['phase_weights']);prob=sampling_probabilities(ph,fs,ss,keys,policy.c['phase_weights']);base=policy.base_action(x);z=core.features(x,base,policy.bc,policy.c)
    initial=policy.predict(x)[0];history=[];start=time.perf_counter();rng_before=seed_descriptor(rng)
    for i in range(c['updates_per_iteration']):
        ii=rng.choice(len(x),c['batch_size'],replace=True,p=prob);delta=policy.residual_tensor(torch.as_tensor(z[ii]));loss=core.fitting_loss(torch.as_tensor(base[ii]),delta,torch.as_tensor(y[ii]),policy.c).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.net.parameters(),c['max_grad_norm']);optimizer.step();policy.updates+=1
        if (i+1)%500==0:history.append(dict(update=policy.updates,loss=float(loss.detach())))
    final=policy.predict(x)[0];policy.assert_frozen()
    groups={f:{str(side):dict(rows=int(np.sum((fs==f)&(ss==side))),sampling_mass=float(prob[(fs==f)&(ss==side)].sum())) for side in (-1,1)} for f in ENCOUNTERS}
    return dict(rows=len(x),sources=sources,updates=policy.updates,new_updates=c['updates_per_iteration'],fit_wall_s=time.perf_counter()-start,history=history,initial_action_mse=float(np.mean((initial-y)**2)),final_action_mse=float(np.mean((final-y)**2)),sampling=groups,batch_rng_before=rng_before,batch_rng_after=seed_descriptor(rng))

def train(output,cpath=CONFIG):
    c=read_config(cpath);out=Path(output).resolve();out.relative_to(ROOT);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    policy,opt,rng,parent=restore_parent(c);paths=verified_parent_sources(parent);cases,rejections=select_cases(c);initial_updates=policy.updates;start=time.perf_counter()
    record=dict(protocol=c['protocol'],config=c,parent_checkpoint=c['parent_checkpoint'],parent_sha256=c['parent_sha256'],parent_manifest_sha256=sha256(str(ROOT/c['parent_checkpoint'])+'.manifest.json'),source_sha256=sha256(__file__),inference_source_sha256=sha256(core.__file__),parent_optimizer_restored=True,parent_rng_restored=True,initial_rng_fingerprint=seed_descriptor(rng),initial_updates=initial_updates,case_rejections=rejections,stage2_contract=parent['stage2_contract'])
    core.write(out/'continuation_manifest.json',record);core.write(out/'cases.json',cases)
    # Archive exact source/config; old v1 inference source is never changed.
    (out/'training_source_snapshot.py').write_bytes(Path(__file__).read_bytes());(out/'experiment_snapshot.yaml').write_bytes(Path(cpath).read_bytes())
    result=core.rollout(policy,cases,out/'teacher_coverage',teacher_execution=True,collect=True)
    if result['success_rate']<1:core.write(out/'result.json',dict(status='teacher_coverage_failed',teacher=result,weights_updated=False));return
    paths.append(out/'teacher_coverage');before=core.rollout(policy,cases,out/'initial_coverage');rounds=[];fit_wall=0.
    for iteration in range(c['iterations']):
        report=fit_iteration(policy,opt,rng,paths,c);fit_wall+=report['fit_wall_s'];core.write(out/f'fit_{iteration}.json',report)
        measured=core.rollout(policy,cases,out/f'student_{iteration}');rounds.append(measured)
        checkpoint=out/f'residual_coverage_{iteration}.pt';extra=dict(training_protocol=c['protocol'],continuation_manifest_sha256=sha256(out/'continuation_manifest.json'),parent_checkpoint=c['parent_checkpoint'],parent_sha256=c['parent_sha256'],coverage_training_source_sha256=sha256(__file__),coverage_config=c,iteration=iteration,data=report['sources'],closed_loop=measured)
        core.save(policy,checkpoint,opt,rng,extra);core.register(policy,checkpoint,f'coverage v2 round {iteration}, {measured["success_rate"]} development success; fixed gates pending')
        if iteration+1<c['iterations']:
            folder=out/f'dagger_{iteration}';core.rollout(policy,cases,folder,collect=True,beta=0.,iteration=iteration+10);paths.append(folder)
    report=dict(status='coverage_complete',parent_updates=initial_updates,new_supervised_updates=policy.updates-initial_updates,total_supervised_updates=policy.updates,initial_coverage=before,rounds=rounds,final_checkpoint=str(checkpoint),wall_s=time.perf_counter()-start,fit_wall_s=fit_wall,base_frozen=True,ppo_updates=0,validation_status='Not automatically promoted; evaluate unchanged fixed suites next')
    core.write(out/'result.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['cases','train']);p.add_argument('--output',required=True);p.add_argument('--experiment',default=str(CONFIG));a=p.parse_args()
    if a.mode=='cases':
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases,rejections=select_cases(read_config(a.experiment));core.write(out/'cases.json',cases);core.write(out/'rejections.json',rejections);print(json.dumps([dict(family=r['family'],episode_seed=r['episode_seed'],side=r['side'],selection=r['selection']) for r in cases],indent=2))
    else:
        result=train(a.output,a.experiment)
        if result is not None:print(json.dumps({k:v for k,v in result.items() if k not in ('rounds','initial_coverage')},indent=2))
if __name__=='__main__':main()
