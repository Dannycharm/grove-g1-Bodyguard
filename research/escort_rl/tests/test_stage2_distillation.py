import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest,torch
import gymnasium as gym
from stable_baselines3 import PPO
p=Path(__file__).resolve().parents[1]/'diagnostics/stage2_distillation.py'
spec=importlib.util.spec_from_file_location('stage2_distillation',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(40,),dtype=np.float32)
    action_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32)
    def reset(self,seed=None,options=None):super().reset(seed=seed);return np.zeros(40,np.float32),{}
    def step(self,a):return np.zeros(40,np.float32),0.,False,True,{}

def synthetic():
    c=m.read_config(m.CONFIG);c['updates']=100;rng=np.random.default_rng(17);x=rng.uniform(-.1,.1,(240,40)).astype(np.float32);y=np.zeros((240,3),np.float32);ids=np.repeat(np.arange(24),10);fs=np.repeat(m.FAMILIES,40)
    active=np.tile(np.arange(40)%5==0,6);y[active,1]=.6;train=np.flatnonzero(ids%4!=3);hold=np.flatnonzero(ids%4==3)
    return (x,y,ids,fs,train,hold,[]),active,c

def test_stratified_batch_really_balances_rare_events():
    data,active,c=synthetic();g=m.pools(data[4],data[3],active);ii=m.sample_batch(g,np.random.default_rng(0),32)
    assert len(ii)==192
    for f in m.FAMILIES:
        a=ii[data[3][ii]==f];assert len(a)==32
        if f in m.ENCOUNTERS:assert active[a].sum()==16
    assert not set(ii)&set(data[5])
    with pytest.raises(ValueError):m.pools(data[4],data[3],np.zeros(len(active),bool))

def test_metrics_expose_rare_errors_instead_of_hiding_them():
    data,active,c=synthetic();x,y,ids,fs,train,hold,_=data;pred=y.copy();pred[active]=0
    report=m.metrics(pred,y,hold,fs,active,np.array([.6,.6,.9]))
    row=report['per_family']['crossing'];assert row['intervention']['mse']>row['all']['mse']>row['ordinary']['mse']
    assert row['intervention']['command_mae'][1]==pytest.approx(.36)

def test_actor_only_fit_preserves_trained_critic_variance_and_optimizer(monkeypatch):
    torch.set_num_threads(1);model=PPO('MlpPolicy',Toy(),policy_kwargs=dict(net_arch=dict(pi=[128,128],vf=[128,128])),seed=42,n_steps=8,batch_size=8)
    # Populate genuine Adam moments for actor, critic and log sigma.
    loss=sum((p**2).sum() for p in model.policy.parameters());loss.backward();model.policy.optimizer.step();model.policy.optimizer.zero_grad()
    before={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    states={n:{k:v.clone() if torch.is_tensor(v) else v for k,v in model.policy.optimizer.state.get(p,{}).items()} for n,p in model.policy.named_parameters()}
    data,active,c=synthetic();monkeypatch.setattr(m,'intervention_mask',lambda *args:active)
    report=m.fit_actor(model,data,c,dict(environment=dict(command_caps=[.6,.6,.9])))
    assert report['critic_and_variance_unchanged'] and model.num_timesteps==0
    assert not set(report['train_episode_ids'])&set(report['holdout_episode_ids'])
    for n,v in before.items():
        if not n.startswith(m.ACTOR_PREFIXES):assert torch.equal(v,model.policy.state_dict()[n])
    for n,p in model.policy.named_parameters():
        if n.startswith(m.ACTOR_PREFIXES):assert p not in model.policy.optimizer.state
        else:
            for k,v in states[n].items():assert torch.equal(v,model.policy.optimizer.state[p][k]) if torch.is_tensor(v) else v==model.policy.optimizer.state[p][k]

def candidate_fixture(folder,monkeypatch):
    folder.mkdir();model=PPO('MlpPolicy',Toy(),seed=42,n_steps=8,batch_size=8);parent=folder/'parent.zip';model.save(parent)
    np.savez_compressed(folder/'candidate_actor.npz',**{n:v.detach().numpy() for n,v in model.policy.state_dict().items() if n.startswith(m.ACTOR_PREFIXES)})
    m.write(folder/'candidate.json',dict(protocol=m.read_config(m.CONFIG)['protocol'],diagnostic_only=True,parent=str(parent),parent_sha256=m.sha256(parent),stage2_contract={},actor_sha256=m.sha256(folder/'candidate_actor.npz')))
    monkeypatch.setattr(m,'manifest',lambda c:{})
    monkeypatch.setattr(m,'load_stage2',lambda *args:(PPO.load(parent),{}))
    return folder

def test_parameter_only_candidate_reload_and_tamper_rejection(tmp_path,monkeypatch):
    folder=candidate_fixture(tmp_path/'candidate',monkeypatch);model,_=m.load_student(folder,m.DEFAULT)
    assert model.predict(np.zeros(40,np.float32),deterministic=True)[0].shape==(3,)
    with (folder/'candidate_actor.npz').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError):m.load_student(folder,m.DEFAULT)

@pytest.mark.parametrize('beta',[0.,1.])
def test_dagger_logs_execution_and_keeps_failed_student_labels(tmp_path,monkeypatch,beta):
    folder=candidate_fixture(tmp_path/'candidate',monkeypatch)
    class Env:
        config={}
        def reset(self,options):self.n=0;return np.zeros(40,np.float32),{}
        def step(self,a):
            self.n+=1;return np.zeros(40,np.float32),0.,True,False,dict(reason='pedestrian_intrusion',elapsed_s=.2,evaluation=dict(formation_error_m=.1))
        def close(self):pass
    class Teacher:
        def __init__(self,c):pass
        def act(self,x):return np.array([-.5,.3,0],np.float32)
    monkeypatch.setattr(m,'Stage2Env',lambda *a:Env());monkeypatch.setattr(m,'Teacher',Teacher)
    out=tmp_path/'labels';result=m.dagger(folder,out,count=3,beta=beta)
    assert result['episodes']==18 and result['transitions']==18
    assert result['teacher_executed_steps']==(18 if beta else 0)
    assert all(r['reason']=='pedestrian_intrusion' and r['episode_seed']%4==0 for r in json.loads((out/'episodes.json').read_text()))
    with np.load(out/'demonstrations.npz') as d:assert np.allclose(d['actions'],[-.5,.3,0])
    trace=json.loads((out/'trace_000.jsonl').read_text());assert trace['teacher_executed']==bool(beta)
    assert (out/'summary.json').exists()

def test_append_preserves_original_holdout_and_rejects_duplicate_or_tampered_data(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'manifest',lambda c:{})
    monkeypatch.setattr(m,'verify_teacher',lambda *args:'fixture')
    def dataset(path,seed):
        path.mkdir();data,active,c=synthetic();x,y,ids,fs,*_=data
        rows=[dict(family=f,episode_seed=seed+4*i,success=True,reason='success',scenario=dict(peak_speed_limit_m_s=.2,route_length_m=3.)) for i,f in enumerate(np.repeat(m.FAMILIES,4))]
        m.write(path/'manifest.json',dict(protocol='stage2_causal_teacher_bootstrap_v2',stage2_contract={}))
        m.write(path/'episodes.json',rows)
        np.savez_compressed(path/'demonstrations.npz',observations=x,actions=y,episode_ids=ids,families=fs)
        m.write(path/'summary.json',dict(data_sha256=m.sha256(path/'demonstrations.npz'),episodes_sha256=m.sha256(path/'episodes.json')))
        return path
    a=dataset(tmp_path/'a',892000);b=dataset(tmp_path/'b',893000)
    first=m.load_data([a],m.DEFAULT);both=m.load_data([a,b],m.DEFAULT)
    assert np.array_equal(first[5],both[5][both[5]<len(first[0])])
    assert not set(both[2][both[4]])&set(both[2][both[5]])
    with pytest.raises(ValueError,match='duplicate'):m.load_data([a,a],m.DEFAULT)
    with (b/'demonstrations.npz').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='integrity'):m.load_data([b],m.DEFAULT)

def test_config_rejects_invalid_mixture(tmp_path):
    c=m.read_config(m.CONFIG);c['intervention_fraction']=.9;p=tmp_path/'bad.json';m.write(p,c)
    with pytest.raises(ValueError):m.read_config(p)

def test_archived_teacher_compatibility_requires_exact_class_and_parameters(tmp_path):
    current=Path(m.__file__).with_name('stage2_causal_teacher.py').read_text()
    snapshot=tmp_path/'teacher_source_snapshot.py';snapshot.write_text(current+'\n# older CLI snapshot\n')
    record=dict(parameters=m.PARAMETERS,teacher_sha256=m.sha256(snapshot))
    assert m.verify_teacher(record,tmp_path)=='exact_archived_source_and_identical_teacher_class'
    snapshot.write_text(current.replace('self.previous=np.zeros(2)','self.previous=np.ones(2)'))
    record['teacher_sha256']=m.sha256(snapshot)
    with pytest.raises(ValueError,match='Teacher source'):m.verify_teacher(record,tmp_path)
    record['parameters']={}
    with pytest.raises(ValueError,match='parameter'):m.verify_teacher(record,tmp_path)

@pytest.mark.parametrize('passing',[False,True])
def test_fit_saves_diagnostic_actor_on_failure_and_prepared_only_on_pass(tmp_path,monkeypatch,passing):
    from stable_baselines3.common.vec_env import DummyVecEnv
    class SeedToy(Toy):
        def __init__(self):self.stream=dict(rng=np.random.default_rng(3).bit_generator.state,episodes=0)
        def rng_state(self):return self.stream
        def set_rng_state(self,s):self.stream=s
    vec=DummyVecEnv([SeedToy]);model=PPO('MlpPolicy',vec,seed=42,n_steps=8,batch_size=8);model.num_timesteps=5000
    parent=tmp_path/'parent';prior=dict(workers=1,root_seed=42,parent_stage1_validation=dict(formation_mean_m=.05))
    m.checkpoint(model,parent,prior);vec.close()
    data,active,c=synthetic()
    monkeypatch.setattr(m,'ROOT',tmp_path)
    monkeypatch.setattr(m,'load_data',lambda *args:data)
    monkeypatch.setattr(m,'intervention_mask',lambda *args:active)
    monkeypatch.setattr(m,'manifest',lambda *args:{})
    monkeypatch.setattr(m,'make_vec',lambda *args:DummyVecEnv([SeedToy]))
    monkeypatch.setattr(m,'load_stage2',lambda path,config,vec=None:(PPO.load(path,env=vec,force_reset=True),prior))
    monkeypatch.setattr(m,'evaluate',lambda *args,**kw:{})
    monkeypatch.setattr(m,'gate',lambda *args:dict(round_pass=passing,reasons=[] if passing else ['pedestrian_success']))
    registered=[];monkeypatch.setattr(m,'register',lambda *args:registered.append(True))
    out=tmp_path/'fit';result=m.fit(['fixture'],out,parent.with_suffix('.zip'),updates=100)
    assert result['saved_prepared']==passing and result['ppo_timesteps']==5000
    assert (out/'candidate_actor.npz').exists() and bool(registered)
    assert (out/'prepared.zip').exists()==passing
    loaded,meta=m.load_student(out,m.DEFAULT);assert loaded.num_timesteps==5000
    assert np.isfinite(loaded.predict(np.zeros(40,np.float32),deterministic=True)[0]).all()
    if passing:
        state=torch.load(out/'prepared.rng.pt',weights_only=False)
        old=torch.load(str(parent)+'.rng.pt',weights_only=False)
        assert torch.equal(state['torch'],old['torch']) and state['environment_streams']==old['environment_streams']
        assert PPO.load(out/'prepared.zip').num_timesteps==5000
