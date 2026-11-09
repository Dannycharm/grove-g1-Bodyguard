import sys,json,subprocess,copy
from pathlib import Path
import numpy as np,pytest,torch
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'diagnostics'))
import residual_ppo as m
from residual_ppo_policy import MaskedGaussian,ResidualActorCriticPolicy,ExecutedPolicy,execute,LATENT_LIMIT

SOURCE=lambda:str(m.ROOT/m.read_config()['source'])

def obs(n=4,active=True):
    x=np.zeros((n,40),np.float32);x[:,0]=1.5/16;x[:,5]=1;x[:,6]=-.5;x[:,8:10]=1
    if active:x[:,22:28]=[.08,.02,-.1,.05,.25,1]
    return x

class Toy(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(40,),dtype=np.float32)
    action_space=gym.spaces.Box(-LATENT_LIMIT,LATENT_LIMIT,(3,),dtype=np.float32)
    def __init__(self):self.stream=dict(rng=np.random.default_rng(13).bit_generator.state,episodes=0)
    def reset(self,seed=None,options=None):super().reset(seed=seed);return obs(1)[0],{}
    def step(self,a):return obs(1)[0],float(-np.sum((a-.1)**2)),False,True,{}
    def rng_state(self):return self.stream
    def set_rng_state(self,s):self.stream=s

def toy_model(c=None):
    c=copy.deepcopy(m.read_config() if c is None else c);c['ppo'].update(n_steps=16,batch_size=8,n_epochs=2)
    v=DummyVecEnv([Toy]);model=m.build(c,v);model.set_logger(configure(None,[]));return model,v,c

def test_masked_logprob_and_inactive_gradients_are_correct():
    mean=torch.tensor([[.2,.3,-.1],[.4,.5,.6]],requires_grad=True);std=torch.tensor([-4.,-4.,-4.]);mask=torch.tensor([True,False]);dist=MaskedGaussian(mask).proba_distribution(mean,std);action=dist.sample()
    assert torch.equal(action[1],torch.zeros(3));expected=dist.distribution.log_prob(action).sum(dim=1)
    assert dist.log_prob(action)[0]==expected[0] and dist.log_prob(action)[1]==0 and dist.entropy()[1]==0
    loss=dist.log_prob(torch.zeros(2,3)).sum();loss.backward();assert torch.equal(mean.grad[1],torch.zeros(3))

def test_import_and_stochastic_inactive_execution_are_exact():
    model,v,c=toy_model()
    try:
        adapter=ExecutedPolicy(model);x=obs();original=model.policy.context
        assert np.array_equal(adapter.predict(x)[0],original.predict(x)[0])
        x=obs(active=False)
        with torch.no_grad():model.policy.residual_net[-1].bias.fill_(5.)
        for _ in range(5):assert np.array_equal(adapter.predict(x,deterministic=False)[0],original.base_action(x))
        u,_=model.predict(x,deterministic=False);assert np.array_equal(u,np.zeros_like(u));model.policy.assert_frozen()
    finally:v.close()

def test_extreme_latents_remain_bounded_and_not_velocity_caps():
    model,v,c=toy_model()
    try:
        final,base,raw=execute(model.policy.context,obs(),np.full((4,3),1e6,np.float32));assert np.isfinite(final).all() and (np.abs(final)<=1).all()
        final0,_,_=execute(model.policy.context,obs(active=False),np.full((4,3),1e6,np.float32));assert np.array_equal(final0,model.policy.context.base_action(obs(active=False)))
    finally:v.close()

def test_optimizer_updates_residual_and_critic_but_not_base_or_std():
    model,v,c=toy_model()
    try:
        before={k:p.detach().clone() for k,p in model.policy.residual_net.state_dict().items()};std=model.policy.log_std.detach().clone();model.learn(32)
        assert model.num_timesteps==32 and model.gradient_steps>0
        assert any(not torch.equal(p,model.policy.residual_net.state_dict()[k]) for k,p in before.items())
        assert torch.equal(std,model.policy.log_std);model.policy.assert_frozen()
    finally:v.close()

def test_calibration_changes_only_critic_and_uses_whole_episode_holdout():
    model,v,c=toy_model();c['critic'].update(updates=400,batch_size=64,learning_rate=.003)
    try:
        x=np.repeat(obs(24),4,axis=0);x[:,0]+=np.linspace(-.02,.02,len(x));y=(4+x[:,0]*10).astype(np.float32);ids=np.repeat(np.arange(24),4)
        before={k:p.clone() for k,p in model.policy.residual_net.state_dict().items()};report=m.fit_critic(model,x,y,ids,c)
        assert report['accepted'] and report['holdout_episode_ids']==[0,4,8,12,16,20]
        assert not model.policy.optimizer.state
        assert all(torch.equal(p,model.policy.residual_net.state_dict()[k]) for k,p in before.items());model.policy.assert_frozen()
    finally:v.close()

def test_save_reload_fresh_process_and_rng_restore(tmp_path):
    model,v,c=toy_model()
    try:
        model.learn(32);path=tmp_path/'model';m.save(model,path,m.provenance(c));loaded,record=m.checked_load(path,c,v)
        assert loaded.num_timesteps==32 and loaded.policy.optimizer.state
        assert np.array_equal(model.predict(obs(),deterministic=True)[0],loaded.predict(obs(),deterministic=True)[0])
        code="import sys,json,numpy as np;sys.path.insert(0,'research/escort_rl/diagnostics');import residual_ppo as m;c=json.load(open(sys.argv[1]+'.manifest.json'))['config'];p,_=m.checked_load(sys.argv[1],c);print('RELOADED',p.num_timesteps,p.policy.residual_net[0].in_features)"
        proc=subprocess.run([sys.executable,'-c',code,str(path)],cwd=m.ROOT,text=True,capture_output=True,check=True);assert 'RELOADED 32 51' in proc.stdout
        with path.with_suffix('.zip').open('ab') as f:f.write(b'tamper')
        with pytest.raises(ValueError,match='hash'):m.checked_load(path,c)
    finally:v.close()

def test_missing_rng_is_not_silently_reseeded(tmp_path):
    model,v,c=toy_model()
    try:
        path=tmp_path/'model';m.save(model,path,m.provenance(c));Path(str(path)+'.rng.pt').unlink()
        with pytest.raises(FileNotFoundError):m.checked_load(path,c,v)
    finally:v.close()

def test_budget_rejected_before_collection_or_output(tmp_path):
    out=tmp_path/'should_not_exist'
    with pytest.raises(ValueError):m.run(out,timesteps=100000)
    assert not out.exists()

def test_real_full_g1_optimizer_smoke_and_scenario_rng(tmp_path):
    c=m.read_config();c['ppo'].update(n_steps=32,batch_size=16,n_epochs=2);torch.set_num_threads(1);v=m.make_vec(c,tmp_path)
    try:
        before=v.env_method('rng_state');model=m.build(c,v);assert v.env_method('rng_state')==before
        model.set_logger(configure(None,[]))
        class Counters(BaseCallback):
            physics=0;gait=0;contacts=0
            def _on_step(self):
                for info in self.locals['infos']:
                    self.physics+=info['physics_steps'];self.gait+=info['gait_calls'];self.contacts+=info['contacts']
                    assert np.max(np.abs(info['ppo_executed_action']))<=1
                return True
        cb=Counters();logging=m.PilotCallback(tmp_path,c,m.provenance(c))
        try:m.exact_learn(model,64,[cb,logging])
        finally:logging.close()
        rows=[json.loads(line) for line in (tmp_path/'training_steps.jsonl').read_text().splitlines()];assert len(rows)==64
        assert model.num_timesteps==64 and model.gradient_steps>0 and cb.physics==6400 and cb.gait==640 and cb.contacts>0
        model.policy.assert_frozen()
        path=tmp_path/'real_checkpoint';m.save(model,path,m.provenance(c));other_dir=tmp_path/'resumed';other_dir.mkdir();other=m.make_vec(c,other_dir)
        try:
            loaded,_=m.checked_load(path,c,other)
            assert other.env_method('rng_state')==v.env_method('rng_state')
            v.reset();other.reset()
            assert v.reset_infos[0]['episode_seed']==other.reset_infos[0]['episode_seed']
            assert loaded.num_timesteps==64
        finally:other.close()
    finally:v.close()

def test_inactive_ppo_samples_do_not_update_residual_actor():
    class NoPed(Toy):
        def reset(self,seed=None,options=None):return obs(1,active=False)[0],{}
        def step(self,a):return obs(1,active=False)[0],1.,False,True,{}
    c=m.read_config();c['ppo'].update(n_steps=16,batch_size=8,n_epochs=2);v=DummyVecEnv([NoPed]);model=m.build(c,v);model.set_logger(configure(None,[]))
    try:
        before={k:p.clone() for k,p in model.policy.residual_net.state_dict().items()};model.learn(32)
        assert model.gradient_steps>0
        assert all(torch.equal(p,model.policy.residual_net.state_dict()[k]) for k,p in before.items())
        model.policy.assert_frozen()
    finally:v.close()

def test_two_spawn_workers_keep_independent_scenario_rng(tmp_path):
    c=m.read_config();c['workers']=2;v=m.make_vec(c,tmp_path)
    try:
        before=v.env_method('rng_state');assert before[0]!=before[1];model=m.build(c,v);assert v.env_method('rng_state')==before
        observations=v.reset();actions,_=model.predict(observations,deterministic=True);new,rewards,dones,infos=v.step(actions)
        assert new.shape==(2,40) and all(i['physics_steps']==100 and i['gait_calls']==10 for i in infos)
        assert v.env_method('rng_state')[0]!=v.env_method('rng_state')[1]
    finally:v.close()

def test_validation_selection_prioritizes_safety_over_success_and_reward():
    ped=dict(reasons={},geometric_intrusions=0,success_rate=.8,per_family={'crossing':{'success_rate':.8}},formation_mean_m=.2)
    stage1=dict(reasons={},success_rate=1.,formation_mean_m=.05)
    unsafe=copy.deepcopy(ped);unsafe.update(geometric_intrusions=1,success_rate=1.)
    assert m.validation_key(ped,stage1)>m.validation_key(unsafe,stage1)
    critical=copy.deepcopy(ped);critical['reasons']={'fall':1}
    assert m.validation_key(unsafe,stage1)>m.validation_key(critical,stage1)


def test_logging_fix_migration_only_accepts_known_prepared_zero_update_source(tmp_path):
    model,v,c=toy_model()
    try:
        record=m.provenance(c);key=str(Path(m.__file__).resolve().relative_to(m.ROOT))
        record['source_hashes'][key]=m.PRE_LOGGING_FIX_DRIVER_SHA256
        record['preparation']={'fit':{'accepted':True}}
        path=tmp_path/'prepared';m.save(model,path,record)
        loaded,meta=m.checked_load(path,c)
        assert loaded.num_timesteps==0 and 'logging_fix_migration' in meta
        manifest=path.with_suffix('.manifest.json');bad=json.loads(manifest.read_text());bad['source_hashes'][key]='0'*64;manifest.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='source_hashes'):m.checked_load(path,c)
        m.save(model,tmp_path/'prepared_second',record)
        manifest=tmp_path/'prepared_second.manifest.json';bad=json.loads(manifest.read_text());bad['num_timesteps']=1;manifest.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match='source_hashes'):m.checked_load(tmp_path/'prepared_second',c)
    finally:v.close()

def test_json_numpy_rejects_unknown_objects():
    assert json.loads(json.dumps({'array':np.array([1.,2.]),'flag':np.bool_(True),'n':np.int64(3)},default=m.json_numpy))=={'array':[1.,2.],'flag':True,'n':3}
    with pytest.raises(TypeError):json.dumps(object(),default=m.json_numpy)
