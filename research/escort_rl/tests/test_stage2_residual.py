import importlib.util
from pathlib import Path
import numpy as np,pytest,torch
p=Path(__file__).resolve().parents[1]/'diagnostics/stage2_residual.py'
s=importlib.util.spec_from_file_location('stage2_residual',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

@pytest.fixture
def policy():
    torch.set_num_threads(1)
    return m.ResidualPolicy()

def observations(n=10):
    x=np.random.default_rng(2).uniform(-.4,.4,(n,40)).astype(np.float32);x[:,22:]=0;return x

def test_exact_stage1_preservation_even_with_nonzero_residual(policy):
    x=observations();base,_=policy.base.predict(x,deterministic=True)
    with torch.no_grad():policy.net[-1].bias.fill_(5)
    assert np.array_equal(policy.predict(x)[0],base)
    assert np.array_equal(policy.predict(x[0])[0],policy.base.predict(x[0],deterministic=True)[0])
    policy.assert_frozen()

def test_base_never_reads_pedestrian_inputs_and_initial_residual_zero(policy):
    x=observations();x[:,22:28]=[.1,-.1,.2,-.2,.25,1]
    assert np.array_equal(policy.base_action(x),policy.base.predict(m.base_observation(x),deterministic=True)[0])
    assert np.array_equal(policy.predict(x)[0],policy.base_action(x))
    with torch.no_grad():policy.net[-1].bias.fill_(10)
    a=policy.predict(x)[0];assert np.isfinite(a).all() and (np.abs(a)<=1).all()
    assert not np.array_equal(a,policy.base_action(x))

def test_only_residual_parameters_update(policy):
    x=observations();x[:,27]=1;base=policy.base_action(x);z=m.features(x,base,policy.bc,policy.c);target=torch.as_tensor(np.clip(base+.4,-1,1))
    opt=torch.optim.Adam(policy.net.parameters(),lr=.001)
    for _ in range(10):
        delta=policy.residual_tensor(torch.as_tensor(z));loss=m.fitting_loss(torch.as_tensor(base),delta,target,policy.c).mean();opt.zero_grad();loss.backward();opt.step()
    policy.assert_frozen();assert all(p.grad is None for p in policy.base.policy.parameters())
    assert np.array_equal(policy.predict(observations())[0],policy.base_action(observations()))

def test_saturation_loss_has_gradient_when_executed_action_is_clipped():
    c=m.config();base=torch.tensor([[.9,.9,.9]]);delta=torch.tensor([[1.,1.,1.]],requires_grad=True);target=torch.zeros(1,3)
    loss=m.fitting_loss(base,delta,target,c).sum();loss.backward();assert (delta.grad>0).all()
    exact=torch.tensor([[-1.,1.,0.]]);target=torch.tensor([[1.,-1.,.5]])
    assert m.fitting_loss(exact,target-exact,target,c).item()==pytest.approx(0)

def test_risk_features_causal_geometry_and_finite(policy):
    x=np.zeros((1,40),np.float32);x[0,5]=1;x[0,22]=2/16;x[0,24]=-.5/2;x[0,27]=1
    r=m.risk_features(x,policy.bc);assert r.shape==(1,8);assert r[0,0]==pytest.approx(.5);assert r[0,1]==pytest.approx(.5);assert r[0,2]==pytest.approx(1);assert r[0,3]==pytest.approx(0)
    assert np.isfinite(m.risk_features(np.zeros(40,np.float32),policy.bc)).all()
    assert m.features(x,policy.base_action(x),policy.bc,policy.c).shape==(1,51)

def test_observable_teacher_does_not_depend_on_hidden_previous(policy):
    x=observations(1)[0];x[22:28]=[.08,.02,-.1,0,.25,1]
    t=m.SupervisionTeacher(policy.bc,'observable_previous_command_v1');t.inner.previous=np.array([.3,.3]);a=t.act(x);t.inner.previous=np.array([-.3,-.3]);assert np.array_equal(a,t.act(x))

def test_save_reload_preserves_frozen_base_and_training_state(policy,tmp_path):
    rng=np.random.default_rng(11);opt=torch.optim.Adam(policy.net.parameters());x=observations();x[:,27]=1
    path=tmp_path/'residual.pt';m.save(policy,path,opt,rng,dict(test=True));loaded,record=m.load(path)
    assert np.array_equal(policy.predict(x)[0],loaded.predict(x)[0]);loaded.assert_frozen()
    state=torch.load(path,weights_only=False);assert state['numpy_rng']==rng.bit_generator.state and state['optimizer'] is not None
    with path.open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='provenance'):m.load(path)

def test_partitions_and_unsupported_modes_are_explicit(policy):
    c=m.config();assert len(m.checked_cases(c,m.DEFAULT))==2;c['development_cases'][0]['episode_seed']+=1
    with pytest.raises(ValueError):m.checked_cases(c,m.DEFAULT)
    with pytest.raises(ValueError):policy.predict(observations()[0],deterministic=False)
    x=observations()[0];x[33]=1
    with pytest.raises(ValueError):policy.predict(x)

def test_registry_accepts_relative_checkpoint_paths(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(m,'ROOT',tmp_path)
    folder=tmp_path/'docs/bodyguard';folder.mkdir(parents=True);(folder/'MODEL_WEIGHTS.md').write_text('# Fixture\n')
    checkpoint=Path('run/residual.pt');checkpoint.parent.mkdir();checkpoint.write_bytes(b'test-parameters')
    p=SimpleNamespace(c=dict(base_checkpoint='base.zip',base_sha256='test',teacher_mode='observable_previous_command_v1'),base=SimpleNamespace(num_timesteps=5000),updates=3)
    m.register(p,checkpoint,'unit fixture')
    text=(folder/'MODEL_WEIGHTS.md').read_text();assert 'run/residual.pt' in text and 'unit fixture' in text
