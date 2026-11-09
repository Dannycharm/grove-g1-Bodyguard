import importlib.util,copy
from pathlib import Path
import numpy as np,pytest,torch
p=Path(__file__).resolve().parents[1]/'diagnostics/stage2_residual_coverage.py'
s=importlib.util.spec_from_file_location('stage2_residual_coverage',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_deterministic_selection_covers_both_sides_and_retains_anchors():
    c=m.read_config();a,rejections=m.select_cases(c);b,_=m.select_cases(c)
    assert a==b and len(a)==8
    assert len({(r['family'],r['episode_seed']) for r in a})==8
    assert all(r['episode_seed']%4==0 for r in a)
    for f in m.ENCOUNTERS:assert {r['side'] for r in a if r['family']==f}=={-1,1}
    for retained in c['retain_cases']:assert any(r['family']==retained['family'] and r['episode_seed']==retained['episode_seed'] for r in a)
    sc=m.load_config();split=__import__('json').loads((Path(sc['stage1_config']).parent/'stage1_splits.json').read_text())
    assert not any(m.reserved(r['metadata'],split) for r in a)
    with pytest.raises(ValueError):m.resolve_case('crossing',920001,sc,split)

def test_sampling_balances_families_sides_cases_despite_unequal_rows():
    phases=[];families=[];sides=[];keys=[]
    for j,f in enumerate(m.ENCOUNTERS):
        for side in (-1,1):
            for case in (0,1):
                n=3+j+case*7;phases+=['avoidance']+['ordinary']*(n-1);families +=[f]*n;sides +=[side]*n;keys +=[(f,100+case+side*10)]*n
    fs=np.array(families);ss=np.array(sides);ph=np.array(phases)
    prob=m.sampling_probabilities(ph,fs,ss,keys,dict(avoidance=8,ordinary=1));assert prob.sum()==pytest.approx(1)
    for f in m.ENCOUNTERS:
        assert prob[fs==f].sum()==pytest.approx(1/3)
        for side in (-1,1):assert prob[(fs==f)&(ss==side)].sum()==pytest.approx(1/6)
    with pytest.raises(ValueError):m.sampling_probabilities(ph[fs!='head_on'],fs[fs!='head_on'],ss[fs!='head_on'],[k for k in keys if k[0]!='head_on'],dict(avoidance=8,ordinary=1))

def test_real_parent_weights_optimizer_and_rng_continue_exactly():
    torch.set_num_threads(1);c=m.read_config();policy,opt,rng,record=m.restore_parent(c)
    saved=torch.load(m.ROOT/c['parent_checkpoint'],weights_only=False)
    assert policy.updates==12000 and rng.bit_generator.state==saved['numpy_rng'] and torch.equal(torch.get_rng_state(),saved['torch_rng'])
    assert all(torch.equal(v,saved['net'][k]) for k,v in policy.net.state_dict().items())
    assert opt.state_dict()['param_groups']==saved['optimizer']['param_groups']
    for k,state in saved['optimizer']['state'].items():
        for name,v in state.items():assert torch.equal(v,opt.state_dict()['state'][k][name])
    policy.assert_frozen();assert len(m.verified_parent_sources(record))==3
    c['learning_rate']=.001
    with pytest.raises(ValueError,match='learning-rate'):m.restore_parent(c)

def test_fit_updates_only_residual_and_preserves_zero_gate(monkeypatch):
    torch.set_num_threads(1);c=m.read_config();c['updates_per_iteration']=4;c['batch_size']=12;policy,opt,rng,record=m.restore_parent(c)
    x=np.zeros((12,40),np.float32);x[:,5]=1;x[:,27]=1;x[:,22]=.1;y=np.ones((12,3),np.float32)*.3
    fs=np.repeat(m.ENCOUNTERS,4);ss=np.tile([-1,-1,1,1],3);keys=[(str(f),int(side)*4+100) for f,side in zip(fs,ss)];ph=np.array(['avoidance']*12)
    monkeypatch.setattr(m,'data_arrays',lambda *a:(x,y,ph,fs,ss,keys,[]))
    before=m.seed_descriptor(rng);report=m.fit_iteration(policy,opt,rng,[],c)
    assert report['updates']==12004 and report['batch_rng_before']==before and report['batch_rng_after']!=before
    policy.assert_frozen();x[:,27]=0;assert np.array_equal(policy.predict(x)[0],policy.base_action(x))

def test_missing_parent_optimizer_is_an_explicit_error(tmp_path,monkeypatch):
    c=m.read_config();p=tmp_path/'bad.pt';torch.save(dict(optimizer=None,numpy_rng=None),p);c['parent_checkpoint']=str(p);c['parent_sha256']=m.sha256(p)
    monkeypatch.setattr(m.core,'load',lambda path:(object(),{}))
    with pytest.raises(ValueError,match='refusing fresh'):m.restore_parent(c)

def test_invalid_seed_and_balancing_configs_fail(tmp_path):
    import yaml
    c=m.read_config();c['candidate_seed_start']=920001;p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):m.read_config(p)
    c['candidate_seed_start']=920000;c['balance_sides']=False;p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):m.read_config(p)

def test_variation_v3_keeps_head_on_and_matches_predeclared_cells():
    path=Path(m.__file__).with_name('stage2_residual_variation_v3.yaml');c=m.read_config(path);cases,_=m.select_cases(c)
    assert len(cases)==16 and len({(r['family'],r['episode_seed']) for r in cases})==16
    assert {(r['family'],r['episode_seed']) for r in cases if r['family']=='head_on'}=={('head_on',930004),('head_on',930020)}
    selected={r['variation_cell']:r for r in cases if 'variation_cell' in r}
    assert set(selected)=={cell['name'] for cell in c['variation_cells']}
    for cell in c['variation_cells']:assert m.matches_cell(selected[cell['name']],cell)
    assert all(r['episode_seed']%4==0 for r in cases)

def test_variation_config_rejects_duplicate_cells_and_invalid_ranges(tmp_path):
    import yaml
    c=m.read_config(Path(m.__file__).with_name('stage2_residual_variation_v3.yaml'));c['variation_cells'].append(copy.deepcopy(c['variation_cells'][0]));p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):m.read_config(p)
    c['variation_cells'].pop();c['variation_cells'][0]['speed_m_s']=[.4,.2];p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):m.read_config(p)

def test_v3_parent_restores_20k_residual_without_reinitialization():
    torch.set_num_threads(1);c=m.read_config(Path(m.__file__).with_name('stage2_residual_variation_v3.yaml'));policy,opt,rng,record=m.restore_parent(c);saved=torch.load(m.ROOT/c['parent_checkpoint'],weights_only=False)
    assert policy.updates==20000 and rng.bit_generator.state==saved['numpy_rng']
    assert all(float(v['step'])==20000 for v in opt.state.values())
    assert len(m.verified_parent_sources(record))==5
    policy.assert_frozen()
