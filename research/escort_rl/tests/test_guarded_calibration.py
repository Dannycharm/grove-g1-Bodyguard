"""Calibration reuse cannot change the measured exploration/environment experiment."""
import copy
import importlib.util
import json
import yaml
import pytest
from escort_rl.contract import PACKAGE
from escort_rl.ppo_training import read_config,provenance


def tool():
    s=importlib.util.spec_from_file_location('prepare_guarded',PACKAGE/'tools/prepare_imitation_guarded.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_reuse_lr_kl_only_and_preserves_original(tmp_path):
    old=read_config(PACKAGE/'configs/ppo_stage1_imitation_warmstart_v1.yaml');manifest=provenance(old)
    summary=dict(sigma=.05,success_rate=1.,per_family={'straight':{'success_rate':1.},'gentle_turn':{'success_rate':1.},'stop_start':{'success_rate':1.}},formation_mean_m=.06,failures={'success':12})
    report=dict(protocol='imitation_pretrained_ppo_v1',config=old,source_hashes=manifest['source_hashes'],selected_sigma=.05,results=[summary])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));path=tmp_path/'noise_report.json';path.write_text(json.dumps(report));original=path.read_bytes()
    target=PACKAGE/'configs/ppo_stage1_imitation_guarded_v1.yaml'
    result=tool().prepare(target,path,tmp_path/'derived');assert result['selected_sigma']==.05
    assert path.read_bytes()==original
    changed=yaml.safe_load(target.read_text());changed['environment_config']=old['environment_path'];changed['split_config']=old['split_path'];changed['imitation']['noise_seed']+=1
    bad=tmp_path/'bad.yaml';bad.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match='ONLY'):tool().prepare(bad,path,tmp_path/'invalid')
    report['source_hashes'][next(iter(report['source_hashes']))]='modified';path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='source changed'):tool().prepare(target,path,tmp_path/'invalid2')
