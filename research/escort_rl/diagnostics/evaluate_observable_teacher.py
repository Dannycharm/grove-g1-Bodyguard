"""Revalidate the observable-history teacher variant; no fitting data exported."""
import argparse,json
from pathlib import Path
import torch
from stage2_residual import SupervisionTeacher,load_stage1,load_config,DEFAULT
from escort_rl.stage2.evaluation import evaluate
from escort_rl.stage2.training import manifest
from escort_rl.contract import sha256

class ObservableTeacherPolicy:
    def __init__(self):
        self.teacher=SupervisionTeacher(load_stage1(load_config()['stage1_config']),'observable_previous_command_v1')
    def predict(self,observation,deterministic=True):
        if not deterministic:raise ValueError('Teacher is deterministic')
        return self.teacher.act(observation),None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(1)
    record=Path(a.output+'.teacher_manifest.json')
    if record.exists():raise FileExistsError(record)
    record.write_text(json.dumps(dict(teacher_mode='observable_previous_command_v1',stage2_contract=manifest(DEFAULT),source_sha256=sha256(__file__),residual_module_sha256=sha256(Path(__file__).with_name('stage2_residual.py')),training_data_exported=False),indent=2))
    c=load_config();c['_config_path']=str(DEFAULT);result=evaluate(ObservableTeacherPolicy(),c,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes_detail'},indent=2))
if __name__=='__main__':main()
