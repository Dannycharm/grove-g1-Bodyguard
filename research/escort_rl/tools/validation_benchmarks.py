"""Repeat the existing benchmark sequentially; never changes runner semantics."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--transitions',type=int,default=500)
    a=p.parse_args()
    output=Path(a.output);output.mkdir(exist_ok=False,parents=True)
    records=[]
    for trial in range(1,a.trials+1):
        for workers in (1,2):
            path=output/f'workers_{workers}_trial_{trial}'
            command=[sys.executable,str(Path(__file__).with_name('run.py')),'throughput',
                     '--workers',str(workers),'--transitions',str(a.transitions),'--output',str(path)]
            start=time.perf_counter()
            result=subprocess.run(command,check=True,capture_output=True,text=True)
            elapsed=time.perf_counter()-start
            data=json.loads((path/'throughput.json').read_text())
            records.append(dict(workers=workers,trial=trial,external_wall_s=elapsed,
                                command=command,result=data))
            print(f"workers={workers} trial={trial} transitions/s={data['aggregate_transitions_s']:.3f}",flush=True)
    with open(output/'repeated_results.json','x') as f:json.dump(records,f,indent=2)
