"""Profile existing full-dynamics runner; instrumented timing is diagnostic only."""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
from escort_rl.mujoco_runner import MujocoRunner

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    r=MujocoRunner(actors=3);r.advance(1.)
    profile=cProfile.Profile()
    with open(out/'transitions.jsonl','x') as log:
        profile.enable()
        for i in range(100):
            if i%50==0:r.reset()
            r.command([.2,0,.1])
            row=r.advance(.2)
            log.write(json.dumps(row)+'\n');log.flush()
        profile.disable()
    profile.dump_stats(str(out/'profile.pstats'))
    stream=io.StringIO();stats=pstats.Stats(profile,stream=stream).sort_stats('cumulative')
    stats.print_stats(35)
    with open(out/'profile.txt','x') as f:f.write(stream.getvalue())
    print(stream.getvalue())
    r.close()
