"""Plot already-recorded Stage 2 traces; no simulation or policy updates."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(x) for x in (Path(a.episode)/'trace.jsonl').read_text().splitlines()];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    info=[r['info'] for r in rows];t=[r['elapsed_s'] for r in info]
    f,ax=plt.subplots(2,2,figsize=(11,8))
    for key,name in [('robot_position','G1'),('client_position','Client'),('pedestrian_position','Pedestrian')]:
        xy=np.array([r[key][:2] for r in info if r.get(key) is not None])
        if len(xy):ax[0,0].plot(xy[:,0],xy[:,1],label=name)
    ax[0,0].axis('equal');ax[0,0].legend();ax[0,0].set(xlabel='World x (m)',ylabel='World y (m)')
    ax[0,1].plot(t,[r.get('geometric_surface_clearance_m') for r in info]);ax[0,1].axhline(0,color='red',ls='--');ax[0,1].axhline(.3,color='gray',ls=':');ax[0,1].set(ylabel='Footprint surrogate surface clearance (m)')
    ax[1,0].plot(t,[r['evaluation']['formation_error_m'] for r in info]);ax[1,0].set(xlabel='Episode time (s)',ylabel='Formation error (m)')
    for i,label in enumerate(('vx','vy','yaw')):
        ax[1,1].plot(t,[r['applied_command'][i] for r in info],label=label+' command')
        ax[1,1].plot(t,[r['body_velocity'][i] for r in info],ls='--',alpha=.6,label=label+' measured')
    ax[1,1].legend(fontsize=7);ax[1,1].set(xlabel='Episode time (s)',ylabel='m/s or rad/s')
    f.suptitle('Stage 2 '+info[-1]['reason']+' | proxy geometry is not physical person contact');f.tight_layout();f.savefig(out,dpi=130);plt.close(f)


if __name__=='__main__':main()
