"""Read-only development plots from residual diagnostic traces."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);out=root/'development_comparison.png'
    if out.exists():raise FileExistsError(out)
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for col,family in enumerate(['crossing','diagonal']):
        old=[json.loads(l) for l in open(root/'initial_student'/f'trace_{col:03d}.jsonl')];rows=[json.loads(l) for l in open(root/'student_2'/f'trace_{col:03d}.jsonl')]
        for rr,label,style in [(old,'Frozen base only','--'),(rows,'Frozen base + residual','-')]:
            xy=np.array([r['info']['robot_position'][:2] for r in rr]);axes[0,col].plot(*xy.T,style,label=label)
        for field,label in [('client_position','Client'),('pedestrian_position','Pedestrian')]:
            xy=np.array([r['info'][field][:2] for r in rows]);axes[0,col].plot(*xy.T,':',label=label)
        axes[0,col].set_title(f'{family}: same development case');axes[0,col].axis('equal');axes[0,col].set(xlabel='World x (m)',ylabel='World y (m)');axes[0,col].legend(fontsize=8)
        for rr,label in [(old,'Base only'),(rows,'Residual')]:axes[1,col].plot([r['info']['elapsed_s'] for r in rr],[r['info']['geometric_surface_clearance_m'] for r in rr],label=label)
        axes[1,col].axhline(0,color='red',ls='--');axes[1,col].set(xlabel='Elapsed time (s)',ylabel='Geometric surface clearance (m)');axes[1,col].legend()
        t=[r['info']['elapsed_s'] for r in rows];base=np.array([r['base_action'] for r in rows]);final=np.array([r['executed_action'] for r in rows]);caps=np.array([.6,.6,.9])
        for j,label in enumerate(['vx','vy']):
            axes[2,col].plot(t,base[:,j]*caps[j],'--',label=f'Base {label}');axes[2,col].plot(t,final[:,j]*caps[j],label=f'Final {label}')
        axes[2,col].set(xlabel='Elapsed time (s)',ylabel='Body-frame command (m/s)');axes[2,col].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out,dpi=140);plt.close(fig)
if __name__=='__main__':main()
