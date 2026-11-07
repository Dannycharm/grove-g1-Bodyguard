"""Run bounded diagnostic imitation and fixed held-out native-physics evaluation."""
import argparse
from escort_rl.capacity_diagnostic import run_capacity

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=4000)
    a=p.parse_args();run_capacity(a.output,updates=a.updates)
