"""Synthetic matched-horizon reward audit, not a dynamics rollout."""
import argparse
from escort_rl.reward_audit import export_audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--candidate',action='store_true');a=p.parse_args()
    results=export_audit(a.output,('legacy_v1','formation_progress_v2') if a.candidate else ('legacy_v1',))
    for r in results:print(r['version'],r['case'],round(r['total_reward'],6),r['reward_components'])
