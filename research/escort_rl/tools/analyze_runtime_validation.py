"""Summarize persisted validation evidence without executing physics."""
import argparse
import json
from pathlib import Path
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args()
    root=Path(a.input)
    summary={}
    repeats=json.loads((root/'benchmarks/repeated_results.json').read_text())
    for count in (1,2):
        results=[x['result'] for x in repeats if x['workers']==count]
        workers=[w for x in results for w in x['workers']]
        rates=np.array([x['aggregate_transitions_s'] for x in results])
        reset=np.concatenate([w['reset_latency_s'] for w in workers])
        summary[str(count)]=dict(transitions_per_s_trials=rates.tolist(),mean=float(rates.mean()),
            sample_std=float(rates.std(ddof=1)),
            physics_steps_s_mean=float(np.mean([x['physics_steps_s'] for x in results])),
            gait_calls_s_mean=float(np.mean([x['gait_inferences_s'] for x in results])),
            sim_seconds_per_wall_second_mean=float(np.mean([x['simulated_seconds_s'] for x in results])),
            end_to_end_transitions_per_s_mean=float(np.mean([count*500/x['end_to_end_wall_s'] for x in results])),
            reset_p50_p95_ms=(np.percentile(reset,[50,95])*1000).tolist(),
            transition_latency_mean_of_worker_p50_p95_ms=[float(np.mean([w['transition_latency_p50_p95_s'][i] for w in workers])*1000) for i in (0,1)],
            physics_latency_mean_of_worker_p50_p95_us=[float(np.mean([w['physics_latency_p50_p95_s'][i] for w in workers])*1e6) for i in (0,1)],
            collection_hours={str(n):float(n/rates.mean()/3600) for n in (100000,500000,1000000)},
            peak_rss_mib_range=[min(w['peak_rss_kib'] for w in workers)/1024,max(w['peak_rss_kib'] for w in workers)/1024])
    with open(root/'throughput_summary.json','x') as f:json.dump(summary,f,indent=2)
    commands=json.loads((root/'commands/summary.json').read_text())
    table=['| Requested (vx, vy, wz) | Achieved (vx, vy, wz) | First crossing latency (s) | Stop time (s) | Stop path (m) |',
           '|---|---|---:|---:|---:|']
    def triple(v):return '('+', '.join(f'{x:.3f}' for x in v)+')'
    def number(v):return '—' if v is None else f'{v:.3f}'
    for r in commands:
        table.append('| '+triple(r['requested'])+' | '+triple(r['achieved'])+' | '+number(r['latency_s'])+' | '+number(r['stopping_time_s'])+' | '+number(r['stopping_distance_m'])+' |')
    with open(root/'command_table.md','x') as f:f.write('\n'.join(table)+'\n')
