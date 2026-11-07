"""Offline only. Exclusive output creation; never opens ROS/DDS."""
import argparse,json,time,resource,concurrent.futures,multiprocessing,hashlib
from pathlib import Path
import numpy as np
from escort_rl.contract import PACKAGE,sha256
from escort_rl.mujoco_runner import MujocoRunner
from escort_rl.measurements import response_metrics

def write(path,value):
    with open(path,'x') as f: json.dump(value,f,indent=2)

def commands(out,scene):
    cfg=json.loads((PACKAGE/'configs/command_suite.json').read_text())
    runner=MujocoRunner(scene)
    write(out/'manifest.json',runner.runtime_manifest())
    write(out/'suite.json',cfg)
    summaries=[]
    for case,cmd in enumerate(cfg['commands']):
        runner.reset(); rows=[]
        segments=[('startup',cfg['startup_s'],[0,0,0]),('stationary',cfg['stationary_s'],[0,0,0]),
                  ('hold',cfg['hold_s'],cmd),('deceleration',cfg['deceleration_s'],(np.array(cmd)*.5).tolist()),
                  ('stop',cfg['stop_s'],[0,0,0]),('correction',cfg['correction_s'],[.1,0,.1]),
                  ('final_stop',cfg['stop_s'],[0,0,0])]
        hold_start=cfg['startup_s']+cfg['stationary_s']; stop_start=hold_start+cfg['hold_s']+cfg['deceleration_s']
        refresh=round(cfg['command_refresh_s']/cfg['sample_period_s'])
        for label,duration,value in segments:
            for i in range(round(duration/cfg['sample_period_s'])):
                if i%refresh==0: runner.command(value)
                row=runner.advance(cfg['sample_period_s']); row['segment']=label; rows.append(row)
        with open(out/f'case_{case:02d}.jsonl','x') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        # Steady tracking is measured before deceleration, stopping from full-zero command.
        result=response_metrics(rows,cmd,hold_start,hold_start+cfg['hold_s'],cfg,stop_command_time=stop_start)
        correction_start=stop_start+cfg['stop_s']
        result['correction']=response_metrics(rows,[.1,0,.1],correction_start,correction_start+cfg['correction_s'],cfg)
        result['valid_capability_trial']=not result['fall'] and not result['emergency']
        result['legacy_com_metrics']=response_metrics(rows,cmd,hold_start,hold_start+cfg['hold_s'],cfg,velocity_key='legacy_com_velocity',stop_command_time=stop_start)
        summaries.append(result)
    write(out/'summary.json',summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for axis,name in enumerate(['vx','vy','yaw_rate']):
        selected=[r for r in summaries if sum(abs(x)>0 for x in r['requested'])<=1 and
                  (r['requested'][axis]!=0 or not any(r['requested']))]
        selected.sort(key=lambda r:r['applied'][axis])
        fig,ax=plt.subplots(); ax.plot([r['applied'][axis] for r in selected],[r['achieved'][axis] for r in selected],'o-')
        ax.set(xlabel=f'Applied {name}',ylabel=f'Achieved {name}',title='Pelvis-origin / pelvis axes; failures retained in summary')
        fig.savefig(out/f'{name}_curve.png'); plt.close(fig)
    for case in range(len(summaries)):
        data=[json.loads(x) for x in (out/f'case_{case:02d}.jsonl').read_text().splitlines()]
        fig,axes=plt.subplots(3,1,sharex=True)
        for axis,ax in enumerate(axes):
            for key in ('requested','applied','body_velocity'):
                ax.plot([r['time'] for r in data],[r[key][axis] for r in data],label=key)
            ax.legend()
        fig.savefig(out/f'case_{case:02d}.png'); plt.close(fig)
    runner.close()

def worker(args):
    transitions,scene,worker_id,output=args
    r=MujocoRunner(scene,actors=3); r.advance(1.)
    reset_lat=[]; latency=[]; step_latency=[]; fingerprint=[]
    log=open(Path(output)/f"worker_{worker_id}.jsonl", "x")
    cpu=time.process_time(); start=time.perf_counter(); physics=r.physics_steps; gait=r.gait.calls; total_gait=0; total_physics=0
    for i in range(transitions):
        if i%50==0:
            total_gait+=r.gait.calls-gait; total_physics+=r.physics_steps-physics
            tick=time.perf_counter(); r.reset(); reset_lat.append(time.perf_counter()-tick)
            gait=physics=0
        r.command([.2,.05*(worker_id%2),.1])
        tick=time.perf_counter()
        for _ in range(100):
            sub=time.perf_counter();r.step_physics();step_latency.append(time.perf_counter()-sub)
        row=r.measure();log.write(json.dumps(row)+'\n');log.flush()
        latency.append(time.perf_counter()-tick)
        if i==0: fingerprint=row['joint_position']
    total_gait+=r.gait.calls-gait; total_physics+=r.physics_steps-physics
    log.close()
    elapsed=time.perf_counter()-start
    result=dict(worker=worker_id,transitions=transitions,wall_s=elapsed,cpu_s=time.process_time()-cpu,
        physics_steps=total_physics,gait_calls=total_gait,sim_s=total_physics*.002,
        reset_latency_s=reset_lat,transition_latency_p50_p95_s=np.percentile(latency,[50,95]).tolist(),
        physics_latency_p50_p95_s=np.percentile(step_latency,[50,95]).tolist(),
        cpu_percent=100*(time.process_time()-cpu)/elapsed,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        first_state_sha256=hashlib.sha256(np.array(fingerprint).tobytes()).hexdigest(),
        final_failure=bool(row['fallen'] or row['emergency']),logging='measurement construction and flushed per-transition JSONL enabled',
        actors='three synthetic noncolliding mocap proxies; no social response')
    result['runtime_manifest']=r.runtime_manifest()
    r.close(); return result

def throughput(out,scene,count,transitions):
    start=time.perf_counter()
    if count==1: results=[worker((transitions,scene,0,str(out)))]
    else:
        with concurrent.futures.ProcessPoolExecutor(count,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(worker,[(transitions,scene,i,str(out)) for i in range(count)]))
    wall=time.perf_counter()-start
    # Aggregate steady worker rate excludes process/model startup, bounded by slowest worker.
    steady=max(r['wall_s'] for r in results); rate=sum(r['transitions'] for r in results)/steady
    write(out/'throughput.json',dict(workers=results,end_to_end_wall_s=wall,aggregate_transitions_s=rate,
        physics_steps_s=sum(r['physics_steps'] for r in results)/steady,
        gait_inferences_s=sum(r['gait_calls'] for r in results)/steady,
        simulated_seconds_s=sum(r['sim_s'] for r in results)/steady,
        collection_seconds={str(n):n/rate for n in [100000,500000,1000000]},
        qualification='collection only; optimizer/evaluation excluded; per-transition disk logging included; failures do not skip physics; validate stability before using estimate'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['commands','throughput']);p.add_argument('--output',required=True)
    p.add_argument('--scene',choices=['flat','navigation'],default='flat');p.add_argument('--workers',type=int,default=1)
    p.add_argument('--transitions',type=int,default=500); a=p.parse_args()
    if a.workers<1 or a.transitions<1: p.error('positive workers/transitions required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:
        if a.mode=='commands': commands(out,a.scene)
        else: throughput(out,a.scene,a.workers,a.transitions)
    except Exception as exc:
        write(out/'failure.json',dict(error=type(exc).__name__,message=str(exc))); raise
