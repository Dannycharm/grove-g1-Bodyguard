"""Summarize already-executed matched reward audit; no simulation/training."""
import argparse,csv,json
from pathlib import Path
import numpy as np

STRATEGIES=('follow','yield','sidestep','stop')
FAMILIES=('crossing','head_on','diagonal')


def summarize(root,output=None):
    root=Path(root);out=Path(output) if output else root/'comparison';out.mkdir(parents=True,exist_ok=False)
    summaries={name:json.loads((root/f'validation_{name}/summary.json').read_text()) for name in STRATEGIES}
    manifests={name:json.loads((root/f'validation_{name}/manifest.json').read_text()) for name in STRATEGIES}
    for name in STRATEGIES:
        assert manifests[name]['cases']==manifests['follow']['cases']
        assert manifests[name]['parameters']==manifests['follow']['parameters']
        assert manifests[name]['stage2_contract']==manifests['follow']['stage2_contract']
    indexed={name:{(r['family'],r['episode_seed']):r for r in summary['episodes_detail']} for name,summary in summaries.items()}
    rows=[];windows=[]
    for name,summary in summaries.items():
        for r in summary['episodes_detail']:
            row=dict(strategy=name,family=r['family'],seed=r['episode_seed'],reason=r['reason'],success=r['success'],duration_s=r['duration_s'],return_total=r['return_total'],discounted_return=r['discounted_return'],minimum_surface_clearance_m=r['minimum_surface_clearance_m'],diagnostic_active_s=r['diagnostic_active_s'])
            row.update({'reward_'+k:v for k,v in r['reward_components'].items()});row.update({'discounted_'+k:v for k,v in r['discounted_reward_components'].items()});rows.append(row)
            assert np.isclose(r['return_total'],sum(r['reward_components'].values()))
            assert np.isclose(r['discounted_return'],sum(r['discounted_reward_components'].values()))
            for start,components in r['five_second_reward_components'].items():windows.append(dict(strategy=name,family=r['family'],seed=r['episode_seed'],window_start_s=int(start),window_end_s=int(start)+5,executed_until_s=min(int(start)+5,r['duration_s']),executed_duration_s=min(int(start)+5,r['duration_s'])-int(start),total=sum(components.values()),**components))
    def export(name,data):
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    export('episode_reward_components.csv',rows);export('five_second_reward_components.csv',windows)
    pairs=[]
    for mode in ('yield','sidestep'):
        for key,r in indexed[mode].items():
            baseline=indexed['follow'][key];stop=indexed['stop'][key]
            pairs.append(dict(strategy=mode,family=key[0],seed=key[1],success=r['success'],follow_success=baseline['success'],
                total_gain_vs_follow=r['return_total']-baseline['return_total'],discounted_gain_vs_follow=r['discounted_return']-baseline['discounted_return'],
                total_gain_vs_stop=r['return_total']-stop['return_total'],discounted_gain_vs_stop=r['discounted_return']-stop['discounted_return']))
    prefixes=[]
    for key in indexed['follow']:
        durations=[indexed[name][key]['duration_s'] for name in STRATEGIES]
        first_end=min(durations)
        traces={name:[json.loads(line) for line in next((root/f'validation_{name}').glob(f'*_{key[0]}_{key[1]}/trace.jsonl')).read_text().splitlines()] for name in STRATEGIES}
        common_count=min(len(t) for t in traces.values())
        for j in range(common_count):
            reference=traces['follow'][j]['info']
            for name in STRATEGIES:
                actual=traces[name][j]['info']
                assert reference['elapsed_s']==actual['elapsed_s']
                assert reference['client_position']==actual['client_position'] and reference['pedestrian_position']==actual['pedestrian_position']
        for mode in ('yield','sidestep'):
            pair=next(p for p in pairs if p['strategy']==mode and (p['family'],p['seed'])==key)
            active=[r['info']['elapsed_s'] for r in traces[mode] if r['diagnostic_active']]
            penalty=[r['info']['elapsed_s'] for r in traces['follow'] if r['info']['reward_components'].get('pedestrian_clearance',0)<-1e-9]
            pair['first_intervention_s']=min(active) if active else None
            pair['follow_first_clearance_penalty_s']=min(penalty) if penalty else None
            pair['intervention_lead_over_clearance_penalty_s']=None if not active or not penalty else min(penalty)-min(active)
            if active:
                start=min(active)-.2;end=min(start+2.,first_end-.2)
                pair['early_window_start_s']=start;pair['early_window_end_s']=end
                pair['early_window_reward_gain_vs_follow']=sum(r['reward'] for r in traces[mode] if start+1e-8<r['info']['elapsed_s']<=end+1e-8)-sum(r['reward'] for r in traces['follow'] if start+1e-8<r['info']['elapsed_s']<=end+1e-8)
        for label,cutoff in [('before_first_outcome',max(0.,first_end-.2)),('including_first_outcome',first_end)]:
            for name,trace in traces.items():
                selected=[r for r in trace if r['info']['elapsed_s']<=cutoff+1e-8]
                components={k:sum(r['info']['reward_components'].get(k,0.) for r in selected) for k in trace[0]['info']['reward_components']}
                prefixes.append(dict(strategy=name,family=key[0],seed=key[1],horizon_kind=label,cutoff_s=cutoff,steps=len(selected),total=sum(r['reward'] for r in selected),discounted_return=sum(r['reward']*r['discount_factor'] for r in selected),**components))
    export('matched_prefix_rewards.csv',prefixes)
    export('paired_comparisons.csv',pairs)
    conclusion={}
    for mode in ('yield','sidestep'):
        successful=[p for p in pairs if p['strategy']==mode and p['success'] and not p['follow_success']]
        conclusion[mode]=dict(successful_avoidance_vs_failed_follow=len(successful),higher_total_than_failed_follow=sum(p['total_gain_vs_follow']>0 for p in successful),higher_discounted_than_failed_follow=sum(p['discounted_gain_vs_follow']>0 for p in successful),
            successful_avoidance_higher_total_than_stop=sum(p['total_gain_vs_stop']>0 for p in successful),successful_avoidance_higher_discounted_than_stop=sum(p['discounted_gain_vs_stop']>0 for p in successful),early_intervention_window_lower_reward_than_follow=sum(p.get('early_window_reward_gain_vs_follow',0)<0 for p in successful),mean_intervention_lead_over_clearance_penalty_s=float(np.mean([p['intervention_lead_over_clearance_penalty_s'] for p in successful if p['intervention_lead_over_clearance_penalty_s'] is not None])) if successful else None)
    compact={name:{k:v for k,v in s.items() if k!='episodes_detail'} for name,s in summaries.items()}
    result=dict(experiment='stage2_matched_reward_audit',gamma=.99,controller_version=manifests['follow']['parameters']['version'],all_cases_and_reward_contracts_matched=True,summaries=compact,paired_successful_avoidance=conclusion,new_policy_training_updates=0)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(3);width=.19
    for i,name in enumerate(STRATEGIES):
        vals=[summaries[name]['per_family'][f] for f in FAMILIES]
        axes[0].bar(x+(i-1.5)*width,[r['mean_return'] for r in vals],width,label=name)
        axes[1].bar(x+(i-1.5)*width,[r['mean_discounted_return'] for r in vals],width,label=name)
    for ax,label in zip(axes,('Total episode reward','Discounted return gamma .99')):ax.set_xticks(x,FAMILIES);ax.set_ylabel(label);ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=8)
    fig.suptitle('Matched full-G1 diagnostic controllers | includes unsuccessful maneuvers');fig.tight_layout();fig.savefig(out/'reward_comparison.png',dpi=150);plt.close(fig)
    selected=next((p for p in pairs if p['strategy']=='yield' and p['family']=='crossing' and p['success'] and not p['follow_success']),None)
    if selected:
        fig,ax=plt.subplots(2,2,figsize=(12,8))
        longest_trace=[]
        for name in STRATEGIES:
            path=next((root/f'validation_{name}').glob(f'*_{selected["family"]}_{selected["seed"]}/trace.jsonl'))
            trace=[json.loads(line) for line in path.read_text().splitlines()]
            if len(trace)>len(longest_trace):longest_trace=trace
            times=[r['info']['elapsed_s'] for r in trace];rxy=np.array([r['info']['robot_position'][:2] for r in trace])
            ax[0,0].plot(rxy[:,0],rxy[:,1],label=name)
            ax[0,1].plot(times,[r['info']['geometric_surface_clearance_m'] for r in trace],label=name)
            ax[1,0].plot(times,[r['info']['evaluation']['formation_error_m'] for r in trace],label=name)
            ax[1,1].plot(times,np.cumsum([r['discount_factor']*r['reward'] for r in trace]),label=name)
        client=np.array([r['info']['client_position'][:2] for r in longest_trace]);ped=np.array([r['info']['pedestrian_position'][:2] for r in longest_trace]);ax[0,0].plot(client[:,0],client[:,1],'k:',label='client recorded prefix');ax[0,0].plot(ped[:,0],ped[:,1],'k--',label='pedestrian recorded prefix')
        ax[0,0].axis('equal');ax[0,0].set(xlabel='World x (m)',ylabel='World y (m)');ax[0,1].axhline(0,color='red',ls='--');ax[0,1].set(ylabel='Surrogate surface clearance (m)');ax[1,0].set(xlabel='Time (s)',ylabel='Formation error (m)');ax[1,1].set(xlabel='Time (s)',ylabel='Cumulative discounted reward')
        for a in ax.ravel():a.legend(fontsize=7)
        fig.suptitle(f'First successful-yield crossing paired with failed follow | seed {selected["seed"]}');fig.tight_layout();fig.savefig(out/'paired_crossing.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output');a=p.parse_args();summarize(a.root,a.output)
