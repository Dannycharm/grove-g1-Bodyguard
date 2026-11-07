from dataclasses import replace
import numpy as np
from escort_rl.stage1_scenarios import load_config,sample_episode
from escort_rl.rewards import EpisodeEvaluator,formation_error


def fixture():
    c=load_config();e=sample_episode(1,'straight');client=e.client_pose_at(0)
    client=replace(client,x=0.,y=0.,yaw=0.)
    return c,client


def update(e,client,**kwargs):
    values=dict(elapsed=.2,dt=.2,robot_xy=[-1.5,0],client=client,robot_progress=-1.5,
        client_progress=0.,client_arrived=None,action=np.zeros(3))
    values.update(kwargs);return e.update(**values)


def test_formation_progress_no_standing_racing_or_repetition():
    c,client=fixture();e=EpisodeEvaluator(c,-1.5,4)
    r,_,_,info=update(e,replace(client,x=.2),client_progress=.2)
    assert r<=0 and info['new_robot_progress_m']==0
    _,_,_,info=update(e,replace(client,x=.2),robot_xy=[-1.3,0],robot_progress=-1.3,client_progress=.2)
    assert info['new_robot_progress_m']>0
    _,_,_,info=update(e,replace(client,x=.2),robot_xy=[-1.5,0],robot_progress=-1.5,client_progress=.2)
    assert info['new_robot_progress_m']==0
    _,_,_,info=update(e,replace(client,x=.2),robot_xy=[-1.3,0],robot_progress=-1.3,client_progress=.2)
    assert info['new_robot_progress_m']==0
    _,_,_,info=update(e,client,robot_xy=[4,0],robot_progress=4,client_progress=0)
    assert info['new_robot_progress_m']==0
    far=EpisodeEvaluator(c,-4.,4)
    _,_,_,info=update(far,client,robot_xy=[-3.2,0],robot_progress=-3.2)
    assert info['formation_error_m']<2 and info['new_robot_progress_m']==0


def test_outcomes_and_recovery_progress_guard():
    c,client=fixture()
    for flag,reason in [('fallen','fall'),('emergency','supervisor_failure'),('physics_failure','physics_failure'),('out_of_bounds','out_of_bounds')]:
        e=EpisodeEvaluator(c,-1.5,4);_,term,trunc,info=update(e,client,**{flag:True})
        assert term and not trunc and info['reason']==reason
    e=EpisodeEvaluator(c,-1.5,4)
    for i in range(15):result=update(e,replace(client,x=10),elapsed=(i+1)*.2)
    assert result[1] and result[3]['reason']=='client_lost'
    e=EpisodeEvaluator(c,-1.5,4)
    update(e,replace(client,x=10));update(e,client)
    assert e.loss_seconds==0
    e=EpisodeEvaluator(c,-1.5,4)
    _,term,trunc,info=update(e,client,elapsed=100.)
    assert not term and trunc and info['reason']=='time_limit'
    e=EpisodeEvaluator(c,-1.5,4)
    _,term,_,info=update(e,client,elapsed=20,client_arrived=20)
    assert not term # perfect formation alone is insufficient without actual robot progress
    _,term,_,info=update(e,client,elapsed=25,client_arrived=20)
    assert term and info['reason']=='recovery_timeout'
    e=EpisodeEvaluator(c,-1.5,4)
    _,term,trunc,info=update(e,replace(client,x=4),robot_xy=[2.5,0],robot_progress=2.5,
                           client_progress=4.,client_arrived=20,elapsed=20)
    assert term and not trunc and info['reason']=='success'


def test_racing_ahead_then_waiting_does_not_satisfy_escort_completion():
    c,client=fixture();e=EpisodeEvaluator(c,-1.5,4)
    update(e,client,robot_xy=[2.5,0],robot_progress=2.5,client_progress=0.)
    _,term,_,info=update(e,replace(client,x=4),robot_xy=[2.5,0],robot_progress=2.5,
                           client_progress=4.,client_arrived=20,elapsed=20)
    assert not term and info['accompanied_robot_progress_m']==0
    _,term,_,info=update(e,replace(client,x=4),robot_xy=[2.5,0],robot_progress=2.5,
                           client_progress=4.,client_arrived=20,elapsed=25)
    assert term and info['reason']=='recovery_timeout'
