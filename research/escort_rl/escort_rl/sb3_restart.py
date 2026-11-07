"""Versioned SB3 2.7.0 restart boundary. No physics/partial-rollout restoration.

SB3 has no public API to cancel a queued seed (seed(None) schedules a random seed).
Keep that exact-version private handling here, never scattered through training code.
"""
import copy
import importlib.metadata
import random
import numpy as np
import torch

PROTOCOL = 'sb3_2_7_0_scenario_restart_v2'


def clear_pending_resets(vec):
    if importlib.metadata.version('stable-baselines3') != '2.7.0':
        raise RuntimeError('Restart adapter is tested only against SB3 2.7.0')
    if not all(hasattr(vec, name) for name in ('_reset_seeds','_reset_options','_seeds','_options')):
        raise RuntimeError('Unsupported SB3 VecEnv reset protocol')
    vec._reset_seeds()
    vec._reset_options()
    if vec._seeds != [None]*vec.num_envs or vec._options != [{} for _ in range(vec.num_envs)]:
        raise RuntimeError('Failed to cancel queued reset seed/options')


def validate_stream(state):
    if not isinstance(state,dict) or set(state) != {'rng','episodes'}:
        raise ValueError('Missing or unknown worker RNG state fields')
    if type(state['episodes']) is not int or state['episodes'] < 0:
        raise ValueError('Invalid worker episode count')
    probe=np.random.default_rng()
    try:probe.bit_generator.state=copy.deepcopy(state['rng'])
    except (ValueError,TypeError,KeyError) as exc:raise ValueError('Invalid worker RNG state') from exc
    return copy.deepcopy(state)


def restore_checkpoint_rng(model,vec,state):
    required={'torch','numpy','python','environment_streams','timesteps','resume_limitation'}
    if not isinstance(state,dict) or not required <= set(state):
        raise ValueError('Incomplete checkpoint RNG state')
    if state['timesteps'] != model.num_timesteps:
        raise ValueError('Checkpoint/RNG timestep mismatch')
    if len(state['environment_streams']) != vec.num_envs:
        raise ValueError('Checkpoint worker count mismatch')
    streams=[validate_stream(s) for s in state['environment_streams']]
    # Validate all worker states before changing any worker. Any error aborts the run.
    clear_pending_resets(vec)
    for i,s in enumerate(streams):vec.env_method('set_rng_state',s,indices=i)
    if vec.env_method('rng_state') != streams:raise RuntimeError('Worker RNG restoration mismatch')
    # Loading/setup has finished. No later setup/seed call is allowed before learn().
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    return PROTOCOL
