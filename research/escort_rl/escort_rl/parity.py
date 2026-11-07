"""Comparisons reject shape/time mismatches and report every tensor."""
import numpy as np
from .contract import load_contract
from .measurements import VELOCITY_CONVENTION

def tensor_report(actual, reference):
    c=load_contract()
    a,b=np.asarray(actual),np.asarray(reference)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=311:
        raise ValueError('Expected equal [ticks,311] arrays for all ten outputs')
    report={}
    start=0
    for spec in c['outputs']:
        end=start+int(np.prod(spec['shape']))
        x,y=a[:,start:end],b[:,start:end]
        report[spec['name']]={'max_abs':float(np.max(np.abs(x-y))),
            'pass':bool(np.allclose(x,y,atol=1e-5,rtol=1e-5,equal_nan=False))}
        start=end
    return report

def trajectory_report(actual, reference):
    if len(actual)!=len(reference) or not actual:
        raise ValueError('Equal nonempty time-aligned trajectories required')
    for rows in (actual,reference):
        if any(row.get('velocity_convention')!=VELOCITY_CONVENTION for row in rows):
            raise ValueError('Explicit corrected pelvis-origin velocity convention required; convert legacy records first')
        for key,width in [('joint_position',29),('body_velocity',3),('position',3),
                          ('quaternion',4),('target',29),('kp',29),('kd',29)]:
            try: array=np.asarray([r[key] for r in rows],dtype=float)
            except (KeyError,ValueError,TypeError) as exc:raise ValueError(f'Missing/invalid reference signal: {key}') from exc
            if array.shape!=(len(rows),width) or not np.isfinite(array).all():
                raise ValueError(f'Incomplete/nonfinite reference signal: {key}')
        times=np.asarray([r['time'] for r in rows],float)
        if not np.isfinite(times).all() or (len(times)>1 and not (np.diff(times)>0).all()):
            raise ValueError('Finite, strictly increasing times required')
    if not np.allclose([x['time'] for x in actual],[x['time'] for x in reference],atol=1e-9,rtol=0):
        raise ValueError('Times differ: align recordings explicitly before comparison')
    def values(rows,key): return np.array([x[key] for x in rows])
    q=values(actual,'joint_position')-values(reference,'joint_position')
    v=values(actual,'body_velocity')[:,:2]-values(reference,'body_velocity')[:,:2]
    endpoint=np.linalg.norm(values(actual,'position')[-1,:2]-values(reference,'position')[-1,:2])
    def yaw(q):
        w,x,y,z=q
        return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    delta=yaw(actual[-1]['quaternion'])-yaw(reference[-1]['quaternion'])
    delta=abs(np.arctan2(np.sin(delta),np.cos(delta)))
    report=dict(joint_rms_rad=float(np.sqrt(np.mean(q*q))),velocity_rms_m_s=float(np.sqrt(np.mean(v*v))),
                endpoint_m=float(endpoint),yaw_difference_rad=float(delta),
                actual_failures=sum(x['fallen'] or x['emergency'] for x in actual),
                reference_failures=sum(x['fallen'] or x['emergency'] for x in reference))
    report['measurement_convention']=VELOCITY_CONVENTION
    report['angular_z_rms_rad_s']=float(np.sqrt(np.mean((values(actual,'body_velocity')[:,2]-values(reference,'body_velocity')[:,2])**2)))
    report['base_position_rms_m']=float(np.sqrt(np.mean((values(actual,'position')-values(reference,'position'))**2)))
    report['full_dynamics_parity_established']=False
    report['required_external_review']=['matching loaded model/checkpoint and initial conditions',
        'executed commands and command receipt timing','time/frame alignment','repeated-reference variability']
    report['controller']={}
    for key in ('target','kp','kd'):
        x,y=values(actual,key),values(reference,key)
        if x.shape!=y.shape: raise ValueError(f'{key} shape mismatch')
        report['controller'][key]={'max_abs':float(np.max(np.abs(x-y))),
            'pass':bool(np.allclose(x,y,atol=1e-5,rtol=0))}
    report['provisional_pass']=bool(all(x['pass'] for x in report['controller'].values()) and report['joint_rms_rad']<=.02 and report['velocity_rms_m_s']<=.05
        and endpoint<=.1 and delta<=.1 and report['actual_failures']<=report['reference_failures'])
    report['qualification']='endpoint gates intended for 10s; reference repeatability and provenance require operator review'
    return report


def observation_fixture_row(reset, *values):
    """Serialize exactly the float32 values consumed by AgilePolicy, not raw float64 sensors.

    Nine decimal digits round-trip float32. Rounding raw doubles to nine digits first
    can cross a float32 midpoint and alter feedback-driven inference comparisons.
    """
    row=np.concatenate([np.asarray([int(bool(reset))],np.float32)]+
                       [np.asarray(v,np.float32).ravel() for v in values])
    if row.shape!=(69,) or not np.isfinite(row).all():
        raise ValueError('Expected reset flag plus 68 finite observation floats')
    return row
