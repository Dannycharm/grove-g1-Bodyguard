"""Unmodified ONNX graph; scaling and normalization live inside the graph."""
import numpy as np
from .contract import ROOT, load_contract, sha256

class FrozenGait:
    def __init__(self):
        import onnxruntime as ort
        if ort.__version__ != '1.20.1':
            raise RuntimeError(f'Reference ORT is 1.20.1, found {ort.__version__}')
        self.contract = load_contract()
        c = self.contract
        path = ROOT / c['checkpoint']
        if sha256(path) != c['checkpoint_sha256']:
            raise ValueError('Checkpoint mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        for actual, expected in ((self.session.get_inputs(), c['inputs']),
                                 (self.session.get_outputs(), c['outputs'])):
            if len(actual) != len(expected):
                raise ValueError('ONNX tensor count mismatch')
            for a, e in zip(actual, expected):
                if a.name != e['name'] or a.shape != e['shape'] or a.type != 'tensor(float)':
                    raise ValueError(f'ONNX signature mismatch: {a.name}')
        self.reset()

    def reset(self):
        self.state = {x['name']: np.zeros(x['shape'], np.float32)
                      for x in self.contract['inputs'][5:]}
        self.calls = 0
        self.last_outputs = None

    def run(self, quat, angular_velocity, command, positions, velocities):
        values = [quat, angular_velocity, command, positions, velocities]
        feed = {x['name']: np.asarray(v, np.float32).reshape(x['shape'])
                for x, v in zip(self.contract['inputs'][:5], values)}
        if not all(np.isfinite(v).all() for v in feed.values()):
            raise ValueError('Nonfinite gait input')
        feed.update(self.state)
        outputs = self.session.run([x['name'] for x in self.contract['outputs']], feed)
        if not all(np.isfinite(v).all() for v in outputs):
            raise RuntimeError('Nonfinite gait output')
        self.state = {x['name']: v.copy() for x, v in
                      zip(self.contract['inputs'][5:], outputs[3:])}
        self.last_outputs = [v.copy() for v in outputs]
        self.calls += 1
        return tuple(v.reshape(-1).astype(np.float64) for v in outputs[:3])
