"""Reference equations with an explicit synchronous event order."""
import numpy as np

class Schedule:
    """Integer nanoseconds: events due at or before physics boundary; no drift."""
    def __init__(self, timestep):
        self.dt_ns = round(timestep * 1e9)
        if abs(self.dt_ns / 1e9 - timestep) > 1e-12:
            raise ValueError('Timestep must resolve to integer nanoseconds')
        self.reset()

    def reset(self):
        self.step = 0
        self.next_controller = 5_000_000
        self.next_bridge = 0

    def events(self):
        now = self.step * self.dt_ns
        events = []
        while min(self.next_controller, self.next_bridge) <= now:
            # Controller precedes bridge on ties. First gait at 20 ms.
            if self.next_controller <= self.next_bridge:
                events.append(('controller', self.next_controller / 1e9))
                self.next_controller += 5_000_000
            else:
                events.append(('bridge', self.next_bridge / 1e9))
                self.next_bridge += 1_000_000
        self.step += 1
        return events

class ControllerReference:
    def __init__(self, names, action_names):
        self.names = list(names)
        self.action_indices = np.array([self.names.index(n) for n in action_names])

    def reset(self, position):
        self.activation = np.array(position, float).copy()
        self.target = self.activation.copy()
        self.kp = np.full(len(self.names), 300.)
        self.kd = np.full(len(self.names), 4.)
        waist = self.names.index('waist_yaw_joint')
        self.kp[waist], self.kd[waist] = 10., 1.
        self.kp[self.action_indices], self.kd[self.action_indices] = 100., 2.
        self.raw = None
        self.blend = 0.
        self.emergency = False
        self.tick = 0
        self.requested = np.zeros(3)
        self.command_stamp = None
        self.applied = np.zeros(3, np.float32)

    def command(self, value, time):
        value = np.asarray(value, float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Expected finite (vx, vy, yaw_rate)')
        self.requested = value.copy()
        self.command_stamp = time

    def command_at(self, time):
        stale = self.command_stamp is None or time - self.command_stamp > .5
        self.applied = (np.zeros(3) if stale else
                        np.clip(self.requested, [-.6, -.6, -.9], [.6, .6, .9])).astype(np.float32)
        return self.applied.copy()

    def update(self, velocity, infer):
        self.tick += 1
        if self.tick % 4 == 0:
            self.raw = infer()
        speed = np.abs(np.asarray(velocity)[self.action_indices])
        self.emergency |= not np.isfinite(speed).all() or speed.mean() > 10 or speed.max() > 35
        idx = self.action_indices
        if self.emergency:
            self.kp[idx], self.kd[idx] = 100., 2.
            return
        self.blend = min(1., self.blend + 20 * .005)
        if self.raw is not None:
            position, kp, kd = self.raw
            if not all(np.isfinite(x).all() for x in self.raw):
                raise ValueError('Nonfinite reference')
            self.target[idx] = self.activation[idx] + self.blend * (position - self.activation[idx])
            self.kp[idx], self.kd[idx] = kp, kd

    def torque(self, position, velocity):
        # LowCmd serializes target and gains as float32; simulator arithmetic is double.
        q = self.target.astype(np.float32).astype(float)
        kp = self.kp.astype(np.float32).astype(float)
        kd = self.kd.astype(np.float32).astype(float)
        return kp * (q - position) - kd * velocity
