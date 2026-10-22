"""
Task A1 (Roadster Ramp Recovery, airborne part): start from ramp-EXIT conditions, recover about +180 deg of body-X roll, land on the wheels.
The ramp's contribution is represented ONLY through the initial state (about -170 deg of pitch, i.e. nose-up past vertical, high altitude,
some pitch rate). The full ramp contact stage (A2) is not part of this environment.
Heading: a roll about body X leaves the horizontal heading of body +X unchanged, so after ramp pitch (about 170 deg) the car faces
back toward where it came from; the target heading is that post-ramp heading (heading(q_exit)), not the pre-ramp heading.
"""
import math
import numpy as np
from sim3d import *
from env3d import *

TASKS['ramp_exit'] = dict(T_max=6.0, roll_total=math.pi)

class RampExitEnv(RoadsterEnv):
    def __init__(self, **kw):
        kw.setdefault('T_roll', 1.2)
        super().__init__(task="ramp_exit", **kw)

    def reset(self, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        rng = self.rng; s = self.sim
        k = self.randomize_cfg
        s.m0 = self.base['m0']*(1 + k*rng.uniform(-1, 1)); s.I0 = self.base['I0']*(1 + k*rng.uniform(-1, 1, 3))
        s.fmax = self.base['fmax']*(1 + k*rng.uniform(-1, 1))
        self.roll_total = self.spec['roll_total']*self.roll_sign
        yaw0 = rng.uniform(-0.1, 0.1)
        pitch = -math.radians(170 + rng.uniform(-8, 8))                  # nose-up past vertical (ramp-induced), reported ~170 deg
        q = qmul(qmul(q_axis_angle([0, 0, 1], yaw0), q_axis_angle([0, 1, 0], pitch)), q_expmap(rng.normal(0, 0.03, 3)))
        self.q_exit = q; self.q_ref0 = q
        self.t_roll0 = 0.15
        q_end = qmul(q, q_axis_angle([1, 0, 0], self.roll_total))
        q_lvl = q_from_euler_zyx(heading(q_end), 0, 0)
        self.q_corr = qmul(q_lvl, qconj(q_end)); 
        if self.q_corr[0] < 0: self.q_corr = -self.q_corr
        self.x0 = 0.0; self.vx0 = 0.0
        p = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(18.0, 21.0))
        v = (rng.uniform(-3, 3), rng.uniform(-1, 1), rng.uniform(-3, 5))
        w = np.array([rng.normal(0, 0.15), rng.uniform(-0.8, 0.0), rng.normal(0, 0.15)])   # residual ramp pitch rate (nose-up sense)
        s.reset(p=p, v=v, q=q, w=w)
        self.sim.t = 0.0
        self.phi = 0.0; self._twist_prev = 0.0
        self.airborne = True; self.touch_t = None; self.touch = None
        self.crashed = False; self.collided = False; self.landed = False; self.min_clear = 99.0
        self.step_n = 0; self.u_prev = np.zeros(self.n_nz); self.hist = [] if self.record else None
        self.max_phi = 0.0; self.ret = 0.0; self.maxz = float(self.sim.p[2])
        self.prev_pot = self._potential(self._state_feats())
        return self._obs(), {}

    def ref_att(self, t):
        th, thd, thdd = roll_profile(t, self.t_roll0, self.T_roll, self.roll_total)
        frac = th/self.roll_total if self.roll_total != 0 else 0.0
        qc = q_expmap(frac*self._qlog(self.q_corr))
        q = qmul(qc, qmul(self.q_exit, q_axis_angle([1, 0, 0], th)))
        return q, np.array([thd, 0, 0]), np.array([thdd, 0, 0]), th

    @staticmethod
    def _qlog(q):
        s = math.sqrt(q[1]**2 + q[2]**2 + q[3]**2)
        if s < 1e-12: return np.zeros(3)
        return q[1:4]*(2*math.atan2(s, q[0])/s)

    def obstacles(self): return []

    def _success(self):
        ok, checks = super()._success()
        return ok, checks
