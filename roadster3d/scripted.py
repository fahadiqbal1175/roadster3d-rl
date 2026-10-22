"""
Classical (NON-learned) reference controller. Used ONLY to (1) prove the nozzle layout has the authority to perform the maneuver,
(2) produce a phase-by-phase feasibility record, (3) optionally provide demonstrations. Results from this file are always
labelled 'scripted', never 'PPO'.
"""
import math
import numpy as np
from scipy.optimize import lsq_linear
from sim3d import *
from env3d import *

class ScriptedController:
    def __init__(self, env, kp=60.0, kd=14.0, a_burn=7.5, a_land=7.0, v_td=-1.0, v_launch=6.25, margin=1.1, lat_ctrl=None):
        self.env = env; self.kp = kp; self.kd = kd; self.a_burn = a_burn; self.a_land = a_land; self.v_td = v_td
        self.v_launch = v_launch; self.margin = margin
        self.lat_ctrl = (env.task != 'kickflip') if lat_ctrl is None else lat_ctrl
        s = env.sim
        self.A = np.vstack([s.nz_fdir.T, s.nz_tmat.T])      # 6 x n : [Fx Fy Fz tx ty tz]
        self.reset()

    def reset(self):
        self.landing_burn = False; self.phase = "launch"; self.launched = False
        self.skip_roll = getattr(self, 'skip_roll', False); self.hover_alt = getattr(self, 'hover_alt', None)

    def _alloc(self, wrench, weights):
        s = self.env.sim
        sc = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
        W = np.asarray(weights)/sc
        A = self.A*W[:, None]; b = np.asarray(wrench)*W
        reg = 1e-4*np.eye(s.nn)/1e4*0   # no hidden regularisation; tiny ridge below
        A2 = np.vstack([A, 1e-3*np.eye(s.nn)/s.fmax[None, :]*1.0]); b2 = np.concatenate([b, np.zeros(s.nn)])
        r = lsq_linear(A2, b2, bounds=(np.zeros(s.nn), s.fmax), method="bvls")
        return r.x

    def act(self):
        env = self.env; s = env.sim; t = s.t; m = s.mass; I = s.inertia
        R = q_to_R(s.q)
        q_ref, w_ref, a_ref, th = env.ref_att(t)
        if getattr(self, 'skip_roll', False):                   # exploit test: ignore the roll reference
            q_ref, w_ref, a_ref = env.q_ref0, np.zeros(3), np.zeros(3)
        rolling0 = abs(env.roll_total) > 0 and (env.phi*env.roll_sign < abs(env.roll_total) - 0.15)
        if self.lat_ctrl and not rolling0 and not (env.touch_t is not None):
            vby = float((R.T @ s.v)[1]); dlt = float(np.clip(0.10*vby, -0.25, 0.25))      # tilt the thrust vector to null body-lateral velocity
            q_ref = qmul(q_ref, q_axis_angle([1, 0, 0], dlt))
        e = q_err_vec(s.q, q_ref)
        tau = I*(self.kp*e + self.kd*(w_ref - s.w) + a_ref) + np.cross(s.w, I*s.w)
        lowest = s.lowest_point_z()
        Fz = 0.0; wz = 0.0
        task = env.task
        roll_total = abs(env.roll_total)
        rolling = roll_total > 0 and (env.phi*env.roll_sign < roll_total - 0.15) and not getattr(self, 'skip_roll', False)
        done_roll = (roll_total == 0) or not rolling
        if s.last['wheel_n'] > 0 and env.airborne and env.touch_t is not None:
            self.phase = "touchdown"; return -np.ones(s.nn)
        if task == "kickflip" and not self.launched:
            self.phase = "launch"
            if s.v[2] >= self.v_launch or t >= env.t_roll0 + 0.3:
                self.launched = True
            else:
                Fz = m*(G0 + self.a_burn)/max(R[2, 2], 0.5); wz = 1.0
        if task != "kickflip" or self.launched:
            if task == "kickflip" and rolling:
                self.phase = "roll"; wz = 0.0
            elif task == "roll360" and rolling:
                self.phase = "roll"; wz = 0.0
            else:
                self.phase = "recover/descent"
                h = max(lowest, 0.05); vz = s.v[2]
                if vz < 0:
                    a_req = (vz*vz - self.v_td**2)/(2*h)
                    if (not self.landing_burn) and a_req >= self.a_land*0.9:
                        self.landing_burn = True
                    if self.landing_burn:
                        self.phase = "landing burn"
                        if vz < self.v_td:
                            a_des = max(a_req*self.margin, 0.0)
                        else:
                            a_des = 3.0*(self.v_td - vz)        # already slow enough: hold the touchdown speed
                        Fz = m*(G0 + a_des)/max(R[2, 2], 0.5); wz = 1.0
        if getattr(self, 'hover_alt', None) is not None and self.launched:     # exploit test: hover instead of landing
            Fz = m*(G0 + 2.0*(self.hover_alt - s.p[2]) - 1.5*s.v[2])/max(R[2, 2], 0.5); wz = 1.0; self.phase = 'hover'
        wrench = [0, 0, Fz, tau[0], tau[1], tau[2]]
        weights = [0, 0, wz*1.0, 3.0, 3.0, 3.0]
        x = self._alloc(wrench, weights)
        return 2*np.clip(x/s.fmax, 0, 1) - 1

def run_scripted(task="kickflip", seed=0, record=True, **kw):
    env = RoadsterEnv(task, seed=seed, record=record, **kw)
    ctl = ScriptedController(env)
    o, _ = env.reset(seed=seed); ctl.reset()
    phases = []; info = {}
    while True:
        a = ctl.act(); phases.append(ctl.phase)
        o, r, te, tr, info = env.step(a)
        if te or tr: break
    return env, info, phases
