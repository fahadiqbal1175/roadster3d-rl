"""
Gymnasium environments for the Roadster-inspired 3D tasks (shared physics: sim3d.Sim3D).

Tasks: landing, recover90, recover180, roll360 (aerial), kickflip (ground launch, +360 deg body-X roll over an obstacle lineup),
       ramp_exit (A1: inverted ramp-exit conditions, about +180 deg body-X roll recovery; added in ramp.py)
Reward: dense tracking costs (always <= 0), potential-based shaping with the SAME gamma PPO uses (terminal potential = 0),
        large sparse terminal terms. Success terms are gated by genuine signed rotation progress.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sim3d import *

REST_Z = 0.8383          # CoM height when resting on the wheels (from the drop test)
OBSTACLES = [(11.0, 20.0, 1.6), (13.5, 17.0, 3.5)]   # (x0, x1, height m): car lineup + Semi-like block (ILLUSTRATIVE, placed around the flight apex)
OBST_HALF_WIDTH = 2.5

def roll_profile(t, t0, T, total):
    """bang-bang (triangular velocity) roll profile about body X: theta, theta_dot, theta_ddot"""
    x = t - t0
    if total == 0.0 or x <= 0.0:
        return 0.0, 0.0, 0.0
    if x >= T:
        return total, 0.0, 0.0
    a = 4*total/T**2
    if x < T/2:
        return 0.5*a*x*x, a*x, a
    y = T - x
    return total - 0.5*a*y*y, a*y, -a

def build_kick_reference(v_l=7.0, a_b=7.5, T_roll=1.4, a_land=7.0, v_td=-1.0, dt=0.002):
    """Piecewise-constant-acceleration height reference: launch burn, ballistic, landing burn. Returns arrays (t, z, vz)."""
    t_b = v_l/a_b
    ts, zs, vs = [0.0], [0.0], [0.0]
    t = 0.0; z = 0.0; v = 0.0; phase = 0
    while t < 8.0:
        if phase == 0:
            a = a_b
            if v >= v_l: phase = 1
        if phase == 1:
            a = -G0
            if v < 0 and (v*v - v_td*v_td)/(2*a_land) >= z and t > t_b + T_roll*0.8: phase = 2
        if phase == 2:
            a = a_land
        v += a*dt; z += v*dt; t += dt
        if phase == 2 and (z <= 0.0 or v >= v_td):
            z = max(z, 0.0); ts.append(t); zs.append(z); vs.append(v); break
        ts.append(t); zs.append(z); vs.append(v)
    return np.array(ts), np.array(zs) + REST_Z, np.array(vs)

KICK_REF = build_kick_reference()
import os as _os
_p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'kick_ref.npz')
if _os.path.exists(_p):                      # height reference taken from the physically simulated scripted run (feasible by construction)
    _d = np.load(_p); KICK_REF = (_d['t'], _d['z'], _d['vz'])

TASKS = dict(
    landing=dict(T_max=6.0, roll_total=0.0),
    recover90=dict(T_max=7.0, roll_total=0.0),
    recover180=dict(T_max=7.0, roll_total=0.0),
    roll360=dict(T_max=7.0, roll_total=2*math.pi),
    kickflip=dict(T_max=5.0, roll_total=2*math.pi),
)

class RoadsterEnv(gym.Env):
    metadata = {"render_modes": []}
    DT = 0.02

    def __init__(self, task="kickflip", cfg=None, gamma=0.99, seed=None, T_roll=1.4, t_roll0=None, roll_sign=1.0,
                 randomize_cfg=0.0, record=False, cfg_factory=None):
        super().__init__()
        self.task = task; self.spec = dict(TASKS.get(task, {}))
        self.cfg = cfg or VehicleConfig()
        self.sim = Sim3D(self.cfg)
        self.gamma = gamma; self.T_roll = T_roll; self.roll_sign = roll_sign
        self.randomize_cfg = randomize_cfg; self.record = record
        self.t_roll0_fixed = t_roll0
        self.rng = np.random.default_rng(seed)
        self.n_nz = self.sim.nn
        self.obs_dim = 54 if self.n_nz == 10 else 44 + self.n_nz
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_nz,), np.float32)
        self.base = dict(I0=self.sim.I0.copy(), m0=self.sim.m0, fmax=self.sim.fmax.copy())

    # ------------------------------------------------------------ reference
    def ref_att(self, t):
        th, thd, thdd = roll_profile(t, self.t_roll0, self.T_roll, self.roll_total)
        q = qmul(self.q_ref0, q_axis_angle([1, 0, 0], th))
        return q, np.array([thd, 0, 0]), np.array([thdd, 0, 0]), th

    def ref_pos(self, t):
        if self.task == "kickflip":
            T, Z, V = KICK_REF
            z = np.interp(t, T, Z); vz = np.interp(t, T, V)
            return np.array([self.x0 + self.vx0*t, 0.0, z]), np.array([self.vx0, 0.0, vz])
        return None, None

    # ------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng; task = self.task
        # optional domain randomisation of mass / inertia / thrust (sensitivity studies)
        k = self.randomize_cfg
        s = self.sim
        s.m0 = self.base['m0']*(1 + k*rng.uniform(-1, 1)); s.I0 = self.base['I0']*(1 + k*rng.uniform(-1, 1, 3))
        s.fmax = self.base['fmax']*(1 + k*rng.uniform(-1, 1))
        self.roll_total = self.spec['roll_total']*self.roll_sign
        yaw0 = rng.uniform(-0.05, 0.05)
        q_up = q_from_euler_zyx(yaw0, 0, 0)
        self.t_roll0 = 0.0; self.x0 = 0.0; self.vx0 = 0.0
        if task == "landing":
            p = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(3.0, 8.0))
            v = (rng.uniform(-3, 3), rng.uniform(-1.5, 1.5), rng.uniform(-6, 0))
            q = qmul(q_up, q_expmap(rng.normal(0, 0.08, 3))); w = rng.normal(0, 0.15, 3)
            self.q_ref0 = q_up; gas = None
        elif task in ("recover90", "recover180"):
            off = (math.pi/2 if task == "recover90" else math.pi)*rng.choice([-1, 1])
            if task == "recover180": off = math.pi*rng.uniform(0.9, 1.0)*rng.choice([-1, 1])
            hlo, hhi = (10, 14) if task == 'recover90' else (16, 22)      # an inverted car needs the altitude to roll over AND brake
            p = (rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(hlo, hhi))
            v = (rng.uniform(-3, 3), rng.uniform(-1, 1), rng.uniform(-2, 3))
            q = qmul(qmul(q_up, q_axis_angle([1, 0, 0], off)), q_expmap(rng.normal(0, 0.05, 3)))
            w = np.array([rng.normal(0, 0.3), rng.normal(0, 0.15), rng.normal(0, 0.15)])
            self.q_ref0 = q_up
        elif task == "roll360":
            p = (rng.uniform(-2, 2), rng.uniform(-1, 1), rng.uniform(11, 15))
            v = (rng.uniform(-2, 2), rng.uniform(-0.5, 0.5), rng.uniform(2, 6))
            q = qmul(q_up, q_expmap(rng.normal(0, 0.03, 3))); w = rng.normal(0, 0.05, 3)
            self.q_ref0 = q_up; self.t_roll0 = self.t_roll0_fixed if self.t_roll0_fixed is not None else 0.1
        elif task == "kickflip":
            self.vx0 = rng.uniform(8.0, 10.0); self.x0 = rng.uniform(-1.0, 1.0)
            p = (self.x0, rng.uniform(-0.15, 0.15), REST_Z)
            v = (self.vx0, rng.normal(0, 0.1), 0.0)
            q = q_up; w = np.zeros(3); self.q_ref0 = q_up
            self.t_roll0 = self.t_roll0_fixed if self.t_roll0_fixed is not None else 0.95 + rng.uniform(-0.05, 0.05)
        else:
            raise ValueError(task)
        self._extra_reset(rng)  # hook for ramp tasks
        if task in ("landing", "recover90", "recover180", "roll360", "kickflip"):
            s.reset(p=p, v=v, q=q, w=w)
        self.sim.t = 0.0
        self.phi = 0.0; self._twist_prev = 0.0
        self.airborne = False; self.touch_t = None; self.touch = None
        self.crashed = False; self.collided = False; self.landed = False
        self.min_clear = 99.0
        self.step_n = 0; self.u_prev = np.zeros(self.n_nz); self.hist = [] if self.record else None
        self.max_phi = 0.0; self.ret = 0.0; self.maxz = float(self.sim.p[2])
        self.prev_pot = self._potential(self._state_feats())
        return self._obs(), {}

    def _extra_reset(self, rng):
        pass

    # ------------------------------------------------------------ features
    def _update_progress(self):
        qr = qmul(qconj(self.q_ref0), self.sim.q)
        if qr[0]**2 + qr[1]**2 > 1e-4:
            tw = 2*math.atan2(qr[1], qr[0])
            self.phi += wrap(tw - self._twist_prev); self._twist_prev = tw
        self.max_phi = max(self.max_phi, self.phi*self.roll_sign)

    def _state_feats(self):
        s = self.sim; t = s.t
        q_ref, w_ref, a_ref, th_ref = self.ref_att(t)
        e_att = geodesic(s.q, q_ref)
        return dict(q_ref=q_ref, w_ref=w_ref, th_ref=th_ref, e_att=e_att)

    def _potential(self, f):
        """>= 0, bounded, evaluated on non-terminal states; terminal potential is defined as 0 in step()."""
        goal = abs(self.roll_total)
        prog = 0.0
        if goal > 0:
            prog = min(1.0, max(0.0, self.phi*self.roll_sign/goal))
        att = math.exp(-(f['e_att']/0.5)**2)
        return 12.0*prog + 6.0*att

    def _clearance(self):
        """vertical clearance (m) of the lowest car point over any obstacle it is above; large if none"""
        s = self.sim; R = q_to_R(s.q)
        pts = s.p + s.cp_body @ R.T
        pts[:4, 2] -= s.cp_rad[:4]
        best = 99.0; hit = False
        for (x0, x1, h) in self.obstacles():
            m = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (np.abs(pts[:, 1]) < OBST_HALF_WIDTH)
            if m.any():
                c = float(np.min(pts[m, 2]) - h)
                best = min(best, c)
                if c < 0: hit = True
        return best, hit

    def obstacles(self):
        return OBSTACLES if self.task == "kickflip" else []

    def _obs(self):
        s = self.sim; R = q_to_R(s.q); t = s.t
        f = self._state_feats()
        e_vec = q_err_vec(s.q, f['q_ref'])
        p_ref, v_ref = self.ref_pos(t)
        if p_ref is None:
            ep = np.zeros(3); ev = np.zeros(3)
        else:
            ep = p_ref - s.p; ev = v_ref - s.v
        clear, _ = self._clearance() if self.obstacles() else (99.0, False)
        nxt = [x0 for (x0, x1, h) in self.obstacles() if x1 > s.p[0]]
        dx = (min(nxt) - s.p[0]) if nxt else 40.0
        th_ref = f['th_ref']
        o = np.concatenate([
            R.flatten(), e_vec/math.pi, s.w/5.0, f['w_ref']/5.0, (R.T @ s.v)/10.0, s.v/10.0, [s.p[2]/10.0],
            ep/5.0, ev/5.0, (R.T @ ep)/5.0,
            [self.phi*self.roll_sign/(2*math.pi), th_ref*self.roll_sign/(2*math.pi)] if True else [],
            [s.t/self.spec['T_max']], [s.gas/s.cfg.gas0], s.f/s.fmax,
            [s.last['wheel_n']/4.0, float(s.last['chassis_n'] > 0)],
            [min(clear, 10.0)/5.0, np.clip(dx, -10, 40)/20.0],
            [float(self.airborne), float(self.landed)]])
        return o.astype(np.float32)

    # ------------------------------------------------------------ step
    def step(self, action):
        s = self.sim
        a = np.clip(np.asarray(action, float), -1, 1)
        u = 0.5*(a + 1.0)
        s.step(u, 4)
        self._update_progress()
        self.step_n += 1
        t = s.t
        lowest = s.lowest_point_z()
        wheel_n = s.last['wheel_n']; chassis_n = s.last['chassis_n']
        if lowest > 0.15: self.airborne = True
        self.maxz = max(self.maxz, s.p[2])
        clear, hit = self._clearance() if self.obstacles() else (99.0, False)
        self.min_clear = min(self.min_clear, clear)
        if hit: self.collided = True
        if s.last['max_pen_chassis'] > s.cfg.crash_pen: self.crashed = True
        if self.airborne and wheel_n > 0 and self.touch_t is None:
            R = q_to_R(s.q)
            self.touch_t = t
            self.touch = dict(vz=float(s.v[2]), vh=float(np.hypot(s.v[0], s.v[1])), w=s.w.copy(), tilt=tilt_angle(s.q),
                              phi=self.phi, v_body=(R.T @ s.v).copy(), p=s.p.copy())
        f = self._state_feats()
        # ---------------- costs
        e_att = f['e_att']; e_w = float(np.linalg.norm(s.w - f['w_ref']))
        p_ref, v_ref = self.ref_pos(t)
        cost = 1.0*(e_att/0.7)**2 + 0.15*(e_w/3.0)**2
        if p_ref is not None:
            ep = p_ref - s.p
            cost += 0.25*((np.hypot(ep[0], ep[1])/4.0)**2 + (ep[2]/2.5)**2)
        phase_done = (abs(self.roll_total) == 0) or (self.phi*self.roll_sign > abs(self.roll_total) - 0.5)
        if phase_done:    # descent-speed shaping: allowed downward speed shrinks with height
            h = max(lowest, 0.0)
            v_allow = 1.0 + math.sqrt(2*6.0*h)
            cost += 0.3*(max(0.0, -s.v[2] - v_allow)/3.0)**2
        cost += 0.02*float(np.mean(u**2)) + 0.02*float(np.mean((u - self.u_prev)**2))*10
        self.u_prev = u
        r = -0.05*min(cost, 4.0)
        # ---------------- termination
        terminated = False; truncated = False; success = False; reason = ""
        if self.crashed:
            terminated = True; reason = "chassis_contact"; r -= 50.0
        elif self.collided:
            terminated = True; reason = "obstacle_collision"; r -= 50.0
        elif self.touch_t is not None:
            maneuver_done = (abs(self.roll_total) == 0) or (self.phi*self.roll_sign > abs(self.roll_total) - math.radians(30))
            if not maneuver_done:
                terminated = True; reason = "touchdown_before_maneuver_complete"; r -= 40.0
            elif t - self.touch_t >= 0.6:
                terminated = True; self.landed = True
                success, info_s = self._success()
                reason = "landed"; r += 60.0 if success else -10.0
        elif s.p[2] > 40 or abs(s.p[0]) > 80 or abs(s.p[1]) > 40:
            terminated = True; reason = "out_of_bounds"; r -= 50.0
        if not terminated and t >= self.spec['T_max'] - 1e-9:
            truncated = True; reason = "timeout"
        # ---------------- potential-based shaping: gamma*Phi(s') - Phi(s), terminal potential = 0
        pot = 0.0 if terminated else self._potential(f)
        r += self.gamma*pot - self.prev_pot
        self.prev_pot = pot
        self.ret += r
        info = {}
        if self.record:
            self.hist.append(dict(t=t, p=s.p.copy(), v=s.v.copy(), q=s.q.copy(), w=s.w.copy(), f=s.f.copy(), u=u.copy(),
                                  phi=self.phi, th_ref=f['th_ref'], e_att=e_att, gas=s.gas, wheel_n=wheel_n, chassis_n=chassis_n,
                                  q_ref=f['q_ref'].copy(), clear=clear, r=r))
        if terminated or truncated:
            info = self._metrics(success, reason)
            info['TimeLimit.truncated'] = bool(truncated)
        return self._obs(), float(r), terminated, truncated, info

    # ------------------------------------------------------------ evaluation
    def _success(self):
        s = self.sim; tch = self.touch
        f = self._state_feats()
        tilt = tilt_angle(s.q); wn = float(np.linalg.norm(s.w))
        vz_ok = -tch['vz'] <= 1.5
        slip_ok = abs(tch['v_body'][1]) <= 1.5
        head_err = abs(wrap(heading(s.q) - heading(self.q_ref0)))
        roll_ok = (abs(self.roll_total) == 0) or abs(self.phi*self.roll_sign - abs(self.roll_total)) < math.radians(30)
        checks = dict(tilt_ok=tilt < math.radians(10), rate_ok=wn < 0.3, vz_ok=vz_ok, slip_ok=slip_ok,
                      heading_ok=head_err < math.radians(20), roll_ok=roll_ok, wheels_ok=s.last['wheel_n'] >= 3,
                      no_chassis=not self.crashed, no_collision=not self.collided, gas_ok=s.gas > 0)
        return all(checks.values()), checks

    def _metrics(self, success, reason):
        s = self.sim
        ok, checks = (self._success() if self.touch is not None else (False, {}))
        tch = self.touch or {}
        m = dict(success=bool(success), reason=reason, task=self.task, t_end=s.t,
                 maneuver_progress_deg=math.degrees(self.phi*self.roll_sign), max_progress_deg=math.degrees(self.max_phi),
                 maneuver_complete=bool((abs(self.roll_total) == 0) or self.max_phi > abs(self.roll_total) - math.radians(30)),
                 final_tilt_deg=math.degrees(tilt_angle(s.q)), final_rate=float(np.linalg.norm(s.w)),
                 touch_vz=tch.get('vz', float('nan')), touch_vh=tch.get('vh', float('nan')),
                 touch_rates=(tch['w'].tolist() if 'w' in tch else None), touch_tilt_deg=math.degrees(tch['tilt']) if 'tilt' in tch else float('nan'),
                 wheel_contacts=s.last['wheel_n'], chassis_contact=bool(self.crashed), obstacle_collision=bool(self.collided),
                 min_obstacle_clearance=self.min_clear, gas_left=s.gas, peak_height=self.maxz - REST_Z, ret=self.ret,
                 heading_err_deg=math.degrees(abs(wrap(heading(s.q) - heading(self.q_ref0)))), checks=checks)
        p_ref, _ = self.ref_pos(s.t)
        m['final_xy'] = s.p[:2].tolist()
        return m
