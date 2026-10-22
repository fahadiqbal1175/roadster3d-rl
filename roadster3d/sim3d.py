"""
Roadster-inspired 3D rigid-body simulator (6-DOF) with a configurable nozzle table.

Conventions
  body +X front, +Y left, +Z roof;  world +Z up;  quaternion q = (w,x,y,z) maps body -> world.
Nozzle convention (IMPORTANT)
  Each nozzle stores its EXHAUST direction e_i (where the gas plume goes, body frame).
  The force on the vehicle is the REACTION:  F_i = -f_i * e_i   (f_i >= 0).
  Torque about the centre of mass:           T_i =  r_i x F_i.
  A -Z exhaust (plume pointing at the floor) therefore pushes the car along +Z (up when upright).
All numbers below are ILLUSTRATIVE assumptions, not verified Tesla values.
"""
import math
from dataclasses import dataclass, field
import numpy as np

G0 = 9.80665

# ---------------------------------------------------------------- quaternions
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qnorm(q):
    return q / math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

def q_axis_angle(axis, ang):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    s = math.sin(0.5*ang) / n
    return np.array([math.cos(0.5*ang), axis[0]*s, axis[1]*s, axis[2]*s])

def q_expmap(rv):
    """quaternion of rotation vector rv (rad)"""
    th = math.sqrt(rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2])
    if th < 1e-9:
        return np.array([1.0, 0.5*rv[0], 0.5*rv[1], 0.5*rv[2]])
    s = math.sin(0.5*th) / th
    return np.array([math.cos(0.5*th), rv[0]*s, rv[1]*s, rv[2]*s])

def q_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                     [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])

def q_from_euler_zyx(yaw, pitch, roll):
    """for initial conditions / logging only (intrinsic z-y-x)"""
    return qmul(qmul(q_axis_angle([0, 0, 1], yaw), q_axis_angle([0, 1, 0], pitch)),
                q_axis_angle([1, 0, 0], roll))

def q_err_vec(q, q_ref):
    """rotation vector (body frame) that takes q to q_ref, shortest way"""
    qe = qmul(qconj(q), q_ref)
    if qe[0] < 0:
        qe = -qe
    s = math.sqrt(qe[1]**2 + qe[2]**2 + qe[3]**2)
    if s < 1e-9:
        return 2*qe[1:4]
    ang = 2*math.atan2(s, qe[0])
    return qe[1:4] * (ang / s)

def geodesic(q, q_ref):
    d = abs(float(np.dot(q, q_ref)))
    return 2*math.acos(min(1.0, d))

def tilt_angle(q):
    """angle between body +Z and world +Z (0 = upright, pi = inverted)"""
    R = q_to_R(q)
    return math.acos(max(-1.0, min(1.0, R[2, 2])))

def heading(q):
    """yaw of body +X projected on the horizontal plane. Invariant under a roll about body X."""
    R = q_to_R(q)
    return math.atan2(R[1, 0], R[0, 0])

def wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

# ---------------------------------------------------------------- configuration
def default_nozzles():
    """L1: HYPOTHETICAL 10-nozzle layout. name, group, position (m, body), EXHAUST dir, Fmax (N)."""
    rows = []
    for i, y in enumerate([-0.7, -0.3, 0.3, 0.7]):
        rows.append((f"rear{i}", "rear", (-2.1, y, 0.0), (-1, 0, 0), 20000.0))     # plume rearward -> force forward
    for name, x, y in [("cornerFL", 1.7, 0.9), ("cornerFR", 1.7, -0.9),
                       ("cornerRL", -1.7, 0.9), ("cornerRR", -1.7, -0.9)]:
        rows.append((name, "corner", (x, y, -0.20), (0, 0, -1), 8000.0))              # plume to floor -> force up
    rows.append(("yawL", "yaw", (1.9, 0.6, 0.0), (0, 1, 0), 3000.0))                  # plume left -> force right
    rows.append(("yawR", "yaw", (1.9, -0.6, 0.0), (0, -1, 0), 3000.0))                # plume right -> force left
    return rows

@dataclass
class VehicleConfig:
    m_total0: float = 1700.0        # kg, including gas at start (assumption)
    gas0: float = 250.0             # kg, reported at the event (assumed usable)
    dims: tuple = (4.3, 2.0, 1.3)   # L, W, H  (m)  illustrative
    inertia: tuple = None           # None -> uniform box
    nozzles: list = field(default_factory=default_nozzles)
    valve_tau: float = 0.03         # s, assumed
    isp: float = 70.0               # s, assumed cold-gas Isp
    gas_depletion: bool = True      # mass and inertia scale with remaining gas
    wheel_radius: float = 0.33
    wheel_xy: tuple = (1.35, 0.9)   # |x|, |y| of wheel centres
    wheel_z: float = -0.55          # wheel centre below CoM; gives ~0.23 m of landing-gear travel before the chassis touches (assumption)
    k_wheel: float = 1.0e5
    c_wheel: float = 6.0e3
    k_chassis: float = 5.0e5
    c_chassis: float = 2.0e4
    mu_wheel_lat: float = 0.9
    mu_wheel_roll: float = 0.03
    mu_chassis: float = 0.6
    v_eps: float = 0.1
    pen_tol: float = 1e-4           # m: contact detection tolerance (numerical)
    crash_pen: float = 0.02         # m: chassis penetration that counts as a crash
    dt: float = 0.005

    def box_inertia(self):
        L, W, H = self.dims; m = self.m_total0
        return (m/12*(W*W + H*H), m/12*(L*L + H*H), m/12*(L*L + W*W))

    def nozzle_table(self):
        import pandas as pd
        return pd.DataFrame([dict(name=n, group=g, x=p[0], y=p[1], z=p[2], ex=e[0], ey=e[1], ez=e[2], fmax_N=f)
                             for n, g, p, e, f in self.nozzles])

class Sim3D:
    def __init__(self, cfg: VehicleConfig = None):
        self.cfg = cfg or VehicleConfig()
        self.configure()

    def configure(self):
        c = self.cfg
        self.m0 = c.m_total0
        self.I0 = np.array(c.inertia if c.inertia is not None else c.box_inertia(), float)
        pos = np.array([n[2] for n in c.nozzles], float)
        ex = np.array([n[3] for n in c.nozzles], float)
        ex /= np.linalg.norm(ex, axis=1, keepdims=True)
        self.nz_pos = pos
        self.nz_exhaust = ex
        self.nz_fdir = -ex                                   # reaction direction on the vehicle
        self.nz_tmat = np.cross(pos, self.nz_fdir)           # torque per newton
        self.fmax = np.array([n[4] for n in c.nozzles], float)
        self.nn = len(c.nozzles)
        L, W, H = c.dims
        corners = np.array([[sx*L/2, sy*W/2, sz*H/2] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)])
        wx, wy = c.wheel_xy
        wheels = np.array([[wx, wy, c.wheel_z], [wx, -wy, c.wheel_z], [-wx, wy, c.wheel_z], [-wx, -wy, c.wheel_z]])
        self.cp_body = np.vstack([wheels, corners])
        self.cp_rad = np.concatenate([np.full(4, c.wheel_radius), np.zeros(8)])
        self.cp_k = np.concatenate([np.full(4, c.k_wheel), np.full(8, c.k_chassis)])
        self.cp_c = np.concatenate([np.full(4, c.c_wheel), np.full(8, c.c_chassis)])
        self.valve_alpha = 1 - math.exp(-c.dt / c.valve_tau)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self, p=(0, 0, 5.0), v=(0, 0, 0), q=(1, 0, 0, 0), w=(0, 0, 0), gas=None, f=None):
        self.p = np.array(p, float); self.v = np.array(v, float)
        self.q = qnorm(np.array(q, float)); self.w = np.array(w, float)
        self.gas = self.cfg.gas0 if gas is None else float(gas)
        self.f = np.zeros(self.nn) if f is None else np.array(f, float)
        self.t = 0.0
        self.last = dict(wheel_n=0, chassis_n=0, max_pen_chassis=0.0, fn_total=0.0)
        self.gas_used = 0.0
        self.impulse = 0.0

    @property
    def mass(self):
        return self.m0 - (self.cfg.gas0 - self.gas) if self.cfg.gas_depletion else self.m0

    @property
    def inertia(self):
        return self.I0 * (self.mass / self.m0)

    def wrench_body(self, f=None):
        f = self.f if f is None else f
        return f @ self.nz_fdir, f @ self.nz_tmat

    # ---------------------------------------------------------------- contact
    def contact(self, R):
        c = self.cfg
        rw = self.cp_body @ R.T
        pen = self.cp_rad - (self.p[2] + rw[:, 2])
        act = pen > c.pen_tol
        F = np.zeros(3); T = np.zeros(3)
        self.last = dict(wheel_n=0, chassis_n=0, max_pen_chassis=0.0, fn_total=0.0)
        if not act.any():
            return F, T
        ww = R @ self.w
        for i in np.nonzero(act)[0]:
            rc = rw[i].copy(); rc[2] -= self.cp_rad[i]            # contact point offset from CoM
            vp = self.v + np.cross(ww, rc)
            fn = self.cp_k[i]*pen[i] - self.cp_c[i]*vp[2]
            if fn <= 0.0:
                continue
            vx, vy = vp[0], vp[1]
            if i < 4:
                hx, hy = R[0, 0], R[1, 0]; n = math.hypot(hx, hy)
                fx, fy = (hx/n, hy/n) if n > 1e-3 else (1.0, 0.0)
                vl = vx*fx + vy*fy; vs = -vx*fy + vy*fx
                fl = -fn*c.mu_wheel_roll*vl/math.sqrt(vl*vl + c.v_eps**2)
                fs = -fn*c.mu_wheel_lat*vs/math.sqrt(vs*vs + c.v_eps**2)
                ft_x = fl*fx - fs*fy; ft_y = fl*fy + fs*fx
                self.last['wheel_n'] += 1
            else:
                vt = math.sqrt(vx*vx + vy*vy + c.v_eps**2)
                ft_x = -fn*c.mu_chassis*vx/vt; ft_y = -fn*c.mu_chassis*vy/vt
                self.last['chassis_n'] += 1
                self.last['max_pen_chassis'] = max(self.last['max_pen_chassis'], pen[i])
            Fi = np.array([ft_x, ft_y, fn])
            F += Fi; T += np.cross(rc, Fi)
            self.last['fn_total'] += fn
        return F, T   # forces and torques in the WORLD frame

    # ------------------------------------------------------------------- step
    def substep(self, u):
        c = self.cfg; dt = c.dt
        target = np.clip(u, 0.0, 1.0) * self.fmax
        if self.gas <= 0.0:
            target = target*0.0
        self.f += (target - self.f) * self.valve_alpha
        if self.gas <= 0.0:
            self.f *= 0.0
        m = self.mass; I = self.inertia
        Fb, Tb = self.wrench_body()
        R = q_to_R(self.q)
        Fc, Tc_w = self.contact(R)
        tau = Tb + R.T @ Tc_w
        wd = (tau - np.cross(self.w, I*self.w)) / I
        w_mid = self.w + 0.5*dt*wd
        q_half = qmul(self.q, q_expmap(w_mid*dt*0.5))
        R_mid = q_to_R(q_half)
        a = (R_mid @ Fb + Fc) / m + np.array([0, 0, -G0])
        self.p = self.p + self.v*dt + 0.5*a*dt*dt
        self.v = self.v + a*dt
        wd_mid = (tau - np.cross(w_mid, I*w_mid)) / I
        self.w = self.w + dt*wd_mid
        self.q = qnorm(qmul(self.q, q_expmap(w_mid*dt)))
        fs = float(self.f.sum())
        used = fs/(c.isp*G0)*dt            # gas is always counted; mass/inertia change only if gas_depletion
        self.gas = max(0.0, self.gas - used); self.gas_used += used
        self.impulse += fs*dt
        self.t += dt

    def step(self, u, n_sub=4):
        for _ in range(n_sub):
            self.substep(u)

    # ---------------------------------------------------------------- queries
    def energy(self):
        I = self.inertia; m = self.mass
        return 0.5*m*float(self.v @ self.v) + 0.5*float(np.sum(I*self.w*self.w)) + m*G0*self.p[2]

    def ang_momentum_world(self):
        I = self.inertia
        return q_to_R(self.q) @ (I*self.w)

    def elastic_energy(self):
        R = q_to_R(self.q)
        pen = self.cp_rad - (self.p[2] + (self.cp_body @ R.T)[:, 2])
        a = pen > self.cfg.pen_tol
        return float(np.sum(0.5*self.cp_k[a]*pen[a]**2))

    def lowest_point_z(self):
        R = q_to_R(self.q)
        return float(np.min(self.p[2] + (self.cp_body @ R.T)[:, 2] - self.cp_rad))
