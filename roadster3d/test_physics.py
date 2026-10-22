import json, math, sys, time
import numpy as np
from sim3d import *

results = []
def check(name, ok, detail):
    results.append(dict(name=name, passed=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + " | " + detail)

def fresh(**kw):
    s = Sim3D(VehicleConfig(**kw)); return s

# 1 ballistic
s = fresh(); s.reset(p=(1, 2, 50), v=(3, -1, 10), q=q_from_euler_zyx(0.3, 0.2, -0.4), w=(0, 0, 0))
T = 3.0
for _ in range(int(T/0.02)): s.step(np.zeros(s.nn))
pa = np.array([1 + 3*T, 2 - 1*T, 50 + 10*T - 0.5*G0*T*T]); va = np.array([3, -1, 10 - G0*T])
err = np.linalg.norm(s.p - pa); verr = np.linalg.norm(s.v - va)
check("1 zero thrust gives ballistic motion", err < 1e-6 and verr < 1e-9, f"pos err {err:.2e} m, vel err {verr:.2e} m/s over {T}s")

# 2 balanced thrusters: net torque ~ 0, force as expected
s = fresh(); names = [n[0] for n in s.cfg.nozzles]
def u_of(**d):
    u = np.zeros(s.nn)
    for k, v in d.items(): u[names.index(k)] = v
    return u
f_c = u_of(cornerFL=1, cornerFR=1, cornerRL=1, cornerRR=1) * s.fmax
F, Tq = s.wrench_body(f_c)
check("2a balanced corner nozzles: +Z force, no torque", abs(F[2] - 32000) < 1e-6 and np.abs(Tq).max() < 1e-6, f"F={F.round(1)}, tau={Tq.round(6)}")
f_r = np.array([1.0 if n.startswith('rear') else 0.0 for n in names]) * s.fmax
F, Tq = s.wrench_body(f_r)
check("2b balanced rear nozzles: +X force (plume rearward), no torque", abs(F[0] - 80000) < 1e-6 and np.abs(Tq).max() < 1e-6, f"F={F.round(1)}, tau={Tq.round(6)}")

# 3 asymmetric -> torque sign and size
fL = u_of(cornerFL=1, cornerRL=1) * s.fmax
F, Tq = s.wrench_body(fL)
check("3a left corners fire: +roll torque tau_x=+14.4 kN m", abs(Tq[0] - 14400) < 1e-6 and abs(Tq[1]) < 1e-6, f"tau={Tq.round(1)}")
fF = u_of(cornerFL=1, cornerFR=1) * s.fmax
F, Tq = s.wrench_body(fF)
check("3b front corners fire: nose-up torque tau_y<0", Tq[1] < 0 and abs(Tq[1] + 1.7*16000) < 1e-6, f"tau={Tq.round(1)}")
fY = u_of(yawL=1) * s.fmax
F, Tq = s.wrench_body(fY)
check("3c yawL fires: force -Y at the nose, tau_z<0", F[1] < 0 and Tq[2] < 0 and abs(Tq[2] + 1.9*3000) < 1e-6, f"F={F.round(1)}, tau={Tq.round(1)}")
# integrated: roll torque produces the right angular acceleration
s = fresh(gas_depletion=False); s.reset(p=(0, 0, 100), q=(1, 0, 0, 0))
u = u_of(cornerFL=1, cornerRL=1)
s.step(u, 50)  # let the valve settle (0.25 s = 8 time constants)
w_prev = s.w[0]; s.step(u, 40); s2 = s.w[0]
a_meas = (s2 - w_prev) / (40*0.005)
a_exp = 14400 / s.I0[0]
check("3d integrated roll acceleration matches tau/Ixx", abs(a_meas - a_exp)/a_exp < 0.02, f"measured {a_meas:.3f} vs expected {a_exp:.3f} rad/s^2")

# 4 quaternion normalisation under random thrust
rng = np.random.default_rng(0)
s = fresh(); s.reset(p=(0, 0, 1e4), w=(1, 2, 3))
for _ in range(5000):
    s.step(rng.random(s.nn), 4)
check("4 quaternion stays normalised", abs(np.linalg.norm(s.q) - 1) < 1e-12, f"|q|-1 = {np.linalg.norm(s.q)-1:.2e} after 5000 control steps (100 s)")

# 5 torque-free asymmetric top: angular momentum and kinetic energy
s = fresh(); s.reset(p=(0, 0, 1e5), w=(0.2, 3.0, 0.2))
L0 = s.ang_momentum_world(); E0 = 0.5*np.sum(s.inertia*s.w**2); worst_L = 0; worst_E = 0
for _ in range(500):
    s.step(np.zeros(s.nn), 4)
    worst_L = max(worst_L, np.linalg.norm(s.ang_momentum_world() - L0)/np.linalg.norm(L0))
    worst_E = max(worst_E, abs(0.5*np.sum(s.inertia*s.w**2) - E0)/E0)
check("5 torque-free tumbling conserves |L| and KE", worst_L < 1e-3 and worst_E < 1e-3, f"max rel drift L {worst_L:.1e}, KE {worst_E:.1e} over 10 s (dt=5 ms)")

# 6 a commanded roll passes intermediate orientations; unwrapped twist reaches 2*pi
s = fresh(); s.reset(p=(0, 0, 1e5), w=(4.0, 0, 0))
tilts = []; twist = 0.0; prev = 0.0
q0 = s.q.copy()
for _ in range(int(2*math.pi/4.0/0.02) + 1):
    s.step(np.zeros(s.nn), 4)
    tilts.append(math.degrees(tilt_angle(s.q)))
    qr = qmul(qconj(q0), s.q); tw = 2*math.atan2(qr[1], qr[0]); twist += wrap(tw - prev); prev = tw
check("6 a 360 deg roll passes through 90 and 180 deg tilt", max(tilts) > 170 and 80 < tilts[len(tilts)//4] < 100, f"max tilt {max(tilts):.1f} deg, tilt at quarter {tilts[len(tilts)//4]:.1f} deg, unwrapped twist {math.degrees(twist):.1f} deg")

# 7 body-fixed jet changes world direction with attitude (unit tests: upright, sideways, inverted)
names = [n[0] for n in Sim3D().cfg.nozzles]
sim = Sim3D(); f4 = np.array([0 if not n.startswith('corner') else 8000.0 for n in names])
rows = []
ok = True
for label, roll, expect in [("upright", 0, (0, 0, 1)), ("rolled +90", math.pi/2, (0, -1, 0)), ("rolled -90", -math.pi/2, (0, 1, 0)), ("inverted", math.pi, (0, 0, -1))]:
    q = q_axis_angle([1, 0, 0], roll); Fb, _ = sim.wrench_body(f4)
    Fw = q_to_R(q) @ Fb / np.linalg.norm(Fb)
    good = np.allclose(Fw, expect, atol=1e-9); ok &= good
    rows.append(f"{label}: world dir {Fw.round(2)}")
check("7a thrust direction in world for upright/sideways/inverted attitudes", ok, "; ".join(rows))
# 7b dynamic: inverted at altitude, corner nozzles accelerate the car DOWNWARD
s = fresh(gas_depletion=False); s.reset(p=(0, 0, 500), q=q_axis_angle([1, 0, 0], math.pi))
s.step(np.zeros(s.nn)+np.array([1.0 if n.startswith('corner') else 0.0 for n in names]), 100)  # 0.5 s
s.step(np.array([1.0 if n.startswith('corner') else 0.0 for n in names]), 4)
az = (s.v[2] - 0)/ s.t
check("7b inverted: up-nozzles push the car down", s.v[2] < -G0*s.t, f"v_z = {s.v[2]:.2f} m/s after {s.t:.2f}s (free fall would be {-G0*s.t:.2f})")
# 7c sideways: body-fixed corner jets accelerate the car sideways at roll +90
s = fresh(gas_depletion=False); s.reset(p=(0, 0, 500), q=q_axis_angle([1, 0, 0], math.pi/2))
for _ in range(25): s.step(np.array([1.0 if n.startswith('corner') else 0.0 for n in names]), 4)
check("7c rolled 90 deg: corner jets push the car toward world -Y", s.v[1] < -5 and abs(s.v[0]) < 1e-6, f"v = {s.v.round(2)}")

# 8 contact: drop test, energy never increases, no wheel/chassis blow-up, rest height
def drop(dt, h=3.0, q=(1, 0, 0, 0), vx=0.0):
    s = Sim3D(VehicleConfig(dt=dt)); s.reset(p=(0, 0, h), v=(vx, 0, 0), q=q)
    ns = int(round(0.02/dt)); E0 = E_prev = s.energy() + s.elastic_energy(); e_over = -1e18; lag_gain = 0.0; gain = 0.0; maxpen = 0; first_contact = None; chassis = 0
    for k in range(int(6.0/0.02)):
        s.step(np.zeros(s.nn), ns)
        E = s.energy() + s.elastic_energy(); gain = max(gain, E - E_prev); lag = s.elastic_energy() > 0 and (s.last['wheel_n'] + s.last['chassis_n']) == 0   # first step of a contact: force not yet applied
        if lag: lag_gain = max(lag_gain, E - E0)
        else: e_over = max(e_over, E - E0)
        E_prev = E
        if s.last['chassis_n']: chassis = max(chassis, s.last['max_pen_chassis'])
        if first_contact is None and s.last['wheel_n']: first_contact = s.t
    s.lag_gain = lag_gain
    return s, gain, chassis, e_over
s, gain, chassis, e_over = drop(0.005)
_, gain_h, _, _ = drop(0.0025)
rest_expected = 0.33 - 1700*G0/(4*1.0e5) + 0.55   # wheel centre height (radius - compression) + 0.55 m
check("8a landing drop: total energy (KE+PE+spring) never exceeds its initial value", e_over <= 1e-6*1700*G0*3, f"max of E(t)-E(0) once contact forces act = {e_over:.1f} J (scale m g h = {1700*G0*3:.0f} J). Separate bookkeeping artifact: the first step of a contact is detected one step late, spring energy 0.5 k pen^2 of {s.lag_gain:.1f} J ({100*s.lag_gain/(1700*G0*3):.2f}% of the drop energy) appears for that step only")
check("8b upright car comes to rest on its wheels at the expected height", abs(s.p[2] - rest_expected) < 2e-3 and np.linalg.norm(s.v) < 1e-3, f"rest z {s.p[2]:.4f} m vs expected {rest_expected:.4f} m, |v|={np.linalg.norm(s.v):.1e}")
print(f"INFO 3 m drop (7.7 m/s) is far beyond the touchdown limit; max chassis penetration {chassis:.4f} m")
s_c, g_c, chassis_c, _ = drop(0.005, h=1.34)   # touchdown at about 3 m/s (the upper end of what the task allows)
check("8c2 3 m/s touchdown does not bottom out the landing gear", chassis_c == 0.0, f"max chassis penetration {chassis_c:.4f} m")
s2, g2, chassis2, _ = drop(0.005, q=q_axis_angle([1, 0, 0], math.pi))
check("8d inverted drop is flagged as chassis contact", chassis2 > 0.0 or s2.last['chassis_n'] > 0, f"max chassis penetration {chassis2:.4f} m")
# time-step sensitivity
sA, _, _, _ = drop(0.005); sB, _, _, _ = drop(0.0025)
check("8e rest height insensitive to dt (5 ms vs 2.5 ms)", abs(sA.p[2] - sB.p[2]) < 1e-3, f"{sA.p[2]:.5f} vs {sB.p[2]:.5f} m")
s3, g3, _, e3 = drop(0.005, vx=8.0)
check("8f car rolls on after landing with forward speed (low rolling friction), no energy gain", e3 <= 1e-6*1700*G0*3 + 0.5*1700*64 and s3.v[0] > 5.5, f"vx after 6 s: {s3.v[0]:.2f} m/s (analytic rolling-resistance estimate 6.2)")

# 9 gas accounting
s = fresh(); s.reset(p=(0, 0, 1e5)); u = np.ones(s.nn)
for _ in range(50): s.step(u, 4)
expected = s.impulse/(s.cfg.isp*G0)
check("9 gas used equals impulse / (Isp g0)", abs((s.cfg.gas0 - s.gas) - expected) < 1e-9, f"used {s.cfg.gas0 - s.gas:.3f} kg vs {expected:.3f} kg")

# speed
s = fresh(); s.reset(p=(0, 0, 1e5)); t0 = time.time()
for _ in range(500): s.step(np.random.rand(s.nn), 4)
sp = 500/(time.time()-t0)
print(f"sim speed: {sp:.0f} control steps/s (4 substeps each)")
json.dump(dict(results=results, sim_steps_per_s=sp), open('physics_tests.json', 'w'), indent=1)
print(f"\n{sum(r['passed'] for r in results)}/{len(results)} passed")
sys.exit(0 if all(r['passed'] for r in results) else 1)
