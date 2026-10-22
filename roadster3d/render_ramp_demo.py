"""Ramp-recovery demo video (Option A): starts at the ramp LIP in the exit state. The ramp is a visual prop only, derived from the exit state."""
import sys, os, math, shutil, subprocess, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from env3d import *
from sim3d import *
from render3d import draw_frame
from render_demo import sample, frame_fig, text_card, run_episode

def ramp_geometry(env, p0):
    cfg = env.cfg; R = q_to_R(env.q_exit)
    psi = heading(env.q_exit); u = np.array([math.cos(psi), math.sin(psi)]); v = np.array([-math.sin(psi), math.cos(psi)])
    S = p0 + R @ np.array([0, 0, cfg.wheel_z - cfg.wheel_radius])           # surface point under the car at the lip
    n = R[:, 2]; n_a = float(n[:2] @ u); n_z = float(n[2])
    Rr = S[2]/(1.0 - n_z)                                                     # radius such that the ramp base is tangent to the ground
    C = (Rr*n_a, S[2] + Rr*n_z)                                               # arc centre in plane coordinates (a along heading, z up)
    th_S = math.atan2(-n_z, -n_a); th_end = th_S - 2*math.pi if th_S > -math.pi/2 else th_S
    return dict(O=S[:2], u=u, v=v, C=C, Rr=Rr, th0=-math.pi/2, th1=th_end, S=S, n=n)

def make_extra(geo, width=5.0, N=80):
    th = np.linspace(geo['th0'], geo['th1'], N)
    a = geo['C'][0] + geo['Rr']*np.cos(th); z = geo['C'][1] + geo['Rr']*np.sin(th)
    def P(ai, zi, w): return np.array([geo['O'][0] + ai*geo['u'][0] + w*geo['v'][0], geo['O'][1] + ai*geo['u'][1] + w*geo['v'][1], zi])
    quads = [[P(a[i], z[i], -width/2), P(a[i+1], z[i+1], -width/2), P(a[i+1], z[i+1], width/2), P(a[i], z[i], width/2)] for i in range(N-1)]
    edges = [[P(a[i], z[i], -width/2), P(a[i+1], z[i+1], -width/2)] for i in range(N-1)] + [[P(a[i], z[i], width/2), P(a[i+1], z[i+1], width/2)] for i in range(N-1)]
    def extra(ax):
        ax.add_collection3d(Poly3DCollection(quads, facecolor="#9aa3ae", edgecolor="#6b7280", alpha=0.55, linewidths=0.2))
        ax.add_collection3d(Line3DCollection(edges, colors="#ff4d4d", linewidths=1.6))
    return extra

def pick_seed():
    best = None
    for sd in range(10000, 10100):
        env, info = run_episode("ramp_exit", "ramp_exit_ppo_best.npz", sd)
        if not info['success']: continue
        R = q_to_R(env.q_exit); v0 = env.hist[0]['v']; p0 = env.hist[0]['p']
        tang = R[:, 0]; nrm = R[:, 2]
        score = float(v0 @ tang) - 1.5*abs(float(v0 @ nrm)) - 2*abs(v0[1])
        if best is None or score > best[0]: best = (score, sd, float(v0 @ tang), float(v0 @ nrm))
    return best

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if seed is None:
        b = pick_seed(); print("chosen seed", b); seed = b[1]
    env, info = run_episode("ramp_exit", "ramp_exit_ppo_best.npz", seed)
    cfg = env.cfg; H = env.hist
    p0 = H[0]['p'] - H[0]['v']*0.02; geo = ramp_geometry(env, p0); extra = make_extra(geo)
    print("ramp radius %.1f m, exit height %.1f m, car heading %.0f deg" % (geo['Rr'], p0[2], math.degrees(heading(env.q_exit))))
    nozzle_pos = np.array([n[2] for n in cfg.nozzles], float); nozzle_exh = np.array([n[3] for n in cfg.nozzles], float); fmax = np.array([n[4] for n in cfg.nozzles], float)
    full = [dict(H[0], t=0.0, p=p0, v=H[0]['v'])] + H
    tr1 = next(h['t'] for h in H if math.degrees(h['phi']) > 175); tt = env.touch_t; t_end = H[-1]['t']
    out = "outputs/demo_ramp_recovery.mp4"; tmp = out + "_frames"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp); n = [0]
    def emit(fig): fig.savefig(os.path.join(tmp, f"f{n[0]:05d}.png"), facecolor=fig.get_facecolor()); n[0] += 1; plt.close(fig)
    def phase(t):
        if t < 0.05: return "AT THE RAMP LIP: car upside down (prescribed exit state)"
        if t < tr1: return "LEAVES THE RAMP: 180 DEGREE ROLL RECOVERY"
        if t < tt - 0.02: return "UPRIGHT: LANDING BURN"
        return "TOUCHDOWN"
    def scene(ts, cam, mode):
        trail = []
        for t in ts:
            st = sample(full, t); trail.append(st['p'].copy())
            fig = frame_fig(); ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
            cx = st['p'][0] if cam['follow'] else cam['cx']; cz = st['p'][2] if cam['follow'] else cam['cz']
            zlim = (cz - cam['half_z'], cz + cam['half_z']) if cam['follow'] else (0, cam['half_z']*2)
            draw_frame(ax, st, cfg, nozzle_exh, nozzle_pos, fmax, [], cam['view'], (cx - cam['half_x'], cx + cam['half_x']), zlim, trail=trail[-80:], zoom=cam['zoom'],
                       extra=extra, show_ground=(zlim[0] < 1.0))
            fig.text(0.5, 0.955, phase(t), color="w", ha="center", fontsize=16, weight="bold")
            fig.text(0.5, 0.912, f"[{mode}]  {cam['name']}  |  t = {t:4.2f} s after the lip  |  altitude {st['p'][2]-REST_Z:4.1f} m  |  roll {math.degrees(st['phi']):4.0f} deg  |  propellant {st['gas']:3.0f} kg", color="#cbd5e1", ha="center", fontsize=10)
            fig.text(0.5, 0.012, "Ramp recovery, airborne part only. The ramp is a visual prop (not simulated).", color="#94a3b8", ha="center", fontsize=9)
            emit(fig)
    text_card(os.path.join(tmp, "card.png"), ["Roadster-inspired Ramp Recovery", "Starts at the ramp lip: car upside down, then jets roll it upright and it lands on its wheels", "The ramp is a visual prop only. Ramp contact is NOT simulated."], "Simulation inspired by public coverage. Not Tesla's controller or hardware.")
    for _ in range(70): shutil.copy(os.path.join(tmp, "card.png"), os.path.join(tmp, f"f{n[0]:05d}.png")); n[0] += 1
    wide = dict(name="wide camera", view=(6, -78), follow=False, cx=float(p0[0]) - 2.0, cz=0, half_x=15.0, half_z=13.5, zoom=1.6)
    chase = dict(name="follow camera", view=(10, -60), follow=True, half_x=8.0, half_z=6.5, zoom=1.5)
    side = dict(name="side camera (like the footage)", view=(3, 90), follow=True, half_x=6.0, half_z=4.5, zoom=1.5)
    front = dict(name="front camera (sees the nose)", view=(4, 180), follow=True, half_x=6.0, half_z=4.5, zoom=1.5)
    dt = 0.04
    scene([0.0]*35, wide, "FREEZE: exit state")                      # hold on the lip
    scene(np.arange(0.0, t_end, dt), wide, "REAL TIME")
    scene(np.arange(0.0, t_end, dt), chase, "REAL TIME")
    w1 = tr1 + 0.3
    scene(np.arange(0.0, w1, dt*0.2), side, "SLOW MOTION 0.2x")
    scene(np.arange(0.0, w1, dt*0.2), front, "SLOW MOTION 0.2x")
    text_card(os.path.join(tmp, "end.png"), ["Result of this episode", f"success = {info['success']}   |   roll {info['max_progress_deg']:.0f} deg   |   touchdown speed {abs(info['touch_vz']):.2f} m/s   |   final tilt {info['final_tilt_deg']:.1f} deg",
              "Policy: imitation-learned (behaviour cloning + DAgger) from a scripted controller.", "Pure PPO from scratch did not converge in my budget.", "Exit state, ramp radius and speed (about 3 m/s) are assumptions, and the ramp itself is not simulated."], "6-DOF rigid body, 10 body-fixed nozzles, physically simulated orientation.")
    for _ in range(100): shutil.copy(os.path.join(tmp, "end.png"), os.path.join(tmp, f"f{n[0]:05d}.png")); n[0] += 1
    os.remove(os.path.join(tmp, "card.png")); os.remove(os.path.join(tmp, "end.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "25", "-i", os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    shutil.rmtree(tmp); print("frames", n[0], "->", out, "seed", seed)
    json.dump(dict(seed=seed, ramp_radius=geo['Rr'], info={k: v for k, v in info.items() if k != 'checks'}), open("outputs/demo_ramp_recovery_episode.json", "w"), indent=1, default=float)
