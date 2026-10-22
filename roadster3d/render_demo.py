"""Complete demonstration video: approach, launch, roll, landing, roll-out, then slow-motion replays. Rendered from the SIMULATED state."""
import sys, os, math, subprocess, json, shutil
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env3d import *
from ramp import RampExitEnv
from ppo_np import Agent
from sim3d import *
import render3d
from render3d import draw_frame

def make_state(h0, h1, a):
    q = (1-a)*h0['q'] + a*h1['q']
    if np.dot(h0['q'], h1['q']) < 0: q = (1-a)*h0['q'] - a*h1['q']
    q = q/np.linalg.norm(q)
    L = lambda k: (1-a)*h0[k] + a*h1[k]
    return dict(t=L('t'), p=L('p'), v=L('v'), q=q, w=L('w'), f=L('f'), phi=L('phi'), gas=L('gas'))

def sample(hist, t):
    ts = np.array([h['t'] for h in hist]); i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(hist)-2))
    a = float(np.clip((t - ts[i])/(ts[i+1]-ts[i]), 0, 1)); return make_state(hist[i], hist[i+1], a)

def approach_hist(env, seconds=1.5, mu=0.03):
    cfg = env.cfg; s = Sim3D(cfg); dec = mu*G0
    v_s = env.vx0 + dec*seconds; x_s = env.x0 - (env.vx0*seconds + 0.5*dec*seconds**2)
    s.reset(p=(x_s, 0.0, REST_Z), v=(v_s, 0, 0), q=env.q_ref0.copy())
    H = []
    for k in range(int(seconds/0.02)):
        s.step(np.zeros(s.nn), 4)
        H.append(dict(t=-seconds + (k+1)*0.02, p=s.p.copy(), v=s.v.copy(), q=s.q.copy(), w=s.w.copy(), f=s.f.copy(), phi=0.0, gas=cfg.gas0))
    return H

def run_episode(task, ckpt, seed):
    env = RampExitEnv(seed=seed, record=True) if task == "ramp_exit" else RoadsterEnv(task, seed=seed, record=True)
    ag = Agent(env.obs_dim, 10); ag.load(ckpt); rng = np.random.default_rng(seed); o, _ = env.reset(seed=seed)
    while True:
        a, _, _ = ag.act(o[None, :], rng, True); o, r, te, tr, info = env.step(a[0])
        if te or tr: return env, info

def frame_fig(W=1280, Hh=720):
    fig = plt.figure(figsize=(W/100, Hh/100), dpi=100); fig.patch.set_facecolor("#0e1116"); return fig

def text_card(path, lines, sub=None):
    fig = frame_fig(); fig.text(0.5, 0.60, lines[0], color="w", ha="center", fontsize=30, weight="bold")
    for k, l in enumerate(lines[1:]): fig.text(0.5, 0.50 - 0.07*k, l, color="#cbd5e1", ha="center", fontsize=17)
    if sub: fig.text(0.5, 0.08, sub, color="#94a3b8", ha="center", fontsize=11)
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)

def build(task, ckpt, seed, out, scene_title, phases_fn, slow_window, obstacles, cams, approach=True, rollout=1.2):
    env, info = run_episode(task, ckpt, seed)
    cfg = env.cfg; H = env.hist
    pre = approach_hist(env) if approach else []
    full = pre + H
    # roll-out after touchdown: keep simulating is not needed, episode already includes 0.6 s of settling; extend by holding state with the sim
    nozzle_pos = np.array([n[2] for n in cfg.nozzles], float); nozzle_exh = np.array([n[3] for n in cfg.nozzles], float); fmax = np.array([n[4] for n in cfg.nozzles], float)
    tmp = out + "_frames"; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    n = [0]
    def emit(fig):
        fig.savefig(os.path.join(tmp, f"f{n[0]:05d}.png"), facecolor=fig.get_facecolor()); n[0] += 1; plt.close(fig)
    # title card (2.4 s)
    text_card(os.path.join(tmp, "card.png"), scene_title, "Simulation inspired by public coverage. Not Tesla's controller or hardware.")
    for _ in range(60): shutil.copy(os.path.join(tmp, "card.png"), os.path.join(tmp, f"f{n[0]:05d}.png")); n[0] += 1
    t_start, t_end = full[0]['t'], full[-1]['t']
    def scene(ts, cam, label_fn, trail_from=None, zlim_top=9.0, box_zoom=1.7):
        trail = []
        for t in ts:
            st = sample(full, t); trail.append(st['p'].copy())
            fig = frame_fig(); ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
            cx = st['p'][0]; draw_frame(ax, st, cfg, nozzle_exh, nozzle_pos, fmax, obstacles, cam['view'], (cx-cam['half_x'], cx+cam['half_x']), (0, zlim_top),
                                        trail=trail[-60:], title="", zoom=box_zoom)
            fig.text(0.5, 0.955, label_fn(st, t), color="w", ha="center", fontsize=15, weight="bold")
            fig.text(0.5, 0.915, f"{cam['name']}   |   altitude {st['p'][2]-REST_Z:4.1f} m   |   body-X roll progress {math.degrees(st['phi']):5.0f} deg   |   propellant left {st['gas']:3.0f} kg", color="#cbd5e1", ha="center", fontsize=11)
            emit(fig)
    # A: full run, real time, chase camera
    dt = 0.04
    scene(np.arange(t_start, t_end, dt), cams[0], phases_fn(env, H, "REAL TIME"))
    # B: slow motion (0.2x) of the maneuver window, two cameras
    w0, w1 = slow_window(env, H)
    for cam in cams[1:]:
        scene(np.arange(w0, w1, dt*0.2), cam, phases_fn(env, H, "SLOW MOTION 0.2x"))
    # end card
    ok = info['success']
    text_card(os.path.join(tmp, "end.png"), ["Result of this episode", f"success = {ok}   |   touchdown speed {abs(info['touch_vz']):.2f} m/s   |   peak height {info['peak_height']:.1f} m",
              "Policy: imitation-learned (behaviour cloning + DAgger) from a scripted controller.", "Pure PPO from scratch did not converge in my budget (0/100).", "All vehicle numbers and the nozzle layout are assumptions."],
              "6-DOF rigid body, 10 body-fixed nozzles, physically simulated orientation (no scripted animation).")
    for _ in range(100): shutil.copy(os.path.join(tmp, "end.png"), os.path.join(tmp, f"f{n[0]:05d}.png")); n[0] += 1
    os.remove(os.path.join(tmp, "card.png")); os.remove(os.path.join(tmp, "end.png"))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "25", "-i", os.path.join(tmp, "f%05d.png"), "-pix_fmt", "yuv420p", "-crf", "20", out], check=True)
    shutil.rmtree(tmp); print("frames", n[0], "->", out); return info

if __name__ == "__main__":
    which = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 10003
    if which == "kickflip":
        def phases(env, H, mode):
            tr0 = next(h['t'] for h in H if math.degrees(h['phi']) > 5); tr1 = next(h['t'] for h in H if math.degrees(h['phi']) > 355); tt = env.touch_t
            def f(st, t):
                if t < 0: ph = "1  APPROACH (rolling, no thrust)"
                elif t < tr0: ph = "2  LAUNCH BURN (corner nozzles)"
                elif t < tr1: ph = "3  360 DEGREE ROLL ABOUT THE LONG AXIS"
                elif t < tt - 0.02: ph = "4  UPRIGHT, LANDING BURN"
                else: ph = "5  TOUCHDOWN AND ROLL-OUT"
                return f"Kickflip (simulation)   {ph}   [{mode}]"
            return f
        def window(env, H):
            return next(h['t'] for h in H if math.degrees(h['phi']) > 5) - 0.25, next(h['t'] for h in H if math.degrees(h['phi']) > 355) + 0.25
        cams = [dict(name="chase camera", view=(14, -58), half_x=9.0), dict(name="side camera (like the footage)", view=(3, 90), half_x=6.5), dict(name="front camera", view=(4, 0), half_x=6.5)]
        build("kickflip", "kick_ppo_best.npz", seed, "outputs/demo_kickflip_full.mp4", ["Roadster-inspired Kickflip", "360 degree roll, powered flight over a car lineup, wheels-down landing"], phases, window, OBSTACLES, cams)
    else:
        def phases(env, H, mode):
            tr1 = next(h['t'] for h in H if math.degrees(h['phi']) > 175); tt = env.touch_t
            def f(st, t):
                ph = "1  RAMP-EXIT STATE: car inverted, high up" if t < 0.15 else ("2  180 DEGREE ROLL RECOVERY" if t < tr1 else ("3  UPRIGHT, LANDING BURN" if t < tt - 0.02 else "4  TOUCHDOWN"))
                return f"Ramp recovery, airborne part only (simulation)   {ph}   [{mode}]"
            return f
        def window(env, H):
            return 0.0, next(h['t'] for h in H if math.degrees(h['phi']) > 175) + 0.3
        cams = [dict(name="chase camera", view=(12, -58), half_x=9.0), dict(name="side camera", view=(3, 90), half_x=6.5), dict(name="front camera", view=(4, 0), half_x=6.5)]
        build("ramp_exit", "ramp_exit_ppo_best.npz", seed, "outputs/demo_ramp_recovery.mp4", ["Roadster-inspired Ramp Recovery", "Airborne recovery from ramp-exit conditions (ramp contact is NOT simulated)"], phases, window, [], cams, approach=False)
