"""Plots, footage comparison and videos from saved checkpoints. Usage: python make_outputs.py plots|compare|video <name>"""
import sys, os, json, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env3d import *
from ramp import RampExitEnv
from scripted import *
from ppo_np import *
from render3d import *
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)

def env_for(task, seed, record=True):
    return RampExitEnv(seed=seed, record=record) if task == "ramp_exit" else RoadsterEnv(task, seed=seed, record=record)

def rollout_policy(task, ckpt, seed, deterministic=True):
    env = env_for(task, seed); ag = Agent(env.obs_dim, 10)
    if ckpt: ag.load(ckpt)
    rng = np.random.default_rng(seed); o, _ = env.reset(seed=seed)
    while True:
        a, _, _ = ag.act(o[None, :], rng, deterministic); o, r, te, tr, info = env.step(a[0])
        if te or tr: return env, info

def rollout_scripted(task, seed):
    env, info, _ = None, None, None
    env = env_for(task, seed); ctl = ScriptedController(env); o, _ = env.reset(seed=seed); ctl.reset()
    while True:
        o, r, te, tr, info = env.step(ctl.act())
        if te or tr: return env, info

def plot_compare(envs, labels, path, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)); cols = ["#2563eb", "#dc2626", "#16a34a"]
    grp = {g: [i for i, n in enumerate(envs[0].cfg.nozzles) if n[1] == g] for g in ("rear", "corner", "yaw")}
    for k, (env, lab) in enumerate(zip(envs, labels)):
        H = env.hist; t = np.array([h['t'] for h in H]); c = cols[k]
        w = np.array([h['w'] for h in H]); p = np.array([h['p'] for h in H]); f = np.array([h['f'] for h in H])
        for j, nm in enumerate("xyz"):
            axs[0, 0].plot(t, w[:, j], color=c, ls=["-", "--", ":"][j], lw=1.3, label=f"{lab} w_{nm}" if k == 0 or True else None)
        axs[0, 1].plot(t, p[:, 2], color=c, label=lab); axs[0, 1].plot(t, p[:, 0]/8.0, color=c, ls=":", lw=1)
        axs[0, 2].plot(t, [math.degrees(h['e_att']) for h in H], color=c, label=lab)
        axs[1, 0].plot(t, [math.degrees(h['phi']) for h in H], color=c, label=f"{lab} progress")
        axs[1, 0].plot(t, [math.degrees(h['th_ref']) for h in H], color="k", ls=":", lw=1)
        axs[1, 1].plot(t, f[:, grp['corner']].sum(1)/1e3, color=c, label=f"{lab} corner")
        axs[1, 1].plot(t, f[:, grp['rear']].sum(1)/1e3, color=c, ls="--", lw=1); axs[1, 1].plot(t, f[:, grp['yaw']].sum(1)/1e3, color=c, ls=":", lw=1)
        axs[1, 2].plot(t, [h['gas'] for h in H], color=c, label=lab)
    axs[0, 0].set_title("body angular velocity (rad/s)  x solid, y dashed, z dotted"); axs[0, 1].set_title("altitude of CoM (m); dotted: x/8 m")
    axs[0, 2].set_title("attitude tracking error vs reference (deg)"); axs[1, 0].set_title("signed body-X roll progress (deg); dotted = reference")
    axs[1, 1].set_title("commanded nozzle force sums (kN): corner solid, rear dashed, yaw dotted"); axs[1, 2].set_title("propellant remaining (kg)")
    for a in axs.flat: a.grid(alpha=0.3); a.set_xlabel("time (s)")
    axs[0, 1].legend(fontsize=8); axs[1, 0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

def footage_comparison(env, path):
    from PIL import Image
    img = Image.open("k_zoom.jpg"); Wt, Ht = 640, img.size[1]//4
    idxs = [2, 4, 6, 8, 10]; times = [13.25 + i/6 for i in idxs]; targets = [0, 90, 180, 270, 330]
    H = env.hist; cfg = env.cfg
    nozzle_pos = np.array([n[2] for n in cfg.nozzles], float); nozzle_exh = np.array([n[3] for n in cfg.nozzles], float); fmax = np.array([n[4] for n in cfg.nozzles], float)
    fig = plt.figure(figsize=(18, 7.4)); 
    for k, (i, tt, tg) in enumerate(zip(idxs, times, targets)):
        r, c = divmod(i, 3); tile = img.crop((c*Wt, r*Ht, (c+1)*Wt, (r+1)*Ht))
        ax = fig.add_subplot(2, 5, k+1); ax.imshow(tile); ax.axis("off"); ax.set_title(f"footage t={tt:.2f}s\n(my visual read: roll about {tg} deg)", fontsize=9)
        j = int(np.argmin([abs(math.degrees(h['phi']) - tg) for h in H]))
        ax3 = fig.add_subplot(2, 5, 5+k+1, projection="3d")
        draw_frame(ax3, H[j], cfg, nozzle_exh, nozzle_pos, fmax, [], (4, 90), (H[j]['p'][0]-6, H[j]['p'][0]+6), (max(0, H[j]['p'][2]-5), H[j]['p'][2]+5), title=f"simulation: roll {math.degrees(H[j]['phi']):.0f} deg, t={H[j]['t']:.2f}s")
    fig.suptitle("Qualitative comparison: Tesla Kickflip footage (top, my read of the roll stage, approximate) vs simulated policy (bottom, same roll stage, camera mirrored to match travel direction)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "plots":
        for task, ck in [("kickflip", "kick_ppo_best.npz"), ("ramp_exit", "ramp_exit_ppo_best.npz")]:
            if not os.path.exists(ck): ck = ck.replace("_ppo_best", "_bc")
            if not os.path.exists(ck): continue
            es, _ = rollout_scripted(task, 10003), None; ep, ip = rollout_policy(task, ck, 10003)
            plot_compare([es[0], ep], ["scripted", "learned"], f"{OUT}/compare_{task}.png", f"{task}: scripted controller vs learned policy, same seed 10003")
    elif mode == "compare":
        if not os.path.exists("k_zoom.jpg"): print("k_zoom.jpg (footage crops) not found, skipping the footage comparison"); sys.exit(0)
        ep, ip = rollout_policy("kickflip", "kick_ppo_best.npz" if os.path.exists("kick_ppo_best.npz") else "kick_bc.npz", 10003)
        footage_comparison(ep, f"{OUT}/footage_vs_sim.png")
    elif mode == "video":
        name = sys.argv[2]
        cfgmap = dict(baseline=("kickflip", None), landing=("landing", "landing_ppo_best.npz"), recover180=("recover180", "recover180_ppo_best.npz"),
                      kickflip=("kickflip", "kick_ppo_best.npz"), ramp=("ramp_exit", "ramp_exit_ppo_best.npz"))
        task, ck = cfgmap[name]
        if ck and not os.path.exists(ck): ck = ck.replace("_ppo_best", "_bc")
        seed = 10003
        if ck is None:
            env = env_for(task, seed); ag = Agent(env.obs_dim, 10); rng = np.random.default_rng(seed); o, _ = env.reset(seed=seed)   # untrained network
            while True:
                a, _, _ = ag.act(o[None, :], rng, True); o, r, te, tr, info = env.step(a[0])
                if te or tr: break
            title = "UNTRAINED policy (random-initialised network, half-throttle bias), Kickflip task"
        else:
            env, info = rollout_policy(task, ck, seed); title = f"Learned policy ({os.path.basename(ck)}), task {task}"
        obs_ = OBSTACLES if task == "kickflip" else []
        res = f"result: {info['reason']}, success={info['success']}"
        render_video(env.hist, env.cfg, f"{OUT}/{name}.mp4", obstacles=obs_, title=title, skip=2, fps=25, label_lines=res)
        json.dump({k: v for k, v in info.items() if k != 'checks'} | {"checks": info['checks']}, open(f"{OUT}/{name}_episode.json", "w"), indent=1, default=float)
