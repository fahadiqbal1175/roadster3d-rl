import os, json, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from env3d import *
from ramp import RampExitEnv
from scripted import *
from ppo_np import Agent
from sim3d import *
from render3d import draw_frame
from render_ramp_demo import ramp_geometry, make_extra
from render_demo import run_episode
OUT = "/home/claude/github_repo/docs/images"; os.makedirs(OUT, exist_ok=True)
cfg = VehicleConfig()
npos = np.array([n[2] for n in cfg.nozzles], float); nexh = np.array([n[3] for n in cfg.nozzles], float); fmax = np.array([n[4] for n in cfg.nozzles], float)
DARK = "#0e1116"

def stills(hist, picks, labels, path, cam, half_x, half_z, extra=None, title="", ncols=4, zoom=1.5, obstacles=()):
    n = len(picks); nr = math.ceil(n/ncols)
    fig = plt.figure(figsize=(4.4*ncols, 3.6*nr + 0.6), dpi=100); fig.patch.set_facecolor(DARK)
    for k, (i, lab) in enumerate(zip(picks, labels)):
        ax = fig.add_subplot(nr, ncols, k+1, projection="3d"); h = hist[i]; cx, cz = h['p'][0], h['p'][2]
        draw_frame(ax, h, cfg, nexh, npos, fmax, list(obstacles), cam, (cx-half_x, cx+half_x), (max(cz-half_z, -0.2) if cz-half_z > 0 else 0.0, cz+half_z), title="", zoom=zoom, extra=extra, show_ground=(cz-half_z < 1.0))
        ax.text2D(0.5, 0.97, lab, transform=ax.transAxes, color="w", ha="center", va="top", fontsize=11, weight="bold")
    fig.suptitle(title, color="w", fontsize=13, y=0.995); fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0, wspace=0, hspace=0.0)
    fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)

# ---- 1. kickflip sequence (learned policy, held-out seed)
env, info = run_episode("kickflip", "kick_ppo_best.npz", 10003); H = env.hist
def near(deg): return int(np.argmin([abs(math.degrees(h['phi']) - deg) + (0 if h['t'] > 0.8 else 1000) for h in H]))
picks = [int(np.argmin([abs(h['t'] - 0.6) for h in H])), near(5), near(90), near(180), near(270), near(355), int(np.argmin([abs(h['t'] - (env.touch_t - 0.25)) for h in H])), int(np.argmin([abs(h['t'] - (env.touch_t + 0.3)) for h in H]))]
labs = ["1  launch burn", "2  roll starts", "3  roll 90 deg", "4  roll 180 deg (inverted)", "5  roll 270 deg", "6  roll complete, upright", "7  landing burn", "8  touchdown"]
stills(H, picks, labs, f"{OUT}/kickflip_sequence.png", (3, 90), 4.2, 3.2, title="Kickflip: eight frames of one simulated episode (side camera, learned policy)", obstacles=OBSTACLES)

# ---- 2. ramp sequence
env2, info2 = run_episode("ramp_exit", "ramp_exit_ppo_best.npz", 10018); H2 = env2.hist
p0 = H2[0]['p'] - H2[0]['v']*0.02; geo = ramp_geometry(env2, p0); extra = make_extra(geo)
H2f = [dict(H2[0], t=0.0, p=p0)] + H2
tr = lambda t: int(np.argmin([abs(h['t'] - t) for h in H2f]))
picks2 = [0, tr(0.4), tr(0.8), tr(1.2), tr(1.6), tr(2.0), tr(env2.touch_t - 0.3), tr(env2.touch_t + 0.25)]
labs2 = ["1  at the ramp lip, upside down", "2  leaves the ramp, jets fire", "3  rolling", "4  rolling", "5  nearly upright", "6  upright, descending", "7  landing burn", "8  touchdown"]
stills(H2f, picks2, labs2, f"{OUT}/ramp_sequence.png", (4, -75), 6.5, 5.0, extra=extra, title="Ramp recovery: eight frames of one simulated episode (the ramp is a visual prop, not simulated)", zoom=1.5)

# ---- 3. results chart
tasks = [("kick", "Kickflip\n(+360 roll)"), ("ramp_exit", "Ramp-exit\nrecovery"), ("roll360", "Aerial\n360 roll"), ("recover180", "Recover\n180"), ("recover90", "Recover\n90"), ("landing", "Landing")]
R = {t: json.load(open(f"{t}_ppo_results.json")) for t, _ in tasks}
fig, ax = plt.subplots(figsize=(11, 4.8)); w = 0.26; x = np.arange(len(tasks))
for j, (key, lab, col) in enumerate([("test_scripted", "Scripted controller (not learned)", "#94a3b8"), ("test_bc_only", "Imitation policy (BC + DAgger)", "#2563eb"), ("test_bc_ppo", "Imitation + PPO fine-tune", "#16a34a")]):
    v = [(R[t][key]['rate']*100 if R[t].get(key) else R[t]['test_bc_only']['rate']*100) for t, _ in tasks]
    b = ax.bar(x + (j-1)*w, v, w, label=lab, color=col)
    for r_, val in zip(b, v): ax.text(r_.get_x() + r_.get_width()/2, val + 1, f"{val:.0f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in tasks]); ax.set_ylim(0, 112); ax.set_ylabel("held-out success rate (%)"); ax.grid(axis="y", alpha=0.3)
fig.text(0.5, 0.01, "For comparison, pure PPO from scratch (no demonstrations): landing 0/100 after 500k steps, kickflip 0/100 after 400k steps", ha="center", color="#dc2626", fontsize=9)
ax.set_title("Held-out success, 100 episodes per bar (seeds 10000 to 10099)"); ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.17), frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/results_chart.png", dpi=130); plt.close(fig)

# ---- 4. sensitivity chart
S = json.load(open("sensitivity.json")); names = [r['condition'] for r in S]
fig, ax = plt.subplots(figsize=(10, 5.2)); y = np.arange(len(S))[::-1]
ax.barh(y + 0.19, [r['scripted_success']/r['n']*100 for r in S], 0.36, color="#94a3b8", label="Scripted controller")
ax.barh(y - 0.19, [r['policy_success']/r['n']*100 for r in S], 0.36, color="#2563eb", label="Learned policy (trained at nominal values only)")
for yy, r in zip(y, S):
    ax.text(r["scripted_success"]/r["n"]*100 + 1, yy + 0.19, f"{r['scripted_success']}/20", va="center", fontsize=7); ax.text(r["policy_success"]/r["n"]*100 + 1, yy - 0.19, f"{r['policy_success']}/20", va="center", fontsize=7, color="#1d4ed8")
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 112); ax.set_xlabel("success rate (%), 20 episodes per row"); ax.grid(axis="x", alpha=0.3); ax.legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
ax.set_title("Kickflip sensitivity: the learned policy is far more brittle than the controller it imitates"); fig.tight_layout(); fig.savefig(f"{OUT}/sensitivity_chart.png", dpi=130); plt.close(fig)

# ---- 5. nozzle layout (hypothetical L1)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6)); cols = dict(rear="#dc2626", corner="#2563eb", yaw="#16a34a")
for a, (ix, iy, ttl) in zip(axs, [(0, 1, "Top view (x forward, y left)"), (0, 2, "Side view (x forward, z up)")]):
    if iy == 1: a.add_patch(Rectangle((-2.15, -1.0), 4.3, 2.0, fill=False, ec="k", lw=1.5))
    else: a.add_patch(Rectangle((-2.15, -0.65), 4.3, 1.3, fill=False, ec="k", lw=1.5))
    for (nm, g, p, e, f) in cfg.nozzles:
        e = np.array(e, float); fvec = -e
        a.plot(p[ix], p[iy], "o", color=cols[g], ms=7)
        if abs(e[ix]) + abs(e[iy]) > 0:
            a.annotate("", xy=(p[ix] + 0.8*e[ix], p[iy] + 0.8*e[iy]), xytext=(p[ix], p[iy]), arrowprops=dict(arrowstyle="-|>", color=cols[g], lw=2))
            a.annotate("", xy=(p[ix] + 0.55*fvec[ix], p[iy] + 0.55*fvec[iy]), xytext=(p[ix], p[iy]), arrowprops=dict(arrowstyle="-|>", color="#111", lw=1, ls="--"))
    a.set_aspect("equal"); a.set_xlim(-3.4, 3.2); a.set_ylim(-1.9, 1.9) if iy == 1 else a.set_ylim(-1.9, 1.4); a.set_title(ttl); a.set_xlabel("x (m)"); a.set_ylabel("y (m)" if iy == 1 else "z (m)"); a.grid(alpha=0.25)
axs[0].plot([], [], color="#dc2626", lw=2, label="rear x4: 20 kN each, exhaust rearward"); axs[0].plot([], [], color="#2563eb", lw=2, label="corner x4: 8 kN each, exhaust down"); axs[0].plot([], [], color="#16a34a", lw=2, label="yaw x2: 3 kN each, exhaust sideways"); axs[0].plot([], [], color="#111", lw=1, ls="--", label="dashed: force on the car (reaction)")
axs[0].legend(fontsize=7, loc="lower left"); fig.suptitle("Hypothetical L1 nozzle layout (an assumption, not Tesla hardware). Coloured arrow = exhaust, dashed arrow = force on the car", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/nozzle_layout.png", dpi=130); plt.close(fig)

# ---- 6. pipeline diagram
fig, ax = plt.subplots(figsize=(12, 4.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2)
def box(x, y, w, h, t, col): 
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=col, ec="#334155", lw=1.2)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=9)
def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color="#334155", lw=1.4))
box(0.1, 2.3, 2.2, 1.4, "6-DOF physics\n10 nozzles, contact,\nGymnasium tasks", "#e0f2fe"); box(2.9, 2.3, 2.2, 1.4, "Scripted controller\n(attitude PD + bounded\nthrust allocation)\nNOT learned", "#fef3c7"); box(5.7, 2.3, 2.2, 1.4, "Demonstrations\n+ DAgger\n(about 65k to 90k\nsamples)", "#fef3c7")
box(8.5, 2.3, 1.7, 1.4, "Behaviour\ncloning\n(imitation)", "#dbeafe"); box(10.4, 2.3, 1.5, 1.4, "PPO\nfine-tune\n(no gain)", "#dcfce7")
box(0.1, 0.3, 2.2, 1.3, "Validation:\n20 physics checks,\n8 reward-exploit tests", "#f1f5f9"); box(2.9, 0.3, 4.9, 1.3, "Pure PPO from scratch (no demonstrations):\nlanding 0/100 (500k steps), kickflip 0/100 (400k steps)", "#fee2e2"); box(8.5, 0.3, 3.4, 1.3, "Evaluation on held-out seeds,\nsensitivity sweeps, 3D videos", "#f1f5f9")
for x0, x1 in [(2.3, 2.9), (5.1, 5.7), (7.9, 8.5), (10.2, 10.4)]: arrow(x0, 3.0, x1, 3.0)
ax.set_title("How the results were produced (the learned policies imitate the scripted controller)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/pipeline.png", dpi=130, facecolor="white"); plt.close(fig)

# ---- 7. kickflip timeline with phases (learned policy)
t = np.array([h['t'] for h in H]); z = np.array([h['p'][2] for h in H]) - REST_Z; ph = np.degrees([h['phi'] for h in H]); f = np.array([h['f'] for h in H]); wx = np.array([h['w'][0] for h in H])
tr0 = t[np.argmax(ph > 5)]; tr1 = t[np.argmax(ph > 355)]
fig, axs = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
for a in axs:
    a.axvspan(0, tr0, color="#fde68a", alpha=0.5); a.axvspan(tr0, tr1, color="#bfdbfe", alpha=0.5); a.axvspan(tr1, env.touch_t, color="#bbf7d0", alpha=0.5); a.grid(alpha=0.3)
axs[0].plot(t, z, "k"); axs[0].set_ylabel("height above ground (m)"); axs[1].plot(t, ph, "#2563eb"); axs[1].set_ylabel("body-X roll progress (deg)")
axs[2].plot(t, f[:, 4:8].sum(1)/1e3, "#2563eb", label="corner nozzles (kN)"); axs[2].plot(t, f[:, 0:4].sum(1)/1e3, "#dc2626", label="rear nozzles (kN)"); axs[2].set_ylabel("commanded thrust (kN)"); axs[2].legend(fontsize=8); axs[2].set_xlabel("time (s)")
for xx, lab in [(tr0/2, "launch burn"), ((tr0 + tr1)/2, "360 deg roll"), ((tr1 + env.touch_t)/2, "landing burn")]: axs[0].text(xx, axs[0].get_ylim()[1]*0.93, lab, ha="center", fontsize=9)
fig.suptitle("Kickflip timeline of the learned policy (seed 10003)"); fig.tight_layout(); fig.savefig(f"{OUT}/kickflip_timeline.png", dpi=130); plt.close(fig)
print("assets done:", os.listdir(OUT))
