"""True 3D rendering of the SIMULATED state (position + quaternion from the physics history). Nothing is animated separately."""
import math, os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from sim3d import *
import env3d

L, W, H = 4.3, 2.0, 1.3

def box_faces(center_off, half, R, p):
    sx, sy, sz = half
    v = np.array([[x*sx, y*sy, z*sz] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) + center_off
    vw = p + v @ R.T
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [vw[i] for i in idx]

def wheel_poly(center_body, R, p, radius=0.33, width=0.28, n=14):
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    ring = np.stack([radius*np.cos(th), np.zeros(n), radius*np.sin(th)], 1)
    a = ring + np.array([0, -width/2, 0]); b = ring + np.array([0, width/2, 0])
    polys = []
    for i in range(n):
        j = (i+1) % n
        polys.append(np.array([a[i], a[j], b[j], b[i]]))
    polys += [a, b]
    return [p + (center_body + q) @ R.T for q in polys]


# ------------------------------------------------------------------ sports-car visual model (VISUAL ONLY: physics still uses the box + wheel contact points)
LIGHT = np.array([0.35, -0.45, 0.82]); LIGHT /= np.linalg.norm(LIGHT)
BODY = [  # x, half-width, z_bottom, z_top
    (2.15, 0.62, -0.40, -0.14), (1.95, 0.86, -0.47, -0.04), (1.40, 0.98, -0.50, 0.08), (0.70, 1.00, -0.50, 0.16),
    (-0.50, 1.00, -0.50, 0.20), (-1.50, 1.00, -0.50, 0.24), (-2.00, 0.95, -0.46, 0.27), (-2.15, 0.84, -0.42, 0.20)]
CABIN = [(0.75, 0.80, 0.16, 0.18), (0.15, 0.80, 0.18, 0.52), (-0.65, 0.78, 0.20, 0.61), (-1.25, 0.76, 0.22, 0.52), (-1.65, 0.72, 0.24, 0.27)]

def _loft(sections, colors):
    polys, cols = [], []
    for i in range(len(sections)-1):
        x0, w0, b0, t0 = sections[i]; x1, w1, b1, t1 = sections[i+1]
        top, bot_, side_c = colors[i]
        polys.append(([x0, -w0, t0], [x0, w0, t0], [x1, w1, t1], [x1, -w1, t1])); cols.append(top)
        polys.append(([x0, -w0, b0], [x0, w0, b0], [x1, w1, b1], [x1, -w1, b1])); cols.append(bot_)
        polys.append(([x0, w0, b0], [x0, w0, t0], [x1, w1, t1], [x1, w1, b1])); cols.append(side_c)
        polys.append(([x0, -w0, b0], [x0, -w0, t0], [x1, -w1, t1], [x1, -w1, b1])); cols.append(side_c)
    x, w, b, t = sections[0]; polys.append(([x, -w, b], [x, w, b], [x, w, t], [x, -w, t])); cols.append(colors[0][2])
    x, w, b, t = sections[-1]; polys.append(([x, -w, b], [x, w, b], [x, w, t], [x, -w, t])); cols.append(colors[-1][2])
    return polys, cols

def _hex(c): return np.array([int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)])/255.0

def draw_car(ax, R, p, cfg, body="#e8ecf2"):
    BODYC, UNDER, GLASS, ROOF, TIRE, RIM, RED, HEAD = _hex(body), _hex("#23262c"), _hex("#2b3a52"), _hex(body), _hex("#15171b"), _hex("#b8bec9"), _hex("#e5262b"), _hex("#fff6c9")
    polys, cols = [], []
    pb, cb = _loft(BODY, [(BODYC, UNDER, BODYC*0.93)]*(len(BODY)-1)); polys += pb; cols += cb
    glass = [(GLASS, UNDER, GLASS), (ROOF, UNDER, GLASS*1.1), (GLASS, UNDER, GLASS), (GLASS, UNDER, GLASS*0.9)]
    pc, cc = _loft(CABIN, glass); polys += pc; cols += cc
    def box(c, h, col):
        for q in box_faces(np.array(c, float), h, np.eye(3), np.zeros(3)): polys.append(tuple(q)); cols.append(col)
    box((-2.17, 0.0, 0.10), (0.02, 0.70, 0.045), RED)                        # tail light bar
    box((-2.17, 0.0, -0.03), (0.02, 0.30, 0.02), RED*0.7)
    box((2.0, 0.62, -0.08), (0.12, 0.2, 0.035), HEAD); box((2.0, -0.62, -0.08), (0.12, 0.2, 0.035), HEAD)   # headlights
    box((-1.95, 0.0, 0.43), (0.20, 0.82, 0.02), UNDER*1.3)                   # rear wing
    box((-1.90, 0.55, 0.35), (0.05, 0.03, 0.09), UNDER*1.3); box((-1.90, -0.55, 0.35), (0.05, 0.03, 0.09), UNDER*1.3)
    wx, wy = cfg.wheel_xy; rad = cfg.wheel_radius; wid = 0.28; nseg = 16
    th = np.linspace(0, 2*np.pi, nseg, endpoint=False)
    for sx in (1, -1):
        for sy in (1, -1):
            c = np.array([sx*wx, sy*wy, cfg.wheel_z])
            ring = lambda r, yo: np.stack([r*np.cos(th), np.full(nseg, yo), r*np.sin(th)], 1) + c
            a, b = ring(rad, -wid/2), ring(rad, wid/2)
            for i in range(nseg):
                j = (i+1) % nseg; polys.append((a[i], a[j], b[j], b[i])); cols.append(TIRE)
            polys.append(tuple(a)); cols.append(TIRE*1.2); polys.append(tuple(b)); cols.append(TIRE*1.2)
            yo = sy*(wid/2 + 0.004)                                           # silver rim on the outer face + dark hub
            polys.append(tuple(ring(rad*0.66, yo))); cols.append(RIM)
            polys.append(tuple(ring(rad*0.22, yo + sy*0.004))); cols.append(UNDER)
            for k in range(5):                                               # spokes
                ang = 2*np.pi*k/5; d = np.array([np.cos(ang), 0, np.sin(ang)])
                pp = [c + d*rad*0.2 + [0, yo + sy*0.006, 0] + np.cross(d, [0, 1, 0])*0.025, c + d*rad*0.64 + [0, yo + sy*0.006, 0] + np.cross(d, [0, 1, 0])*0.025,
                      c + d*rad*0.64 + [0, yo + sy*0.006, 0] - np.cross(d, [0, 1, 0])*0.025, c + d*rad*0.2 + [0, yo + sy*0.006, 0] - np.cross(d, [0, 1, 0])*0.025]
                polys.append(tuple(pp)); cols.append(UNDER*1.4)
    W = [p + np.array(q) @ R.T for q in polys]
    fc = []
    for q, col in zip(W, cols):
        n = np.cross(q[1] - q[0], q[2] - q[0]); nn = np.linalg.norm(n)
        sh = 0.55 + 0.45*abs(float(n @ LIGHT))/nn if nn > 1e-9 else 1.0
        fc.append(np.clip(col*sh, 0, 1))
    ax.add_collection3d(Poly3DCollection(W, facecolors=fc, edgecolors=(0.05, 0.06, 0.08, 0.55), linewidths=0.25))

def draw_frame(ax, h, cfg, nozzle_exh, nozzle_pos, fmax, obstacles, view, xlim, zlim, trail=None, title="", zoom=1.25, extra=None, show_ground=True):
    ax.cla()
    R = q_to_R(h['q']); p = h['p']
    draw_car(ax, R, p, cfg)
    # thruster plumes: along the EXHAUST direction, length proportional to thrust
    segs = []
    for i in range(len(nozzle_pos)):
        thr = h['f'][i]/fmax[i]
        if thr > 0.03:
            a = p + R @ nozzle_pos[i]; b = a + R @ nozzle_exh[i]*(0.4 + 2.4*thr)
            segs.append([a, b])
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors="#38bdf8", linewidths=3.0))
    # ground and obstacles
    gx = np.array([xlim[0], xlim[1]]); gy = np.array([-5, 5])
    if show_ground:
        ax.add_collection3d(Poly3DCollection([[[gx[0], gy[0], 0], [gx[1], gy[0], 0], [gx[1], gy[1], 0], [gx[0], gy[1], 0]]], facecolor="#2b2f36", edgecolor="#444", alpha=0.55))
    if extra is not None: extra(ax)
    for (x0, x1, hh) in obstacles:
        c = np.array([(x0+x1)/2, 0, hh/2]); half = ((x1-x0)/2, env3d.OBST_HALF_WIDTH, hh/2)
        ax.add_collection3d(Poly3DCollection(box_faces(np.zeros(3), half, np.eye(3), c), facecolor="#b0b7c3" if hh < 2 else "#e9ecef", edgecolor="#667", alpha=0.35, linewidths=0.4))
    if trail is not None and len(trail) > 1:
        tr = np.array(trail); ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="#f59e0b", lw=1.2)
    # world axes (at the corner) and body axes (at the car)
    o = np.array([xlim[0]+0.8, -4.0, 0.0])
    for d, c, n in [((1, 0, 0), "r", "X"), ((0, 1, 0), "g", "Y"), ((0, 0, 1), "b", "Z")]:
        ax.plot([o[0], o[0]+2*d[0]], [o[1], o[1]+2*d[1]], [o[2], o[2]+2*d[2]], color=c, lw=2); ax.text(o[0]+2.2*d[0], o[1]+2.2*d[1], o[2]+2.2*d[2], n, color=c, fontsize=7)
    for k, (c, n) in enumerate([("r", "x"), ("g", "y"), ("b", "z")]):
        e = np.zeros(3); e[k] = 1.0; b = p + R @ e*2.6
        ax.plot([p[0], b[0]], [p[1], b[1]], [p[2], b[2]], color=c, lw=1.4, ls="--"); ax.text(b[0], b[1], b[2], n, color=c, fontsize=7)
    ax.set_xlim(*xlim); ax.set_ylim(-5, 5); ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1]-xlim[0], 10, zlim[1]-zlim[0]), zoom=zoom)
    ax.view_init(elev=view[0], azim=view[1]); ax.set_title(title, fontsize=8, pad=-2)
    ax.set_facecolor("#0e1116"); ax.patch.set_facecolor("#0e1116"); ax.set_axis_off()

VIEWS = [("three-quarter", (22, -55)), ("side (looking along +Y)", (3, -90)), ("front", (3, 0)), ("rear", (3, 180))]

def render_video(hist, cfg, out_path, obstacles=(), title="", fps=25, follow=True, max_frames=None, skip=1, label_lines=None):
    nozzle_pos = np.array([n[2] for n in cfg.nozzles], float); nozzle_exh = np.array([n[3] for n in cfg.nozzles], float)
    nozzle_exh /= np.linalg.norm(nozzle_exh, axis=1, keepdims=True); fmax = np.array([n[4] for n in cfg.nozzles], float)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=72)
    axs = [fig.add_subplot(2, 2, i+1, projection="3d") for i in range(4)]
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.0, wspace=0.0, hspace=0.0)
    frames = list(range(0, len(hist), skip))[:max_frames]
    tmp = out_path + "_frames"; os.makedirs(tmp, exist_ok=True)
    trail = []
    zmax = max(h['p'][2] for h in hist) + 3
    for n, i in enumerate(frames):
        h = hist[i]; trail.append(h['p'].copy())
        cx = h['p'][0]
        xlim = (cx-7, cx+7) if follow else (-5, 40)
        zlim = (0, max(zmax, 8))
        tilt = math.degrees(tilt_angle(h['q'])); roll = math.degrees(h.get('phi', 0))
        for ax, (nm, view) in zip(axs, VIEWS):
            draw_frame(ax, h, cfg, nozzle_exh, nozzle_pos, fmax, obstacles, view, xlim, zlim, trail=trail, title=nm)
        fig.suptitle(f"{title}\nt={h['t']:.2f}s  alt={h['p'][2]:.1f} m  tilt={tilt:.0f} deg  body-X roll progress={roll:.0f} deg  gas={h['gas']:.0f} kg  {label_lines or ''}", fontsize=9, color="w")
        fig.patch.set_facecolor("#0e1116")
        fig.savefig(os.path.join(tmp, f"f{n:04d}.png"), facecolor=fig.get_facecolor())
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "f%04d.png"), "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path], check=True)
    for f in os.listdir(tmp): os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return out_path
