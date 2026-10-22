import numpy as np, math, json, sys
from scripted import *
def phase_table(env, phases):
    H = env.hist; rows = []
    names = [n[0] for n in env.cfg.nozzles]
    grp = {g: [i for i, n in enumerate(env.cfg.nozzles) if n[1] == g] for g in ("rear", "corner", "yaw")}
    start = 0
    seq = []
    for i, ph in enumerate(phases):
        if i == 0 or ph != phases[i-1]: seq.append([ph, i, i])
        else: seq[-1][2] = i
    prev_v = np.zeros(3)
    out = []
    for ph, i0, i1 in seq:
        seg = H[i0:i1+1]
        t0 = seg[0]['t'] - 0.02; t1 = seg[-1]['t']
        z = np.array([h['p'][2] for h in seg]); v = np.array([h['v'] for h in seg]); f = np.array([h['f'] for h in seg])
        acc = (v[-1] - (H[i0-1]['v'] if i0 > 0 else v[0]))/ (t1 - t0 if t1 > t0 else 1)
        gas_u = (H[i0-1]['gas'] if i0 > 0 else env.cfg.gas0) - seg[-1]['gas']
        out.append(dict(phase=ph, t0=round(t0, 2), t1=round(t1, 2), z_start=round(float(H[i0-1]['p'][2]) if i0 > 0 else float(z[0]), 2), z_end=round(float(z[-1]), 2),
                        z_max=round(float(z.max()), 2), vz_end=round(float(v[-1][2]), 2), vx_end=round(float(v[-1][0]), 2), vy_end=round(float(v[-1][1]), 2),
                        mean_corner_kN=round(float(f[:, grp['corner']].sum(1).mean())/1e3, 1), peak_corner_kN=round(float(f[:, grp['corner']].sum(1).max())/1e3, 1),
                        mean_yaw_kN=round(float(f[:, grp['yaw']].sum(1).mean())/1e3, 2), mean_rear_kN=round(float(f[:, grp['rear']].sum(1).mean())/1e3, 2),
                        mean_acc_xyz=[round(float(a), 1) for a in acc], roll_deg_end=round(math.degrees(seg[-1]['phi']), 1), gas_kg=round(float(gas_u), 1)))
    return out
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env, info, phases = run_scripted("kickflip", seed=seed)
    for r in phase_table(env, phases): print(r)
    print({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in info.items() if k not in ('checks', 'touch_rates')})
    print(info['checks'])
