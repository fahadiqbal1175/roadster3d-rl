import numpy as np, json
from ramp import *
from scripted import *
out = []
for a_land in (4.0, 5.0, 6.0, 7.0):
    for margin in (1.1, 1.3):
        res = []
        for sd in range(40000, 40020):
            env = RampExitEnv(seed=sd); ctl = ScriptedController(env, a_land=a_land, margin=margin)
            o, _ = env.reset(seed=sd); ctl.reset()
            while True:
                o, r, te, tr, info = env.step(ctl.act())
                if te or tr: res.append(info); break
        row = dict(a_land=a_land, margin=margin, success=int(sum(r['success'] for r in res)), chassis=int(sum(r['chassis_contact'] for r in res)),
                   mean_gas=float(np.mean([r['gas_left'] for r in res])))
        out.append(row); print(row, flush=True)
json.dump(out, open("tune_ramp.json", "w"), indent=1)
