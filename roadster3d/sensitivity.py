"""Sensitivity of the kickflip to mass, inertia and actuator strength (policy trained at NOMINAL values; scripted controller for reference)."""
import json, sys, numpy as np
from env3d import *
from scripted import *
from ppo_np import *
from finetune import summarize

def cfg_with(mass=1.0, ixx=1.0, thrust=1.0, corner=1.0):
    base = VehicleConfig(); I = list(base.box_inertia())
    nz = []
    for n, g, p, e, f in base.nozzles:
        nz.append((n, g, p, e, f*thrust*(corner if g == "corner" else 1.0)))
    return VehicleConfig(m_total0=base.m_total0*mass, inertia=(I[0]*mass*ixx, I[1]*mass, I[2]*mass), nozzles=nz)

conds = [("nominal", {})] + [(f"thrust x{k}", dict(thrust=k)) for k in (0.6, 0.7, 0.8, 0.9, 1.1, 1.2)] + \
        [(f"corner nozzles x{k}", dict(corner=k)) for k in (0.6, 0.75)] + \
        [(f"mass x{k}", dict(mass=k)) for k in (0.85, 1.15)] + [(f"roll inertia x{k}", dict(ixx=k)) for k in (0.8, 1.25)]
ck = "kick_ppo_best.npz"
ag = Agent(54, 10); ag.load(ck)
seeds = range(30000, 30020); out = []
for name, kw in conds:
    cfg = cfg_with(**kw)
    pol = []; scr = []
    for sd in seeds:
        env = RoadsterEnv("kickflip", cfg=cfg); rng = np.random.default_rng(sd); o, _ = env.reset(seed=sd)
        while True:
            a, _, _ = ag.act(o[None, :], rng, True); o, r, te, tr, info = env.step(a[0])
            if te or tr: pol.append(info); break
        env = RoadsterEnv("kickflip", cfg=cfg); ctl = ScriptedController(env); o, _ = env.reset(seed=sd); ctl.reset()
        while True:
            o, r, te, tr, info = env.step(ctl.act())
            if te or tr: scr.append(info); break
    sp, ss = summarize(pol), summarize(scr)
    row = dict(condition=name, policy_success=sp['successes'], policy_reasons=sp['reasons'], scripted_success=ss['successes'], scripted_reasons=ss['reasons'], n=len(seeds))
    out.append(row); print(row, flush=True)
    json.dump(out, open("sensitivity.json", "w"), indent=1)
