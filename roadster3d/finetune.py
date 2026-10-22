"""BC-initialised PPO fine-tuning with validation-based checkpoint selection. Reports BC-only and BC+PPO separately."""
import sys, os, json, time, numpy as np
from env3d import *
from ramp import RampExitEnv
from scripted import *
from ppo_np import *

def make_env_fn(task, **kw):
    if task == "ramp_exit": return lambda: RampExitEnv(**kw)
    return lambda: RoadsterEnv(task, **kw)

def summarize(infos):
    n = len(infos); succ = [i['success'] for i in infos]
    k = int(sum(succ)); z = 1.96; p = k/n
    den = 1 + z*z/n; c = (p + z*z/(2*n))/den; hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    ci = (max(0.0, c-hw), min(1.0, c+hw))
    reasons = {}
    for i in infos: reasons[i['reason']] = reasons.get(i['reason'], 0) + 1
    fails = {}
    for i in infos:
        if not i['success']:
            for kk, v in (i['checks'] or {}).items():
                if not v: fails[kk] = fails.get(kk, 0) + 1
    ok = [i for i in infos if i['success']]
    return dict(n=n, successes=k, rate=p, wilson95=ci, reasons=reasons, failed_checks=fails,
                mean_progress_deg=float(np.mean([i['max_progress_deg'] for i in infos])), mean_touch_vz=float(np.nanmean([i['touch_vz'] for i in infos])),
                mean_gas_left=float(np.mean([i['gas_left'] for i in infos])), mean_peak=float(np.mean([i['peak_height'] for i in infos])))

if __name__ == "__main__":
    task, bc_path, out_prefix, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    def log(*a): print(*a, flush=True)
    env_fn = make_env_fn(task)
    ag = Agent(env_fn().obs_dim, 10, seed=0); ag.load(bc_path)
    val_seeds = range(20000, 20020); test_seeds = range(10000, 10100)
    res = {}
    v0 = summarize(evaluate(ag, env_fn, val_seeds)); log("BC-only validation:", v0['successes'], "/", v0['n'])
    best = dict(rate=v0['rate'], path=bc_path, steps=0)
    ag.log_std[:] = -2.0
    hist = []
    def eval_fn(agent, steps_done):
        v = summarize(evaluate(agent, env_fn, val_seeds)); log(f"   validation @ {steps_done}: {v['successes']}/{v['n']}  reasons {v['reasons']}")
        if v['rate'] >= best['rate'] and steps_done > 0:
            best.update(rate=v['rate'], path=out_prefix + "_best.npz", steps=steps_done); agent.save(out_prefix + "_best.npz")
    if steps < 0 and os.path.exists(out_prefix + '_best.npz'):      # resume-eval: training already ran, only evaluate the saved best-validation checkpoint
        best.update(rate=1.0, path=out_prefix + '_best.npz', steps=-1)
    if steps > 0:
        train(ag, env_fn, steps, n_envs=8, n_steps=256, lr=1e-4, epochs=6, ent_coef=0.0, log=log, ckpt_path=out_prefix + "_last.npz",
              eval_fn=eval_fn, eval_every=25, pi_warmup=12, history=hist, lr_end_frac=0.3)
    res['train_history'] = hist
    bc_ag = Agent(env_fn().obs_dim, 10); bc_ag.load(bc_path)
    res['test_bc_only'] = summarize(evaluate(bc_ag, env_fn, test_seeds)); log("TEST BC-only:", res['test_bc_only']['successes'], "/ 100", res['test_bc_only']['reasons'])
    if best['path'] != bc_path:
        fa = Agent(env_fn().obs_dim, 10); fa.load(best['path'])
        res['test_bc_ppo'] = summarize(evaluate(fa, env_fn, test_seeds)); res['selected'] = best
        log("TEST BC+PPO (best-validation checkpoint):", res['test_bc_ppo']['successes'], "/ 100", res['test_bc_ppo']['reasons'])
    else:
        res['test_bc_ppo'] = None; res['selected'] = best; log("PPO fine-tuning did not beat the BC policy on validation; BC policy kept")
    # scripted reference on the same test seeds
    infos = []
    for sd in test_seeds:
        env = env_fn(); ctl = ScriptedController(env); o, _ = env.reset(seed=sd); ctl.reset()
        while True:
            o, r, te, tr, info = env.step(ctl.act())
            if te or tr: infos.append(info); break
    res['test_scripted'] = summarize(infos); log("TEST scripted controller:", res['test_scripted']['successes'], "/ 100")
    json.dump(res, open(out_prefix + "_results.json", "w"), indent=1, default=float)
