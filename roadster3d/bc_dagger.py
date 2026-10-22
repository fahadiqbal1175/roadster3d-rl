import sys, time, json, numpy as np
from env3d import *
from scripted import *
from ppo_np import *
from ramp import RampExitEnv

def make_env(task, seed=None, **kw):
    return RampExitEnv(seed=seed, **kw) if task == 'ramp_exit' else RoadsterEnv(task, seed=seed, **kw)

def collect(task, n_eps, seed0, agent=None, beta=1.0, noise=0.15, env_kw=None, log=print):
    """roll out; label every visited state with the scripted controller's action. Executed action = scripted (prob beta) else learner."""
    env_kw = env_kw or {}
    O = []; A = []; rng = np.random.default_rng(seed0); succ = 0
    for ep in range(n_eps):
        env = make_env(task, seed0+ep, **env_kw); ctl = ScriptedController(env)
        o, _ = env.reset(seed=seed0+ep); ctl.reset()
        while True:
            a_s = ctl.act()
            O.append(o.copy()); A.append(a_s.copy())
            if agent is not None and rng.random() > beta:
                a_l, _, _ = agent.act(o[None, :], rng, True); a = a_l[0]
            else:
                a = a_s
            a = np.clip(a + rng.normal(0, noise, a.shape), -1, 1)
            o, r, te, tr, info = env.step(a)
            if te or tr: succ += info['success']; break
    log(f"  collected {len(O)} samples from {n_eps} episodes (executed-policy success {succ}/{n_eps})")
    return np.array(O), np.array(A)

if __name__ == "__main__":
    task = sys.argv[1]; out = sys.argv[2]
    t0 = time.time()
    def log(*a): print(*a, flush=True)
    env_fn = lambda: make_env(task)
    ag = Agent(env_fn().obs_dim, 10, seed=0, log_std=-1.8)
    O, A = collect(task, 120, 1000, noise=0.12, log=log)
    O2, A2 = collect(task, 80, 2000, noise=0.30, log=log)
    O = np.vstack([O, O2]); A = np.vstack([A, A2])
    bc_pretrain(ag, O, A, epochs=60, lr=1e-3, log=log)
    for it in range(3):
        beta = [0.5, 0.2, 0.0][it]
        On, An = collect(task, 60, 3000+100*it, agent=ag, beta=beta, noise=0.05, log=log)
        O = np.vstack([O, On]); A = np.vstack([A, An])
        bc_pretrain(ag, O, A, epochs=40, lr=5e-4, log=log)
        infos = evaluate(ag, env_fn, range(20000, 20020))
        log(f"DAgger iter {it}: dataset {len(O)}, eval success {np.mean([i['success'] for i in infos]):.2f}, roll progress {np.mean([i['max_progress_deg'] for i in infos]):.0f} deg, reasons {[(k, sum(i['reason']==k for i in infos)) for k in sorted(set(i['reason'] for i in infos))]}  [{time.time()-t0:.0f}s]")
        ag.save(out)
    np.savez(out.replace('.npz', '_data.npz'), O=O, A=A)
