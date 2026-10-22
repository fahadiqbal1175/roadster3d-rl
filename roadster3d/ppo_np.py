"""
Minimal PPO (clip objective, GAE, Gaussian policy, running observation normalisation) in pure NumPy with manual back-propagation.
Used because it is fast on a single CPU core and has no framework dependency. Same code runs in Colab.
"""
import time, json, math, os
import numpy as np

class RunningNorm:
    def __init__(self, dim):
        self.n = 1e-4; self.mean = np.zeros(dim); self.var = np.ones(dim)
    def update(self, x):
        bm = x.mean(0); bv = x.var(0); bn = x.shape[0]
        d = bm - self.mean; tot = self.n + bn
        self.mean = self.mean + d*bn/tot
        self.var = (self.var*self.n + bv*bn + d*d*self.n*bn/tot)/tot; self.n = tot
    def __call__(self, x):
        return np.clip((x - self.mean)/np.sqrt(self.var + 1e-8), -8, 8)

def _init(rng, i, o, scale=1.0):
    return rng.normal(0, 1, (i, o))*scale/math.sqrt(i)

class MLP:
    """obs -> h1 -> h2 -> out, tanh hidden units"""
    def __init__(self, rng, i, o, h=96, out_scale=0.01):
        self.p = dict(W1=_init(rng, i, h, math.sqrt(2)), b1=np.zeros(h), W2=_init(rng, h, h, math.sqrt(2)), b2=np.zeros(h),
                      W3=_init(rng, h, o, out_scale*math.sqrt(h)), b3=np.zeros(o))
    def forward(self, x, cache=False):
        p = self.p
        h1 = np.tanh(x @ p['W1'] + p['b1']); h2 = np.tanh(h1 @ p['W2'] + p['b2']); y = h2 @ p['W3'] + p['b3']
        return (y, (x, h1, h2)) if cache else y
    def backward(self, cache, dy):
        x, h1, h2 = cache; p = self.p; g = {}
        g['W3'] = h2.T @ dy; g['b3'] = dy.sum(0)
        d2 = (dy @ p['W3'].T)*(1 - h2*h2)
        g['W2'] = h1.T @ d2; g['b2'] = d2.sum(0)
        d1 = (d2 @ p['W2'].T)*(1 - h1*h1)
        g['W1'] = x.T @ d1; g['b1'] = d1.sum(0)
        return g

class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}; self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
    def step(self, params, grads, lr=None):
        self.t += 1; lr = lr or self.lr
        for k in params:
            g = grads[k]
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g; self.v[k] = self.b2*self.v[k] + (1-self.b2)*g*g
            mh = self.m[k]/(1-self.b1**self.t); vh = self.v[k]/(1-self.b2**self.t)
            params[k] -= lr*mh/(np.sqrt(vh) + self.eps)

class Agent:
    def __init__(self, obs_dim, act_dim, seed=0, hidden=96, init_bias=-0.5, log_std=-0.7):
        rng = np.random.default_rng(seed)
        self.pi = MLP(rng, obs_dim, act_dim, hidden, 0.01); self.vf = MLP(rng, obs_dim, 1, hidden, 1.0)
        self.pi.p['b3'][:] = init_bias
        self.log_std = np.full(act_dim, log_std)
        self.norm = RunningNorm(obs_dim); self.act_dim = act_dim; self.obs_dim = obs_dim
    def act(self, obs, rng, deterministic=False):
        x = self.norm(obs); mu = self.pi.forward(x)
        if deterministic: return mu, mu, self.vf.forward(x)[:, 0]
        a = mu + np.exp(self.log_std)*rng.normal(size=mu.shape)
        logp = -0.5*(((a - mu)/np.exp(self.log_std))**2).sum(1) - self.log_std.sum() - 0.5*self.act_dim*math.log(2*math.pi)
        return a, logp, self.vf.forward(x)[:, 0]
    def value(self, obs):
        return self.vf.forward(self.norm(obs))[:, 0]
    def save(self, path):
        d = {f"pi_{k}": v for k, v in self.pi.p.items()}; d.update({f"vf_{k}": v for k, v in self.vf.p.items()})
        d.update(log_std=self.log_std, nmean=self.norm.mean, nvar=self.norm.var, nn=self.norm.n)
        np.savez(path, **d)
    def load(self, path):
        d = np.load(path)
        for k in self.pi.p: self.pi.p[k] = d[f"pi_{k}"].copy()
        for k in self.vf.p: self.vf.p[k] = d[f"vf_{k}"].copy()
        self.log_std = d['log_std'].copy(); self.norm.mean = d['nmean'].copy(); self.norm.var = d['nvar'].copy(); self.norm.n = float(d['nn'])

def bc_pretrain(agent, obs, acts, epochs=60, lr=1e-3, batch=512, seed=0, log=print):
    """supervised regression of the policy mean onto scripted-controller actions (behaviour cloning). LABEL RESULTS ACCORDINGLY."""
    rng = np.random.default_rng(seed)
    agent.norm.update(obs)
    X = agent.norm(obs); Y = acts; n = len(X); opt = Adam(agent.pi.p, lr=lr)
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]; y, cache = agent.pi.forward(X[idx], True)
            err = y - Y[idx]; tot += (err**2).sum()
            g = agent.pi.backward(cache, 2*err/len(idx)); opt.step(agent.pi.p, g)
        if ep % 10 == 0 or ep == epochs-1: log(f"  BC epoch {ep}: mse {tot/(n*Y.shape[1]):.4f}")

def evaluate(agent, env_fn, seeds, deterministic=True):
    infos = []
    for sd in seeds:
        env = env_fn(); rng = np.random.default_rng(sd)
        o, _ = env.reset(seed=sd)
        while True:
            a, _, _ = agent.act(o[None, :], rng, deterministic)
            o, r, te, tr, info = env.step(a[0])
            if te or tr: infos.append(info); break
    return infos

def train(agent, env_fn, total_steps, n_envs=8, n_steps=256, gamma=0.99, lam=0.95, lr=3e-4, epochs=8, minibatch=1024, clip=0.2,
          ent_coef=0.0, vf_coef=0.5, rew_scale=0.1, seed=0, log=print, ckpt_path=None, ckpt_every=10, eval_fn=None, eval_every=20,
          lr_end_frac=0.2, max_grad=1.0, target_kl=0.03, start_step=0, history=None, pi_warmup=0):
    rng = np.random.default_rng(seed)
    envs = [env_fn() for _ in range(n_envs)]
    obs = np.stack([e.reset(seed=int(rng.integers(1e9)))[0] for e in envs])
    ep_ret = np.zeros(n_envs); ep_len = np.zeros(n_envs)
    popt = Adam(agent.pi.p, lr); vopt = Adam(agent.vf.p, lr)
    n_iter = max(1, total_steps // (n_envs*n_steps))
    hist = history if history is not None else []
    recent = []; t0 = time.time(); steps_done = start_step
    for it in range(n_iter):
        frac = 1 - it/n_iter; cur_lr = lr*(lr_end_frac + (1-lr_end_frac)*frac)
        B_obs = np.zeros((n_steps, n_envs, agent.obs_dim)); B_act = np.zeros((n_steps, n_envs, agent.act_dim))
        B_logp = np.zeros((n_steps, n_envs)); B_rew = np.zeros((n_steps, n_envs)); B_done = np.zeros((n_steps, n_envs))
        B_val = np.zeros((n_steps, n_envs)); B_boot = np.zeros((n_steps, n_envs))
        agent.norm.update(obs)
        for t in range(n_steps):
            a, logp, v = agent.act(obs, rng)
            B_obs[t] = obs; B_act[t] = a; B_logp[t] = logp; B_val[t] = v
            for i, e in enumerate(envs):
                o2, r, te, tr, info = e.step(a[i])
                ep_ret[i] += r; ep_len[i] += 1
                B_rew[t, i] = r*rew_scale
                if te or tr:
                    if tr and not te: B_boot[t, i] = float(agent.value(o2[None, :])[0])   # time-limit: bootstrap
                    B_done[t, i] = 1.0
                    recent.append(dict(ret=ep_ret[i], len=ep_len[i], success=info['success'], reason=info['reason'], prog=info['max_progress_deg']))
                    ep_ret[i] = 0; ep_len[i] = 0
                    o2, _ = e.reset(seed=int(rng.integers(1e9)))
                obs[i] = o2
        last_v = agent.value(obs)
        adv = np.zeros((n_steps, n_envs)); gae = 0
        for t in reversed(range(n_steps)):
            nv = last_v if t == n_steps-1 else B_val[t+1]
            nonterm = 1.0 - B_done[t]
            delta = B_rew[t] + gamma*(nv*nonterm + B_boot[t]*rew_scale) - B_val[t]
            gae = delta + gamma*lam*nonterm*gae; adv[t] = gae
        ret = adv + B_val
        N = n_steps*n_envs
        X = agent.norm(B_obs.reshape(N, -1)); A = B_act.reshape(N, -1); LP = B_logp.reshape(N); AD = adv.reshape(N); RT = ret.reshape(N)
        AD = (AD - AD.mean())/(AD.std() + 1e-8)
        kl_last = 0
        for ep in range(epochs):
            perm = rng.permutation(N); stop = False
            for i in range(0, N, minibatch):
                idx = perm[i:i+minibatch]; m = len(idx)
                mu, cache = agent.pi.forward(X[idx], True)
                std = np.exp(agent.log_std)
                logp = -0.5*(((A[idx] - mu)/std)**2).sum(1) - agent.log_std.sum() - 0.5*agent.act_dim*math.log(2*math.pi)
                ratio = np.exp(logp - LP[idx]); a_ = AD[idx]
                unclipped = (ratio*a_ < np.clip(ratio, 1-clip, 1+clip)*a_) | ((ratio >= 1-clip) & (ratio <= 1+clip))
                coef = np.where(unclipped, -a_*ratio, 0.0)/m              # dL/dlogp
                dmu = coef[:, None]*(A[idx] - mu)/std**2
                g = agent.pi.backward(cache, dmu)
                gls = (coef[:, None]*(((A[idx] - mu)/std)**2 - 1)).sum(0) - ent_coef/ m * m
                gn = math.sqrt(sum(float((v*v).sum()) for v in g.values()))
                if gn > max_grad:
                    for k in g: g[k] *= max_grad/gn
                if it >= pi_warmup:      # policy frozen during value-function warm-up (used when starting from a behaviour-cloned policy)
                    popt.step(agent.pi.p, g, cur_lr); agent.log_std = np.clip(agent.log_std - cur_lr*3*np.sign(gls)*np.minimum(np.abs(gls), 1.0), -2.5, 0.0)
                v_, vcache = agent.vf.forward(X[idx], True)
                dv = vf_coef*(v_[:, 0] - RT[idx])[:, None]/m
                gv = agent.vf.backward(vcache, dv)
                gvn = math.sqrt(sum(float((v*v).sum()) for v in gv.values()))
                if gvn > max_grad:
                    for k in gv: gv[k] *= max_grad/gvn
                vopt.step(agent.vf.p, gv, cur_lr)
                kl_last = float(np.mean(LP[idx] - logp))
            if kl_last > target_kl: break
        steps_done += N
        if (it % 5 == 0) or it == n_iter-1:
            rr = recent[-60:]
            row = dict(it=it, steps=steps_done, ret=float(np.mean([r['ret'] for r in rr])) if rr else float('nan'),
                       succ=float(np.mean([r['success'] for r in rr])) if rr else float('nan'),
                       prog=float(np.mean([r['prog'] for r in rr])) if rr else float('nan'), std=float(np.exp(agent.log_std).mean()),
                       sec=time.time()-t0)
            hist.append(row)
            log(f"it {it:4d} steps {steps_done:8d} | ret {row['ret']:7.2f} succ {row['succ']:.2f} roll {row['prog']:6.1f} deg | std {row['std']:.2f} | {row['sec']:.0f}s")
        if ckpt_path and (it % ckpt_every == 0 or it == n_iter-1):
            agent.save(ckpt_path)
        if eval_fn and (it % eval_every == 0 or it == n_iter-1):
            eval_fn(agent, steps_done)
    return hist
