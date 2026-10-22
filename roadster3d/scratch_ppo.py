"""Pure PPO from scratch (NO behaviour cloning, no scripted controller involved in training). Reported separately."""
import sys, json, numpy as np
from env3d import *
from ramp import RampExitEnv
from ppo_np import *
from finetune import summarize, make_env_fn
task, steps = sys.argv[1], int(sys.argv[2])
env_fn = make_env_fn(task)
ag = Agent(env_fn().obs_dim, 10, seed=1, log_std=-0.7)
hist = []
def log(*a): print(*a, flush=True)
train(ag, env_fn, steps, n_envs=8, n_steps=256, lr=3e-4, epochs=8, log=log, ckpt_path=f"scratch_{task}.npz", history=hist, seed=1)
infos = evaluate(ag, env_fn, range(10000, 10100))
res = dict(task=task, steps=steps, test=summarize(infos), history=hist)
log("PURE PPO TEST:", res['test']['successes'], "/ 100", res['test']['reasons'], "mean roll progress", res['test']['mean_progress_deg'])
json.dump(res, open(f"scratch_{task}_results.json", "w"), indent=1, default=float)
