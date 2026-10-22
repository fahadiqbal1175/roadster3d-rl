# Roadster-Inspired 3D Reinforcement Learning: Kickflip and Ramp Recovery

A physically consistent 6-DOF rigid-body simulation of two Roadster-style stunts with a configurable 10-nozzle thruster model, a Gymnasium environment, a NumPy PPO implementation, a classical reference controller, validation tests, reward-exploit tests, and 3D rendering of the simulated state.

> **Disclaimer.** Inspired by public coverage of the Tesla Roadster demonstration. This project does not imply that Tesla uses reinforcement learning or PPO, and it does not recover Tesla's actuators or flight controller. All vehicle numbers other than a few reported event figures are illustrative assumptions, and the 10-nozzle layout ("L1") is hypothetical. This is an independent student project, not affiliated with Tesla.

## Demo videos

Rendered from the simulated state (position and quaternion from the physics, nothing is a separately scripted animation).

- `videos/demo_kickflip_full.mp4`: approach, launch burn, 360 degree roll about the long axis, landing burn, touchdown, slow-motion replays.
- `videos/demo_ramp_recovery.mp4`: starts at the ramp lip with the car upside down, 180 degree roll recovery, landing. **The ramp is a visual prop only. Ramp contact is not simulated.**
- Shorter clips: `kickflip.mp4`, `landing.mp4`, `recover180.mp4`, `ramp.mp4`, and `baseline.mp4` (untrained network).

## Results (honest summary)

Held-out test seeds 10000 to 10099 (100 episodes per cell). Success means every landing criterion was met and the maneuver was completed.

| Task | Scripted controller (not learned) | Behaviour cloning + DAgger | BC + PPO fine-tune |
|---|---|---|---|
| Kickflip (+360 deg roll) | 100/100 | 100/100 | 100/100 |
| Ramp-exit recovery (about +180 deg roll) | 81/100 | 79/100 | 79/100 |
| Aerial 360 roll | 100/100 | 100/100 | 98/100 |
| Recover 180 | 93/100 | 87/100 | 87/100 |
| Recover 90 | 100/100 | 100/100 | 100/100 |
| Landing | 100/100 | 98/100 | 96/100 |

**Read this before quoting the numbers:**

- The learned policies **imitate the scripted controller** (behaviour cloning plus DAgger). PPO fine-tuning did not measurably improve them.
- **Pure PPO from scratch did not converge** in my compute budget (one CPU core): landing 0/100 after 500k steps, kickflip 0/100 after 400k steps.
- The learned policies are **brittle**: with the kickflip policy trained at nominal values, thrust x0.9 gave 0/20 and mass x0.85 gave 1/20, while the scripted controller succeeded in 20/20 for both. See `results/RESULTS.md` for the full sensitivity table.
- Footage reading (roll about the long axis, about 1.4 s for a full revolution) is my own rough estimate from monocular night footage, not telemetry. The roll direction is an arbitrary choice.

## What is modelled

- 6-DOF rigid body, quaternion attitude, body angular velocity, `I w_dot = tau - w x (I w)`, constant box inertia scaled with remaining gas.
- 10 body-fixed nozzles (4 rear, 4 corner, 2 yaw) with exhaust directions, first-order valve lag, bounded thrust and gas consumption from an assumed Isp. **Sign convention:** each nozzle stores its exhaust direction, the force on the car is the reaction (`F = -f * exhaust_dir`). A downward plume pushes the car up when upright and down when inverted.
- Wheel and chassis contact with spring-damper normal forces, friction, and a crash flag when the chassis touches the ground.
- Tasks: landing, recover 90, recover 180, aerial 360 roll, kickflip (ground launch, +360 deg roll over an obstacle lineup), ramp-exit recovery (airborne part only).
- Reward: dense tracking costs, potential-based shaping with the same discount factor PPO uses (terminal potential 0), success gated by signed roll progress, explicit exploit tests (skip the roll, premature landing, hover to timeout, collisions).

## Not modelled

- Ramp contact (tire contact, friction, suction on the curved ramp).
- Aerodynamic drag, wheel rolling dynamics, suspension beyond a spring at each wheel, real Tesla nozzle layout, gas properties, and valve response.
- The car model in the videos is a visual mesh only. Physics uses the box and wheel contact points.

## Quick start

```bash
git clone https://github.com/fahadiqbal1175/roadster3d-rl.git
cd roadster3d-rl
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
sudo apt install ffmpeg        # needed only for video rendering (macOS: brew install ffmpeg)
cd roadster3d
python test_physics.py          # 20 physics validation checks
python phase_report.py 3        # phase-by-phase feasibility record of the scripted kickflip
python exploit_tests.py         # reward-exploit tests
python make_outputs.py video kickflip     # short 4-camera video from the shipped checkpoint
python render_demo.py kickflip            # complete demo video (about 30 s)
python render_ramp_demo.py 10018          # ramp recovery demo video
```

Run everything from inside the `roadster3d/` folder (scripts use relative paths).

Colab: open `notebooks/Roadster_3D_RL_Kickflip_RampRecovery_Colab.ipynb` and run all cells. It has not been executed in Colab, only in a sandbox with Python 3.12.

## Repository layout

```
roadster3d/            simulator, environments, controllers, PPO, rendering, tests, checkpoints
  sim3d.py             quaternions, nozzle model, contact, integrator
  env3d.py             Gymnasium tasks (landing, recover90/180, roll360, kickflip)
  ramp.py              ramp-exit recovery task
  scripted.py          classical reference controller (not learned)
  ppo_np.py            PPO and behaviour cloning in NumPy
  bc_dagger.py         behaviour cloning + DAgger from the scripted controller
  finetune.py          PPO fine-tuning with validation checkpoint selection
  scratch_ppo.py       pure PPO from scratch
  test_physics.py, exploit_tests.py, sensitivity.py, phase_report.py
  render3d.py, render_demo.py, render_ramp_demo.py, make_outputs.py
  *.npz                checkpoints and the feasible height reference
notebooks/             Colab notebook
results/               RESULTS.md, JSON result files, comparison plots
videos/                demo videos
```

## Retraining

```bash
cd roadster3d
python bc_dagger.py kickflip kick_bc.npz
python finetune.py kickflip kick_bc.npz kick_ppo 250000
python scratch_ppo.py kickflip 400000     # pure PPO baseline, expected to fail at this budget
```

Training uses a single CPU core (about 1000 steps per second).

## License

MIT, see `LICENSE`. Tesla, Roadster, and related names belong to their owners. No Tesla footage is included in this repository.
