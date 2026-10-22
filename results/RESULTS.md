# Roadster-inspired 3D RL: measured results

All numbers below were measured in a sandbox (Python 3.12, one CPU core, NumPy). Held-out test seeds are 10000+, validation seeds 20000+. The Colab notebook itself has not been run in Colab.

Three sources of results are kept apart on purpose: (S) scripted classical controller, not learned; (BC) behaviour cloning plus DAgger from that controller; (BC+PPO) the BC policy fine-tuned with PPO and selected on validation seeds. Pure PPO from scratch is reported separately.

## Physics validation
20/20 checks passed (simulator speed about 1334 control steps/s).

## Held-out success rates (100 episodes each, success = every landing criterion met and the maneuver completed)

| task | scripted (S) | BC+DAgger (BC) | BC+PPO fine-tune | selected PPO checkpoint |
|---|---|---|---|---|
| kickflip (+360 deg roll) | 100/100 (95% CI 0.96 to 1.00) | 100/100 (95% CI 0.96 to 1.00) | 100/100 (95% CI 0.96 to 1.00) | kick_ppo_best.npz (best-validation checkpoint at 155,648 PPO steps; run cut short by a sandbox restart, only this saved checkpoint was tested) |
| ramp-exit recovery (about +180 deg roll) | 81/100 (95% CI 0.72 to 0.87) | 79/100 (95% CI 0.70 to 0.86) | 79/100 (95% CI 0.70 to 0.86) | ramp_exit_ppo_best.npz (98304 PPO steps) |
| aerial 360 roll | 100/100 (95% CI 0.96 to 1.00) | 100/100 (95% CI 0.96 to 1.00) | 98/100 (95% CI 0.93 to 0.99) | roll360_ppo_best.npz (98304 PPO steps) |
| recover 180 | 93/100 (95% CI 0.86 to 0.97) | 87/100 (95% CI 0.79 to 0.92) | 87/100 (95% CI 0.79 to 0.92) | recover180_ppo_best.npz (2048 PPO steps) |
| recover 90 | 100/100 (95% CI 0.96 to 1.00) | 100/100 (95% CI 0.96 to 1.00) | 100/100 (95% CI 0.96 to 1.00) | recover90_ppo_best.npz (53248 PPO steps) |
| landing | 100/100 (95% CI 0.96 to 1.00) | 98/100 (95% CI 0.93 to 0.99) | 96/100 (95% CI 0.90 to 0.98) | landing_ppo_best.npz (98304 PPO steps) |

Failure modes (BC+PPO policy): kick: {}; ramp_exit: {'vz_ok': 21, 'rate_ok': 9, 'no_chassis': 9}; roll360: {'vz_ok': 2}; recover180: {'rate_ok': 2, 'vz_ok': 13, 'no_chassis': 2, 'wheels_ok': 1}; recover90: {}; landing: {'vz_ok': 4}

vz_ok = touchdown vertical speed above 1.5 m/s, no_chassis = chassis contact, rate_ok = angular rate above 0.3 rad/s at the end, slip_ok = body-lateral speed above 1.5 m/s.

## Pure PPO from scratch (no demonstrations)

- landing: 500000 steps, test 0/100, end reasons {'out_of_bounds': 91, 'landed': 9}, mean roll progress 4.9 deg
- kickflip: 400000 steps, test 0/100, end reasons {'obstacle_collision': 100}, mean roll progress 0.3 deg
- Earlier smoke test: landing, 60k steps, 0/20.

## Reward-exploit tests (10 seeds each, scripted policies)

- A reference: scripted full maneuver: mean return 27.2, success 10/10, outcomes {'landed': 10}
- B exploit: skip the rotation (stay upright): mean return -63.5, success 0/10, outcomes {'touchdown_before_maneuver_complete': 5, 'obstacle_collision': 5}
- C exploit: premature landing (no thrust, drop): mean return -59.3, success 0/10, outcomes {'obstacle_collision': 10}
- D exploit: delay touchdown (hover to timeout): mean return -21.2, success 0/10, outcomes {'timeout': 10}
- E exploit: hover after completing the roll: mean return -25.1, success 0/10, outcomes {'timeout': 10}
- F exploit: crash into the obstacle / chassis contact: mean return -59.3, success 0/10, outcomes {'obstacle_collision': 10}
- G exploit: collide on purpose (rear thrust, no lift): mean return -57.5, success 0/10, outcomes {'obstacle_collision': 10}

Checks: PASS full maneuver beats skipping the rotation; PASS skipping the rotation never counts as success; PASS full maneuver beats premature landing; PASS full maneuver beats hovering to timeout; PASS hovering after the roll without landing is worse than landing; PASS colliding on purpose is worse than any non-colliding failure (hover to timeout); PASS ending the episode early by crashing is never rewarded over continuing; PASS no exploit policy succeeds

## Sensitivity of the kickflip (20 episodes per row; policies were trained at nominal values only)

| condition | BC+PPO policy | scripted controller |
|---|---|---|
| nominal | 20/20 | 20/20 |
| thrust x0.6 | 0/20 | 0/20 |
| thrust x0.7 | 0/20 | 0/20 |
| thrust x0.8 | 0/20 | 0/20 |
| thrust x0.9 | 0/20 | 20/20 |
| thrust x1.1 | 20/20 | 20/20 |
| thrust x1.2 | 16/20 | 20/20 |
| corner nozzles x0.6 | 0/20 | 0/20 |
| corner nozzles x0.75 | 0/20 | 0/20 |
| mass x0.85 | 1/20 | 20/20 |
| mass x1.15 | 0/20 | 13/20 |
| roll inertia x0.8 | 8/20 | 20/20 |
| roll inertia x1.25 | 0/20 | 19/20 |

The scripted controller tolerates most changes except weaker thrust (0.8x and below, or weaker corner nozzles), which makes the maneuver physically infeasible for this layout. The imitation policies are much more brittle than the controller they imitate.

## Phase-by-phase feasibility record (scripted, kickflip seed 3)

| phase | t0 | t1 | z end (m) | z max (m) | vz end | mean corner kN | peak corner kN | roll end (deg) | gas (kg) |
|---|---|---|---|---|---|---|---|---|---|
| launch | 0.0 | 0.86 | 3.57 | 3.57 | 6.4 | 28.4 | 29.3 | -0.0 | 35.3 |
| roll | 0.86 | 2.2 | 5.96 | 7.21 | -3.84 | 11.5 | 16.0 | 353.1 | 22.6 |
| recover/descent | 2.2 | 2.6 | 4.03 | 5.89 | -6.38 | 5.5 | 11.0 | 359.2 | 3.3 |
| landing burn | 2.6 | 3.46 | 0.87 | 3.9 | -0.9 | 26.2 | 28.7 | 360.0 | 32.7 |
| touchdown | 3.46 | 4.08 | 0.84 | 0.86 | 0.0 | 0.7 | 10.0 | 359.9 | 0.8 |
