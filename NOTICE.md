# Sources and licenses

The locomotion environment and PPO implementation use RoboDuet source code and
Go2-X5 robot assets. The MIT notice is retained in `LICENSE`. The legged_gym and
rsl_rl BSD notices are retained in `LICENSES/legged_gym.txt` and
`LICENSES/rsl_rl.txt`.

| Source component | Submission component |
| --- | --- |
| RoboDuet `roboduet/legged_robot.py` | Simulation, joint control, reset and gait logic in `stage1_gym/` |
| RoboDuet `roboduet/wbc_env.py` | Arm disturbance generation and locomotion observations |
| RoboDuet `rewards/rewards.py` and task configs | Locomotion rewards and control parameters |
| RoboDuet `ppo_cse_automatic/dog_ac.py` | `stage1_rl/actor_critic.py` |
| RoboDuet `ppo_cse_automatic/ppo.py` | `stage1_rl/ppo.py` |
| RoboDuet `ppo_cse_automatic/rollout_storage.py` | `stage1_rl/rollout_storage.py` |
| RoboDuet locomotion rollout loop | `stage1_rl/runner.py` |
| WBCTrackingShare `legged_robot.py` performance metrics | Physical velocity, tilt and height measurements in `stage1_rl/evaluation.py` |
| RoboDuet `resources/robots/go2_x5_v3` | Robot URDF and meshes in `stage1_gym/assets/` |

The implementation uses a single locomotion policy, standard Gym episode
semantics and single-frame observations. Network dimensions, arm target velocity
observations, posture commands, timeout handling and checkpoint interfaces are
specified in `README.md`.

The Learning Agent additions use the local WBCTrackingShare/RoboDuet project's
stage-1 rollout/save/evaluate workflow as a reference. Periodic atomic checkpoints,
Adam-aware training restart, bounded standalone evaluation and the end-to-end demo
are implemented for this submission's single-environment Gym interface. The
evaluation metrics use the source project's physical-error definitions, adapted
to step-weighted NumPy accumulation. No WBCTracking installation or online logging
service is needed. Stage-2 arm-policy training, history/adaptation networks and
upstream checkpoint formats are outside this submission's scope.

External libraries include PyTorch, NumPy and OpenAI Gym. NVIDIA Isaac Gym Preview 4
is an external proprietary simulator and is not redistributed. The optional custom
PyTorch binary in `vendor/` retains its upstream notices; its identity and checksum
are recorded in `vendor/README.md`.

AI assistance: OpenAI Codex assisted with code integration, tests and documentation.
