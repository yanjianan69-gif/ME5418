"""Run the official Gym checker and a real PhysX rollout (GPU by default)."""

import argparse
from isaacgym import gymapi  # Import before torch.
import gym
from gym.utils.env_checker import check_env
import numpy as np
import stage1_gym


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-device', default='cuda:0')
    args = parser.parse_args()
    env = gym.make("ArmDisturbanceLocomotion-v0", sim_device=args.sim_device)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=123)
        initial_arm = env.unwrapped.state.arm_pos.copy()
        max_arm_motion = 0.0
        episodes = 0
        for step in range(500):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12, dtype=np.float32))
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            max_arm_motion = max(max_arm_motion, float(np.max(np.abs(env.unwrapped.state.arm_pos - initial_arm))))
            if terminated or truncated:
                episodes += 1
                env.reset()
                initial_arm = env.unwrapped.state.arm_pos.copy()
        assert max_arm_motion > 1e-3, "Arm did not move in real physics"
        print(f"PASS ({args.sim_device}): Gym checker + 500 PhysX steps; {episodes} completed episodes; "
              f"max arm joint motion={max_arm_motion:.4f} rad")
    finally:
        env.close()


if __name__ == "__main__":
    main()
