"""Run an actor-critic policy in Isaac Gym, optionally loading a trained checkpoint."""

import argparse
from isaacgym import gymapi  # Must precede torch imports in any downstream learner.
import gym
import numpy as np
import torch
import stage1_gym
from stage1_gym.task import OBSERVATION_SIZE
from stage1_rl import ActorCritic
from stage1_rl.runner import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Open the Isaac Gym viewer")
    parser.add_argument("--sim-device", default="cuda:0", help="cuda:N (default cuda:0), or explicit cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="PyTorch device for policy inference")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--checkpoint", help="Policy checkpoint saved by train.py")
    policy.add_argument("--random-actions", action="store_true")
    policy.add_argument("--zero-actions", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.steps < 1 or args.seed < 0:
        parser.error("steps must be positive and seed must be nonnegative")
    torch.manual_seed(args.seed)
    config = stage1_gym.TaskConfig()
    model = None
    if args.checkpoint:
        model, config = load_policy(args.checkpoint, args.device)
        print(f"policy={args.checkpoint}")
    elif not (args.random_actions or args.zero_actions):
        model = ActorCritic(OBSERVATION_SIZE).to(args.device).eval()
        print("policy=untrained actor (interface demonstration)")
    env = gym.make("ArmDisturbanceLocomotion-v0", sim_device=args.sim_device,
                   config=config,
                   render_mode="human" if args.viewer else None)
    try:
        obs, info = env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        print(f"observation={obs.shape}; action={env.action_space.shape}; dt={env.unwrapped.cfg.dt}s")
        for step in range(args.steps):
            if model is not None:
                with torch.no_grad():
                    inputs = torch.as_tensor(obs, device=args.device).unsqueeze(0)
                    action = model.act_inference(inputs)[0].cpu().numpy()
            else:
                action = env.action_space.sample() if args.random_actions else np.zeros(12, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if step % 100 == 0:
                print(f"step={step} reward={reward:.5f} height={info['base_height']:.3f} "
                      f"command={info['command']} pose={info['pose_command']} "
                      f"arm_target={info['arm_target'].round(3)}")
            if terminated or truncated:
                print(f"episode={info['episode']} terminated={terminated} truncated={truncated} "
                      f"reason={info['termination_reasons']}")
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
