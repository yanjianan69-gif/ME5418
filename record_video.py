"""Record a headless Isaac Gym rollout to an MP4 file."""

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

from isaacgym import gymapi  # Must precede torch imports.
import gym
import imageio.v2 as imageio
import numpy as np
import torch
import stage1_gym
from stage1_rl.runner import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--output", default="rollout.mp4")
    parser.add_argument("--device", default="cpu", help="PyTorch device for policy inference")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--disturbance-intensity", type=float, default=1.0)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--checkpoint", help="Policy checkpoint saved by train.py or train_vec.py")
    policy.add_argument("--random-actions", action="store_true")
    policy.add_argument("--zero-actions", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model, config = None, replace(stage1_gym.TaskConfig(), disturbance_intensity=args.disturbance_intensity)
    if args.checkpoint:
        model, config = load_policy(args.checkpoint, args.device)
        config = replace(config, disturbance_intensity=args.disturbance_intensity)
    elif not (args.random_actions or args.zero_actions):
        args.zero_actions = True

    env = gym.make("ArmDisturbanceLocomotion-v0", sim_device=args.sim_device, config=config, render_mode="rgb_array")
    options = {"command": args.command} if args.command is not None else None
    obs, info = env.reset(seed=args.seed, options=options)
    env.action_space.seed(args.seed)
    with imageio.get_writer(output, fps=args.fps, codec="libx264", quality=8) as writer:
        frame = env.render()
        if frame is not None:
            writer.append_data(frame)
        for step in range(args.steps):
            if model is not None:
                with torch.no_grad():
                    inputs = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
                    action = model.act_inference(inputs)[0].cpu().numpy()
            else:
                action = env.action_space.sample() if args.random_actions else np.zeros(12, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                writer.append_data(frame)
            if step % 100 == 0:
                print(f"step={step} reward={reward:.5f} height={info['base_height']:.3f}")
            if terminated or truncated:
                obs, info = env.reset(options=options)
                frame = env.render()
                if frame is not None:
                    writer.append_data(frame)
    print(f"Wrote {output}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
