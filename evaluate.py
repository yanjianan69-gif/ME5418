"""Evaluate a saved locomotion agent in real Isaac Gym and write physical metrics."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from isaacgym import gymapi  # Must precede torch.
import gym
import torch
import stage1_gym
from stage1_rl.evaluation import evaluate_policy
from stage1_rl.runner import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, help="Optional per-episode cap; capped runs are not successes")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW_RATE"))
    parser.add_argument("--pose-command", type=float, nargs=2, metavar=("PITCH", "ROLL"))
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("episodes/max-steps must be positive and seed nonnegative")
    if args.output.exists():
        parser.error("output file already exists; choose a new --output")
    torch.set_num_threads(1)
    model, config = load_policy(args.checkpoint, args.device)
    options = {key: value for key, value in {
        "command": args.command, "pose_command": args.pose_command}.items() if value is not None}
    env = gym.make("ArmDisturbanceLocomotion-v0", config=config, sim_device=args.sim_device,
                   render_mode="human" if args.viewer else None)
    try:
        result = evaluate_policy(env, model, args.episodes, args.seed, args.max_steps, options)
    finally:
        env.close()
    result["settings"] = {"checkpoint": str(args.checkpoint.resolve()), "task": asdict(config),
                          "seed": args.seed, "max_steps": args.max_steps, "reset_options": options,
                          "sim_device": args.sim_device, "device": args.device,
                          "policy": "deterministic actor mean", "aggregate": "step-weighted physical errors"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
