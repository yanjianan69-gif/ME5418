"""Train the locomotion actor-critic using PPO and the Isaac Gym environment."""

import argparse
import csv
import json
from pathlib import Path
from dataclasses import asdict

from isaacgym import gymapi  # Isaac Gym must precede PyTorch.
import gym
import torch
import stage1_gym
from stage1_rl import PPOConfig
from stage1_rl.runner import Runner, read_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100, help="Additional learning iterations")
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minibatches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--device", default="cpu", help="PyTorch device for the network")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="Restore training state from this submission's checkpoint")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/train"))
    args = parser.parse_args()
    torch.set_num_threads(1)  # Small single-environment MLP batches.
    if args.iterations < 1 or args.save_interval < 1 or (args.seed is not None and args.seed < 0):
        parser.error("iterations/save-interval must be positive and seed must be nonnegative")
    overrides = {name: value for name, value in {
        "num_steps_per_env": args.rollout_steps, "num_learning_epochs": args.epochs,
        "num_mini_batches": args.minibatches, "learning_rate": args.learning_rate}.items()
                 if value is not None}
    if args.resume and (overrides or args.seed is not None):
        parser.error("resume restores PPO settings and seed; omit their command-line overrides")
    task_config = stage1_gym.TaskConfig()
    try:
        if args.resume:
            data = read_checkpoint(args.resume)
            task_config = stage1_gym.TaskConfig(**data["task_config"])
            config = PPOConfig(**data["ppo_config"])
        else:
            config = PPOConfig(**overrides)
        if config.num_mini_batches > config.num_steps_per_env:
            raise ValueError("minibatches cannot exceed rollout-steps")
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("output directory already exists; select a new --output directory")
    env = gym.make("ArmDisturbanceLocomotion-v0", sim_device=args.sim_device,
                   config=task_config,
                   render_mode="human" if args.viewer else None)
    try:
        runner = (Runner.from_checkpoint(env, args.resume, device=args.device) if args.resume else
                  Runner(env, config, device=args.device, seed=42 if args.seed is None else args.seed))
        args.output.mkdir(parents=True)
        settings = {"seed": runner.seed, "sim_device": args.sim_device, "device": args.device,
                    "resume": str(args.resume.resolve()) if args.resume else None,
                    "start_iteration": runner.iteration, "additional_iterations": args.iterations,
                    "task": asdict(env.unwrapped.cfg), "network": asdict(runner.model.config), "ppo": asdict(config)}
        (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        runner.save(args.output / "initial.pt")
        with (args.output / "training.csv").open("w", newline="") as stream:
            writer = None
            for _ in range(args.iterations):
                metrics = runner.learn_iteration()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                stream.flush()
                if runner.iteration % args.save_interval == 0:
                    runner.save(args.output / "checkpoints" / f"iteration_{runner.iteration:06d}.pt")
                    runner.save(args.output / "policy.pt")
                print(f"iteration={runner.iteration} steps={runner.environment_steps} "
                      f"reward={metrics['reward_mean']:.5f} policy_loss={metrics['policy_loss']:.5f} "
                      f"value_loss={metrics['value_loss']:.5f} grad_norm={metrics['grad_norm']:.5f}")
        runner.save(args.output / "policy.pt")
        print(f"Saved {args.output / 'policy.pt'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
