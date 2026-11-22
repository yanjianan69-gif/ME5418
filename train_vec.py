"""Train PPO with many Go2-X5 environments inside one Isaac Gym simulation."""

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from dataclasses import asdict, replace

from isaacgym import gymapi
import numpy as np
import torch

import stage1_gym
from stage1_gym.task import OBSERVATION_LAYOUT, OBSERVATION_SIZE
from stage1_gym.vec_env import Stage1VecEnv
from stage1_rl import ActorCritic, PPO, PPOConfig


def save_checkpoint(path, env, model, algorithm, config, seed, iteration, environment_steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": 1,
        "environment_id": "ArmDisturbanceLocomotion-v0",
        "observation_layout": OBSERVATION_LAYOUT,
        "num_obs": model.num_obs,
        "num_actions": model.num_actions,
        "network_config": asdict(model.config),
        "ppo_config": asdict(config),
        "task_config": asdict(env.cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": algorithm.optimizer.state_dict(),
        "iteration": iteration,
        "environment_steps": environment_steps,
        "seed": seed,
    }
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def vector_metrics(info, rewards, terminated, truncated, episode_returns, target_height):
    velocity_error = info["base_velocity"] - info["command"]
    tilt = info["base_rpy"][:, :2]
    height_error = info["base_height"] - target_height
    completed = episode_returns[np.isfinite(episode_returns)]
    return {
        "reward_mean": float(np.mean(rewards)),
        "vx_mae_mps": float(np.mean(np.abs(velocity_error[:, 0]))),
        "vy_mae_mps": float(np.mean(np.abs(velocity_error[:, 1]))),
        "yaw_rate_mae_rad_s": float(np.mean(np.abs(velocity_error[:, 2]))),
        "lin_vel_rmse_mps": float(np.sqrt(np.mean(np.sum(velocity_error[:, :2] ** 2, axis=1)))),
        "yaw_rate_rmse_rad_s": float(np.sqrt(np.mean(velocity_error[:, 2] ** 2))),
        "roll_rms_rad": float(np.sqrt(np.mean(tilt[:, 0] ** 2))),
        "pitch_rms_rad": float(np.sqrt(np.mean(tilt[:, 1] ** 2))),
        "base_height_rmse_m": float(np.sqrt(np.mean(height_error ** 2))),
        "terminations": int(np.count_nonzero(terminated)),
        "timeouts": int(np.count_nonzero(truncated & ~terminated)),
        "episodes": int(len(completed)),
        "episode_return_mean": float(np.mean(completed)) if len(completed) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disturbance-intensity", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("runs/train_vec"))
    args = parser.parse_args()
    if min(args.num_envs, args.iterations, args.rollout_steps, args.epochs, args.minibatches, args.save_interval) < 1:
        parser.error("num-envs, iterations, rollout-steps, epochs, minibatches and save-interval must be positive")
    if args.minibatches > args.num_envs * args.rollout_steps:
        parser.error("minibatches cannot exceed num_envs * rollout_steps")
    if args.output.exists():
        parser.error("output directory already exists; select a new --output directory")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = PPOConfig(num_steps_per_env=args.rollout_steps, num_learning_epochs=args.epochs,
                       num_mini_batches=args.minibatches, learning_rate=args.learning_rate)
    task_config = replace(stage1_gym.TaskConfig(), disturbance_intensity=args.disturbance_intensity)
    env = Stage1VecEnv(args.num_envs, task_config, args.sim_device, args.seed)
    model = ActorCritic(OBSERVATION_SIZE, 12).to(args.device)
    algorithm = PPO(model, config)
    algorithm.init_storage(num_envs=args.num_envs)
    args.output.mkdir(parents=True)
    settings = {"seed": args.seed, "num_envs": args.num_envs, "sim_device": args.sim_device, "device": args.device,
                "iterations": args.iterations, "task": asdict(env.cfg), "network": asdict(model.config),
                "ppo": asdict(config)}
    (args.output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    iteration = 0
    environment_steps = 0
    save_checkpoint(args.output / "initial.pt", env, model, algorithm, config, args.seed, iteration, environment_steps)
    try:
        obs, _ = env.reset(seed=args.seed)
        with (args.output / "training.csv").open("w", newline="") as stream:
            writer = None
            for _ in range(args.iterations):
                rewards_seen = []
                last_info = None
                last_terminated = last_truncated = last_episode_returns = None
                for _ in range(config.num_steps_per_env):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
                    action = algorithm.act(obs_tensor)
                    next_obs, rewards, terminated, truncated, info = env.step(action.cpu().numpy())
                    algorithm.process_env_step(torch.as_tensor(rewards, dtype=torch.float32, device=args.device),
                                               torch.as_tensor(terminated, dtype=torch.bool, device=args.device),
                                               torch.as_tensor(truncated, dtype=torch.bool, device=args.device),
                                               torch.as_tensor(next_obs, dtype=torch.float32, device=args.device))
                    environment_steps += args.num_envs
                    rewards_seen.append(rewards)
                    last_info, last_terminated, last_truncated = info, terminated, truncated
                    last_episode_returns = info["episode_returns"]
                    obs = next_obs
                algorithm.compute_returns()
                metrics = vector_metrics(last_info, np.concatenate(rewards_seen), last_terminated, last_truncated,
                                         last_episode_returns, env.cfg.target_height)
                metrics.update(algorithm.update())
                iteration += 1
                metrics.update(iteration=iteration, environment_steps=environment_steps)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(metrics))
                    writer.writeheader()
                writer.writerow(metrics)
                stream.flush()
                if iteration % args.save_interval == 0:
                    save_checkpoint(args.output / "checkpoints" / f"iteration_{iteration:06d}.pt",
                                    env, model, algorithm, config, args.seed, iteration, environment_steps)
                    save_checkpoint(args.output / "policy.pt", env, model, algorithm, config,
                                    args.seed, iteration, environment_steps)
                print(f"iteration={iteration} envs={args.num_envs} steps={environment_steps} "
                      f"reward={metrics['reward_mean']:.5f} policy_loss={metrics['policy_loss']:.5f} "
                      f"value_loss={metrics['value_loss']:.5f} grad_norm={metrics['grad_norm']:.5f}")
        save_checkpoint(args.output / "policy.pt", env, model, algorithm, config, args.seed, iteration, environment_steps)
        print(f"Saved {args.output / 'policy.pt'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
