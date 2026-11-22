"""Gym rollout collection, PPO updates and portable policy checkpoints."""

from dataclasses import asdict
from pathlib import Path
import os
import tempfile
import numpy as np
import torch

from stage1_gym.config import TaskConfig
from stage1_gym.task import OBSERVATION_LAYOUT, OBSERVATION_SIZE
from .actor_critic import ActorCritic
from .config import NetworkConfig, PPOConfig
from .ppo import PPO
from .evaluation import TrackingMetrics


class Runner:
    def __init__(self, env, config=None, network_config=None, device="cpu", seed=42):
        self.env = env
        self.device = torch.device(device)
        self.config = config or PPOConfig()
        self.seed = seed
        torch.manual_seed(seed)
        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], network_config).to(self.device)
        self.algorithm = PPO(self.model, self.config)
        self.algorithm.init_storage(num_envs=1)
        self.observation = None
        self.iteration = 0
        self.environment_steps = 0

    def _tensor(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

    def collect_rollout(self):
        if self.observation is None:
            self.observation, _ = self.env.reset(seed=self.seed)
        episode_returns = []
        tracking = TrackingMetrics(self.env.unwrapped.cfg.target_height)
        terminations = timeouts = 0
        rewards = []
        for _ in range(self.config.num_steps_per_env):
            action = self.algorithm.act(self._tensor(self.observation))
            # Keep the sampled action for PPO's log probability. The environment
            # clips only the executed action, as part of its control dynamics.
            next_obs, reward, terminated, truncated, info = self.env.step(action[0].cpu().numpy())
            self.algorithm.process_env_step(
                torch.tensor([reward], device=self.device),
                torch.tensor([terminated], dtype=torch.bool, device=self.device),
                torch.tensor([truncated], dtype=torch.bool, device=self.device),
                self._tensor(next_obs))
            self.environment_steps += 1
            rewards.append(reward)
            tracking.add(info)
            terminations += int(terminated)
            timeouts += int(truncated and not terminated)
            if terminated or truncated:
                episode_returns.append(info["episode"]["r"])
                next_obs, _ = self.env.reset()
            self.observation = next_obs
        self.algorithm.compute_returns()
        return {**tracking.result(), "terminations": terminations, "timeouts": timeouts,
                "reward_mean": float(np.mean(rewards)), "episodes": len(episode_returns),
                "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else None}

    def learn_iteration(self):
        metrics = self.collect_rollout()
        metrics.update(self.algorithm.update())
        self.iteration += 1
        metrics.update(iteration=self.iteration, environment_steps=self.environment_steps)
        return metrics

    def save(self, path):
        # A checkpoint represents a completed update, never a partial rollout.
        if self.algorithm.storage.step or self.algorithm.transition.observations is not None:
            raise RuntimeError("Save checkpoints only between completed learning iterations")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "format_version": 1,
            "environment_id": "ArmDisturbanceLocomotion-v0",
            "observation_layout": OBSERVATION_LAYOUT,
            "num_obs": self.model.num_obs,
            "num_actions": self.model.num_actions,
            "network_config": asdict(self.model.config),
            "ppo_config": asdict(self.config),
            "task_config": asdict(self.env.unwrapped.cfg),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
            "iteration": self.iteration,
            "environment_steps": self.environment_steps,
            "seed": self.seed,
        }
        # Replace atomically so interruption cannot corrupt the last good policy.
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(data, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def from_checkpoint(cls, env, path, device="cpu"):
        """Resume weights, Adam, adaptive LR and counters, starting a fresh episode.

        PhysX state and episode RNG are not serialized: this is a training restart,
        not a bit-for-bit continuation of the previous simulation trajectory.
        """
        data = read_checkpoint(path, device)
        if asdict(env.unwrapped.cfg) != data["task_config"]:
            raise ValueError("Resume environment must use the checkpoint task_config")
        runner = cls(env, PPOConfig(**data["ppo_config"]), NetworkConfig(**data["network_config"]),
                     device=device, seed=data["seed"])
        runner.model.load_state_dict(data["model_state_dict"], strict=True)
        runner.algorithm.optimizer.load_state_dict(data["optimizer_state_dict"])
        runner.algorithm.learning_rate = runner.algorithm.optimizer.param_groups[0]["lr"]
        runner.iteration = data["iteration"]
        runner.environment_steps = data["environment_steps"]
        # Reproducible new episode; do not replay the original seed's first episode.
        restart_seed = runner.seed + runner.iteration
        torch.manual_seed(restart_seed)
        runner.observation, _ = env.reset(seed=restart_seed)
        return runner


def read_checkpoint(path, device="cpu"):
    """Read this submission's checkpoint format (also accepts the NN submission)."""
    data = torch.load(path, map_location=device, weights_only=True)
    if (data.get("format_version") != 1
            or data.get("environment_id") != "ArmDisturbanceLocomotion-v0"
            or data.get("observation_layout") != OBSERVATION_LAYOUT
            or data.get("num_obs") != OBSERVATION_SIZE or data.get("num_actions") != 12):
        raise ValueError("Checkpoint does not match this environment's observation/action interface")
    return data


def load_policy(path, device="cpu"):
    """Return (model, task_config); validate observation semantics before playback."""
    data = read_checkpoint(path, device)
    model = ActorCritic(data["num_obs"], data["num_actions"], NetworkConfig(**data["network_config"])).to(device)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.eval()
    return model, TaskConfig(**data["task_config"])
