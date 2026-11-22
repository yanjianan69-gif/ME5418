"""Demonstrate network forward/backward passes on synthetic rollout data; no simulator required."""

import argparse
import json
import torch

from stage1_gym.task import OBSERVATION_SIZE
from stage1_rl import ActorCritic, PPO, PPOConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    model = ActorCritic(OBSERVATION_SIZE).to(args.device)
    algorithm = PPO(model, PPOConfig(num_steps_per_env=8, num_learning_epochs=2, num_mini_batches=4))
    algorithm.init_storage(num_envs=4)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    observations = torch.randn(4, OBSERVATION_SIZE, device=args.device) * 0.1
    mean, value = model(observations)
    print(model)
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print(f"input={tuple(observations.shape)} mean={tuple(mean.shape)} value={tuple(value.shape)}")
    print(f"action_mean[0]={mean[0].detach().cpu().tolist()}; value[0]={value[0].item():.6f}")
    for step in range(8):
        actions = algorithm.act(observations)
        next_observations = torch.randn_like(observations) * 0.1
        # Synthetic rewards exercise PPO numerics, not robot performance.
        rewards = 0.1 - 0.01 * actions.square().sum(-1)
        terminated = torch.tensor([step == 3, False, False, False], device=args.device)
        truncated = torch.tensor([False, step == 5, False, False], device=args.device)
        algorithm.process_env_step(rewards, terminated, truncated, next_observations)
        observations = next_observations
    algorithm.compute_returns()
    metrics = algorithm.update()
    changes = {group: max((p.detach() - before[name]).abs().max().item()
                         for name, p in model.named_parameters() if name.startswith(group))
               for group in ("actor_body", "critic_body", "log_std")}
    if not all(torch.isfinite(p).all() and p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters()):
        raise RuntimeError("Network parameters or gradients are invalid")
    if not all(change > 0 for change in changes.values()):
        raise RuntimeError("Expected actor, critic and action-noise parameters to update")
    print("PPO:", json.dumps(metrics, sort_keys=True))
    print("Maximum parameter changes:", json.dumps(changes, sort_keys=True))
    print("PASS: forward pass, GAE, PPO losses, backward pass, gradient clipping and parameter updates")


if __name__ == "__main__":
    main()
