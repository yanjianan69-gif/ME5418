"""Clipped PPO with value regression, entropy regularization and Adam updates."""

import torch
from torch import nn, optim

from .config import PPOConfig
from .rollout_storage import RolloutStorage


class PPO:
    def __init__(self, actor_critic, config=None):
        self.actor_critic = actor_critic
        self.config = config or PPOConfig()
        self.device = next(actor_critic.parameters()).device
        self.learning_rate = self.config.learning_rate
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.learning_rate)
        self.storage = None
        self.transition = RolloutStorage.Transition()

    def init_storage(self, num_envs=1):
        if num_envs * self.config.num_steps_per_env < self.config.num_mini_batches:
            raise ValueError("Rollout size must be at least num_mini_batches")
        self.storage = RolloutStorage(num_envs, self.config.num_steps_per_env,
                                      self.actor_critic.num_obs, self.actor_critic.num_actions, self.device)

    @torch.no_grad()
    def act(self, observations):
        if self.storage is None:
            raise RuntimeError("Call init_storage before collecting a rollout")
        if self.transition.observations is not None:
            raise RuntimeError("Record the previous environment step before selecting another action")
        if observations.shape != (self.storage.num_envs, self.actor_critic.num_obs):
            raise ValueError("Observation batch shape does not match the rollout storage")
        self.transition.actions = self.actor_critic.act(observations).detach()
        self.transition.values = self.actor_critic.evaluate(observations).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = observations.detach().clone()
        return self.transition.actions.clone()

    @torch.no_grad()
    def process_env_step(self, rewards, terminated, truncated, next_observations):
        if self.transition.observations is None:
            raise RuntimeError("Select an action before recording its outcome")
        self.transition.rewards = rewards
        self.transition.terminated = terminated
        self.transition.truncated = truncated
        # This must run before resetting an ended episode.
        self.transition.next_values = self.actor_critic.evaluate(next_observations)
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self):
        self.storage.compute_returns(self.config.gamma, self.config.lam)

    def update(self):
        self.actor_critic.train()
        logs = []
        generator = self.storage.mini_batch_generator(self.config.num_mini_batches, self.config.num_learning_epochs)
        for (obs_batch, actions_batch, target_values_batch, advantages_batch,
             returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch) in generator:
            self.actor_critic.update_distribution(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            with torch.no_grad():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch)
                    + (old_sigma_batch.square() + (old_mu_batch - mu_batch).square())
                    / (2 * sigma_batch.square()) - 0.5, dim=-1).mean()
                if self.config.desired_kl is not None:
                    if kl > self.config.desired_kl * 2:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0 < kl < self.config.desired_kl / 2:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch.squeeze(-1))
            surrogate = -advantages_batch.squeeze(-1) * ratio
            surrogate_clipped = -advantages_batch.squeeze(-1) * torch.clamp(
                ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.config.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.config.clip_param, self.config.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.config.value_loss_coef * value_loss - self.config.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()

            with torch.no_grad():
                self.actor_critic.log_std.clamp_(-5.0, 2.0)
            logs.append(torch.stack((loss.detach(), surrogate_loss.detach(), value_loss.detach(),
                                     entropy_batch.mean().detach(), grad_norm.detach(), kl.detach())))
        metrics = dict(zip(("loss", "policy_loss", "value_loss", "entropy", "grad_norm", "kl"),
                           torch.stack(logs).mean(0).cpu().tolist()))
        metrics["learning_rate"] = self.learning_rate
        self.storage.clear()
        return metrics
