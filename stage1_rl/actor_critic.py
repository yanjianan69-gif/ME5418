"""Gaussian locomotion actor and value MLP. See NOTICE.md for licenses."""

import math
import torch
from torch import nn
from torch.distributions import Normal

from .config import NetworkConfig


class ActorCritic(nn.Module):
    """Input [batch, num_obs]; action mean [batch, 12], value [batch, 1]."""

    def __init__(self, num_obs, num_actions=12, config=None):
        super().__init__()
        if num_obs < 1 or num_actions < 1:
            raise ValueError("Observation and action dimensions must be positive")
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.config = config or NetworkConfig()
        activation = nn.ELU()
        actor_input_dim = num_obs
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, self.config.actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(self.config.actor_hidden_dims)):
            if l == len(self.config.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(self.config.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(self.config.actor_hidden_dims[l], self.config.actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(
            nn.Linear(num_obs, self.config.critic_hidden_dims[0])
        )
        critic_layers.append(activation)
        for l in range(len(self.config.critic_hidden_dims)):
            if l == len(self.config.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(self.config.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(self.config.critic_hidden_dims[l], self.config.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        # Log standard deviation keeps the Gaussian scale positive.
        self.log_std = nn.Parameter(torch.full((num_actions,), math.log(self.config.init_noise_std)))
        self.distribution = None

    @property
    def std(self):
        return self.log_std.exp()

    def forward(self, observations):
        if observations.ndim != 2 or observations.shape[-1] != self.num_obs:
            raise ValueError(f"Expected observations [batch, {self.num_obs}]")
        return self.actor_body(observations), self.critic_body(observations)

    def update_distribution(self, observations):
        mean = self.actor_body(observations)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.actor_body(observations)

    def evaluate(self, observations):
        return self.critic_body(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
