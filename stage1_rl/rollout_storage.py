"""On-policy rollout storage and generalized advantage estimation (GAE)."""

import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.terminated = None
            self.truncated = None
            self.values = None
            self.next_values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, num_obs, num_actions, device="cpu"):
        if min(num_envs, num_transitions_per_env, num_obs, num_actions) < 1:
            raise ValueError("Rollout dimensions must be positive")
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        shape = (num_transitions_per_env, num_envs)
        self.observations = torch.zeros(*shape, num_obs, device=device)
        self.actions = torch.zeros(*shape, num_actions, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.terminated = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.truncated = torch.zeros_like(self.terminated)
        self.values = torch.zeros_like(self.rewards)
        self.next_values = torch.zeros_like(self.rewards)
        self.returns = torch.zeros_like(self.rewards)
        self.advantages = torch.zeros_like(self.rewards)
        self.actions_log_prob = torch.zeros_like(self.rewards)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)
        self.step = 0
        self.ready = False

    @torch.no_grad()
    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Rollout buffer overflow; update before collecting more transitions")
        names = {"mu": "action_mean", "sigma": "action_sigma"}
        for name in ("observations", "actions", "rewards", "terminated", "truncated",
                     "values", "next_values", "actions_log_prob", "mu", "sigma"):
            target = getattr(self, name)[self.step]
            value = getattr(transition, names.get(name, name))
            if value is None or value.numel() != target.numel():
                raise ValueError(f"Invalid transition field: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite transition field: {name}")
            target.copy_(value.reshape_as(target))
        self.step += 1
        self.ready = False

    def clear(self):
        self.step = 0
        self.ready = False

    @torch.no_grad()
    def compute_returns(self, gamma, lam):
        if self.step != self.num_transitions_per_env:
            raise RuntimeError("Collect a complete rollout before computing returns")
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        for step in reversed(range(self.num_transitions_per_env)):
            # A fall has no future value. A time limit uses the FINAL observation's
            # value, but neither boundary may propagate GAE into the next episode.
            bootstrap = (~self.terminated[step]).float()
            continue_episode = (~(self.terminated[step] | self.truncated[step])).float()
            delta = self.rewards[step] + gamma * bootstrap * self.next_values[step] - self.values[step]
            advantage = delta + gamma * lam * continue_episode * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std(unbiased=False) + 1e-8)
        self.ready = True

    def mini_batch_generator(self, num_mini_batches, num_epochs=5):
        if not self.ready:
            raise RuntimeError("Compute returns before generating minibatches")
        batch_size = self.num_envs * self.step
        if not 1 <= num_mini_batches <= batch_size:
            raise ValueError("num_mini_batches must be between 1 and rollout size")
        fields = [field.flatten(0, 1) for field in (
            self.observations, self.actions, self.values, self.advantages,
            self.returns, self.actions_log_prob, self.mu, self.sigma)]
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            # tensor_split includes the remainder when batch sizes are unequal.
            for batch_idx in torch.tensor_split(indices, num_mini_batches):
                yield tuple(field[batch_idx] for field in fields)
