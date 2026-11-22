"""Network and PPO hyperparameters."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NetworkConfig:
    actor_hidden_dims: tuple = (512, 256, 128)
    critic_hidden_dims: tuple = (512, 256, 128)
    init_noise_std: float = 1.0

    def __post_init__(self):
        for widths in (self.actor_hidden_dims, self.critic_hidden_dims):
            if not widths or any(type(w) is not int or w < 1 for w in widths):
                raise ValueError("Hidden layer widths must be positive integers")
        if not math.isfinite(self.init_noise_std) or not 0.01 <= self.init_noise_std <= 5.0:
            raise ValueError("init_noise_std must be between 0.01 and 5")


@dataclass(frozen=True)
class PPOConfig:
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    use_clipped_value_loss: bool = True

    def __post_init__(self):
        for name in ("num_steps_per_env", "num_learning_epochs", "num_mini_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "clip_param", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "lam"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("value_loss_coef", "entropy_coef"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.desired_kl is not None and (not math.isfinite(self.desired_kl) or self.desired_kl <= 0):
            raise ValueError("desired_kl must be positive or None")
