"""Reward-independent tracking measurements and bounded deterministic evaluation.

Adapted conceptually from WBCTrackingShare's locomotion performance metrics.
All errors use unscaled physical state and the command of the completed step.
"""

import numpy as np
import torch


class TrackingMetrics:
    def __init__(self, target_height):
        self.target_height = target_height
        self.steps = 0
        self.velocity_abs = np.zeros(3, dtype=np.float64)
        self.velocity_sq = np.zeros(3, dtype=np.float64)
        self.tilt_sq = np.zeros(2, dtype=np.float64)
        self.height_sq = 0.0

    def add(self, info):
        # `command` can already contain the next command after resampling.
        error = np.asarray(info["base_velocity"], dtype=np.float64) - info["reward_command"]
        tilt = np.asarray(info["base_rpy"][:2], dtype=np.float64)
        height_error = info["base_height"] - self.target_height
        if not (np.isfinite(error).all() and np.isfinite(tilt).all() and np.isfinite(height_error)):
            raise FloatingPointError("Non-finite physical tracking measurement")
        self.steps += 1
        self.velocity_abs += np.abs(error)
        self.velocity_sq += error ** 2
        self.tilt_sq += tilt ** 2
        self.height_sq += height_error ** 2

    def result(self):
        if not self.steps:
            raise ValueError("Tracking metrics require at least one step")
        mae = self.velocity_abs / self.steps
        return {
            "vx_mae_mps": float(mae[0]), "vy_mae_mps": float(mae[1]),
            "yaw_rate_mae_rad_s": float(mae[2]),
            "lin_vel_rmse_mps": float(np.sqrt(self.velocity_sq[:2].sum() / self.steps)),
            "yaw_rate_rmse_rad_s": float(np.sqrt(self.velocity_sq[2] / self.steps)),
            "roll_rms_rad": float(np.sqrt(self.tilt_sq[0] / self.steps)),
            "pitch_rms_rad": float(np.sqrt(self.tilt_sq[1] / self.steps)),
            "base_height_rmse_m": float(np.sqrt(self.height_sq / self.steps)),
        }


@torch.no_grad()
def evaluate_policy(env, model, episodes=5, seed=1000, max_steps=None, reset_options=None):
    """Evaluate policy means; a caller-imposed cap is reported separately from timeout.

    Use a separate environment from a live training runner. Aggregate physical
    errors are step-weighted; episode return/duration are episode-weighted.
    """
    if episodes < 1 or seed < 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("episodes/max_steps must be positive and seed nonnegative")
    limit = env.unwrapped.cfg.max_steps if max_steps is None else max_steps
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    tracking = TrackingMetrics(env.unwrapped.cfg.target_height)
    records = []
    try:
        for index in range(episodes):
            obs, _ = env.reset(seed=seed + index, options=reset_options)
            episode_tracking = TrackingMetrics(env.unwrapped.cfg.target_height)
            total_reward = 0.0
            for step in range(1, limit + 1):
                action = model.act_inference(torch.as_tensor(obs, dtype=torch.float32,
                                                            device=device).unsqueeze(0))
                obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                if not np.isfinite(reward):
                    raise FloatingPointError("Non-finite evaluation reward")
                total_reward += reward
                tracking.add(info)
                episode_tracking.add(info)
                if terminated or truncated:
                    break
            records.append({"episode": index + 1, "seed": seed + index, "return": total_reward,
                            "steps": step, "duration_s": step * env.unwrapped.cfg.dt,
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "capped": not (terminated or truncated),
                            "termination_reasons": list(info["termination_reasons"]),
                            **episode_tracking.result()})
    finally:
        model.train(was_training)
    return {"summary": {**tracking.result(), "episodes": episodes, "steps": tracking.steps,
                        "episode_return_mean": float(np.mean([r["return"] for r in records])),
                        "episode_duration_mean_s": float(np.mean([r["duration_s"] for r in records])),
                        "termination_rate": sum(r["terminated"] for r in records) / episodes,
                        "timeout_rate": sum(r["truncated"] and not r["terminated"] for r in records) / episodes,
                        "capped_rate": sum(r["capped"] for r in records) / episodes},
            "episodes": records}
