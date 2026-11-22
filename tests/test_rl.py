"""Numerical checks for PPO, episode boundaries and minibatch coverage."""

import math
import unittest
import torch
import numpy as np

from stage1_rl import ActorCritic, NetworkConfig, PPO, PPOConfig
from stage1_rl.rollout_storage import RolloutStorage
from stage1_rl.evaluation import TrackingMetrics


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_gaussian_log_probability_and_deterministic_action(self):
        model = ActorCritic(5, 3, NetworkConfig((16,), (16,)))
        obs = torch.randn(4, 5)
        mean, value = model(obs)
        model.update_distribution(obs)
        torch.testing.assert_close(model.act_inference(obs), mean)
        expected = -3 * (0.5 * math.log(2 * math.pi) + math.log(model.config.init_noise_std))
        torch.testing.assert_close(model.get_actions_log_prob(mean), torch.full((4,), expected))
        self.assertEqual(value.shape, (4, 1))
        torch.testing.assert_close(model.std, torch.ones(3))

    def test_tracking_metrics_use_completed_command_and_physical_units(self):
        metrics = TrackingMetrics(.3)
        for velocity in ([2., 4., 6.], [0., 2., 2.]):
            metrics.add({"base_velocity": np.array(velocity), "reward_command": np.array([1., 2., 3.]),
                         "command": np.array([99., 99., 99.]), "base_rpy": np.array([.1, -.2, 0.]),
                         "base_height": .4})
        result = metrics.result()
        self.assertAlmostEqual(result["vx_mae_mps"], 1.)
        self.assertAlmostEqual(result["vy_mae_mps"], 1.)
        self.assertAlmostEqual(result["lin_vel_rmse_mps"], math.sqrt(3.))
        self.assertAlmostEqual(result["yaw_rate_rmse_rad_s"], math.sqrt(5.))
        self.assertAlmostEqual(result["base_height_rmse_m"], .1)
        self.assertAlmostEqual(result["roll_rms_rad"], .1)
        self.assertAlmostEqual(result["pitch_rms_rad"], .2)

    def test_gae_terminal_timeout_and_episode_boundaries(self):
        storage = RolloutStorage(4, 2, 1, 1)
        # Four environments: ongoing, fall, timeout, fall at the time limit.
        storage.rewards[0].fill_(1)
        storage.values[0].fill_(2)
        storage.next_values[0].fill_(3)
        storage.terminated[0, 1] = True
        storage.truncated[0, 2] = True
        storage.terminated[0, 3] = True
        storage.truncated[0, 3] = True
        # A high reward in the next episode must not leak across a boundary.
        storage.rewards[1].fill_(100)
        storage.values[1].fill_(3)
        storage.next_values[1].fill_(4)
        storage.step = 2
        storage.compute_returns(gamma=.9, lam=.8)
        torch.testing.assert_close(storage.returns[0, :, 0], torch.tensor([76.132, 1., 3.7, 1.]))
        torch.testing.assert_close(storage.returns[1, :, 0], torch.full((4,), 103.6))
        self.assertTrue(torch.isfinite(storage.advantages).all())

    def test_timeout_uses_final_observation_value(self):
        model = ActorCritic(2, 1, NetworkConfig((4,), (4,)))
        with torch.no_grad():
            for p in model.critic_body.parameters():
                p.zero_()
            model.critic_body[-1].bias.fill_(3)
        ppo = PPO(model, PPOConfig(num_steps_per_env=1, num_mini_batches=1, gamma=.9))
        ppo.init_storage()
        ppo.act(torch.zeros(1, 2))
        # Change the critic so V(next_obs) differs from the stored V(obs).
        with torch.no_grad():
            model.critic_body[-1].bias.fill_(7)
        ppo.process_env_step(torch.tensor([1.]), torch.tensor([False]),
                             torch.tensor([True]), torch.ones(1, 2))
        ppo.compute_returns()
        self.assertAlmostEqual(ppo.storage.returns.item(), 7.3, places=5)

    def test_minibatches_cover_remainder_and_singleton(self):
        storage = RolloutStorage(1, 5, 1, 1)
        storage.observations[:, 0, 0] = torch.arange(5)
        storage.step = 5
        storage.compute_returns(.99, .95)
        batches = list(storage.mini_batch_generator(2, 2))
        for start in (0, 2):
            ids = torch.cat([batch[0][:, 0] for batch in batches[start:start + 2]])
            torch.testing.assert_close(ids.sort().values, torch.arange(5, dtype=torch.float32))
        singleton = RolloutStorage(1, 1, 1, 1)
        singleton.step = 1
        singleton.compute_returns(.99, .95)
        self.assertEqual(singleton.advantages.item(), 0.0)

    def test_ppo_updates_actor_critic_noise_and_clips_gradients(self):
        model = ActorCritic(5, 3, NetworkConfig((16, 8), (16, 8)))
        config = PPOConfig(num_steps_per_env=5, num_learning_epochs=2, num_mini_batches=3,
                           max_grad_norm=.05, desired_kl=None)
        ppo = PPO(model, config)
        ppo.init_storage(2)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        for _ in range(5):
            obs = torch.randn(2, 5)
            actions = ppo.act(obs)
            # Mutating the action delivered to a caller must not alter PPO data.
            sampled = ppo.transition.actions.clone()
            actions.zero_()
            torch.testing.assert_close(ppo.transition.actions, sampled)
            ppo.process_env_step(torch.tensor([2., -1.]), torch.zeros(2, dtype=torch.bool),
                                 torch.zeros(2, dtype=torch.bool), torch.randn(2, 5))
        ppo.compute_returns()
        metrics = ppo.update()
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        for group in ("actor_body", "critic_body", "log_std"):
            self.assertTrue(any(not torch.equal(p, before[name]) for name, p in model.named_parameters()
                                if name.startswith(group)))
        norm = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in model.parameters()])).item()
        self.assertLessEqual(norm, config.max_grad_norm + 1e-6)
        self.assertEqual(ppo.storage.step, 0)


if __name__ == "__main__":
    unittest.main()
