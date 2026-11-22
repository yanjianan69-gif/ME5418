"""Integration tests use the actual Isaac Gym simulator, never a mock backend."""

import unittest
import os
from pathlib import Path
import tempfile
from dataclasses import replace
from isaacgym import gymapi
import gym
import numpy as np
import stage1_gym
import torch

from stage1_rl import NetworkConfig, PPOConfig
from stage1_rl.runner import Runner, load_policy
from stage1_rl.evaluation import evaluate_policy


class PhysicsTests(unittest.TestCase):
    def make_env(self, **kwargs):
        config = replace(stage1_gym.TaskConfig(), **kwargs)
        env = gym.make("ArmDisturbanceLocomotion-v0", config=config,
                       sim_device=os.environ.get("STAGE1_SIM_DEVICE", "cuda:0"))
        self.addCleanup(env.close)
        return env

    def test_reset_seed_and_timeout_boundary(self):
        env = self.make_env(episode_seconds=.06)
        first, _ = env.reset(seed=8, options={"command": [0, 0, 0]})
        self.assertEqual(env.unwrapped.steps, 0)
        self.assertTrue(env.observation_space.contains(first))
        obs, _ = env.reset(seed=8, options={"command": [0, 0, 0]})
        np.testing.assert_array_equal(first, obs)
        for i in range(3):
            obs, reward, terminated, truncated, info = env.step(np.zeros(12))
            self.assertFalse(terminated)
            self.assertEqual(truncated, i == 2)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)
        self.assertEqual(env.unwrapped.steps, 3)  # No automatic reset.
        self.assertEqual(info['episode']['l'], 3)
        self.assertAlmostEqual(float(obs[52]), info['base_height'])
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(12))
        obs, _ = env.reset(seed=8, options={"command": [0, 0, 0]})
        np.testing.assert_array_equal(first, obs)
        self.assertEqual(env.unwrapped.disturbance.steps, 0)

    def test_tilt_is_failure_not_timeout(self):
        env = self.make_env(terminal_tilt=.001, reset_tilt=.1, episode_seconds=10)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertIn('excessive_tilt', info['termination_reasons'])
        self.assertTrue(env.observation_space.contains(obs))

    def test_arm_motion_and_invalid_actions(self):
        env = self.make_env(randomize_dynamics=False)
        env.reset(seed=42, options={"command": [0, 0, 0]})
        initial = env.unwrapped.state.arm_pos.copy()
        for invalid in (np.zeros(18), np.full(12, np.nan)):
            with self.assertRaises(ValueError):
                env.step(invalid)
        for _ in range(25):
            obs, reward, terminated, truncated, _ = env.step(np.zeros(12))
            self.assertFalse(terminated or truncated)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
        self.assertGreater(np.max(np.abs(env.unwrapped.state.arm_pos-initial)), .01)
        env.close()
        env.close()
        with self.assertRaises(RuntimeError):
            env.reset()

    def test_ppo_rollout_update_and_checkpoint(self):
        torch.set_num_threads(1)
        env = self.make_env(episode_seconds=.06, randomize_dynamics=False)
        runner = Runner(env, PPOConfig(num_steps_per_env=8, num_learning_epochs=1, num_mini_batches=2),
                        NetworkConfig((32, 16), (32, 16)))
        metrics = runner.learn_iteration()
        self.assertEqual(runner.environment_steps, 8)
        self.assertGreaterEqual(metrics['episodes'], 2)
        self.assertTrue(np.isfinite(metrics['loss']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pt'
            runner.save(path)
            loaded, config = load_policy(path)
            self.assertEqual(config, env.unwrapped.cfg)
            obs = torch.as_tensor(runner.observation).unsqueeze(0)
            with torch.no_grad():
                torch.testing.assert_close(loaded.act_inference(obs), runner.model.act_inference(obs))
            resumed = Runner.from_checkpoint(env, path)
            self.assertEqual(resumed.iteration, runner.iteration)
            self.assertEqual(resumed.environment_steps, runner.environment_steps)
            self.assertEqual(resumed.algorithm.learning_rate, runner.algorithm.learning_rate)
            self.assertEqual(resumed.algorithm.storage.step, 0)
            old_optimizer = runner.algorithm.optimizer.state_dict()
            new_optimizer = resumed.algorithm.optimizer.state_dict()
            self.assertTrue(old_optimizer["state"])
            self.assertEqual(old_optimizer["param_groups"], new_optimizer["param_groups"])
            for key, state in old_optimizer["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, new_optimizer["state"][key][field])
            metrics = resumed.learn_iteration()
            self.assertEqual(resumed.iteration, 2)
            self.assertEqual(resumed.environment_steps, 16)
            self.assertTrue(np.isfinite(metrics["loss"]))
            incompatible = gym.make("ArmDisturbanceLocomotion-v0",
                                    config=replace(env.unwrapped.cfg, disturbance_intensity=.5))
            try:
                with self.assertRaisesRegex(ValueError, "task_config"):
                    Runner.from_checkpoint(incompatible, path)
            finally:
                incompatible.close()
            data = torch.load(path, weights_only=True)
            data['observation_layout'] = ()
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, 'interface'):
                load_policy(path)

    def test_evaluation_distinguishes_cap_and_timeout_without_updating_weights(self):
        torch.set_num_threads(1)
        env = self.make_env(episode_seconds=.06, randomize_dynamics=False)
        runner = Runner(env, network_config=NetworkConfig((16,), (16,)))
        before = {name: value.clone() for name, value in runner.model.state_dict().items()}
        options = {"command": [0, 0, 0], "pose_command": [0, 0]}
        capped = evaluate_policy(env, runner.model, episodes=2, max_steps=1, reset_options=options)
        self.assertEqual(capped["summary"]["capped_rate"], 1.)
        self.assertEqual(capped["summary"]["timeout_rate"], 0.)
        self.assertEqual(capped["summary"]["steps"], 2)
        ended = evaluate_policy(env, runner.model, episodes=2, reset_options=options)
        self.assertEqual(ended["summary"]["capped_rate"], 0.)
        self.assertEqual(ended["summary"]["timeout_rate"], 1.)
        self.assertTrue(runner.model.training)
        for name, value in runner.model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_pose_command_and_arm_velocity_observations(self):
        env = self.make_env(randomize_dynamics=False)
        obs, info = env.reset(seed=4, options={'command': [0, 0, 0], 'pose_command': [.1, -.1]})
        np.testing.assert_allclose(obs[79:81], [.1, -.1])
        np.testing.assert_array_equal(obs[73:79], np.zeros(6))
        obs, _, _, _, info = env.step(np.zeros(12))
        np.testing.assert_allclose(obs[73:79], info['arm_command_velocity'] * .05)
        np.testing.assert_allclose(info['reward_pose_command'], [.1, -.1])
        for invalid in ([1, 0], [0], [float('nan'), 0]):
            with self.assertRaises(ValueError):
                env.reset(options={'pose_command': invalid})


if __name__ == "__main__":
    unittest.main()
