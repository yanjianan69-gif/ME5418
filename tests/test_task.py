import unittest
from dataclasses import replace
import numpy as np

from stage1_gym.config import TaskConfig, LEG_HOME
from stage1_gym.task import (ArmDisturbance, RobotState, Reward, make_observation,
                             OBSERVATION_SIZE, quat_from_rpy, rotate_inverse, gait_targets)


def standing_state():
    root = np.zeros(13, dtype=np.float32)
    root[2], root[6] = 0.30, 1.0
    feet = np.array([[.225, .15, .02], [.225, -.15, .02],
                     [-.225, .15, .02], [-.225, -.15, .02]])
    return RobotState(root, np.array(LEG_HOME), np.zeros(12), np.zeros(6), np.zeros(6),
                      feet, np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((1, 3)))


class TaskTests(unittest.TestCase):
    def test_disturbance_is_bounded_and_seeded(self):
        cfg = replace(TaskConfig(), arm_max_accel=100, arm_max_velocity=5)
        lower, upper = np.full(6, -.05), np.full(6, .05)
        first, second = ArmDisturbance(cfg, lower, upper), ArmDisturbance(cfg, lower, upper)
        rng1, rng2 = np.random.default_rng(12), np.random.default_rng(12)
        first.sample_motion(rng1)
        second.sample_motion(rng2)
        excursion = 0.0
        for _ in range(2000):
            a, b = first.step(rng1), second.step(rng2)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.all(a >= lower-1e-7) and np.all(a <= upper+1e-7))
            self.assertLessEqual(np.max(np.abs(first.command_velocity)), cfg.arm_target_max_velocity + 1e-6)
            excursion = max(excursion, np.max(np.abs(a)))
        self.assertGreater(excursion, .04)

    def test_zero_intensity_holds_reset_pose(self):
        cfg = replace(TaskConfig(), disturbance_intensity=0)
        d = ArmDisturbance(cfg, np.full(6, -1), np.ones(6))
        pose = np.arange(6, dtype=np.float32) * .1
        d.reset(pose)
        d.sample_motion(np.random.default_rng(2))
        for _ in range(50):
            np.testing.assert_array_equal(d.step(np.random.default_rng(1)), pose)

    def test_observation_slots_and_frames(self):
        state = standing_state()
        state.root[3:7] = quat_from_rpy(0, 0, np.pi/2)
        state.root[8] = 0.5  # world +y is robot +x
        obs = make_observation(state, np.ones(12), np.array([.5, 0, 0]), 0, TaskConfig())
        self.assertEqual(obs.shape, (OBSERVATION_SIZE,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[:3], [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(obs[27:39], 1)
        np.testing.assert_allclose(obs[49:52], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(obs[58:61], 0, atol=1e-6)

    def test_arm_command_velocity_matches_executed_target(self):
        cfg = replace(TaskConfig(), arm_max_accel=100, arm_max_velocity=5)
        disturbance = ArmDisturbance(cfg, np.full(6, -.03), np.full(6, .03))
        rng = np.random.default_rng(3)
        disturbance.sample_motion(rng)
        for _ in range(100):
            previous = disturbance.position.copy()
            target = disturbance.step(rng)
            np.testing.assert_allclose(disturbance.command_velocity, (target - previous) / cfg.dt, atol=1e-6)
        obs = make_observation(standing_state(), np.zeros(12), np.zeros(3), 0, cfg,
                               disturbance.command_velocity, np.array([.1, -.1]))
        np.testing.assert_allclose(obs[73:79], disturbance.command_velocity * .05)
        np.testing.assert_allclose(obs[79:81], [.1, -.1])
        np.testing.assert_allclose(obs[56:58], [.1, -.1])

    def test_orientation_reward_tracks_pose_command(self):
        c = TaskConfig()
        state = standing_state()
        state.root[3:7] = quat_from_rpy(-.1, .15, 0)
        model = Reward(c, np.full(12, -3), np.full(12, 3))
        model.reset(state)
        args = (state, np.zeros(12), np.zeros(3), 0, np.array(LEG_HOME), np.zeros(12))
        _, correct = model.compute(*args, pose_command=np.array([.15, -.1]))
        _, wrong = model.compute(*args, pose_command=np.zeros(2))
        self.assertAlmostEqual(correct['orientation'], 0.0)
        self.assertLess(wrong['orientation'], correct['orientation'])

    def test_reward_favors_tracking_and_stability(self):
        c = TaskConfig()
        def evaluate(state):
            model = Reward(c, np.full(12, -3), np.full(12, 3))
            model.reset(standing_state())
            return model.compute(state, np.zeros(12), np.zeros(3), 0, np.array(LEG_HOME), np.zeros(12))[0]
        good = evaluate(standing_state())
        bad = standing_state()
        bad.root[7] = 1.0
        self.assertLess(evaluate(bad), good)
        bad = standing_state()
        bad.root[3:7] = quat_from_rpy(.4, .3, 0)
        self.assertLess(evaluate(bad), good)

    def test_standing_gait_and_reset_history(self):
        phase, clock, contact = gait_targets(.71, np.zeros(3))
        np.testing.assert_allclose(phase, .25)
        np.testing.assert_allclose(clock, 1)
        self.assertTrue(np.all(contact > .99))
        model = Reward(TaskConfig(), np.full(12, -3), np.full(12, 3))
        state = standing_state()
        model.reset(state)
        args = (state, np.zeros(12), np.zeros(3), 0, np.array(LEG_HOME), np.zeros(12))
        first = model.compute(*args)
        model.compute(state, np.ones(12), np.ones(3), .4, np.array(LEG_HOME)+.2, np.ones(12))
        model.reset(state)
        self.assertEqual(first, model.compute(*args))


if __name__ == "__main__":
    unittest.main()
