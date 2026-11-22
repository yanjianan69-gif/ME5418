"""Vectorized locomotion environment with batched NumPy task logic."""

from gym import spaces
import numpy as np

from .config import TaskConfig, LEG_HOME
from .task import OBSERVATION_SIZE, REWARD_WEIGHTS, quat_from_rpy, sample_command
from .vec_physics import VecPhysics


def rotate_inverse_batch(q, vector):
    xyz = -q[:, :3]
    t = 2 * np.cross(xyz, vector)
    return vector + q[:, 3:4] * t + np.cross(xyz, t)


def rpy_from_quat_batch(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.column_stack([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ]).astype(np.float32)


def gait_targets_batch(phase, command):
    offsets = np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32)
    phases = np.remainder(phase[:, None] + offsets[None, :], 1.0)
    phases[np.linalg.norm(command, axis=1) < 0.1] = 0.25
    clock = np.sin(2 * np.pi * phases)
    erf = np.vectorize(lambda v: 0.5 * (1 + np.math.erf(float(v) / (0.07 * np.sqrt(2)))))
    contact = erf(phases) * (1 - erf(phases - 0.5)) + erf(phases - 1) * (1 - erf(phases - 1.5))
    return phases.astype(np.float32), clock.astype(np.float32), contact.astype(np.float32)


class VecArmMotion:
    def __init__(self, cfg, num_envs, lower, upper, rng):
        self.cfg = cfg
        self.num_envs = num_envs
        self.lower = np.asarray(lower, dtype=np.float32)
        self.upper = np.asarray(upper, dtype=np.float32)
        self.position = np.zeros((num_envs, 6), dtype=np.float32)
        self.position_at_reset = np.zeros_like(self.position)
        self.command_velocity = np.zeros_like(self.position)
        self.start = np.zeros_like(self.position)
        self.goal = np.zeros_like(self.position)
        self.elapsed = np.zeros(num_envs, dtype=np.float32)
        self.duration = np.zeros(num_envs, dtype=np.float32)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.reset(np.arange(num_envs), self.position, rng)

    def reset(self, env_ids, positions, rng):
        ids = np.asarray(env_ids, dtype=np.int64)
        self.position[ids] = np.clip(positions, self.lower, self.upper)
        self.position_at_reset[ids] = self.position[ids]
        self.command_velocity[ids] = 0
        self.start[ids] = self.goal[ids] = self.position[ids]
        self.elapsed[ids] = self.duration[ids] = 0
        self.steps[ids] = 0
        self._sample(ids, rng)

    def _sample(self, ids, rng):
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return
        c = self.cfg
        extension = rng.uniform(1.3, 3.0, len(ids))
        goals = np.column_stack([
            rng.uniform(-1.8, 1.8, len(ids)),
            extension,
            np.clip(extension + rng.uniform(-0.3, 0.3, len(ids)), 0.8, 3.15),
            rng.uniform(-0.45, 0.45, len(ids)),
            rng.uniform(-0.6, 0.6, len(ids)),
            rng.uniform(-0.8, 0.8, len(ids)),
        ]).astype(np.float32)
        goals = np.clip(goals, np.maximum(self.lower, self.position_at_reset[ids] - c.arm_position_limit),
                        np.minimum(self.upper, self.position_at_reset[ids] + c.arm_position_limit))
        self.start[ids] = self.position[ids]
        self.goal[ids] = goals
        speed = max(c.arm_target_max_velocity, 1e-6)
        duration = np.maximum(rng.uniform(5.0, 8.0, len(ids)) * 0.6 / speed,
                              1.875 * np.max(np.abs(goals - self.start[ids]), axis=1) / speed)
        duration = np.where(self.steps[ids] == 0, np.maximum(duration, c.arm_startup_seconds), duration)
        self.duration[ids] = duration.astype(np.float32)
        self.elapsed[ids] = 0

    def step(self, rng):
        c = self.cfg
        if c.disturbance_intensity == 0:
            self.command_velocity.fill(0)
            return self.position.copy()
        due = self.elapsed >= self.duration
        self._sample(np.nonzero(due)[0], rng)
        u = np.clip(self.elapsed / np.maximum(self.duration, c.dt), 0, 1)
        blend = u**3 * (10 - 15*u + 6*u*u)
        raw = self.start + blend[:, None] * (self.goal - self.start)
        delta = c.arm_target_max_velocity * c.dt
        position = (self.position + np.clip(raw - self.position, -delta, delta)).astype(np.float32)
        self.command_velocity[:] = (position - self.position) / c.dt
        self.position[:] = position
        self.elapsed += c.dt
        self.steps += 1
        return self.position.copy()


class Stage1VecEnv:
    def __init__(self, num_envs, config=None, sim_device="cuda:0", seed=42):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.cfg = config if config is not None else TaskConfig()
        if not isinstance(self.cfg, TaskConfig):
            raise TypeError("config must be a TaskConfig")
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-self.cfg.action_limit, self.cfg.action_limit, (12,), np.float32)
        self.observation_space = spaces.Box(-self.cfg.observation_limit, self.cfg.observation_limit,
                                            (OBSERVATION_SIZE,), np.float32)
        self.physics = VecPhysics(self.cfg, sim_device, num_envs, self.rng)
        self.arm = VecArmMotion(self.cfg, num_envs, self.physics.lower[self.physics.arms],
                                self.physics.upper[self.physics.arms], self.rng)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.phase = np.zeros(num_envs, dtype=np.float32)
        self.previous_action = np.zeros((num_envs, 12), dtype=np.float32)
        self.previous_velocity = np.zeros((num_envs, 12), dtype=np.float32)
        self.previous_target = np.repeat(np.asarray(LEG_HOME, dtype=np.float32)[None, :], num_envs, axis=0)
        self.previous_previous_target = self.previous_target.copy()
        self.previous_contact = np.zeros((num_envs, 4), dtype=bool)
        self.velocity_reference = np.zeros((num_envs, 2), dtype=np.float32)
        self.command = np.zeros((num_envs, 3), dtype=np.float32)
        self.command_start = np.zeros((num_envs, 3), dtype=np.float32)
        self.command_goal = np.zeros((num_envs, 3), dtype=np.float32)
        self.command_elapsed = np.zeros(num_envs, dtype=np.float32)
        self.command_duration = np.zeros(num_envs, dtype=np.float32)
        self.pose_command = np.zeros((num_envs, 2), dtype=np.float32)
        self.episode_return = np.zeros(num_envs, dtype=np.float64)
        self.state = None

    @property
    def unwrapped(self):
        return self

    def reset(self, seed=None, env_ids=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        env_ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        n, c, p, rng = len(env_ids), self.cfg, self.physics, self.rng
        roots = np.zeros((n, 13), dtype=np.float32)
        roots[:, :2] = rng.uniform(-0.1, 0.1, (n, 2))
        roots[:, 2] = c.reset_height + rng.uniform(-c.reset_height_noise, c.reset_height_noise, n)
        roots[:, 3:7] = np.array([quat_from_rpy(*values) for values in
                                  rng.uniform([-c.reset_tilt, -c.reset_tilt, -np.pi],
                                              [c.reset_tilt, c.reset_tilt, np.pi], (n, 3))], dtype=np.float32)
        roots[:, 7:13] = rng.uniform(-c.reset_velocity_noise, c.reset_velocity_noise, (n, 6))
        positions = np.repeat(p.home[None, :], n, axis=0)
        positions[:, p.legs] += rng.uniform(-c.reset_joint_noise, c.reset_joint_noise, (n, 12))
        positions[:, p.arms] += rng.uniform(-c.arm_initial_noise, c.arm_initial_noise, (n, 6))
        positions = np.clip(positions, p.lower, p.upper).astype(np.float32)
        p.reset(env_ids, roots, positions, rng)
        self.state = p.state()
        self.arm.reset(env_ids, self.state.arm_pos[env_ids], rng)
        self.steps[env_ids] = 0
        self.phase[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.previous_velocity[env_ids] = self.state.leg_vel[env_ids]
        self.previous_target[env_ids] = LEG_HOME
        self.previous_previous_target[env_ids] = LEG_HOME
        self.previous_contact[env_ids] = False
        self.velocity_reference[env_ids] = 0
        bounds = np.asarray(c.pose_command_ranges)
        self.command[env_ids] = 0
        self.command_start[env_ids] = 0
        self.command_goal[env_ids] = 0
        self.command_elapsed[env_ids] = self.command_duration[env_ids] = 0
        self._advance_commands(env_ids)
        self.pose_command[env_ids] = rng.uniform(bounds[:, 0], bounds[:, 1], (n, 2)).astype(np.float32)
        self.episode_return[env_ids] = 0
        return self._observations(), self._info()

    def _advance_commands(self, env_ids=None):
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
        due = ids[self.command_elapsed[ids] >= self.command_duration[ids]]
        if len(due):
            bounds = np.asarray(self.cfg.command_target_ranges)
            self.command_start[due] = self.command[due]
            self.command_goal[due] = self.rng.uniform(bounds[:, 0], bounds[:, 1], (len(due), 3)).astype(np.float32)
            self.command_elapsed[due] = 0
            self.command_duration[due] = self.rng.uniform(*self.cfg.command_transition_seconds, len(due)).astype(np.float32)
        self.command_elapsed[ids] += self.cfg.dt
        u = np.clip(self.command_elapsed[ids] / self.command_duration[ids], 0, 1)[:, None]
        blend = u**3 * (10 - 15*u + 6*u*u)
        self.command[ids] = self.command_start[ids] + blend * (self.command_goal[ids] - self.command_start[ids])

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_envs, 12) or not np.isfinite(action).all():
            raise ValueError(f"action must have shape ({self.num_envs}, 12) and finite values")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled = action * self.cfg.action_scale
        scaled[:, [0, 3, 6, 9]] *= self.cfg.hip_scale
        target = np.asarray(LEG_HOME, dtype=np.float32)[None, :] + scaled
        arm_target = self.arm.step(self.rng)
        self.state, torques = self.physics.step(target, arm_target)
        self.steps += 1
        self.phase = (self.phase + self.cfg.dt * self.cfg.gait_frequency) % 1.0
        rewards, terminated, reasons = self._rewards(action, target, torques)
        truncated = self.steps >= self.cfg.max_steps
        done = terminated | truncated
        self.episode_return += rewards
        episode_returns = np.full(self.num_envs, np.nan, dtype=np.float64)
        episode_returns[done] = self.episode_return[done]
        self.previous_action = action.copy()
        if np.any(~done):
            self._advance_commands(np.nonzero(~done)[0])
        obs = self._observations()
        if np.any(done):
            self.reset(env_ids=np.nonzero(done)[0])
            obs = self._observations()
        info = self._info()
        info.update(episode_returns=episode_returns, termination_reasons=reasons, arm_target=arm_target)
        return obs, rewards.astype(np.float32), terminated, truncated, info

    def _observations(self):
        q = self.state.root[:, 3:7]
        gravity = rotate_inverse_batch(q, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        lin = rotate_inverse_batch(q, self.state.root[:, 7:10])
        ang = rotate_inverse_batch(q, self.state.root[:, 10:13])
        rpy = rpy_from_quat_batch(q)
        pose = np.column_stack([self.state.root[:, 2], rpy[:, 1], rpy[:, 0]])
        pose_error = np.column_stack([np.full(self.num_envs, self.cfg.target_height), self.pose_command]) - pose
        velocity_error = (self.command - np.column_stack([lin[:, 0], lin[:, 1], ang[:, 2]])) * [2.0, 2.0, 0.25]
        _, clock, _ = gait_targets_batch(self.phase, self.command)
        values = (gravity, self.state.leg_pos - LEG_HOME, self.state.leg_vel * 0.05, self.previous_action,
                  self.command * [2.0, 2.0, 0.25], clock, ang * 0.25, lin * 2.0, pose, pose_error,
                  velocity_error, self.state.arm_pos, self.state.arm_vel * 0.05, self.arm.command_velocity * 0.05,
                  self.pose_command)
        obs = np.concatenate(values, axis=1).astype(np.float32)
        assert obs.shape == (self.num_envs, OBSERVATION_SIZE)
        return np.clip(obs, -self.cfg.observation_limit, self.cfg.observation_limit)

    def _rewards(self, action, target, torques):
        c = self.cfg
        q = self.state.root[:, 3:7]
        lin = rotate_inverse_batch(q, self.state.root[:, 7:10])
        ang = rotate_inverse_batch(q, self.state.root[:, 10:13])
        gravity = rotate_inverse_batch(q, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        desired_q = np.array([quat_from_rpy(roll, pitch, 0) for pitch, roll in self.pose_command], dtype=np.float32)
        desired_gravity = rotate_inverse_batch(desired_q, np.tile([0.0, 0.0, -1.0], (self.num_envs, 1)))
        self.velocity_reference += (1 - np.exp(-c.dt / 0.4)) * (self.command[:, :2] - self.velocity_reference)
        phases, _, desired_contact = gait_targets_batch(self.phase, self.command)
        contact = self.state.foot_force[:, :, 2] > 1.0
        slip_contact = contact | self.previous_contact
        swing_phase = 1 - np.abs(1 - np.clip(2 * phases - 1, 0, 1) * 2)
        foot_target_z = c.swing_height * swing_phase + 0.02
        yaw = rpy_from_quat_batch(q)[:, 2]
        yaw_q = np.array([quat_from_rpy(0, 0, value) for value in yaw], dtype=np.float32)
        feet_body = rotate_inverse_batch(np.repeat(yaw_q, 4, axis=0),
                                         (self.state.foot_pos - self.state.root[:, None, :3]).reshape(-1, 3)
                                         ).reshape(self.num_envs, 4, 3)
        phase_offset = np.abs(1 - 2 * phases) - 0.5
        target_x = np.array([1, 1, -1, -1]) * c.stance_length / 2 + phase_offset * self.command[:, :1] * 0.5 / c.gait_frequency
        target_y = (np.array([1, -1, 1, -1]) * c.stance_width / 2
                    + phase_offset * self.command[:, 2:3] * c.stance_length / 2 * 0.5 / c.gait_frequency
                    * np.array([1, 1, -1, -1]))
        terms = {
            "tracking_lin_vel": np.exp(-np.sum((self.command[:, :2] - lin[:, :2])**2, axis=1) / 0.25),
            "tracking_ang_vel": np.exp(-(self.command[:, 2] - ang[:, 2])**2 / 0.25),
            "response_consistency": np.sum((lin[:, :2] - self.velocity_reference)**2, axis=1),
            "lin_vel_z": lin[:, 2]**2,
            "ang_vel_xy": np.sum(ang[:, :2]**2, axis=1),
            "orientation": np.sum((gravity[:, :2] - desired_gravity[:, :2])**2, axis=1),
            "base_height": (self.state.root[:, 2] - c.target_height)**2,
            "torques": np.sum(torques**2, axis=1),
            "dof_vel": np.sum(self.state.leg_vel**2, axis=1),
            "dof_acc": np.sum(((self.state.leg_vel - self.previous_velocity) / c.dt)**2, axis=1),
            "collision": np.count_nonzero(np.linalg.norm(self.state.collision_forces, axis=-1) > 0.1, axis=1),
            "action_rate": np.sum((action - self.previous_action)**2, axis=1),
            "action_smoothness_1": np.sum((target - self.previous_target)**2, axis=1),
            "action_smoothness_2": np.sum((target - 2*self.previous_target + self.previous_previous_target)**2, axis=1),
            "dof_pos_limits": np.sum(np.maximum(self.physics.lower[self.physics.legs] - self.state.leg_pos, 0)
                                     + np.maximum(self.state.leg_pos - self.physics.upper[self.physics.legs], 0), axis=1),
            "feet_slip": np.sum(slip_contact * np.sum(self.state.foot_vel[:, :, :2]**2, axis=2), axis=1),
            "feet_clearance": np.sum((foot_target_z - self.state.foot_pos[:, :, 2])**2 * (1 - desired_contact), axis=1),
            "tracking_contacts_force": -np.mean((1-desired_contact) * (1 - np.exp(-np.sum(self.state.foot_force**2, axis=2) / 100.0)), axis=1),
            "tracking_contacts_velocity": -np.mean(desired_contact * (1 - np.exp(-np.sum(self.state.foot_vel**2, axis=2) / 10.0)), axis=1),
            "raibert_heuristic": np.sum((feet_body[:, :, :2] - np.stack([target_x, target_y], axis=2))**2, axis=(1, 2)),
            "loco_energy": np.sum((torques * self.state.leg_vel)**2, axis=1),
        }
        weighted = np.zeros(self.num_envs, dtype=np.float32)
        positive = np.zeros(self.num_envs, dtype=np.float32)
        negative = np.zeros(self.num_envs, dtype=np.float32)
        for name, value in terms.items():
            term = value * REWARD_WEIGHTS[name] * c.dt
            positive += np.maximum(term, 0)
            negative += np.minimum(term, 0)
        weighted = positive * np.exp(negative / 0.02)
        self.previous_velocity = self.state.leg_vel.copy()
        self.previous_previous_target = self.previous_target.copy()
        self.previous_target = target.copy()
        self.previous_contact = contact.copy()
        rpy = rpy_from_quat_batch(q)
        terminated = (self.state.root[:, 2] < c.terminal_height) | (np.max(np.abs(rpy[:, :2]), axis=1) > c.terminal_tilt)
        reasons = [[] for _ in range(self.num_envs)]
        return weighted.astype(np.float32), terminated, reasons

    def _info(self):
        q = self.state.root[:, 3:7]
        lin = rotate_inverse_batch(q, self.state.root[:, 7:10])
        ang = rotate_inverse_batch(q, self.state.root[:, 10:13])
        return {
            "command": self.command.copy(),
            "pose_command": self.pose_command.copy(),
            "arm_command_velocity": self.arm.command_velocity.copy(),
            "arm_targets_active": self.arm.goal.copy(),
            "base_height": self.state.root[:, 2].astype(np.float32),
            "base_rpy": rpy_from_quat_batch(q),
            "base_velocity": np.column_stack([lin[:, 0], lin[:, 1], ang[:, 2]]).astype(np.float32),
        }

    def close(self):
        self.physics.close()
