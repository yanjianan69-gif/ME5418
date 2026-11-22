"""Single-agent, single-robot OpenAI Gym 0.26 environment."""

import copy
import gym
from gym import spaces
import numpy as np

from .config import TaskConfig, LEG_HOME
from .task import (OBSERVATION_SIZE, ArmDisturbance, Reward, SmoothRandomCommand, make_observation,
                   quat_from_rpy, rpy_from_quat, sample_command, rotate_inverse)


class Stage1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, config=None, render_mode=None, sim_device="cuda:0"):
        self.cfg = config if config is not None else TaskConfig()
        if not isinstance(self.cfg, TaskConfig):
            raise TypeError("config must be a TaskConfig")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human' or 'rgb_array'")
        self.render_mode = render_mode
        self.sim_device = sim_device
        self.metadata = dict(self.metadata, render_fps=round(1 / self.cfg.dt))
        self.action_space = spaces.Box(-self.cfg.action_limit, self.cfg.action_limit, (12,), np.float32)
        self.observation_space = spaces.Box(-self.cfg.observation_limit, self.cfg.observation_limit,
                                            (OBSERVATION_SIZE,), np.float32)
        self.physics = None
        self._needs_reset = True
        self._closed = False

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("This environment is closed; create a new instance.")
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        unknown = set(options) - {"command", "pose_command"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        fixed_command = options.get("command")
        if fixed_command is not None:
            fixed_command = np.asarray(fixed_command, dtype=np.float32)
            limits = np.asarray(self.cfg.command_ranges)
            if (fixed_command.shape != (3,) or not np.isfinite(fixed_command).all()
                    or np.any(fixed_command < limits[:, 0]) or np.any(fixed_command > limits[:, 1])):
                raise ValueError("options['command'] must be finite [vx, vy, yaw] within command_ranges")
        fixed_pose = options.get("pose_command")
        if fixed_pose is not None:
            fixed_pose = np.asarray(fixed_pose, dtype=np.float32)
            limits = np.asarray(self.cfg.pose_command_ranges)
            if (fixed_pose.shape != (2,) or not np.isfinite(fixed_pose).all()
                    or np.any(fixed_pose < limits[:, 0]) or np.any(fixed_pose > limits[:, 1])):
                raise ValueError("options['pose_command'] must be [pitch, roll] within pose_command_ranges")
        if self.physics is None:
            from .physics import Physics
            # Separate instance DR from episode RNG: reset(seed=s) remains
            # reproducible even though rigid-body properties are set only once.
            instance_seed = seed if seed is not None else int(self.np_random.integers(2**31))
            self.physics = Physics(self.cfg, self.sim_device, self.render_mode, np.random.default_rng(instance_seed))
            self.disturbance = ArmDisturbance(self.cfg, self.physics.lower[self.physics.arms],
                                              self.physics.upper[self.physics.arms])
            self.command_sampler = SmoothRandomCommand(self.cfg)
            self.reward_model = Reward(self.cfg, self.physics.lower[self.physics.legs],
                                        self.physics.upper[self.physics.legs])
        rng, c, p = self.np_random, self.cfg, self.physics
        root = np.zeros(13, dtype=np.float32)
        root[:2] = rng.uniform(-0.1, 0.1, 2)
        root[2] = c.reset_height + rng.uniform(-c.reset_height_noise, c.reset_height_noise)
        root[3:7] = quat_from_rpy(*rng.uniform([-c.reset_tilt, -c.reset_tilt, -np.pi],
                                             [c.reset_tilt, c.reset_tilt, np.pi]))
        root[7:13] = rng.uniform(-c.reset_velocity_noise, c.reset_velocity_noise, 6)
        positions = p.home.copy()
        positions[p.legs] += rng.uniform(-c.reset_joint_noise, c.reset_joint_noise, 12)
        positions[p.arms] += rng.uniform(-c.arm_initial_noise, c.arm_initial_noise, 6)
        positions = np.clip(positions, p.lower, p.upper).astype(np.float32)
        self.state = p.reset(root, positions, rng)
        self.disturbance.reset(self.state.arm_pos)
        self.disturbance.sample_motion(rng)
        self.reward_model.reset(self.state)
        self.steps = 0
        self.phase = 0.0
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.fixed_command = fixed_command is not None
        self.command_sampler.reset(fixed_command)
        self.command = fixed_command.copy() if self.fixed_command else self.command_sampler.step(rng, c.dt)
        self.fixed_pose_command = fixed_pose is not None
        pose_limits = np.asarray(c.pose_command_ranges)
        self.pose_command = (fixed_pose.copy() if self.fixed_pose_command else
                             rng.uniform(pose_limits[:, 0], pose_limits[:, 1]).astype(np.float32))
        self.episode_return = 0.0
        self._needs_reset = False
        obs = self._observation()
        info = self._info()
        info["instance_randomization"] = copy.deepcopy(p.instance_randomization)
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        if self._closed or self._needs_reset:
            raise RuntimeError("Call reset() before step(), including after an episode ends.")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("action must contain exactly 12 finite values")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled = action * self.cfg.action_scale
        scaled[[0, 3, 6, 9]] *= self.cfg.hip_scale
        target = np.asarray(LEG_HOME, dtype=np.float32) + scaled
        arm_target = self.disturbance.step(self.np_random)
        self.state, torques = self.physics.step(target, arm_target)
        self.steps += 1
        self.phase = (self.phase + self.cfg.dt * self.cfg.gait_frequency) % 1.0
        if not self.state.finite():
            self._needs_reset = True
            raise FloatingPointError("Non-finite Isaac Gym state; reset before continuing.")
        reward_command = self.command.copy()
        reward_pose_command = self.pose_command.copy()
        reward, terms = self.reward_model.compute(self.state, action, reward_command, self.phase, target, torques,
                                                  reward_pose_command)
        roll, pitch, _ = rpy_from_quat(self.state.root[3:7])
        reasons = []
        if self.state.root[2] < self.cfg.terminal_height:
            reasons.append("low_height")
        if max(abs(roll), abs(pitch)) > self.cfg.terminal_tilt:
            reasons.append("excessive_tilt")
        terminated = bool(reasons)
        truncated = bool(self.steps >= self.cfg.max_steps)
        self._needs_reset = terminated or truncated
        self.episode_return += reward
        self.previous_action = action.copy()
        # Reward belongs to the command that governed the completed transition.
        # The returned observation carries the command for the NEXT action.
        if not self.fixed_command and not self._needs_reset:
            self.command = self.command_sampler.step(self.np_random, self.cfg.dt)
        obs = self._observation()
        info = self._info()
        info.update(reward_terms=terms, reward_command=reward_command, reward_pose_command=reward_pose_command,
                    termination_reasons=tuple(reasons), arm_target=arm_target)
        if self._needs_reset:
            info["episode"] = {"r": self.episode_return, "l": self.steps, "t": self.steps * self.cfg.dt}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _observation(self):
        return make_observation(self.state, self.previous_action, self.command, self.phase, self.cfg,
                                self.disturbance.command_velocity, self.pose_command)

    def _info(self):
        q = self.state.root[3:7]
        lin = rotate_inverse(q, self.state.root[7:10])
        ang = rotate_inverse(q, self.state.root[10:13])
        return {
            "command": self.command.copy(), "elapsed_seconds": self.steps * self.cfg.dt,
            "pose_command": self.pose_command.copy(),
            "arm_command_velocity": self.disturbance.command_velocity.copy(),
            "arm_targets_accepted": self.disturbance.accepted,
            "disturbance_intensity": self.cfg.disturbance_intensity,
            "base_height": float(self.state.root[2]),
            "base_rpy": rpy_from_quat(q).astype(np.float32),
            "base_velocity": np.array([lin[0], lin[1], ang[2]], dtype=np.float32),
        }

    def render(self):
        if self.physics is not None and not self._closed:
            return self.physics.render()
        return None

    def close(self):
        if self.physics is not None:
            self.physics.close()
        self._closed = True
