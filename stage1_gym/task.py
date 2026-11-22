"""Observation, disturbance and reward logic, independent of the simulator.

First-stage equations adapted from RoboDuet; all arrays describe one robot.
Quaternion convention: xyzw. Leg/foot order: FL, FR, RL, RR.
"""

from dataclasses import dataclass
import math
import numpy as np

from .config import LEG_HOME


OBSERVATION_LAYOUT = (
    ("projected_gravity", 3), ("leg_position", 12), ("leg_velocity", 12),
    ("previous_action", 12), ("velocity_command", 3), ("gait_clock", 4),
    ("base_angular_velocity", 3), ("base_linear_velocity", 3),
    ("height_pitch_roll", 3), ("pose_error", 3), ("velocity_error", 3),
    ("arm_position", 6), ("arm_velocity", 6),
    ("arm_command_velocity", 6), ("posture_command", 2),
)
OBSERVATION_SIZE = sum(width for _, width in OBSERVATION_LAYOUT)


def quat_from_rpy(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw]) / 2)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy], dtype=np.float32)


def rotate_inverse(q, vector):
    xyz = -np.asarray(q[:3])
    t = 2 * np.cross(xyz, vector)
    return vector + q[3] * t + np.cross(xyz, t)


def rpy_from_quat(q):
    x, y, z, w = q
    return np.array([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ])


@dataclass
class RobotState:
    root: np.ndarray            # xyz, xyzw, world linear velocity, world angular velocity
    leg_pos: np.ndarray
    leg_vel: np.ndarray
    arm_pos: np.ndarray
    arm_vel: np.ndarray
    foot_pos: np.ndarray        # world frame, (4, 3)
    foot_vel: np.ndarray        # world frame, (4, 3)
    foot_force: np.ndarray      # net world contact force, (4, 3)
    collision_forces: np.ndarray

    def finite(self):
        return all(np.isfinite(value).all() for value in vars(self).values())


class ArmDisturbance:
    """Smooth random reaching targets, matching the final group task style."""
    def __init__(self, cfg, lower, upper):
        self.cfg = cfg
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.reset(np.clip(np.zeros(6), self.lower, self.upper))

    def reset(self, position):
        self.position = np.clip(position, self.lower, self.upper).astype(np.float32)
        self.position_at_reset = self.position.copy()
        self.command_velocity = np.zeros(6, dtype=np.float32)
        self.start = self.position.copy()
        self.goal = self.position.copy()
        self.elapsed = 0.0
        self.duration = 0.0
        self.accepted = 0
        self.rejected = 0
        self.steps = 0

    def sample_motion(self, rng):
        self.start = self.position.copy()
        self.goal = self.position.copy()
        self.elapsed = self.duration = 0.0
        self.command_velocity.fill(0)
        self._sample(rng)

    def _sample(self, rng):
        c = self.cfg
        for _ in range(c.arm_sample_attempts):
            extension = rng.uniform(1.3, 3.0)
            goal = np.array([
                rng.uniform(-1.8, 1.8),
                extension,
                np.clip(extension + rng.uniform(-0.3, 0.3), 0.8, 3.15),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.6, 0.6),
                rng.uniform(-0.8, 0.8),
            ], dtype=np.float32)
            goal = np.clip(goal, np.maximum(self.lower, self.position_at_reset - c.arm_position_limit),
                           np.minimum(self.upper, self.position_at_reset + c.arm_position_limit))
            self.start = self.position.copy()
            self.goal = goal
            speed = max(c.arm_target_max_velocity, 1e-6)
            self.duration = max(rng.uniform(5.0, 8.0) * 0.6 / speed,
                                1.875 * np.max(np.abs(goal - self.start)) / speed)
            if self.steps == 0:
                self.duration = max(self.duration, c.arm_startup_seconds)
            self.elapsed = 0.0
            self.accepted += 1
            return
        self.start = self.goal = self.position.copy()
        self.elapsed = 0.0
        self.duration = c.arm_retry_seconds
        self.rejected += 1

    def step(self, rng):
        c = self.cfg
        if c.disturbance_intensity == 0:
            self.command_velocity.fill(0)
            return self.position.copy()
        if self.elapsed >= self.duration:
            self._sample(rng)
        u = np.clip(self.elapsed / max(self.duration, c.dt), 0.0, 1.0)
        blend = u**3 * (10 - 15*u + 6*u*u)
        raw = self.start + blend * (self.goal - self.start)
        delta = c.arm_target_max_velocity * c.dt
        position = (self.position + np.clip(raw - self.position, -delta, delta)).astype(np.float32)
        self.command_velocity[:] = (position - self.position) / c.dt
        self.position[:] = position
        self.elapsed += c.dt
        self.steps += 1
        return self.position.copy()


class SmoothRandomCommand:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start = np.zeros(3, dtype=np.float32)
        self.goal = self.start.copy()
        self.current = self.start.copy()
        self.elapsed = 0.0
        self.duration = 0.0

    def reset(self, command=None):
        base = np.zeros(3, dtype=np.float32) if command is None else np.asarray(command, dtype=np.float32)
        self.start = self.goal = self.current = base.copy()
        self.elapsed = self.duration = 0.0

    def step(self, rng, dt):
        if self.elapsed >= self.duration:
            self.start = self.current.copy()
            bounds = np.asarray(self.cfg.command_target_ranges)
            self.goal = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
            self.elapsed = 0.0
            self.duration = rng.uniform(*self.cfg.command_transition_seconds)
        self.elapsed += dt
        u = np.clip(self.elapsed / self.duration, 0.0, 1.0)
        blend = u**3 * (10 - 15*u + 6*u*u)
        self.current = self.start + blend * (self.goal - self.start)
        return self.current.copy()


def sample_command(rng, cfg):
    bounds = np.asarray(cfg.command_target_ranges)
    command = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
    if rng.random() < cfg.stand_probability:
        command.fill(0)
    return command


def gait_targets(phase, command):
    phases = np.remainder(phase + np.array([0.5, 0.0, 0.0, 0.5]), 1.0)
    if np.linalg.norm(command) < 0.1:
        phases.fill(0.25)
    clock = np.sin(2 * np.pi * phases)
    # Smoothed stance probabilities, as in _step_contact_targets().
    def cdf(x):
        return np.array([0.5 * (1 + math.erf(float(v) / (0.07 * math.sqrt(2)))) for v in x])
    contact = cdf(phases) * (1 - cdf(phases - 0.5)) + cdf(phases - 1) * (1 - cdf(phases - 1.5))
    return phases, clock, contact


def make_observation(state, action, command, phase, cfg, arm_command_velocity=None, pose_command=None):
    arm_command_velocity = np.zeros(6) if arm_command_velocity is None else arm_command_velocity
    pose_command = np.zeros(2) if pose_command is None else pose_command
    q = state.root[3:7]
    gravity = rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
    lin = rotate_inverse(q, state.root[7:10])
    ang = rotate_inverse(q, state.root[10:13])
    roll, pitch, _ = rpy_from_quat(q)
    pose = np.array([state.root[2], pitch, roll])
    pose_error = np.array([cfg.target_height, *pose_command]) - pose
    velocity_error = (command - np.array([lin[0], lin[1], ang[2]])) * [2.0, 2.0, 0.25]
    _, clock, _ = gait_targets(phase, command)
    values = (gravity, state.leg_pos - LEG_HOME, state.leg_vel * 0.05,
              action, command * [2.0, 2.0, 0.25], clock, ang * 0.25, lin * 2.0,
              pose, pose_error, velocity_error, state.arm_pos, state.arm_vel * 0.05,
              np.asarray(arm_command_velocity) * 0.05, pose_command)
    observation = np.concatenate(values).astype(np.float32)
    assert observation.shape == (OBSERVATION_SIZE,)
    return np.clip(observation, -cfg.observation_limit, cfg.observation_limit)


# Locomotion reward weights; each contribution is scaled by the control period.
REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.5,
    "response_consistency": -0.05, "lin_vel_z": -0.02, "ang_vel_xy": -0.001,
    "orientation": -5.0, "base_height": -10.0, "torques": -0.0001,
    "dof_vel": -0.0001, "dof_acc": -2.5e-7, "collision": -10.0,
    "action_rate": -0.001, "action_smoothness_1": -0.1,
    "action_smoothness_2": -0.1, "dof_pos_limits": -10.0,
    "feet_slip": -0.04, "feet_clearance": -30.0,
    "tracking_contacts_force": 4.0, "tracking_contacts_velocity": 4.0,
    "raibert_heuristic": -10.0, "loco_energy": -0.00004,
}


class Reward:
    def __init__(self, cfg, leg_lower, leg_upper):
        self.cfg = cfg
        middle = (leg_lower + leg_upper) / 2
        half = (leg_upper - leg_lower) * 0.45
        self.lower, self.upper = middle - half, middle + half

    def reset(self, state):
        self.previous_action = np.zeros(12)
        self.previous_velocity = state.leg_vel.copy()
        self.previous_target = np.array(LEG_HOME)
        self.previous_previous_target = np.array(LEG_HOME)
        self.previous_contact = np.zeros(4, dtype=bool)
        self.velocity_reference = np.zeros(2)
        self.steps = 0

    def compute(self, state, action, command, phase, target, torques, pose_command=None):
        c = self.cfg
        q = state.root[3:7]
        lin = rotate_inverse(q, state.root[7:10])
        ang = rotate_inverse(q, state.root[10:13])
        gravity = rotate_inverse(q, np.array([0., 0., -1.]))
        pitch_cmd, roll_cmd = (0.0, 0.0) if pose_command is None else pose_command
        desired_gravity = rotate_inverse(quat_from_rpy(roll_cmd, pitch_cmd, 0), np.array([0., 0., -1.]))
        self.velocity_reference += (1 - math.exp(-c.dt / 0.4)) * (command[:2] - self.velocity_reference)
        phases, _, desired_contact = gait_targets(phase, command)
        contact = state.foot_force[:, 2] > 1.0
        slip_contact = contact | self.previous_contact
        swing_phase = 1 - np.abs(1 - np.clip(2 * phases - 1, 0, 1) * 2)
        foot_target_z = c.swing_height * swing_phase + 0.02
        yaw = rpy_from_quat(q)[2]
        yaw_q = quat_from_rpy(0, 0, yaw)
        feet_body = rotate_inverse(yaw_q, state.foot_pos - state.root[:3])
        phase_offset = np.abs(1 - 2 * phases) - 0.5
        target_x = np.array([1, 1, -1, -1]) * c.stance_length / 2
        target_y = np.array([1, -1, 1, -1]) * c.stance_width / 2
        target_x += phase_offset * command[0] * 0.5 / c.gait_frequency
        target_y += phase_offset * command[2] * c.stance_length / 2 * 0.5 / c.gait_frequency * [1, 1, -1, -1]
        terms = {
            "tracking_lin_vel": np.exp(-np.sum((command[:2] - lin[:2])**2) / 0.25),
            "tracking_ang_vel": np.exp(-(command[2] - ang[2])**2 / 0.25),
            "response_consistency": np.sum((lin[:2] - self.velocity_reference)**2),
            "lin_vel_z": lin[2]**2, "ang_vel_xy": np.sum(ang[:2]**2),
            "orientation": np.sum((gravity[:2] - desired_gravity[:2])**2),
            "base_height": (state.root[2] - c.target_height)**2,
            "torques": np.sum(torques**2), "dof_vel": np.sum(state.leg_vel**2),
            "dof_acc": np.sum(((state.leg_vel - self.previous_velocity) / c.dt)**2),
            "collision": np.count_nonzero(np.linalg.norm(state.collision_forces, axis=-1) > 0.1),
            "action_rate": np.sum((action - self.previous_action)**2),
            "action_smoothness_1": np.sum((target - self.previous_target)**2) if self.steps > 0 else 0.0,
            "action_smoothness_2": np.sum((target - 2*self.previous_target + self.previous_previous_target)**2) if self.steps > 1 else 0.0,
            "dof_pos_limits": np.sum(np.maximum(self.lower - state.leg_pos, 0) + np.maximum(state.leg_pos - self.upper, 0)),
            "feet_slip": np.sum(slip_contact * np.sum(state.foot_vel[:, :2]**2, axis=1)),
            "feet_clearance": np.sum((foot_target_z - state.foot_pos[:, 2])**2 * (1 - desired_contact)),
            "tracking_contacts_force": -np.mean((1-desired_contact) * (1 - np.exp(-np.sum(state.foot_force**2, axis=1) / 100.0))),
            "tracking_contacts_velocity": -np.mean(desired_contact * (1 - np.exp(-np.sum(state.foot_vel**2, axis=1) / 10.0))),
            "raibert_heuristic": np.sum((feet_body[:, :2] - np.column_stack([target_x, target_y]))**2),
            "loco_energy": np.sum((torques * state.leg_vel)**2),
        }
        weighted = {name: float(value * REWARD_WEIGHTS[name] * c.dt) for name, value in terms.items()}
        positive = sum(max(0.0, v) for v in weighted.values())
        negative = sum(min(0.0, v) for v in weighted.values())
        reward = positive * math.exp(negative / 0.02)
        self.previous_action = action.copy()
        self.previous_velocity = state.leg_vel.copy()
        self.previous_previous_target = self.previous_target.copy()
        self.previous_target = target.copy()
        self.previous_contact = contact.copy()
        self.steps += 1
        return float(reward), weighted
