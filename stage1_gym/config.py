"""Only parameters used by the single-agent locomotion task.

Adapted from RoboDuet config/go1.py, wtw.py and wbc.py; see ../NOTICE.md.
"""

from dataclasses import dataclass
import math


LEG_NAMES = tuple(
    f"{leg}_{joint}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf")
)
ARM_NAMES = tuple(f"x5_joint{i}" for i in range(1, 7))
FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEG_HOME = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
            0.1, 1.0, -1.5, -0.1, 1.0, -1.5)
ARM_KP = (50.0, 50.0, 80.0, 30.0, 20.0, 20.0)
ARM_KD = (5.0, 10.0, 10.0, 2.5, 2.0, 1.0)


@dataclass(frozen=True)
class TaskConfig:
    physics_dt: float = 0.005
    decimation: int = 4
    episode_seconds: float = 120.0
    action_scale: float = 0.25
    action_limit: float = 10.0
    hip_scale: float = 0.5
    leg_kp: float = 35.0
    leg_kd: float = 1.0
    observation_limit: float = 100.0
    target_height: float = 0.30
    reset_height: float = 0.34
    reset_height_noise: float = 0.02
    reset_tilt: float = 0.05
    reset_joint_noise: float = 0.05
    reset_velocity_noise: float = 0.05
    arm_initial_noise: float = 0.0
    terminal_height: float = 0.17
    terminal_tilt: float = 1.2
    command_ranges: tuple = ((0.0, 0.35), (-0.1, 0.1), (-0.55, 0.55))
    command_target_ranges: tuple = ((0.12, 0.35), (-0.1, 0.1), (-0.55, 0.55))
    command_transition_seconds: tuple = (5.0, 9.0)
    pose_command_ranges: tuple = ((0.0, 0.0), (0.0, 0.0))  # pitch, roll (rad)
    command_seconds: float = 10.0
    stand_probability: float = 0.1
    gait_frequency: float = 3.0
    swing_height: float = 0.04
    stance_width: float = 0.30
    stance_length: float = 0.45
    # Fixed intensity: the Gym environment never needs a trainer/global counter.
    disturbance_intensity: float = 1.0
    arm_accel_seconds: float = 0.01
    arm_max_accel: float = 10.0
    arm_max_velocity: float = 5.0
    arm_random_position_limit: float = 0.08
    arm_position_limit: float = 3.2
    arm_startup_seconds: float = 2.0
    arm_target_max_velocity: float = 2.4
    arm_gripper_max_drop: float = 0.1
    arm_gripper_height_margin: float = 0.02
    arm_planning_height: float = 0.08
    arm_body_clearance: float = 0.03
    arm_path_resolution: float = 0.025
    arm_sample_attempts: int = 48
    arm_retry_seconds: float = 0.5
    arm_zero_accel_probability: float = 0.3
    arm_zero_velocity_probability: float = 0.005
    # Rigid-body DR is applied once before prepare_sim, never during reset.
    randomize_dynamics: bool = True
    arm_mass_scale: tuple = (0.8, 1.2)
    arm_com_noise: float = 0.01
    friction_range: tuple = (0.5, 1.25)
    motor_strength_range: tuple = (0.9, 1.1)
    motor_offset_noise: float = 0.02

    @property
    def dt(self):
        return self.physics_dt * self.decimation

    @property
    def max_steps(self):
        return int(math.ceil(self.episode_seconds / self.dt))

    def __post_init__(self):
        positive = ("physics_dt", "episode_seconds", "action_scale", "action_limit",
                    "observation_limit", "command_seconds", "gait_frequency",
                    "arm_accel_seconds", "terminal_height", "terminal_tilt")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("decimation must be a positive integer")
        for name in ("disturbance_intensity", "stand_probability",
                     "arm_zero_accel_probability", "arm_zero_velocity_probability"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if len(self.command_ranges) != 3 or len(self.command_target_ranges) != 3:
            raise ValueError("command_ranges must contain vx, vy, yaw ranges")
        if len(self.pose_command_ranges) != 2:
            raise ValueError("pose_command_ranges must contain pitch and roll ranges")
        for bounds in (*self.command_ranges, *self.command_target_ranges, self.command_transition_seconds,
                       *self.pose_command_ranges, self.arm_mass_scale, self.friction_range, self.motor_strength_range):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Invalid range: {bounds}")
        for name in ("reset_height_noise", "reset_tilt", "reset_joint_noise",
                      "reset_velocity_noise", "arm_initial_noise", "arm_com_noise",
                      "motor_offset_noise", "arm_max_accel", "arm_max_velocity",
                      "arm_random_position_limit", "arm_position_limit", "arm_startup_seconds",
                      "arm_target_max_velocity", "arm_gripper_max_drop", "arm_gripper_height_margin",
                      "arm_planning_height", "arm_body_clearance", "arm_path_resolution",
                      "arm_retry_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.arm_sample_attempts, int) or self.arm_sample_attempts < 1:
            raise ValueError("arm_sample_attempts must be a positive integer")
        if self.reset_height - self.reset_height_noise <= self.terminal_height:
            raise ValueError("Reset height must be above terminal_height")
        if min(self.arm_mass_scale) <= 0 or min(self.motor_strength_range) <= 0:
            raise ValueError("Mass and motor strength scales must be positive")
