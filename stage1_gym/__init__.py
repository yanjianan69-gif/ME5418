"""OpenAI Gym 0.26 registration for the standalone first-stage task."""

from gym.envs.registration import register

from .config import TaskConfig

register(id="ArmDisturbanceLocomotion-v0", entry_point="stage1_gym.env:Stage1Env")

__all__ = ["TaskConfig"]
