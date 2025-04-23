import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import gym_franka  # noqa: F401


@pytest.mark.parametrize(
    "env_task, image_obs",
    [
        ("PandaPickCube-v0", False),
        ("PandaPickCube-v0", True),
    ],
)
def test_franka(env_task, image_obs):
    env = gym.make(f"gym_franka/{env_task}", image_obs=image_obs)
    check_env(env.unwrapped)
