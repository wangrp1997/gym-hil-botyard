from gym_franka.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gymnasium.envs.registration import register

register(
    id="gym_franka/PandaPickCube-v0",
    entry_point="gym_franka.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
