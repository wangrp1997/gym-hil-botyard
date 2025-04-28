import gymnasium as gymnasium
from gym_franka.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
from gym_franka.wrappers.viewer_wrapper import PassiveViewerWrapper

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
    "PassiveViewerWrapper",
]

from gymnasium.envs.registration import register

register(
    id="gym_franka/PandaPickCube-v0",
    entry_point="gym_franka.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)

register(
    id="gym_franka/PandaPickCubeViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(
        gymnasium.make("gym_franka/PandaPickCube-v0", **kwargs)  # type: ignore
    ),
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
