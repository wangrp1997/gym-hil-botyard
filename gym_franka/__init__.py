#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
