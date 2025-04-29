import time

import imageio
import gymnasium as gym
import numpy as np

import gym_hil

env = gym.make("gym_hil/PandaPickCubeViewer-v0", image_obs=True)
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(200):
    a = action_spec.sample()
    obs, rew, done, truncated, info = env.step(a)
    images = obs["images"]
    frames.append(np.concatenate((images["front"], images["wrist"]), axis=0))

    if done or truncated:
        obs, info = env.reset()

env.close()
imageio.mimsave("franka_render_test.mp4", frames, fps=20)