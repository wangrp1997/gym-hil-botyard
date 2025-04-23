# gym-franka

A gymnasium environment for Franka Emika Panda arm

## Installation

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n franka python=3.10 && conda activate franka
```

Install gym-franka:
```bash
pip install gym-franka
```

## Quick start

```python
import time

import imageio
import gymnasium as gym
import numpy as np

import franka_sim

env = gym.make("PandaPickCubeVision-v0", render_mode="human", image_obs=True)
action_spec = env.action_space


def sample():
    a = np.random.uniform(action_spec.low, action_spec.high, action_spec.shape)
    return a.astype(action_spec.dtype)


obs, info = env.reset()
frames = []

for i in range(200):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)
    images = obs["images"]
    frames.append(np.concatenate((images["front"], images["wrist"]), axis=0))

    if done:
        obs, info = env.reset()

env.close()
imageio.mimsave("franka_render_test.mp4", frames, fps=20)
```

## Description

Franka Emika Panda environment.

### Action Space


### Observation Space


### Rewards


### Success Criteria


### Starting State


### Episode Termination


### Arguments


### Reset Arguments


## Version History

* v0: Original version


## References


## Contribute

Instead of using `pip` directly, we use `poetry` for development purposes to easily track our dependencies.
If you don't have it already, follow the [instructions](https://python-poetry.org/docs/#installation) to install it.

Install the project with dev dependencies:
```bash
poetry install --all-extras
```

### Follow our style

```bash
# install pre-commit hooks
pre-commit install

# apply style and linter checks on staged files
pre-commit
```

## Acknowledgment

gym-franka is adapted from [franka-sim](https://github.com/rail-berkeley/serl/tree/main/franka_sim) initially built by [Kevin Zakka](https://kzakka.com/).