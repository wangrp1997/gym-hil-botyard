from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_franka.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymEnv(FrankaGymEnv):
    """Environment for a Panda robot picking up a cube."""
    
    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.05, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 20.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
    ):
        self.reward_type = reward_type
        
        super().__init__(
            action_scale=action_scale,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )
        
        # Task-specific setup
        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position
        
        # Enhance observation space with block position if not using image observations
        if not self.image_obs:
            state_dict = self.observation_space["state"].spaces
            state_dict["block_pos"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
            self.observation_space = spaces.Dict({
                "state": spaces.Dict(state_dict)
            })

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetData(self._model, self._data)
        
        # Reset the robot to home position
        self.reset_robot()
        
        # Sample a new block position
        if self._random_block_position:
            block_xy = np.random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        else:
            block_xy = np.asarray([0.3, 0.0])
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)
        
        # Cache the initial block height
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2
        
        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot
        self.apply_action(action)
        
        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()
        
        # Check if block is outside bounds
        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = (
            np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or 
            np.any(block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05))
        )
        
        terminated = self.time_limit_exceeded() or success or exceeded_bounds
        
        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        obs = {}

        # Get robot state
        obs["agent_pos"] = self.get_robot_state()
        
        # Add block position if not using image observations
        if not self.image_obs:
            block_pos = self._data.sensor("block_pos").data.astype(np.float32)
            obs["environment_state"] = block_pos
        
        # Add camera images if using image observations
        if self.image_obs:
            front_view, wrist_view = self.render()
            obs["pixels"] = {
                "front": front_view,
                "wrist": wrist_view
            }
        
        return obs

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        block_pos = self._data.sensor("block_pos").data
        
        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.2)

    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.2


if __name__ == "__main__":
    from gym_franka import PassiveViewerWrapper
    env = PandaPickCubeGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    env.reset()
    for _ in range(100):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()
