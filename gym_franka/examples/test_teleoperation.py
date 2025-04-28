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

import argparse
import time
import numpy as np

from gym_franka.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_franka.wrappers.viewer_wrapper import PassiveViewerWrapper
from gym_franka.wrappers.hil_wrappers import EEActionWrapper, InputsControlWrapper, EEActionSpaceParams


def main():
    parser = argparse.ArgumentParser(description="Control Franka robot interactively")
    parser.add_argument(
        "--step-size", 
        type=float, 
        default=0.05,
        help="Step size for movement in meters"
    )
    parser.add_argument(
        "--render-mode", 
        type=str, 
        default="human",
        choices=["human", "rgb_array"],
        help="Rendering mode"
    )
    parser.add_argument(
        "--use-keyboard", 
        action="store_true",
        help="Use keyboard control"
    )
    args = parser.parse_args()

    # Create Franka environment
    env = PandaPickCubeGymEnv(
        render_mode=args.render_mode,
        image_obs=True,
        reward_type="dense",
    )
    
    # Add PassiveViewer wrapper for visualization
    env = PassiveViewerWrapper(env, show_left_ui=True, show_right_ui=True)
    
    # Add EEActionWrapper to change action space to end-effector control
    ee_params = EEActionSpaceParams(args.step_size, args.step_size, args.step_size)
    env = EEActionWrapper(env, ee_action_space_params=ee_params, use_gripper=True)
    

    # Add GamepadControlWrapper to enable gamepad control
    env = InputsControlWrapper(
        env,
        x_step_size=args.step_size,
        y_step_size=args.step_size,
        z_step_size=args.step_size,
        use_gripper=True,
        use_gamepad=not args.use_keyboard,
    )
    
    # Reset environment
    obs, _ = env.reset()
    dummy_action = np.zeros(4, dtype=np.float32)
    
    try:
        while True:

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            
            # Print some feedback
            if info.get("succeed", False):
                print("\nSuccess! Block has been picked up.")
            
            # If auto-reset is disabled, manually reset when episode ends
            if terminated or truncated:
                print("Episode ended, resetting environment")
                obs, _ = env.reset()
                
            # Add a small delay to control update rate
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        env.close()
        print("Session ended")


if __name__ == "__main__":
    main() 