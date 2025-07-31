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

import gymnasium as gym
import numpy as np

import gym_hil  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description='Test automatic picking functionality')
    parser.add_argument('--step-size', type=float, default=0.01, help='Step size for movement in meters')
    parser.add_argument(
        '--render-mode', type=str, default='rgb_array', choices=['human', 'rgb_array'], help='Rendering mode'
    )
    args = parser.parse_args()

    # Create Franka environment - Use base environment first to debug
    env = gym.make(
        'gym_hil/PandaPickCubeBase-v0',  # Use the base environment for debugging
        render_mode=args.render_mode,
        image_obs=True,
        mode='auto',
    )

    # Print observation space for debugging
    print('Observation space:', env.observation_space)

    # Reset and check observation structure
    obs, _ = env.reset()

    print('Observation keys:', list(obs.keys()))
    if 'pixels' in obs:
        print('Pixels keys:', list(obs['pixels'].keys()))

    # Now try with the wrapped version
    print('\nTrying wrapped environment...')
    env = gym.make(
        'gym_hil/PandaPickCubeAuto-v0',
        render_mode=args.render_mode,
        image_obs=False,
        use_gamepad=False,
        mode='auto',
    )

    # Print observation space for the wrapped environment
    print('Wrapped observation space:', env.observation_space)

    # Reset and check wrapped observation structure
    obs, _ = env.reset()
    print('Wrapped observation keys:', list(obs.keys()))

    # Reset environment
    obs, _ = env.reset()

    dummy_action = np.zeros(4, dtype=np.float32)
    # This ensures the "stay gripper" action is set when the intervention button is not pressed
    dummy_action[-1] = 1

    # Statistics variables
    successful_picks = 0
    failed_picks = 0
    total_episodes = 0  # 初始化episode计数器

    try:             
        print('Starting automatic picking...')
        while True:
            # 重置环境开始新的episode
            obs, _ = env.reset()
            episode_success = False  # 初始化episode成功标志
            total_episodes += 1  # 增加episode计数
            
            print(f'Starting episode {total_episodes}...')
            
            while True:
                # Step the environment
                obs, reward, terminated, truncated, info = env.step(dummy_action)
                
                # Check for success
                if info.get('succeed', False):
                    episode_success = True
                    successful_picks += 1
                    print(f'✅ Episode {total_episodes} pick successful!')
                    break
                
                # Check for failure
                if terminated or truncated:
                    if not episode_success:
                        failed_picks += 1
                        print(f'❌ Episode {total_episodes} pick failed')
                    break
                
                # Add a small delay to control update rate
                time.sleep(0.05)

    except KeyboardInterrupt:
        print('\nInterrupted by user')
        # 在打断时total_episodes保持当前值，不再增加
    except Exception as e:
        print(f'\nError during testing: {e}')
    finally:
        # Clean up resources
        env.close()
        
        # Print statistics
        print('\n' + '='*50)
        print('Test Results:')
        print(f'Total episodes: {total_episodes}')
        print(f'Successful picks: {successful_picks}')
        print(f'Failed picks: {failed_picks}')
        if total_episodes > 0:
            print(f'Success rate: {successful_picks/total_episodes*100:.1f}%')
        else:
            print('Success rate: N/A (no episodes completed)')
        print('='*50)
        print('Session ended')


if __name__ == '__main__':
    main()
    