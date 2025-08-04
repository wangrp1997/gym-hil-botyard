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

import json
from pathlib import Path
import time
import numpy as np

def load_controller_config(controller_name: str, config_path: str | None = None) -> dict:
    """
    Load controller configuration from a JSON file.

    Args:
        controller_name: Name of the controller to load.
        config_path: Path to the config file. If None, uses the package's default config.

    Returns:
        Dictionary containing the selected controller's configuration.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "controller_config.json"

    with open(config_path) as f:
        config = json.load(f)

    controller_config = config[controller_name] if controller_name in config else config["default"]

    if controller_name not in config:
        print(f"Controller {controller_name} not found in config. Using default configuration.")

    return controller_config


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def reset(self):
        """Reset the controller."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "no-op"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, mode, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, **kwargs):
        super().__init__(x_step_size, y_step_size, z_step_size, **kwargs)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "success": False,
            "failure": False,
            "intervention": False,
            "rerecord": False,
        }
        self.listener = None
        # 新增：记录按下时间
        self._space_down_time = None
        self._enter_down_time = None
        self._r_down_time = None
        self._long_press_threshold = 1.0  # 长按1秒
        self._record_mode = mode == "record"

    def start(self):
        """Start the keyboard listener."""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = True
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = True
                elif key == keyboard.Key.enter:
                    if self._enter_down_time is None:
                        self._enter_down_time = time.time()
                elif key == keyboard.Key.esc:
                    self.key_states["failure"] = True
                    self.episode_end_status = "failure"
                if key == keyboard.Key.space:
                    if self._space_down_time is None:
                        self._space_down_time = time.time()
                elif key == keyboard.Key.r:
                    if self._r_down_time is None:
                        self._r_down_time = time.time()
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.ctrl_r:
                    self.open_gripper_command = False
                elif key == keyboard.Key.ctrl_l:
                    self.close_gripper_command = False
                if self._space_down_time is not None:
                    duration = time.time() - self._space_down_time
                    if self._record_mode or duration >= self._long_press_threshold:
                        self.key_states["intervention"] = not self.key_states["intervention"]
                    self._space_down_time = None

                # enter
                if self._enter_down_time is not None:
                    duration = time.time() - self._enter_down_time
                    if self._record_mode or duration >= self._long_press_threshold:
                        self.key_states["success"] = True
                        self.episode_end_status = "success"
                    self._enter_down_time = None

                # r
                if self._r_down_time is not None:
                    duration = time.time() - self._r_down_time
                    if self._record_mode or duration >= self._long_press_threshold:
                        self.key_states["rerecord"] = True
                    self._r_down_time = None                            
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Right Ctrl and Left Ctrl: Open and close gripper")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  Space: Start/Stop Intervention")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z

    def should_save(self):
        """Return True if Enter was pressed (save episode)."""
        return self.key_states["success"] or self.key_states["failure"]

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.key_states["intervention"]

    def reset(self):
        """Reset the controller."""
        for key in self.key_states:
            self.key_states[key] = False


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01, deadzone=0.1, config_path=None):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.intervention_flag = False
        self.config_path = config_path
        self.controller_config = None

    def start(self):
        """Initialize pygame and the gamepad."""
        import pygame

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            print("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        joystick_name = self.joystick.get_name()
        print(f"Initialized gamepad: {joystick_name}")

        # Load controller configuration based on joystick name
        self.controller_config = load_controller_config(joystick_name, self.config_path)

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})

        print("Gamepad controls:")
        print(f"  {buttons.get('rb', 'RB')} button: Intervention")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print(f"  {buttons.get('lt', 'LT')} button: Close gripper")
        print(f"  {buttons.get('rt', 'RT')} button: Open gripper")
        print(f"  {buttons.get('b', 'B')}/Circle button: Exit")
        print(f"  {buttons.get('y', 'Y')}/Triangle button: End episode with SUCCESS")
        print(f"  {buttons.get('a', 'A')}/Cross button: End episode with FAILURE")
        print(f"  {buttons.get('x', 'X')}/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        import pygame

        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        import pygame

        # Get button mappings from config
        buttons = self.controller_config.get("buttons", {})
        y_button = buttons.get("y", 3)  # Default to 3 if not found
        a_button = buttons.get("a", 0)  # Default to 0 if not found (Logitech F310)
        x_button = buttons.get("x", 2)  # Default to 2 if not found (Logitech F310)
        lt_button = buttons.get("lt", 6)  # Default to 6 if not found
        rt_button = buttons.get("rt", 7)  # Default to 7 if not found
        rb_button = buttons.get("rb", 5)  # Default to 5 if not found

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == y_button:
                    self.episode_end_status = "success"
                elif event.button == a_button:
                    self.episode_end_status = "failure"
                elif event.button == x_button:
                    self.episode_end_status = "rerecord_episode"
                elif event.button == lt_button:
                    self.close_gripper_command = True
                elif event.button == rt_button:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [x_button, a_button, y_button]:
                    self.episode_end_status = None
                elif event.button == lt_button:
                    self.close_gripper_command = False
                elif event.button == rt_button:
                    self.open_gripper_command = False

            # Check for RB button for intervention flag
            if self.joystick.get_button(rb_button):
                self.intervention_flag = True
            else:
                self.intervention_flag = False

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        import pygame

        try:
            # Get axis mappings from config
            axes = self.controller_config.get("axes", {})
            axis_inversion = self.controller_config.get("axis_inversion", {})

            # Get axis indices from config (with defaults if not found)
            left_x_axis = axes.get("left_x", 0)
            left_y_axis = axes.get("left_y", 1)
            right_y_axis = axes.get("right_y", 3)

            # Get axis inversion settings (with defaults if not found)
            invert_left_x = axis_inversion.get("left_x", False)
            invert_left_y = axis_inversion.get("left_y", True)
            invert_right_y = axis_inversion.get("right_y", True)

            # Read joystick axes
            x_input = self.joystick.get_axis(left_x_axis)  # Left/Right
            y_input = self.joystick.get_axis(left_y_axis)  # Up/Down
            z_input = self.joystick.get_axis(right_y_axis)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Apply inversion if configured
            if invert_left_x:
                x_input = -x_input
            if invert_left_y:
                y_input = -y_input
            if invert_right_y:
                z_input = -z_input

            # Calculate deltas
            delta_x = y_input * self.y_step_size  # Forward/backward
            delta_y = x_input * self.x_step_size  # Left/right
            delta_z = z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            print("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0

        # Button states
        self.buttons = {}
        self.quit_requested = False
        self.save_requested = False

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        import hid

        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        print("No gamepad found, check the connection and the product string in HID to add your gamepad")
        return None

    def start(self):
        """Connect to the gamepad using HIDAPI."""
        import hid

        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            print(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            print(f"Connected to {manufacturer} {product}")

            print("Gamepad controls (HID mode):")
            print("  Left analog stick: Move in X-Y plane")
            print("  Right analog stick: Move in Z axis (vertical)")
            print("  Button 1/B/Circle: Exit")
            print("  Button 2/A/Cross: End episode with SUCCESS")
            print("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            print(f"Error opening gamepad: {e}")
            print("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_x = (data[1] - 128) / 128.0
                self.left_y = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = data[6] in [2, 6, 10, 14]

                # Check if RT is pressed
                self.open_gripper_command = data[6] in [8, 10, 12]

                # Check if LT is pressed
                self.close_gripper_command = data[6] in [4, 6, 12]

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = "success"
                elif buttons & 1 << 5:
                    self.episode_end_status = "failure"
                elif buttons & 1 << 4:
                    self.episode_end_status = "rerecord_episode"
                else:
                    self.episode_end_status = None

        except OSError as e:
            print(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_y * self.x_step_size  # Forward/backward
        delta_y = -self.left_x * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def should_quit(self):
        """Return True if quit button was pressed."""
        return self.quit_requested

    def should_save(self):
        """Return True if save button was pressed."""
        return self.save_requested


class AutoPickController(InputController):
    """自动抓取控制器，根据物体位置自动生成运动指令"""
    
    def __init__(self, env, x_step_size=0.01, y_step_size=0.01, z_step_size=0.01):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.env = env
        self.target_pos = None
        self.gripper_closing = False         # 是否正在闭合计数
        self.gripper_closing_count = 0       # 夹爪闭合计数器
        self.gripper_closing_wait = 10       # 闭合后等待帧数
        self.ready_to_lift = False           # 是否可以提升
        self.lifting_count = 0               # 新增：提升阶段计数
        self.lifting_count_max = 5          # 新增：最大提升帧数
        self.lift_start_z = None             # 新增：开始提升时的z坐标
        
    def start(self):
        print("自动抓取控制器已启动")
        
    def stop(self):
        print("自动抓取控制器已停止")
        
    def reset(self):
        self.target_pos = None
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.gripper_closing = False
        self.gripper_closing_count = 0
        self.ready_to_lift = False
        self.lifting_count = 10  # 新增
        self.lift_start_z = None  # 新增
        
    def _get_current_tcp_pos(self):
        try:
            current_obs = self.env.unwrapped.get_observation()
            if "agent_pos" in current_obs:
                robot_state = current_obs["agent_pos"]
                return robot_state[-3:]
            else:
                return [0.0, 0.0, 0.0]
        except Exception as e:
            print(f"获取TCP位置时出错: {e}")
            return [0.0, 0.0, 0.0]
        
    def update(self):
        self.target_pos = self.detect_object()
        if self.ready_to_lift:
            # 新增：提升阶段失败检测
            try:
                obs = self.env.unwrapped.get_observation()
                if "environment_state" in obs:
                    env_state = obs["environment_state"]
                    if len(env_state) >= 3:
                        current_z = env_state[2]
                        # 记录开始提升时的z坐标
                        if self.lift_start_z is None:
                            self.lift_start_z = current_z
                        
                        # 检查是否成功提升（z坐标上升超过0.05米）
                        if current_z > self.lift_start_z + 0.1:
                            # 成功提升，重置计数
                            self.lifting_count = 0
                        else:
                            # 没有成功提升，增加失败计数
                            self.lifting_count += 1
                            
                        if self.lifting_count > self.lifting_count_max:
                            print("提升失败，自动重置抓取状态")
                            self.reset()
                            return
            except Exception as e:
                print(f"提升阶段检测出错: {e}")
            self.close_gripper_command = True
            self.open_gripper_command = False
            return
        if self.target_pos is not None:
            current_pos = self._get_current_tcp_pos()
            self.target_pos[0] = self.target_pos[0] - 0.03
            self.target_pos[1] = self.target_pos[1] + 0.03
            distance = np.linalg.norm(np.array(self.target_pos) - np.array(current_pos))
            if not self.gripper_closing:
                if distance < 0.05:
                    self.gripper_closing = True
                    self.gripper_closing_count = 0
                else:
                    self.open_gripper_command = True
                    self.close_gripper_command = False
            else:
                self.close_gripper_command = True
                self.open_gripper_command = False
                self.gripper_closing_count += 1
                if self.gripper_closing_count >= self.gripper_closing_wait:
                    self.ready_to_lift = True
                    self.lifting_count = 0  # 新增：开始提升时计数归零
                    self.lift_start_z = None  # 新增：重置开始提升时的z坐标
        else:
            self.open_gripper_command = True
            self.close_gripper_command = False
            self.gripper_closing = False
            self.gripper_closing_count = 0
            self.ready_to_lift = False
            self.lifting_count = 0  # 新增
            self.lift_start_z = None  # 新增
        
    def get_deltas(self):
        """获取当前帧的运动增量"""
        if self.target_pos is None:
            return 0.0, 0.0, 0.0
        try:
            current_pos = self._get_current_tcp_pos()
            if not self.ready_to_lift:
                delta = np.array(self.target_pos) - np.array(current_pos)
                delta_x = np.clip(delta[0], -self.x_step_size, self.x_step_size)
                delta_y = np.clip(delta[1], -self.y_step_size, self.y_step_size)
                delta_z = np.clip(delta[2], -self.z_step_size, self.z_step_size)
                return delta_x, delta_y, delta_z
            else:
                # 夹爪闭合后，持续向上提升
                return 0.0, 0.0, self.z_step_size
        except Exception as e:
            print(f"计算运动增量时出错: {e}")
            return 0.0, 0.0, 0.0
            
    def detect_object(self):
        """检测物体位置"""
        try:
            # 获取当前观测
            obs = self.env.unwrapped.get_observation()
            
            # 方法1: 如果有environment_state，直接使用（这是物体的位置）
            if "environment_state" in obs:
                env_state = obs["environment_state"]
                if len(env_state) >= 3:
                    return env_state[:3]  # 返回物体位置
                    
            # 方法2: 如果没有environment_state，尝试从agent_pos推断
            # 这里可以添加更复杂的物体检测逻辑
            # 暂时返回一个固定的目标位置作为示例
            else:
                return self.env.unwrapped._data.sensor("block_pos").data.astype(np.float32)
                
        except Exception as e:
            print(f"物体检测出错: {e}")
            return None

    def should_intervene(self):
        """自动控制器始终返回True，表示一直处于干预状态"""
        return True
        
    def get_episode_end_status(self):
        """
        Get the current episode end status from environment.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        try:
            # 直接调用环境的成功判断方法
            if hasattr(self.env.unwrapped, '_is_success'):
                if self.env.unwrapped._is_success():
                    return "success"
            
            # 检查是否失败（物体掉落）
            obs = self.env.unwrapped.get_observation()
            if "environment_state" in obs:
                env_state = obs["environment_state"]
                if len(env_state) >= 3:
                    # 如果物体z坐标低于0.4，认为失败
                    if env_state[2] < 0.4:
                        return "failure"
            
            return None
            
        except Exception as e:
            print(f"获取episode状态时出错: {e}")
            return None
