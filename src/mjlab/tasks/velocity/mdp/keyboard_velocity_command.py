"""Keyboard-controlled velocity command with tank controls for testing."""

from __future__ import annotations

__all__ = [
  "KeyboardState",
  "KeyboardVelocityCommand",
  "KeyboardVelocityCommandCfg",
  "create_keyboard_callback",
]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class KeyboardState:
  """Shared state for keyboard input with toggle behavior.

  Camera-relative controls with auto-rotate:
  - W/Up: Move forward, auto-rotate toward camera direction
  - S/Down: Move backward
  - A/Left: Turn left, D/Right: Turn right (yaw)
  - SPACE: Stop all movement
  """

  def __init__(self) -> None:
    self._active: dict[str, bool] = {
      "forward": False,
      "backward": False,
      "turn_left": False,
      "turn_right": False,
    }
    self._get_camera_pose: Callable[[], np.ndarray | None] | None = None

  def toggle(self, key: str) -> None:
    """Toggle a movement key on/off."""
    if key in self._active:
      self._active[key] = not self._active[key]
      # Clear opposing direction
      opposites = {
        "forward": "backward",
        "backward": "forward",
        "turn_left": "turn_right",
        "turn_right": "turn_left",
      }
      if self._active[key] and key in opposites:
        self._active[opposites[key]] = False

  def stop_all(self) -> None:
    """Stop all movement."""
    for key in self._active:
      self._active[key] = False

  @property
  def forward(self) -> bool:
    return self._active["forward"]

  @property
  def backward(self) -> bool:
    return self._active["backward"]

  @property
  def turn_left(self) -> bool:
    return self._active["turn_left"]

  @property
  def turn_right(self) -> bool:
    return self._active["turn_right"]

  @property
  def camera_pose(self) -> np.ndarray | None:
    """Get current camera pose from viewer."""
    if self._get_camera_pose is not None:
      return self._get_camera_pose()
    return None

  def set_camera_pose_callback(self, callback: Callable[[], np.ndarray | None]) -> None:
    """Set callback to get camera pose from viewer."""
    self._get_camera_pose = callback

  def reset(self) -> None:
    """Reset all key states."""
    self.stop_all()


def create_keyboard_callback(
  keyboard_state: KeyboardState,
  get_camera_pose: Callable[[], np.ndarray | None] | None = None,
) -> Callable[[int], None]:
  """Create a key callback for the native viewer with toggle behavior.

  Camera-relative controls - press to toggle on/off, SPACE to stop all:
  - W/Up: Forward (auto-rotates toward camera direction)
  - S/Down: Backward
  - A/Left: Turn left
  - D/Right: Turn right

  Args:
      keyboard_state: Shared state to update on key press.
      get_camera_pose: Optional callback to get camera pose for auto-rotate.

  Returns:
      Key callback function for NativeMujocoViewer.
  """
  if get_camera_pose is not None:
    keyboard_state.set_camera_pose_callback(get_camera_pose)

  # GLFW key codes
  KEY_SPACE = 32
  KEY_UP = 265
  KEY_DOWN = 264
  KEY_LEFT = 263
  KEY_RIGHT = 262
  KEY_W = 87
  KEY_S = 83
  KEY_A = 65
  KEY_D = 68

  def callback(key: int) -> None:
    if key == KEY_SPACE:
      keyboard_state.stop_all()
      return

    if key == KEY_UP or key == KEY_W:
      keyboard_state.toggle("forward")
    elif key == KEY_DOWN or key == KEY_S:
      keyboard_state.toggle("backward")
    elif key == KEY_LEFT or key == KEY_A:
      keyboard_state.toggle("turn_left")
    elif key == KEY_RIGHT or key == KEY_D:
      keyboard_state.toggle("turn_right")

  return callback


class KeyboardVelocityCommand(CommandTerm):
  """Velocity command controlled by keyboard with camera-relative auto-rotate.

  Camera-relative controls (like 3rd person games):
  - W / UP: Move forward, auto-rotate toward camera direction
  - S / DOWN: Move backward
  - A / LEFT: Turn left (yaw)
  - D / RIGHT: Turn right (yaw)
  - SPACE: Stop all movement

  When moving forward/backward, the robot automatically rotates to align
  with the camera's viewing direction.
  """

  cfg: KeyboardVelocityCommandCfg

  def __init__(self, cfg: KeyboardVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    # keyboard_state is created in build() if not provided
    assert cfg.keyboard_state is not None
    self.keyboard_state: KeyboardState = cfg.keyboard_state

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Keyboard command doesn't resample - it's always from keyboard input
    pass

  def _update_command(self) -> None:
    """Update velocity command with auto-rotate toward camera direction."""
    ks = self.keyboard_state

    # Default: no movement
    lin_vel_x = 0.0
    ang_vel_z: float | torch.Tensor = 0.0

    # Check if we need auto-rotate (forward/backward pressed and camera available)
    if (ks.forward or ks.backward) and ks.camera_pose is not None:
      # Get camera forward direction (projected to XY)
      rot = ks.camera_pose[:3, :3]
      cam_forward_3d = -rot[:, 2]  # -Z is forward in OpenGL convention
      cam_forward_xy = cam_forward_3d[:2]
      cam_norm = np.linalg.norm(cam_forward_xy)
      if cam_norm > 1e-6:
        cam_forward_xy = cam_forward_xy / cam_norm

      # Desired yaw from camera direction
      desired_yaw = np.arctan2(cam_forward_xy[1], cam_forward_xy[0])

      # Get robot's current yaw from quaternion (wxyz format)
      root_quat = self.robot.data.root_link_quat_w
      w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
      current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

      # Compute yaw error (normalized to [-pi, pi])
      yaw_error = desired_yaw - current_yaw
      yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

      # P-controller for auto-rotate
      yaw_gain = self.cfg.yaw_gain
      ang_vel_z = yaw_gain * yaw_error
      ang_vel_z = torch.clamp(
        ang_vel_z, -self.cfg.ang_vel_scale, self.cfg.ang_vel_scale
      )

      # Set forward/backward velocity
      if ks.forward:
        lin_vel_x = self.cfg.lin_vel_scale
      else:
        lin_vel_x = -self.cfg.lin_vel_scale

    else:
      # No camera or not moving forward/backward: use tank controls
      if ks.forward:
        lin_vel_x = self.cfg.lin_vel_scale
      elif ks.backward:
        lin_vel_x = -self.cfg.lin_vel_scale

      if ks.turn_left:
        ang_vel_z = self.cfg.ang_vel_scale
      elif ks.turn_right:
        ang_vel_z = -self.cfg.ang_vel_scale

    # Set commands
    self.vel_command_b[:, 0] = lin_vel_x
    self.vel_command_b[:, 1] = 0.0
    if isinstance(ang_vel_z, torch.Tensor):
      self.vel_command_b[:, 2] = ang_vel_z
    else:
      self.vel_command_b[:, 2] = ang_vel_z

  # Visualization (reuse from UniformVelocityCommand)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw velocity command and actual velocity arrows."""
    batch = visualizer.env_idx

    if batch >= self.num_envs:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    base_pos_w = base_pos_ws[batch]
    base_mat_w = base_mat_ws[batch]
    cmd = cmds[batch]
    lin_vel_b = lin_vel_bs[batch]
    ang_vel_b = ang_vel_bs[batch]

    if np.linalg.norm(base_pos_w) < 1e-6:
      return

    def local_to_world(
      vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
    ) -> np.ndarray:
      return pos + mat @ vec

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    # Command linear velocity (blue)
    cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    cmd_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
    )
    visualizer.add_arrow(
      cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
    )

    # Command angular velocity (green)
    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
    )
    visualizer.add_arrow(
      cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
    )

    # Actual linear velocity (cyan)
    act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
    act_lin_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
    )
    visualizer.add_arrow(
      act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
    )

    # Actual angular velocity (light green)
    act_ang_from = act_lin_from
    act_ang_to = local_to_world(
      (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
    )
    visualizer.add_arrow(
      act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
    )


@dataclass(kw_only=True)
class KeyboardVelocityCommandCfg(CommandTermCfg):
  """Configuration for keyboard-controlled velocity command."""

  entity_name: str
  keyboard_state: KeyboardState | None = None  # Created automatically if None
  lin_vel_scale: float = 1.0  # Max linear velocity (m/s)
  ang_vel_scale: float = 1.0  # Max angular velocity (rad/s)
  yaw_gain: float = 2.0  # P-gain for auto-rotate (rad/s per rad error)
  # Keyboard command doesn't resample, but parent class requires this field.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> KeyboardVelocityCommand:
    # Create keyboard state if not provided
    if self.keyboard_state is None:
      self.keyboard_state = KeyboardState()
    return KeyboardVelocityCommand(self, env)
