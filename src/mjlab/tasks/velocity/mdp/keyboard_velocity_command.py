"""Keyboard-controlled velocity command for testing with camera-relative controls."""

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

  Press a direction key to toggle it on, press again to toggle off.
  Press SPACE to stop all movement.
  """

  def __init__(self) -> None:
    self._active: dict[str, bool] = {
      "forward": False,
      "backward": False,
      "left": False,
      "right": False,
      "turn_left": False,
      "turn_right": False,
    }
    # Store camera pose from viewer (updated every frame via callback)
    self._camera_pose: np.ndarray | None = None
    # Callback to get current camera pose from viewer
    self._get_camera_pose: Callable[[], np.ndarray | None] | None = None

  def toggle(self, key: str) -> None:
    """Toggle a movement key on/off."""
    if key in self._active:
      self._active[key] = not self._active[key]
      # Clear opposing direction
      opposites = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
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
  def left(self) -> bool:
    return self._active["left"]

  @property
  def right(self) -> bool:
    return self._active["right"]

  @property
  def turn_left(self) -> bool:
    return self._active["turn_left"]

  @property
  def turn_right(self) -> bool:
    return self._active["turn_right"]

  @property
  def camera_pose(self) -> np.ndarray | None:
    # Get fresh camera pose if callback is set
    if self._get_camera_pose is not None:
      self._camera_pose = self._get_camera_pose()
    return self._camera_pose

  def set_camera_pose_callback(self, callback: Callable[[], np.ndarray | None]) -> None:
    """Set callback to get camera pose from viewer."""
    self._get_camera_pose = callback

  def reset(self) -> None:
    """Reset all key states."""
    self.stop_all()


def create_keyboard_callback(
  keyboard_state: KeyboardState,
  get_camera_pose: Callable[[], np.ndarray | None],
) -> Callable[[int], None]:
  """Create a key callback for the native viewer with toggle behavior.

  Press direction keys to toggle movement on/off. Press SPACE to stop all.

  Args:
      keyboard_state: Shared state to update on key press.
      get_camera_pose: Function that returns current camera pose (e.g., viewer.camera_pose).

  Returns:
      Key callback function for NativeMujocoViewer.
  """
  # Set up camera pose callback for continuous updates
  keyboard_state.set_camera_pose_callback(get_camera_pose)

  # GLFW key codes
  KEY_SPACE = 32
  KEY_UP = 265
  KEY_DOWN = 264
  KEY_LEFT = 263
  KEY_RIGHT = 262
  KEY_Q = 81  # Turn left
  KEY_E = 69  # Turn right
  KEY_W = 87  # Alternative forward
  KEY_S = 83  # Alternative backward
  KEY_A = 65  # Alternative left
  KEY_D = 68  # Alternative right

  def callback(key: int) -> None:
    # SPACE stops all movement
    if key == KEY_SPACE:
      keyboard_state.stop_all()
      return

    # Toggle direction keys
    if key == KEY_UP or key == KEY_W:
      keyboard_state.toggle("forward")
    elif key == KEY_DOWN or key == KEY_S:
      keyboard_state.toggle("backward")
    elif key == KEY_LEFT or key == KEY_A:
      keyboard_state.toggle("left")
    elif key == KEY_RIGHT or key == KEY_D:
      keyboard_state.toggle("right")
    elif key == KEY_Q:
      keyboard_state.toggle("turn_left")
    elif key == KEY_E:
      keyboard_state.toggle("turn_right")

  return callback


class KeyboardVelocityCommand(CommandTerm):
  """Velocity command controlled by keyboard with camera-relative directions.

  Hold keys to move (like a game):
  - W / UP: Move forward (camera's forward direction projected to ground)
  - S / DOWN: Move backward
  - A / LEFT: Strafe left
  - D / RIGHT: Strafe right
  - Q: Turn left (yaw)
  - E: Turn right (yaw)

  Release key to stop. Movement is relative to camera view direction.
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
    """Update velocity command based on keyboard state and camera pose."""
    ks = self.keyboard_state

    # Compute velocity in world frame based on camera orientation
    ang_vel_z = 0.0

    # Get camera forward/right directions (projected to ground plane)
    cam_forward = np.array([1.0, 0.0])  # Default: world X
    cam_right = np.array([0.0, -1.0])  # Default: world -Y

    if ks.camera_pose is not None:
      # Camera pose is 4x4 matrix, extract forward direction (-Z in camera frame)
      # and right direction (X in camera frame)
      rot = ks.camera_pose[:3, :3]
      # Note: We negate because we want "forward" to mean "away from camera"
      # (the direction the user is looking), not "toward camera target"
      cam_forward_3d = rot[:, 2]  # Z points away from where camera looks
      cam_right_3d = -rot[:, 0]  # Negate X to fix left/right

      # Project to ground plane (XY) and normalize
      cam_forward = cam_forward_3d[:2]
      cam_right = cam_right_3d[:2]
      fwd_norm = np.linalg.norm(cam_forward)
      right_norm = np.linalg.norm(cam_right)
      if fwd_norm > 1e-6:
        cam_forward = cam_forward / fwd_norm
      if right_norm > 1e-6:
        cam_right = cam_right / right_norm

    # Compute world-frame velocity from keyboard input
    vel_world = np.zeros(2)
    if ks.forward:
      vel_world += cam_forward * self.cfg.lin_vel_scale
    if ks.backward:
      vel_world -= cam_forward * self.cfg.lin_vel_scale
    if ks.right:
      vel_world += cam_right * self.cfg.lin_vel_scale
    if ks.left:
      vel_world -= cam_right * self.cfg.lin_vel_scale

    if ks.turn_left:
      ang_vel_z = self.cfg.ang_vel_scale
    if ks.turn_right:
      ang_vel_z = -self.cfg.ang_vel_scale

    # Transform world velocity to body frame for each environment
    root_quat = self.robot.data.root_link_quat_w
    root_mat = matrix_from_quat(root_quat)  # (num_envs, 3, 3)

    # World velocity as tensor
    vel_world_t = torch.tensor(
      [vel_world[0], vel_world[1], 0.0], device=self.device, dtype=torch.float32
    )

    # Transform to body frame: v_b = R^T @ v_w
    vel_body = torch.einsum("nij,j->ni", root_mat.transpose(1, 2), vel_world_t)

    # Set command for all environments
    self.vel_command_b[:, 0] = vel_body[:, 0]
    self.vel_command_b[:, 1] = vel_body[:, 1]
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
