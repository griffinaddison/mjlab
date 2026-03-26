"""Standard MuJoCo C-backend simulation for real-time single-env playback.

This module provides :class:`MujocoCSimulation`, a lightweight alternative to
:class:`~mjlab.sim.sim.Simulation` that uses ``mujoco.mj_step`` /
``mujoco.mj_forward`` instead of mujoco_warp. It is designed for the play/viewer
path where a single environment needs to run in real-time on CPU (e.g. Mac
Apple Silicon without CUDA).

The class exposes the same property interface as ``Simulation`` — ``model``,
``data``, ``mj_model``, ``mj_data``, ``step()``, ``forward()``, ``reset()`` —
so the rest of the framework (EntityData, managers, viewers) can operate
without changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.sim.numpy_bridge import NumpyBridge
from mjlab.sim.sim import SimulationCfg

if TYPE_CHECKING:
  from mjlab.managers.event_manager import RecomputeLevel
  from mjlab.sensor.sensor_context import SensorContext


class MujocoCSimulation:
  """CPU-only MuJoCo simulation using the standard C library.

  Intended for single-environment playback where GPU acceleration is
  unnecessary. Domain randomisation, CUDA graphs, and GPU sensor
  rendering are not supported; the corresponding methods are no-ops.
  """

  def __init__(
    self,
    num_envs: int,
    cfg: SimulationCfg,
    model: mujoco.MjModel,
    device: str,
  ) -> None:
    if num_envs != 1:
      raise ValueError(
        f"MujocoCSimulation only supports num_envs=1, got {num_envs}. "
        "Use the mujoco_warp backend for multi-environment simulation."
      )

    self.cfg = cfg
    self.device = device
    self.num_envs = num_envs

    # Standard MuJoCo model and data.
    self._mj_model = model
    cfg.mujoco.apply(self._mj_model)
    self._mj_data = mujoco.MjData(model)
    mujoco.mj_forward(self._mj_model, self._mj_data)

    # Bridges expose numpy arrays as torch tensors with [nworld, ...] shape.
    self._model_bridge = NumpyBridge(self._mj_model, nworld=num_envs)
    self._data_bridge = NumpyBridge(self._mj_data, nworld=num_envs)

    self._expanded_fields: set[str] = set()
    self._default_model_fields: dict[str, torch.Tensor] = {}

    self.use_cuda_graph = False

  # -- Properties matching Simulation interface --

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mj_data(self) -> mujoco.MjData:
    return self._mj_data

  @property
  def data(self) -> NumpyBridge:
    return self._data_bridge

  @property
  def model(self) -> NumpyBridge:
    return self._model_bridge

  @property
  def default_model_fields(self) -> dict[str, torch.Tensor]:
    return self._default_model_fields

  @property
  def expanded_fields(self) -> set[str]:
    return self._expanded_fields

  # -- Simulation stepping --

  def step(self) -> None:
    self._data_bridge.sync_to_struct()
    mujoco.mj_step(self._mj_model, self._mj_data)
    self._data_bridge.sync_from_struct()

  def forward(self) -> None:
    self._data_bridge.sync_to_struct()
    mujoco.mj_forward(self._mj_model, self._mj_data)
    self._data_bridge.sync_from_struct()

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Reset simulation state to initial keyframe."""
    mujoco.mj_resetData(self._mj_model, self._mj_data)
    if self._mj_model.nkey > 0:
      mujoco.mj_resetDataKeyframe(self._mj_model, self._mj_data, 0)
    mujoco.mj_forward(self._mj_model, self._mj_data)
    # Clear and rebuild float32 copies from the fresh float64 state.
    self._data_bridge.clear_cache()

  # -- No-ops for GPU-only features --

  def expand_model_fields(self, fields: tuple[str, ...]) -> None:
    """No-op: domain randomisation is not supported on the C backend."""

  def recompute_constants(self, level: RecomputeLevel) -> None:
    """No-op: DR constant recomputation is not needed on the C backend."""

  def set_sensor_context(self, ctx: SensorContext) -> None:
    """No-op: GPU sensor rendering is not supported on the C backend."""

  def sense(self) -> None:
    """No-op: GPU sensor pipeline is not available on the C backend."""

  def create_graph(self) -> None:
    """No-op: CUDA graphs are not used on the C backend."""

  def get_default_field(self, field: str) -> torch.Tensor:
    """Get the default value for a model field, caching for reuse."""
    if field not in self._default_model_fields:
      if not hasattr(self._mj_model, field):
        raise ValueError(f"Field '{field}' not found in model")
      default_value = getattr(self._mj_model, field)
      model_field = getattr(self.model, field)
      self._default_model_fields[field] = torch.as_tensor(
        default_value, dtype=model_field.dtype, device=self.device
      ).clone()
    return self._default_model_fields[field]
