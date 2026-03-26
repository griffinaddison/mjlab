"""Bridge for standard MuJoCo C-backend: wraps numpy arrays as torch tensors.

Provides the same interface as WarpBridge so that EntityData, managers, and
viewers can operate identically regardless of whether the simulation is running
on mujoco_warp (GPU) or standard mujoco (CPU).

Because MuJoCo C uses float64 while mujoco_warp uses float32, the bridge
maintains float32 copies of all float64 arrays. The :class:`MujocoCSimulation`
calls :meth:`NumpyBridge.sync_to_struct` before ``mj_step`` and
:meth:`NumpyBridge.sync_from_struct` after to keep the copies in sync.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class NumpyTorchArray:
  """Numpy array wrapped as a torch.Tensor with an optional leading nworld dim.

  Mirrors :class:`~mjlab.sim.sim_data.TorchArray` but backed by a float32
  numpy copy (for float64 sources) or a zero-copy view (for other dtypes).
  """

  def __init__(
    self,
    np_array: np.ndarray,
    *,
    add_batch_dim: bool = True,
    source_f64: np.ndarray | None = None,
  ) -> None:
    self._source_f64 = source_f64
    base = torch.from_numpy(np_array)
    if add_batch_dim:
      base = base.unsqueeze(0)
    self._tensor = base

  # --- indexing ---------------------------------------------------------

  def __getitem__(self, idx: Any) -> Any:
    return self._tensor[idx]

  def __setitem__(self, idx: Any, value: Any) -> None:
    self._tensor[idx] = value

  def __getattr__(self, name: str) -> Any:
    return getattr(self._tensor, name)

  def __repr__(self) -> str:
    return repr(self._tensor)

  # --- torch function protocol ------------------------------------------

  @classmethod
  def __torch_function__(
    cls, func: Any, types: tuple[type, ...], args: tuple[Any, ...] = (), kwargs: dict | None = None
  ) -> Any:
    if kwargs is None:
      kwargs = {}
    if not any(issubclass(t, cls) for t in types):
      return NotImplemented

    def _unwrap(x: Any) -> Any:
      return x._tensor if isinstance(x, cls) else x

    return func(*tuple(_unwrap(a) for a in args), **{k: _unwrap(v) for k, v in kwargs.items()})

  # --- arithmetic -------------------------------------------------------

  def __add__(self, o: Any) -> Any:
    return self._tensor + o

  def __radd__(self, o: Any) -> Any:
    return o + self._tensor

  def __sub__(self, o: Any) -> Any:
    return self._tensor - o

  def __rsub__(self, o: Any) -> Any:
    return o - self._tensor

  def __mul__(self, o: Any) -> Any:
    return self._tensor * o

  def __rmul__(self, o: Any) -> Any:
    return o * self._tensor

  def __truediv__(self, o: Any) -> Any:
    return self._tensor / o

  def __rtruediv__(self, o: Any) -> Any:
    return o / self._tensor

  def __pow__(self, o: Any) -> Any:
    return self._tensor**o

  def __neg__(self) -> Any:
    return -self._tensor

  def __pos__(self) -> Any:
    return +self._tensor

  def __abs__(self) -> Any:
    return abs(self._tensor)

  # --- comparison -------------------------------------------------------

  def __eq__(self, o: Any) -> Any:
    return self._tensor == o

  def __ne__(self, o: Any) -> Any:
    return self._tensor != o

  def __lt__(self, o: Any) -> Any:
    return self._tensor < o

  def __le__(self, o: Any) -> Any:
    return self._tensor <= o

  def __gt__(self, o: Any) -> Any:
    return self._tensor > o

  def __ge__(self, o: Any) -> Any:
    return self._tensor >= o

  # --- sync helpers -----------------------------------------------------

  def sync_to_source(self) -> None:
    """Copy float32 tensor values back to the float64 C struct array."""
    if self._source_f64 is not None:
      np.copyto(self._source_f64, self._tensor.squeeze(0).numpy(), casting="unsafe")

  def sync_from_source(self) -> None:
    """Copy float64 C struct array values into the float32 tensor."""
    if self._source_f64 is not None:
      np.copyto(
        self._tensor.squeeze(0).numpy(),
        self._source_f64.astype(np.float32),
        casting="same_kind",
      )


# Integer dtypes that should NOT receive a leading nworld dimension.
_INT_DTYPES = frozenset({np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64})

# MjData fields that are written by user code and must sync to the C struct
# before mj_step. Other fields (xpos, xquat, etc.) are computed by mj_step.
_WRITABLE_DATA_FIELDS = frozenset({
  "qpos", "qvel", "ctrl", "xfrc_applied", "mocap_pos", "mocap_quat", "qfrc_applied",
})

# Fields whose last dimension of 9 should be reshaped to (3, 3) to match
# mjwarp's mat33f representation.
_MAT33_FIELDS = frozenset({"xmat", "site_xmat", "geom_xmat"})


class NumpyBridge:
  """Wraps ``mujoco.MjModel`` or ``mujoco.MjData`` to expose numpy arrays as torch tensors.

  Provides the same attribute-access interface as
  :class:`~mjlab.sim.sim_data.WarpBridge`:

  * Float arrays are served as float32 tensors with a leading ``nworld=1``
    batch dimension so that downstream code can index as
    ``data.qpos[env_ids, addr]``.
  * Integer lookup arrays (``geom_bodyid``, ``site_bodyid``, …) are returned
    without a batch dimension, matching mujoco_warp's convention.

  Call :meth:`sync_to_struct` before ``mj_step`` and :meth:`sync_from_struct`
  after to keep the float32 tensors and the float64 C struct in sync.
  """

  def __init__(self, struct: Any, nworld: int = 1) -> None:
    object.__setattr__(self, "_struct", struct)
    object.__setattr__(self, "_nworld", nworld)
    object.__setattr__(self, "_cache", {})

  @property
  def nworld(self) -> int:
    return object.__getattribute__(self, "_nworld")

  # Scalar fields that mjwarp stores as per-world arrays of shape (nworld,).
  _SCALAR_TO_ARRAY_FIELDS = frozenset({"time"})

  def __getattr__(self, name: str) -> Any:
    cache = object.__getattribute__(self, "_cache")
    if name in cache:
      return cache[name]

    struct = object.__getattribute__(self, "_struct")
    val = getattr(struct, name)

    if isinstance(val, np.ndarray) and val.ndim > 0:
      add_batch = val.dtype.type not in _INT_DTYPES
      # Reshape flat rotation matrices (N, 9) → (N, 3, 3) to match
      # mjwarp's mat33f representation.
      is_mat33 = name in _MAT33_FIELDS and val.ndim >= 2 and val.shape[-1] == 9
      if is_mat33:
        val = val.reshape(*val.shape[:-1], 3, 3)
      # Float64 arrays need a float32 copy to match mjwarp convention.
      if val.dtype == np.float64:
        source = val  # reshaped view if mat33, original otherwise
        f32_copy = val.astype(np.float32)
        wrapped = NumpyTorchArray(f32_copy, add_batch_dim=add_batch, source_f64=source)
      else:
        wrapped = NumpyTorchArray(val, add_batch_dim=add_batch)
      cache[name] = wrapped
      return wrapped

    # Scalar fields that mjwarp exposes as (nworld,) arrays.
    if name in self._SCALAR_TO_ARRAY_FIELDS and isinstance(val, (int, float)):
      nworld = object.__getattribute__(self, "_nworld")
      tensor = torch.tensor([val], dtype=torch.float32)
      if nworld > 1:
        tensor = tensor.expand(nworld)
      cache[name] = tensor
      return tensor

    return val

  def __setattr__(self, name: str, value: Any) -> None:
    raise AttributeError(
      f"Cannot set attribute '{name}' on NumpyBridge. "
      f"Use in-place operations instead: obj.{name}[:] = value"
    )

  def __repr__(self) -> str:
    struct = object.__getattribute__(self, "_struct")
    return f"NumpyBridge({repr(struct)})"

  def sync_to_struct(self) -> None:
    """Sync writable float32 tensor values back to the float64 C struct.

    Must be called before ``mj_step`` so the C library sees any changes
    made through tensor indexing (e.g. ``data.qpos[0, :] = ...``).
    """
    cache = object.__getattribute__(self, "_cache")
    for name, wrapped in cache.items():
      if isinstance(wrapped, NumpyTorchArray) and name in _WRITABLE_DATA_FIELDS:
        wrapped.sync_to_source()

  def sync_from_struct(self) -> None:
    """Sync all cached float32 tensors from the float64 C struct.

    Must be called after ``mj_step`` / ``mj_forward`` so the tensors
    reflect the updated simulation state.
    """
    cache = object.__getattribute__(self, "_cache")
    for wrapped in cache.values():
      if isinstance(wrapped, NumpyTorchArray):
        wrapped.sync_from_source()

  def clear_cache(self) -> None:
    """Clear cached tensors. Call after ``mj_resetData`` to rebind views."""
    object.__setattr__(self, "_cache", {})

  @property
  def struct(self) -> Any:
    return object.__getattribute__(self, "_struct")
