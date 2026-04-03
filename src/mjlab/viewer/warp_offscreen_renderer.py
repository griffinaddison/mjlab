"""MuJoCo Warp GPU ray-tracing offscreen renderer.

Replaces the EGL/OpenGL OffscreenRenderer with a pure-CUDA path that
avoids glFinish() stalls on multi-tenant GPU nodes.
"""

import logging

import mujoco
import mujoco_warp as mjw
import numpy as np

from mjlab.viewer.viewer_config import ViewerConfig

logger = logging.getLogger(__name__)


class WarpOffscreenRenderer:
  """GPU ray-tracing renderer using MuJoCo Warp.

  Renders directly from Warp GPU state — no CPU round-trip for geometry.
  Requires an MJCF-defined camera specified via ``cfg.camera_name``.
  """

  def __init__(
    self,
    mj_model: mujoco.MjModel,
    wp_model: mjw.Model,
    wp_data: mjw.Data,
    cfg: ViewerConfig,
  ) -> None:
    self._mj_model = mj_model
    self._wp_model = wp_model
    self._wp_data = wp_data
    self._cfg = cfg

    if not cfg.camera_name:
      raise ValueError(
        "WarpOffscreenRenderer requires ViewerConfig.camera_name to be set "
        "to the name of an MJCF-defined camera."
      )

    # Resolve camera name -> MuJoCo camera id.
    self._cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.camera_name)
    if self._cam_id < 0:
      raise ValueError(
        f"Camera '{cfg.camera_name}' not found in the MuJoCo model. "
        f"Available cameras: {[mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mj_model.ncam)]}"
      )

    self._rc: mjw.RenderContext | None = None
    self._height = cfg.height
    self._width = cfg.width

  @property
  def is_initialized(self) -> bool:
    return self._rc is not None

  def initialize(self) -> None:
    if self._rc is not None:
      raise RuntimeError(
        "Renderer is already initialized. Call 'close()' first to reinitialize."
      )

    # Build a boolean mask: only the selected camera is active.
    cam_active = [False] * self._mj_model.ncam
    cam_active[self._cam_id] = True

    self._rc = mjw.create_render_context(
      self._mj_model,
      nworld=self._wp_data.nworld,
      cam_res=[(self._width, self._height)],
      render_rgb=[True],
      render_depth=[False],
      use_textures=True,
      use_shadows=self._cfg.enable_shadows,
      cam_active=cam_active,
    )

    logger.info(
      "WarpOffscreenRenderer initialized: camera='%s' (id=%d), "
      "resolution=%dx%d, nworld=%d",
      self._cfg.camera_name,
      self._cam_id,
      self._width,
      self._height,
      self._wp_data.nworld,
    )

  def update(self, data: mjw.Data | None = None, **kwargs) -> None:
    """Update the BVH and render all worlds in one GPU kernel launch.

    Args:
      data: Optional override for the Warp Data. If None, uses the Data
            passed at construction time. The ``**kwargs`` absorb extra
            arguments (e.g. ``debug_vis_callback``, ``camera``) so that
            call-sites shared with OffscreenRenderer do not break.
    """
    if self._rc is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    d = data if data is not None else self._wp_data
    mjw.refit_bvh(self._wp_model, d, self._rc)
    mjw.render(self._wp_model, d, self._rc)

  def render(self) -> np.ndarray:
    """Return an RGB image for a single env as ``np.ndarray (H, W, 3) uint8``.

    The environment is selected by ``cfg.env_idx``.
    """
    if self._rc is None:
      raise ValueError("Renderer not initialized. Call 'initialize()' first.")

    env_idx = self._cfg.env_idx
    num_pixels = self._width * self._height

    # rc.rgb_data is (nworld, total_rgb_pixels) of uint32 packed as ARGB.
    # For a single active camera, rgb_adr[0] == 0 and total pixels == W*H.
    rgb_packed = self._rc.rgb_data.numpy()[env_idx, :num_pixels]  # (W*H,) uint32

    # Unpack ARGB uint32 -> (H, W, 3) uint8.
    r = ((rgb_packed >> 16) & 0xFF).astype(np.uint8)
    g = ((rgb_packed >> 8) & 0xFF).astype(np.uint8)
    b = (rgb_packed & 0xFF).astype(np.uint8)

    # cam_res is (W, H), so reshape accordingly: H rows, W cols.
    img = np.stack([r, g, b], axis=-1).reshape(self._height, self._width, 3)
    return img

  def close(self) -> None:
    self._rc = None
