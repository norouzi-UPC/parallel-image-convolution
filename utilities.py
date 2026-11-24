"""
utilities.py — Local-Image Helper Functions for the Parallel 2D Convolution Lab
-------------------------------------------------------------------------------
Provides:
  1. `DEFAULT_SMALL_IMAGE = "small_image.png"` (600 x 800 grayscale test image)
     and `DEFAULT_LARGE_IMAGE = "large_image.png"` (1600 x 2400 high-res test image).
  2. `load_grayscale_image()`: Loads a local image file (PNG, JPG, etc.), converts
     it to grayscale, optionally resizes it if `target_size` is specified, and
     returns a contiguous 2D `np.float64` array normalized to [0.0, 1.0].
  3. `get_kernels()`: Returns standard 5x5 `blur` and `edges` convolution kernels.
  4. `plot_comparison()`: Saves a clean side-by-side comparison PNG (automatically
     normalizing signed edge-detection magnitudes so both positive and negative edges
     are clearly visible when inspecting for parallel row-chunking seams).
"""

from __future__ import annotations

import os
import numpy as np

DEFAULT_SMALL_IMAGE = "small_image.png"
DEFAULT_LARGE_IMAGE = "large_image.png"


def _resize_bilinear(arr: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Pure-NumPy bilinear resize fallback when Pillow (PIL) is not installed."""
    old_h, old_w = arr.shape
    if (old_h, old_w) == (new_h, new_w):
        return np.ascontiguousarray(arr, dtype=np.float64)

    row_idx = np.linspace(0, old_h - 1, new_h)
    col_idx = np.linspace(0, old_w - 1, new_w)

    r0 = np.floor(row_idx).astype(np.int64)
    c0 = np.floor(col_idx).astype(np.int64)
    r1 = np.minimum(r0 + 1, old_h - 1)
    c1 = np.minimum(c0 + 1, old_w - 1)

    dr = (row_idx - r0)[:, None]
    dc = (col_idx - c0)[None, :]

    top = arr[r0[:, None], c0[None, :]] * (1.0 - dc) + arr[r0[:, None], c1[None, :]] * dc
    bot = arr[r1[:, None], c0[None, :]] * (1.0 - dc) + arr[r1[:, None], c1[None, :]] * dc
    return np.ascontiguousarray(top * (1.0 - dr) + bot * dr, dtype=np.float64)


def load_grayscale_image(
    filepath: str = DEFAULT_SMALL_IMAGE,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a local image file as a contiguous 2D `float64` array in [0.0, 1.0].

    Parameters
    ----------
    filepath : str
        Path to a local image file (e.g. 'small_image.png' or 'large_image.png').
    target_size : tuple[int, int] | None
        Optional `(height, width)` to resize the image. If `None` (default), the
        image's native resolution is preserved.

    Returns
    -------
    np.ndarray
        Contiguous 2D array of dtype `np.float64` with values in [0.0, 1.0].
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Local image file '{filepath}' was not found in '{os.getcwd()}'. "
            f"Please place '{DEFAULT_SMALL_IMAGE}' and '{DEFAULT_LARGE_IMAGE}' "
            f"in your working directory (or pass a valid local image path)."
        )

    # Prefer Pillow (PIL) if installed
    try:
        from PIL import Image

        with Image.open(filepath) as im:
            gray = im.convert("L")
            if target_size is not None:
                target_h, target_w = target_size
                if gray.size != (target_w, target_h):
                    gray = gray.resize((target_w, target_h), Image.Resampling.BILINEAR)
            arr = np.asarray(gray, dtype=np.float64) / 255.0
            return np.ascontiguousarray(arr, dtype=np.float64)
    except ImportError:
        pass

    # Fallback: matplotlib.image
    import matplotlib.image as mpimg

    raw = mpimg.imread(filepath).astype(np.float64)
    if raw.ndim == 3:
        # Standard ITU-R BT.601 luminance weights (RGB -> Gray)
        raw = 0.2989 * raw[..., 0] + 0.5870 * raw[..., 1] + 0.1140 * raw[..., 2]
    if raw.max() > 1.0:
        raw /= 255.0
    if target_size is not None and raw.shape != target_size:
        raw = _resize_bilinear(raw, target_size[0], target_size[1])
    return np.ascontiguousarray(raw, dtype=np.float64)


def get_kernels() -> dict[str, np.ndarray]:
    """Return a dictionary of standard 5x5 float64 convolution kernels."""
    # 5x5 Gaussian-like smoothing kernel (weights sum to 1.0)
    weights_1d = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64)
    blur = np.outer(weights_1d, weights_1d)
    blur /= blur.sum()

    # 5x5 Laplacian-style center-surround edge detector (weights sum to 0.0)
    edges = np.full((5, 5), -1.0, dtype=np.float64)
    edges[2, 2] = 24.0
    edges /= 8.0  # Scale so edge magnitudes sit nicely near [0, 1]

    return {
        "blur": np.ascontiguousarray(blur, dtype=np.float64),
        "edges": np.ascontiguousarray(edges, dtype=np.float64),
    }


def plot_comparison(
    original: np.ndarray,
    filtered: np.ndarray,
    title: str = "5x5 2D Convolution Result",
    save_path: str = "result.png",
    show: bool = False,
) -> None:
    """Save a side-by-side comparison plot of `original` vs `filtered`."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # If filtered contains negative values (e.g., edge detector), visualize magnitude
    if filtered.min() < -1e-9:
        vis = np.abs(filtered)
        vmax = np.percentile(vis, 99.5) if vis.max() > 0 else 1.0
        vis = np.clip(vis / max(vmax, 1e-12), 0.0, 1.0)
        right_label = "Filtered Output (|magnitude|, normalized)"
    else:
        vis = np.clip(filtered, 0.0, 1.0)
        right_label = "Filtered Output"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Original ({original.shape[0]}x{original.shape[1]})")
    axes[0].axis("off")

    axes[1].imshow(vis, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(right_label)
    axes[1].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[utilities] Saved comparison image to '{save_path}'.")
