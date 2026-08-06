"""
_background.py — Median-based background processing.

Background is estimated as the median of the full stack ± 10%.
Since background occupies the majority of the volume, the median
naturally falls within the background intensity range.

  bg_median = median(full stack)
  bg_min    = bg_median * 0.90
  bg_max    = bg_median * 1.10

All three modes use a single-sided threshold:

  threshold = bg_max + tolerance

  pixels <= threshold  →  treated as background
  pixels >  threshold  →  treated as signal (kept)

Three processing modes:
  1. remove_outside_brain  — inference-guided: zero background pixels
                             outside the brain mask only
  2. remove_global         — zero all background pixels in the full stack
  3. fill_outside_brain    — fill outside-brain pixels with random samples
                             drawn from the background pixel distribution
"""

import numpy as np


def _estimate_background(volume, brain_mask):
    """
    Estimate background from the mode of brain pixels (post-skin-removal).

    Only pixels inside the brain mask are used for the probe, giving a clean
    estimate of the background within the brain region.

    Returns bg_values (pool of background pixels), bg_median, bg_min, bg_max.
    """
    brain_pixels = volume[brain_mask.astype(bool)].ravel()
    hist, edges  = np.histogram(brain_pixels, bins=1000)
    bg_mode      = float(edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2)
    bg_values    = brain_pixels[brain_pixels <= bg_mode]
    print(f"   Background probe (inside brain): mode={bg_mode:.2f}"
          f"  ({len(bg_values):,} voxels = {100.*len(bg_values)/volume.size:.1f}% of stack)")
    return bg_values, bg_mode, bg_mode, bg_mode


def _threshold(volume, brain_mask, tolerance_pct=0.05):
    """
    Compute the background threshold and background mask.

    threshold = bg_max + tolerance  (tolerance can be negative)
    background mask = pixels <= threshold

    Returns bg_values, bg_max, threshold, bg_mask
    """
    bg_values, bg_median, bg_min, bg_max = _estimate_background(volume, brain_mask)
    data_range = float(volume.max()) - float(volume.min())
    tol        = data_range * (tolerance_pct / 100.0)
    thresh     = bg_max + tol
    bg_mask    = volume <= thresh
    return bg_values, bg_max, thresh, bg_mask


# ------------------------------------------------------------------ #
# Mode 1 — inference-guided removal (outside brain only)
# ------------------------------------------------------------------ #

def remove_outside_brain(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    tolerance_pct: float = 0.05,
):
    """
    Zero background pixels OUTSIDE the brain mask.
    Pixels inside the brain are always preserved.
    """
    bg_values, bg_max, thresh, bg_mask = _threshold(volume, brain_mask, tolerance_pct)
    outside   = ~brain_mask.astype(bool)
    to_zero   = outside & bg_mask
    n_removed = int(to_zero.sum())

    print(f"   Threshold: {thresh:.2f}  (tol={tolerance_pct:+.3f}%)")
    print(f"   Removed {n_removed:,} outside-brain background voxels"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[to_zero] = 0
    return result, bg_max, thresh, n_removed


# ------------------------------------------------------------------ #
# Mode 2 — global threshold removal (whole stack)
# ------------------------------------------------------------------ #

def remove_global(
    volume: np.ndarray,
    brain_mask: np.ndarray,
    tolerance_pct: float = 0.05,
):
    """
    Zero ALL pixels in the full stack at or below the background threshold.
    Negative tolerance shrinks the threshold (removes less),
    positive tolerance raises it (removes more).
    """
    bg_values, bg_max, thresh, bg_mask = _threshold(volume, brain_mask, tolerance_pct)
    n_removed = int(bg_mask.sum())

    print(f"   Threshold: {thresh:.2f}  (tol={tolerance_pct:+.3f}%)")
    print(f"   Removed {n_removed:,} voxels globally"
          f"  ({100.*n_removed/volume.size:.1f}% of stack)")

    result = volume.copy()
    result[bg_mask] = 0
    return result, bg_max, thresh, n_removed


# ------------------------------------------------------------------ #
# Mode 3 — random background fill (outside brain only)
# ------------------------------------------------------------------ #

def fill_outside_brain_random(
    volume: np.ndarray,
    brain_mask: np.ndarray,
):
    """
    Fill all pixels zeroed by skin removal (outside the brain mask) with
    random samples drawn from the background pixel distribution
    (median ± 10%, Gaussian-filtered at ±2σ to remove outliers).

    Inside brain  → original pixel values preserved
    Outside brain → random samples from background distribution
    """
    bg_values, bg_mode, _, _ = _estimate_background(volume, brain_mask)

    # Fill range: mode ± 0.10% of data range
    data_range = float(volume.max()) - float(volume.min())
    tol        = data_range * 0.001   # 0.10%
    low_fill   = bg_mode - tol
    high_fill  = bg_mode + tol

    outside  = ~brain_mask.astype(bool)
    n_filled = int(outside.sum())
    print(f"   Fill range: [{low_fill:.2f}, {high_fill:.2f}]"
          f"  (mode={bg_mode:.2f} ± 0.10%)")
    print(f"   Filling {n_filled:,} outside-brain voxels"
          f"  ({100.*n_filled/volume.size:.1f}% of stack)")

    result = volume.copy()
    if n_filled > 0:
        random_fill = np.random.uniform(low_fill, high_fill, size=n_filled)
        result[outside] = random_fill.astype(volume.dtype)
    return result, n_filled
