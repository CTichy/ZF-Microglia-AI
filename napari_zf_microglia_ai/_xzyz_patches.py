"""
_xzyz_patches.py — ported from microglia_segmentation/generate_xzyz_patches.py.

Generates 2D training crops (default 512x512) for Cellpose-SAM fine-
tuning across all three orientations: XY at native resolution, XZ/YZ
with the Z axis stretched by anisotropy (scipy.ndimage.zoom) to match
XY's pixel scale. Only crops containing GT signal are kept.

This is the crop-generation methodology this project has actually used
for every real Cellpose-SAM training run since May 2026
(train_cellpose_512, train_cellpose_512_multi, train_cellpose_512_multi3
-- including the branch-weighted-loss run this whole session has been
tracking) -- NOT the bbox-based single/double/triple/quadruple crops
_crop_extraction.py ports from the earlier (abandoned in April 2026)
extract_cellpose_crops.py approach. Both stay available in the plugin;
this one matches current practice.

The original script had no __main__ guard at all -- it ran top-to-
bottom at module import time. Refactored into generate_xzyz_patches()
so it's safe to import and call from the GUI. stretch_z/stretch_z_mask
and the random-crop sampling logic are otherwise unchanged from the
original -- _crop_truncation.py depends on stretch_z_mask producing
exactly the same transform used here, at generation time, to correctly
look up a crop's true full-slice cross-section later.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom


def stretch_z(vol_2d, anisotropy):
    """Bilinear Z-axis (first axis) upsample of an image slice by anisotropy."""
    return zoom(vol_2d.astype(np.float32), (anisotropy, 1.0), order=1)


def stretch_z_mask(vol_2d, anisotropy):
    """Nearest-neighbour Z-axis upsample of an integer mask slice by anisotropy."""
    return zoom(vol_2d.astype(np.float32), (anisotropy, 1.0), order=0).astype(np.int32)


def _random_crops(img_slice, gt_slice, crop_size, n_crops, rng, min_gt_pixels):
    """
    Extract up to n_crops random crop_size x crop_size patches from a 2D
    slice, anchored on GT-positive pixels. Identical logic to the
    original script's random_crops().
    """
    H, W = img_slice.shape[:2]
    if H < crop_size or W < crop_size:
        return []

    gt_ys, gt_xs = np.where(gt_slice > 0)
    if len(gt_ys) == 0:
        return []

    crops = []
    seen = set()
    max_attempts = n_crops * 30

    for _ in range(max_attempts):
        if len(crops) >= n_crops:
            break
        idx = int(rng.integers(len(gt_ys)))
        cy, cx = int(gt_ys[idx]), int(gt_xs[idx])
        offset_y = int(rng.integers(0, crop_size))
        offset_x = int(rng.integers(0, crop_size))
        ly = max(0, min(cy - offset_y, H - crop_size))
        lx = max(0, min(cx - offset_x, W - crop_size))
        pos = (ly, lx)
        if pos in seen:
            continue
        seen.add(pos)
        gt_crop = gt_slice[ly:ly + crop_size, lx:lx + crop_size]
        if np.count_nonzero(gt_crop) < min_gt_pixels:
            continue
        img_crop = img_slice[ly:ly + crop_size, lx:lx + crop_size]
        crops.append((img_crop, gt_crop))

    return crops


def generate_xzyz_patches(image_path, gt_path, out_dir, anisotropy,
                           crop_size=512, ncrops_per_slice=5,
                           max_per_orientation=320, min_gt_pixels=10,
                           seed=42, progress_cb=None, cancel_event=None):
    """
    Generates training crops in all three orientations, saving
    <out_dir>/{xy,xz,yz}_{slice:03d or 04d}_{i:02d}.tif +
    _masks.tif pairs -- identical filenames/logic to
    generate_xzyz_patches.py (see module docstring), so downstream tools
    that parse this naming convention (_crop_truncation.py, and any
    existing train_xzyz.py dataset) work unchanged.

    Matches the original script's two-phase per-orientation behavior
    exactly: collect all qualifying crops for an orientation first, cap
    to max_per_orientation, only then write files -- not an early-exit
    mid-slice that could overshoot the cap.

    cancel_event, if given, is checked once per orientation (not
    per-slice) -- each orientation's own generation loop already
    completes quickly relative to the whole run.

    Returns dict: {n_xy, n_xz, n_yz, out_dir, cancelled}.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    _report("loading image + GT ...")
    img = tifffile.imread(image_path)
    gt = tifffile.imread(gt_path).astype(np.int32)
    Z, Y, X = gt.shape
    _report(f"image {img.shape}, GT {gt.shape}, {len(np.unique(gt)) - 1} cells")

    # ── XY (native resolution) ──────────────────────────────────────── #
    _report("XY slices (native resolution) ...")
    xy_signal = np.array([z for z in range(Z) if gt[z].any()])
    rng.shuffle(xy_signal)
    all_xy_crops = []
    for z in xy_signal:
        crops = _random_crops(img[z], gt[z], crop_size, ncrops_per_slice, rng, min_gt_pixels)
        all_xy_crops.extend([(z, i, ic, gc) for i, (ic, gc) in enumerate(crops)])
        if len(all_xy_crops) >= max_per_orientation:
            break
    all_xy_crops = all_xy_crops[:max_per_orientation]
    for z, i, ic, gc in all_xy_crops:
        stem = f"xy_{z:03d}_{i:02d}"
        tifffile.imwrite(out_dir / f"{stem}.tif", ic.astype(np.uint16))
        tifffile.imwrite(out_dir / f"{stem}_masks.tif", gc.astype(np.int32))
    n_xy = len(all_xy_crops)
    _report(f"XY done: {n_xy} crops")
    if _cancelled():
        return dict(n_xy=n_xy, n_xz=0, n_yz=0, out_dir=out_dir, cancelled=True)

    # ── XZ (Z-stretched) ────────────────────────────────────────────── #
    _report("XZ slices (Z-stretched) ...")
    xz_signal = np.array([y for y in range(Y) if gt[:, y, :].any()])
    rng.shuffle(xz_signal)
    all_xz_crops = []
    for y in xz_signal:
        img_xz = stretch_z(img[:, y, :], anisotropy).astype(np.uint16)
        gt_xz = stretch_z_mask(gt[:, y, :], anisotropy)
        crops = _random_crops(img_xz, gt_xz, crop_size, ncrops_per_slice, rng, min_gt_pixels)
        all_xz_crops.extend([(y, i, ic, gc) for i, (ic, gc) in enumerate(crops)])
        if len(all_xz_crops) >= max_per_orientation:
            break
    all_xz_crops = all_xz_crops[:max_per_orientation]
    for y, i, ic, gc in all_xz_crops:
        stem = f"xz_{y:04d}_{i:02d}"
        tifffile.imwrite(out_dir / f"{stem}.tif", ic.astype(np.uint16))
        tifffile.imwrite(out_dir / f"{stem}_masks.tif", gc.astype(np.int32))
    n_xz = len(all_xz_crops)
    _report(f"XZ done: {n_xz} crops")
    if _cancelled():
        return dict(n_xy=n_xy, n_xz=n_xz, n_yz=0, out_dir=out_dir, cancelled=True)

    # ── YZ (Z-stretched) ────────────────────────────────────────────── #
    _report("YZ slices (Z-stretched) ...")
    yz_signal = np.array([x for x in range(X) if gt[:, :, x].any()])
    rng.shuffle(yz_signal)
    all_yz_crops = []
    for x in yz_signal:
        img_yz = stretch_z(img[:, :, x], anisotropy).astype(np.uint16)
        gt_yz = stretch_z_mask(gt[:, :, x], anisotropy)
        crops = _random_crops(img_yz, gt_yz, crop_size, ncrops_per_slice, rng, min_gt_pixels)
        all_yz_crops.extend([(x, i, ic, gc) for i, (ic, gc) in enumerate(crops)])
        if len(all_yz_crops) >= max_per_orientation:
            break
    all_yz_crops = all_yz_crops[:max_per_orientation]
    for x, i, ic, gc in all_yz_crops:
        stem = f"yz_{x:04d}_{i:02d}"
        tifffile.imwrite(out_dir / f"{stem}.tif", ic.astype(np.uint16))
        tifffile.imwrite(out_dir / f"{stem}_masks.tif", gc.astype(np.int32))
    n_yz = len(all_yz_crops)
    _report(f"YZ done: {n_yz} crops")

    _report(f"done — {n_xy + n_xz + n_yz} total crops in {out_dir}")
    return dict(n_xy=n_xy, n_xz=n_xz, n_yz=n_yz, out_dir=out_dir, cancelled=False)
