"""
_xzyz_patches.py — ported from microglia_segmentation/generate_xzyz_patches.py.

Generates 2D training crops (default 512x512) for Cellpose-SAM fine-
tuning across all three orientations: XY at native resolution, XZ/YZ
with the Z axis stretched by anisotropy (scipy.ndimage.zoom) to match
XY's pixel scale. Cell crops are anchored on GT cells; when the GT
carries the skin label (-1), cell-skin boundary and skin-only crops are
added (see generate_xzyz_patches).

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
from scipy.ndimage import binary_dilation, zoom


# Crop-index offsets in the {xy,xz,yz}_<slice>_<index> stem: cell crops use
# 0..ncrops_per_slice-1, skin crops start here. train_xzyz.py keeps an
# empty-mask crop only when its index is >= SKIN_ONLY_INDEX_OFFSET.
BOUNDARY_INDEX_OFFSET = 100
SKIN_ONLY_INDEX_OFFSET = 200


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
        if np.count_nonzero(gt_crop > 0) < min_gt_pixels:
            continue
        img_crop = img_slice[ly:ly + crop_size, lx:lx + crop_size]
        crops.append((img_crop, gt_crop))

    return crops


def _anchored_crops(img_slice, gt_slice, anchor_mask, crop_size, n_crops, rng,
                    accept):
    """
    Up to n_crops random crop_size x crop_size patches whose window
    contains a pixel of anchor_mask (same placement rule as
    _random_crops). accept(gt_crop) decides whether a window is kept.
    """
    H, W = img_slice.shape[:2]
    if H < crop_size or W < crop_size:
        return []
    ys, xs = np.where(anchor_mask)
    if len(ys) == 0:
        return []

    crops = []
    seen = set()
    for _ in range(n_crops * 30):
        if len(crops) >= n_crops:
            break
        idx = int(rng.integers(len(ys)))
        cy, cx = int(ys[idx]), int(xs[idx])
        offset_y = int(rng.integers(0, crop_size))
        offset_x = int(rng.integers(0, crop_size))
        ly = max(0, min(cy - offset_y, H - crop_size))
        lx = max(0, min(cx - offset_x, W - crop_size))
        if (ly, lx) in seen:
            continue
        seen.add((ly, lx))
        gt_crop = gt_slice[ly:ly + crop_size, lx:lx + crop_size]
        if not accept(gt_crop):
            continue
        crops.append((img_slice[ly:ly + crop_size, lx:lx + crop_size], gt_crop))
    return crops


def _skin_crops(get_slice, slice_ids, crop_size, ncrops_per_slice, rng,
                min_gt_pixels, n_boundary, n_skin_only, touch_px):
    """
    Skin crops for one orientation, from a GT that carries the skin
    label (-1). get_slice(i) -> (img_2d, gt_2d) for slice i.

    Boundary crops: anchored on cell pixels within touch_px of skin; the
    window must hold >= min_gt_pixels cell pixels and some skin.
    Skin-only crops: anchored on skin pixels; the window must hold
    >= min_gt_pixels skin pixels and no cell pixel at all.

    Returns ([(slice, index, img, gt), ...] boundary,
             [(slice, index, img, gt), ...] skin-only).
    """
    boundary, skin_only = [], []
    struct = np.ones((3, 3), dtype=bool)
    for i in slice_ids:
        if len(boundary) >= n_boundary and len(skin_only) >= n_skin_only:
            break
        img_s, gt_s = get_slice(i)
        skin = gt_s < 0
        if not skin.any():
            continue
        cells = gt_s > 0
        if len(boundary) < n_boundary and cells.any():
            near_skin = binary_dilation(skin, structure=struct, iterations=touch_px)
            crops = _anchored_crops(
                img_s, gt_s, cells & near_skin, crop_size, ncrops_per_slice, rng,
                lambda g: (np.count_nonzero(g > 0) >= min_gt_pixels
                           and bool((g < 0).any())),
            )
            boundary.extend((i, BOUNDARY_INDEX_OFFSET + k, ic, gc)
                            for k, (ic, gc) in enumerate(crops))
        if len(skin_only) < n_skin_only:
            crops = _anchored_crops(
                img_s, gt_s, skin, crop_size, ncrops_per_slice, rng,
                lambda g: (not (g > 0).any()
                           and np.count_nonzero(g < 0) >= min_gt_pixels),
            )
            skin_only.extend((i, SKIN_ONLY_INDEX_OFFSET + k, ic, gc)
                             for k, (ic, gc) in enumerate(crops))
    return boundary[:n_boundary], skin_only[:n_skin_only]


def generate_xzyz_patches(image_path, gt_path, out_dir, anisotropy,
                           crop_size=512, ncrops_per_slice=5,
                           max_per_orientation=320, min_gt_pixels=10,
                           seed=42, skin_crops=True, boundary_fraction=0.20,
                           skin_only_fraction=0.10, touch_px=3,
                           progress_cb=None, cancel_event=None):
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

    Skin crops (skin_crops=True and the GT carries the skin label -1):
    per orientation, on top of the cell crops, round(boundary_fraction x
    n_cell_crops) boundary crops (a cell within touch_px of skin, plus
    that skin) and round(skin_only_fraction x n_cell_crops) skin-only
    crops (skin, no cell -- an empty mask). They use their own random
    generator (seed + 1), so the cell crops are the same with or without
    them. Crop indices start at BOUNDARY_INDEX_OFFSET /
    SKIN_ONLY_INDEX_OFFSET. Every mask crop is written with negative
    labels as 0 (background for Cellpose); the GT file is not changed.

    cancel_event, if given, is checked once per orientation (not
    per-slice) -- each orientation's own generation loop already
    completes quickly relative to the whole run.

    Returns dict: {n_xy, n_xz, n_yz, n_boundary, n_skin_only,
    skin_label_found, out_dir, cancelled}.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rng_skin = np.random.default_rng(seed + 1)

    _report("loading image + GT ...")
    img = tifffile.imread(image_path)
    gt = tifffile.imread(gt_path).astype(np.int32)
    Z, Y, X = gt.shape
    n_cells = len(np.unique(gt[gt > 0]))
    n_neg = int(np.count_nonzero(gt < 0))
    skin_label_found = n_neg > 0
    do_skin = skin_crops and skin_label_found
    _report(f"image {img.shape}, GT {gt.shape}, {n_cells} cells, "
            f"{n_neg} skin voxels (-1)"
            + ("" if do_skin else
               " -- no skin crops" + ("" if skin_crops else " (off)")))

    def _write(prefix, width, crops):
        for i, k, ic, gc in crops:
            stem = f"{prefix}_{i:0{width}d}_{k:02d}"
            tifffile.imwrite(out_dir / f"{stem}.tif", ic.astype(np.uint16))
            tifffile.imwrite(out_dir / f"{stem}_masks.tif",
                             np.where(gc < 0, 0, gc).astype(np.int32))

    # Each orientation: (prefix, slice-number width, slice count, getter)
    def _xy(z):
        return img[z], gt[z]

    def _xz(y):
        return (stretch_z(img[:, y, :], anisotropy).astype(np.uint16),
                stretch_z_mask(gt[:, y, :], anisotropy))

    def _yz(x):
        return (stretch_z(img[:, :, x], anisotropy).astype(np.uint16),
                stretch_z_mask(gt[:, :, x], anisotropy))

    orientations = (
        ("xy", 3, "XY slices (native resolution)", Z, _xy,
         lambda z: (gt[z] > 0).any(), lambda z: (gt[z] < 0).any()),
        ("xz", 4, "XZ slices (Z-stretched)", Y, _xz,
         lambda y: (gt[:, y, :] > 0).any(), lambda y: (gt[:, y, :] < 0).any()),
        ("yz", 4, "YZ slices (Z-stretched)", X, _yz,
         lambda x: (gt[:, :, x] > 0).any(), lambda x: (gt[:, :, x] < 0).any()),
    )

    counts = dict(n_xy=0, n_xz=0, n_yz=0, n_boundary=0, n_skin_only=0)
    for prefix, width, title, n_slices, get_slice, has_cells, has_skin in orientations:
        _report(f"{title} ...")
        signal = np.array([i for i in range(n_slices) if has_cells(i)])
        rng.shuffle(signal)
        cell_crops = []
        for i in signal:
            img_s, gt_s = get_slice(i)
            crops = _random_crops(img_s, gt_s, crop_size, ncrops_per_slice, rng, min_gt_pixels)
            cell_crops.extend([(i, k, ic, gc) for k, (ic, gc) in enumerate(crops)])
            if len(cell_crops) >= max_per_orientation:
                break
        cell_crops = cell_crops[:max_per_orientation]
        _write(prefix, width, cell_crops)
        counts[f"n_{prefix}"] = len(cell_crops)
        msg = f"{prefix.upper()} done: {len(cell_crops)} crops"

        if do_skin:
            n_b = int(round(boundary_fraction * len(cell_crops)))
            n_s = int(round(skin_only_fraction * len(cell_crops)))
            skin_ids = np.array([i for i in range(n_slices) if has_skin(i)])
            rng_skin.shuffle(skin_ids)
            boundary, skin_only = _skin_crops(
                get_slice, skin_ids, crop_size, ncrops_per_slice, rng_skin,
                min_gt_pixels, n_b, n_s, touch_px)
            _write(prefix, width, boundary)
            _write(prefix, width, skin_only)
            counts["n_boundary"] += len(boundary)
            counts["n_skin_only"] += len(skin_only)
            msg += (f" + {len(boundary)}/{n_b} cell-skin boundary"
                    f" + {len(skin_only)}/{n_s} skin-only")
        _report(msg)
        if _cancelled():
            return dict(**counts, skin_label_found=skin_label_found,
                        out_dir=out_dir, cancelled=True)

    n_total = sum(counts.values())
    _report(f"done — {n_total} total crops in {out_dir}")
    return dict(**counts, skin_label_found=skin_label_found,
                out_dir=out_dir, cancelled=False)
