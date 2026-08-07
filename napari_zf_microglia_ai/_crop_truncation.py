"""
_crop_truncation.py — clean incidental-neighbor truncation in
generate_xzyz_patches.py-style training crops (see _xzyz_patches.py).

A crop framed around one target cell often also grazes the corner of a
different, nearby cell purely by chance -- sometimes showing only a
tiny sliver of it. Left as-is, that sliver is still a valid-looking
training label with a wildly wrong flow-center target (Cellpose points
flows toward each labeled object's own centroid; a fragment's visible
centroid is nowhere near the true cell's center). This zeros out any
label in any crop whose visible pixel count is below a threshold
(default 90%) of its TRUE full-slice cross-section, looked up from the
fish's own full GT volume with the same Z-stretch transform used at
crop-generation time for XZ/YZ orientations -- fixing exactly this,
without touching crop framing itself. A crop's own intended target
cell is essentially never affected by this (it's already
near-complete in the crop it was generated for); this specifically
catches incidental neighbors.

This is the exact fix already applied by hand to this project's real
training data (D1F1/D1F2/D1F4, 2026-08-05: 1,348/5,673 (crop,label)
pairs across 1,058/2,880 crop files) -- ported into the plugin instead
of remaining a one-off research script.
"""

import re
import shutil
from pathlib import Path

import numpy as np
import tifffile

from ._xzyz_patches import stretch_z_mask

_STEM_RE = re.compile(r"^(xy|xz|yz)_(\d+)_(\d+)$")


def find_crop_pairs(crop_dir):
    """
    Return [(stem, img_path, mask_path), ...] for every
    <stem>.tif / <stem>_masks.tif pair in crop_dir matching the
    {xy,xz,yz}_NNN_NN naming convention from _xzyz_patches.py.
    """
    crop_dir = Path(crop_dir)
    pairs = []
    for mask_path in sorted(crop_dir.glob("*_masks.tif")):
        stem = mask_path.name[:-len("_masks.tif")]
        if not _STEM_RE.match(stem):
            continue
        img_path = crop_dir / f"{stem}.tif"
        if img_path.exists():
            pairs.append((stem, img_path, mask_path))
    return pairs


def clean_crop_truncation(crop_dir, gt_full_path, anisotropy, threshold=0.9,
                           backup_suffix="_pretrunc_backup",
                           progress_cb=None, cancel_event=None):
    """
    For every crop mask in crop_dir (xy/xz/yz_NNN_NN naming, see
    find_crop_pairs), zero out any label whose crop-visible pixel count
    is below threshold (default 90%) of its true full-slice
    cross-section in gt_full_path.

    Backs up crop_dir to <crop_dir><backup_suffix> first (a plain
    directory copy -- these crop folders are typically not
    git-tracked) -- skipped if that backup already exists, so re-runs
    don't clobber an earlier, still-valid backup with already-modified
    files.

    cancel_event, if given, is checked once per crop pair.

    Returns dict: {n_files_scanned, n_files_modified, n_labels_zeroed,
    backup_dir, cancelled}.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    crop_dir = Path(crop_dir)
    backup_dir = crop_dir.parent / f"{crop_dir.name}{backup_suffix}"
    if not backup_dir.exists():
        _report(f"backing up {crop_dir} -> {backup_dir} ...")
        shutil.copytree(crop_dir, backup_dir)
    else:
        _report(f"backup already exists at {backup_dir}, not overwriting.")

    _report("loading full GT volume ...")
    gt_full = tifffile.imread(gt_full_path).astype(np.int32)

    # Cache stretched full-slice cross-sections per (orientation, slice_idx)
    # -- multiple crops commonly share the same source slice.
    slice_cache = {}

    def _true_slice(orientation, slice_idx):
        key = (orientation, slice_idx)
        if key in slice_cache:
            return slice_cache[key]
        if orientation == "xy":
            true_slice = gt_full[slice_idx]
        elif orientation == "xz":
            true_slice = stretch_z_mask(gt_full[:, slice_idx, :], anisotropy)
        else:  # yz
            true_slice = stretch_z_mask(gt_full[:, :, slice_idx], anisotropy)
        slice_cache[key] = true_slice
        return true_slice

    pairs = find_crop_pairs(crop_dir)
    _report(f"{len(pairs)} crop pairs found.")

    n_modified = 0
    n_zeroed = 0
    n_scanned = 0
    cancelled = False
    for stem, img_path, mask_path in pairs:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            _report(f"cancelled after {n_scanned}/{len(pairs)}.")
            break

        m = _STEM_RE.match(stem)
        orientation, slice_idx = m.group(1), int(m.group(2))
        true_slice = _true_slice(orientation, slice_idx)

        crop_mask = tifffile.imread(mask_path).astype(np.int32)
        labels_present = np.unique(crop_mask[crop_mask > 0])
        modified = False
        for label_id in labels_present:
            visible = int((crop_mask == label_id).sum())
            true_count = int((true_slice == label_id).sum())
            if true_count == 0:
                continue  # not findable in the true slice -- leave alone, don't guess
            if visible < threshold * true_count:
                crop_mask[crop_mask == label_id] = 0
                n_zeroed += 1
                modified = True
        if modified:
            tifffile.imwrite(mask_path, crop_mask)
            n_modified += 1
        n_scanned += 1
        if progress_cb and n_scanned % 50 == 0:
            _report(f"{n_scanned}/{len(pairs)} scanned, {n_modified} files modified so far, {n_zeroed} labels zeroed")

    _report(f"done — {n_modified}/{n_scanned} files modified, {n_zeroed} labels zeroed")
    return dict(n_files_scanned=n_scanned, n_files_modified=n_modified,
                n_labels_zeroed=n_zeroed, backup_dir=backup_dir, cancelled=cancelled)
