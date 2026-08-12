"""
_pixel_sweep.py — GT-verified BG Threshold x Signal Erosion sweep for
microglia labels produced by the Pixel Classifier path (Tab 1 background
removal + Tab 2 Create Labels / union-find), scored the same way as
_epoch_sweep.py's Cellpose-SAM checkpoint sweep: crop to the N most
morphologically complex GT cells, best-IoU-match the resulting labels
against GT, average.

Deliberately does NOT re-run MONAI inference per grid point -- neither
Signal Erosion nor BG Threshold change what MONAI predicts, only how
that prediction gets turned into brain_only and then labels. Takes a
pre-computed, un-eroded brain_mask.tif (produced once via a normal Tab 1
run) as input instead -- MONAI Erosion is out of scope for this sweep
entirely, same as for a real Tab 1 run's Background mode 2 step. This
makes the whole sweep union-find-bound, not GPU-inference-bound -- a
full grid finishes in minutes, not hours, and doesn't need a GPU at all
(Create Labels already has a CPU fallback).

Uses Background mode 2 ("Remove globally") -- the documented recommended
setting before labeling -- since that's the mode this sweep validates.
Mirrors _background.py's remove_global() threshold + signal-erosion math
by hand (erode what survives the intensity threshold, not the probe
mask feeding it) rather than calling it directly, since this sweep
operates on small padded crops instead of full volumes -- see
remove_global()'s signal_erosion_voxels docstring for why eroding the
probe/brain-boundary mask instead has no real effect and was the wrong
design the first time this sweep was written.
"""

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, find_objects

from ._background import _threshold as _bg_threshold
from ._epoch_sweep import find_complex_cells, bbox_crop, _best_gt_match
from ._labeling import create_labels

_DEFAULT_MIN_VOLUME = 7500  # fallback only -- see min_volume_from_gt()
_DEFAULT_MIN_HOLE_SIZE = 20  # fallback only -- see min_hole_size_from_gt()


def min_hole_size_from_gt(gt_labels, min_hole_size_to_trust=5):
    """Recommended per-slice hole-fill cutoff (voxels) for create_labels()'s
    min_hole_size parameter, measured directly from real GT rather than
    guessed.

    A hand-corrected GT cell that still has an internal, unlabeled gap in
    some Z-slice is exactly the kind of real biological hole min_hole_size
    is meant to protect -- a human looked at that gap and deliberately did
    not label it as part of the cell. The smallest such real gap found
    anywhere in this GT sets the floor directly: min_hole_size should be
    no larger than it, so create_labels() never risks erasing a hole a
    real annotator confirmed as genuine background. This mirrors
    min_volume_from_gt()'s never-guess-when-you-can-measure logic exactly,
    just applied to holes instead of whole objects -- and, like
    min_volume, names the size a region must clear to survive as real,
    not the size at which it gets discarded.

    min_hole_size_to_trust discards any hole strictly below this size
    before taking the minimum. This matters in practice: real GT checked
    during development showed a sharp bimodal split, several 1-2 voxel
    gaps (near-certainly single pixels missed while manually painting a
    cell, not deliberate holes) alongside a cluster of 400+ voxel gaps
    (clearly real structure) -- nothing in between. Trusting every hole
    GT reports, including single-pixel annotation slips, collapsed the
    recommended cutoff to 0 and made the feature useless. Treating
    anything below min_hole_size_to_trust as annotation noise rather than
    evidence fixes this without needing a per-fish judgment call.

    Returns _DEFAULT_MIN_HOLE_SIZE if no GT cell has any internal hole at
    or above min_hole_size_to_trust in any slice -- either because the
    cells are genuinely solid, or because every hole found was itself
    below the trust threshold -- in which case there is nothing to
    measure and a small, conservative guess is used instead."""
    from scipy.ndimage import binary_fill_holes, label as _cc_label

    real_hole_sizes = []
    objs = find_objects(gt_labels)
    for lbl in np.unique(gt_labels[gt_labels > 0]):
        sl = objs[int(lbl) - 1]
        if sl is None:
            continue
        for z in range(sl[0].start, sl[0].stop):
            cell_slice = gt_labels[z, sl[1], sl[2]] == lbl
            if not cell_slice.any():
                continue
            filled = binary_fill_holes(cell_slice)
            holes = filled & ~cell_slice
            if not holes.any():
                continue
            hole_labels, n_holes = _cc_label(holes)
            if n_holes == 0:
                continue
            counts = np.bincount(hole_labels.ravel())[1:]  # drop background bin
            real_hole_sizes.extend(
                int(c) for c in counts if c >= min_hole_size_to_trust
            )

    if not real_hole_sizes:
        return _DEFAULT_MIN_HOLE_SIZE
    return min(real_hole_sizes)


def min_volume_from_gt(gt_labels):
    """Smallest true voxel volume among the labeled cells in gt_labels.

    min_volume (the small-blob cleanup threshold Create Labels drops
    after union-find) used to always default to a single hardcoded
    constant (7500) regardless of which fish was being processed -- a
    guess, not a measurement, and not necessarily right for a fish whose
    real microglia run smaller or larger than whatever fish that number
    happened to come from. Since a real gt_labels volume is already an
    input to this sweep, there's no reason to guess: the true smallest
    labeled cell in the GT itself is exactly the number that should
    never be discarded, so it's measured directly here every time,
    mirroring _krendl_sweep.gt_min_from_labels()'s identical fix for the
    same category of problem (Krendl safe-merge's gt_min parameter).

    Falls back to _DEFAULT_MIN_VOLUME if gt_labels has no labeled cells
    at all (degenerate input, shouldn't happen in practice)."""
    _, counts = np.unique(gt_labels[gt_labels > 0], return_counts=True)
    if len(counts) == 0:
        return _DEFAULT_MIN_VOLUME
    return int(counts.min())


def run_pixel_sweep(image_path, brain_mask_path, gt_labels_path,
                     bg_thresholds, erosions, scale_zyx,
                     sigma_xy=1.5, sigma_z=3.0, min_volume=None,
                     min_hole_size=None, final_min_fraction=0.618,
                     n_cells=5, pad_z=15, pad_xy=40,
                     progress_cb=None, cancel_event=None):
    """
    Sweep every (bg_threshold, erosion) combination in the given lists,
    scoring the resulting Pixel Classifier labels against the N most
    complex GT cells.

    image_path        : raw/original volume (same one Tab 1 ran on)
    brain_mask_path    : the RAW (un-eroded) brain_mask.tif from a prior
                         Tab 1 run -- MONAI inference itself is not
                         re-run here
    gt_labels_path     : corrected GT microglia label volume
    bg_thresholds       : list of BG Threshold values to sweep (same
                         units as Tab 1's BG Threshold field --
                         percent-of-data-range tolerance)
    erosions            : list of Signal Erosion radii (voxels) to sweep --
                         erodes what survives the BG Threshold, not the
                         brain mask itself
    scale_zyx           : (Z, Y, X) um/voxel -- drives complexity ranking
    sigma_xy/sigma_z    : held fixed, passed straight to create_labels()
                         (same defaults as Tab 2)
    min_volume          : small-blob cleanup threshold passed to
                         create_labels(). If None (default), measured
                         from gt_labels itself via min_volume_from_gt()
                         instead of guessed -- see that function's
                         docstring. Pass an explicit value to override.
    min_hole_size        : per-slice hole-fill floor passed to
                         create_labels(). If None (default), measured
                         from gt_labels itself via
                         min_hole_size_from_gt() instead of guessed.
                         Pass an explicit value to override.
    final_min_fraction  : passed straight to create_labels() -- the
                         actual small-blob deletion cutoff used there is
                         final_min_fraction * min_volume, not min_volume
                         itself. Default 0.618 (golden ratio) matches
                         Common Settings' Tab 2 field and Cellpose-SAM's
                         own final_min_size_cleanup() default, so this
                         sweep is validated against the exact same
                         cutoff production actually uses.

    progress_cb(str), if given, is called with a one-line status message
    as work proceeds. cancel_event (threading.Event), if given, is
    checked between erosion values and between bg_threshold values --
    stops early and returns whatever grid points completed, same
    partial-results contract as _epoch_sweep.run_epoch_sweep.

    Returns dict: {
      'cells': [label_id, ...],                      # most-complex-first
      'grid': [(bg_threshold, erosion), ...],         # completed points
      'results': {(bg_threshold, erosion, label_id): {...}},
      'per_point_avg': {(bg_threshold, erosion): {'iou': x, 'dice': y}},
      'best_point': (bg_threshold, erosion) or None,  # highest average IoU
      'min_volume_used': int,                         # see min_volume above
      'min_hole_size_used': int,                      # see min_hole_size above
      'cancelled': bool,
    }
    """
    image = tifffile.imread(image_path)
    brain_mask = tifffile.imread(brain_mask_path).astype(bool)
    gt_labels = tifffile.imread(gt_labels_path).astype(np.int32)
    data_range = float(image.max()) - float(image.min())

    if min_volume is None:
        min_volume = min_volume_from_gt(gt_labels)
        if progress_cb:
            progress_cb(f"min_volume: measured {min_volume} vox from this GT's smallest labeled cell.")

    if min_hole_size is None:
        min_hole_size = min_hole_size_from_gt(gt_labels)
        if progress_cb:
            progress_cb(f"min_hole_size: measured {min_hole_size} vox from this GT's own real holes.")

    objs = find_objects(gt_labels)
    cells = find_complex_cells(gt_labels, scale_zyx, n_cells=n_cells, objs=objs)
    if not cells:
        raise ValueError("No labeled cells found in the GT volume.")

    crops = {}
    for label_id in cells:
        img_crop, gt_mask, gt_vox = bbox_crop(image, gt_labels, label_id, pad_z, pad_xy, objs=objs)
        mask_crop, _, _ = bbox_crop(brain_mask.astype(np.int32), gt_labels, label_id, pad_z, pad_xy, objs=objs)
        crops[label_id] = dict(img=img_crop, raw_mask=mask_crop.astype(bool),
                                gt_mask=gt_mask, gt_vox=gt_vox)
        if progress_cb:
            progress_cb(f"cell {label_id}: bbox shape={img_crop.shape}  gt_vox={gt_vox}")

    # Global background estimate -- a genuinely global quantity (histogram
    # over ALL brain-masked pixels in the full volume, not something safe
    # to compute from a small crop), and independent of Signal Erosion:
    # Signal Erosion no longer touches this probe, only what survives the
    # threshold afterward -- see remove_global()'s signal_erosion_voxels.
    _, bg_max, _, _ = _bg_threshold(image, brain_mask, tolerance_pct=0.0)
    if progress_cb:
        progress_cb(f"global bg_max={bg_max:.2f}")

    results = {}
    cancelled = False
    for bg_threshold in bg_thresholds:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        thresh = bg_max + data_range * (bg_threshold / 100.0)
        for erosion in erosions:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            for label_id in cells:
                c = crops[label_id]
                img_thresholded = np.where(c["img"] <= thresh, 0, c["img"])
                # Erode what survived the threshold (the signal), not the
                # brain-boundary probe mask -- matches remove_global().
                signal_mask = (img_thresholded > 0) & c["raw_mask"]
                if erosion > 0:
                    signal_mask = binary_erosion(signal_mask, iterations=erosion)
                brain_only_crop = (img_thresholded * signal_mask).astype(c["img"].dtype)
                pred_labels = create_labels(
                    brain_only_crop, sigma_xy=sigma_xy, sigma_z=sigma_z,
                    min_volume=min_volume, min_hole_size=min_hole_size,
                    final_min_fraction=final_min_fraction,
                )
                r = _best_gt_match(pred_labels, c["gt_mask"], c["gt_vox"])
                results[(bg_threshold, erosion, label_id)] = r
                if progress_cb:
                    progress_cb(
                        f"bg={bg_threshold}, erosion={erosion}, cell {label_id}: "
                        f"IoU={r['iou']:.1f}%  Dice={r['dice']:.1f}%  n_obj={r['n_obj']}"
                    )
        if cancelled:
            break

    grid = sorted({(bt, er) for (bt, er, _) in results.keys()})
    per_point_avg = {}
    for point in grid:
        bt, er = point
        vals = [results[(bt, er, c)] for c in cells if (bt, er, c) in results]
        if len(vals) == len(cells):
            per_point_avg[point] = dict(
                iou=sum(v["iou"] for v in vals) / len(vals),
                dice=sum(v["dice"] for v in vals) / len(vals),
            )
    best_point = max(per_point_avg, key=lambda k: per_point_avg[k]["iou"]) if per_point_avg else None
    scored_grid = [k for k in grid if k in per_point_avg]

    return dict(cells=cells, grid=scored_grid, results=results,
                per_point_avg=per_point_avg, best_point=best_point,
                min_volume_used=min_volume, min_hole_size_used=min_hole_size,
                cancelled=cancelled)


def run_sigma_sweep(image_path, brain_mask_path, gt_labels_path,
                     sigma_xy_values, sigma_z_values, scale_zyx,
                     bg_threshold, erosion, min_volume=None,
                     min_hole_size=None, final_min_fraction=0.618,
                     n_cells=5, pad_z=15, pad_xy=40,
                     progress_cb=None, cancel_event=None):
    """
    Sweep every (sigma_xy, sigma_z) combination in the given lists, scoring
    the resulting Pixel Classifier labels against the N most complex GT
    cells -- the Smooth sigma counterpart to run_pixel_sweep's BG
    Threshold x Signal Erosion sweep, with the roles reversed: BG
    Threshold and Signal Erosion are held fixed here (at whatever Tab 1
    is currently set to), and Smooth sigma XY/Z -- previously never swept
    at all, always left at whatever default (1.5/3.0) happened to be
    guessed -- is what varies.

    Cheaper per grid point than run_pixel_sweep: sigma only affects the
    create_labels() call, not the background-threshold/signal-erosion
    step that builds each cell's brain_only crop, so that crop is
    computed once per cell (not once per grid point) and every sigma
    combination reuses it.

    image_path/brain_mask_path/gt_labels_path : same as run_pixel_sweep
    sigma_xy_values/sigma_z_values : lists of Smooth sigma values to sweep
    scale_zyx      : (Z, Y, X) um/voxel -- drives complexity ranking
    bg_threshold/erosion : held fixed, same units as Tab 1's own fields
                     (erosion = Signal Erosion, not MONAI Erosion)
    min_volume     : small-blob cleanup threshold. If None (default),
                     measured from gt_labels via min_volume_from_gt()
                     instead of guessed.
    min_hole_size  : per-slice hole-fill floor. If None (default), measured
                     from gt_labels via min_hole_size_from_gt() instead
                     of guessed.
    final_min_fraction : passed straight to create_labels() -- see
                     run_pixel_sweep's docstring for the same parameter.

    progress_cb(str)/cancel_event : same contract as run_pixel_sweep.

    Returns dict: {
      'cells': [label_id, ...],
      'grid': [(sigma_xy, sigma_z), ...],
      'results': {(sigma_xy, sigma_z, label_id): {...}},
      'per_point_avg': {(sigma_xy, sigma_z): {'iou': x, 'dice': y}},
      'best_point': (sigma_xy, sigma_z) or None,
      'min_volume_used': int,
      'min_hole_size_used': int,
      'cancelled': bool,
    }
    """
    image = tifffile.imread(image_path)
    brain_mask = tifffile.imread(brain_mask_path).astype(bool)
    gt_labels = tifffile.imread(gt_labels_path).astype(np.int32)
    data_range = float(image.max()) - float(image.min())

    if min_volume is None:
        min_volume = min_volume_from_gt(gt_labels)
        if progress_cb:
            progress_cb(f"min_volume: measured {min_volume} vox from this GT's smallest labeled cell.")

    if min_hole_size is None:
        min_hole_size = min_hole_size_from_gt(gt_labels)
        if progress_cb:
            progress_cb(f"min_hole_size: measured {min_hole_size} vox from this GT's own real holes.")

    objs = find_objects(gt_labels)
    cells = find_complex_cells(gt_labels, scale_zyx, n_cells=n_cells, objs=objs)
    if not cells:
        raise ValueError("No labeled cells found in the GT volume.")

    # bg_threshold/erosion are fixed, so the global background estimate and
    # each cell's thresholded brain_only crop are each computed exactly
    # once here, not once per (sigma_xy, sigma_z) grid point. Signal
    # Erosion erodes what survives the threshold (the signal), not the
    # brain-boundary probe mask -- matches remove_global().
    _, bg_max, _, _ = _bg_threshold(image, brain_mask, tolerance_pct=0.0)
    thresh = bg_max + data_range * (bg_threshold / 100.0)
    if progress_cb:
        progress_cb(f"Fixed BG Threshold={bg_threshold}, Signal Erosion={erosion} -> bg_max={bg_max:.2f}")

    crops = {}
    for label_id in cells:
        img_crop, gt_mask, gt_vox = bbox_crop(image, gt_labels, label_id, pad_z, pad_xy, objs=objs)
        mask_crop, _, _ = bbox_crop(brain_mask.astype(np.int32), gt_labels, label_id, pad_z, pad_xy, objs=objs)
        raw_mask = mask_crop.astype(bool)
        img_thresholded = np.where(img_crop <= thresh, 0, img_crop)
        signal_mask = (img_thresholded > 0) & raw_mask
        if erosion > 0:
            signal_mask = binary_erosion(signal_mask, iterations=erosion)
        brain_only_crop = (img_thresholded * signal_mask).astype(img_crop.dtype)
        crops[label_id] = dict(brain_only=brain_only_crop, gt_mask=gt_mask, gt_vox=gt_vox)
        if progress_cb:
            progress_cb(f"cell {label_id}: bbox shape={img_crop.shape}  gt_vox={gt_vox}")

    results = {}
    cancelled = False
    for sigma_xy in sigma_xy_values:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        for sigma_z in sigma_z_values:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            for label_id in cells:
                c = crops[label_id]
                pred_labels = create_labels(
                    c["brain_only"], sigma_xy=sigma_xy, sigma_z=sigma_z,
                    min_volume=min_volume, min_hole_size=min_hole_size,
                    final_min_fraction=final_min_fraction,
                )
                r = _best_gt_match(pred_labels, c["gt_mask"], c["gt_vox"])
                results[(sigma_xy, sigma_z, label_id)] = r
                if progress_cb:
                    progress_cb(
                        f"sigma_xy={sigma_xy}, sigma_z={sigma_z}, cell {label_id}: "
                        f"IoU={r['iou']:.1f}%  Dice={r['dice']:.1f}%  n_obj={r['n_obj']}"
                    )
        if cancelled:
            break

    grid = sorted({(sx, sz) for (sx, sz, _) in results.keys()})
    per_point_avg = {}
    for point in grid:
        sx, sz = point
        vals = [results[(sx, sz, c)] for c in cells if (sx, sz, c) in results]
        if len(vals) == len(cells):
            per_point_avg[point] = dict(
                iou=sum(v["iou"] for v in vals) / len(vals),
                dice=sum(v["dice"] for v in vals) / len(vals),
            )
    best_point = max(per_point_avg, key=lambda k: per_point_avg[k]["iou"]) if per_point_avg else None
    scored_grid = [k for k in grid if k in per_point_avg]

    return dict(cells=cells, grid=scored_grid, results=results,
                per_point_avg=per_point_avg, best_point=best_point,
                min_volume_used=min_volume, min_hole_size_used=min_hole_size,
                cancelled=cancelled)


def format_sigma_sweep_report(sweep, current_sigma_xy=None, current_sigma_z=None):
    """Plain-text 2D grid report (rows = sigma Z, columns = sigma XY), same
    spirit as format_pixel_sweep_report."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    sigma_xys = sorted({sx for sx, _ in grid})
    sigma_zs = sorted({sz for _, sz in grid})

    header = f"{'sigmaZ':>8} | " + " | ".join(f"xy={sx:>5} " for sx in sigma_xys)
    lines = [header, "-" * len(header)]
    for sz in sigma_zs:
        row_cells = []
        for sx in sigma_xys:
            point = (sx, sz)
            if point in sweep["per_point_avg"]:
                row_cells.append(f"{sweep['per_point_avg'][point]['iou']:>8.1f}")
            else:
                row_cells.append(f"{'--':>8}")
        marker = "  <- current" if sz == current_sigma_z else ""
        lines.append(f"{sz:>8} | " + " | ".join(row_cells) + marker)
    lines.append("-" * len(header))
    lines.append("(values are average IoU% across the tested cells)")

    best = sweep["best_point"]
    if best is not None:
        best_sx, best_sz = best
        lines.append("")
        lines.append(
            f"Best: Smooth sigma XY={best_sx}, sigma Z={best_sz} "
            f"(avg IoU={sweep['per_point_avg'][best]['iou']:.1f}%, "
            f"avg Dice={sweep['per_point_avg'][best]['dice']:.1f}%)"
        )
        if current_sigma_xy is not None and current_sigma_z is not None:
            current = (current_sigma_xy, current_sigma_z)
            if current in sweep["per_point_avg"] and current != best:
                lines.append(
                    f"Current setting (sigma XY={current_sigma_xy}, sigma Z={current_sigma_z}): "
                    f"avg IoU={sweep['per_point_avg'][current]['iou']:.1f}% -- "
                    f"the sweep found a better combination above."
                )
            elif current == best:
                lines.append("Current setting matches the sweep's best -- confirmed.")

    if sweep.get("cancelled"):
        lines.append("\n(sweep was cancelled -- results above are partial.)")

    lines.append("")
    lines.append("Per-cell winner (grid point with highest IoU for that cell):")
    for c in sweep["cells"]:
        best_c = max(grid, key=lambda pt: sweep["results"][(pt[0], pt[1], c)]["iou"])
        sx_c, sz_c = best_c
        lines.append(
            f"  cell {c:>4}: best at sigma XY={sx_c}, sigma Z={sz_c} "
            f"(IoU={sweep['results'][(sx_c, sz_c, c)]['iou']:.1f}%)"
        )
    return "\n".join(lines)


def format_pixel_sweep_report(sweep, current_bg_threshold=None, current_erosion=None):
    """Plain-text 2D grid report (rows = erosion, columns = BG Threshold),
    same spirit as _epoch_sweep.format_sweep_report."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    bg_thresholds = sorted({bt for bt, _ in grid})
    erosions = sorted({er for _, er in grid})

    header = f"{'SigErode':>8} | " + " | ".join(f"bg={bt:>5} " for bt in bg_thresholds)
    lines = [header, "-" * len(header)]
    for er in erosions:
        row_cells = []
        for bt in bg_thresholds:
            point = (bt, er)
            if point in sweep["per_point_avg"]:
                row_cells.append(f"{sweep['per_point_avg'][point]['iou']:>8.1f}")
            else:
                row_cells.append(f"{'--':>8}")
        marker = "  <- current" if er == current_erosion else ""
        lines.append(f"{er:>8} | " + " | ".join(row_cells) + marker)
    lines.append("-" * len(header))
    lines.append("(values are average IoU% across the tested cells)")

    best = sweep["best_point"]
    if best is not None:
        best_bt, best_er = best
        lines.append("")
        lines.append(
            f"Best: BG Threshold={best_bt}, Signal Erosion={best_er} "
            f"(avg IoU={sweep['per_point_avg'][best]['iou']:.1f}%, "
            f"avg Dice={sweep['per_point_avg'][best]['dice']:.1f}%)"
        )
        if current_bg_threshold is not None and current_erosion is not None:
            current = (current_bg_threshold, current_erosion)
            if current in sweep["per_point_avg"] and current != best:
                lines.append(
                    f"Current setting (BG Threshold={current_bg_threshold}, Signal Erosion={current_erosion}): "
                    f"avg IoU={sweep['per_point_avg'][current]['iou']:.1f}% -- "
                    f"the sweep found a better combination above."
                )
            elif current == best:
                lines.append(f"Current setting matches the sweep's best -- confirmed.")

    if sweep.get("cancelled"):
        lines.append("\n(sweep was cancelled -- results above are partial.)")

    lines.append("")
    lines.append("Per-cell winner (grid point with highest IoU for that cell):")
    for c in sweep["cells"]:
        best_c = max(grid, key=lambda pt: sweep["results"][(pt[0], pt[1], c)]["iou"])
        bt_c, er_c = best_c
        lines.append(
            f"  cell {c:>4}: best at BG Threshold={bt_c}, Signal Erosion={er_c} "
            f"(IoU={sweep['results'][(bt_c, er_c, c)]['iou']:.1f}%)"
        )
    return "\n".join(lines)
