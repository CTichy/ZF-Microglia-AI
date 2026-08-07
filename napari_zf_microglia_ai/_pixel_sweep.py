"""
_pixel_sweep.py — GT-verified BG Threshold x Erosion sweep for microglia
labels produced by the Pixel Classifier path (Tab 1 background removal +
Tab 2 Create Labels / union-find), scored the same way as
_epoch_sweep.py's Cellpose-SAM checkpoint sweep: crop to the N most
morphologically complex GT cells, best-IoU-match the resulting labels
against GT, average.

Deliberately does NOT re-run MONAI inference per grid point -- neither
Erosion nor BG Threshold change what MONAI predicts, only how that
prediction gets turned into brain_only and then labels. Takes a
pre-computed, un-eroded brain_mask.tif (produced once via a normal Tab 1
run) as input instead. This makes the whole sweep union-find-bound, not
GPU-inference-bound -- a full grid finishes in minutes, not hours, and
doesn't need a GPU at all (Create Labels already has a CPU fallback).

Uses Background mode 2 ("Remove globally") -- the documented recommended
setting before labeling -- since that's the mode this sweep validates.
Reuses _background.py's own threshold math (not reimplemented) so swept
results match exactly what a real Tab 1 run at the same values would
produce -- see the Erosion/background-mode composition fix in _on_run,
which this sweep depends on to be meaningful in the first place.
"""

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, find_objects

from ._background import _threshold as _bg_threshold
from ._epoch_sweep import find_complex_cells, bbox_crop, _best_gt_match
from ._labeling import create_labels


def run_pixel_sweep(image_path, brain_mask_path, gt_labels_path,
                     bg_thresholds, erosions, scale_zyx,
                     sigma_xy=1.5, sigma_z=3.0, min_volume=7500,
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
    erosions            : list of erosion radii (voxels) to sweep
    scale_zyx           : (Z, Y, X) um/voxel -- drives complexity ranking
    sigma_xy/sigma_z/min_volume : held fixed, passed straight to
                         create_labels() (same defaults as Tab 2)

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
      'cancelled': bool,
    }
    """
    image = tifffile.imread(image_path)
    brain_mask = tifffile.imread(brain_mask_path).astype(bool)
    gt_labels = tifffile.imread(gt_labels_path).astype(np.int32)
    data_range = float(image.max()) - float(image.min())

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

    results = {}
    cancelled = False
    for erosion in erosions:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        # Global background estimate for this erosion level -- needs
        # full-volume context (_background.py's _estimate_background is a
        # histogram over ALL brain-masked pixels, a genuinely global
        # quantity, not something safe to compute from a small crop).
        eroded_full = (binary_erosion(brain_mask, iterations=erosion).astype(np.uint8)
                       if erosion > 0 else brain_mask.astype(np.uint8))
        _, bg_max, _, _ = _bg_threshold(image, eroded_full, tolerance_pct=0.0)
        if progress_cb:
            progress_cb(f"erosion {erosion}: global bg_max={bg_max:.2f}")

        # Padding (default 15 vox Z, 40 vox XY) comfortably exceeds any
        # sane erosion radius, so eroding the padded crop directly here
        # (instead of eroding the full volume per cell) gives the same
        # result much more cheaply.
        for label_id in cells:
            c = crops[label_id]
            c["eroded_mask"] = (binary_erosion(c["raw_mask"], iterations=erosion)
                                 if erosion > 0 else c["raw_mask"])

        for bg_threshold in bg_thresholds:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            thresh = bg_max + data_range * (bg_threshold / 100.0)
            for label_id in cells:
                c = crops[label_id]
                img_thresholded = np.where(c["img"] <= thresh, 0, c["img"])
                brain_only_crop = (img_thresholded * c["eroded_mask"]).astype(c["img"].dtype)
                pred_labels = create_labels(
                    brain_only_crop, sigma_xy=sigma_xy, sigma_z=sigma_z, min_volume=min_volume
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
                per_point_avg=per_point_avg, best_point=best_point, cancelled=cancelled)


def format_pixel_sweep_report(sweep, current_bg_threshold=None, current_erosion=None):
    """Plain-text 2D grid report (rows = erosion, columns = BG Threshold),
    same spirit as _epoch_sweep.format_sweep_report."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    bg_thresholds = sorted({bt for bt, _ in grid})
    erosions = sorted({er for _, er in grid})

    header = f"{'Erosion':>8} | " + " | ".join(f"bg={bt:>5} " for bt in bg_thresholds)
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
            f"Best: BG Threshold={best_bt}, Erosion={best_er} "
            f"(avg IoU={sweep['per_point_avg'][best]['iou']:.1f}%, "
            f"avg Dice={sweep['per_point_avg'][best]['dice']:.1f}%)"
        )
        if current_bg_threshold is not None and current_erosion is not None:
            current = (current_bg_threshold, current_erosion)
            if current in sweep["per_point_avg"] and current != best:
                lines.append(
                    f"Current setting (BG Threshold={current_bg_threshold}, Erosion={current_erosion}): "
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
            f"  cell {c:>4}: best at BG Threshold={bt_c}, Erosion={er_c} "
            f"(IoU={sweep['results'][(bt_c, er_c, c)]['iou']:.1f}%)"
        )
    return "\n".join(lines)
