"""
_brain_sweep.py — GT-verified MONAI Threshold x Erosion sweep for the
brain mask itself (Tab 1), the third of this plugin's three GT-sweep
tools (see _epoch_sweep.py for Cellpose-SAM checkpoints, _pixel_sweep.py
for the Pixel Classifier's BG Threshold x Erosion).

Unlike the other two, this scores a single global binary mask against a
hand-corrected GT brain mask (from GT Annotation, Tab 4), not multiple
labeled cells -- so there's no "N complex cells" step here, just a
whole-volume Dice/IoU/precision/recall comparison.

Only the MONAI sliding-window inference itself (predict_probability) is
expensive and GPU-bound; it runs exactly once regardless of how many
threshold/erosion values are swept. Thresholding, largest-component +
fill-holes, and erosion are all cheap and reused from _inference.py
(postprocess_probability) rather than reimplemented, so results match a
real Tab 1 run at the same values exactly.
"""

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion

from ._inference import predict_probability, postprocess_probability


def _dice_iou(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    pred_vox = int(pred.sum())
    gt_vox = int(gt.sum())
    union = pred_vox + gt_vox - inter
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (pred_vox + gt_vox) if (pred_vox + gt_vox) > 0 else 0.0
    precision = inter / pred_vox if pred_vox > 0 else 0.0
    recall = inter / gt_vox if gt_vox > 0 else 0.0
    return dict(dice=dice * 100, iou=iou * 100, precision=precision * 100,
                recall=recall * 100, pred_vox=pred_vox, gt_vox=gt_vox)


def run_brain_sweep(volume_path, gt_brain_mask_path, model_path, device,
                     thresholds, erosions, progress_cb=None, cancel_event=None):
    """
    Runs MONAI inference once, then sweeps every (threshold, erosion)
    combination cheaply on top of the resulting probability map, scoring
    the whole-volume mask against a hand-corrected GT brain mask.

    thresholds : list of MONAI Threshold values (0-1) to sweep
    erosions   : list of erosion radii (voxels) to sweep

    progress_cb(str) / cancel_event: same contract as the other two
    sweep tools -- cancel_event is checked between threshold values;
    stops early and returns whatever grid points completed.

    Returns dict: {
      'grid': [(threshold, erosion), ...],
      'results': {(threshold, erosion): {dice, iou, precision, recall, pred_vox, gt_vox}},
      'best_point': (threshold, erosion) or None,   # highest Dice
      'cancelled': bool,
    }
    """
    volume = tifffile.imread(volume_path)
    gt_mask = tifffile.imread(gt_brain_mask_path).astype(bool)

    if progress_cb:
        progress_cb("running MONAI inference (once) ...")
    pred_prob = predict_probability(volume, model_path, device)
    if progress_cb:
        progress_cb("inference done -- sweeping threshold/erosion (cheap from here) ...")

    results = {}
    cancelled = False
    for threshold in thresholds:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        raw_mask = postprocess_probability(pred_prob, threshold)
        for erosion in erosions:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            mask = (binary_erosion(raw_mask, iterations=erosion).astype(np.uint8)
                    if erosion > 0 else raw_mask)
            r = _dice_iou(mask, gt_mask)
            results[(threshold, erosion)] = r
            if progress_cb:
                progress_cb(
                    f"threshold={threshold}, erosion={erosion}: "
                    f"Dice={r['dice']:.1f}%  IoU={r['iou']:.1f}%  "
                    f"precision={r['precision']:.1f}%  recall={r['recall']:.1f}%"
                )
        if cancelled:
            break

    grid = sorted(results.keys())
    best_point = max(results, key=lambda k: results[k]["dice"]) if results else None

    return dict(grid=grid, results=results, best_point=best_point, cancelled=cancelled)


def format_brain_sweep_report(sweep, current_threshold=None, current_erosion=None):
    """Plain-text 2D grid report (rows = erosion, columns = threshold),
    same spirit as the other two sweep tools' reports."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    thresholds = sorted({t for t, _ in grid})
    erosions = sorted({e for _, e in grid})

    header = f"{'Erosion':>8} | " + " | ".join(f"th={t:>5} " for t in thresholds)
    lines = [header, "-" * len(header)]
    for er in erosions:
        row = []
        for th in thresholds:
            point = (th, er)
            row.append(f"{sweep['results'][point]['dice']:>8.1f}" if point in sweep["results"] else f"{'--':>8}")
        marker = "  <- current" if er == current_erosion else ""
        lines.append(f"{er:>8} | " + " | ".join(row) + marker)
    lines.append("-" * len(header))
    lines.append("(values are Dice% against the GT brain mask)")

    best = sweep["best_point"]
    if best is not None:
        best_th, best_er = best
        r = sweep["results"][best]
        lines.append("")
        lines.append(
            f"Best: MONAI Threshold={best_th}, Erosion={best_er} "
            f"(Dice={r['dice']:.1f}%, IoU={r['iou']:.1f}%, "
            f"precision={r['precision']:.1f}%, recall={r['recall']:.1f}%)"
        )
        if current_threshold is not None and current_erosion is not None:
            current = (current_threshold, current_erosion)
            if current in sweep["results"] and current != best:
                lines.append(
                    f"Current setting (MONAI Threshold={current_threshold}, Erosion={current_erosion}): "
                    f"Dice={sweep['results'][current]['dice']:.1f}% -- "
                    f"the sweep found a better combination above."
                )
            elif current == best:
                lines.append("Current setting matches the sweep's best -- confirmed.")

    if sweep.get("cancelled"):
        lines.append("\n(sweep was cancelled -- results above are partial.)")
    return "\n".join(lines)
