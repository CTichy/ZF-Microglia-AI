"""
_epoch_sweep.py — GT-verified epoch sweep for Cellpose-SAM checkpoints.

Automates the bbox-restricted GT-IoU sweep methodology developed by hand
across several training rounds to confirm a run's recommended checkpoint
(picked from test_loss, a proxy metric) against real ground truth: find
the N most morphologically complex cells in a GT-annotated fish, crop
each to its bounding box, run raw do_3D inference at each candidate
epoch, best-IoU-match every predicted object against the GT cell, and
average. Complex cells (highly branched, low sphericity) are used
deliberately instead of large/simple ones -- a large cell is often
amoeboid and easy, while a genuinely branchy cell is what actually
stresses the branch-weighted loss this project's training targets.

Not ported as a subprocess: `cellpose` is already an in-process
dependency of this plugin (same as _cellpose_seg.py's do_3D inference
used by Tab 2), so there's no separate conda env to shell out to here.
"""

import math
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import find_objects

from ._statistics import _skeleton_stats, _surface_area


def _complexity_score(binary, scale_zyx):
    """(n_branches, -sphericity) sort key -- most-branched first, least
    spherical as tiebreak. Same skeleton/surface-area code and sphericity
    formula (pi^(1/3) * (6V)^(2/3) / A) as the Statistics tab, so this
    ranking is consistent with what's already reported to the user
    elsewhere in the plugin."""
    n_branches = _skeleton_stats(binary, scale_zyx)[0]
    volume = float(binary.sum()) * scale_zyx[0] * scale_zyx[1] * scale_zyx[2]
    sa = _surface_area(binary, scale_zyx)
    sphericity = (math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / sa) if sa > 0 else 1.0
    return n_branches, -sphericity


def find_complex_cells(gt_labels, scale_zyx, n_cells=5, objs=None):
    """
    Rank every labeled cell in gt_labels by complexity (most skeleton
    branches first, least spherical as tiebreak) -- deliberately NOT by
    volume, which was confirmed misleading in this project's manual
    sweeps: a large cell is often simple/amoeboid, while a genuinely
    branchy cell that stresses the branch-weighted loss can be modest
    in size. Returns up to n_cells label IDs, most complex first.

    `objs` (a precomputed scipy.ndimage.find_objects(gt_labels) result)
    can be passed in to avoid recomputing it when the caller already has
    one -- optional, computed here if omitted.
    """
    if objs is None:
        objs = find_objects(gt_labels)
    scored = []
    for label_id in range(1, len(objs) + 1):
        sl = objs[label_id - 1]
        if sl is None:
            continue
        binary = gt_labels[sl] == label_id
        if not binary.any():
            continue
        scored.append((label_id, _complexity_score(binary, scale_zyx)))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [label_id for label_id, _ in scored[:n_cells]]


def bbox_crop(image, gt_labels, label_id, pad_z=15, pad_xy=40, objs=None):
    """
    Crop image+gt_labels to label_id's bounding box + padding. Returns
    (img_crop, gt_mask, gt_voxel_count). Default padding matches the
    convention used throughout this project's manual epoch-confirmation
    sweeps (15 vox Z, 40 vox XY).
    """
    if objs is None:
        objs = find_objects(gt_labels)
    sl = objs[label_id - 1] if label_id - 1 < len(objs) else None
    if sl is None:
        raise ValueError(f"Label {label_id} not found in GT volume.")
    Z, Y, X = gt_labels.shape
    z0 = max(0, sl[0].start - pad_z); z1 = min(Z, sl[0].stop + pad_z)
    y0 = max(0, sl[1].start - pad_xy); y1 = min(Y, sl[1].stop + pad_xy)
    x0 = max(0, sl[2].start - pad_xy); x1 = min(X, sl[2].stop + pad_xy)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    img_crop = image[crop_sl]
    gt_crop = gt_labels[crop_sl]
    gt_mask = gt_crop == label_id
    return img_crop, gt_mask, int(gt_mask.sum())


def pick_sweep_epochs(recommended_epoch, save_every, available_epochs, n_below=2, n_above=2):
    """
    Return up to (n_below + 1 + n_above) epochs centered on
    recommended_epoch, stepping by save_every in each direction, clipped
    to whatever checkpoints actually exist on disk (available_epochs) --
    fewer than the full spread near the start/end of a run, never padded
    with placeholders that don't exist.
    """
    available = sorted(set(available_epochs))
    if recommended_epoch not in available:
        raise ValueError(f"Recommended epoch {recommended_epoch} has no checkpoint on disk.")
    wanted = [recommended_epoch + k * save_every for k in range(-n_below, n_above + 1)]
    return [e for e in wanted if e in available]


def _best_gt_match(pred_masks, gt_mask, gt_vox):
    """Best-IoU-vs-GT match among predicted objects -- same logic used
    throughout this project's manual sweeps: try every predicted object
    that overlaps at all, keep the highest-IoU one."""
    pred_ids = np.unique(pred_masks[pred_masks > 0])
    best_iou, best_vol = 0.0, 0
    for pid in pred_ids:
        pm = pred_masks == pid
        inter = int(np.logical_and(pm, gt_mask).sum())
        if inter == 0:
            continue
        union = int(pm.sum()) + gt_vox - inter
        iou = inter / union
        if iou > best_iou:
            best_iou, best_vol = iou, int(pm.sum())
    dice = 2 * best_iou / (1 + best_iou) if best_iou > 0 else 0.0
    size_delta = ((best_vol - gt_vox) / gt_vox * 100) if best_vol > 0 else -100.0
    return dict(n_obj=len(pred_ids), iou=best_iou * 100, dice=dice * 100,
                pred_vox=best_vol, size_delta=size_delta)


def run_epoch_sweep(image_path, gt_labels_path, models_dir, model_name,
                     epochs, scale_zyx, cellprob=-2.5, flow=0.4,
                     n_cells=5, pad_z=15, pad_xy=40, gpu=True,
                     progress_cb=None, cancel_event=None):
    """
    The full sweep: find the n_cells most complex GT cells, crop each to
    its bbox, then for every candidate epoch load that checkpoint once
    and run do_3D inference on all n_cells crops (the model is loaded
    once per epoch, not once per cell -- same efficient pattern as the
    manual sweep scripts this automates). `scale_zyx` (Z, Y, X in
    um/voxel) drives both the complexity ranking and the do_3D
    anisotropy (Z/XY ratio) -- same convention as the rest of the
    plugin's metadata-driven scale handling.

    progress_cb(str), if given, is called with a one-line status message
    after each crop is prepared and after each (epoch, cell) inference.

    cancel_event (threading.Event), if given, is checked once between
    epochs (not mid-inference -- a single do_3D call can't be
    interrupted cleanly). If set, the sweep stops early and returns
    whatever completed so far with cancelled=True, rather than
    discarding it.

    Returns dict: {
      'cells': [label_id, ...],              # most-complex-first
      'epochs': [epoch, ...],                # swept & scored, ascending
      'results': {(epoch, label_id): {...}}, # see _best_gt_match
      'per_epoch_avg': {epoch: {'iou': x, 'dice': y}},
      'best_epoch': epoch or None,           # highest average IoU
      'cancelled': bool,
    }
    """
    from cellpose import models as cp_models

    gt_labels = tifffile.imread(gt_labels_path).astype(np.int32)
    image = tifffile.imread(image_path)
    anisotropy = scale_zyx[0] / scale_zyx[1]

    objs = find_objects(gt_labels)
    cells = find_complex_cells(gt_labels, scale_zyx, n_cells=n_cells, objs=objs)
    if not cells:
        raise ValueError("No labeled cells found in the GT volume.")

    crops = {}
    for label_id in cells:
        img_crop, gt_mask, gt_vox = bbox_crop(image, gt_labels, label_id, pad_z, pad_xy, objs=objs)
        crops[label_id] = (img_crop, gt_mask, gt_vox)
        if progress_cb:
            progress_cb(f"cell {label_id}: bbox shape={img_crop.shape}  gt_vox={gt_vox}")

    results = {}
    cancelled = False
    for epoch in epochs:
        if cancel_event is not None and cancel_event.is_set():
            if progress_cb:
                progress_cb(f"cancelled before epoch {epoch} -- reporting what completed so far.")
            cancelled = True
            break
        model_path = Path(models_dir) / f"{model_name}_epoch_{epoch:04d}"
        if not model_path.exists():
            if progress_cb:
                progress_cb(f"epoch {epoch}: checkpoint not found at {model_path}, skipping")
            continue
        if progress_cb:
            progress_cb(f"loading epoch {epoch} ...")
        model = cp_models.CellposeModel(pretrained_model=str(model_path), gpu=gpu)
        for label_id in cells:
            img_crop, gt_mask, gt_vox = crops[label_id]
            masks, _, _ = model.eval(
                img_crop, do_3D=True, anisotropy=anisotropy, z_axis=0, channel_axis=None,
                cellprob_threshold=cellprob, flow_threshold=flow,
                diameter=None, normalize=True, augment=False,
            )
            masks = np.asarray(masks, dtype=np.int32)
            r = _best_gt_match(masks, gt_mask, gt_vox)
            results[(epoch, label_id)] = r
            if progress_cb:
                progress_cb(
                    f"epoch {epoch}, cell {label_id}: IoU={r['iou']:.1f}%  Dice={r['dice']:.1f}%  "
                    f"SizeD={r['size_delta']:+.1f}%  n_obj={r['n_obj']}"
                )
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    per_epoch_avg = {}
    for epoch in epochs:
        vals = [results[(epoch, c)] for c in cells if (epoch, c) in results]
        if not vals:
            continue
        per_epoch_avg[epoch] = dict(
            iou=sum(v["iou"] for v in vals) / len(vals),
            dice=sum(v["dice"] for v in vals) / len(vals),
        )
    best_epoch = max(per_epoch_avg, key=lambda e: per_epoch_avg[e]["iou"]) if per_epoch_avg else None
    scored_epochs = [e for e in epochs if e in per_epoch_avg]

    return dict(cells=cells, epochs=scored_epochs, results=results,
                per_epoch_avg=per_epoch_avg, best_epoch=best_epoch, cancelled=cancelled)


def format_sweep_report(sweep, recommended_epoch):
    """Plain-text summary table, matching the format used throughout
    this project's manual sweep scripts -- so results read the same way
    whether they came from the GUI or a one-off research script."""
    cells = sweep["cells"]
    epochs = sweep["epochs"]
    if not epochs:
        return "No checkpoints could be evaluated (none found on disk for the swept epochs)."

    prefix = "CANCELLED -- partial results below.\n\n" if sweep.get("cancelled") else ""
    header = f"{'Epoch':>6} | " + " | ".join(f"Cell{c:>4} IoU%" for c in cells) + " | AvgIoU% | AvgDice%"
    lines = [header, "-" * len(header)]
    for epoch in epochs:
        ious = [sweep["results"][(epoch, c)]["iou"] for c in cells]
        avg = sweep["per_epoch_avg"][epoch]
        marker = "  <- recommended" if epoch == recommended_epoch else ""
        row = (f"{epoch:>6} | " + " | ".join(f"{v:>10.1f}" for v in ious) +
               f" | {avg['iou']:>7.1f} | {avg['dice']:>8.1f}{marker}")
        lines.append(row)
    lines.append("-" * len(header))

    best = sweep["best_epoch"]
    if recommended_epoch not in sweep["per_epoch_avg"]:
        lines.append(f"Recommended epoch {recommended_epoch} was not part of the sweep or has no checkpoint on disk.")
    elif best == recommended_epoch:
        lines.append(f"CONFIRMED: recommended epoch {recommended_epoch} is also the sweep's best "
                      f"(avg IoU={sweep['per_epoch_avg'][best]['iou']:.1f}%).")
    else:
        lines.append(
            f"NOTE: the sweep's best epoch is {best} (avg IoU={sweep['per_epoch_avg'][best]['iou']:.1f}%), "
            f"not the recommended {recommended_epoch} "
            f"(avg IoU={sweep['per_epoch_avg'][recommended_epoch]['iou']:.1f}%). "
            f"Only {len(cells)} cells x {len(epochs)} epochs were tested -- treat as a signal to look "
            f"closer, not an automatic override."
        )

    lines.append("")
    lines.append("Per-cell winner (epoch with highest IoU for that cell):")
    for c in cells:
        best_c = max(epochs, key=lambda e: sweep["results"][(e, c)]["iou"])
        lines.append(f"  cell {c:>4}: best at epoch {best_c} (IoU={sweep['results'][(best_c, c)]['iou']:.1f}%)")
    return prefix + "\n".join(lines)
