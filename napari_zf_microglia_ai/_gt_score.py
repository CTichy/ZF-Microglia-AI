"""
_gt_score.py — whole-fish, Hungarian-matched instance segmentation
scoring against ground truth: TP/FP/FN/Score + mean IoU/Dice over
matched pairs.

This is the compare_pred_gt.py methodology that's been used throughout
this project to validate essentially every real modeling decision
(checkpoint picks, cellprob/large_contact tuning, before/after
comparisons) -- ported into the plugin as a reusable, GPU-free scoring
function instead of remaining a standalone CLI script. It's what the
other three GT-sweep tools (_epoch_sweep.py, _pixel_sweep.py,
_brain_sweep.py) approximate with a handful of complex cells / a single
mask; this scores every object in the whole fish at once.

Score = TP - 0.5*(FP + FN), matching this project's own established
convention (verified against multiple historical result tables, e.g.
TP=34/FP=8/FN=5 -> Score=+27.5, TP=10/FP=28/FN=18 -> Score=-13.0).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._cellpose_seg import _get_info, _bboxes_close, _joint_bbox


def _iou_and_volumes(labels_a, id_a, labels_b, id_b, joint_bbox):
    sl = tuple(slice(lo, hi) for lo, hi in joint_bbox)
    a = labels_a[sl] == id_a
    b = labels_b[sl] == id_b
    inter = int(np.logical_and(a, b).sum())
    vol_a = int(a.sum())
    vol_b = int(b.sum())
    if inter == 0:
        return 0.0, vol_a, vol_b
    union = vol_a + vol_b - inter
    return inter / union, vol_a, vol_b


def score_against_gt(pred_labels, gt_labels, iou_threshold=0.5):
    """
    Hungarian-matched instance segmentation scoring of pred_labels
    against gt_labels (both (Z, Y, X) int label arrays, whole-fish
    scale). Only object pairs whose bounding boxes actually intersect
    are ever compared (IoU is necessarily 0 otherwise) -- this keeps the
    cost-matrix construction fast even for a full fish with dozens of
    objects, without ever missing a possible match.

    Returns dict: {
      'tp': int, 'fp': int, 'fn': int, 'score': float,
      'mean_iou': float, 'mean_dice': float,          # over TP matches only
      'n_pred': int, 'n_gt': int,
      'matches': [{'pred_id', 'gt_id', 'iou', 'dice', 'pred_vox', 'gt_vox', 'size_delta'}, ...],
      'fp_ids': [pred_id, ...], 'fn_ids': [gt_id, ...],
    }
    """
    pred_info = _get_info(pred_labels)
    gt_info = _get_info(gt_labels)
    pred_ids = sorted(pred_info.keys())
    gt_ids = sorted(gt_info.keys())

    iou_mat = np.zeros((len(gt_ids), len(pred_ids)))
    vox_cache = {}
    for gi, gid in enumerate(gt_ids):
        for pi, pid in enumerate(pred_ids):
            if not _bboxes_close(gt_info[gid]["bbox"], pred_info[pid]["bbox"], margin=0):
                continue
            jb = _joint_bbox(gt_info[gid]["bbox"], pred_info[pid]["bbox"])
            iou, pv, gv = _iou_and_volumes(gt_labels, gid, pred_labels, pid, jb)
            iou_mat[gi, pi] = iou
            vox_cache[(gid, pid)] = (pv, gv)

    matches = []
    matched_gt, matched_pred = set(), set()
    if iou_mat.size:
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        for gi, pi in zip(row_ind, col_ind):
            iou = iou_mat[gi, pi]
            if iou >= iou_threshold:
                gid, pid = gt_ids[gi], pred_ids[pi]
                gv, pv = vox_cache[(gid, pid)]
                dice = 2 * iou / (1 + iou)
                size_delta = ((pv - gv) / gv * 100) if gv > 0 else 0.0
                matches.append(dict(
                    pred_id=pid, gt_id=gid, iou=iou * 100, dice=dice * 100,
                    pred_vox=pv, gt_vox=gv, size_delta=size_delta,
                ))
                matched_gt.add(gid)
                matched_pred.add(pid)

    fn_ids = [g for g in gt_ids if g not in matched_gt]
    fp_ids = [p for p in pred_ids if p not in matched_pred]
    tp, fp, fn = len(matches), len(fp_ids), len(fn_ids)
    score = tp - 0.5 * (fp + fn)
    mean_iou = sum(m["iou"] for m in matches) / tp if tp else 0.0
    mean_dice = sum(m["dice"] for m in matches) / tp if tp else 0.0

    return dict(tp=tp, fp=fp, fn=fn, score=score, mean_iou=mean_iou, mean_dice=mean_dice,
                n_pred=len(pred_ids), n_gt=len(gt_ids),
                matches=matches, fp_ids=fp_ids, fn_ids=fn_ids)


def format_gt_score_report(result, pred_name="prediction", gt_name="GT"):
    """Plain-text summary, matching the TP/FP/FN/Score/MeanIoU/MeanDice
    tables this project has used throughout its history."""
    lines = [
        f"{pred_name} vs {gt_name}  ({result['n_pred']} predicted objects, {result['n_gt']} GT objects)",
        "-" * 70,
        f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  Score={result['score']:+.1f}",
        f"Mean IoU (matched)  = {result['mean_iou']:.1f}%",
        f"Mean Dice (matched) = {result['mean_dice']:.1f}%",
        "",
    ]
    if result["matches"]:
        lines.append(f"{'Pred':>6} {'GT':>6} {'IoU%':>7} {'Dice%':>7} {'PredVox':>9} {'GTVox':>9} {'SizeD%':>8}")
        for m in sorted(result["matches"], key=lambda m: m["gt_id"]):
            lines.append(
                f"{m['pred_id']:>6} {m['gt_id']:>6} {m['iou']:>7.1f} {m['dice']:>7.1f} "
                f"{m['pred_vox']:>9} {m['gt_vox']:>9} {m['size_delta']:>+8.1f}"
            )
    if result["fn_ids"]:
        lines.append("")
        lines.append(f"FN (missed GT objects): {result['fn_ids']}")
    if result["fp_ids"]:
        lines.append("")
        lines.append(f"FP (spurious predicted objects): {result['fp_ids']}")
    return "\n".join(lines)
