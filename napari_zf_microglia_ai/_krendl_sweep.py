"""
_krendl_sweep.py — GT-verified cellprob x large_contact sweep for the
Cellpose-SAM Segmentation pipeline (Tab 2), scored against a full-fish
GT via _gt_score.score_against_gt -- the same whole-fish Hungarian-
matched methodology this project has used throughout its own parameter
tuning history (e.g. the cellprob=-2.5/large_contact=20 discovery that
became the current default).

cellprob changes what do_3D actually predicts, so it requires a real
do_3D re-inference per value -- the one expensive, GPU-bound dimension.
large_contact is a post-processing merge threshold applied after do_3D
+ GMM cleanup + Krendl safe-merge, so it's cheap to sweep on top of a
single do_3D result: run do_3D + GMM + safe-merge once per cellprob
value, then vary large_contact freely on that same intermediate result
-- mirrors this project's own established `--skip_inference` shortcut
for exactly this kind of sweep. max_gap/min_contact (Krendl safe-merge
parameters) are held fixed at whatever Tab 2 is currently set to; only
cellprob and large_contact vary here, matching how every historical
sweep in this project's history was actually run.
"""

from ._cellpose_seg import (
    run_do3d_inference, gmm_cleanup, krendl_safe_merge,
    large_contact_merge, relabel_sequential, GT_MIN,
)
from ._gt_score import score_against_gt


def run_krendl_sweep(volume, gt_labels, model_path, cellprobs, large_contacts,
                      flow=0.4, anisotropy=5.747, max_gap=2, min_contact=10,
                      gt_min=GT_MIN, iou_threshold=0.5, gpu=True,
                      progress_cb=None, cancel_event=None):
    """
    Sweep every (cellprob, large_contact) combination, scoring the
    resulting Krendl-pipeline labels against gt_labels with
    _gt_score.score_against_gt.

    progress_cb(str) / cancel_event: same contract as the other sweep
    tools -- cancel_event is checked between cellprob values (not
    between large_contact values, since those are cheap and fast enough
    that checking every one adds no real responsiveness).

    Returns dict: {
      'grid': [(cellprob, large_contact), ...],
      'results': {(cellprob, large_contact): <score_against_gt() dict>},
      'best_point': (cellprob, large_contact) or None,   # highest Score
      'cancelled': bool,
    }
    """
    results = {}
    cancelled = False
    for cellprob in cellprobs:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        if progress_cb:
            progress_cb(f"cellprob={cellprob}: running do_3D inference...")
        masks = run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=gpu)
        n0 = len(set(masks[masks > 0].tolist()))
        if progress_cb:
            progress_cb(f"cellprob={cellprob}: {n0} raw cells — GMM + Krendl safe-merge...")
        masks, _, _ = gmm_cleanup(masks)
        masks, _ = krendl_safe_merge(masks, max_gap, min_contact, gt_min)

        for large_contact in large_contacts:
            merged, _ = large_contact_merge(masks, large_contact)
            labels, n_labels = relabel_sequential(merged)
            r = score_against_gt(labels, gt_labels, iou_threshold=iou_threshold)
            results[(cellprob, large_contact)] = r
            if progress_cb:
                progress_cb(
                    f"cellprob={cellprob}, large_contact={large_contact}: "
                    f"TP={r['tp']} FP={r['fp']} FN={r['fn']} Score={r['score']:+.1f} "
                    f"MeanIoU={r['mean_iou']:.1f}%"
                )

    grid = sorted(results.keys())
    best_point = max(results, key=lambda k: results[k]["score"]) if results else None

    return dict(grid=grid, results=results, best_point=best_point, cancelled=cancelled)


def format_krendl_sweep_report(sweep, current_cellprob=None, current_large_contact=None):
    """Plain-text 2D grid report (rows = large_contact, columns = cellprob),
    same spirit as the plugin's other sweep-tool reports."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    cellprobs = sorted({c for c, _ in grid})
    large_contacts = sorted({lc for _, lc in grid})

    header = f"{'LrgCnt':>8} | " + " | ".join(f"cp={c:>5} " for c in cellprobs)
    lines = [header, "-" * len(header)]
    for lc in large_contacts:
        row = []
        for cp in cellprobs:
            point = (cp, lc)
            row.append(f"{sweep['results'][point]['score']:>+8.1f}" if point in sweep["results"] else f"{'--':>8}")
        marker = "  <- current" if lc == current_large_contact else ""
        lines.append(f"{lc:>8} | " + " | ".join(row) + marker)
    lines.append("-" * len(header))
    lines.append("(values are Score = TP - 0.5*(FP+FN))")

    best = sweep["best_point"]
    if best is not None:
        best_cp, best_lc = best
        r = sweep["results"][best]
        lines.append("")
        lines.append(
            f"Best: cellprob={best_cp}, large_contact={best_lc} "
            f"(TP={r['tp']} FP={r['fp']} FN={r['fn']} Score={r['score']:+.1f}, "
            f"MeanIoU={r['mean_iou']:.1f}%, MeanDice={r['mean_dice']:.1f}%)"
        )
        if current_cellprob is not None and current_large_contact is not None:
            current = (current_cellprob, current_large_contact)
            if current in sweep["results"] and current != best:
                cr = sweep["results"][current]
                lines.append(
                    f"Current setting (cellprob={current_cellprob}, large_contact={current_large_contact}): "
                    f"Score={cr['score']:+.1f} -- the sweep found a better combination above."
                )
            elif current == best:
                lines.append("Current setting matches the sweep's best -- confirmed.")

    if sweep.get("cancelled"):
        lines.append("\n(sweep was cancelled -- results above are partial.)")
    return "\n".join(lines)
