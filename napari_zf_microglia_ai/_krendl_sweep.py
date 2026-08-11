"""
_krendl_sweep.py — GT-verified cellprob x large_contact sweep for the
Cellpose-SAM Segmentation pipeline (Tab 2), scored against a full-fish
GT via _gt_score.score_against_gt -- the same whole-fish Hungarian-
matched methodology this project has used throughout its own parameter
tuning history (e.g. the cellprob=-2.5/large_contact=20 discovery that
became the current default).

Originally this called run_do3d_inference() fresh for every cellprob
value, on the assumption that cellprob changes what do_3D predicts and
therefore needs a real re-inference each time -- true in spirit, but a
needless full re-run in practice. Reading cellpose/models.py directly
shows CellposeModel.eval() internally splits into two independent
steps: self._run_net() (the actual GPU network forward pass -- the
genuinely expensive part, unrelated to any threshold) and
self._compute_masks(..., cellprob_threshold=..., flow_threshold=...)
(cheap flow-following + thresholding on the already-computed flow
field). cellprob_threshold only feeds the cheap second step, so the
network pass only needs to run ONCE per sweep, not once per cellprob
value -- predict_flows()/masks_from_flows() in _cellpose_seg.py expose
exactly that split. This sweep now costs roughly one do_3D network
pass total (~3h on a full-size fish, this project's own historical
figure) regardless of how many cellprob values are in the grid,
instead of one pass per value (~3h x N).

flow (flow_threshold) was considered as a second swept axis alongside
cellprob, but reading cellpose/dynamics.py's compute_masks() shows its
flow-error QC filter (remove_bad_flow_masks) is called only inside
`if not do_3D:` -- under do_3D=True (this project's pipeline, always)
it never runs, confirmed both by that code path and by a call-count
spy test. Sweeping it here would be a wasted axis; it's held fixed
purely because do_3D's own function signature still accepts it.

large_contact is a post-processing merge threshold applied after
do_3D + GMM cleanup + Krendl safe-merge, and stays cheap to sweep on
top of a single do_3D+GMM+safe-merge result exactly as before: GMM +
safe-merge run once per cellprob value, large_contact then varies
freely on that same intermediate result -- mirrors this project's own
established `--skip_inference` shortcut for exactly this kind of
sweep. max_gap/min_contact (Krendl safe-merge parameters) are held
fixed at whatever Tab 2 is currently set to; only cellprob and
large_contact vary here, matching how every historical sweep in this
project's history was actually run.

gt_min (the smallest real-cell volume Krendl safe-merge trusts as
"already a whole cell", below which a fragment is a merge candidate)
used to be a single hardcoded historical constant (GT_MIN=10230,
"smallest real microglia volume seen in validated GT data" as of
whenever that constant was last set). That's a snapshot of one past
GT, not necessarily representative of the GT actually being swept
against here. Since a real gt_labels volume is already an input to
every sweep, gt_min is now measured directly from it (the smallest
labeled cell's true voxel volume) unless the caller explicitly
overrides -- the sweep's own GT statistics recalibrate this parameter
every time it runs, instead of trusting a frozen number.
"""

from ._cellpose_seg import (
    predict_flows, masks_from_flows, gmm_cleanup,
    krendl_safe_merge, large_contact_merge, final_min_size_cleanup, relabel_sequential,
)
from ._gt_score import score_against_gt
from ._pixel_sweep import min_volume_from_gt as gt_min_from_labels
from ._pixel_sweep import min_hole_size_from_gt
# gt_min_from_labels is kept as a name here for readability at this
# module's call sites (Krendl safe-merge's "already a whole cell"
# floor), but it is no longer its own implementation: gt_min and the
# Pixel Classifier's min_volume are literally the same measurement --
# the smallest true voxel volume among GT-labeled cells -- and were
# only ever tracked as two separate config histories by historical
# accident. Both now read and update the single shared
# min_volume_vox/min_volume_recommended_vox floor (see
# _widget.py's _update_gt_history calls), so a fish checked through
# either the Pixel Classifier sweeps, this sweep, or Tab 3 Statistics
# (when marked as verified GT) all contribute to the same number.


def run_krendl_sweep(volume, gt_labels, model_path, cellprobs, large_contacts,
                      flow=0.4, anisotropy=5.747, max_gap=2, min_contact=10,
                      gt_min=None, iou_threshold=0.5, gpu=True, min_hole_size=None,
                      min_size=15, final_min_fraction=0.618,
                      progress_cb=None, cancel_event=None):
    """
    Sweep every (cellprob, large_contact) combination, scoring the
    resulting Krendl-pipeline labels against gt_labels with
    _gt_score.score_against_gt.

    gt_min: if None (default), computed from gt_labels itself via
    gt_min_from_labels() -- the sweep recalibrates this parameter from
    the real GT statistics every time it runs. Pass an explicit value
    to override.

    min_hole_size: passed through to masks_from_flows() -- see
    _cellpose_seg._make_capped_fill_holes()'s docstring. Shared with the
    Pixel Classifier route's Min hole size value. If None (default),
    computed from gt_labels itself via _pixel_sweep.min_hole_size_from_gt()
    -- the same real-GT measurement the Pixel Classifier's own two GT
    sweeps already use, so this route's recommendation is measured, not
    guessed, and every sweep tool feeds the same never-rising floor.
    Pass an explicit value (e.g. 0, matching Cellpose's own unconditional
    hole-filling) to override.

    final_min_fraction: passed through to final_min_size_cleanup(), run
    after large_contact_merge on every grid point exactly like
    run_full_pipeline() does in production -- see that function's
    docstring for why 0.618 (golden ratio) is the default. Keeping this
    sweep's pipeline shape identical to production is the whole point of
    testing here rather than trusting the proxy metrics alone.

    progress_cb(str) / cancel_event: same contract as the other sweep
    tools -- cancel_event is checked between cellprob values (not
    between large_contact values, since those are cheap and fast enough
    that checking every one adds no real responsiveness).

    Returns dict: {
      'grid': [(cellprob, large_contact), ...],
      'results': {(cellprob, large_contact): <score_against_gt() dict>},
      'best_point': (cellprob, large_contact) or None,   # highest Score
      'gt_min_used': int,   # the gt_min value actually applied (measured
                             # or overridden), for reporting/auto-apply
      'min_hole_size_used': int,   # the min_hole_size value actually
                             # applied (measured or overridden)
      'cancelled': bool,
    }
    """
    if gt_min is None:
        gt_min = gt_min_from_labels(gt_labels)
        if progress_cb:
            progress_cb(f"gt_min computed from this GT's smallest labeled cell: {gt_min} vox")

    if min_hole_size is None:
        min_hole_size = min_hole_size_from_gt(gt_labels)
        if progress_cb:
            progress_cb(f"min_hole_size computed from this GT's own real holes: {min_hole_size} vox")

    if progress_cb:
        progress_cb("Predicting flows (do_3D network pass -- the one expensive step, runs once)...")
    precomputed = predict_flows(volume, model_path, anisotropy, gpu=gpu)
    if progress_cb:
        progress_cb("Flows ready — forming masks per Cellprob value (cheap, no re-inference)...")

    results = {}
    cancelled = False
    for cellprob in cellprobs:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break

        model, dP, cellprob_map, shape = precomputed
        masks = masks_from_flows(model, dP, cellprob_map, shape, cellprob, flow,
                                  min_size=min_size, min_hole_size=min_hole_size)
        n0 = len(set(masks[masks > 0].tolist()))
        if progress_cb:
            progress_cb(f"cellprob={cellprob}: {n0} raw cells — GMM + Krendl safe-merge...")
        masks, _, _ = gmm_cleanup(masks)
        masks, _ = krendl_safe_merge(masks, max_gap, min_contact, gt_min)

        for large_contact in large_contacts:
            merged, _ = large_contact_merge(masks, large_contact)
            merged, _ = final_min_size_cleanup(merged, gt_min, final_min_fraction)
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

    return dict(grid=grid, results=results, best_point=best_point,
                gt_min_used=gt_min, min_hole_size_used=min_hole_size, cancelled=cancelled)


def format_krendl_sweep_report(sweep, current_cellprob=None, current_large_contact=None):
    """Plain-text 2D grid report (rows = large_contact, columns = cellprob),
    same spirit as the plugin's other sweep-tool reports."""
    grid = sweep["grid"]
    if not grid:
        return "No grid points completed."

    cellprobs = sorted({c for c, _ in grid})
    large_contacts = sorted({lc for _, lc in grid})

    lines0 = []
    if sweep.get("gt_min_used") is not None:
        lines0.append(
            f"gt_min used for Safe-merge: {sweep['gt_min_used']} vox "
            f"(measured from this GT's smallest labeled cell)\n"
        )
    if sweep.get("min_hole_size_used") is not None:
        lines0.append(
            f"min_hole_size used: {sweep['min_hole_size_used']} vox "
            f"(measured from this GT's own real holes)\n"
        )

    header = f"{'LrgCnt':>8} | " + " | ".join(f"cp={c:>5} " for c in cellprobs)
    lines = list(lines0) + [header, "-" * len(header)]
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
