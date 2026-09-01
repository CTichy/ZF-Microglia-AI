"""
_auto_correction.py -- Automatic full-stack contrast correction, chained
onto the end of a Cellpose-SAM Segmentation run.

Ties together three tools this plugin already has, in the one order that
makes each of them safe to run completely unattended, on every cell:

  1. Calibrate Correct-Label Contrast's own self-referential sweep
     (_contrast_sweep.py) -- finds the one intensity threshold `lo` that
     best REPRODUCES what Cellpose-SAM just segmented, no external GT
     needed (this is exactly what that sweep was built for).

  2. Pass 1 -- cell-by-cell, whole-volume: correct_label_from_intensity_3d()
     run independently for EVERY label at that calibrated `lo`. Each call
     is foreign-protected (can never claim territory another label
     already holds), so cells can't invade each other's already-corrected
     shape -- but that also makes the result order-dependent exactly
     where two cells' newly-recalibrated shapes meet: whichever label was
     corrected first wins the contested boundary pixels, purely by
     accident of processing order, not by anything about the real signal
     there.

  3. Pass 2 -- slice-by-slice, for touching groups only: after pass 1,
     touching_groups_for_stack() finds every (slice, cell-group) where
     two or more cells ended up directly bordering each other. Exactly
     those are re-derived jointly with correct_label_group_2d()'s
     marker-seeded watershed, seeded from each cell's own pass-1 shape --
     replacing the greedy, order-dependent boundary with a real one that
     treats every cell in the group symmetrically. Slices where cells
     don't touch at all are left as pass 1 produced them; nothing here
     runs a full second correction of every label everywhere.

  4. A final whole-layer Remove Debris pass (same golden-ratio floor as
     every other final-safety-net stage in this plugin) -- pass 2's
     watershed splits can leave a small disconnected sliver behind at a
     cut, on top of whatever pass 1's own per-label debris cleanup didn't
     already catch.

This directly answers "cell-by-cell or slice-by-slice, several labels at
once?" -- both, each for the specific failure mode it actually solves:
whole-cell 3D consistency by default, joint multi-label re-derivation
only where cells actually turn out to touch.
"""

from __future__ import annotations

import numpy as np

from ._labeling import (
    correct_label_from_intensity_3d,
    correct_label_group_2d,
    touching_groups_for_stack,
    remove_debris,
)
from ._contrast_sweep import (
    select_calibration_samples,
    default_lo_candidates,
    sweep_contrast_lower_value,
)


def auto_contrast_correct_stack(
    labels: np.ndarray,
    image: np.ndarray,
    scale_zyx: "tuple[float, float, float]",
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    pad: int = 15,
    sigma: float = 1.0,
    n_cells_calib: int = 5,
    slices_per_cell_calib: int = 10,
    n_lo_steps: int = 40,
    edge_margin_um: float = 50.0,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Full automatic post-segmentation correction. See the module
    docstring for the 4-step pipeline this runs.

    labels, image      : (Z, Y, X) volumes, same shape -- labels is the
                          just-produced Cellpose-SAM result, image is
                          the raw ExtRm signal it was segmented from
    scale_zyx           : voxel size (µm), only used to pick calibration
                          cells away from the volume's own edge
    min_volume          : Common Settings' Min volume (voxels) -- drives
                          both pass 1's per-label debris cleanup and the
                          final whole-layer pass. None skips debris
                          cleanup entirely (report will show 0 removed)
    final_min_fraction  : golden ratio (0.618) by default, matching
                          every other final-safety-net stage
    pad, sigma          : same meaning as every other Correct Label tool
    n_cells_calib, slices_per_cell_calib, n_lo_steps, edge_margin_um
                        : passed straight through to the contrast sweep
                          (select_calibration_samples / default_lo_candidates)
    progress_cb          : optional callable(str), called with a
                          human-readable status line as each stage/step
                          advances

    Returns (new_labels, report). report is a dict:
        best_lo               -- the calibrated intensity threshold used
        sweep_mean_iou         -- that threshold's own mean IoU against
                                  the calibration samples
        n_calibration_samples  -- how many (label, z) samples the sweep
                                  actually used
        n_cells_total           -- labels present at the start
        n_cells_corrected       -- how many pass-1 whole-cell corrections
                                  actually succeeded
        skipped_cells           -- {label_id: reason} for every pass-1
                                  correction that raised (left as
                                  Cellpose-SAM originally produced it,
                                  never crashes the whole run)
        n_group_slices           -- how many (slice, group) joint
                                  corrections pass 2 attempted
        n_groups_corrected       -- how many of those actually succeeded
        skipped_groups           -- [{z, labels, reason}, ...] for every
                                  pass-2 joint correction that raised
                                  (left as pass 1 produced it there)
        n_debris_fragments_removed -- fragments cleared by the final pass

    Raises ValueError only for conditions that make the WHOLE run
    meaningless (no labels present at all, or no cell suitable for
    contrast calibration) -- a failure on any individual cell or group
    is caught and recorded in the report instead, never aborts the run.
    """

    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    labels = np.asarray(labels).astype(np.int32)
    image = np.asarray(image)
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids > 0]
    if unique_ids.size == 0:
        raise ValueError("no labels present -- nothing to correct")

    # ── Step 1: self-referential contrast calibration ──────────────────
    _report("Auto-correct: selecting contrast-calibration samples...")
    samples = select_calibration_samples(
        labels, scale_zyx, n_cells=n_cells_calib, slices_per_cell=slices_per_cell_calib,
        edge_margin_um=edge_margin_um,
    )
    if not samples:
        raise ValueError(
            "no interior/complex-enough cells found for contrast calibration "
            "-- can't auto-correct this stack"
        )
    lo_candidates = default_lo_candidates(image, samples, pad, n_steps=n_lo_steps)

    def _sweep_progress(msg: str) -> None:
        _report(f"Auto-correct: {msg}")

    sweep = sweep_contrast_lower_value(
        labels, image, samples, lo_candidates, pad=pad, progress_cb=_sweep_progress,
    )
    best_lo = sweep["best_lo"]
    _report(
        f"Auto-correct: calibrated lo={best_lo:.4g} "
        f"(mean IoU={sweep['best_mean_iou']:.3f} on {sweep['n_samples']} samples) "
        f"-- correcting {unique_ids.size} cell(s)..."
    )

    # ── Step 2 (pass 1): cell-by-cell, whole-volume correction ─────────
    new_labels = labels
    n_corrected = 0
    skipped_cells: "dict[int, str]" = {}
    n_total = int(unique_ids.size)
    for idx, lid in enumerate(unique_ids.tolist()):
        try:
            new_labels, _cell_report = correct_label_from_intensity_3d(
                new_labels, image, lid, best_lo, pad=pad,
                min_volume=min_volume, final_min_fraction=final_min_fraction,
            )
            n_corrected += 1
        except ValueError as exc:
            skipped_cells[lid] = str(exc)
        if idx % 5 == 0 or idx == n_total - 1:
            _report(
                f"Auto-correct: cell-by-cell pass {idx + 1}/{n_total} "
                f"(label {lid}) -- {n_corrected} corrected, "
                f"{len(skipped_cells)} skipped so far"
            )

    # ── Step 3: find where cells now actually touch ────────────────────
    _report("Auto-correct: detecting touching-label groups per slice...")
    groups_by_z = touching_groups_for_stack(new_labels)
    group_jobs = [(z, group) for z, groups in groups_by_z.items() for group in groups]
    n_group_slices = len(group_jobs)

    # ── Step 4 (pass 2): joint slice-by-slice correction for those groups ─
    n_groups_corrected = 0
    skipped_groups: "list[dict]" = []
    for done, (z, group) in enumerate(group_jobs, start=1):
        try:
            new_labels, _group_info = correct_label_group_2d(
                new_labels, image, group, z, best_lo, pad=pad, sigma=sigma,
            )
            n_groups_corrected += 1
        except ValueError as exc:
            skipped_groups.append({"z": z, "labels": group, "reason": str(exc)})
        if done % 10 == 0 or done == n_group_slices:
            _report(
                f"Auto-correct: touching-group pass {done}/{n_group_slices} "
                f"(z={z}, labels={group})"
            )

    # ── Step 5: final whole-layer debris cleanup ────────────────────────
    n_debris_removed = 0
    if min_volume is not None:
        threshold = int(round(final_min_fraction * min_volume))
        _report(f"Auto-correct: removing debris below {threshold} vox...")
        new_labels, n_debris_removed = remove_debris(new_labels, threshold)

    report = {
        "best_lo": best_lo,
        "sweep_mean_iou": sweep["best_mean_iou"],
        "n_calibration_samples": sweep["n_samples"],
        "n_cells_total": n_total,
        "n_cells_corrected": n_corrected,
        "skipped_cells": skipped_cells,
        "n_group_slices": n_group_slices,
        "n_groups_corrected": n_groups_corrected,
        "skipped_groups": skipped_groups,
        "n_debris_fragments_removed": n_debris_removed,
    }
    return new_labels.astype(np.int32), report


def format_auto_correction_report(report: dict) -> str:
    lines = []
    lines.append(
        f"Auto-correction: lo={report['best_lo']:.4g} "
        f"(calibration mean IoU={report['sweep_mean_iou']:.3f}, "
        f"{report['n_calibration_samples']} samples)"
    )
    lines.append(
        f"  Cell-by-cell (3D): {report['n_cells_corrected']}/{report['n_cells_total']} corrected"
        + (f", {len(report['skipped_cells'])} skipped" if report["skipped_cells"] else "")
    )
    for lid, reason in report["skipped_cells"].items():
        lines.append(f"    label {lid} skipped: {reason}")
    lines.append(
        f"  Touching-groups (2D): {report['n_groups_corrected']}/{report['n_group_slices']} corrected"
        + (f", {len(report['skipped_groups'])} skipped" if report["skipped_groups"] else "")
    )
    for item in report["skipped_groups"]:
        lines.append(f"    z={item['z']} labels={item['labels']} skipped: {item['reason']}")
    lines.append(f"  Debris removed: {report['n_debris_fragments_removed']} fragment(s)")
    return "\n".join(lines)
