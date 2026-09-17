"""
_auto_correction.py -- Automatic full-stack contrast correction, chained
onto the end of a Cellpose-SAM Segmentation run.

Ties together several tools this plugin already has, in the one order
that makes each of them safe to run completely unattended, on every
cell:

  1. Calibrate Correct-Label Contrast's own self-referential sweep
     (_contrast_sweep.py) -- finds the one intensity threshold `lo` that
     best REPRODUCES what Cellpose-SAM just segmented, no external GT
     needed (this is exactly what that sweep was built for).

  2. Protect Skin as Label, BEFORE any real cell is touched -- seeded
     (seed_skin_label()) and trimmed (trim_skin_label()) at `lo + 1`,
     deliberately one step stricter than the calibrated `lo` used for
     every real cell in step 4 below. This mirrors the interactive
     workflow of dialing the signal layer's contrast lower limit up by
     one before running Protect Skin as Label, then back down to the
     calibrated value before correcting real cells -- here that's just
     two different `lo` arguments to the same underlying calls, since
     this runs headless on raw arrays rather than through a napari
     layer's live contrast slider. Skin becomes a real, ordinary label
     (-1) at this point, so every per-cell correction in step 4 already
     excludes it as foreign territory, by construction -- exactly why
     this has to happen before, not after, the per-cell loop.

  3. Resort every real cell by Centroid Z (resort_labels()) -- so the
     sequential per-cell loop below always walks the fish in the same
     deep-to-shallow (or shallow-to-deep) order, not whatever arbitrary
     order Cellpose-SAM happened to assign label IDs in.

  4. Sequential 3D correction, one cell at a time, in that new
     Centroid-Z order -- grow_correct_label_3d(), the same auto-grow +
     until-stable engine Tab 3's own "Correct Label" (3D mode) button
     uses, not a single plain pass. Its own per-slice walk already
     resolves a genuinely adjacent label (skin or another cell)
     entirely on its own -- see grow_correct_label_3d()'s own
     docstring -- so no separate touching-groups joint pass is needed
     after this loop (the earlier version of this pipeline had one,
     because the plain correct_label_from_intensity_3d() it used here
     instead couldn't do that on its own). Every cell's own report is
     kept and merged into one consolidated report, not one report per
     cell.

  5. A final whole-layer Remove Debris pass (same golden-ratio floor as
     every other final-safety-net stage in this plugin) -- step 4's
     per-cell corrections can each leave a small disconnected sliver
     behind on top of their own already-applied per-label debris
     cleanup.
"""

from __future__ import annotations

import numpy as np

from ._labeling import (
    seed_skin_label,
    trim_skin_label,
    resort_labels,
    remove_debris,
)
from ._grow_correct import grow_correct_label_3d, format_grow_report
from ._contrast_sweep import (
    select_calibration_samples,
    default_lo_candidates,
    sweep_contrast_lower_value,
)


def auto_contrast_correct_stack(
    labels: np.ndarray,
    image: np.ndarray,
    scale_zyx: "tuple[float, float, float]",
    brain_mask: np.ndarray,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    pad: int = 15,
    sigma: float = 1.0,
    n_cells_calib: int = 5,
    slices_per_cell_calib: int = 10,
    n_lo_steps: int = 40,
    edge_margin_um: float = 50.0,
    growth_step: int = 5,
    max_iterations: int = 10,
    until_stable: bool = True,
    max_stability_passes: int = 100,
    auto_grow: bool = True,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Full automatic post-segmentation correction. See the module
    docstring for the 5-step pipeline this runs.

    labels, image      : (Z, Y, X) volumes, same shape -- labels is the
                          just-produced Cellpose-SAM result, image is
                          the raw ExtRm signal it was segmented from
    scale_zyx           : voxel size (µm), only used to pick calibration
                          cells away from the volume's own edge
    brain_mask          : (Z, Y, X) boolean-like array, same shape --
                          same convention as Protect Skin as Label's own
                          (nonzero = brain, kept). Required: skin
                          protection is not optional in this pipeline.
    min_volume          : Common Settings' Min volume (voxels) -- drives
                          both step 4's per-cell debris cleanup and the
                          final whole-layer pass. None skips debris
                          cleanup entirely (report will show 0 removed)
    final_min_fraction  : golden ratio (0.618) by default, matching
                          every other final-safety-net stage
    pad, sigma          : same meaning as every other Correct Label tool
                          -- pad is also reused as-is for the skin trim's
                          own bounding-box padding (step 2)
    n_cells_calib, slices_per_cell_calib, n_lo_steps, edge_margin_um
                        : passed straight through to the contrast sweep
                          (select_calibration_samples / default_lo_candidates)
    growth_step, max_iterations, until_stable, max_stability_passes,
    auto_grow            : forwarded straight through to
                          grow_correct_label_3d() for every cell in step
                          4 -- same meaning as Tab 3's own "Correct
                          Label" (3D mode) auto-grow / until-stable
                          controls.
    progress_cb          : optional callable(str), called with a
                          human-readable status line as each stage/step
                          advances

    Returns (new_labels, report). report is a dict:
        best_lo               -- the calibrated intensity threshold used
                                  for every real cell
        sweep_mean_iou         -- that threshold's own mean IoU against
                                  the calibration samples
        n_calibration_samples  -- how many (label, z) samples the sweep
                                  actually used
        skin_label_id           -- the ID skin was seeded as (-1)
        skin_report              -- trim_skin_label()'s own report dict
                                  for the skin correction
        n_cells_total           -- real cells present after skin
                                  protection + resorting
        n_cells_corrected       -- how many per-cell 3D corrections
                                  actually succeeded
        skipped_cells           -- {label_id: reason} for every per-cell
                                  correction that raised (left as it was
                                  going into this pipeline, never crashes
                                  the whole run)
        cell_reports             -- {label_id: grow_correct_label_3d()'s
                                  own report dict}, one entry per
                                  successfully corrected cell, in
                                  Centroid-Z order
        n_debris_fragments_removed -- fragments cleared by the final pass

    Raises ValueError only for conditions that make the WHOLE run
    meaningless (no labels present at all, shape mismatches, or no cell
    suitable for contrast calibration) -- a failure on any individual
    cell is caught and recorded in the report instead, never aborts the
    run.
    """

    def _report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    labels = np.asarray(labels).astype(np.int32)
    image = np.asarray(image)
    brain_mask = np.asarray(brain_mask).astype(bool)
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")
    if labels.shape != brain_mask.shape:
        raise ValueError(f"labels shape {labels.shape} != brain mask shape {brain_mask.shape}")

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
        f"(mean IoU={sweep['best_mean_iou']:.3f} on {sweep['n_samples']} samples)"
    )

    # ── Step 2: protect skin BEFORE any real cell is touched, at lo+1 ──
    skin_lo = best_lo + 1.0
    _report(f"Auto-correct: protecting skin (lo={skin_lo:.4g})...")
    seeded, skin_id = seed_skin_label(labels, brain_mask)
    labels_with_skin, skin_report = trim_skin_label(
        seeded, image, brain_mask, skin_id, skin_lo, pad=pad, sigma=sigma,
    )

    # ── Step 3: resort every real cell by Centroid Z ────────────────────
    _report("Auto-correct: resorting cells by Centroid Z...")
    new_labels = resort_labels(labels_with_skin, sort_by="centroid_z")

    # ── Step 4: sequential 3D correction, one cell at a time, in that
    #    new Centroid-Z order, back at the calibrated lo (not lo+1) ─────
    unique_ids2 = np.unique(new_labels)
    unique_ids2 = unique_ids2[unique_ids2 > 0]
    n_total = int(unique_ids2.size)
    n_corrected = 0
    skipped_cells: "dict[int, str]" = {}
    cell_reports: "dict[int, dict]" = {}
    for idx, lid in enumerate(unique_ids2.tolist()):
        try:
            new_labels, cell_report = grow_correct_label_3d(
                new_labels, image, lid, best_lo,
                initial_pad=pad, growth_step=growth_step, max_iterations=max_iterations,
                sigma=sigma, min_volume=min_volume, final_min_fraction=final_min_fraction,
                until_stable=until_stable, max_stability_passes=max_stability_passes,
                auto_grow=auto_grow,
            )
            cell_reports[lid] = cell_report
            n_corrected += 1
        except ValueError as exc:
            skipped_cells[lid] = str(exc)
        if idx % 5 == 0 or idx == n_total - 1:
            _report(
                f"Auto-correct: cell-by-cell 3D pass {idx + 1}/{n_total} "
                f"(label {lid}) -- {n_corrected} corrected, "
                f"{len(skipped_cells)} skipped so far"
            )

    # ── Step 5: final whole-layer debris cleanup (skin included) ────────
    n_debris_removed = 0
    if min_volume is not None:
        threshold = int(round(final_min_fraction * min_volume))
        _report(f"Auto-correct: removing debris below {threshold} vox...")
        new_labels, n_debris_removed = remove_debris(new_labels, threshold, skin_label_id=skin_id)

    report = {
        "best_lo": best_lo,
        "sweep_mean_iou": sweep["best_mean_iou"],
        "n_calibration_samples": sweep["n_samples"],
        "skin_label_id": skin_id,
        "skin_report": skin_report,
        "n_cells_total": n_total,
        "n_cells_corrected": n_corrected,
        "skipped_cells": skipped_cells,
        "cell_reports": cell_reports,
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
    skin_report = report["skin_report"]
    lines.append(
        f"  Skin protected as label {report['skin_label_id']} "
        f"(lo={report['best_lo'] + 1.0:.4g}) -- "
        f"{skin_report.get('n_debris_removed_px', 0)} px trimmed as debris, "
        f"{skin_report.get('n_reclaimed_px', 0)} px reclaimed from inside the brain mask."
    )
    lines.append(
        f"  Cells resorted by Centroid Z before correction "
        f"({report['n_cells_total']} cell(s))."
    )
    lines.append("")
    lines.append(
        f"Cell-by-cell 3D correction (Centroid-Z order): "
        f"{report['n_cells_corrected']}/{report['n_cells_total']} corrected"
        + (f", {len(report['skipped_cells'])} skipped" if report["skipped_cells"] else "")
    )
    for lid, reason in report["skipped_cells"].items():
        lines.append(f"  label {lid} skipped: {reason}")
    lines.append("")

    # Per-cell detail, reusing the exact same formatter the interactive
    # Correct Label (3D) button uses -- so a cell corrected here reads
    # identically to one corrected by hand.
    n_converged = 0
    n_not_stable = 0
    total_debris = 0
    total_slices_grown = 0
    total_slices_stability = 0
    for lid in sorted(report["cell_reports"]):
        cell_report = report["cell_reports"][lid]
        lines.append(f"--- Label {lid} ---")
        lines.append(format_grow_report(cell_report, mode="3D"))
        lines.append("")
        if cell_report.get("converged"):
            n_converged += 1
        if not cell_report.get("stable", True):
            n_not_stable += 1
        total_debris += cell_report.get("n_debris_removed_px", 0)
        total_slices_grown += len(cell_report.get("slices_grown", {}))
        total_slices_stability += len(cell_report.get("slices_stability_passes", {}))

    n_reported = len(report["cell_reports"])
    lines.append(
        f"Summary: {n_converged}/{n_reported} cell(s) converged, "
        f"{n_not_stable} cell(s) still changing at the stability-pass cap, "
        f"{total_slices_grown} slice(s) across all cells needed a bigger pad, "
        f"{total_slices_stability} slice(s) took more than one pass to settle, "
        f"{total_debris} px of debris removed during per-cell correction."
    )
    lines.append(f"Final whole-layer debris removed: {report['n_debris_fragments_removed']} fragment(s)")
    return "\n".join(lines)
