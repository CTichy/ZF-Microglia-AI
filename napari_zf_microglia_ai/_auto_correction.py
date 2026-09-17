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
     (seed_skin_label()) and trimmed (trim_skin_label()) at the SAME
     calibrated `lo` real cells get in step 4 below (no longer offset
     by +1 -- that margin existed to keep skin from greedily grabbing
     marginal real-cell-adjacent signal back when skin's own trim only
     ever excluded a touching cell; now that it jointly resolves the
     boundary against one instead, see below, the margin is no longer
     needed). Skin becomes a real, ordinary label (-1) at this point.
     Its own trim uses the same per-slice auto-grow + until-stable
     machinery real cells get in step 4 (see trim_skin_label()'s own
     docstring), and jointly resolves its boundary against any real
     cell it finds along the way -- but only ever writes back skin's
     OWN resulting territory, never the cell's, since at this point in
     the pipeline that cell hasn't been through its own correction yet.
     Step 4's per-cell corrections, once it's each cell's own turn, are
     free to adjust both themselves AND whatever they touch (skin
     included) -- exactly why this has to happen before, not after,
     the per-cell loop: skin adapts to whatever a cell currently looks
     like, then the cell gets the final say once it's actually
     corrected.

  3. Resort every real cell by Centroid Z (resort_labels()) -- gives
     step 4's wave partitioning below a stable, deterministic starting
     order to break ties from, not whatever arbitrary order Cellpose-
     SAM happened to assign label IDs in.

  4. 3D correction, batched into parallel-safe WAVES instead of one
     cell strictly at a time -- grow_correct_label_3d(), the same
     auto-grow + until-stable engine Tab 3's own "Correct Label" (3D
     mode) button uses. Its own per-slice walk already resolves a
     genuinely adjacent label (skin or another cell) entirely on its
     own -- see grow_correct_label_3d()'s own docstring -- so no
     separate touching-groups joint pass is needed after this loop.
     Two cells whose own maximum-possible working areas (bounding box
     + auto-grow's own worst-case reach) can never overlap are
     provably unable to interact, so they're grouped into the same
     wave and corrected concurrently on a ThreadPoolExecutor (capped
     at 75% of CPU cores); cells that CAN conflict are pushed into
     later waves, which still see every earlier wave's own already-
     corrected state -- preserving the same "resolve against an
     already-corrected neighbor" behavior the old strictly-sequential
     loop had, just at wave granularity instead of one cell at a time.
     See _compute_correction_waves()'s own docstring for the exact
     partitioning scheme. Every cell's own report is kept and merged
     into one consolidated report, not one report per cell.

  5. A final whole-layer Remove Debris pass (same golden-ratio floor as
     every other final-safety-net stage in this plugin) -- step 4's
     per-cell corrections can each leave a small disconnected sliver
     behind on top of their own already-applied per-label debris
     cleanup.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import find_objects

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


def _compute_correction_waves(
    labels: np.ndarray,
    cell_ids: "list[int]",
    pad: int,
    growth_step: int,
    max_iterations: int,
    auto_grow: bool,
) -> "tuple[list[list[int]], dict[int, tuple[int, int, int, int, int, int]]]":
    """
    Partitions cell_ids into sequential WAVES -- groups where every cell
    is guaranteed to never spatially interact with any other cell in
    the same wave -- so every cell in one wave can be corrected in
    parallel, safely, on a shared labels array. Two cells only need to
    stay in different waves if their own maximum-POSSIBLE working areas
    could ever overlap; confirmed on 2 real, densely-packed fish (78
    and 88 cells, one with literally zero fully-isolated cells) that
    this still yields real parallelism -- crowding increases the number
    of waves needed, not whether batching helps at all.

    "Maximum possible working area" = each cell's own bounding box,
    expanded by `reach` on every side: pad + growth_step * max_iterations
    when auto_grow is on (0 otherwise) -- a strict upper bound on how
    far correct_label_from_intensity_3d()'s own per-slice auto-grow can
    ever push a single slice's working window (see that function's own
    auto_grow docstring: use_pad starts at `pad` and grows by
    growth_step at most max_iterations-1 more times before the attempt
    cap is hit). Z-extension beyond a cell's own original range is
    capped by z_extent_pad (== pad by default, never multiplied by
    growth), so using the same (larger) Y/X reach for Z too is
    deliberately conservative, not exact -- erring toward fewer, safer
    waves rather than a tighter but riskier bound.

    Cells across DIFFERENT waves still see each other's already-
    corrected state, in wave order -- preserving the same "resolve
    against an already-corrected neighbor" behavior a fully-sequential,
    one-cell-at-a-time loop would have, just at wave granularity
    instead of strictly one cell at a time.

    Greedy, not globally optimal: repeatedly extracts the largest
    mutually-non-conflicting set from whichever cells remain unassigned,
    processing candidates in ascending order of how many others they
    conflict with (a lightly-conflicted cell is more likely to still
    fit into whatever's already been added to the current wave).

    Returns (waves, boxes) -- boxes is {cell_id: (z0,z1,y0,y1,x0,x1)},
    each cell's own expanded working-area bounds, reused by the caller
    to merge a worker thread's single-cell result back into the shared
    array (safe precisely because wave members' own boxes never
    overlap).
    """
    max_lbl = int(labels.max()) if labels.size else 0
    objs = find_objects(labels, max_label=max_lbl)
    reach = pad + (growth_step * max_iterations if auto_grow else 0)
    Z, Y, X = labels.shape

    boxes: "dict[int, tuple[int, int, int, int, int, int]]" = {}
    for lid in cell_ids:
        sl = objs[lid - 1] if 0 < lid <= len(objs) else None
        if sl is None:
            continue
        z0, z1 = sl[0].start, sl[0].stop
        y0, y1 = sl[1].start, sl[1].stop
        x0, x1 = sl[2].start, sl[2].stop
        boxes[lid] = (
            max(z0 - reach, 0), min(z1 + reach, Z),
            max(y0 - reach, 0), min(y1 + reach, Y),
            max(x0 - reach, 0), min(x1 + reach, X),
        )

    def _overlaps(a, b) -> bool:
        az0, az1, ay0, ay1, ax0, ax1 = a
        bz0, bz1, by0, by1, bx0, bx1 = b
        return not (
            az1 <= bz0 or bz1 <= az0
            or ay1 <= by0 or by1 <= ay0
            or ax1 <= bx0 or bx1 <= ax0
        )

    ids = list(boxes.keys())
    conflicts: "dict[int, set[int]]" = {lid: set() for lid in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if _overlaps(boxes[ids[i]], boxes[ids[j]]):
                conflicts[ids[i]].add(ids[j])
                conflicts[ids[j]].add(ids[i])

    order = sorted(ids, key=lambda lid: len(conflicts[lid]))
    remaining = set(ids)
    waves: "list[list[int]]" = []
    while remaining:
        wave: "list[int]" = []
        blocked: "set[int]" = set()
        for lid in order:
            if lid not in remaining or lid in blocked:
                continue
            wave.append(lid)
            blocked.add(lid)
            blocked |= conflicts[lid]
        for lid in wave:
            remaining.discard(lid)
        waves.append(wave)
    return waves, boxes


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
        n_skin_debris_removed_px -- px of stray skin debris swept up
                                  right after protection (see the debris
                                  pass right after step 2, below)
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

    # ── Step 2: protect skin BEFORE any real cell is touched, at the
    #    same calibrated lo real cells get in step 4 (no longer +1 --
    #    skin's own trim jointly resolves against a touching cell now,
    #    instead of just excluding it, so that safety margin is no
    #    longer needed) ───────────────────────────────────────────────
    _report(f"Auto-correct: protecting skin (lo={best_lo:.4g})...")
    seeded, skin_id = seed_skin_label(labels, brain_mask)
    labels_with_skin, skin_report = trim_skin_label(
        seeded, image, skin_id, best_lo, pad=pad, sigma=sigma,
        auto_grow=auto_grow, growth_step=growth_step, max_iterations=max_iterations,
        until_stable=until_stable, max_stability_passes=max_stability_passes,
    )

    # trim_skin_label() no longer clamps against the brain mask (that
    # clamp used to block legitimate inner-boundary correction manual
    # Correct Label never had to fight -- see its own docstring) -- a
    # debris pass right here, before any real cell's own turn, sweeps
    # up whatever small stray blob skin absorbed instead, by size alone
    # rather than a hard "never inside the brain" rule.
    n_skin_debris_removed = 0
    if min_volume is not None:
        threshold = int(round(final_min_fraction * min_volume))
        _report(f"Auto-correct: removing debris skin absorbed (below {threshold} vox)...")
        labels_with_skin, n_skin_debris_removed = remove_debris(
            labels_with_skin, threshold, skin_label_id=skin_id,
        )

    # ── Step 3: resort every real cell by Centroid Z ────────────────────
    _report("Auto-correct: resorting cells by Centroid Z...")
    new_labels = resort_labels(labels_with_skin, sort_by="centroid_z")

    # ── Step 4: 3D correction, batched into parallel-safe waves ─────────
    # See _compute_correction_waves()'s own docstring for the exact
    # partitioning scheme and why it's safe.
    unique_ids2 = np.unique(new_labels)
    unique_ids2 = unique_ids2[unique_ids2 > 0]
    n_total = int(unique_ids2.size)
    n_corrected = 0
    n_done = 0
    skipped_cells: "dict[int, str]" = {}
    cell_reports: "dict[int, dict]" = {}

    waves, boxes = _compute_correction_waves(
        new_labels, unique_ids2.tolist(), pad, growth_step, max_iterations, auto_grow,
    )
    n_workers = max(1, int((os.cpu_count() or 4) * 0.75))
    _report(
        f"Auto-correct: {n_total} cell(s) split into {len(waves)} "
        f"parallel-safe wave(s) (up to {n_workers} cell(s) at once)..."
    )

    for wave_idx, wave in enumerate(waves):
        wave_snapshot = new_labels  # read-only for this wave -- each
        # worker's own grow_correct_label_3d() call copies it internally
        # before mutating, so concurrent reads here are safe; nothing
        # writes to new_labels itself until every worker in this wave
        # has finished and its own single-cell result is merged back
        # below, so no wave-mate ever sees a partially-updated array.

        def _correct_one(lid, _snapshot=wave_snapshot):
            try:
                result_labels, cell_report = grow_correct_label_3d(
                    _snapshot, image, lid, best_lo,
                    initial_pad=pad, growth_step=growth_step, max_iterations=max_iterations,
                    sigma=sigma, min_volume=min_volume, final_min_fraction=final_min_fraction,
                    until_stable=until_stable, max_stability_passes=max_stability_passes,
                    auto_grow=auto_grow,
                )
                return lid, result_labels, cell_report, None
            except ValueError as exc:
                return lid, None, None, str(exc)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            wave_results = list(pool.map(_correct_one, wave))

        for lid, result_labels, cell_report, err in wave_results:
            n_done += 1
            if err is not None:
                skipped_cells[lid] = err
                continue
            # Wave members' own boxes never overlap (that's the whole
            # point of the partition), so copying just this cell's own
            # expanded working area back is safe and unambiguous --
            # every other voxel in result_labels is identical to
            # wave_snapshot anyway (this cell's own correction couldn't
            # have reached beyond its own box).
            z0, z1, y0, y1, x0, x1 = boxes[lid]
            new_labels[z0:z1, y0:y1, x0:x1] = result_labels[z0:z1, y0:y1, x0:x1]
            cell_reports[lid] = cell_report
            n_corrected += 1

        _report(
            f"Auto-correct: wave {wave_idx + 1}/{len(waves)} done "
            f"({len(wave)} cell(s), {n_done}/{n_total} total) -- "
            f"{n_corrected} corrected, {len(skipped_cells)} skipped so far"
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
        "n_skin_debris_removed_px": n_skin_debris_removed,
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
        f"(lo={report['best_lo']:.4g}) -- "
        f"{report.get('n_skin_debris_removed_px', 0)} px of stray skin debris removed."
    )
    skin_slices_grown = skin_report.get("slices_grown", {})
    skin_slices_stability = skin_report.get("slices_stability_passes", {})
    if skin_slices_grown:
        lines.append(f"  Skin: {len(skin_slices_grown)} slice(s) needed a bigger pad to grow.")
    if skin_slices_stability:
        lines.append(f"  Skin: {len(skin_slices_stability)} slice(s) took more than one pass to settle.")
    if not skin_report.get("stable", True):
        lines.append(
            "  Skin: STILL CHANGING on at least one slice -- hit the stability-pass "
            "cap without settling; a larger cap may let it finish converging."
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
