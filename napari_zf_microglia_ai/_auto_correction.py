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

  2. Protect Skin as Label, BEFORE any real cell is touched -- unless the
     labels handed in ALREADY carry a skin label (-1), in which case it
     is reused exactly as it is and never re-seeded/re-trimmed. Either
     way the pipeline then VERIFIES skin is present before it corrects
     a single cell (see the "Gate" comment in step 2-3) and refuses to
     continue if it is not. When protecting, skin is seeded
     (seed_skin_label()) and trimmed (trim_skin_label()) at the SAME
     calibrated `lo` real cells get in step 5 below (no longer offset
     by +1 -- that margin existed to keep skin from greedily grabbing
     marginal real-cell-adjacent signal back when skin's own trim only
     ever excluded a touching cell; now that it jointly resolves the
     boundary against one instead, see below, the margin is no longer
     needed), at its OWN dedicated padding (skin_pad, wider than a real
     cell's own pad by default -- skin's own bounding box already spans
     nearly the whole frame on most slices, so a slightly bigger
     starting window costs little and leaves less for auto-grow to have
     to do). Skin becomes a real, ordinary label (-1) at this point. Its
     own trim uses the same per-slice auto-grow + until-stable
     machinery real cells get below (see trim_skin_label()'s own
     docstring), and jointly resolves its boundary against any real
     cell it finds along the way -- but only ever writes back skin's
     OWN resulting territory, never the cell's, since at this point in
     the pipeline that cell hasn't been through its own correction yet.
     No brain-mask clamp any more (see trim_skin_label()'s own
     docstring) -- step 3 below sweeps up whatever small stray blob it
     absorbed instead.

  3. Remove Debris right after skin protection, before any real cell is
     touched -- catches whatever small stray blob skin's own unclamped
     trim absorbed, by size alone (same golden-ratio floor as every
     other debris pass in this plugin).

  4. Figure out which real cells touch skin -- by GEOMETRY of skin's
     final territory (after the trim and its debris pass): a cell
     "touches" skin if, on any slice, some pixel of it is within
     skin_touch_px pixels (default 3, in-plane) of a skin pixel --
     see _labels_touching_skin()'s own docstring for why this is NOT
     read from trim_skin_label()'s foreign_nearby report any more (that
     report lists every label ever folded into a growth attempt, which
     flagged all 33 of 33 cells of a real fish although none was within
     45 px of skin's real territory). THEN partition every real cell into
     parallel-safe WAVES (_compute_correction_waves() -- two cells
     whose own maximum-possible working areas can never overlap are
     grouped into the same wave and corrected concurrently on a
     ThreadPoolExecutor capped at 75% of CPU cores; cells that CAN
     conflict are pushed into later waves, which still see every
     earlier wave's own already-corrected state). This one wave
     partition covers every real cell, touching skin or not -- both
     correction modes below share the identical worst-case reach
     formula, so one shared conflict graph is exact for both.

  5. Each cell's own correction, mode chosen by whether it touches skin:
       - Touching skin: corrected in 2D, slice by slice, jointly
         against skin -- correct_label_2d_stack() (label_id=the cell),
         the SAME per-slice mechanism trim_skin_label() itself uses,
         just now writing back the CELL's own side instead of skin's.
         This is deliberate, not incidental: skin can only ever be
         corrected in 2D (see trim_skin_label()'s own docstring), so a
         cell meeting it has to do so on skin's own per-slice terms --
         running the cell's usual 3D cross-slice walk here would let
         the cell's own boundary drift across slices in ways skin's
         fixed-per-slice shape structurally can't reciprocate.
       - Not touching skin: corrected in 3D as before --
         grow_correct_label_3d(), the same auto-grow + until-stable
         engine Tab 3's own "Correct Label" (3D mode) button uses. Its
         own per-slice walk already resolves a genuinely adjacent
         label (another cell) entirely on its own, so no separate
         touching-groups joint pass is needed here either.
     Every cell's own report is kept and merged into one consolidated
     report, not one report per cell.

  6. A final whole-layer Remove Debris pass (same golden-ratio floor as
     every other final-safety-net stage in this plugin) -- step 5's
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
    skin_voxel_count,
    correct_label_2d_stack,
    resort_labels,
    remove_debris,
)
from ._grow_correct import grow_correct_label_3d, format_grow_report
from ._contrast_sweep import (
    select_calibration_samples,
    default_lo_candidates,
    sweep_contrast_lower_value,
)


def _labels_touching_skin(
    labels: np.ndarray,
    cell_ids: "list[int]",
    skin_id: int,
    max_gap_px: int,
) -> "set[int]":
    """
    Every real cell within max_gap_px pixels of skin's FINAL territory on
    at least one slice -- pure geometry on the labels array as it stands
    (after skin's trim and debris pass), nothing inferred from how the
    trim got there.

    "Within max_gap_px" means at most max_gap_px EMPTY pixels between the
    cell and skin, in-plane (each slice is independent -- skin is
    corrected per-slice in 2D, so 2D distance is what matters): the cell
    is dilated by max_gap_px + 1 with a 3x3 structure (Chebyshev, the
    same neighborhood every other adjacency test in this plugin uses),
    and it counts if that reaches any skin pixel. max_gap_px=0 is plain
    adjacency.

    Why not read trim_skin_label()'s foreign_nearby report, as this
    function used to: that report lists every label that was EVER folded
    into a skin-growth attempt on any slice. Skin's first attempt on a
    slice floods every above-threshold pixel of the (whole-slice) crop --
    including the halo around every cell -- so every cell gets folded in
    for a moment, and stays listed after the joint split hands those
    pixels back. Confirmed on a real fish (NT36-3dpf D1F4): all 33 of 33
    cells flagged, yet post-trim adjacency was 0 for every one and the
    nearest real skin was 45-950 px away. And the old alternative -- a
    strict 0-px adjacency scan -- misses a cell that the shared
    _clear_split_interface() convention or a sub-threshold band leaves
    1-2 px from skin, which is a genuine "touching" case.

    Each cell only ever examines its own bounding box (grown by the gap),
    so this stays cheap on a full fish.
    """
    from scipy.ndimage import binary_dilation

    Z, Y, X = labels.shape
    max_lbl = int(max(cell_ids)) if cell_ids else 0
    objs = find_objects(labels, max_label=max_lbl)
    reach = int(max_gap_px) + 1
    struct = np.ones((1, 3, 3), dtype=bool)  # in-plane only: slices are independent
    touching: "set[int]" = set()
    for lid in cell_ids:
        sl = objs[lid - 1] if 0 < lid <= len(objs) else None
        if sl is None:
            continue
        y0 = max(sl[1].start - reach, 0); y1 = min(sl[1].stop + reach, Y)
        x0 = max(sl[2].start - reach, 0); x1 = min(sl[2].stop + reach, X)
        crop = labels[sl[0], y0:y1, x0:x1]
        skin = crop == skin_id
        if not skin.any():
            continue
        grown = binary_dilation(crop == lid, structure=struct, iterations=reach)
        if (grown & skin).any():
            touching.add(int(lid))
    return touching


def _flag_possible_non_microglia(
    labels: np.ndarray,
    brain_mask: np.ndarray,
    cell_ids: "list[int]",
    skin_id: int,
    skin_touch_px: int,
    fraction: float,
) -> "dict[int, dict]":
    """
    REPORT-ONLY screen for blobs that are probably not microglia:
    macrophages (usually outside the skin, or lying over it) and skin
    that MONAI did not remove and Cellpose then segmented as a cell.
    Never changes a label -- it only returns what to mention.

    A cell is flagged when EITHER criterion reaches `fraction` (0-1):

      outside_brain -- share of the cell's VOXELS lying outside the brain
                       mask (the cell sits beyond the brain boundary, or
                       mostly over it).
      skin_contact  -- share of the cell's own boundary SURFACE (its outer
                       shell of voxels: the cell minus its 3D erosion) that
                       is within skin_touch_px pixels, in-plane, of skin --
                       the same reach the "touching skin" rule uses (a cell
                       wrapped against skin even while inside the mask).

    Returns {label_id: {"outside_brain_frac", "skin_surface_frac",
    "reasons": ["outside_brain" and/or "skin_contact"], "volume_vox",
    "centroid_zyx"}} for flagged cells only. Each cell only examines its
    own bounding box, so this stays cheap on a full fish.
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    Z, Y, X = labels.shape
    max_lbl = int(max(cell_ids)) if cell_ids else 0
    objs = find_objects(labels, max_label=max_lbl)
    reach = int(skin_touch_px) + 1
    struct2d = np.ones((1, 3, 3), dtype=bool)
    flagged: "dict[int, dict]" = {}
    for lid in cell_ids:
        sl = objs[lid - 1] if 0 < lid <= len(objs) else None
        if sl is None:
            continue
        z0, z1 = sl[0].start, sl[0].stop
        y0 = max(sl[1].start - reach, 0); y1 = min(sl[1].stop + reach, Y)
        x0 = max(sl[2].start - reach, 0); x1 = min(sl[2].stop + reach, X)
        crop = labels[z0:z1, y0:y1, x0:x1]
        cell = crop == lid
        n_vox = int(cell.sum())
        if n_vox == 0:
            continue

        outside_frac = float((cell & ~brain_mask[z0:z1, y0:y1, x0:x1]).sum()) / n_vox

        skin_frac = 0.0
        skin = crop == skin_id
        if skin.any():
            shell = cell & ~binary_erosion(cell)
            n_shell = int(shell.sum())
            if n_shell:
                near_skin = binary_dilation(skin, structure=struct2d, iterations=reach)
                skin_frac = float((shell & near_skin).sum()) / n_shell

        reasons = []
        if outside_frac >= fraction:
            reasons.append("outside_brain")
        if skin_frac >= fraction:
            reasons.append("skin_contact")
        if reasons:
            zz, yy, xx = np.nonzero(cell)
            flagged[int(lid)] = {
                "outside_brain_frac": outside_frac,
                "skin_surface_frac": skin_frac,
                "reasons": reasons,
                "volume_vox": n_vox,
                "centroid_zyx": (float(zz.mean()) + z0, float(yy.mean()) + y0, float(xx.mean()) + x0),
            }
    return flagged


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
    skin_image: "np.ndarray | None" = None,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
    pad: int = 15,
    skin_pad: int = 25,
    skin_touch_px: int = 3,
    non_microglia_fraction: float = 0.40,
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
    lo_override: "float | None" = None,
    lo_adjustment: float = 0.0,
    progress_cb=None,
) -> "tuple[np.ndarray, dict]":
    """
    Full automatic post-segmentation correction. See the module
    docstring for the 6-step pipeline this runs.

    labels, image      : (Z, Y, X) volumes, same shape -- labels is the
                          just-produced Cellpose-SAM result, image is
                          the raw ExtRm signal it was segmented from
    scale_zyx           : voxel size (µm), only used to pick calibration
                          cells away from the volume's own edge
    brain_mask          : (Z, Y, X) boolean-like array, same shape --
                          same convention as Protect Skin as Label's own
                          (nonzero = brain, kept). Required: skin
                          protection is not optional in this pipeline.
    skin_image          : optional (Z, Y, X) array, same shape as image --
                          used ONLY for skin's own trim (step 2), in
                          place of `image`. `image` (the just-segmented
                          signal) is often background-processed in a way
                          that zeroes everything outside the brain mask
                          by design (Tab 1 Background mode 1 "_ExtRm" or
                          mode 2 "_NoBG" -- see _background.py), which
                          leaves nothing for skin's own trim to find no
                          matter what threshold is used -- and neither
                          does a "Background=Off" brain_only, despite
                          the name: run_inference() always returns
                          volume * eroded_mask regardless of bg_mode
                          (see _inference.py's own docstring), "Off"
                          only skips the EXTRA background-threshold
                          pass. Pass the raw, never-masked imported
                          channel itself here when available (or any
                          other image that still retains real
                          outside-brain signal). None (default) falls
                          back to `image` itself, matching this
                          function's behavior before this parameter
                          existed.
    min_volume          : Common Settings' Min volume (voxels) -- drives
                          both the debris pass right after skin
                          protection and the final whole-layer pass.
                          None skips debris cleanup entirely (report
                          will show 0 removed everywhere)
    final_min_fraction  : golden ratio (0.618) by default, matching
                          every other final-safety-net stage
    pad, sigma          : same meaning as every other Correct Label tool
                          -- pad is used for every real cell's own
                          correction (step 5), both the 2D-vs-skin and
                          3D modes
    skin_pad             : the STARTING pad for skin's own trim (step
                          2), separate from `pad` above -- skin's own
                          bounding box already spans nearly the whole
                          frame on most slices (see trim_skin_label()'s
                          own docstring), so it gets its own, wider
                          default rather than sharing a real cell's
                          tighter one
    skin_touch_px        : a real cell counts as "touching skin" (and is
                          corrected in 2D against it, step 5) if some
                          pixel of it, on any slice, has at most this
                          many empty pixels between it and skin's final
                          territory -- 0 = plain adjacency. Default 3.
                          Measured on the final skin mask, not inferred
                          from the skin trim's own group discovery (see
                          _labels_touching_skin()).
    non_microglia_fraction : REPORT-ONLY. A cell is listed in the report as a
                          "possible non-microglia blob" (macrophage, or
                          skin MONAI left behind that Cellpose segmented)
                          when at least this share (0-1, default 0.40) of
                          its voxels lies outside the brain mask, OR of its
                          boundary surface is within skin_touch_px of skin.
                          Nothing is removed or altered -- see
                          _flag_possible_non_microglia().
    n_cells_calib, slices_per_cell_calib, n_lo_steps, edge_margin_um
                        : passed straight through to the contrast sweep
                          (select_calibration_samples / default_lo_candidates)
    growth_step, max_iterations, until_stable, max_stability_passes,
    auto_grow            : forwarded straight through to both skin's own
                          trim (step 2) and every real cell's own
                          correction (step 5, both modes) -- same
                          meaning as Tab 3's own "Correct Label"
                          auto-grow / until-stable controls.
    lo_override          : if given, SKIPS the contrast sweep (step 1)
                          entirely and uses this value directly as
                          `best_lo` for everything downstream -- the
                          "use the signal layer's current low contrast
                          limit instead of auto-sweeping" path. Mutually
                          exclusive with `lo_adjustment` in effect: an
                          override is used exactly as given, never
                          further adjusted.
    lo_adjustment        : only applied when `lo_override` is None (the
                          sweep DID run). Subtracted from the swept
                          `best_lo` before it's used for anything --
                          "auto-sweep, but capture N units of fainter
                          signal than the sweep itself would pick,"
                          since the sweep optimizes for reproducing
                          Cellpose-SAM's own existing footprint, not for
                          recovering real signal the model may have
                          under-segmented. Default 0.0 (no change).
                          Report's `best_lo_swept` always holds the
                          RAW sweep result even when this shifts the
                          value actually used (`best_lo`).
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
        n_skin_debris_fragments_removed -- fragments of stray skin
                                  debris swept up right after protection
                                  (step 3) -- a FRAGMENT count, not a
                                  pixel count (matches remove_debris()'s
                                  own return contract -- the pre-fix
                                  version of this report mislabeled it
                                  "px", which on a fish where skin ended
                                  up fully fragmented made a complete
                                  wipeout of skin read as a trivial "N px
                                  removed" debris cleanup)
        touching_skin_cell_ids -- sorted [label_id, ...] -- every real
                                  cell corrected in 2D against skin
                                  (step 5's first mode) rather than 3D
        n_cells_total           -- real cells present after skin
                                  protection + resorting
        n_cells_corrected       -- how many per-cell corrections
                                  actually succeeded (either mode)
        skipped_cells           -- {label_id: reason} for every per-cell
                                  correction that raised (left as it was
                                  going into this pipeline, never crashes
                                  the whole run)
        cell_reports             -- {label_id: report dict}, one entry
                                  per successfully corrected cell, in
                                  Centroid-Z order -- correct_label_2d_
                                  stack()'s own report shape for a
                                  touching-skin cell (also tagged with
                                  "_mode": "2d_vs_skin"), or
                                  grow_correct_label_3d()'s own report
                                  shape otherwise (tagged "_mode": "3d")
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
    # Skipped entirely when lo_override is given -- see that parameter's
    # own docstring. sweep stays None in that case; every report field
    # that would normally read from it falls back to a value that says
    # plainly "the sweep didn't run" rather than crashing on a missing key.
    sweep = None
    best_lo_swept = None
    if lo_override is not None:
        best_lo = float(lo_override)
        _report(f"Auto-correct: using given lo={best_lo:.4g} (sweep skipped).")
    else:
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
        best_lo_swept = float(sweep["best_lo"])
        best_lo = best_lo_swept - float(lo_adjustment)
        adj_note = f" (swept {best_lo_swept:.4g}, adjusted by -{lo_adjustment:.4g})" if lo_adjustment else ""
        _report(
            f"Auto-correct: calibrated lo={best_lo:.4g}{adj_note} "
            f"(mean IoU={sweep['best_mean_iou']:.3f} on {sweep['n_samples']} samples)"
        )

    # ── Step 2: protect skin BEFORE any real cell is touched, at the
    #    SAME calibrated best_lo real cells get (reusing it, not a
    #    separately-computed skin threshold -- confirmed directly by
    #    the user: skin's real intensity range is approximately the
    #    same as real cells' own, since the contrast sweep already
    #    adapts best_lo per-fish to capture even faint cells Cellpose-
    #    SAM recognizes -- the SAME reasoning applies to skin. A
    #    separate Otsu-based threshold was tried and confirmed WRONG on
    #    real data: this fish's outside-brain intensity histogram has
    #    no real second mode (93% of voxels sit in one narrow spike,
    #    then a smoothly decaying tail with no bump) -- Otsu just
    #    chopped an arbitrary point off that tail (0.095% of voxels
    #    survived, visually nothing), nowhere near where the user could
    #    actually see real skin structure (~90-115). The REAL bug was
    #    only ever the image source (`image` here is often "_ExtRm"/
    #    "_NoBG", zeroed outside the brain mask by design -- see
    #    skin_image's own docstring above) -- best_lo was fine all
    #    along once given real signal to work with ─────────────────────
    skin_id = -1
    n_skin_existing = skin_voxel_count(labels, skin_id)
    if n_skin_existing > 0:
        # Skin is ALREADY protected on the labels handed in (e.g. the user
        # clicked Protect Skin as Label first, or loaded a labels file that
        # was protected in an earlier session) -- never seed/trim it a
        # second time: re-protecting would re-trim a territory the user
        # may have already inspected or hand-corrected, and (before this
        # check existed) re-seeded on top of it. Reuse it exactly as it is.
        _report(
            f"Auto-correct: skin already protected (label {skin_id}, "
            f"{n_skin_existing:,} voxels) -- reusing it, skipping skin protection."
        )
        labels_with_skin = labels
        skin_report = {
            "already_protected": True,
            "n_skin_voxels_reused": n_skin_existing,
            "foreign_touching": {}, "foreign_nearby": {},
            "slices_grown": {}, "slices_stability_passes": {},
            "stable": True, "converged": True, "slices_not_converged": [],
            "n_debris_removed_px": 0,
        }
        n_skin_debris_fragments_removed = 0
    else:
        skin_src = skin_image if skin_image is not None else image
        _report(f"Auto-correct: protecting skin (lo={best_lo:.4g})...")
        seeded, skin_id = seed_skin_label(labels, brain_mask)
        labels_with_skin, skin_report = trim_skin_label(
            seeded, skin_src, skin_id, best_lo, pad=skin_pad, sigma=sigma,
            auto_grow=auto_grow, growth_step=growth_step, max_iterations=max_iterations,
            until_stable=until_stable, max_stability_passes=max_stability_passes,
        )
        skin_report["already_protected"] = False

        # ── Step 3: remove debris skin just absorbed, right here, before
        #    any real cell's own turn -- trim_skin_label() no longer clamps
        #    against the brain mask (that clamp used to block legitimate
        #    inner-boundary correction manual Correct Label never had to
        #    fight -- see its own docstring), so this is what actually
        #    catches a small stray blob it picked up along the way. Counts
        #    FRAGMENTS removed, not pixels -- see remove_debris()'s own
        #    docstring; do not relabel this "_px" again ─────────────────
        n_skin_debris_fragments_removed = 0
        if min_volume is not None:
            threshold = int(round(final_min_fraction * min_volume))
            _report(f"Auto-correct: removing debris skin absorbed (below {threshold} vox)...")
            labels_with_skin, n_skin_debris_fragments_removed = remove_debris(
                labels_with_skin, threshold, skin_label_id=skin_id,
            )

    # ── Gate: NO cell is corrected unless skin is verifiably protected
    #    on the array about to be corrected -- whichever branch above got
    #    us here. A protection that ended up with nothing (e.g. a brain
    #    mask covering the whole frame, so there is no outside to seed,
    #    or every skin fragment swept as debris) would silently let every
    #    cell's correction bleed into skin residue -- the exact failure
    #    skin protection exists to prevent -- so this refuses instead. ──
    n_skin_final = skin_voxel_count(labels_with_skin, skin_id)
    if n_skin_final == 0:
        raise ValueError(
            "skin is not protected (no voxel carries the skin label after "
            "protection) -- refusing to auto-correct cells without it. "
            "Check that the brain mask really excludes skin/outside tissue "
            "and that the signal layer used for skin still has signal "
            "outside the brain."
        )

    _report("Auto-correct: resorting cells by Centroid Z...")
    new_labels = resort_labels(labels_with_skin, sort_by="centroid_z")

    # ── Step 4: which cells touch skin's final territory (geometry, see
    #    _labels_touching_skin()'s own docstring), THEN partition every
    #    real cell into parallel-safe waves ────────────────────────────
    # See _compute_correction_waves()'s own docstring for the exact
    # partitioning scheme and why it's safe -- one shared wave partition
    # covers every real cell, touching skin or not, since both
    # correction modes in step 5 share the identical worst-case reach
    # formula (pad + growth_step*max_iterations).
    touching_skin_ids = _labels_touching_skin(
        new_labels, [int(i) for i in np.unique(new_labels) if i > 0], skin_id, skin_touch_px,
    )
    _report(f"Auto-correct: {len(touching_skin_ids)} cell(s) within {skin_touch_px}px of skin.")

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
        f"Auto-correct: {n_total} cell(s) "
        + (f"({len(touching_skin_ids)} within {skin_touch_px}px of skin, corrected in 2D against it, "
           f"the rest in 3D) " if touching_skin_ids else "(none near skin, all corrected in 3D) ")
        + f"split into {len(waves)} parallel-safe "
        f"wave(s) (up to {n_workers} thread(s) at once)..."
    )

    # ── Step 5: each cell's own correction -- 2D against skin for a
    #    touching cell, 3D otherwise ─────────────────────────────────────
    for wave_idx, wave in enumerate(waves):
        wave_snapshot = new_labels  # read-only for this wave -- each
        # worker's own correction call copies it internally before
        # mutating, so concurrent reads here are safe; nothing writes to
        # new_labels itself until every worker in this wave has
        # finished and its own single-cell result is merged back below,
        # so no wave-mate ever sees a partially-updated array.

        # ONE shared thread budget (n_workers = 75% of cores), split
        # between cells-in-this-wave (outer) and slices-within-a-cell
        # (inner), never multiplied: a wave of 12 cells on 6 cores runs
        # 6 cells x 1 slice-thread; a wave of 1 cell runs 1 cell x 6
        # slice-threads. The old code gave every cell its own full
        # 75%-of-cores slice pool on top of the cell pool (6 x 6 = 36
        # concurrent slice corrections), each allocating a full-volume
        # copy -- that's what exhausted 119 GB of RAM on 2026-09-18.
        n_outer = max(1, min(n_workers, len(wave)))
        n_inner = max(1, n_workers // n_outer)

        def _correct_one(lid, _snapshot=wave_snapshot, _n_inner=n_inner):
            try:
                if lid in touching_skin_ids:
                    result_labels, cell_report = correct_label_2d_stack(
                        _snapshot, image, lid, best_lo,
                        pad=pad, sigma=sigma, auto_grow=auto_grow,
                        growth_step=growth_step, max_iterations=max_iterations,
                        until_stable=until_stable, max_stability_passes=max_stability_passes,
                        n_workers=_n_inner,
                    )
                    cell_report = dict(cell_report)
                    cell_report["_mode"] = "2d_vs_skin"
                else:
                    result_labels, cell_report = grow_correct_label_3d(
                        _snapshot, image, lid, best_lo,
                        initial_pad=pad, growth_step=growth_step, max_iterations=max_iterations,
                        sigma=sigma, min_volume=min_volume, final_min_fraction=final_min_fraction,
                        until_stable=until_stable, max_stability_passes=max_stability_passes,
                        auto_grow=auto_grow,
                    )
                    cell_report["_mode"] = "3d"
                # Hand back ONLY this cell's own box, then let the full-
                # volume result_labels go out of scope right here. The
                # merge below only ever reads this box anyway (see
                # _compute_correction_waves()'s own docstring), but the
                # old code returned the whole array, so `list(pool.map())`
                # kept one full-volume copy alive per cell in the wave
                # until the wave's last cell finished.
                z0, z1, y0, y1, x0, x1 = boxes[lid]
                box_result = result_labels[z0:z1, y0:y1, x0:x1].copy()
                del result_labels
                return lid, box_result, cell_report, None
            except ValueError as exc:
                return lid, None, None, str(exc)

        with ThreadPoolExecutor(max_workers=n_outer) as pool:
            wave_results = list(pool.map(_correct_one, wave))

        for lid, box_result, cell_report, err in wave_results:
            n_done += 1
            if err is not None:
                skipped_cells[lid] = err
                continue
            # Wave members' own boxes never overlap (that's the whole
            # point of the partition), so copying just this cell's own
            # expanded working area back is safe and unambiguous --
            # every other voxel of its result was identical to
            # wave_snapshot anyway (this cell's own correction couldn't
            # have reached beyond its own box).
            z0, z1, y0, y1, x0, x1 = boxes[lid]
            new_labels[z0:z1, y0:y1, x0:x1] = box_result
            cell_reports[lid] = cell_report
            n_corrected += 1
        del wave_results

        _report(
            f"Auto-correct: wave {wave_idx + 1}/{len(waves)} done "
            f"({len(wave)} cell(s), {n_done}/{n_total} total) -- "
            f"{n_corrected} corrected, {len(skipped_cells)} skipped so far"
        )

    # ── Step 6: final whole-layer debris cleanup (skin included) ────────
    n_debris_removed = 0
    if min_volume is not None:
        threshold = int(round(final_min_fraction * min_volume))
        _report(f"Auto-correct: removing debris below {threshold} vox...")
        new_labels, n_debris_removed = remove_debris(new_labels, threshold, skin_label_id=skin_id)

    # ── Report-only screen: possible non-microglia blobs (no label is
    #    changed by this) ────────────────────────────────────────────────
    _report("Auto-correct: checking for possible non-microglia blobs (report only)...")
    possible_non_microglia = _flag_possible_non_microglia(
        new_labels, brain_mask, [int(i) for i in np.unique(new_labels) if i > 0],
        skin_id, skin_touch_px, non_microglia_fraction,
    )
    if possible_non_microglia:
        _report(f"Auto-correct: {len(possible_non_microglia)} possible non-microglia blob(s) flagged (report only).")

    report = {
        "best_lo": best_lo,
        "best_lo_swept": best_lo_swept,  # None when lo_override skipped the sweep
        "lo_override_used": lo_override is not None,
        "lo_adjustment": float(lo_adjustment),
        "sweep_mean_iou": sweep["best_mean_iou"] if sweep is not None else None,
        "n_calibration_samples": sweep["n_samples"] if sweep is not None else 0,
        "skin_label_id": skin_id,
        "skin_report": skin_report,
        "n_skin_debris_fragments_removed": n_skin_debris_fragments_removed,
        "touching_skin_cell_ids": sorted(touching_skin_ids),
        "skin_touch_px": skin_touch_px,
        "n_cells_total": n_total,
        "n_cells_corrected": n_corrected,
        "skipped_cells": skipped_cells,
        "cell_reports": cell_reports,
        "n_debris_fragments_removed": n_debris_removed,
        "possible_non_microglia": possible_non_microglia,
        "non_microglia_fraction": non_microglia_fraction,
    }
    return new_labels.astype(np.int32), report


def _format_2d_vs_skin_report(report: dict, skin_id: int) -> str:
    """
    Companion to format_grow_report()'s own "3D, per-slice" style, for
    a cell corrected by correct_label_2d_stack() instead (touching
    skin -- see auto_contrast_correct_stack()'s own module docstring,
    step 5). Same per-slice-report shape, deliberately not run through
    format_grow_report() itself: that function's "3D" text always says
    "(3D, per-slice)", which would misdescribe what actually ran here.
    """
    lines = [
        f"Auto-grow (2D per-slice, jointly resolved against skin label "
        f"{skin_id}): group={report['group']}"
    ]
    slices_grown = report.get("slices_grown", {})
    if slices_grown:
        grown_txt = ", ".join(f"{z}: {p}px" for z, p in sorted(slices_grown.items()))
        lines.append(f"  Slice(s) that needed a bigger pad: {grown_txt}")
    else:
        lines.append("  No slice needed more than the base pad.")
    slices_stability = report.get("slices_stability_passes", {})
    if slices_stability:
        stab_txt = ", ".join(f"{z}: {p} pass(es)" for z, p in sorted(slices_stability.items()))
        lines.append(f"  Slice(s) that took more than one pass to settle: {stab_txt}")
    if not report.get("stable", True):
        lines.append(
            "  STILL CHANGING -- at least one slice hit the stability-pass "
            "cap without settling; a larger cap may let it finish converging."
        )
    if report.get("converged", True):
        lines.append("  Converged -- no part of the result touches the padded region's own edge.")
    else:
        still_touching = report.get("slices_not_converged", [])
        lines.append(
            "  NOT converged -- signal still reaches the padded region's edge on "
            f"slice(s) {still_touching} even after auto-grow was exhausted there. "
            "Real signal may extend further; consider a larger pad, or correct "
            "this cell by hand."
        )
    lines.append(f"  Debris removed: {report.get('n_debris_removed_px', 0)} px")
    return "\n".join(lines)


def format_auto_correction_report(report: dict) -> str:
    lines = []
    if report.get("lo_override_used"):
        lines.append(
            f"Auto-correction: lo={report['best_lo']:.4g} "
            f"(given directly -- contrast sweep skipped)"
        )
    else:
        adj = report.get("lo_adjustment", 0.0)
        swept = report.get("best_lo_swept")
        adj_note = (
            f", adjusted -{adj:.4g} from the swept {swept:.4g}" if adj and swept is not None else ""
        )
        lines.append(
            f"Auto-correction: lo={report['best_lo']:.4g}{adj_note} "
            f"(calibration mean IoU={report['sweep_mean_iou']:.3f}, "
            f"{report['n_calibration_samples']} samples)"
        )
    skin_report = report["skin_report"]
    n_skin_frag = report.get("n_skin_debris_fragments_removed", 0)
    if skin_report.get("already_protected"):
        lines.append(
            f"  Skin was ALREADY protected as label {report['skin_label_id']} "
            f"({skin_report.get('n_skin_voxels_reused', 0):,} voxels) -- reused exactly "
            f"as it was, not re-protected or re-trimmed."
        )
    else:
        lines.append(
            f"  Skin protected as label {report['skin_label_id']} "
            f"(lo={report['best_lo']:.4g}, same threshold real cells use) -- "
            f"{n_skin_frag} debris fragment(s) of stray skin removed."
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
    touching_ids = report.get("touching_skin_cell_ids", [])
    lines.append("")
    lines.append(
        f"Cell-by-cell correction (Centroid-Z order): "
        f"{report['n_cells_corrected']}/{report['n_cells_total']} corrected"
        + (f" ({len(touching_ids)} touching skin, corrected in 2D against it; "
           f"the rest in 3D)" if touching_ids else " (all in 3D, none touch skin)")
        + (f", {len(report['skipped_cells'])} skipped" if report["skipped_cells"] else "")
    )
    for lid, reason in report["skipped_cells"].items():
        lines.append(f"  label {lid} skipped: {reason}")
    lines.append("")

    # Report-only: blobs that are probably not microglia. Nothing below
    # was changed in the labels on account of this list.
    flagged = report.get("possible_non_microglia", {})
    pct = int(round(100 * report.get("non_microglia_fraction", 0.40)))
    if flagged:
        lines.append(
            f"Possible NON-microglia blob(s) -- {len(flagged)} cell(s) with >= {pct}% "
            f"outside the brain mask and/or of their surface against skin "
            f"(macrophage or leftover skin?). REPORT ONLY -- nothing was removed or changed:"
        )
        for lid, f in sorted(flagged.items()):
            z, y, x = f["centroid_zyx"]
            why = " + ".join(
                {"outside_brain": f"{100 * f['outside_brain_frac']:.0f}% of volume outside brain",
                 "skin_contact": f"{100 * f['skin_surface_frac']:.0f}% of surface within "
                                 f"{report.get('skin_touch_px', 3)}px of skin"}[r]
                for r in f["reasons"]
            )
            lines.append(
                f"  label {lid}: {why} ({f['volume_vox']:,} vox, centroid z={z:.0f} y={y:.0f} x={x:.0f})"
            )
    else:
        lines.append(f"No possible non-microglia blobs (>= {pct}% outside brain / against skin).")
    lines.append("")

    # Per-cell detail: a cell touching skin was corrected 2D-per-slice
    # against it (_format_2d_vs_skin_report -- same per-slice report
    # shape trim_skin_label() itself uses); every other cell reuses the
    # exact same formatter the interactive Correct Label (3D) button
    # uses, so it reads identically to one corrected by hand.
    n_converged = 0
    n_not_stable = 0
    total_debris = 0
    total_slices_grown = 0
    total_slices_stability = 0
    for lid in sorted(report["cell_reports"]):
        cell_report = report["cell_reports"][lid]
        lines.append(f"--- Label {lid} ---")
        if cell_report.get("_mode") == "2d_vs_skin":
            lines.append(_format_2d_vs_skin_report(cell_report, report["skin_label_id"]))
        else:
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
