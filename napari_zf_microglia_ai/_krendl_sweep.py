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

import numpy as np

from ._cellpose_seg import predict_flows, masks_from_flows, gmm_cleanup, krendl_safe_merge
from ._pixel_sweep import min_volume_from_gt as gt_min_from_labels
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


def _voxel_dice_iou(pred_mask, gt_mask):
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


def run_cellprob_voxel_sweep(volume, gt_labels, model_path, cellprobs,
                              anisotropy=5.747, gpu=True, min_size=15,
                              min_hole_size=0, niter=None,
                              progress_cb=None, cancel_event=None, precomputed=None):
    """
    Score cellprob PURELY on raw voxel-level signal quality against GT
    (Dice/IoU/precision/recall on the binarized foreground, ignoring
    instance identity entirely) -- deliberately has NO dependency on
    max_gap/min_contact/large_contact at all.

    Why this exists alongside run_krendl_sweep(): that tool scores
    cellprob using score_against_gt() on the FULLY CORRECTED result
    (after GMM + Krendl safe-merge + large-contact-merge), which means
    its "optimal cellprob" is entangled with whatever those merge
    parameters happen to be set to at sweep time -- circular if those
    parameters haven't themselves been calibrated yet (see
    measure_merge_params_from_prediction()/recommend_merge_params(),
    which need a real cp_masks at SOME cellprob to calibrate from).
    This sweep breaks that circularity: whether do_3D fragments one
    real cell into 5 pieces is irrelevant to "did cellprob correctly
    separate true cell signal from background" -- fragmentation is
    exactly what the merge stage exists to fix afterward, so it
    shouldn't feed back into picking cellprob in the first place.
    Mirrors _brain_sweep.run_brain_sweep's identical reasoning for
    MONAI Threshold (mask-level Dice, not instance-matched), applied
    here to cellprob instead.

    Correct order to calibrate a fish end-to-end: run THIS sweep first
    to pick cellprob on signal quality alone, generate cp_masks at that
    cellprob, calibrate max_gap/min_contact/large_contact from it, THEN
    optionally cross-check with run_krendl_sweep()'s instance-matched
    Score using the newly-calibrated merge parameters.

    Reuses predict_flows()/masks_from_flows()'s split (see this
    module's own docstring) -- one do_3D pass regardless of how many
    cellprob values are tested; precomputed= lets a caller reuse an
    already-computed pass, same contract as run_krendl_sweep().

    Returns dict: {
      'results': {cellprob: {dice, iou, precision, recall, pred_vox, gt_vox}},
      'best_cellprob': float or None,   # highest Dice
      'cancelled': bool,
      'precomputed': tuple,   # (model, dP, cellprob_map, shape)
    }
    """
    gt_mask = gt_labels > 0

    if precomputed is not None:
        model, dP, cellprob_map, shape = precomputed
        if progress_cb:
            progress_cb("Reusing precomputed flows from an earlier call -- no re-inference.")
    else:
        if progress_cb:
            progress_cb("Predicting flows (do_3D network pass -- the one expensive step, runs once)...")
        model, dP, cellprob_map, shape = predict_flows(volume, model_path, anisotropy, gpu=gpu)

    results = {}
    cancelled = False
    for cp in cellprobs:
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        masks = masks_from_flows(model, dP, cellprob_map, shape, cp, flow_threshold=0.4,
                                  min_size=min_size, min_hole_size=min_hole_size, niter=niter)
        r = _voxel_dice_iou(masks > 0, gt_mask)
        results[cp] = r
        if progress_cb:
            progress_cb(f"cellprob={cp}: Dice={r['dice']:.1f}%  IoU={r['iou']:.1f}%  "
                        f"precision={r['precision']:.1f}%  recall={r['recall']:.1f}%")

    best_cellprob = max(results, key=lambda k: results[k]["dice"]) if results else None
    return dict(results=results, best_cellprob=best_cellprob, cancelled=cancelled,
                precomputed=(model, dP, cellprob_map, shape))


def format_cellprob_voxel_sweep_report(sweep, current_cellprob=None):
    """Plain-text 1D report (one row per cellprob value) for
    run_cellprob_voxel_sweep() -- single-axis, unlike
    format_krendl_sweep_report()'s 2D grid, since this sweep has no
    second parameter to grid against."""
    results = sweep["results"]
    if not results:
        return "No grid points completed."

    cellprobs = sorted(results.keys())
    header = f"{'cellprob':>10} | {'Dice%':>8} | {'IoU%':>8} | {'Prec%':>8} | {'Recall%':>8}"
    lines = [header, "-" * len(header)]
    for cp in cellprobs:
        r = results[cp]
        marker = "  <- current" if current_cellprob is not None and cp == current_cellprob else ""
        lines.append(
            f"{cp:>10} | {r['dice']:>8.1f} | {r['iou']:>8.1f} | "
            f"{r['precision']:>8.1f} | {r['recall']:>8.1f}{marker}"
        )
    lines.append("-" * len(header))
    lines.append("(voxel-level Dice/IoU/precision/recall against binarized GT -- instance identity ignored)")

    best = sweep.get("best_cellprob")
    if best is not None:
        r = results[best]
        lines.append("")
        lines.append(
            f"Best: cellprob={best} (Dice={r['dice']:.1f}%, IoU={r['iou']:.1f}%, "
            f"precision={r['precision']:.1f}%, recall={r['recall']:.1f}%)"
        )
        if current_cellprob is not None:
            if current_cellprob in results and current_cellprob != best:
                cr = results[current_cellprob]
                lines.append(
                    f"Current setting (cellprob={current_cellprob}): Dice={cr['dice']:.1f}% "
                    f"-- the sweep found a better value above."
                )
            elif current_cellprob == best:
                lines.append("Current setting matches the sweep's best -- confirmed.")

    if sweep.get("cancelled"):
        lines.append("\n(sweep was cancelled -- results above are partial.)")
    return "\n".join(lines)


def measure_merge_params_from_prediction(cp_masks, gt_labels, scale_zyx=(1.0, 0.174, 0.174),
                                          min_overlap_vox=5, search_pad_um=3.0):
    """
    Directly measure max_gap/min_contact/large_contact from a REAL raw
    (pre-GMM, pre-Krendl) Cellpose-SAM prediction compared against GT --
    a strictly better source than pure-GT geometry (see
    min_intercell_gap_um()'s own docstring for the ambiguity that
    approach has: is the smallest real gap between two GT cells actual
    biology, or just how an annotator happened to draw the boundary?).
    This sidesteps that question entirely by using GT only to LABEL
    which raw fragments are/aren't the same real cell, then measuring
    the actual gap/contact the network's own fragmentation produced.

    Every raw cp_masks fragment is assigned to whichever GT cell it
    overlaps most (by voxel count); a fragment whose largest overlap is
    below min_overlap_vox is dropped as noise, not a real fragment of
    anything.

    'should_merge' samples: pairs of fragments assigned to the SAME GT
    cell (the raw prediction over-fragmented one real cell) -- the gap/
    contact between them is exactly what max_gap/min_contact need to
    bridge to correctly reunite them.

    'should_not_merge' samples: pairs of fragments assigned to
    DIFFERENT GT cells, found via a cheap bbox-proximity pre-filter
    (search_pad_um) rather than checking every fragment against every
    other -- the gap/contact between them is a real, unambiguous safety
    ceiling (GT already confirms these are genuinely different cells).

    Returns dict: {
      'should_merge_gaps_um': [float, ...],
      'should_merge_contacts_vox': [int, ...],
      'should_not_merge_gaps_um': [float, ...],
      'should_not_merge_contacts_vox': [int, ...],
      'n_gt_cells_fragmented': int,   # GT cells with >=2 assigned fragments
      'n_fragments_assigned': int,
      'n_fragments_dropped_as_noise': int,
    }
    """
    from scipy.ndimage import find_objects, distance_transform_edt, binary_dilation
    from ._cellpose_seg import _touch_struct

    cp_masks = np.asarray(cp_masks)
    gt_labels = np.asarray(gt_labels)
    Z, Y, X = cp_masks.shape

    frag_ids = np.unique(cp_masks)
    frag_ids = frag_ids[frag_ids > 0]
    frag_objs = find_objects(cp_masks, max_label=int(frag_ids.max()) if len(frag_ids) else 0)

    # Assign each fragment to its dominant GT cell.
    assigned = {}   # frag_id -> gt_label
    frag_bbox = {}  # frag_id -> (z0,z1,y0,y1,x0,x1) in full-volume coords
    n_dropped = 0
    for fid in frag_ids:
        sl = frag_objs[fid - 1]
        if sl is None:
            continue
        crop_gt = gt_labels[sl]
        crop_frag = cp_masks[sl] == fid
        overlap_vals = crop_gt[crop_frag]
        overlap_vals = overlap_vals[overlap_vals > 0]
        if overlap_vals.size == 0:
            n_dropped += 1
            continue
        counts = np.bincount(overlap_vals)
        best_gt = int(np.argmax(counts))
        if counts[best_gt] < min_overlap_vox:
            n_dropped += 1
            continue
        assigned[fid] = best_gt
        frag_bbox[fid] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop)

    by_gt = {}
    for fid, gt_lbl in assigned.items():
        by_gt.setdefault(gt_lbl, []).append(fid)

    def _gap_and_contact(fid_a, fid_b):
        ba = frag_bbox[fid_a]; bb = frag_bbox[fid_b]
        z0 = min(ba[0], bb[0]); z1 = max(ba[1], bb[1])
        y0 = min(ba[2], bb[2]); y1 = max(ba[3], bb[3])
        x0 = min(ba[4], bb[4]); x1 = max(ba[5], bb[5])
        region = cp_masks[z0:z1, y0:y1, x0:x1]
        mask_a = region == fid_a
        mask_b = region == fid_b
        if not mask_a.any() or not mask_b.any():
            return None, None
        distmap = distance_transform_edt(~mask_a, sampling=scale_zyx)
        gap = float(distmap[mask_b].min())
        dilated = binary_dilation(mask_a, structure=_touch_struct)
        contact = int((dilated & mask_b).sum())
        return gap, contact

    should_merge_gaps, should_merge_contacts = [], []
    n_fragmented = 0
    for gt_lbl, fids in by_gt.items():
        if len(fids) < 2:
            continue
        n_fragmented += 1
        # krendl_safe_merge() merges each small fragment to its NEAREST
        # larger same-cell neighbor, iteratively -- it never needs to
        # bridge two fragments directly if a shorter path exists via a
        # third fragment in between. All-pairs gaps overstate what
        # max_gap actually needs (e.g. two small pieces on opposite ends
        # of one large, sprawling real cell would never need to merge
        # directly). So for every fragment, only its single nearest
        # same-cell sibling gap/contact is recorded -- one sample per
        # fragment, not one per pair.
        for i in range(len(fids)):
            best_gap = None; best_contact = None
            for j in range(len(fids)):
                if i == j:
                    continue
                gap, contact = _gap_and_contact(fids[i], fids[j])
                if gap is not None and (best_gap is None or gap < best_gap):
                    best_gap = gap; best_contact = contact
            if best_gap is not None:
                should_merge_gaps.append(best_gap)
                should_merge_contacts.append(best_contact)

    # bbox-proximity pre-filter for cross-GT-cell fragment pairs, in voxel
    # space (generous -- converts search_pad_um using the finest axis so
    # no genuinely-close pair is missed).
    pad_vox = search_pad_um / min(scale_zyx)
    all_fids = list(assigned.keys())
    should_not_merge_gaps, should_not_merge_contacts = [], []
    for i in range(len(all_fids)):
        fid_a = all_fids[i]
        ba = frag_bbox[fid_a]
        for j in range(i + 1, len(all_fids)):
            fid_b = all_fids[j]
            if assigned[fid_a] == assigned[fid_b]:
                continue  # same GT cell -- already counted above
            bb = frag_bbox[fid_b]
            close = not (
                ba[1] + pad_vox < bb[0] or bb[1] + pad_vox < ba[0] or
                ba[3] + pad_vox < bb[2] or bb[3] + pad_vox < ba[2] or
                ba[5] + pad_vox < bb[4] or bb[5] + pad_vox < ba[4]
            )
            if not close:
                continue
            gap, contact = _gap_and_contact(fid_a, fid_b)
            if gap is not None:
                should_not_merge_gaps.append(gap)
                should_not_merge_contacts.append(contact)

    return dict(
        should_merge_gaps_um=should_merge_gaps,
        should_merge_contacts_vox=should_merge_contacts,
        should_not_merge_gaps_um=should_not_merge_gaps,
        should_not_merge_contacts_vox=should_not_merge_contacts,
        n_gt_cells_fragmented=n_fragmented,
        n_fragments_assigned=len(assigned),
        n_fragments_dropped_as_noise=n_dropped,
    )


def _count_gap_gt_and_contact_lt(gaps, contacts, gap_candidates, contact_candidates):
    """
    For every (g, c) in the cross product of gap_candidates x
    contact_candidates, count how many (gap_i, contact_i) sample pairs
    satisfy gap_i > g AND contact_i < c -- i.e. samples that neither
    threshold would catch on its own. Returns a (len(gap_candidates),
    len(contact_candidates)) int array.

    Done as one matrix multiply instead of a G x C x N triple loop:
    A[i, g] = 1 if gap_i > gap_candidates[g] else 0        -- (N, G)
    B[i, c] = 1 if contact_i < contact_candidates[c] else 0 -- (N, C)
    count[g, c] = sum_i A[i, g] * B[i, c] = (A.T @ B)[g, c]
    Exact (candidates are the real observed breakpoints, not a grid
    approximation), and fast even for N/G/C in the hundreds.
    """
    gaps = np.asarray(gaps, dtype=float)
    contacts = np.asarray(contacts, dtype=float)
    if gaps.size == 0:
        return np.zeros((len(gap_candidates), len(contact_candidates)), dtype=int)
    A = (gaps[:, None] > np.asarray(gap_candidates)[None, :]).astype(np.int32)
    B = (contacts[:, None] < np.asarray(contact_candidates)[None, :]).astype(np.int32)
    return A.T @ B


def recommend_merge_params(merge_stats_list):
    """
    Turn one or more measure_merge_params_from_prediction() results
    (e.g. one per fish, pooled) into a single recommended (max_gap,
    min_contact) PAIR -- the last step this project's other GT
    calibrations (min_volume_from_gt(), recommend_branch_radius())
    already take automatically, not yet built for these two.

    Jointly optimizes both parameters together against the REAL
    krendl_safe_merge() decision rule -- a merge triggers if
    gap <= max_gap OR contact >= min_contact (see that function's own
    docstring) -- rather than picking each threshold in isolation as
    if only one criterion existed. Gap and contact are measured as
    PAIRED samples per fragment pair (the same pair's gap and contact
    both come from measure_merge_params_from_prediction()'s single
    pass over it), so the pairing is preserved here: a sample only
    counts as "missed" if BOTH its gap exceeds max_gap AND its contact
    falls short of min_contact, matching the OR-combined rule exactly,
    not two independent AND-combined single-criterion misses.

    Every (gap, contact) combination actually observed in the data is
    tried as a candidate threshold pair (exact, not a coarse grid) via
    a vectorized matrix-multiply rather than a triple-nested loop --
    see _count_gap_gt_and_contact_lt().

    Optimizes for FEWEST TOTAL MANUAL CORRECTIONS, not zero risk of one
    error type. An earlier version of this function treated bridging
    two genuinely distinct real cells as something to avoid almost no
    matter the cost, clamping the threshold defensively low even when
    that meant catching close to none of the real should-merge cases.
    That's the wrong objective for this pipeline specifically: an
    over-merge is trivially fixable with the existing watershed-based
    Split Label tool, exactly as an under-merge (a cell left in pieces)
    is fixable with Join Labels. Neither failure is silent or
    catastrophic -- both are expected, easy things a human reviews and
    fixes by hand, since this tool (like every automated step here) is
    meant to help, never to be trusted blindly. So the right threshold
    pair is whichever leaves the fewest total mistakes for that review
    to catch, not whichever is most paranoid about a single direction
    of error.

    Returns dict: {
      'max_gap_um': float or None, 'contact_vox': int or None,
      'missed_merges': int, 'false_merges': int,
      'n_should_merge': int, 'n_should_not_merge': int,
    }
    Missed/false counts are against the pooled sample data itself --
    read them as "how many of the real cases in this data landed on
    the wrong side", not a guaranteed rate on unseen fish.
    """
    sm_gaps, sm_contacts, snm_gaps, snm_contacts = [], [], [], []
    for ms in merge_stats_list:
        sm_gaps.extend(ms["should_merge_gaps_um"])
        sm_contacts.extend(ms["should_merge_contacts_vox"])
        snm_gaps.extend(ms["should_not_merge_gaps_um"])
        snm_contacts.extend(ms["should_not_merge_contacts_vox"])

    if not sm_gaps and not snm_gaps:
        return dict(max_gap_um=None, contact_vox=None, missed_merges=0, false_merges=0,
                    n_should_merge=0, n_should_not_merge=0)

    all_gaps = sm_gaps + snm_gaps
    all_contacts = sm_contacts + snm_contacts
    # Sentinels so a purely-gap-only or purely-contact-only solution is
    # representable too: a gap candidate below every real gap disables
    # the gap criterion entirely (gap_i > g always true, i.e. never
    # triggers on its own); a contact candidate above every real
    # contact disables the contact criterion the same way.
    gap_candidates = sorted(set(all_gaps) | {min(all_gaps) - 1.0})
    contact_candidates = sorted(set(all_contacts) | {max(all_contacts) + 1})

    missed_matrix = _count_gap_gt_and_contact_lt(sm_gaps, sm_contacts, gap_candidates, contact_candidates)
    avoided_matrix = _count_gap_gt_and_contact_lt(snm_gaps, snm_contacts, gap_candidates, contact_candidates)
    false_matrix = len(snm_gaps) - avoided_matrix
    total_matrix = missed_matrix + false_matrix

    best_flat = int(np.argmin(total_matrix))
    gi, ci = np.unravel_index(best_flat, total_matrix.shape)
    best_gap = float(gap_candidates[gi])
    best_contact = int(contact_candidates[ci])

    return dict(
        max_gap_um=best_gap, contact_vox=best_contact,
        missed_merges=int(missed_matrix[gi, ci]), false_merges=int(false_matrix[gi, ci]),
        n_should_merge=len(sm_gaps), n_should_not_merge=len(snm_gaps),
    )


def measure_min_size_from_prediction(cp_masks_unfiltered, gt_labels, min_overlap_vox=5):
    """
    Directly measure a safe `min_size` early-noise-filter threshold from
    a REAL raw, UNFILTERED (min_size=0, pre-GMM, pre-Krendl) Cellpose-SAM
    prediction compared against GT -- same spirit as
    measure_merge_params_from_prediction(), but for fragment VOLUME
    instead of gap/contact.

    cp_masks_unfiltered MUST come from masks_from_flows(..., min_size=0)
    (or run_do3d_inference(..., min_size=0)) -- any positive min_size
    already discards exactly the small fragments this function needs to
    see, before it ever gets a chance to measure them. Confirmed
    directly by reading _make_capped_fill_holes()'s own _capped(): it
    strips anything under threshold at TWO points before returning, so
    a masks array formed with the pipeline's normal min_size=15 has
    already lost this information irrecoverably.

    Every fragment is assigned to whichever GT cell it overlaps most (by
    voxel count) -- identical rule to measure_merge_params_from_
    prediction(); a fragment whose best overlap is under
    min_overlap_vox is dropped as noise. NOT discarded here the way
    that other function discards it -- this IS the population min_size
    exists to catch, so its size is recorded instead.

    Unlike measure_merge_params_from_prediction(), every GT-assigned
    fragment counts, not only ones belonging to a GT cell with >=2
    fragments: the question here is "does this fragment, however small,
    deserve to survive to GMM/Krendl," not "does it need merging," so a
    GT cell matched by exactly one already-correctly-sized fragment
    still contributes that fragment's own size to real_sizes_vox.

    Returns dict: {
      'real_sizes_vox': [int, ...],   # GT-assigned fragments, any size
      'noise_sizes_vox': [int, ...],  # no real GT correspondence at all
      'n_fragments_assigned': int,
      'n_fragments_dropped_as_noise': int,
    }
    """
    from scipy.ndimage import find_objects

    cp_masks_unfiltered = np.asarray(cp_masks_unfiltered)
    gt_labels = np.asarray(gt_labels)

    frag_ids = np.unique(cp_masks_unfiltered)
    frag_ids = frag_ids[frag_ids > 0]
    if frag_ids.size == 0:
        return dict(real_sizes_vox=[], noise_sizes_vox=[],
                    n_fragments_assigned=0, n_fragments_dropped_as_noise=0)
    frag_objs = find_objects(cp_masks_unfiltered, max_label=int(frag_ids.max()))

    real_sizes, noise_sizes = [], []
    n_assigned = 0
    n_dropped = 0
    for fid in frag_ids:
        sl = frag_objs[fid - 1]
        if sl is None:
            continue
        crop_frag = cp_masks_unfiltered[sl] == fid
        size = int(crop_frag.sum())
        overlap_vals = gt_labels[sl][crop_frag]
        overlap_vals = overlap_vals[overlap_vals > 0]
        if overlap_vals.size == 0:
            noise_sizes.append(size)
            n_dropped += 1
            continue
        best_overlap = int(np.bincount(overlap_vals).max())
        if best_overlap < min_overlap_vox:
            noise_sizes.append(size)
            n_dropped += 1
            continue
        real_sizes.append(size)
        n_assigned += 1

    return dict(
        real_sizes_vox=real_sizes,
        noise_sizes_vox=noise_sizes,
        n_fragments_assigned=n_assigned,
        n_fragments_dropped_as_noise=n_dropped,
    )


def recommend_min_size(stats_list, miss_weight=1000):
    """
    Turn one or more measure_min_size_from_prediction() results (e.g.
    one per fish, pooled) into a single recommended min_size -- same
    "pool across fish, then search once over the real observed values"
    pattern as recommend_merge_params().

    Deliberately NOT the same objective as recommend_merge_params(),
    even though the mechanics are similar. There, an over-merge (fixable
    with Split Label) and an under-merge (fixable with Join Labels) are
    both easy, expected things a human corrects by hand -- genuinely
    symmetric failure costs, which is why that function optimizes for
    fewest TOTAL corrections. Here the two directions are NOT symmetric:
    a real fragment discarded by min_size is simply gone -- there is no
    "restore a deleted raw fragment" tool anywhere in this plugin. A
    noise fragment that survives too long, by contrast, still gets three
    more chances to be caught (GMM cleanup, Krendl's own gt_min floor,
    the final golden-ratio safety net) before it could ever become a
    final cell. So missing a real fragment is treated as far more costly
    than letting a noise fragment survive a little longer.

    miss_weight: how many "noise fragments surviving" one "real fragment
    discarded" is worth in the cost function minimized below. Default
    1000 -- in practice this means "never discard a real fragment if ANY
    threshold avoids it," falling back to a genuine trade-off only if
    the two size distributions actually overlap (some real fragment is
    smaller than some noise fragment, so no single threshold gets both
    right).

    Every size actually observed in the pooled data is tried as a
    candidate threshold (exact, not a coarse grid); ties on cost prefer
    the LARGEST candidate (cleans up more noise without costing any more
    missed real fragments).

    Returns dict: {
      'min_size_vox': int or None, 'missed': int, 'false_survivors': int,
      'n_real': int, 'n_noise': int,
    }
    Missed/false counts are against the pooled sample data itself, same
    caveat as recommend_merge_params(): read them as "how many of the
    real cases in this data landed on the wrong side," not a guaranteed
    rate on an unseen fish.
    """
    real_sizes, noise_sizes = [], []
    for st in stats_list:
        real_sizes.extend(st["real_sizes_vox"])
        noise_sizes.extend(st["noise_sizes_vox"])

    if not real_sizes and not noise_sizes:
        return dict(min_size_vox=None, missed=0, false_survivors=0, n_real=0, n_noise=0)

    real_arr = np.asarray(real_sizes, dtype=float)
    noise_arr = np.asarray(noise_sizes, dtype=float)

    # Candidate thresholds: every observed size, plus a sentinel of 1 so
    # "keep absolutely everything" (today's min_size=0 behaviour) stays
    # representable even if every observed fragment happens to be large.
    candidates = sorted(set(int(s) for s in (real_sizes + noise_sizes)) | {1})

    best_t = best_cost = best_missed = best_false = None
    for t in candidates:
        missed = int((real_arr < t).sum()) if real_arr.size else 0
        false_survivors = int((noise_arr >= t).sum()) if noise_arr.size else 0
        cost = miss_weight * missed + false_survivors
        # candidates is ascending, so on an exact tie this keeps
        # replacing with the larger t -- ends up preferring the biggest
        # threshold among every candidate achieving the minimum cost.
        if best_cost is None or cost <= best_cost:
            best_cost, best_t, best_missed, best_false = cost, t, missed, false_survivors

    return dict(
        min_size_vox=int(best_t), missed=best_missed, false_survivors=best_false,
        n_real=len(real_sizes), n_noise=len(noise_sizes),
    )


def measure_large_contact_from_prediction(post_safe_merge_masks, gt_labels,
                                           min_overlap_vox=5, bbox_margin_vox=2):
    """
    Directly measure a safe `large_contact` threshold (large_contact_
    merge()'s own, single-criterion "merge if contact >= large_contact"
    rule) from a REAL post-GMM, post-Krendl-safe-merge prediction
    compared against GT -- same spirit as measure_merge_params_from_
    prediction(), but at the LATER pipeline stage large_contact_merge()
    actually operates on.

    post_safe_merge_masks MUST come from AFTER gmm_cleanup() and
    krendl_safe_merge() have both already run, but BEFORE
    large_contact_merge() itself -- i.e. exactly what large_contact_
    merge() would receive as its own input in the real pipeline. Safe-
    merge's own gt_min-based size gate has already reunited every small
    fragment it could by then; large_contact_merge() is the next, size-
    agnostic catch-all for whatever's still split through a thick
    junction rather than a thin neck -- passing raw cp_masks here
    instead would mix in everything safe_merge already fixes, which
    isn't what this stage needs to decide.

    Every object is assigned to whichever GT cell it overlaps most (same
    rule as measure_merge_params_from_prediction); an object whose best
    overlap is under min_overlap_vox is dropped as noise, not used.

    'should_merge' samples: for every GT cell still matched by >=2
    objects at this stage (safe_merge couldn't fully reunite it), each
    object's contact area with its STRONGEST (largest-contact) same-GT-
    cell sibling -- large_contact must be <= this value to actually
    trigger that merge. "Strongest contact" here plays the same role
    "nearest by gap" plays in measure_merge_params_from_prediction: the
    one pair large_contact_merge()'s own greedy loop would actually act
    on first, since recording every pair would overstate what a single
    threshold needs to bridge.

    'should_not_merge' samples: contact area between any two bbox-close
    (bbox_margin_vox -- matches large_contact_merge()'s OWN proximity
    pre-filter exactly, not Krendl's wider search_pad_um) objects
    assigned to DIFFERENT GT cells -- GT already confirms these are
    genuinely separate cells, so this is a real, unambiguous safety
    ceiling.

    Returns dict: {
      'should_merge_contacts_vox': [int, ...],
      'should_not_merge_contacts_vox': [int, ...],
      'n_gt_cells_still_fragmented': int,
      'n_objects_assigned': int,
      'n_objects_dropped_as_noise': int,
    }
    """
    from scipy.ndimage import find_objects, binary_dilation
    from ._cellpose_seg import _touch_struct, _bboxes_close, _joint_bbox

    post_safe_merge_masks = np.asarray(post_safe_merge_masks)
    gt_labels = np.asarray(gt_labels)

    obj_ids = np.unique(post_safe_merge_masks)
    obj_ids = obj_ids[obj_ids > 0]
    if obj_ids.size == 0:
        return dict(should_merge_contacts_vox=[], should_not_merge_contacts_vox=[],
                    n_gt_cells_still_fragmented=0, n_objects_assigned=0,
                    n_objects_dropped_as_noise=0)
    obj_objs = find_objects(post_safe_merge_masks, max_label=int(obj_ids.max()))

    assigned = {}
    obj_bbox = {}
    n_dropped = 0
    for oid in obj_ids:
        sl = obj_objs[oid - 1]
        if sl is None:
            continue
        crop_obj = post_safe_merge_masks[sl] == oid
        overlap_vals = gt_labels[sl][crop_obj]
        overlap_vals = overlap_vals[overlap_vals > 0]
        if overlap_vals.size == 0:
            n_dropped += 1
            continue
        counts = np.bincount(overlap_vals)
        best_gt = int(np.argmax(counts))
        if counts[best_gt] < min_overlap_vox:
            n_dropped += 1
            continue
        assigned[oid] = best_gt
        obj_bbox[oid] = tuple((s.start, s.stop) for s in sl)

    by_gt = {}
    for oid, gt_lbl in assigned.items():
        by_gt.setdefault(gt_lbl, []).append(oid)

    def _contact(oid_a, oid_b):
        jbbox = _joint_bbox(obj_bbox[oid_a], obj_bbox[oid_b])
        slZ = slice(jbbox[0][0], jbbox[0][1])
        slY = slice(jbbox[1][0], jbbox[1][1])
        slX = slice(jbbox[2][0], jbbox[2][1])
        region = post_safe_merge_masks[slZ, slY, slX]
        mask_a = region == oid_a
        mask_b = region == oid_b
        if not mask_a.any() or not mask_b.any():
            return None
        dilated = binary_dilation(mask_a, structure=_touch_struct)
        return int((dilated & mask_b).sum())

    should_merge = []
    n_fragmented = 0
    for gt_lbl, oids in by_gt.items():
        if len(oids) < 2:
            continue
        n_fragmented += 1
        for i in range(len(oids)):
            best_contact = None
            for j in range(len(oids)):
                if i == j:
                    continue
                c = _contact(oids[i], oids[j])
                if c is not None and (best_contact is None or c > best_contact):
                    best_contact = c
            if best_contact is not None:
                should_merge.append(best_contact)

    all_oids = list(assigned.keys())
    should_not_merge = []
    for i in range(len(all_oids)):
        oid_a = all_oids[i]
        ba = obj_bbox[oid_a]
        for j in range(i + 1, len(all_oids)):
            oid_b = all_oids[j]
            if assigned[oid_a] == assigned[oid_b]:
                continue
            if not _bboxes_close(ba, obj_bbox[oid_b], margin=bbox_margin_vox):
                continue
            c = _contact(oid_a, oid_b)
            if c is not None:
                should_not_merge.append(c)

    return dict(
        should_merge_contacts_vox=should_merge,
        should_not_merge_contacts_vox=should_not_merge,
        n_gt_cells_still_fragmented=n_fragmented,
        n_objects_assigned=len(assigned),
        n_objects_dropped_as_noise=n_dropped,
    )


def recommend_large_contact(stats_list, miss_weight=1.0):
    """
    Turn one or more measure_large_contact_from_prediction() results
    into a single recommended large_contact threshold.

    Unlike recommend_min_size(), this stage's two failure directions ARE
    genuinely symmetric -- same reasoning as recommend_merge_params():
    a false merge here (bridging two real, separate GT cells) is
    fixable with Split Label; a missed merge (leaving one real GT cell
    split across >=2 objects) is fixable with Join Labels. Neither is
    silent or catastrophic the way a min_size-discarded fragment is
    (nothing here is ever permanently lost -- both directions are
    reviewable, expected things). So this defaults to miss_weight=1.0 --
    fewest total corrections, deliberately NOT the heavily asymmetric
    weighting min_size's genuinely irreversible failure mode needs.

    large_contact_merge()'s own rule, "merge if contact >= large_contact",
    means a BIGGER threshold here causes FEWER merges (missed merges
    grow, false merges shrink as the threshold rises) -- the mirror
    image of min_size's "keep if size >= min_size" direction, where a
    bigger threshold means MORE gets discarded. Close enough in shape to
    reuse the same candidate-threshold-search pattern, different enough
    in which direction each failure count moves that this gets its own
    loop rather than literally calling recommend_min_size().

    Every contact value actually observed in the pooled data is tried
    as a candidate threshold (exact, not a coarse grid); ties on cost
    prefer the LARGEST candidate -- merges less aggressively without
    costing any more missed real merges, the safer default when nothing
    in the data distinguishes two candidates.

    Returns dict: {
      'large_contact_vox': int or None, 'missed': int, 'false_merges': int,
      'n_should_merge': int, 'n_should_not_merge': int,
    }
    Missed/false counts are against the pooled sample data itself, same
    caveat as recommend_merge_params()/recommend_min_size().
    """
    should_merge, should_not_merge = [], []
    for st in stats_list:
        should_merge.extend(st["should_merge_contacts_vox"])
        should_not_merge.extend(st["should_not_merge_contacts_vox"])

    if not should_merge and not should_not_merge:
        return dict(large_contact_vox=None, missed=0, false_merges=0,
                    n_should_merge=0, n_should_not_merge=0)

    sm_arr = np.asarray(should_merge, dtype=float)
    snm_arr = np.asarray(should_not_merge, dtype=float)

    candidates = sorted(set(int(c) for c in (should_merge + should_not_merge)) | {1})

    best_t = best_cost = best_missed = best_false = None
    for t in candidates:
        missed = int((sm_arr < t).sum()) if sm_arr.size else 0
        false_merges = int((snm_arr >= t).sum()) if snm_arr.size else 0
        cost = miss_weight * missed + false_merges
        if best_cost is None or cost <= best_cost:
            best_cost, best_t, best_missed, best_false = cost, t, missed, false_merges

    return dict(
        large_contact_vox=int(best_t), missed=best_missed, false_merges=best_false,
        n_should_merge=len(should_merge), n_should_not_merge=len(should_not_merge),
    )
