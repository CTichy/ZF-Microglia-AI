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
        masks = masks_from_flows(model, dP, cellprob_map, shape, cp, flow=0.4,
                                  min_size=min_size, min_hole_size=min_hole_size, niter=niter)
        r = _voxel_dice_iou(masks > 0, gt_mask)
        results[cp] = r
        if progress_cb:
            progress_cb(f"cellprob={cp}: Dice={r['dice']:.1f}%  IoU={r['iou']:.1f}%  "
                        f"precision={r['precision']:.1f}%  recall={r['recall']:.1f}%")

    best_cellprob = max(results, key=lambda k: results[k]["dice"]) if results else None
    return dict(results=results, best_cellprob=best_cellprob, cancelled=cancelled,
                precomputed=(model, dP, cellprob_map, shape))


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


def run_krendl_sweep(volume, gt_labels, model_path, cellprobs, large_contacts,
                      flow=0.4, anisotropy=5.747, max_gap=1.0, min_contact=10,
                      gt_min=None, iou_threshold=0.5, gpu=True, min_hole_size=None,
                      min_size=15, final_min_fraction=0.618,
                      progress_cb=None, cancel_event=None, precomputed=None,
                      scale_zyx=(1.0, 0.174, 0.174)):
    """
    Sweep every (cellprob, large_contact) combination, scoring the
    resulting Krendl-pipeline labels against gt_labels with
    _gt_score.score_against_gt.

    max_gap, scale_zyx: passed through to krendl_safe_merge() -- max_gap
    is in PHYSICAL MICRONS (Z, Y, X um/voxel = scale_zyx), not voxels.
    See that function's docstring for why (anisotropic voxels).

    precomputed: pass a (model, dP, cellprob_map, shape) tuple -- as
    returned in this call's own result dict under 'precomputed' -- to
    skip predict_flows() entirely and reuse an already-computed do_3D
    network pass. cellprob_threshold only feeds the cheap
    masks_from_flows() step (see this module's docstring), so a
    multi-stage narrowing sweep (coarse pass, then a finer pass zoomed
    around the winner, then finer still) run against the SAME volume +
    model can share one ~3h inference pass across every stage instead
    of repeating it per stage. If None (default), predict_flows() runs
    as before.

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
      'precomputed': tuple,   # (model, dP, cellprob_map, shape) -- pass
                             # back in as precomputed= on a later call
                             # against the same volume+model to skip
                             # predict_flows() entirely.
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

    if precomputed is not None:
        if progress_cb:
            progress_cb("Reusing precomputed flows from an earlier call -- no re-inference.")
    else:
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
        masks, _ = krendl_safe_merge(masks, max_gap, min_contact, gt_min, scale_zyx=scale_zyx)

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
    # Score (TP - 0.5*(FP+FN)) is coarse -- built from integer counts, so
    # many grid points can tie on it even though they differ meaningfully
    # in how tightly the matched cells are actually segmented. Breaking
    # ties by mean_iou then mean_dice picks the most precise segmentation
    # among equally-good-on-Score candidates, instead of just the first
    # one encountered in grid order.
    #
    # A point with tp==0 detects no real cells at all -- score_against_gt()
    # then reports mean_iou=mean_dice=0.0 (nothing to average), which can
    # still tie or even "win" against a genuinely-detecting point whose FPs
    # made its own Score just as bad or worse. That's never a usable
    # result, so exclude tp==0 points from the winner search entirely
    # unless literally every grid point detected nothing (in which case
    # there's no better answer to give).
    non_degenerate = {k: v for k, v in results.items() if v["tp"] > 0}
    candidates = non_degenerate if non_degenerate else results
    best_point = (
        max(candidates, key=lambda k: (results[k]["score"], results[k]["mean_iou"], results[k]["mean_dice"]))
        if candidates else None
    )

    return dict(grid=grid, results=results, best_point=best_point,
                gt_min_used=gt_min, min_hole_size_used=min_hole_size, cancelled=cancelled,
                precomputed=precomputed)


def _grid_table(sweep, cellprobs, large_contacts, value_key, fmt, current_large_contact):
    """Build one metric's 2D grid table (rows = large_contact, columns =
    cellprob). Shared by Score/MeanIoU/MeanDice below so the three tables
    stay in lockstep -- same column layout, same missing-point handling."""
    header = f"{'LrgCnt':>8} | " + " | ".join(f"cp={c:>5} " for c in cellprobs)
    lines = [header, "-" * len(header)]
    for lc in large_contacts:
        row = []
        for cp in cellprobs:
            point = (cp, lc)
            if point in sweep["results"]:
                row.append(fmt(sweep["results"][point][value_key]))
            else:
                row.append(f"{'--':>8}")
        marker = "  <- current" if lc == current_large_contact else ""
        lines.append(f"{lc:>8} | " + " | ".join(row) + marker)
    lines.append("-" * len(header))
    return lines


def format_krendl_sweep_report(sweep, current_cellprob=None, current_large_contact=None):
    """Plain-text 2D grid report (rows = large_contact, columns = cellprob),
    same spirit as the plugin's other sweep-tool reports.

    Prints three grid tables -- Score, MeanIoU, MeanDice -- not just Score,
    so the rise-peak-fall shape of cellprob's effect (too lenient -> noisy
    FPs; sweet spot; too strict -> real signal gets excluded, IoU/Dice
    collapse toward the TP=0 degenerate case) is directly visible in the
    report instead of only inferable from where "Best" happens to land.
    """
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

    lines = list(lines0)
    lines.append("Score = TP - 0.5*(FP+FN):")
    lines += _grid_table(
        sweep, cellprobs, large_contacts, "score",
        lambda v: f"{v:>+8.1f}", current_large_contact,
    )
    lines.append("")
    lines.append("Mean IoU % (matched cells only -- 0.0 at a fully degenerate/TP=0 point):")
    lines += _grid_table(
        sweep, cellprobs, large_contacts, "mean_iou",
        lambda v: f"{v:>8.1f}", current_large_contact,
    )
    lines.append("")
    lines.append("Mean Dice % (matched cells only -- 0.0 at a fully degenerate/TP=0 point):")
    lines += _grid_table(
        sweep, cellprobs, large_contacts, "mean_dice",
        lambda v: f"{v:>8.1f}", current_large_contact,
    )

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
