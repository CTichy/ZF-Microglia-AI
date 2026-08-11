"""
_cellpose_seg.py — Cellpose-SAM do_3D segmentation + Krendl corrections.

Same pipeline as microglia_segmentation/krendl_do3d.py (do_3D inference ->
3-component GMM -> Krendl safe merge -> large-contact merge), refactored into
plain, importable functions so the plugin can run it directly instead of
shelling out to the CLI script. No GT-based relabeling/scoring here — that
stays a CLI/research workflow; this produces a clean, sequentially-labeled
instance mask ready for manual correction or export.
"""

import numpy as np
from scipy.ndimage import find_objects, distance_transform_edt, binary_dilation

GT_MIN = 10230  # smallest real microglia volume (vox) seen in validated GT data

_touch_struct = np.zeros((3, 3, 3), dtype=bool)
_touch_struct[1, 1, :] = True
_touch_struct[1, :, 1] = True
_touch_struct[:, 1, 1] = True


def _get_info(m):
    sl = find_objects(m)
    info = {}
    for pid in np.unique(m[m > 0]):
        psl = sl[pid - 1]
        if psl is None:
            continue
        coords = np.where(m[psl] == pid)
        vol = len(coords[0])
        info[int(pid)] = {
            "vol": vol,
            "centroid": np.array([coords[0].mean() + psl[0].start,
                                   coords[1].mean() + psl[1].start,
                                   coords[2].mean() + psl[2].start]),
            "bbox": tuple((s.start, s.stop) for s in psl),
        }
    return info


def _bboxes_close(b1, b2, margin):
    return all(not (hi1 + margin < lo2 or hi2 + margin < lo1)
               for (lo1, hi1), (lo2, hi2) in zip(b1, b2))


def _joint_bbox(b1, b2):
    return tuple((min(lo1, lo2), max(hi1, hi2))
                 for (lo1, hi1), (lo2, hi2) in zip(b1, b2))


def run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=True,
                        min_hole_size=0, min_size=15):
    """Raw Cellpose-SAM do_3D inference. Returns an int32 label array.

    Convenience wrapper around predict_flows()+masks_from_flows() for a
    single (cellprob, flow) point -- see those two for the split used when
    sweeping multiple cellprob values against the same volume.

    min_size : deliberately tiny early noise filter, not this project's
    Min volume floor -- see the "Common Settings" note in _widget.py for
    why the two are kept separate. Default 15 matches Cellpose's own
    default and this project's prior, unexposed hardcoded value."""
    model, dP, cellprob_map, shape = predict_flows(volume, model_path, anisotropy, gpu=gpu)
    return masks_from_flows(model, dP, cellprob_map, shape, cellprob, flow,
                             min_size=min_size, min_hole_size=min_hole_size)


def predict_flows(volume, model_path, anisotropy, gpu=True):
    """The one genuinely expensive, GPU-bound step of do_3D inference: the
    network forward pass producing a per-voxel flow field (dP) and cell
    probability map (cellprob). Depends on neither cellprob_threshold nor
    flow_threshold at all -- both are applied afterward in masks_from_flows(),
    a cheap CPU/light-GPU step (confirmed by reading cellpose/models.py:
    CellposeModel.eval() calls self._run_net() -- the expensive part -- then
    separately calls self._compute_masks(shape, dP, cellprob,
    flow_threshold=..., cellprob_threshold=..., ...) -- the cheap part).

    Splitting these mirrors _inference.py's predict_probability() /
    postprocess_probability() split for MONAI: run the expensive network
    pass once, then re-threshold as many times as needed for a sweep
    without paying for a second forward pass.

    Returns (model, dP, cellprob, shape) -- shape is the original volume
    shape, needed by masks_from_flows() to resize correctly if dP/cellprob
    ever come back at a different resolution (only happens with
    diameter-based rescaling; unused here, this project always passes
    diameter=None).
    """
    from cellpose import models as cp_models
    model = cp_models.CellposeModel(pretrained_model=str(model_path), gpu=gpu)
    _, flows, _ = model.eval(
        volume, do_3D=True, anisotropy=anisotropy, z_axis=0, channel_axis=None,
        diameter=None, normalize=True, augment=False, compute_masks=False,
    )
    dP, cellprob = flows[1], flows[2]
    return model, dP, cellprob, volume.shape


def _make_capped_fill_holes(min_hole_size):
    """Builds a drop-in replacement for cellpose.utils's own
    fill_holes_and_remove_small_masks(masks, min_size=15), monkey-patched
    in for the duration of a single _compute_masks() call (see
    masks_from_flows() below).

    Cellpose's own version fills every enclosed void in each predicted
    mask's full 3D crop completely unconditionally, via
    `fill_voids.fill(msk)` -- no size threshold anywhere in that call.
    This is the exact same category of bug create_labels() had before
    min_hole_size was added there (see _labeling.py): a single-voxel
    prediction artifact and a genuine internal structural void are
    treated identically, since there is no way to tell them apart. The
    installed package can't be edited directly (breaks on every
    reinstall), so this follows the same monkey-patching convention
    train_xzyz.py already uses for the branch-weighted loss: swap in a
    size-aware replacement, restore the original afterward.

    min_hole_size<=0 keeps the exact original behaviour (unconditional
    fill_voids.fill()); a positive value switches to skimage's
    area-limited remove_small_holes, which works natively in 3D so no
    per-slice loop is needed here the way _labeling.py's 2D case needed
    one. The min_size small-mask-removal logic is otherwise reproduced
    unchanged from cellpose/utils.py."""
    import fastremap
    from scipy.ndimage import find_objects as _find_objects
    from skimage.morphology import remove_small_holes

    def _capped(masks, min_size=15):
        if masks.ndim > 3 or masks.ndim < 2:
            raise ValueError(f"masks_to_outlines takes 2D or 3D array, not {masks.ndim}D array")
        if min_size > 0:
            uniq, counts = fastremap.unique(masks, return_counts=True)
            small = uniq[1:][np.nonzero(counts[1:] < min_size)[0]]
            masks = fastremap.mask(masks, small)
            fastremap.renumber(masks, in_place=True)

        slices = _find_objects(masks)
        j = 0
        for i, slc in enumerate(slices):
            if slc is not None:
                msk = masks[slc] == (i + 1)
                if min_hole_size <= 0:
                    import fill_voids
                    msk = fill_voids.fill(msk)
                else:
                    msk = remove_small_holes(msk, area_threshold=min_hole_size)
                masks[slc][msk] = (j + 1)
                j += 1

        if min_size > 0:
            uniq, counts = fastremap.unique(masks, return_counts=True)
            small = uniq[1:][np.nonzero(counts[1:] < min_size)[0]]
            masks = fastremap.mask(masks, small)
            fastremap.renumber(masks, in_place=True)
        return masks

    return _capped


def masks_from_flows(model, dP, cellprob, shape, cellprob_threshold, flow_threshold=0.4,
                      min_size=15, max_size_fraction=0.4, niter=None, min_hole_size=0):
    """Cheap step: form instance masks from an already-computed flow field
    (see predict_flows()). do_3D=True is baked in -- this project's
    pipeline never uses 2D/stitch mode.

    flow_threshold is accepted only to match do_3D's own call signature and
    is a documented NO-OP here: reading cellpose/dynamics.py's
    compute_masks() shows its flow-error QC filter (remove_bad_flow_masks)
    is called only inside `if not do_3D:` -- under do_3D=True it never
    runs, confirmed both by that unconditional code-path check and by a
    call-count spy test (0 calls under do_3D=True regardless of value).
    It's kept as a parameter (rather than silently dropped) so callers
    that still pass a Flow value don't get a confusing signature error --
    it just has no effect on the result, by Cellpose's own design.

    niter=None is CellposeModel.eval()'s own public default, but eval()
    only resolves it to a real integer (200, unless diameter-based
    rescaling is active) inside its own top-level body before calling
    _compute_masks() internally -- calling _compute_masks() directly, as
    this function does, skips that resolution entirely and passes None
    straight down to dynamics.follow_flows()'s `range(niter)`, crashing
    with "TypeError: 'NoneType' object cannot be interpreted as an
    integer". This project always calls predict_flows() with
    diameter=None and no rescale, the exact case eval() itself resolves
    to niter=200, so that same value is applied here explicitly.

    min_hole_size : passed through to _make_capped_fill_holes() -- see
    that function's docstring. 0 (default) matches cellpose's own
    unconditional hole-filling exactly.
    """
    if niter is None:
        niter = 200
    from cellpose import utils as _cp_utils
    _original_fill_holes = _cp_utils.fill_holes_and_remove_small_masks
    _cp_utils.fill_holes_and_remove_small_masks = _make_capped_fill_holes(min_hole_size)
    try:
        masks = model._compute_masks(
            shape, dP, cellprob, flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold, min_size=min_size,
            max_size_fraction=max_size_fraction, niter=niter, do_3D=True,
        )
    finally:
        _cp_utils.fill_holes_and_remove_small_masks = _original_fill_holes
    return np.asarray(masks, dtype=np.int32)


def gmm_cleanup(masks):
    """3-component GMM on the raw object-size distribution — separates
    noise / gray-zone / real-cell populations and drops everything below
    the auto-detected gray->cell cutoff. Returns (masks, cutoff_vox, n_removed)."""
    from skimage.measure import regionprops
    from sklearn.mixture import GaussianMixture

    masks = masks.copy()
    props_raw = regionprops(masks)
    n0 = len(props_raw)
    if n0 < 3:
        return masks, 0.0, 0  # not enough objects to fit 3 components

    vols_raw = np.array([p.area for p in props_raw], dtype=np.float64)
    x_raw    = np.log1p(vols_raw).reshape(-1, 1)

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0)
    gmm.fit(x_raw)

    means     = gmm.means_.flatten()
    variances = gmm.covariances_.reshape(-1)
    weights   = gmm.weights_.flatten()
    oi = np.argsort(means)
    _, mid_i, large_i = oi

    def gaussian_intersection(i, j):
        mu_a, mu_b   = means[i], means[j]
        var_a, var_b = max(variances[i], 1e-12), max(variances[j], 1e-12)
        sig_a, sig_b = np.sqrt(var_a), np.sqrt(var_b)
        w_a,  w_b    = max(weights[i], 1e-12), max(weights[j], 1e-12)
        a  = (1.0 / (2 * var_a)) - (1.0 / (2 * var_b))
        b  = (mu_b / var_b) - (mu_a / var_a)
        c0 = (mu_a**2 / (2 * var_a) - mu_b**2 / (2 * var_b)
              + np.log((w_b * sig_a) / (w_a * sig_b)))
        if abs(a) < 1e-12:
            t = -c0 / max(b, 1e-12)
        else:
            disc = b * b - 4 * a * c0
            if disc < 0:
                t = 0.5 * (mu_a + mu_b)
            else:
                r1 = (-b + np.sqrt(disc)) / (2 * a)
                r2 = (-b - np.sqrt(disc)) / (2 * a)
                lo, hi = min(mu_a, mu_b), max(mu_a, mu_b)
                cands = [r for r in (r1, r2) if lo <= r <= hi]
                t = cands[0] if cands else (
                    r1 if abs(r1 - 0.5 * (mu_a + mu_b)) < abs(r2 - 0.5 * (mu_a + mu_b)) else r2
                )
        return float(np.expm1(t))

    cutoff = gaussian_intersection(mid_i, large_i)

    removed = 0
    for p in props_raw:
        if p.area < cutoff:
            masks[masks == p.label] = 0
            removed += 1

    return masks, cutoff, removed


def krendl_safe_merge(masks, max_gap=2, min_contact=10, gt_min=GT_MIN):
    """Merge only sub-gt_min fragments into their nearest larger neighbour,
    when either close enough (<=max_gap) or touching with enough contact
    area (>=min_contact). Returns (masks, n_merges)."""
    masks = masks.copy()
    total_merges = 0

    for _ in range(200):
        info = _get_info(masks)
        candidates = sorted(
            [p for p, d in info.items() if d["vol"] < gt_min],
            key=lambda p: info[p]["vol"]
        )
        merged_any = False
        for fid in candidates:
            if fid not in info:
                continue
            fvol = info[fid]["vol"]; fcent = info[fid]["centroid"]; fbbox = info[fid]["bbox"]
            best_tid = None; best_dist = 1e9
            for tid, tdata in info.items():
                if tid == fid or tdata["vol"] <= fvol:
                    continue
                if not _bboxes_close(fbbox, tdata["bbox"], margin=max_gap + 2):
                    continue
                d = float(np.linalg.norm(tdata["centroid"] - fcent))
                if d < best_dist:
                    best_dist = d; best_tid = tid
            if best_tid is None:
                continue
            jbbox = _joint_bbox(fbbox, info[best_tid]["bbox"])
            slZ = slice(jbbox[0][0], jbbox[0][1])
            slY = slice(jbbox[1][0], jbbox[1][1])
            slX = slice(jbbox[2][0], jbbox[2][1])
            region = masks[slZ, slY, slX]
            fmask = (region == fid); tmask = (region == best_tid)
            if not fmask.any() or not tmask.any():
                continue
            distmap = distance_transform_edt(~fmask)
            do_merge = float(distmap[tmask].min()) <= max_gap
            if not do_merge:
                dilated = binary_dilation(fmask, structure=_touch_struct)
                do_merge = int((dilated & tmask).sum()) >= min_contact
            if not do_merge:
                continue
            region[fmask] = best_tid; masks[slZ, slY, slX] = region
            del info[fid]
            nc = np.where(masks == best_tid)
            if len(nc[0]) > 0:
                info[best_tid]["vol"] = len(nc[0])
                info[best_tid]["centroid"] = np.array([nc[0].mean(), nc[1].mean(), nc[2].mean()])
                info[best_tid]["bbox"] = (
                    (int(nc[0].min()), int(nc[0].max()) + 1),
                    (int(nc[1].min()), int(nc[1].max()) + 1),
                    (int(nc[2].min()), int(nc[2].max()) + 1),
                )
            merged_any = True; total_merges += 1
        if not merged_any:
            break

    return masks, total_merges


def large_contact_merge(masks, large_contact=20):
    """Merge any two objects (regardless of size) that share a contact area
    of >= large_contact voxels — catches blobs split through a thick
    junction rather than a thin neck. Returns (masks, n_merges)."""
    masks = masks.copy()
    lc_merges = 0

    for _ in range(50):
        info = _get_info(masks)
        sorted_ids = sorted(info.keys(), key=lambda p: info[p]["vol"])
        merged_any = False

        for fid in sorted_ids:
            if fid not in info:
                continue
            fbbox = info[fid]["bbox"]

            for tid in list(info.keys()):
                if tid == fid or tid not in info:
                    continue
                if not _bboxes_close(fbbox, info[tid]["bbox"], margin=2):
                    continue

                jbbox = _joint_bbox(fbbox, info[tid]["bbox"])
                slZ = slice(jbbox[0][0], jbbox[0][1])
                slY = slice(jbbox[1][0], jbbox[1][1])
                slX = slice(jbbox[2][0], jbbox[2][1])
                region = masks[slZ, slY, slX]
                fmask = (region == fid)
                tmask = (region == tid)
                if not fmask.any() or not tmask.any():
                    continue

                dilated = binary_dilation(fmask, structure=_touch_struct)
                contact = int((dilated & tmask).sum())

                if contact >= large_contact:
                    keep, drop = (tid, fid) if info[tid]["vol"] >= info[fid]["vol"] else (fid, tid)
                    region[region == drop] = keep
                    masks[slZ, slY, slX] = region
                    del info[drop]
                    nc = np.where(masks == keep)
                    if len(nc[0]) > 0:
                        info[keep]["vol"] = len(nc[0])
                        info[keep]["centroid"] = np.array([nc[0].mean(), nc[1].mean(), nc[2].mean()])
                        info[keep]["bbox"] = (
                            (int(nc[0].min()), int(nc[0].max()) + 1),
                            (int(nc[1].min()), int(nc[1].max()) + 1),
                            (int(nc[2].min()), int(nc[2].max()) + 1),
                        )
                    merged_any = True; lc_merges += 1
                    break

        if not merged_any:
            break

    return masks, lc_merges


def final_min_size_cleanup(masks, gt_min, fraction=0.618):
    """Last-resort safety net, run after every other correction stage
    (GMM cleanup, Krendl safe-merge, large-contact merge): removes any
    surviving object smaller than fraction * gt_min.

    gt_min is the smallest true voxel volume ever confirmed in real GT
    (the same unified floor Safe-merge's own gt_min parameter uses --
    see _widget.py's "Common Settings" note and _krendl_sweep.py's
    gt_min_from_labels alias). Nothing upstream is guaranteed to remove
    every possible debris object: GMM cleanup separates populations by
    the raw size distribution, which can still leave a gray-zone object
    standing, and safe-merge/large-contact only act when a nearby
    neighbor exists to merge into. This stage is the final backstop --
    not a replacement for those, but a floor under all of them.

    fraction defaults to the golden ratio, 1/phi ~= 0.618: a fragment
    genuinely that much smaller than the smallest real GT cell ever
    measured is a defensible cutoff for "almost certainly not a real
    cell" without being as aggressive as gt_min itself, which would
    reject legitimately smaller-than-average real cells too.

    Returns (masks, n_removed)."""
    threshold = max(1, round(gt_min * fraction))
    info = _get_info(masks)
    below = [pid for pid, v in info.items() if v["vol"] < threshold]
    if not below:
        return masks, 0
    masks = masks.copy()
    for pid in below:
        masks[masks == pid] = 0
    return masks, len(below)


def relabel_sequential(masks):
    """Renumber labels 1..N with no gaps. Returns (masks, n_labels)."""
    ids = np.unique(masks[masks > 0])
    if ids.size == 0:
        return masks, 0
    lut = np.zeros(int(ids.max()) + 1, dtype=np.int32)
    for new_id, old_id in enumerate(ids, start=1):
        lut[old_id] = new_id
    return lut[masks], int(ids.size)


def run_full_pipeline(volume, model_path, cellprob=-2.5, flow=0.4, anisotropy=5.747,
                       max_gap=2, min_contact=10, large_contact=20, gt_min=GT_MIN,
                       gpu=True, progress_cb=None, precomputed_flows=None,
                       min_hole_size=0, min_size=15, final_min_fraction=0.618):
    """
    Full do_3D + 3-GMM + Krendl safe merge + large-contact merge + final
    min-size safety net pipeline — identical math to krendl_do3d.py plus
    one additional final stage, minus the GT-based relabeling/scoring
    (that stays a CLI/research workflow). Returns (labels, stats).

    progress_cb, if given, is called with a short status string before each
    stage — safe to call from a worker thread (just writes a string).

    precomputed_flows: optional (model, dP, cellprob_map, shape) tuple from
    a prior predict_flows() call on this same volume/model -- skips the
    expensive network pass entirely and goes straight to mask formation.
    Used by the Cellprob/Large-contact sweep to call this once per Cellprob
    value without re-running do_3D's network forward pass each time.

    min_hole_size : passed through to masks_from_flows() -- see
    _make_capped_fill_holes()'s docstring. 0 (default) matches Cellpose's
    own unconditional hole-filling exactly.

    min_size : deliberately tiny early noise filter, not this project's
    Min volume floor -- kept as a separate parameter on purpose (see the
    "Common Settings" note in _widget.py). Default 15 matches Cellpose's
    own default.

    final_min_fraction : passed through to final_min_size_cleanup(), run
    as the very last stage after large-contact merge -- see that
    function's docstring for why 0.618 (golden ratio) is the default.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    if precomputed_flows is not None:
        model, dP, cellprob_map, shape = precomputed_flows
        _report(f"cellprob={cellprob}: forming masks from precomputed flows...")
        masks = masks_from_flows(model, dP, cellprob_map, shape, cellprob, flow,
                                  min_size=min_size, min_hole_size=min_hole_size)
    else:
        _report("Running do_3D inference...")
        masks = run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=gpu,
                                    min_hole_size=min_hole_size, min_size=min_size)
    n0 = len(np.unique(masks[masks > 0]))

    _report(f"{n0} raw cells — 3-component GMM cleanup...")
    masks, gmm_cutoff, gmm_removed = gmm_cleanup(masks)
    n1 = len(np.unique(masks[masks > 0]))

    _report(f"{n1} cells — Krendl safe merge...")
    masks, safe_merges = krendl_safe_merge(masks, max_gap, min_contact, gt_min)
    n2 = len(np.unique(masks[masks > 0]))

    _report(f"{n2} cells — large-contact merge...")
    masks, lc_merges = large_contact_merge(masks, large_contact)
    n3 = len(np.unique(masks[masks > 0]))

    final_min_threshold = max(1, round(gt_min * final_min_fraction))
    _report(f"{n3} cells — final min-size safety net (< {final_min_threshold} vox)...")
    masks, final_removed = final_min_size_cleanup(masks, gt_min, final_min_fraction)
    n4 = len(np.unique(masks[masks > 0]))

    _report(f"{n4} cells — relabeling...")
    masks, n_final = relabel_sequential(masks)

    stats = {
        "n_raw":               n0,
        "n_after_gmm":         n1,
        "n_after_safe_merge":  n2,
        "n_after_large_contact": n3,
        "n_after_final_min_size": n4,
        "n_final":             n_final,
        "gmm_cutoff_vox":      gmm_cutoff,
        "gmm_removed":         gmm_removed,
        "safe_merges":         safe_merges,
        "large_contact_merges": lc_merges,
        "final_min_threshold_vox": final_min_threshold,
        "final_min_removed":   final_removed,
    }
    return masks, stats
