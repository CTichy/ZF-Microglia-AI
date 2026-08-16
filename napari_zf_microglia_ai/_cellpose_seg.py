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
                        min_hole_size=0, min_size=15, niter=None):
    """Raw Cellpose-SAM do_3D inference. Returns an int32 label array.

    Convenience wrapper around predict_flows()+masks_from_flows() for a
    single (cellprob, flow) point -- see those two for the split used when
    sweeping multiple cellprob values against the same volume.

    min_size : deliberately tiny early noise filter, not this project's
    Min volume floor -- see the "Common Settings" note in _widget.py for
    why the two are kept separate. Default 15 matches Cellpose's own
    default and this project's prior, unexposed hardcoded value.

    niter : passed through to masks_from_flows() -- see its docstring.
    None (default) resolves to Cellpose's own default of 200."""
    model, dP, cellprob_map, shape = predict_flows(volume, model_path, anisotropy, gpu=gpu)
    return masks_from_flows(model, dP, cellprob_map, shape, cellprob, flow,
                             min_size=min_size, min_hole_size=min_hole_size, niter=niter)


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
    area-limited remove_small_holes, applied one Z-slice at a time --
    same per-slice loop _labeling.py's 2D case needs, and for the same
    reason: min_hole_size is calibrated as a 2D per-slice area (see
    min_hole_size_from_gt()), and a single remove_small_holes() call
    over the whole 3D crop treats a real per-slice-small hole that
    persists across many Z-slices as one 3D-connected void whose total
    volume can dwarf that threshold -- "too big to be noise" by 3D
    volume, even though it's exactly the per-slice-small gap the
    threshold is meant to catch. The min_size small-mask-removal logic
    is otherwise reproduced unchanged from cellpose/utils.py."""
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
                    # Per-slice, not one remove_small_holes() call over the
                    # whole 3D crop: min_hole_size is calibrated as a 2D
                    # per-slice area (see min_hole_size_from_gt()), but a
                    # real per-slice-small hole that persists across many
                    # Z-slices is one 3D-connected void whose total volume
                    # can dwarf that threshold -- remove_small_holes() in
                    # 3D then judges it "too big to be noise" and leaves it
                    # unfilled on every slice, producing a visible ring
                    # inside the cell on each of those slices. _labeling.py
                    # (the Pixel Classifier route) already loops per-slice
                    # for exactly this reason; this path just never did.
                    filled = np.empty_like(msk)
                    for z in range(msk.shape[0]):
                        filled[z] = remove_small_holes(msk[z], area_threshold=min_hole_size)
                    msk = filled
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

    Raising niter above 200 gives each voxel's flow trajectory more
    Euler-integration steps to fully converge before follow_flows() bins
    final positions into instances -- relevant for structurally complex
    cells under this pipeline's heavy Z-anisotropy stretch, where some
    voxels can fail to converge within 200 steps and get scattered into
    inconsistent bins near their true instance, producing a porous/
    "pumice stone" 3D shape (parallel banding on any given 2D slice) for
    that one cell instead of a solid blob. See compute_porosity() below
    for a way to detect this after the fact rather than guessing.

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


def compute_porosity(masks, solidity_threshold=0.5):
    """Flag any label whose 3D shape is abnormally porous -- solidity =
    actual voxel volume / convex-hull volume, via skimage's regionprops
    (each label's own bounding box, so cheap even though convex-hull
    computation itself is not free).

    A solid blob sits close to 1.0. A "pumice stone"/skeletonized label
    -- voxels scattered inside a much larger convex envelope than they
    actually fill -- sits well below it. This is the shape Cellpose's
    do_3D flow-following produces when some voxels' trajectories fail
    to fully converge within niter steps on a structurally complex cell
    (see masks_from_flows()'s niter docstring) -- an algorithmic
    convergence artifact, not something this pipeline's own merge/
    cleanup stages can introduce or repair.

    Returns {label_id: solidity, ...} for every label below
    solidity_threshold -- a list of suspects to inspect/re-run (e.g. at
    a higher niter), not an automatic fix. There is no safe way to
    "repair" a label like this in place."""
    from skimage.measure import regionprops
    flagged = {}
    for p in regionprops(masks):
        try:
            sol = float(p.solidity)
        except Exception:
            continue
        if sol < solidity_threshold:
            flagged[int(p.label)] = sol
    return flagged


def run_full_pipeline(volume, model_path, cellprob=-2.5, flow=0.4, anisotropy=5.747,
                       max_gap=2, min_contact=10, large_contact=20, gt_min=GT_MIN,
                       gpu=True, progress_cb=None, precomputed_flows=None,
                       min_hole_size=0, min_size=15, final_min_fraction=0.618,
                       niter=None, solidity_threshold=0.5):
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

    niter : passed through to masks_from_flows()/run_do3d_inference() --
    see masks_from_flows()'s docstring. None (default) resolves to
    Cellpose's own default of 200.

    solidity_threshold : passed to compute_porosity(), run as the very
    last step against the final relabeled masks. 0.5 (default) is a
    conservative catch-most-real-artifacts starting point, not a
    calibrated GT-verified value (unlike this project's other
    thresholds) -- there is no GT for "is this shape porous" to
    calibrate against.
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    if precomputed_flows is not None:
        model, dP, cellprob_map, shape = precomputed_flows
        _report(f"cellprob={cellprob}: forming masks from precomputed flows...")
        masks = masks_from_flows(model, dP, cellprob_map, shape, cellprob, flow,
                                  min_size=min_size, min_hole_size=min_hole_size, niter=niter)
    else:
        _report("Running do_3D inference...")
        masks = run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=gpu,
                                    min_hole_size=min_hole_size, min_size=min_size, niter=niter)
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

    _report(f"{n_final} cells — checking shape solidity...")
    porous_cells = compute_porosity(masks, solidity_threshold=solidity_threshold)

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
        "porous_cells":        porous_cells,
        "solidity_threshold":  solidity_threshold,
    }
    return masks, stats


def rerun_single_cell(volume, labels, label_id, model_path, cellprob=-2.5, flow=0.4,
                       anisotropy=5.747, max_gap=2, min_contact=10, large_contact=20,
                       gt_min=GT_MIN, gpu=True, min_hole_size=0, min_size=15,
                       final_min_fraction=0.618, niter=None, solidity_threshold=0.5,
                       pad_z=15, pad_xy=40, progress_cb=None):
    """
    Re-run do_3D inference (+ the same GMM/Krendl-safe-merge/large-
    contact-merge/final-min-size cleanup as run_full_pipeline) on just
    the small padded bounding-box crop around one existing label,
    instead of the whole volume -- turns "fix one porous/mis-segmented
    cell" (see compute_porosity()) into seconds instead of the hours a
    full-fish re-run costs.

    Safe to reuse run_full_pipeline() unmodified on a tiny crop: GMM
    cleanup already no-ops below 3 objects (see its own docstring), and
    Krendl safe-merge / large-contact merge / final-min-size cleanup
    all use a fixed gt_min floor, not population statistics fit from
    whatever happens to be in the crop.

    Only objects from the crop's re-run that actually overlap
    label_id's own original footprint (within that same crop) are
    spliced back in, as brand-new label IDs appended after the current
    max -- anything else the crop's do_3D pass also happens to detect
    (e.g. a neighbouring cell caught by the padding) is discarded, not
    duplicated into the full volume. If the fix genuinely reveals more
    than one real cell where there was one label before, all of them
    are kept.

    volume : the SAME intensity volume label_id's original segmentation
    was produced from (not the label array itself) -- do_3D needs
    pixel intensities, not previous predictions.

    Returns (new_labels, info):
      new_labels : full-size label array, label_id's voxels replaced.
      info : {'old_label': label_id, 'new_labels': [id, ...],
              'n_new': int, 'porous_cells': {...}, 'crop_stats': {...}}
             -- porous_cells is compute_porosity() re-run restricted to
             just the newly spliced-in objects; crop_stats is
             run_full_pipeline()'s own stats dict from the crop.

    Raises ValueError if label_id isn't present in labels at all.
    """
    from scipy.ndimage import find_objects as _find_objects

    volume = np.asarray(volume)
    labels = np.asarray(labels)
    if volume.ndim != 3 or labels.ndim != 3:
        raise ValueError(
            f"volume and labels must both be 3D (Z, Y, X); got "
            f"volume.shape={volume.shape}, labels.shape={labels.shape}."
        )
    if volume.shape != labels.shape:
        raise ValueError(
            f"volume and labels must have matching shapes; got "
            f"volume.shape={volume.shape}, labels.shape={labels.shape}."
        )
    label_id = int(label_id)
    obj_slices = _find_objects(labels, max_label=label_id)
    if label_id < 1 or label_id > len(obj_slices) or obj_slices[label_id - 1] is None:
        raise ValueError(f"Label {label_id} not found in the current labels layer.")
    sl = obj_slices[label_id - 1]

    Z, Y, X = labels.shape
    z0 = max(0, sl[0].start - pad_z); z1 = min(Z, sl[0].stop + pad_z)
    y0 = max(0, sl[1].start - pad_xy); y1 = min(Y, sl[1].stop + pad_xy)
    x0 = max(0, sl[2].start - pad_xy); x1 = min(X, sl[2].stop + pad_xy)
    crop_sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    vol_crop = volume[crop_sl]
    orig_footprint = labels[crop_sl] == label_id

    if progress_cb:
        progress_cb(f"Re-running label {label_id} only: crop shape={vol_crop.shape} ...")

    crop_labels, crop_stats = run_full_pipeline(
        vol_crop, model_path, cellprob=cellprob, flow=flow, anisotropy=anisotropy,
        max_gap=max_gap, min_contact=min_contact, large_contact=large_contact,
        gt_min=gt_min, gpu=gpu, min_hole_size=min_hole_size, min_size=min_size,
        final_min_fraction=final_min_fraction, niter=niter,
        solidity_threshold=solidity_threshold, progress_cb=progress_cb,
    )

    keep_ids = sorted({int(v) for v in np.unique(crop_labels[orig_footprint]) if v > 0})

    new_labels = labels.copy()
    new_labels[labels == label_id] = 0

    region = new_labels[crop_sl]
    next_id = int(new_labels.max()) + 1 if new_labels.max() > 0 else 1
    spliced_ids = []
    for old_local_id in keep_ids:
        region[crop_labels == old_local_id] = next_id
        spliced_ids.append(next_id)
        next_id += 1
    new_labels[crop_sl] = region

    porous_recheck = compute_porosity(new_labels, solidity_threshold=solidity_threshold)
    porous_recheck = {k: v for k, v in porous_recheck.items() if k in spliced_ids}

    info = dict(
        old_label=label_id, new_labels=spliced_ids, n_new=len(spliced_ids),
        porous_cells=porous_recheck, crop_stats=crop_stats,
    )
    return new_labels, info
