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


def run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=True):
    """Raw Cellpose-SAM do_3D inference. Returns an int32 label array."""
    from cellpose import models as cp_models
    model = cp_models.CellposeModel(pretrained_model=str(model_path), gpu=gpu)
    masks, _, _ = model.eval(
        volume, do_3D=True, anisotropy=anisotropy, z_axis=0, channel_axis=None,
        cellprob_threshold=cellprob, flow_threshold=flow,
        diameter=None, normalize=True, augment=False,
    )
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
                       gpu=True, progress_cb=None):
    """
    Full do_3D + 3-GMM + Krendl safe merge + large-contact merge pipeline —
    identical math to krendl_do3d.py, minus the GT-based relabeling/scoring
    (that stays a CLI/research workflow). Returns (labels, stats).

    progress_cb, if given, is called with a short status string before each
    stage — safe to call from a worker thread (just writes a string).
    """
    def _report(msg):
        if progress_cb:
            progress_cb(msg)

    _report("Running do_3D inference...")
    masks = run_do3d_inference(volume, model_path, cellprob, flow, anisotropy, gpu=gpu)
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

    _report(f"{n3} cells — relabeling...")
    masks, n_final = relabel_sequential(masks)

    stats = {
        "n_raw":               n0,
        "n_after_gmm":         n1,
        "n_after_safe_merge":  n2,
        "n_after_large_contact": n3,
        "n_final":             n_final,
        "gmm_cutoff_vox":      gmm_cutoff,
        "gmm_removed":         gmm_removed,
        "safe_merges":         safe_merges,
        "large_contact_merges": lc_merges,
    }
    return masks, stats
