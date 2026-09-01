"""
_labeling.py — True 3D connected component labeling.

Backend priority
----------------
1. CUDA  — CuPy + cupyx.scipy.ndimage  (full GPU path)
2. MPS   — Apple Silicon Metal          (threaded CPU; MPS lacks ndimage ops)
3. CPU   — scipy.ndimage + ThreadPool   (multithreaded, portable fallback)

Workflow
--------
1. Binary mask  : volume > 0
2. Gaussian smooth (σ_xy, σ_z) → re-threshold at 0.5
3. Fill holes per Z slice
4. 3D connected components (26-connectivity via ones(3,3,3) structure)
5. Remove blobs < final_min_fraction * min_volume voxels (golden ratio
   safety-net relaxation by default, see create_labels()'s docstring)
6. Renumber 1…N by descending volume  (label 1 = largest)
"""

from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import (
    gaussian_filter as cpu_gaussian,
    label        as cpu_label,
    binary_fill_holes as cpu_fill_holes,
)
from skimage.morphology import remove_small_holes as _cpu_remove_small_holes


# ─────────────────────────────────────────────────────────────────────────────
# Backend detection  (run once at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backend() -> tuple[str, object, object]:
    """Return (backend_name, cupy_module, cupyx_ndimage_module)."""

    # ── CUDA via CuPy ──────────────────────────────────────────────────────
    import io, sys
    _saved, sys.stdout = sys.stdout, io.StringIO()
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpnd
        # Exercise NVRTC/JIT so a broken install fails here, not mid-run
        _t = cp.zeros((4, 4), dtype=cp.float32)
        cpnd.gaussian_filter(_t, sigma=1.0)
        return "cuda", cp, cpnd
    except Exception:
        pass
    finally:
        sys.stdout = _saved

    # ── Apple Silicon MPS ──────────────────────────────────────────────────
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps", None, None
    except Exception:
        pass

    return "cpu", None, None


_BACKEND, _CP, _CPND = _detect_backend()
_N_THREADS = max(1, (os.cpu_count() or 4) // 2)


def _free_gpu_cache() -> None:
    """Free CuPy and PyTorch GPU memory pools to prevent OOM."""
    if _CP is not None:
        try:
            _CP.get_default_memory_pool().free_all_blocks()
            _CP.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CUDA path
# ─────────────────────────────────────────────────────────────────────────────

def _fill_holes_capped_gpu(slice_gpu, min_hole_size, cp, cpnd):
    """Per-slice hole fill: a hole survives as background only if its area
    is >= min_hole_size voxels; anything smaller is filled as noise.
    <=0 means no floor -- fills every enclosed background region,
    matching the old unconditional binary_fill_holes behaviour exactly."""
    if min_hole_size <= 0:
        return cpnd.binary_fill_holes(slice_gpu)
    filled = cpnd.binary_fill_holes(slice_gpu)
    holes = filled & (~slice_gpu)
    if not bool(holes.any()):
        return slice_gpu
    hole_labels, n_holes = cpnd.label(holes)
    if n_holes == 0:
        return slice_gpu
    counts = cp.bincount(hole_labels.ravel().astype(cp.int64), minlength=n_holes + 1)
    fillable = counts < min_hole_size
    fillable[0] = False
    return slice_gpu | fillable[hole_labels]


def _create_labels_cuda(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_volume: int,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    cp   = _CP
    cpnd = _CPND
    Z, Y, X = volume.shape

    # ── Steps 1–2: binary → Gaussian smooth → re-threshold ────────────────
    vol_gpu     = cp.asarray(volume, dtype=cp.float32)
    binary_gpu  = (vol_gpu > 0).astype(cp.float32)
    blurred_gpu = cpnd.gaussian_filter(binary_gpu, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_gpu  = blurred_gpu > 0.5
    del vol_gpu, binary_gpu, blurred_gpu

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_gpu.sum()):,}")

    # ── Step 3: fill holes per slice (GPU loop), floored at min_hole_size ──
    for z in range(Z):
        smooth_gpu[z] = _fill_holes_capped_gpu(smooth_gpu[z], min_hole_size, cp, cpnd)

    # ── Step 4: true 3D connected components (26-connectivity) ────────────
    structure  = cp.ones((3, 3, 3), dtype=cp.int32)
    labeled_gpu, n_objects = cpnd.label(smooth_gpu, structure=structure)
    del smooth_gpu
    n_objects = int(n_objects)
    print(f"   3D blobs: {n_objects}")

    if n_objects == 0:
        result = labeled_gpu.get().astype(np.int32)
        del labeled_gpu
        _free_gpu_cache()
        return result

    # ── Step 5: remove small blobs — vectorised on GPU ────────────────────
    # Final cutoff is final_min_fraction * min_volume, not min_volume
    # itself -- same golden-ratio safety-net philosophy _cellpose_seg.py's
    # final_min_size_cleanup() uses, applied here since this route has no
    # merge/reattach stage of its own to leave a gray-zone object standing
    # for a later stage to reconsider (unlike Cellpose-SAM's GMM/safe-
    # merge/large-contact chain): min_volume alone as a hard cutoff would
    # discard a legitimately smaller-than-average real cell just as
    # readily as real debris. final_min_fraction=1.0 recovers the exact
    # historical behaviour (cutoff == min_volume) for any caller that
    # doesn't pass a fraction.
    threshold = max(1, round(final_min_fraction * min_volume))
    max_out = int(labeled_gpu.max())
    counts  = cp.bincount(labeled_gpu.ravel().astype(cp.int64), minlength=max_out + 1)

    keep_lut    = counts >= threshold
    keep_lut[0] = True
    output_gpu  = cp.where(keep_lut[labeled_gpu], labeled_gpu, cp.int32(0))
    removed     = int(((counts[1:] > 0) & (counts[1:] < threshold)).sum())
    del labeled_gpu

    # ── Step 6: renumber 1…N by descending volume ─────────────────────────
    remaining      = cp.unique(output_gpu[output_gpu > 0]).get().tolist()
    counts_cpu     = counts.get()
    volumes_sorted = sorted(
        [(int(counts_cpu[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output_gpu.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = cp.asarray(lut2)[output_gpu].get()
    n_final = int(output.max())
    print(f"   3D blobs removed (< {threshold} vox = {final_min_fraction:.3f} x min_volume {min_volume}): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    del output_gpu, counts, keep_lut
    _free_gpu_cache()
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded CPU path  (also used for Apple MPS — MPS lacks ndimage ops)
# ─────────────────────────────────────────────────────────────────────────────

def _create_labels_threaded(
    volume: np.ndarray,
    sigma_xy: float,
    sigma_z: float,
    min_volume: int,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    Z, Y, X = volume.shape

    # ── Steps 1–2: Gaussian smooth (scipy already multi-threaded internally)
    binary      = (volume > 0).astype(np.float32)
    blurred     = cpu_gaussian(binary, sigma=(sigma_z, sigma_xy, sigma_xy))
    smooth_mask = blurred > 0.5
    del binary, blurred

    print(f"   σ_xy={sigma_xy:.1f}  σ_z={sigma_z:.1f}  "
          f"signal voxels: {int(smooth_mask.sum()):,}")

    # ── Step 3: fill holes per slice in parallel, floored at min_hole_size ─
    # min_hole_size<=0 keeps the old unconditional-fill behaviour exactly
    # (cpu_fill_holes fills every enclosed background region regardless of
    # size); a positive floor switches to skimage's area-limited fill,
    # which leaves any hole at or above the floor as real background
    # instead of erasing it.
    def _fill_slice(args: tuple) -> tuple:
        z, slc = args
        if min_hole_size <= 0:
            return z, cpu_fill_holes(slc)
        return z, _cpu_remove_small_holes(slc, area_threshold=min_hole_size)

    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        results = list(pool.map(_fill_slice, [(z, smooth_mask[z]) for z in range(Z)]))
    del smooth_mask

    results.sort(key=lambda r: r[0])
    filled_3d = np.stack([r[1] for r in results])

    # ── Step 4: true 3D connected components (26-connectivity) ────────────
    structure         = np.ones((3, 3, 3), dtype=np.int32)
    labeled, n_objects = cpu_label(filled_3d, structure=structure)
    del filled_3d
    print(f"   3D blobs: {n_objects}")

    if n_objects == 0:
        return labeled.astype(np.int32)

    # ── Step 5: remove small blobs ────────────────────────────────────────
    # See _create_labels_cuda's matching comment: the deletion cutoff is
    # final_min_fraction * min_volume, not min_volume itself.
    threshold = max(1, round(final_min_fraction * min_volume))
    max_out = int(labeled.max())
    counts  = np.bincount(labeled.ravel().astype(np.int64), minlength=max_out + 1)

    keep_lut    = counts >= threshold
    keep_lut[0] = True
    output      = np.where(keep_lut[labeled], labeled, 0).astype(np.int32)
    removed     = int(((counts[1:] > 0) & (counts[1:] < threshold)).sum())
    del labeled

    # ── Step 6: renumber 1…N by descending volume ─────────────────────────
    remaining      = np.unique(output[output > 0]).tolist()
    volumes_sorted = sorted(
        [(int(counts[lbl]), int(lbl)) for lbl in remaining], reverse=True
    )
    max_out2 = int(output.max())
    lut2     = np.zeros(max_out2 + 1, dtype=np.int32)
    for new_id, (_vol, old_id) in enumerate(volumes_sorted, start=1):
        lut2[old_id] = new_id

    output  = lut2[output]
    n_final = int(output.max())
    print(f"   3D blobs removed (< {threshold} vox = {final_min_fraction:.3f} x min_volume {min_volume}): {removed}")
    print(f"   Final 3D labels: {n_final}  (label 1 = largest)")
    return output.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def resort_labels(
    labels: np.ndarray,
    sort_by: str = "size",
    reverse: bool = False,
) -> np.ndarray:
    """
    Renumber labels 1…N by the chosen criterion.

    Parameters
    ----------
    labels   : (Z, Y, X) int32 ndarray — existing label volume (0 = background)
    sort_by  : "size" | "centroid_z" | "centroid_y" | "centroid_x" | "complexity"
    reverse  : reverse the natural sort order
                 size        — natural = descending (largest = label 1)
                 centroid    — natural = ascending  (smallest coord = label 1)
                 complexity  — natural = descending (most branched = label 1)

    Returns
    -------
    (Z, Y, X) int32 ndarray — same objects, renumbered 1…N
    """
    from scipy.ndimage import center_of_mass as _com

    unique = np.unique(labels)
    unique = unique[unique > 0]
    if unique.size == 0:
        return labels.copy()

    label_list = unique.tolist()
    max_lbl    = int(unique.max())

    if sort_by == "size":
        counts = np.bincount(labels.ravel().astype(np.int64), minlength=max_lbl + 1)
        keyed  = [(int(counts[lbl]), int(lbl)) for lbl in label_list]
        # natural: descending (largest first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=not reverse)
    elif sort_by == "complexity":
        # "Complexity" = skeleton branch count, the same morphology proxy
        # this project has used throughout (see microglia_cellpose.md) to
        # find genuinely branched/ramified cells rather than just large
        # ones. Branch *count* is scale-independent (pure topology), so a
        # dummy (1,1,1) spacing is fine here -- unlike Statistics' own
        # _skeleton_stats() use, no physical branch length is needed.
        from scipy.ndimage import find_objects
        from ._statistics import _skeleton_stats

        slices = find_objects(labels, max_label=max_lbl)

        def _branch_count(lbl):
            sl = slices[lbl - 1]
            if sl is None:
                return lbl, 0
            binary = labels[sl] == lbl
            n_branches, *_ = _skeleton_stats(binary, (1.0, 1.0, 1.0))
            return lbl, n_branches

        with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 2)) as ex:
            results = list(ex.map(_branch_count, label_list))
        counts = dict(results)
        keyed  = [(counts[lbl], int(lbl)) for lbl in label_list]
        # natural: descending (most branched first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=not reverse)
    else:
        axis_map = {"centroid_z": 0, "centroid_y": 1, "centroid_x": 2}
        axis     = axis_map[sort_by]
        # center_of_mass returns a proper list of tuples for a LIST index
        # (label_list always is one here), even a length-1 list -- the
        # "single bare tuple" case only applies to a scalar index, which
        # this never passes. No extra wrapping needed (a prior version of
        # this code wrapped unconditionally on unique.size==1, which was
        # wrong and would have crashed resorting a single-label volume by
        # centroid -- found and fixed while building a similar sweep that
        # copied the same mistaken assumption from here).
        raw      = _com(labels > 0, labels, label_list)
        keyed = [(float(c[axis]), int(lbl)) for lbl, c in zip(label_list, raw)]
        # natural: ascending (smallest coordinate first → label 1)
        keyed.sort(key=lambda t: t[0], reverse=reverse)

    lut = np.zeros(max_lbl + 1, dtype=np.int32)
    for new_id, (_key, old_id) in enumerate(keyed, start=1):
        lut[old_id] = new_id

    return lut[labels].astype(np.int32)


def remove_debris(labels: np.ndarray, threshold: int) -> "tuple[np.ndarray, int]":
    """Zero out every spatially-connected fragment smaller than threshold
    voxels. A manual edit in napari -- deleting a whole label that turned
    out to be misclassified skin, splitting a label, painting part of
    one away -- can leave small disconnected fragments behind that never
    went through create_labels()'s own volume filter (which only ever
    ran once, before the edit). This is that same filter, callable again
    on demand against whatever the Labels layer currently looks like.

    Deliberately evaluated **per connected component, not per raw label
    ID** -- a naive per-label-ID voxel count (bincount over the whole
    array) silently missed exactly the case this function exists for: a
    manually-erased label's own small leftover fragments still carry
    that same original ID, and summing their voxels together can clear
    the threshold even though each individual disconnected piece is
    genuine debris on its own. 26-connectivity, matching create_labels()'s
    own connected-component convention.

    Deliberately does NOT renumber surviving labels -- unlike
    create_labels()'s own filter, this is a targeted cleanup step on an
    already-in-use label set, not a fresh labeling pass, so IDs the user
    may already be tracking (via Split Label, manual annotation, etc.)
    are left exactly as they were. A label ID with one real surviving
    piece and one small disconnected debris piece keeps its ID on the
    real piece; only the debris piece is zeroed.

    threshold : typically final_min_fraction * min_volume (Common
    Settings), the same golden-ratio-relaxed cutoff create_labels()'s
    own filter and Cellpose-SAM's final_min_size_cleanup() already use.

    Returns (labels, n_removed) -- n_removed counts fragments, not label
    IDs (one ID can contribute more than one removed fragment)."""
    from scipy.ndimage import find_objects

    labels = np.asarray(labels)
    max_lbl = int(labels.max()) if labels.size else 0
    if max_lbl == 0:
        return labels.copy(), 0

    out = labels.copy()
    objs = find_objects(labels)
    n_removed = 0
    for lbl in range(1, max_lbl + 1):
        sl = objs[lbl - 1] if lbl - 1 < len(objs) else None
        if sl is None:
            continue
        n_removed += _remove_debris_from_crop(out[sl], lbl, threshold)

    return out.astype(np.int32), n_removed


def _remove_debris_from_crop(crop: np.ndarray, lbl: int, threshold: int) -> int:
    """
    In-place: zero out connected fragments of value `lbl` within `crop`
    smaller than `threshold` voxels (26-connectivity, matching
    create_labels()'s own convention). Returns the number of fragments
    removed. Shared by remove_debris() (loops over every label in a
    layer) and remove_debris_for_label() (scoped to just one).
    """
    from scipy.ndimage import label as _cc_label
    structure = np.ones((3, 3, 3), dtype=np.int32)
    mask = crop == lbl
    if not mask.any():
        return 0
    cc, n_cc = _cc_label(mask, structure=structure)
    if n_cc <= 1:
        if int(mask.sum()) < threshold:
            crop[mask] = 0
            return 1
        return 0
    counts = np.bincount(cc.ravel())
    removed = 0
    for piece_id in range(1, n_cc + 1):
        if counts[piece_id] < threshold:
            crop[cc == piece_id] = 0
            removed += 1
    return removed


def remove_debris_for_label(labels: np.ndarray, label_id: int, threshold: int) -> "tuple[np.ndarray, int]":
    """
    Same connected-fragment volume filter as remove_debris(), but
    scoped to exactly ONE label -- every other label elsewhere in the
    volume is left completely untouched, unlike running the general
    Remove Debris tool on a whole layer (used as the automatic cleanup
    step after correct_label_from_intensity_3d(), which shouldn't have
    side effects on unrelated cells just because it corrected one).

    Returns (new_labels, n_removed_px) -- pixel count removed (not
    fragment count, unlike remove_debris() -- more informative for a
    single-label report).
    """
    mask = labels == label_id
    if not np.any(mask):
        return labels.copy(), 0
    nz = np.argwhere(mask)
    lo = nz.min(axis=0)
    hi = nz.max(axis=0)
    sl = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo, hi))

    out = labels.copy()
    crop = out[sl]
    before = int((crop == label_id).sum())
    _remove_debris_from_crop(crop, label_id, threshold)
    after = int((crop == label_id).sum())
    return out.astype(np.int32), before - after


def _watershed_split_mask(
    mask: np.ndarray,
    n_splits: int,
    sigma: float,
    min_distance: int,
    surface_full: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Core watershed-split engine, dimension-agnostic (works identically on a
    2D mask -- one Z-slice -- or a full 3D mask) -- extracted so
    split_label() can reuse the exact same pipeline for both its 3D and 2D
    modes instead of maintaining two copies.

    mask         : bool ndarray, any ndim (2D slice or full 3D volume)
    surface_full : optional ndarray, SAME shape as mask (uncropped) -- the
                   raw signal intensity to watershed on instead of the
                   mask's own shape. When None (default, used by 3D mode),
                   the surface is the mask's own EDT -- a purely geometric
                   split that cuts at the shape's thinnest neck, regardless
                   of what the underlying signal looks like there. When
                   given (used by 2D mode), the surface is the actual
                   image intensity -- seeds are placed at local brightness
                   peaks (real cell interiors) and the cut runs along the
                   dimmest ridge between them, i.e. wherever the real
                   signal is weakest, not wherever the mask's silhouette
                   happens to be geometrically narrowest. These can find
                   completely different cuts: a mask can be geometrically
                   wide at a point where the real signal already dips low
                   (e.g. a genuine cell edge, with a merged skin-residue
                   fragment sitting past it) -- EDT has no way to see that
                   dip at all, since it only looks at the mask's outline.

    Returns split_full : int32 ndarray, same shape as mask -- 0=background,
    1..n_splits=parts (part 1 is the seed with the highest surface peak,
    i.e. the thickest chunk in geometric mode, or the brightest chunk in
    intensity mode).

    Raises ValueError if fewer than n_splits distinct sub-regions/peaks are
    found (mirrors the original error messages).
    """
    from scipy.ndimage import distance_transform_edt

    # ── 1. Crop to bounding box (avoids running EDT on the full array) ─────
    nz  = np.argwhere(mask)
    lo  = nz.min(axis=0)
    hi  = nz.max(axis=0)
    pad = max(int(min_distance), int(sigma) + 2, 2)
    lo_p = np.maximum(lo - pad, 0)
    hi_p = np.minimum(hi + pad, np.array(mask.shape) - 1)
    sl   = tuple(slice(int(a), int(b) + 1) for a, b in zip(lo_p, hi_p))

    mask_crop = mask[sl]

    # ── 2. The surface to split: signal intensity if given, else the
    #       mask's own EDT (geometric neck-finding, the original behavior)
    if surface_full is not None:
        dist = surface_full[sl].astype(np.float32)
    else:
        dist = distance_transform_edt(mask_crop).astype(np.float32)

    # ── 3. Gaussian smoothing — GPU if available, CPU fallback ─────────────
    if _BACKEND == "cuda" and _CP is not None:
        try:
            dist_gpu  = _CP.asarray(dist)
            dist_gpu  = _CPND.gaussian_filter(dist_gpu, sigma=float(sigma))
            dist_smooth = dist_gpu.get()
            del dist_gpu
            _free_gpu_cache()
            print(f"   Split: Gaussian smooth on GPU")
        except Exception as exc:
            print(f"   Split: GPU smooth failed ({exc}), using CPU")
            dist_smooth = cpu_gaussian(dist, sigma=float(sigma)) if sigma > 0 else dist
    else:
        dist_smooth = cpu_gaussian(dist, sigma=float(sigma)) if sigma > 0 else dist

    # ── 4. Seed detection via h-maxima (topological prominence) ────────────
    #
    #    peak_local_max uses Euclidean distance — it fails when two big chunks
    #    are spatially close (thin neck) because their centres may be within
    #    min_distance of each other.
    #
    #    h_maxima finds peaks that stand at least h ABOVE their lowest saddle
    #    to any higher peak.  The thin neck IS that saddle, so the two chunk
    #    centres are always separated regardless of their Euclidean distance.
    #
    #    We auto-reduce h (starting at 50% of max EDT) until >= n_splits
    #    topologically distinct peaks are found.  Each peak is then placed at
    #    the EDT maximum inside its h-maxima connected region.
    #
    #    min_distance is used as a final Euclidean guard: if two chosen seeds
    #    are closer than min_distance voxels, the weaker one is dropped.
    from skimage.morphology import h_maxima
    from scipy.ndimage import label as _nd_label

    dist_in_mask = dist_smooth * mask_crop.astype(np.float32)
    max_dist = float(dist_in_mask.max())
    if max_dist == 0:
        raise ValueError("distance transform is zero — blob too flat?")

    # Iteratively reduce h until >= n_splits prominent peaks found
    h_val   = max_dist * 0.50
    h_floor = max_dist * 0.005          # never go below 0.5 % of max EDT
    labeled_hmax = None
    n_found = 0
    while h_val >= h_floor:
        hmax = h_maxima(dist_in_mask, h=float(h_val))
        labeled_hmax, n_found = _nd_label(hmax)
        if n_found >= n_splits:
            break
        h_val *= 0.75

    if n_found < n_splits:
        raise ValueError(
            f"Only {n_found} distinct sub-volume(s) found — "
            f"try reducing Smooth σ"
        )

    # For each h-maxima region pick the voxel with the highest EDT value
    region_peaks = []
    for i in range(1, n_found + 1):
        region_dist = np.where(labeled_hmax == i, dist_in_mask, 0.0)
        coord       = np.array(np.unravel_index(region_dist.argmax(), region_dist.shape))
        peak_val    = float(dist_in_mask[tuple(coord)])
        region_vol  = int((labeled_hmax == i).sum())
        region_peaks.append((peak_val, region_vol, coord))

    # Sort by EDT peak value (thickest chunk centre first) then apply
    # Euclidean min_distance guard to avoid two seeds in the same chunk
    region_peaks.sort(key=lambda t: t[0], reverse=True)
    seeds = []
    for peak_val, _vol, coord in region_peaks:
        if all(np.linalg.norm(coord - s) >= min_distance for s in seeds):
            seeds.append(coord)
        if len(seeds) == n_splits:
            break

    if len(seeds) < n_splits:
        raise ValueError(
            f"Only {len(seeds)} well-separated peak(s) after min-distance "
            f"guard — try reducing Min distance"
        )

    # ── 5. Watershed on negative distance map (finds narrowest boundary) ───
    #    Runs on the cropped region only — fast even on CPU.
    from skimage.segmentation import watershed
    markers = np.zeros(mask_crop.shape, dtype=np.int32)
    for i, c in enumerate(seeds, start=1):
        markers[tuple(c)] = i
    split_crop = watershed(-dist_smooth, markers, mask=mask_crop)

    # ── 6. Clear only the cut interface (1 voxel each side) ──────────────
    eroded_crop = _clear_split_interface(split_crop)

    # ── 7. Write result back into a full-size (mask.shape) array ───────────
    split_full = np.zeros(mask.shape, dtype=np.int32)
    split_full[sl] = eroded_crop
    return split_full


def _clear_split_interface(split_arr: np.ndarray) -> np.ndarray:
    """
    Given an int array with 0=background and 1..N=parts (from any
    watershed split), zero out only the 1-voxel-wide interface where two
    different parts directly touch -- the outer surface of each part is
    left completely untouched. Shared by _watershed_split_mask() (auto
    peak-found seeds) and any marker-seeded watershed split that wants
    the same "clean gap between parts, not a jagged shared boundary"
    treatment.
    """
    eroded = split_arr.copy()
    interface = np.zeros(split_arr.shape, dtype=bool)
    for axis in range(split_arr.ndim):
        slc_lo = [slice(None)] * split_arr.ndim
        slc_hi = [slice(None)] * split_arr.ndim
        slc_lo[axis] = slice(None, -1)
        slc_hi[axis] = slice(1, None)
        slc_lo = tuple(slc_lo)
        slc_hi = tuple(slc_hi)
        both = (
            (split_arr[slc_lo] > 0) &
            (split_arr[slc_hi] > 0) &
            (split_arr[slc_lo] != split_arr[slc_hi])
        )
        tmp_lo = np.zeros(split_arr.shape, dtype=bool)
        tmp_hi = np.zeros(split_arr.shape, dtype=bool)
        tmp_lo[slc_lo] = both
        tmp_hi[slc_hi] = both
        interface |= tmp_lo | tmp_hi
    eroded[interface] = 0
    return eroded


def split_label(
    labels: np.ndarray,
    target_label: int,
    n_splits: int = 2,
    sigma: float = 1.0,
    min_distance: int = 5,
    mode: str = "3d",
    z: "int | None" = None,
    image: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, list[int]]":
    """
    Split one label into n_splits parts using watershed.

    mode="3d" (default) splits the whole 3D blob using the mask's own
    distance transform -- the boundary is placed where the SHAPE is
    geometrically narrowest (the saddle point of the distance map
    between local maxima), independent of what the underlying signal
    looks like there.

    mode="2d" restricts the entire operation to ONE Z-slice (`z`) AND
    watersheds on the raw signal intensity (`image[z]`) instead of the
    mask's shape -- seeds are placed at local brightness peaks (real
    cell interiors) and the cut runs along the dimmest ridge between
    them, i.e. wherever the actual signal is weakest. This is for the
    case where two things only touch on a single cross-section -- e.g.
    real signal happening to graze a skin-residue fragment right at
    that slice -- where the mask's own outline can be geometrically
    wide right through a point the real signal already dips low at (a
    true 3D neck doesn't exist there, and even in 2D the mask's own
    EDT has no way to see an intensity dip it isn't shaped around).
    Only that one slice is touched -- every other Z-slice of the label
    is left completely untouched, and the new part(s) exist only on
    that slice (they don't extend into neighboring slices the way a
    3D split's parts would).

    Speed notes
    -----------
    - All operations run on the bounding box of the target label (or its
      footprint on the given slice, in 2D mode), not the full volume/slice
      — critical for large stacks.
    - Gaussian smoothing runs on GPU (CuPy) when available.

    Parameters
    ----------
    labels       : (Z, Y, X) int32 ndarray
    target_label : label value to split
    n_splits     : number of parts to produce (≥ 2)
    sigma        : Gaussian smoothing of the split surface (higher = broader peaks)
    min_distance : minimum voxel distance between seed peaks
    mode         : "3d" (default) or "2d"
    z            : slice index, required when mode="2d" (ignored for "3d")
    image        : (Z, Y, X) raw signal volume, same shape as labels --
                   required when mode="2d" (ignored for "3d", which never
                   looks at signal intensity)

    Returns
    -------
    (new_labels, new_ids)
        new_labels — same shape as labels, blob split into n_splits parts
        new_ids    — list of n_splits-1 new label IDs created
                     (target_label is kept for part 1)

    Raises
    ------
    ValueError  if mode is invalid, z/image is missing or the wrong shape
                for mode="2d", the label is not found (in the volume, or
                on slice z), or fewer peaks than n_splits are found
    """
    if mode not in ("3d", "2d"):
        raise ValueError(f"mode must be '3d' or '2d', got {mode!r}")

    if mode == "2d":
        if z is None:
            raise ValueError("z (slice index) is required when mode='2d'")
        if not (0 <= z < labels.shape[0]):
            raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
        if image is None:
            raise ValueError("image (raw signal volume) is required when mode='2d'")
        if image.shape != labels.shape:
            raise ValueError(f"image shape {image.shape} != labels shape {labels.shape}")
        mask = labels[z] == target_label
        if not np.any(mask):
            raise ValueError(f"Label {target_label} not found on slice {z}")
        surface_full = image[z].astype(np.float32)
    else:
        mask = labels == target_label
        if not np.any(mask):
            raise ValueError(f"Label {target_label} not found")
        surface_full = None

    split_full = _watershed_split_mask(mask, n_splits, sigma, min_distance, surface_full=surface_full)

    out     = labels.copy()
    out_view = out[z] if mode == "2d" else out
    new_ids = []
    max_lbl = int(labels.max())

    # Zero out the original blob first (gap voxels become background)
    out_view[mask] = 0
    out_view[split_full == 1] = target_label
    for i in range(2, n_splits + 1):
        new_id = max_lbl + (i - 1)
        out_view[split_full == i] = new_id
        new_ids.append(new_id)

    for i, nid in enumerate([target_label] + new_ids, start=1):
        n_vox = int((split_full == i).sum())
        print(f"   Part {i}: {n_vox:,} vox  (id {nid})")

    return out.astype(np.int32), new_ids


def join_labels(labels: np.ndarray, label_a: int, label_b: int) -> np.ndarray:
    """
    Merge label_b into label_a -- every voxel currently labeled label_b
    becomes label_a instead. The inverse of split_label(): two labels
    that are really one cell, wrongly segmented into two pieces (e.g. a
    thin neck that fooled the segmenter into cutting it in half),
    collapsed back into one. label_a survives; label_b's ID disappears.

    A single vectorized boolean assignment over the whole volume --
    unlike split_label(), there's no bounding-box crop to compute
    (nothing here depends on shape/geometry) and no GPU path needed,
    so this stays fast even on a full-fish volume without one.

    Returns new_labels (same shape, same dtype). Raises ValueError if
    either label is not found, or if label_a == label_b.
    """
    if label_a == label_b:
        raise ValueError("Label A and Label B must be different labels.")
    mask_a = labels == label_a
    mask_b = labels == label_b
    if not np.any(mask_a):
        raise ValueError(f"Label {label_a} not found")
    if not np.any(mask_b):
        raise ValueError(f"Label {label_b} not found")

    new_labels = labels.copy()
    new_labels[mask_b] = label_a
    return new_labels


def correct_label_from_intensity(
    labels: np.ndarray,
    image: np.ndarray,
    label_id: int,
    z: int,
    lo: float,
    hi: float,
    pad: int = 15,
) -> np.ndarray:
    """
    Regenerate one label's shape on one Z-slice from a raw-signal
    intensity window, instead of hand-painting it -- for the case
    where a particular contrast window (read straight from the signal
    image layer's own contrast_limits) happens to trace the cell's
    true silhouette better than the existing segmentation does on
    that slice.

    labels    : (Z, Y, X) label volume
    image     : (Z, Y, X) raw signal volume, same shape as labels
    label_id  : the label being corrected
    z         : slice index to correct -- only this slice is touched
    lo, hi    : the signal layer's own contrast_limits. Only lo is
                actually used as the foreground cutoff (candidate =
                image >= lo) -- hi is napari's display SATURATION
                ceiling, not an upper bound on what still counts as
                real signal. A band threshold (lo <= image <= hi) was
                tried first and got this backwards: with a narrow
                window like [100, 101] chosen specifically to make
                the display saturate into a clean silhouette, a
                cell's true bright interior sits well ABOVE hi (it's
                just displayed as solid white, i.e. saturated) --
                excluding it as "not candidate" emptied the label,
                while scattered background pixels that happened to
                fall inside the narrow band got pulled in instead.
                hi is still accepted (and shown in status messages)
                for context, but never restricts the mask.
    pad       : bounding-box padding in pixels around the label's
                existing footprint on this slice, XY only (matches
                extract_cellpose_crops.py's own convention)

    A naive threshold within the padded bounding box would happily
    pick up a neighboring cell that happens to fall inside the same
    box -- two real problems, not one: (a) a bright neighbor sitting
    fully inside the crop but never touching label_id at all, and (b)
    a neighbor whose thresholded pixels touch label_id's own
    candidate pixels, which a plain connected-component pass would
    then fuse into one blob. Both are handled the same way: any pixel
    already claimed by a DIFFERENT existing label is excluded from
    the candidate mask outright, before connected components ever
    run -- so a foreign label can never be grown into, merged in, or
    even present in the corrected shape, regardless of how the
    intensity window or connectivity falls.

    Returns new_labels (same shape, same dtype) with only that one
    slice's label_id footprint changed. Raises ValueError if label_id
    isn't present on slice z, or if the intensity window leaves
    nothing connected to the label's own existing footprint (a wrong
    contrast window would otherwise silently erase the label instead
    of correcting it).
    """
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z  = image[z]
    try:
        corrected, crop_existing, (y0, y1, x0, x1) = _intensity_correct_2d(
            labels_z, image_z, label_id, lo, pad
        )
    except ValueError as exc:
        raise ValueError(f"{exc} (slice {z})") from None

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    crop[crop_existing] = 0
    crop[corrected] = label_id
    return new_labels


def correct_label_from_intensity_3d(
    labels: np.ndarray,
    image: np.ndarray,
    label_id: int,
    lo: float,
    pad: int = 15,
    min_volume: "int | None" = None,
    final_min_fraction: float = 0.618,
) -> "tuple[np.ndarray, dict]":
    """
    3D version of correct_label_from_intensity(): corrects the WHOLE
    cell, not just one slice.

    Finds label_id's own 3D centroid, corrects that slice first (same
    engine as the 2D tool, seeded by its own existing footprint there),
    then walks outward in both +Z and -Z from the centroid slice. Each
    step is seeded by the PREVIOUS step's own corrected shape (not
    necessarily that slice's original footprint) -- so the correction
    can both reshape slices that already carried the label AND grow
    into a slice the original label never touched at all, as long as
    the signal and connectivity support it. Each direction's walk stops
    the moment a step produces nothing (no candidate pixels connect to
    the previous step's seed there) -- a natural stopping point, not a
    fixed slice count.

    Beyond that stopping point, any of this label's OWN original pixels
    still remaining are TRIMMED (cleared to background), contiguously,
    until the original label's own extent genuinely ends in that
    direction -- Cellpose-SAM having labeled something there that the
    recalibrated contrast threshold no longer supports as real signal
    is exactly the case this exists for; left alone, that's garbage
    Remove Debris might catch (if it happens to be small) or might not
    (nothing about size alone distinguishes a false extension from a
    real, differently-shaped continuation of the cell), so it's swept
    unconditionally instead of relying on a size heuristic. Only this
    label's own pixels are ever touched by the trim -- it never reaches
    past a genuine gap to affect something unrelated sharing the ID.

    After the walk (and trim), remove_debris_for_label() (golden-ratio-relaxed
    floor, same as Cellpose-SAM's own final safety net) cleans up any
    small disconnected fragment left over from the correction -- scoped
    to ONLY this label, unlike running the general Remove Debris tool
    on the whole layer.

    labels, image      : (Z, Y, X) volumes, same shape
    label_id            : the label being corrected
    lo                  : one-sided intensity cutoff (signal = image >= lo),
                          held constant across every slice in the walk
    pad                 : bbox padding in pixels, XY only, per slice
    min_volume          : if given (with final_min_fraction), the debris
                          floor is final_min_fraction * min_volume voxels
                          -- pass None to skip the debris-cleanup step
                          entirely (report will show n_debris_removed_px=0)
    final_min_fraction  : golden ratio (0.618) by default, matching
                          every other final-safety-net stage in this plugin

    Returns (new_labels, report). report is a dict:
        z_center            -- the starting slice (nearest to the
                               label's own pre-correction 3D centroid)
        slices_corrected    -- sorted list of every Z the walk itself
                               regenerated
        slices_trimmed      -- sorted list of every Z where original
                               label_id pixels beyond the walk's own
                               reach were cleared (see above)
        n_trimmed_px        -- total pixels cleared by trimming
        n_debris_removed_px -- pixels removed by the debris-cleanup step
        foreign_touching    -- {z: sorted [foreign label ids]} -- IDs
                               whose pixels directly border (8-connected)
                               this label's corrected footprint on that
                               slice, for slices where this is non-empty
        foreign_nearby      -- {z: sorted [foreign label ids]} -- IDs
                               present anywhere inside the padded bbox
                               region this slice's correction actually
                               worked within, even where not touching,
                               for slices where this is non-empty. (A
                               literal "foreign pixels included inside
                               this label" check is not meaningful for a
                               label array -- each voxel holds exactly
                               one label value, so true overlap is
                               structurally impossible; this is the
                               closest real, useful signal: a foreign
                               blob sitting inside the correction's own
                               working neighborhood, worth a manual
                               look even though it was never actually
                               absorbed -- the foreign-exclusion guard
                               inside _intensity_grow_2d makes that part
                               impossible by construction.)

    Raises ValueError if label_id isn't found anywhere in the volume,
    or if even the centroid slice can't be corrected (nothing connects
    to its own existing footprint there).
    """
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    mask3d = labels == label_id
    if not np.any(mask3d):
        raise ValueError(f"label {label_id} not found anywhere in the volume")

    from scipy.ndimage import center_of_mass as _com
    cz, _cy, _cx = _com(mask3d)
    z_dim = labels.shape[0]
    z_center = int(round(cz))
    z_center = max(0, min(z_dim - 1, z_center))

    zs_with_label = np.unique(np.nonzero(mask3d)[0])
    if not np.any(labels[z_center] == label_id):
        # centroid can land on a slice the label doesn't actually occupy
        # (a branchy/non-convex 3D shape) -- snap to the nearest real one
        z_center = int(zs_with_label[np.argmin(np.abs(zs_with_label - z_center))])

    new_labels = labels.copy()
    bboxes: "dict[int, tuple[int, int, int, int]]" = {}

    corrected0, crop_seed0, bbox0 = _intensity_correct_2d(
        new_labels[z_center], image[z_center], label_id, lo, pad
    )
    y0, y1, x0, x1 = bbox0
    crop = new_labels[z_center, y0:y1, x0:x1]
    crop[crop_seed0] = 0
    crop[corrected0] = label_id
    bboxes[z_center] = bbox0

    seed_full = np.zeros(labels.shape[1:], dtype=bool)
    seed_full[y0:y1, x0:x1] = corrected0
    center_seed_full = seed_full  # kept to start the -Z walk from the same slice

    slices_corrected = {z_center}

    slices_trimmed = set()
    n_trimmed_px = 0

    for direction, z_range in ((1, range(z_center + 1, z_dim)),
                                (-1, range(z_center - 1, -1, -1))):
        prev_seed = center_seed_full
        stop_z = None
        for z in z_range:
            result = _intensity_grow_2d(image[z], new_labels[z], label_id, lo, pad, prev_seed)
            if result is None:
                stop_z = z  # natural stop: nothing here connects to the previous slice
                break
            corrected, _crop_seed, (gy0, gy1, gx0, gx1) = result
            crop = new_labels[z, gy0:gy1, gx0:gx1]
            crop[crop == label_id] = 0  # clear any of this label's own old pixels here first
            crop[corrected] = label_id
            bboxes[z] = (gy0, gy1, gx0, gx1)
            slices_corrected.add(z)

            prev_seed = np.zeros(labels.shape[1:], dtype=bool)
            prev_seed[gy0:gy1, gx0:gx1] = corrected

        # Trim: beyond the point the walk stopped, any of this label's OWN
        # original pixels still remaining are exactly the case this exists
        # for -- Cellpose-SAM labeled something there that the recalibrated
        # contrast threshold no longer supports as real signal. Left alone,
        # that's garbage that Remove Debris might catch (if it's small) or
        # might not (nothing about size alone tells you a chunk is a false
        # extension rather than a real, differently-shaped continuation of
        # the cell) -- so it's cleared explicitly here instead of relying
        # on that. Contiguous only: stops the instant the original label's
        # own extent genuinely ends in this direction, never reaching past
        # it to touch something unrelated that happens to share the ID.
        if stop_z is not None:
            trim_z = stop_z
            while 0 <= trim_z < z_dim:
                mask_here = new_labels[trim_z] == label_id
                if not np.any(mask_here):
                    break
                n_trimmed_px += int(mask_here.sum())
                new_labels[trim_z][mask_here] = 0
                slices_trimmed.add(trim_z)
                trim_z += direction

    n_debris_removed_px = 0
    if min_volume is not None:
        threshold = final_min_fraction * min_volume
        new_labels, n_debris_removed_px = remove_debris_for_label(new_labels, label_id, threshold)

    from scipy.ndimage import binary_dilation
    foreign_touching: "dict[int, list[int]]" = {}
    foreign_nearby: "dict[int, list[int]]" = {}
    for z in sorted(slices_corrected):
        own = new_labels[z] == label_id
        if not np.any(own):
            continue  # debris cleanup removed this slice's contribution entirely
        dilated = binary_dilation(own, structure=np.ones((3, 3), dtype=bool))
        touching_here = new_labels[z][dilated & ~own]
        touching_ids = sorted(int(i) for i in np.unique(touching_here) if i not in (0, label_id))
        if touching_ids:
            foreign_touching[z] = touching_ids

        gy0, gy1, gx0, gx1 = bboxes.get(z, (0, new_labels.shape[1], 0, new_labels.shape[2]))
        nearby_here = new_labels[z, gy0:gy1, gx0:gx1]
        nearby_ids = sorted(int(i) for i in np.unique(nearby_here) if i not in (0, label_id))
        if nearby_ids:
            foreign_nearby[z] = nearby_ids

    report = {
        "z_center": z_center,
        "slices_corrected": sorted(slices_corrected),
        "slices_trimmed": sorted(slices_trimmed),
        "n_trimmed_px": n_trimmed_px,
        "n_debris_removed_px": n_debris_removed_px,
        "foreign_touching": foreign_touching,
        "foreign_nearby": foreign_nearby,
    }
    return new_labels.astype(np.int32), report


def _intensity_correct_2d(
    labels_z: np.ndarray,
    image_z: np.ndarray,
    label_id: int,
    lo: float,
    pad: int,
) -> "tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]":
    """
    Core 2D intensity-threshold engine behind correct_label_from_intensity(),
    extracted so a calibration sweep (many candidate `lo` values) can reuse
    it directly without copying/returning a full label volume on every
    candidate -- this returns just the small cropped boolean result.

    labels_z, image_z : one Z-slice each (2D), same shape
    label_id           : the label being corrected
    lo                  : one-sided intensity cutoff (signal = image >= lo)
    pad                 : bbox padding in pixels

    Returns (corrected, crop_existing, (y0, y1, x0, x1)):
        corrected     — bool array, shape (y1-y0, x1-x0): the label's new
                         footprint within the crop
        crop_existing — bool array, same crop shape: the label's OLD
                         footprint within the crop (what production code
                         needs to clear before painting `corrected`)
        (y0,y1,x0,x1) — the crop's bounds within labels_z/image_z, for the
                         caller to splice back with

    Raises ValueError if label_id isn't on this slice, or if the threshold
    leaves nothing connected to the label's existing footprint.
    """
    existing = labels_z == label_id
    if not np.any(existing):
        raise ValueError(f"label {label_id} not found")

    result = _intensity_grow_2d(image_z, labels_z, label_id, lo, pad, existing)
    if result is None:
        raise ValueError(
            f"threshold >= {lo} leaves nothing connected to "
            f"label {label_id}'s existing footprint -- refusing to erase "
            f"the label; adjust the contrast window and try again."
        )
    corrected, crop_seed, bbox = result
    return corrected, crop_seed, bbox


def _intensity_grow_2d(
    image_z: np.ndarray,
    labels_z: np.ndarray,
    label_id: int,
    lo: float,
    pad: int,
    seed_mask: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None":
    """
    Threshold + connected-component engine shared by
    _intensity_correct_2d() (per-slice correction, seed = that slice's
    own existing label_id footprint) and
    correct_label_from_intensity_3d()'s Z-walk (seed = the previous
    step's own corrected shape, projected onto the next slice -- so the
    correction can grow into a slice the original label never touched
    at all, not just reshape a slice that already carried it).

    seed_mask : bool, same shape as image_z/labels_z -- non-empty
                (caller's responsibility; this function doesn't itself
                know whether an empty seed means "label not found" or
                "the Z-walk should stop here", since that's a different
                error for each caller)

    Returns (corrected, crop_seed, (y0,y1,x0,x1)) within the crop only
    -- same convention _intensity_correct_2d() already returned.
    Returns None (not an exception) if the threshold leaves nothing
    connected to seed_mask -- the caller decides what that means.
    """
    ys, xs = np.nonzero(seed_mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_seed   = seed_mask[y0:y1, x0:x1]

    candidate = crop_image >= lo  # one-sided: signal is "at/above lo", not a narrow band
    foreign = (crop_labels != 0) & (crop_labels != label_id)
    candidate &= ~foreign  # never claim another label's territory

    cc, _ = cpu_label(candidate)
    keep_ids = set(np.unique(cc[crop_seed & (cc > 0)]))
    keep_ids.discard(0)
    if not keep_ids:
        return None
    corrected = np.isin(cc, list(keep_ids))
    return corrected, crop_seed, (y0, y1, x0, x1)


def correct_adjacent_labels_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_a: int,
    label_b: int,
    z: int,
    lo: float,
    pad: int = 15,
    sigma: float = 1.0,
) -> "tuple[np.ndarray, dict]":
    """
    Corrects TWO adjacent/touching labels on ONE slice simultaneously
    from the raw signal's intensity threshold, with the boundary
    between them placed by watershed on the signal itself -- for the
    case where two things (two real cells, or a cell and a skin-residue
    fragment) end up merged together on a particular cross-section
    (e.g. right after Copy Label to Adjacent Slice pastes a shape that
    now touches a neighbor there).

    Correcting each label independently with correct_label_from_intensity()
    doesn't handle this: at a shared contrast threshold, both labels'
    regenerated regions can fuse into one connected blob exactly where
    they touch, and each single-label correction's own foreign-exclusion
    guard would just draw the boundary wherever the OTHER label's stale
    ORIGINAL pixels happened to already sit -- not the real signal
    boundary between them, which is usually different once both are
    corrected together.

    The cut is placed by a MARKER-SEEDED watershed, anchored at each
    label's own existing footprint -- not by Split Label's blind
    peak-finding (auto-detecting the two most prominent intensity peaks
    anywhere in the combined region and cutting between THOSE). Peak-
    finding has no notion of "these two regions already have their own
    separate identities and an existing boundary between them" -- on a
    combined region with more than one real intensity dip (e.g. a
    genuine internal texture variation inside one of the two cells, in
    addition to the real seam between them), it can just as easily lock
    onto the wrong valley entirely, one unrelated to where the two
    labels actually meet. Seeding directly from each label's own
    existing region instead means the watershed floods outward from
    each cell's own known territory and the two fronts meet wherever
    they meet -- the resulting boundary naturally tracks close to the
    ORIGINAL touching boundary's neighborhood while still following the
    real per-pixel signal (so it isn't just re-drawing the stale shape
    either), rather than being pulled toward some other, unrelated dip.

    labels, image    : (Z, Y, X) volumes, same shape
    label_a, label_b : the two labels being corrected together (must be
                       different, both present on slice z)
    z                : slice index -- only this slice is touched
    lo               : one-sided intensity cutoff (signal = image >= lo)
    pad              : bbox padding in pixels around the UNION of both
                       labels' existing footprints on this slice
    sigma            : Gaussian smoothing of the signal before watershed
                       (higher = less sensitive to single-pixel noise
                       nudging the boundary around)

    Returns (new_labels, info). info is a dict:
        n_a, n_b -- final pixel counts for label_a/label_b after correction
        n_lost   -- pixels that were label_a or label_b before, but ended
                    up neither after the joint correction/split (e.g. a
                    sliver the threshold no longer supports, or that
                    neither watershed part reached) -- reported, not
                    hidden, so a meaningful loss doesn't go unnoticed

    Raises ValueError if label_a == label_b, either label isn't present
    on slice z, the joint threshold connects to neither label's existing
    footprint at all, or one label's own marker ends up with zero
    reachable candidate pixels within the combined region (its own
    existing footprint doesn't meet the new threshold at all) -- refuses
    to silently erase one label rather than returning a 1-label result.
    """
    if label_a == label_b:
        raise ValueError("label_a and label_b must be different labels")
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z = image[z]
    existing_a = labels_z == label_a
    existing_b = labels_z == label_b
    if not np.any(existing_a):
        raise ValueError(f"label {label_a} not found on slice {z}")
    if not np.any(existing_b):
        raise ValueError(f"label {label_b} not found on slice {z}")

    seed = existing_a | existing_b
    ys, xs = np.nonzero(seed)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_seed   = seed[y0:y1, x0:x1]
    crop_a      = existing_a[y0:y1, x0:x1]
    crop_b      = existing_b[y0:y1, x0:x1]

    candidate = crop_image >= lo  # one-sided, same convention as every other Correct Label tool
    foreign = (crop_labels != 0) & (crop_labels != label_a) & (crop_labels != label_b)
    candidate &= ~foreign  # a THIRD label's territory is still off-limits

    cc, _ = cpu_label(candidate)
    keep_ids = set(np.unique(cc[crop_seed & (cc > 0)]))
    keep_ids.discard(0)
    if not keep_ids:
        raise ValueError(
            f"threshold >= {lo} leaves nothing connected to either label "
            f"{label_a} or {label_b}'s existing footprint on slice {z} -- "
            f"refusing to erase both labels; adjust the contrast window "
            f"and try again."
        )
    combined = np.isin(cc, list(keep_ids))

    # Marker-seeded watershed: markers are exactly each label's own
    # EXISTING footprint (not auto-detected peaks) -- floods outward from
    # each cell's own known territory on the (smoothed) real signal, so
    # the two fronts meet near the actual current boundary's neighborhood
    # rather than being pulled toward some other, unrelated intensity dip
    # elsewhere in the combined region. See the docstring above for why
    # Split Label's peak-finding approach was tried first and rejected.
    from skimage.segmentation import watershed
    if sigma > 0:
        crop_image_smooth = cpu_gaussian(crop_image.astype(np.float32), sigma=float(sigma))
    else:
        crop_image_smooth = crop_image.astype(np.float32)

    markers = np.zeros(combined.shape, dtype=np.int32)
    markers[crop_a] = 1
    markers[crop_b] = 2
    split_crop = watershed(-crop_image_smooth, markers, mask=combined)
    split_crop = _clear_split_interface(split_crop)

    crop_final_a = split_crop == 1
    crop_final_b = split_crop == 2
    if not np.any(crop_final_a) or not np.any(crop_final_b):
        # A marker with no candidate pixels of its own reachable within
        # `combined` gets 0 output pixels from skimage's watershed,
        # silently -- e.g. label_a's own existing footprint doesn't meet
        # the new threshold at all. Refuse rather than erase one label
        # outright, matching every other Correct Label tool's guard.
        empty_lbl = label_a if not np.any(crop_final_a) else label_b
        raise ValueError(
            f"threshold >= {lo} leaves label {empty_lbl} with nothing -- "
            f"refusing to erase it; adjust the contrast window and try again."
        )

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    crop[crop_a] = 0
    crop[crop_b] = 0
    crop[crop_final_a] = label_a
    crop[crop_final_b] = label_b

    n_lost = int((crop_seed & ~(crop_final_a | crop_final_b)).sum())
    info = {
        "n_a": int(crop_final_a.sum()),
        "n_b": int(crop_final_b.sum()),
        "n_lost": n_lost,
    }
    return new_labels.astype(np.int32), info


def correct_label_group_2d(
    labels: np.ndarray,
    image: np.ndarray,
    label_ids: "list[int]",
    z: int,
    lo: float,
    pad: int = 15,
    sigma: float = 1.0,
) -> "tuple[np.ndarray, dict]":
    """
    N-label generalization of correct_adjacent_labels_2d(): jointly
    corrects an arbitrary GROUP of mutually-touching labels on ONE
    slice from the raw signal, via the same marker-seeded watershed
    (markers = each label's own existing footprint, never an
    auto-detected peak) -- see correct_adjacent_labels_2d()'s own
    docstring for why that anchoring matters. Built for
    auto_contrast_correct_stack()'s slice-by-slice pass, where more
    than two cells can end up mutually touching on a given slice after
    an independent cell-by-cell 3D correction pass, not just pairs.

    label_ids : 2 or more distinct label IDs, all present on slice z.
                A single-ID call degenerates to plain single-label
                intensity correction (one marker floods everything the
                threshold connects it to) -- supported for uniformity,
                though the pipeline that drives this only ever calls it
                with real touching groups (size >= 2).

    Returns (new_labels, info). info is a dict:
        n_lost   -- pixels that were one of label_ids before, but ended
                    up belonging to NONE of them after the joint
                    correction/split (reported, not hidden)
        <label_id>: <final pixel count>  -- one entry per label in the
                    group

    Raises ValueError if label_ids has duplicates, fewer than 1 label
    is found on slice z, the joint threshold connects to none of the
    group's existing footprint at all, or any one label's own marker
    ends up with zero reachable pixels within the combined region --
    refuses to silently erase a label rather than returning a result
    missing one.
    """
    if len(label_ids) < 1:
        raise ValueError("label_ids must contain at least one label")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(f"label_ids must not contain duplicates: {label_ids}")
    if not (0 <= z < labels.shape[0]):
        raise ValueError(f"slice {z} out of range for a {labels.shape[0]}-slice volume")
    if labels.shape != image.shape:
        raise ValueError(f"labels shape {labels.shape} != image shape {image.shape}")

    labels_z = labels[z]
    image_z = image[z]

    existing: "dict[int, np.ndarray]" = {}
    seed = np.zeros(labels_z.shape, dtype=bool)
    for lid in label_ids:
        m = labels_z == lid
        if not np.any(m):
            raise ValueError(f"label {lid} not found on slice {z}")
        existing[lid] = m
        seed |= m

    ys, xs = np.nonzero(seed)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, labels_z.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, labels_z.shape[1])

    crop_labels = labels_z[y0:y1, x0:x1]
    crop_image  = image_z[y0:y1, x0:x1]
    crop_seed   = seed[y0:y1, x0:x1]
    crop_existing = {lid: m[y0:y1, x0:x1] for lid, m in existing.items()}

    candidate = crop_image >= lo  # one-sided, same convention as every other Correct Label tool
    foreign = (crop_labels != 0) & (~np.isin(crop_labels, list(label_ids)))
    candidate &= ~foreign  # any label OUTSIDE this group is still off-limits

    cc, _ = cpu_label(candidate)
    keep_ids = set(np.unique(cc[crop_seed & (cc > 0)]))
    keep_ids.discard(0)
    if not keep_ids:
        raise ValueError(
            f"threshold >= {lo} leaves nothing connected to any of "
            f"{list(label_ids)}'s existing footprint on slice {z} -- "
            f"refusing to erase the whole group; adjust the contrast "
            f"window and try again."
        )
    combined = np.isin(cc, list(keep_ids))

    # Marker-seeded watershed: markers are exactly each label's own
    # EXISTING footprint (not auto-detected peaks) -- see
    # correct_adjacent_labels_2d()'s docstring for the full rationale.
    from skimage.segmentation import watershed
    if sigma > 0:
        crop_image_smooth = cpu_gaussian(crop_image.astype(np.float32), sigma=float(sigma))
    else:
        crop_image_smooth = crop_image.astype(np.float32)

    markers = np.zeros(combined.shape, dtype=np.int32)
    for i, lid in enumerate(label_ids, start=1):
        markers[crop_existing[lid]] = i
    split_crop = watershed(-crop_image_smooth, markers, mask=combined)
    split_crop = _clear_split_interface(split_crop)

    finals = {}
    empties = []
    for i, lid in enumerate(label_ids, start=1):
        finals[lid] = split_crop == i
        if not np.any(finals[lid]):
            empties.append(lid)
    if empties:
        raise ValueError(
            f"threshold >= {lo} leaves label(s) {empties} with nothing -- "
            f"refusing to erase; adjust the contrast window and try again."
        )

    new_labels = labels.copy()
    crop = new_labels[z, y0:y1, x0:x1]
    for lid in label_ids:
        crop[crop_existing[lid]] = 0
    for lid in label_ids:
        crop[finals[lid]] = lid

    any_final = np.zeros(combined.shape, dtype=bool)
    for lid in label_ids:
        any_final |= finals[lid]
    n_lost = int((crop_seed & ~any_final).sum())

    info = {"n_lost": n_lost}
    for lid in label_ids:
        info[lid] = int(finals[lid].sum())
    return new_labels.astype(np.int32), info


def _touching_pairs_on_slice(labels_z: np.ndarray) -> "set[frozenset]":
    """
    Every pair of distinct, nonzero labels that directly (8-connected)
    border each other on this one 2D slice. Bbox-cropped per label (via
    find_objects) so cost scales with how many cells are actually on
    this slice, not the slice's full size.
    """
    from scipy.ndimage import binary_dilation, find_objects

    present = np.unique(labels_z)
    present = present[present > 0]
    pairs: "set[frozenset]" = set()
    if present.size < 2:
        return pairs

    objs = find_objects(labels_z, max_label=int(present.max()))
    struct = np.ones((3, 3), dtype=bool)
    for lid in present.tolist():
        sl = objs[lid - 1] if lid - 1 < len(objs) else None
        if sl is None:
            continue
        pad_sl = tuple(
            slice(max(s.start - 1, 0), min(s.stop + 1, labels_z.shape[i]))
            for i, s in enumerate(sl)
        )
        crop = labels_z[pad_sl]
        own = crop == lid
        if not np.any(own):
            continue
        dilated = binary_dilation(own, structure=struct)
        neighbor_ids = np.unique(crop[dilated & ~own])
        for nid in neighbor_ids.tolist():
            if nid != 0 and nid != lid:
                pairs.add(frozenset((int(lid), int(nid))))
    return pairs


def _union_find_groups(elements: "list[int]", pairs: "set[frozenset]") -> "list[list[int]]":
    parent = {e: e for e in elements}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in pairs:
        a, b = tuple(pair)
        if a in parent and b in parent:
            union(a, b)

    groups: "dict[int, list[int]]" = {}
    for e in elements:
        r = find(e)
        groups.setdefault(r, []).append(e)
    return list(groups.values())


def touching_groups_for_stack(labels: np.ndarray) -> "dict[int, list[list[int]]]":
    """
    For every Z slice, groups the labels present there into
    mutually-touching (8-connected) clusters. Returns {z: [group, ...]}
    -- slices with no group of size >= 2 (every label on that slice has
    no neighbor there) are omitted entirely, since they need no joint
    correction. Each group is a sorted list of label IDs.

    Used by auto_contrast_correct_stack()'s second pass: an independent
    cell-by-cell correction can leave two (or more) cells directly
    touching wherever their newly-recalibrated shapes meet, and which
    cell "wins" a contested boundary pixel there is an arbitrary
    accident of processing order -- this is how that pass finds exactly
    which (slice, cell-group) combinations need to be re-derived
    jointly instead of left as that greedy artifact.
    """
    result: "dict[int, list[list[int]]]" = {}
    for z in range(labels.shape[0]):
        labels_z = labels[z]
        present = np.unique(labels_z)
        present = present[present > 0]
        if present.size < 2:
            continue
        pairs = _touching_pairs_on_slice(labels_z)
        if not pairs:
            continue
        groups = _union_find_groups([int(p) for p in present.tolist()], pairs)
        multi = [sorted(g) for g in groups if len(g) >= 2]
        if multi:
            result[z] = multi
    return result


def copy_label_to_adjacent_slice(
    labels: np.ndarray,
    label_id: int,
    z_src: int,
    direction: int,
) -> "tuple[np.ndarray, int]":
    """
    Copy label_id's 2D footprint from slice z_src onto the adjacent
    slice z_src + direction -- e.g. to patch a slice where this cell's
    cross-section is missing or broken by reusing a good neighboring
    slice's shape.

    labels    : (Z, Y, X) label volume
    label_id  : the label whose shape is being copied
    z_src     : source slice index (the currently-viewed slice)
    direction : +1 (copy to the next slice) or -1 (copy to the
                previous slice)

    label_id's own OLD footprint on the target slice is cleared first,
    then the copied shape is painted in -- re-running this doesn't
    leave a stale double outline behind. Any pixel on the target slice
    already claimed by a DIFFERENT existing label is excluded from the
    copy outright and left completely untouched (same foreign-label
    protection principle as correct_label_from_intensity() and
    rerun_single_cell()'s crop-splice guard) -- natural shape drift
    between slices means the copied footprint can land partly on a
    neighboring cell's territory on the target slice, and that
    territory must never be overwritten.

    Returns (new_labels, n_excluded_px) -- new_labels is the same
    shape/dtype with only the target slice's label_id footprint
    changed; n_excluded_px is how many of the source shape's pixels
    were dropped because they landed on a foreign label (0 for a
    clean copy), so the caller can report a partial-overlap copy
    honestly instead of silently pretending it was exact. Raises
    ValueError if direction isn't +-1, if the source or target slice
    is out of range, if label_id isn't present on z_src, or if every
    pixel would be dropped to foreign-label territory (refuses to
    paste nothing).
    """
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or +1, got {direction}")
    if not (0 <= z_src < labels.shape[0]):
        raise ValueError(f"slice {z_src} out of range for a {labels.shape[0]}-slice volume")
    z_dst = z_src + direction
    if not (0 <= z_dst < labels.shape[0]):
        raise ValueError(
            f"target slice {z_dst} out of range for a {labels.shape[0]}-slice volume "
            f"-- there is no {'next' if direction > 0 else 'previous'} slice"
        )

    src_mask = labels[z_src] == label_id
    if not np.any(src_mask):
        raise ValueError(f"label {label_id} not found on slice {z_src}")

    dst_slice = labels[z_dst]
    foreign = (dst_slice != 0) & (dst_slice != label_id)
    paint_mask = src_mask & ~foreign  # never claim another label's territory
    n_excluded_px = int(np.sum(src_mask & foreign))

    if not np.any(paint_mask):
        raise ValueError(
            f"every pixel of label {label_id}'s shape on slice {z_src} lands on "
            f"a different label on slice {z_dst} -- refusing to paste nothing."
        )

    new_labels = labels.copy()
    new_dst = new_labels[z_dst]
    new_dst[new_dst == label_id] = 0  # clear this label's own old footprint on the target slice
    new_dst[paint_mask] = label_id
    return new_labels, n_excluded_px


def create_labels(
    volume: np.ndarray,
    sigma_xy: float = 1.0,
    sigma_z: float = 0.5,
    min_volume: int = 7500,
    min_hole_size: int = 0,
    final_min_fraction: float = 0.618,
) -> np.ndarray:
    """
    Create 3D labels from brain_only volume using true 3D connected components.

    Dispatches to the fastest available backend:
      CUDA (CuPy)  →  Apple MPS (threaded CPU)  →  CPU threaded

    Parameters
    ----------
    volume        : (Z, Y, X) ndarray — brain_only output
    sigma_xy      : Gaussian smoothing sigma in XY (voxels)
    sigma_z       : Gaussian smoothing sigma in Z (voxels)
    min_volume    : minimum 3D blob size in voxels
    min_hole_size : per-slice hole-fill floor, in voxels. A background
                    region fully enclosed by signal in a 2D slice
                    survives as real background only if its area is
                    >= this value; anything smaller is filled in as
                    noise instead of being left as a stray gap. Named
                    to match min_volume: both name the size a region
                    must clear to be trusted as real, not the size at
                    which it gets discarded/filled. <=0 (default) fills
                    every enclosed hole regardless of size -- the
                    original, unconditional behaviour, kept as the
                    default so existing callers are unaffected unless
                    they opt in.
    final_min_fraction : the actual 3D-blob deletion cutoff is
                    final_min_fraction * min_volume, not min_volume
                    itself -- same golden-ratio safety-net idea as
                    _cellpose_seg.py's final_min_size_cleanup(), applied
                    here too since this route has no merge/reattach
                    stage to leave a gray-zone object standing for later
                    reconsideration the way Cellpose-SAM's GMM/safe-
                    merge/large-contact chain does. Default 0.618 (the
                    golden ratio, 1/phi) matches that route's default;
                    pass 1.0 to recover the exact historical behaviour
                    (cutoff == min_volume, no relaxation).

    Returns
    -------
    (Z, Y, X) int32 ndarray — 0=background, 1..N=objects (1=largest)
    """
    backend_label = {
        "cuda": "CUDA (CuPy)",
        "mps":  f"Apple MPS → threaded CPU  (threads={_N_THREADS})",
        "cpu":  f"CPU threaded  (threads={_N_THREADS})",
    }[_BACKEND]
    print(f"   Backend: {backend_label}")

    if _BACKEND == "cuda":
        try:
            return _create_labels_cuda(
                volume, sigma_xy, sigma_z, min_volume, min_hole_size, final_min_fraction
            )
        except Exception as exc:
            # e.g. out-of-memory — fall back gracefully
            print(f"   CUDA error ({exc}), falling back to CPU.")
            _free_gpu_cache()

    return _create_labels_threaded(
        volume, sigma_xy, sigma_z, min_volume, min_hole_size, final_min_fraction
    )
