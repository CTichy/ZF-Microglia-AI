"""
_statistics.py — Per-label morphological statistics + natural-language descriptions.

Statistics computed per label (46 columns always present; up to 51 when an
intensity image and/or brain-region boundary lines are supplied)
-----------------------------------------------------------------------
  label                   : label ID
  volume_vox              : voxel count
  volume_um3              : physical volume (µm³)
  centroid_z/y/x_vox      : centroid in voxels
  centroid_z/y/x_um       : centroid in µm
  bbox_z0/y0/x0_vox       : bounding box start corner (voxels, inclusive)
  bbox_z1/y1/x1_vox       : bounding box end corner (voxels, exclusive)
  bbox_dz/dy/dx_um        : bounding box size (µm)
  eq_diam_um              : equivalent sphere diameter (µm)
  axis1/2/3_um            : principal axis lengths (µm), axis1 = longest
  elongation              : axis1 / axis3  (1 = sphere, > 1 = elongated)
  principal_axis_dir      : dominant axis of elongation ('Z', 'Y', or 'X')
  solidity                : volume / convex_hull_volume  (1 = convex)
  extent                  : volume / bounding_box_volume
  surface_area_um2        : mesh surface area via marching cubes (µm²)
  sphericity              : π^(1/3) × (6V)^(2/3) / A  (1 = perfect sphere)
  surface_to_volume_ratio : surface_area_um2 / volume_um3
  n_branches              : skeleton branch count
  n_endpoints             : free-end branch tip count
  mean_branch_len_um      : mean branch path length (µm)
  max_branch_len_um       : longest single branch (µm)
  branch_tortuosity       : mean(path_len / euclidean_len) per branch (1 = straight)
  branch_density          : n_branches / volume_um3  (×10⁶ for readability)
  endpoint_density        : n_endpoints / volume_um3  (×10⁶)
  process_complexity      : n_endpoints × mean_branch_len_um / volume_um3
  morphotype              : rule-based class (Ramified/Amoeboid/Rod-shaped/Intermediate)
  nearest_neighbor_label  : label ID of the 1st nearest neighbour
  nearest_neighbor_dist_um: distance to nearest centroid (µm)
  nearest_neighbor_2_label: label ID of the 2nd nearest neighbour
  nearest_neighbor_2_dist_um: distance to 2nd nearest centroid (µm)
  nearest_neighbor_ratio  : observed NND / expected NND (3-D Clark-Evans index)
  local_density_100um     : cells within 100 µm radius
  depth_normalized        : centroid_z_um / max_centroid_z_um  (0 = top, 1 = bottom)
  mean_intensity          : mean voxel intensity inside mask  [optional]
  integrated_intensity    : sum of voxel intensities  [optional]
  intensity_cv            : coefficient of variation of intensity (std/mean)  [optional]
  brain_region            : name of brain region assigned by boundary lines  [optional]
  region_boundary_dist_um : distance to nearest region boundary line (µm)  [optional]
  description             : natural-language description

GPU acceleration
----------------
  Phase 1 — batch regionprops: cuCIM (GPU) → skimage regionprops_table (CPU)
  Phase 2 — per-label marching cubes + skeleton: ThreadPoolExecutor (parallel CPU)
  Phase 3 — description generation: online API or rule-based (CPU)

Description backends
--------------------
  rule   – built-in rule-based templates (always available, fully offline)
  ollama – local Ollama LLM (free, no API key; install from https://ollama.com)
  openai – OpenAI API (paid; requires API key from https://platform.openai.com)
  claude – Anthropic Claude API (paid; requires API key from https://console.anthropic.com)
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

_N_THREADS = max(1, (os.cpu_count() or 4) // 2)


# ── GPU backend detection ─────────────────────────────────────────────────────

def _detect_stats_backend():
    """Return (cupy, cucim_measure) or (None, None)."""
    import io, sys
    # cupy / cucim print a spurious cuVS/pylibraft warning to stdout on import;
    # suppress it — it doesn't affect regionprops functionality.
    _saved, sys.stdout = sys.stdout, io.StringIO()
    try:
        import cupy as cp
        import cucim.skimage.measure as cucim_measure
        # Quick smoke-test
        t = cp.zeros((4, 4, 4), dtype=cp.int32)
        t[1:3, 1:3, 1:3] = 1
        cucim_measure.regionprops_table(t, properties=["label", "area"])
        return cp, cucim_measure
    except Exception:
        return None, None
    finally:
        sys.stdout = _saved


_CP_STATS, _CUCIM = _detect_stats_backend()
_STATS_BACKEND = "cuda" if _CP_STATS is not None else "cpu"

# Properties requested from regionprops_table (both skimage and cuCIM)
_RPROPS = [
    "label", "area", "centroid", "bbox",
    "inertia_tensor", "inertia_tensor_eigvals",
    "axis_major_length", "axis_minor_length",
    "solidity", "extent",
]


# ── Phase 1: batch regionprops (GPU or CPU) ──────────────────────────────────

def _batch_regionprops(labels: np.ndarray) -> dict:
    """
    Compute regionprops for ALL labels in one vectorised pass.
    Returns a dict of flat numpy arrays keyed by property name.
    Uses cuCIM (GPU) when available, falls back to skimage (CPU).
    """
    from skimage.measure import regionprops_table as _cpu_rpt

    if _STATS_BACKEND == "cuda":
        try:
            labels_gpu = _CP_STATS.asarray(labels)
            table = _CUCIM.regionprops_table(labels_gpu, properties=_RPROPS)
            result = {}
            for k, v in table.items():
                result[k] = v.get() if hasattr(v, "get") else np.asarray(v)
            del labels_gpu
            _CP_STATS.get_default_memory_pool().free_all_blocks()
            print(f"   regionprops: cuCIM GPU ({len(result['label'])} labels)")
            return result
        except Exception as exc:
            print(f"   regionprops: cuCIM failed ({exc}), using CPU")

    table = _cpu_rpt(labels, properties=_RPROPS)
    print(f"   regionprops: skimage CPU ({len(table['label'])} labels)")
    return {k: np.asarray(v) for k, v in table.items()}


# ── Phase 2a: per-label slow ops (surface area + skeleton) ───────────────────

def _surface_area(binary: np.ndarray, scale_zyx: tuple) -> float:
    """Mesh surface area in µm² via marching cubes. Returns 0.0 on failure."""
    try:
        from skimage.measure import marching_cubes, mesh_surface_area
        padded = np.pad(binary.astype(np.uint8), 1)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=scale_zyx)
        return float(mesh_surface_area(verts, faces))
    except Exception:
        return 0.0


def _skeleton_stats(binary: np.ndarray, scale_zyx: tuple) -> tuple:
    """
    Return (n_branches, n_endpoints, mean_branch_len_um, max_branch_len_um, tortuosity).
    Uses skan if available, otherwise (0, 0, 0.0, 0.0, 1.0).
    Tortuosity = mean(path_length / euclidean_distance) per branch; 1.0 = perfectly straight.
    """
    try:
        from skimage.morphology import skeletonize
        import skan
        skeleton = skeletonize(binary)
        if not skeleton.any():
            return 0, 0, 0.0, 0.0, 1.0
        sk = skan.Skeleton(
            skeleton, spacing=scale_zyx,
            source_image=binary.astype(np.float32),
        )
        bd = skan.summarize(sk, separator="-")
        if len(bd) == 0:
            return 0, 0, 0.0, 0.0, 1.0
        n_branches  = len(bd)
        n_endpoints = int((bd["branch-type"] == 1).sum())
        euc         = bd["euclidean-distance"].values
        mean_len    = float(euc.mean())
        max_len     = float(euc.max())
        # Tortuosity: path length / straight-line distance
        try:
            path  = bd["branch-distance"].values
            valid = euc > 0
            tort  = float((path[valid] / euc[valid]).mean()) if valid.any() else 1.0
        except (KeyError, Exception):
            tort = 1.0
        return n_branches, n_endpoints, mean_len, max_len, tort
    except Exception:
        return 0, 0, 0.0, 0.0, 1.0


def _slow_stats_worker(args: tuple) -> tuple:
    """Worker: returns (lbl, sa_um2, n_br, n_ep, br_len, max_br_len, tortuosity)."""
    lbl, labels, bbox, scale_zyx = args
    z0, y0, x0, z1, y1, x1 = bbox
    binary = (labels[z0:z1, y0:y1, x0:x1] == lbl)
    sa = _surface_area(binary, scale_zyx)
    n_br, n_ep, br_len, max_br_len, tort = _skeleton_stats(binary, scale_zyx)
    return lbl, sa, n_br, n_ep, br_len, max_br_len, tort


# ── Phase 2b: intensity stats (optional) ─────────────────────────────────────

def _intensity_stats_worker(args: tuple) -> tuple:
    """Worker: returns (lbl, mean_intensity, integrated_intensity, intensity_cv)."""
    lbl, labels, image, bbox = args
    z0, y0, x0, z1, y1, x1 = bbox
    mask = (labels[z0:z1, y0:y1, x0:x1] == lbl)
    vals = image[z0:z1, y0:y1, x0:x1][mask].astype(np.float64)
    if len(vals) == 0:
        return lbl, 0.0, 0.0, 0.0
    mean_i    = float(vals.mean())
    integrated = float(vals.sum())
    cv         = float(vals.std() / mean_i) if mean_i > 0 else 0.0
    return lbl, mean_i, integrated, cv


# ── Post-assembly: spatial statistics ────────────────────────────────────────

def _spatial_stats(centroids_zyx_um: np.ndarray, label_ids: np.ndarray) -> dict:
    """
    Compute NND, Clark-Evans 3-D ratio, local density, and normalised depth
    for all centroids at once.

    centroids_zyx_um : (N, 3) array of (z, y, x) in µm
    label_ids        : (N,) array of integer label IDs (same order)
    """
    from scipy.spatial import cKDTree

    N   = len(centroids_zyx_um)
    pts = np.asarray(centroids_zyx_um, dtype=np.float64)

    if N < 2:
        return {
            "nearest_neighbor_label":     label_ids.copy(),
            "nearest_neighbor_dist_um":   np.zeros(N),
            "nearest_neighbor_2_label":   label_ids.copy(),
            "nearest_neighbor_2_dist_um": np.zeros(N),
            "nearest_neighbor_ratio":     np.ones(N),
            "local_density_100um":        np.zeros(N, dtype=int),
            "depth_normalized":           np.zeros(N),
        }

    tree = cKDTree(pts)

    # k=3: self + 1st NN + 2nd NN (use min(N,3) to handle N==2 safely)
    k       = min(N, 3)
    dists, idxs = tree.query(pts, k=k)
    nnd      = dists[:, 1]
    nn_lbls  = label_ids[idxs[:, 1]]
    # 2nd nearest neighbour (falls back to 1st if only 2 cells total)
    nn2_col  = 2 if k == 3 else 1
    nnd2     = dists[:, nn2_col]
    nn2_lbls = label_ids[idxs[:, nn2_col]]

    # Clark-Evans 3-D index: E[NND] = Γ(4/3) · (3 / (4π·ρ))^(1/3)
    bbox_vol  = float(np.prod(pts.max(axis=0) - pts.min(axis=0) + 1e-10))
    density   = N / bbox_vol
    exp_nnd   = math.gamma(4 / 3) * (3.0 / (4.0 * math.pi * density)) ** (1.0 / 3.0)
    nnd_ratio = nnd / exp_nnd if exp_nnd > 0 else np.ones(N)

    # Local density: cells within 100 µm sphere (excluding self)
    try:
        local_density = np.array(
            tree.query_ball_point(pts, r=100.0, return_length=True), dtype=int
        ) - 1
    except TypeError:                                       # scipy < 1.8
        local_density = np.array(
            [len(idx) - 1 for idx in tree.query_ball_point(pts, r=100.0)], dtype=int
        )

    # Normalised depth (Z axis)
    z_max      = float(pts[:, 0].max())
    depth_norm = np.clip(pts[:, 0] / z_max, 0.0, 1.0) if z_max > 0 else np.zeros(N)

    return {
        "nearest_neighbor_label":     nn_lbls,
        "nearest_neighbor_dist_um":   nnd,
        "nearest_neighbor_2_label":   nn2_lbls,
        "nearest_neighbor_2_dist_um": nnd2,
        "nearest_neighbor_ratio":     nnd_ratio,
        "local_density_100um":        local_density,
        "depth_normalized":           depth_norm,
    }


# ── Post-assembly: brain region assignment ───────────────────────────────────

def _polyline_side_and_dist(cy: float, cx: float, pts: np.ndarray):
    """
    Given a point (cy, cx) and a polyline defined by pts (M, 2) in [Y, X] order,
    return (side, dist) where:
      side  — +1 if the point is left of the nearest segment, -1 if right
      dist  — minimum distance from the point to any segment of the polyline

    'Right of segment' means the cross-product at the nearest segment is negative
    (when the segment runs left→right, i.e. anterior→posterior).
    """
    best_dist = float("inf")
    best_cross_sign = 1  # default: left side (more anterior)

    for i in range(len(pts) - 1):
        ay, ax = float(pts[i][0]),   float(pts[i][1])
        by, bx = float(pts[i+1][0]), float(pts[i+1][1])
        dy_seg, dx_seg = by - ay, bx - ax
        denom = dx_seg * dx_seg + dy_seg * dy_seg
        if denom < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0,
                ((cx - ax) * dx_seg + (cy - ay) * dy_seg) / denom))
        py = ay + t * dy_seg
        px = ax + t * dx_seg
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        if d < best_dist:
            best_dist = d
            # Cross product at the closest segment
            cross = dx_seg * (cy - ay) - dy_seg * (cx - ax)
            best_cross_sign = -1 if cross < 0 else 1

    return best_cross_sign, best_dist


def _assign_brain_regions(
    centroids_yx_um: np.ndarray,
    region_lines_um: list,
    region_names: list,
) -> tuple[list, list]:
    """
    Classify each centroid into a named brain region using dividing curves.

    Expected fish / image orientation
    ----------------------------------
    The fish lies along the X axis: head at X=0 (anterior), tail at X=max
    (posterior). Y runs top to bottom (0 → max). Each boundary curve should
    be drawn top→bottom (increasing Y), so that the anterior region (smaller X)
    is to the LEFT of the path and the posterior region (larger X) is to the RIGHT.

    Each boundary may be a 2-point straight line OR a multi-point polyline
    (both represented as (M, 2) arrays of [Y, X] in µm). Boundaries are
    sorted by mean X of their vertices (ascending = most anterior first).

    For each centroid, region_index = number of boundaries whose nearest
    segment has the centroid on its right side (cross-product < 0 = more
    posterior / larger X).

    Parameters
    ----------
    centroids_yx_um  : (N, 2) array of (y_um, x_um)
    region_lines_um  : list of (M, 2) arrays, M >= 2, in µm
    region_names     : list of strings, length = len(region_lines_um) + 1

    Returns
    -------
    (regions, boundary_dists) — both length-N lists
    """
    if not region_lines_um:
        N = len(centroids_yx_um)
        return [""] * N, [0.0] * N

    # Sort boundaries by mean X of all vertices (ascending = anterior first)
    lines_sorted = sorted(
        region_lines_um,
        key=lambda ln: float(np.mean(ln[:, 1])),
    )

    n_regions = len(lines_sorted) + 1
    names = list(region_names)[:n_regions]
    while len(names) < n_regions:
        names.append(f"region_{len(names)}")

    result_regions = []
    result_dists   = []

    for cy, cx in centroids_yx_um:
        idx      = 0
        min_dist = float("inf")
        for pts in lines_sorted:
            side, d = _polyline_side_and_dist(cy, cx, pts)
            if side < 0:   # point is right of (more posterior than) this boundary
                idx += 1
            min_dist = min(min_dist, d)
        result_regions.append(names[min(idx, n_regions - 1)])
        result_dists.append(float(min_dist))

    return result_regions, result_dists


# ── Morphotype classification ─────────────────────────────────────────────────

def _classify_morphotype(spher: float, solid: float, elong: float,
                          n_br: int, sav: float) -> str:
    """
    Rule-based morphotype assignment for microglia.

    Classes (in priority order):
      Rod-shaped          — highly elongated, few branches
      Amoeboid            — round, convex, few branches  (activated)
      Ramified            — many branches, high SA/V, low sphericity  (resting)
      Intermediate-ramified — partial ramification
      Intermediate        — catch-all
    """
    if elong > 3.5 and n_br <= 3:
        return "Rod-shaped"
    if spher > 0.70 and solid > 0.80 and n_br <= 2:
        return "Amoeboid"
    if n_br >= 6 and sav > 2.0 and spher < 0.55:
        return "Ramified"
    if n_br >= 4 and spher < 0.65:
        return "Intermediate-ramified"
    return "Intermediate"


# ── Description backends ──────────────────────────────────────────────────────

def _rule_based_description(row: dict) -> str:
    lbl    = row["label"]
    vol    = row["volume_um3"]
    elong  = row["elongation"]
    adir   = row["principal_axis_dir"]
    spher  = row["sphericity"]
    solid  = row["solidity"]
    n_br   = row["n_branches"]
    n_ep   = row["n_endpoints"]
    br_len = row["mean_branch_len_um"]
    cz, cy, cx = row["centroid_z_um"], row["centroid_y_um"], row["centroid_x_um"]
    morph  = row.get("morphotype", "")

    if spher > 0.85:
        shape = "spherical"
    elif spher > 0.70:
        shape = f"rounded, elongated along {adir}-axis ({elong:.1f}:1)" if elong > 1.8 else "rounded"
    elif elong > 2.5:
        shape = f"elongated along {adir}-axis ({elong:.1f}:1)"
    elif elong > 1.5:
        shape = f"moderately elongated along {adir}-axis"
    else:
        shape = "compact, irregular"

    surface = (
        "smooth surface" if solid > 0.90 else
        "slightly lobulated surface" if solid > 0.75 else
        "lobulated/irregular surface"
    )

    if n_br == 0:
        branch_str = "no branching detected"
    elif n_br <= 2:
        branch_str = f"{n_ep} protrusion(s)" + (f" (mean {br_len:.1f} µm)" if br_len > 0 else "")
    else:
        branch_str = f"{n_br} branches, {n_ep} endpoints" + (f" (mean {br_len:.1f} µm)" if br_len > 0 else "")

    morph_str = f" Morphotype: {morph}." if morph else ""
    return (
        f"Label {lbl}: {shape.capitalize()}, volume {vol:,.0f} µm³, "
        f"centroid Z={cz:.1f} Y={cy:.1f} X={cx:.1f} µm. "
        f"{surface.capitalize()}, sphericity {spher:.2f}, solidity {solid:.2f}. "
        f"{branch_str.capitalize()}.{morph_str}"
    )


_STATS_PROMPT = """\
You are analyzing a single 3D cell from a zebrafish brain confocal microscopy image.
Given the morphological statistics below, write ONE concise sentence (max 40 words) \
describing the cell's shape, size, and notable features.

Label: {label}
Volume: {volume_um3:.0f} µm³
Elongation (longest/shortest axis): {elongation:.2f}
Dominant axis: {principal_axis_dir}
Sphericity (1=sphere): {sphericity:.3f}
Solidity (1=convex): {solidity:.3f}
Branches: {n_branches}  Endpoints: {n_endpoints}  Mean branch length: {mean_branch_len_um:.1f} µm
Morphotype: {morphotype}
Centroid: Z={centroid_z_um:.1f} Y={centroid_y_um:.1f} X={centroid_x_um:.1f} µm

Description:"""


def _ollama_description(row: dict, endpoint: str, model: str) -> str:
    prompt  = _STATS_PROMPT.format(**row)
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def _openai_description(row: dict, api_key: str, model: str, api_url: str) -> str:
    url     = (api_url or "https://api.openai.com").rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model or "gpt-4o-mini",
        "messages": [{"role": "user", "content": _STATS_PROMPT.format(**row)}],
        "max_tokens": 80,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"[OpenAI error: {exc}]"


def _claude_description(row: dict, api_key: str, model: str) -> str:
    payload = json.dumps({
        "model": model or "claude-haiku-4-5-20251001",
        "max_tokens": 80,
        "messages": [{"role": "user", "content": _STATS_PROMPT.format(**row)}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()
    except Exception as exc:
        return f"[Claude error: {exc}]"


def _make_desc_fn(backend_config: dict | None):
    if not backend_config or backend_config.get("backend", "rule") == "rule":
        return _rule_based_description
    b = backend_config["backend"]
    if b == "ollama":
        ep = backend_config.get("ollama_endpoint", "http://localhost:11434")
        mo = backend_config.get("ollama_model", "llama3")
        return lambda row: _ollama_description(row, ep, mo)
    if b == "openai":
        return lambda row: _openai_description(
            row,
            backend_config.get("api_key", ""),
            backend_config.get("api_model", "gpt-4o-mini"),
            backend_config.get("api_url", ""),
        )
    if b == "claude":
        return lambda row: _claude_description(
            row,
            backend_config.get("api_key", ""),
            backend_config.get("api_model", "claude-haiku-4-5-20251001"),
        )
    return _rule_based_description


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_stats(
    labels: np.ndarray,
    scale_zyx: tuple,
    image: np.ndarray | None = None,
    region_lines: list | None = None,
    region_names: list | None = None,
    backend_config: dict | None = None,
) -> "Any":
    """
    Compute per-label morphological statistics and generate descriptions.

    Parameters
    ----------
    labels        : (Z, Y, X) int32 ndarray, 0 = background
    scale_zyx     : (z_um, y_um, x_um) — physical voxel size in µm
    image         : optional (Z, Y, X) ndarray — source image for intensity stats
    region_lines  : optional list of (2,2) arrays [[y0,x0],[y1,x1]] in µm —
                    boundary lines for brain region assignment
    region_names  : optional list of strings, len = len(region_lines)+1
    backend_config: description backend config dict (see module docstring)

    Returns
    -------
    pd.DataFrame — one row per label
    """
    import pandas as pd

    sz, sy, sx = float(scale_zyx[0]), float(scale_zyx[1]), float(scale_zyx[2])
    vox_vol    = sz * sy * sx
    desc_fn    = _make_desc_fn(backend_config)
    has_image  = image is not None
    has_regions = region_lines is not None and len(region_lines) > 0

    print(f"\n{'='*60}")
    print(f"GENERATE STATISTICS  (GPU={_STATS_BACKEND}  threads={_N_THREADS})")
    print(f"Scale: Z={sz:.4f}  Y={sy:.4f}  X={sx:.4f} µm/vox")
    if has_image:   print(f"   Intensity stats: enabled")
    if has_regions: print(f"   Brain regions: {len(region_lines)} boundaries")
    print(f"{'='*60}")

    # ── Phase 1: batch regionprops ────────────────────────────────────────
    table = _batch_regionprops(labels)
    N     = len(table["label"])

    # Batch inertia tensor eigenvectors
    IT = np.stack([
        [table["inertia_tensor-0-0"], table["inertia_tensor-0-1"], table["inertia_tensor-0-2"]],
        [table["inertia_tensor-1-0"], table["inertia_tensor-1-1"], table["inertia_tensor-1-2"]],
        [table["inertia_tensor-2-0"], table["inertia_tensor-2-1"], table["inertia_tensor-2-2"]],
    ], axis=0).T                                  # shape (N, 3, 3)
    _, eigvecs    = np.linalg.eigh(IT)
    longest_vecs  = eigvecs[:, :, 0]
    axis_dirs     = [
        ["Z", "Y", "X"][int(np.argmax(np.abs(v)))]
        for v in longest_vecs
    ]

    axis_scale_map = {"Z": sz, "Y": sy, "X": sx}
    axis1_scales   = np.array([axis_scale_map[d] for d in axis_dirs])
    scale_mean     = (sz + sy + sx) / 3.0

    a1_um      = table["axis_major_length"] * axis1_scales
    a3_um      = table["axis_minor_length"] * scale_mean
    a2_um      = (a1_um + a3_um) / 2.0
    elongation = a1_um / np.maximum(a3_um, 1e-10)

    dz_um   = (table["bbox-3"] - table["bbox-0"]) * sz
    dy_um   = (table["bbox-4"] - table["bbox-1"]) * sy
    dx_um   = (table["bbox-5"] - table["bbox-2"]) * sx
    vol_um3 = table["area"] * vox_vol
    eq_diam = (6.0 * vol_um3 / math.pi) ** (1.0 / 3.0)

    # ── Phase 2a: per-label marching cubes + skeleton ─────────────────────
    print(f"   Surface area + skeleton ({_N_THREADS} threads)...")
    bbox_list = list(zip(
        table["bbox-0"].astype(int), table["bbox-1"].astype(int),
        table["bbox-2"].astype(int), table["bbox-3"].astype(int),
        table["bbox-4"].astype(int), table["bbox-5"].astype(int),
    ))
    worker_args = [
        (int(lbl), labels, bbox, (sz, sy, sx))
        for lbl, bbox in zip(table["label"], bbox_list)
    ]
    with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
        slow = list(pool.map(_slow_stats_worker, worker_args))

    slow_by_lbl = {
        lbl: (sa, n_br, n_ep, br_len, max_br, tort)
        for lbl, sa, n_br, n_ep, br_len, max_br, tort in slow
    }

    # ── Phase 2b: intensity stats (optional) ──────────────────────────────
    if has_image:
        print(f"   Intensity stats ({_N_THREADS} threads)...")
        int_args = [
            (int(lbl), labels, image, bbox)
            for lbl, bbox in zip(table["label"], bbox_list)
        ]
        with ThreadPoolExecutor(max_workers=_N_THREADS) as pool:
            int_results = list(pool.map(_intensity_stats_worker, int_args))
        int_by_lbl = {
            lbl: (mean_i, integ, cv)
            for lbl, mean_i, integ, cv in int_results
        }

    # ── Phase 3: assemble rows + derived metrics + descriptions ───────────
    print(f"   Assembling rows and generating descriptions...")
    rows: list[dict] = []
    for i in range(N):
        lbl = int(table["label"][i])
        sa, n_br, n_ep, br_len, max_br, tort = slow_by_lbl[lbl]

        vol   = float(vol_um3[i])
        spher = 0.0
        if sa > 0:
            spher = min(
                float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol) ** (2.0 / 3.0)) / sa),
                1.0,
            )
        try:
            solid = float(table["solidity"][i])
        except Exception:
            solid = 1.0

        elong_val = float(elongation[i])
        sav       = round(sa / vol, 4) if vol > 0 else 0.0
        br_dens   = round(n_br  / vol * 1e6, 4) if vol > 0 else 0.0
        ep_dens   = round(n_ep  / vol * 1e6, 4) if vol > 0 else 0.0
        proc_cmpl = round(n_ep * br_len / vol, 6) if vol > 0 else 0.0
        morph     = _classify_morphotype(spher, solid, elong_val, n_br, sav)

        row = {
            "label":                lbl,
            "volume_vox":           int(table["area"][i]),
            "volume_um3":           round(vol, 2),
            "centroid_z_vox":       round(float(table["centroid-0"][i]), 2),
            "centroid_y_vox":       round(float(table["centroid-1"][i]), 2),
            "centroid_x_vox":       round(float(table["centroid-2"][i]), 2),
            "centroid_z_um":        round(float(table["centroid-0"][i]) * sz, 2),
            "centroid_y_um":        round(float(table["centroid-1"][i]) * sy, 2),
            "centroid_x_um":        round(float(table["centroid-2"][i]) * sx, 2),
            "bbox_z0_vox":          int(table["bbox-0"][i]),
            "bbox_y0_vox":          int(table["bbox-1"][i]),
            "bbox_x0_vox":          int(table["bbox-2"][i]),
            "bbox_z1_vox":          int(table["bbox-3"][i]),
            "bbox_y1_vox":          int(table["bbox-4"][i]),
            "bbox_x1_vox":          int(table["bbox-5"][i]),
            "bbox_dz_um":           round(float(dz_um[i]), 2),
            "bbox_dy_um":           round(float(dy_um[i]), 2),
            "bbox_dx_um":           round(float(dx_um[i]), 2),
            "eq_diam_um":           round(float(eq_diam[i]), 2),
            "axis1_um":             round(float(a1_um[i]), 2),
            "axis2_um":             round(float(a2_um[i]), 2),
            "axis3_um":             round(float(a3_um[i]), 2),
            "elongation":           round(elong_val, 3),
            "principal_axis_dir":   axis_dirs[i],
            "solidity":             round(solid, 4),
            "extent":               round(float(table["extent"][i]), 4),
            "surface_area_um2":     round(sa, 2),
            "sphericity":           round(spher, 4),
            "surface_to_volume_ratio": sav,
            "n_branches":           n_br,
            "n_endpoints":          n_ep,
            "mean_branch_len_um":   round(br_len, 2),
            "max_branch_len_um":    round(max_br, 2),
            "branch_tortuosity":    round(tort, 4),
            "branch_density":       br_dens,
            "endpoint_density":     ep_dens,
            "process_complexity":   proc_cmpl,
            "morphotype":           morph,
        }

        if has_image:
            mean_i, integ, cv = int_by_lbl[lbl]
            row["mean_intensity"]        = round(mean_i, 2)
            row["integrated_intensity"]  = round(integ, 2)
            row["intensity_cv"]          = round(cv, 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Post-assembly: spatial statistics ─────────────────────────────────
    print(f"   Spatial statistics...")
    centroids_zyx = df[["centroid_z_um", "centroid_y_um", "centroid_x_um"]].values
    label_ids     = df["label"].values.astype(int)
    sp = _spatial_stats(centroids_zyx, label_ids)
    df["nearest_neighbor_label"]     = sp["nearest_neighbor_label"]
    df["nearest_neighbor_dist_um"]   = np.round(sp["nearest_neighbor_dist_um"],   2)
    df["nearest_neighbor_2_label"]   = sp["nearest_neighbor_2_label"]
    df["nearest_neighbor_2_dist_um"] = np.round(sp["nearest_neighbor_2_dist_um"], 2)
    df["nearest_neighbor_ratio"]     = np.round(sp["nearest_neighbor_ratio"],     4)
    df["local_density_100um"]        = sp["local_density_100um"]
    df["depth_normalized"]           = np.round(sp["depth_normalized"],           4)

    # ── Post-assembly: brain region assignment ────────────────────────────
    if has_regions:
        print(f"   Brain region assignment...")
        centroids_yx = df[["centroid_y_um", "centroid_x_um"]].values
        rnames = list(region_names) if region_names else []
        regions, bdists = _assign_brain_regions(centroids_yx, region_lines, rnames)
        df["brain_region"]             = regions
        df["region_boundary_dist_um"]  = np.round(bdists, 2)

    # ── Descriptions ──────────────────────────────────────────────────────
    df["description"] = [desc_fn(row) for row in df.to_dict("records")]

    print(f"   Done — {len(df)} labels.")
    print(f"{'='*60}\n")
    return df
