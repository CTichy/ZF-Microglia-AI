"""
_gt_toolkit.py -- single-folder auto-discovery for the GT Toolkit Tuning
Tool: given one fish's original source file (or its already-created
output folder), locates every artifact this project's established
naming convention already produces there, and decides which of the
plugin's individual GT-sweep/calibration steps can run without the
user hunting down and browsing to each file by hand.

Established convention (_output_dir() in _widget.py, _gt_annotation.py,
_gt_package.py, and the real 3-fish production pipeline's own file
layout): every fish's derived files live in ONE folder named after the
original source file, every filename inside prefixed with that same
stem --

  <parent>/<stem>.ims                          (raw source, sits BESIDE
                                                  the folder, not in it)
  <parent>/<stem>/<stem>_original.tif           (GT Annotation only --
                                                  a normal Tab 1 run
                                                  never writes this)
  <parent>/<stem>/<stem>_brain_mask.tif         (raw un-eroded MONAI
                                                  mask -- or hand-
                                                  corrected, if this
                                                  fish went through GT
                                                  Annotation; the two
                                                  are indistinguishable
                                                  by filename alone, so
                                                  _original.tif's
                                                  presence -- which
                                                  ONLY GT Annotation
                                                  writes -- is used as
                                                  the signal that this
                                                  one is hand-corrected)
  <parent>/<stem>/<stem>_brain_only_ExtRm.tif   (Background mode 1)
  <parent>/<stem>/<stem>_brain_only_NoBG.tif    (Background mode 2 --
                                                  Pixel Classifier route)
  <parent>/<stem>/<stem>_brain_only_RndFill.tif (Background mode 3)
  <parent>/<stem>/<stem>_cp.tif                 (raw Cellpose-SAM output,
                                                  pre-merge)
  <parent>/<stem>/<stem>_cp_krendl.tif          (post-Krendl safe-merge +
                                                  large-contact merge)
  <parent>/<stem>/<stem>_cp_krendl_ac.tif       (post auto-correct, if
                                                  that stage ran)
  <parent>/<stem>/<stem>_cp_krendl_ac_snd.tif   (post sanding, if that
                                                  stage ran too)
  <parent>/<stem>/<stem>_GROUND_TRUTH.tif       (hand-corrected cell
                                                  instance labels)
  <parent>/<stem>/<stem>_statistics.csv

Cellpose-SAM naming is cumulative and self-documenting: each stage's
filename is the previous stage's filename with one more suffix
appended, so reading it left to right tells you exactly which
processing steps were actually applied -- `_cp_krendl_ac_snd.tif`
went through Krendl, then auto-correct, then sanding; `_cp_krendl.tif`
only went through Krendl. Only `_cp.tif` (raw) and `_cp_krendl.tif`
(post-Krendl) are always saved by a normal segmentation run --
`_cp_krendl_ac.tif`/`_cp_krendl_ac_snd.tif` only exist if
auto-correct/sanding were left enabled for that run.
"""
from pathlib import Path

SUFFIXES = {
    "original":            "_original.tif",
    "brain_mask":          "_brain_mask.tif",
    "ext_rm":              "_brain_only_ExtRm.tif",
    "no_bg":               "_brain_only_NoBG.tif",
    "rnd_fill":            "_brain_only_RndFill.tif",
    "cp":                  "_cp.tif",
    "cp_krendl":           "_cp_krendl.tif",
    "cp_krendl_ac":        "_cp_krendl_ac.tif",
    "cp_krendl_ac_snd":    "_cp_krendl_ac_snd.tif",
    "ground_truth":        "_GROUND_TRUTH.tif",
    "statistics_csv":      "_statistics.csv",
}

# Priority order for "most-advanced Cellpose-SAM correction stage this
# fish actually has" -- sanded beats auto-corrected beats Krendl-only.
# Deliberately does NOT include "cp" (the raw pre-merge stage): that one
# is always kept as a separate reference file, never used as the GT
# package's own correction starting point.
_CORRECTED_MASKS_PRIORITY = ("cp_krendl_ac_snd", "cp_krendl_ac", "cp_krendl")


def best_corrected_masks(found):
    """Returns (path, key) for the most-advanced Cellpose-SAM correction
    stage present in a discover_fish_files() `found` dict, or (None, None)
    if this fish has none of them (e.g. it only went through the Pixel
    Classifier route). See _CORRECTED_MASKS_PRIORITY for the order."""
    for key in _CORRECTED_MASKS_PRIORITY:
        if found.get(key) is not None:
            return found[key], key
    return None, None


def resolve_fish_folder(source_path) -> Path:
    """A fish's canonical folder is <original_parent>/<original_stem> --
    the same convention _output_dir() already enforces for every file
    this plugin writes. Accepts that folder directly, any file already
    living inside it, or the original raw source file sitting beside it
    (the folder may not exist yet in that case)."""
    p = Path(source_path)
    if p.is_dir():
        return p
    sibling = p.parent / p.stem
    if sibling.is_dir():
        return sibling
    return p.parent


def discover_fish_files(source_path):
    """Returns (folder, stem, found). found maps each key in SUFFIXES to
    a Path if <stem><suffix> exists exactly, else to the single match of
    a same-suffix glob if exactly one exists (tolerates the folder name
    and a file's own stem drifting apart slightly), else None."""
    folder = resolve_fish_folder(source_path)
    stem = folder.name
    found = {}
    for key, suffix in SUFFIXES.items():
        exact = folder / f"{stem}{suffix}"
        if exact.is_file():
            found[key] = exact
            continue
        matches = sorted(folder.glob(f"*{suffix}")) if folder.is_dir() else []
        found[key] = matches[0] if len(matches) == 1 else None
    return folder, stem, found


def format_discovery_report(folder, stem, found) -> str:
    lines = [f"Fish folder: {folder}", f"Stem: {stem}", ""]
    for key, suffix in SUFFIXES.items():
        mark = "found" if found[key] else "missing"
        lines.append(f"  [{mark:>7}] {key:<20} ({stem}{suffix})")
    return "\n".join(lines)


def step_preconditions(found):
    """Which of the 5 folded-in GT-sweep/calibration steps can run
    against this fish's discovered files, and why not for the ones that
    can't.

    Returns {step_key: (ok: bool, reason: str)}. reason is always
    populated (not just on failure) so the caller can show it either
    way -- "why this ran" is as useful to see as "why it didn't".
    """
    have_gt = found["ground_truth"] is not None
    have_original = found["original"] is not None
    have_ext_rm = found["ext_rm"] is not None
    have_brain_mask = found["brain_mask"] is not None
    have_raw_or_ext = have_original or have_ext_rm

    return {
        "monai": (
            have_original and have_brain_mask,
            "needs <stem>_original.tif + <stem>_brain_mask.tif -- both only "
            "exist if this fish went through GT Annotation (Tab 4), which is "
            "also what makes brain_mask.tif hand-corrected rather than a raw "
            "MONAI prediction."
        ),
        "bg": (
            have_raw_or_ext and have_brain_mask and have_gt,
            "needs a raw or _ExtRm image + <stem>_brain_mask.tif + "
            "<stem>_GROUND_TRUTH.tif."
        ),
        "sigma": (
            have_raw_or_ext and have_brain_mask and have_gt and found["no_bg"] is not None,
            "needs the same as BG Threshold/Erosion, plus <stem>_brain_only_"
            "NoBG.tif as evidence this fish went through the Pixel Classifier "
            "route (Sigma XY/Z only affects that route's pre-threshold "
            "smoothing)."
        ),
        "cellprob": (
            have_ext_rm and have_gt,
            "needs <stem>_brain_only_ExtRm.tif + <stem>_GROUND_TRUTH.tif "
            "(the Cellpose-SAM route)."
        ),
        "branch_radius": (
            have_gt,
            "needs <stem>_GROUND_TRUTH.tif (measures branch thickness "
            "directly from the labeled cells, no image needed)."
        ),
    }
