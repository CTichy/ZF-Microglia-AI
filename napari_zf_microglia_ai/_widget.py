"""
_widget.py — ZFMicrogliaAIWidget napari dock panel.
"""

import json
import os
import threading
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import napari

from qtpy.QtWidgets import (
    QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFileDialog, QSizePolicy, QButtonGroup, QRadioButton,
    QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QScrollArea, QGroupBox,
    QTextEdit,
)
from qtpy.QtCore import Qt, QTimer
from superqt import QLabeledSlider, QLabeledDoubleSlider

from ._io import load_file
from ._inference import DEFAULT_MODEL, _SKIN_SEG_DIR, run_inference
from ._background import remove_outside_brain, remove_global, fill_outside_brain_random
from ._labeling import (
    create_labels, resort_labels, split_label, join_labels, remove_debris,
    correct_label_from_intensity, copy_label_to_adjacent_slice,
)
from ._statistics import compute_stats
from ._cellpose_seg import run_full_pipeline as _run_cellpose_pipeline
from ._cellpose_seg import rerun_single_cell as _rerun_single_cell
from ._cellpose_seg import masks_from_flows as _masks_from_flows
from . import _pixel_sweep as _psw
from . import _brain_sweep as _bsw
from . import _gt_score as _gts
from . import _krendl_sweep as _ksw
from . import _sieve
from . import _gt_package as _gtp
from ._gpu_check import GPU_HAS_CUDA, GPU_VRAM_GB, GPU_MEETS_RECOMMENDED, GPU_NAME, GPU_MSG
from . import _gt_annotation as _gt
from . import _ai_tools as _ait
from . import _training_jobs as _tj
from . import _xzyz_patches as _xzp
from . import _crop_truncation as _ctr
from . import _epoch_sweep as _esw
from . import _branch_calibration as _bcal
from ._live_progress import capture_live_output
from . import _secrets
from . import _gt_toolkit as _gtk

_CONFIG_PATH = Path.home() / ".config" / "napari-zf-microglia-ai" / "config.json"

# Cellpose's flow_threshold QC filter only runs in 2D/stitch mode -- under
# do_3D=True, which every do_3D call in this plugin uses, it is provably a
# no-op (confirmed by reading cellpose/dynamics.py and by a call-count spy
# test showing zero calls regardless of value). Not exposed as a user
# control anywhere in the UI; kept only because do_3D's own call signature
# still accepts a flow_threshold argument.
_FLOW_THRESHOLD_FIXED = 0.4

# Suffix added to brain_only filename for each background mode
_BG_SUFFIX = {
    0: "",          # Off — no processing
    1: "_ExtRm",    # Exterior Removed (outside-brain BG stripped)
    2: "_NoBG",     # No Background (global removal)
    3: "_RndFill",  # Random Fill (background replaced with noise)
}

# (column_key, display_label, default_on)
# label is always included and its checkbox is disabled.
# Optional columns (intensity / brain region / description) are shown but only
# appear in the DataFrame when the respective inputs are provided.
_STATS_COLUMNS = [
    ("label",                   "label  (identifier)",                           True),
    ("volume_vox",              "volume_vox  (voxels)",                          True),
    ("volume_um3",              "volume_um3  (µm³)",                             True),
    ("centroid_vox",            "centroid_vox  (z/y/x, voxels)",                 True),
    ("centroid_um",             "centroid_um  (z/y/x, µm)",                      True),
    ("sphericity",              "sphericity",                                     True),
    ("solidity",                "solidity",                                       True),
    ("elongation",              "elongation",                                     True),
    ("axis1_um",                "axis1_um  (longest axis, µm)",                  True),
    ("axis3_um",                "axis3_um  (shortest axis, µm)",                 True),
    ("surface_area_um2",        "surface_area_um2  (µm²)",                       True),
    ("surface_to_volume_ratio", "surface_to_volume_ratio",                       True),
    ("n_branches",              "n_branches",                                    True),
    ("n_endpoints",             "n_endpoints",                                   True),
    ("mean_branch_len_um",      "mean_branch_len_um  (µm)",                      True),
    ("nn_1st",                  "nearest_neighbor 1st  (label + dist µm)",       True),
    ("nn_2nd",                  "nearest_neighbor 2nd  (label + dist µm)",       False),
    ("local_density_100um",     "local_density_100um",                           True),
    ("is_volume_outlier",       "is_volume_outlier  (smaller/larger than any GT cell confirmed so far)", True),
    # ── default OFF ──────────────────────────────────────────────────────────
    ("eq_diam_um",              "eq_diam_um  (equiv. sphere diam.)",             False),
    ("axis2_um",                "axis2_um  (middle axis, derived)",              False),
    ("principal_axis_dir",      "principal_axis_dir  (Z/Y/X orientation)",       False),
    ("bbox_vox",                "bbox_vox  (z0/y0/x0/z1/y1/x1, voxels)",        True),
    ("bbox_um",                 "bbox_um  (dz/dy/dx, µm)",                       False),
    ("extent",                  "extent  (bbox fill fraction 0–1)",              False),
    ("nearest_neighbor_ratio",  "nearest_neighbor_ratio  (Clark-Evans 3D)",      False),
    ("depth_normalized",        "depth_normalized  (Z position 0–1)",            False),
    ("max_branch_len_um",       "max_branch_len_um  (µm)",                       False),
    ("branch_tortuosity",       "branch_tortuosity",                             False),
    ("branch_density",          "branch_density  (per 10⁶ µm³)",                False),
    ("endpoint_density",        "endpoint_density  (per 10⁶ µm³)",              False),
    ("process_complexity",      "process_complexity  (custom composite)",        False),
    ("morphotype",              "morphotype  (unvalidated rule-based)",          False),
    # ── optional: only present when respective inputs are provided ────────────
    ("mean_intensity",          "mean_intensity  [intensity opt.]",              True),
    ("integrated_intensity",    "integrated_intensity  [intensity opt.]",        True),
    ("intensity_cv",            "intensity_cv  [intensity opt.]",                False),
    ("brain_region",            "brain_region  [region opt.]",                   True),
    ("region_boundary_dist_um", "region_boundary_dist_um  [region opt.]",        True),
    ("description",             "description  [AI backend]",                     False),
]

# Group keys that expand to multiple DataFrame columns when selected
_COL_GROUPS = {
    "centroid_vox": ["centroid_z_vox", "centroid_y_vox", "centroid_x_vox"],
    "centroid_um":  ["centroid_z_um",  "centroid_y_um",  "centroid_x_um"],
    "bbox_vox":     ["bbox_z0_vox", "bbox_y0_vox", "bbox_x0_vox",
                     "bbox_z1_vox", "bbox_y1_vox", "bbox_x1_vox"],
    "bbox_um":      ["bbox_dz_um", "bbox_dy_um", "bbox_dx_um"],
    "nn_1st":       ["nearest_neighbor_label",   "nearest_neighbor_dist_um"],
    "nn_2nd":       ["nearest_neighbor_2_label", "nearest_neighbor_2_dist_um"],
}


def _load_config() -> dict:
    """Load full config dict from disk, return {} on any failure."""
    try:
        if _CONFIG_PATH.exists():
            return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_config(data: dict) -> None:
    """Persist config dict to disk."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _add_reliable_spinbox(row_layout, slider, minimum, maximum, step,
                           decimals=None, slider_max_width=85):
    """
    Replace superqt's built-in numeric entry with a real QSpinBox /
    QDoubleSpinBox synced to the slider.

    superqt's QLabeled(Double)Slider has an internal numeric label
    (`._label`) whose width it silently resets on first show to a value
    that isn't reliably predictable outside the real running app — direct
    attempts to fix its width (`._label.setFixedWidth(...)`) kept getting
    overridden or landing too narrow in practice. QSpinBox/QDoubleSpinBox
    are ordinary Qt widgets with correctly self-computed sizeHints and no
    such override behaviour, so we hide the broken label and use one of
    these instead — still perfectly in sync with the slider via signals.

    Returns the spinbox (e.g. to enable/disable alongside the slider).
    """
    slider._label.setVisible(False)
    slider._slider.setMaximumWidth(slider_max_width)

    spin = QDoubleSpinBox() if decimals is not None else QSpinBox()
    if decimals is not None:
        spin.setDecimals(decimals)
    spin.setMinimum(minimum)
    spin.setMaximum(maximum)
    spin.setSingleStep(step)
    spin.setValue(slider.value())
    spin.setFixedWidth(spin.sizeHint().width() + 6)  # +6px margin for safety

    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)

    row_layout.addWidget(spin)
    return spin


def _add_gt_checkbox(layout, hint, visible=True):
    """Shared "This is verified ground truth" opt-in, one per sweep tool
    -- same one-shot pattern as Tab 3 Statistics used to: off by
    default, only ticking it lets that specific run move any never-
    rising/never-falling recommended value the sweep tracks. A sweep
    with a bad or unverified GT input can still be run and read for its
    own report (best point / score), it just can't corrupt a shared
    floor/ceiling unless this box is ticked for that run. Callers must
    un-tick it again after each run completes (see _finish_gt_sweep
    below) -- this helper only builds the widget.

    visible=False keeps the checkbox fully functional (still exists,
    still gates the apply logic exactly the same way) but hides it from
    this tool's own layout -- used by the 4 sweeps folded into the GT
    Toolkit Tuning Tool (Tab 5), so tuning can ONLY ever be triggered
    from that one place: the checkbox is only ever ticked programmatically
    by the Tuning Tool's own orchestrator right before it calls this
    tool's run handler, never by a user clicking it directly here.

    Returns the checkbox."""
    cb = QCheckBox("This is verified ground truth")
    cb.setChecked(False)
    if visible:
        layout.addWidget(cb)
    note_text = (
        f"  Off by default. Only tick when the GT labels used for {hint} "
        "have actually been manually corrected/verified as real ground "
        "truth -- an unverified/uncorrected prediction can never move a "
        "recommended value this way. Un-ticks itself automatically once "
        "the run finishes, so it must be re-ticked deliberately every "
        "time you want this specific run to count."
        if visible else
        "  GT-verified auto-apply for this sweep now only ever happens "
        "via the GT Toolkit Tuning Tool (Tab 5, top) -- running it here "
        "directly always produces a preview report only, never moves a "
        "shared recommended value."
    )
    note = QLabel(note_text)
    note.setWordWrap(True)
    note.setStyleSheet("color: #888; font-size: 10px;")
    layout.addWidget(note)
    return cb


def _add_sieve_controls(layout, axis_name, default_w2, default_s2, default_w3, default_s3):
    """Shared "Sieve" opt-in for a sweep tool's primary continuous axis --
    same idea across all four sweep tools that have one (MONAI Threshold,
    BG Threshold, Smooth sigma XY, Cellprob): when ticked, the tool's own
    min/max/step fields become stage 1 (the coarse full-range pass), then
    two more narrowing stages run automatically, each centered on the
    previous stage's winner and clipped to the axis's own min/max bounds
    -- e.g. stage1 -6..6 step 1.0 -> stage2 winner+-1.0 step 0.25 ->
    stage3 winner+-0.5 step 0.1, the exact narrowing validated by hand on
    real GT data before this control existed. Only the primary axis given
    here is narrowed across stages -- any secondary axis (e.g.
    Large-contact, Erosion, sigma Z) keeps using its own single min/max/
    step grid unchanged at every stage, same as it does today.

    Returns (sieve_checkbox, stage2_width_spin, stage2_step_spin,
             stage3_width_spin, stage3_step_spin).
    """
    cb = QCheckBox(f"Sieve: auto-narrow {axis_name} coarse -> fine (3 stages)")
    cb.setChecked(False)
    layout.addWidget(cb)

    row2 = QHBoxLayout()
    row2.addWidget(QLabel("  Stage 2 +/-:"))
    w2 = QDoubleSpinBox()
    w2.setDecimals(3)
    w2.setRange(0.001, 1000.0)
    w2.setValue(default_w2)
    row2.addWidget(w2)
    row2.addWidget(QLabel("step:"))
    s2 = QDoubleSpinBox()
    s2.setDecimals(3)
    s2.setRange(0.001, 1000.0)
    s2.setValue(default_s2)
    row2.addWidget(s2)
    layout.addLayout(row2)

    row3 = QHBoxLayout()
    row3.addWidget(QLabel("  Stage 3 +/-:"))
    w3 = QDoubleSpinBox()
    w3.setDecimals(3)
    w3.setRange(0.001, 1000.0)
    w3.setValue(default_w3)
    row3.addWidget(w3)
    row3.addWidget(QLabel("step:"))
    s3 = QDoubleSpinBox()
    s3.setDecimals(3)
    s3.setRange(0.001, 1000.0)
    s3.setValue(default_s3)
    row3.addWidget(s3)
    layout.addLayout(row3)

    note = QLabel(
        f"  Off by default. When ticked, {axis_name} min/max/step above "
        "become stage 1 (the coarse pass); stages 2 and 3 each narrow "
        "around the previous stage's winner using the widths/steps here, "
        "clipped to stage 1's own min/max. The one expensive computation "
        "(inference/do_3D, where applicable) still runs only once and is "
        "reused across all 3 stages."
    )
    note.setWordWrap(True)
    note.setStyleSheet("color: #888; font-size: 10px;")
    layout.addWidget(note)

    return cb, w2, s2, w3, s3


def _is_valid_cellpose_checkpoint(path) -> bool:
    """True if `path` looks like a real PyTorch checkpoint (a zip
    archive -- the format torch.save() has used since PyTorch 1.6) rather
    than some other file picked by mistake.

    Cellpose-SAM checkpoints are typically extensionless, so the browse
    dialog can't filter by extension (see _on_browse_cp_model) -- without
    this check, selecting the wrong file (e.g. an input image .tif) was
    silently accepted and stored, and only failed much later, deep
    inside cellpose's own torch.load(), with a cryptic
    "mmap can only be used with files saved with torch.save(...)" error
    in a background thread partway through a live run.
    """
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def _add_recommended_label(layout, initial=None, unit="", noun="value"):
    """Read-only "Recommended: X" display shown next to a slider that a
    GT sweep can auto-apply a value to -- distinct from the slider
    itself, which stays freely editable afterward. Shows what the last
    GT-confirmed sweep found even after the user has since moved the
    slider away from it, so that history isn't silently lost the moment
    someone nudges the control.

    Returns the label; call .setText() on it from the sweep's own
    result handler once a GT-confirmed recommendation is available."""
    text = (
        f"  Recommended {noun}: {initial}{unit}"
        if initial is not None else
        f"  Recommended {noun}: not yet measured — run a GT-verified sweep below."
    )
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #aaa; font-size: 10px;")
    lbl.setWordWrap(True)
    layout.addWidget(lbl)
    return lbl


def _set_layout_widgets_visible(layout, visible):
    """Recursively show/hide every widget reachable from `layout` -- rows in
    this file are a mix of addWidget(...) and addLayout(...) (nested
    QHBoxLayout rows), and only widgets (not layouts) support setVisible(),
    so nested layouts need walking rather than a single flat loop."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        w = item.widget()
        if w is not None:
            w.setVisible(visible)
        else:
            sub = item.layout()
            if sub is not None:
                _set_layout_widgets_visible(sub, visible)


def _make_collapsible(groupbox: QGroupBox, start_expanded: bool = True) -> QGroupBox:
    """Turn an already-built QGroupBox into a collapsible section: clicking
    its title checkbox hides/shows its contents, so a collapsed group
    shrinks to just its title bar and whatever comes after it in the same
    QVBoxLayout moves up -- the fix for a tab having more groups than fit
    on a laptop screen at once (no scroll area wraps each tab, so content
    below the fold was previously just unreachable). Works generically on
    any groupbox already populated via setLayout(...), whether its rows
    were added with addWidget or addLayout -- no need to restructure how
    each group was built."""
    groupbox.setCheckable(True)
    groupbox.setChecked(start_expanded)

    def _on_toggled(checked):
        lay = groupbox.layout()
        if lay is not None:
            _set_layout_widgets_visible(lay, checked)

    groupbox.toggled.connect(_on_toggled)
    _on_toggled(start_expanded)
    return groupbox


def _wrap_scroll(widget: QWidget) -> QScrollArea:
    """Wrap a fully-built tab page in a vertical-only QScrollArea. Collapsible
    groups (_make_collapsible) help but aren't enough on their own -- a tab
    with many groups (all expanded, or just genuinely a lot of controls) can
    still be taller than a laptop screen, and without this the content below
    the fold was completely unreachable (no scrollbar existed at all).
    setWidgetResizable(True) makes the scroll area resize `widget` to match
    its own width, so only a vertical scrollbar ever appears -- horizontal
    content isn't meant to scroll, per the panel-width work."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    return scroll


def _sep():
    """Thin horizontal separator line."""
    w = QWidget()
    w.setFixedHeight(1)
    w.setStyleSheet("background-color: #666;")
    return w


def _make_notify_checkbox():
    """'Email me when done' checkbox for the plugin's longer-running, non-
    detached operations (Tab 1 Run, Tab 2 Cellpose-SAM Segmentation, Tab
    5's Cellprob/Large-contact and Best-Epoch sweeps) -- each can run
    well past 30 minutes. Unchecked by default; uses the shared SMTP
    credentials configured once in Tab 5 -- Sweeps & Utilities, General
    category, Email notification panel. Kept as a plain function (not a
    QCheckBox subclass) since every call site still needs to persist its
    own checked-state to config and wire completion separately -- this
    only removes the repeated construction/tooltip boilerplate."""
    cb = QCheckBox("Email me when done")
    cb.setToolTip(
        "Sends one email when this finishes (or errors), using the SMTP "
        "credentials set once in Tab 5 -- Sweeps & Utilities, General "
        "category, Email notification panel."
    )
    return cb


def _extract_region_lines_um(shapes_lyr):
    """
    Extract boundary curves from a Shapes layer as (M, 2) YX arrays in µm.

    Accepted shape types:
      - 'line'  — 2-point straight line
      - 'path'  — multi-point polyline (any number of vertices)

    Each returned array has shape (M, 2) where M >= 2 and columns are [Y, X].
    """
    scale = np.array(shapes_lyr.scale)
    lines = []
    for data, stype in zip(shapes_lyr.data, shapes_lyr.shape_type):
        if stype not in ("line", "path"):
            continue
        pts = np.array(data) * scale   # (M, ndim)
        lines.append(pts[:, -2:])      # last 2 dims = YX → shape (M, 2)
    return lines


class ZFMicrogliaAIWidget(QWidget):
    """
    Napari dock panel for zebrafish brain extraction, microglia
    segmentation, and statistics (ZF-Microglia-AI).

    Layout
    ------
    [Open TIF / IMS file]
    ─────────────────────
    Model (.pth):
    [path…]  [...]
    ─────────────────────
    Input: bottom layer (auto)
      "{name}"  (Z×Y×X  dtype)
    ─────────────────────
      Z=1.0000  Y=0.1740  X=0.1740 µm
      Anisotropy 5.75:1  |  TIF ImageJ metadata
    ─────────────────────
    Threshold: [────●──]  0.30
    ─────────────────────
    [x] Save brain_only.tif
    [x] Save brain_mask.tif
    ─────────────────────
    [     Run Skin-Remover     ]
    Status: Ready
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = napari_viewer
        cfg = _load_config()
        cfg, _migrated_secrets = _secrets.migrate_plaintext_secrets(cfg)
        if _migrated_secrets:
            _save_config(cfg)
        # Model path priority: saved config > hardcoded default > None
        # NOTE: must use is_file(), not exists() -- Path(cfg.get(key, ""))
        # is Path("") when nothing was ever saved, which pathlib silently
        # normalizes to Path(".") (current directory). exists() returns
        # True for a directory too, so it was incorrectly treating "no
        # model configured yet" as "a valid model is loaded", which then
        # failed much later and far more confusingly inside torch.load()
        # ("Permission denied: '.'" instead of a clear "no model selected").
        saved_model = Path(cfg.get("model_path", ""))
        if saved_model.is_file():
            initial_model = saved_model
        elif DEFAULT_MODEL.is_file():
            initial_model = DEFAULT_MODEL
        else:
            initial_model = None
        # Cellpose-SAM checkpoint: saved config only (no bundled default —
        # this is a project-specific fine-tuned model, not shipped with the plugin)
        saved_cp_model = Path(cfg.get("cellpose_model_path", ""))
        initial_cp_model = saved_cp_model if saved_cp_model.is_file() else None
        self._state = {
            "model_path":         initial_model,
            "cellpose_model_path": initial_cp_model,
            "last_file_path":     None,
            "metadata":           None,
            "config":             cfg,
        }
        self._build_ui()
        self._connect_signals()
        self._refresh_layer_info()
        self._refresh_stats_layers()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        tabs = QTabWidget()
        self._tabs = tabs

        # Tab 5 — "Sweeps & Utilities": declared up front (not at the end,
        # where it's added to the QTabWidget) so the six GT-sweep/utility
        # groups built further down in each of Tabs 1-4's own sections can
        # simply target t5.addWidget(...) instead of their original tab's
        # layout at construction time -- moving *where a group is displayed*
        # doesn't require moving *how it's built*, since every widget it
        # references is a plain self.-attribute, reachable regardless of
        # which tab visually contains it. Consolidates every "Verify ... (GT
        # Sweep)" tool plus two adjacent GT-utility tools (Build
        # GT-Correction Package, Score Against GT) that were previously
        # scattered across four tabs, cluttering the primary workflow
        # sections they were sitting in.
        tab5 = QWidget()
        t5 = QVBoxLayout()
        t5.setSpacing(6)
        t5_note = QLabel(
            "GT-verification sweeps and related utilities, consolidated from "
            "Tabs 1-4 -- each tool below still operates on its own tab's data/"
            "sliders (e.g. the MONAI sweep still recalibrates Tab 1's "
            "Threshold/Erosion) and auto-applies its findings back there. "
            "Nothing here is a separate workflow of its own."
        )
        t5_note.setWordWrap(True)
        t5_note.setStyleSheet("color: #888; font-size: 10px;")
        t5.addWidget(t5_note)
        t5.addWidget(_sep())

        # Category filter -- with 7 tools stacked in one tab and no
        # indication of which pipeline each belongs to, it was unclear at a
        # glance what any given tool was even for. Each checkbox toggles
        # visibility of every tool tagged with that category (populated via
        # self._t5_category_groups.setdefault(category, []).append(group)
        # at each tool's own construction site below, since the groups
        # themselves don't exist yet at this point in the method -- the
        # checkboxes are wired and given their initial visibility only once
        # all seven groups have been built, see _on_t5_filter_changed()
        # near the end of this tab's section). Independent checkboxes, not
        # a mutually-exclusive radio switch like Tab 4's MONAI/Cellpose-SAM
        # mode -- unlike training, a user may genuinely want e.g. both
        # Pixel Classifier and Cellpose-SAM tools visible at once if they
        # use both pipelines across different fish.
        self._t5_category_groups = {}
        t5fg = QGroupBox("Show tools for...")
        t5fl = QVBoxLayout()
        t5fl.setSpacing(4)
        _t5f_cfg = self._state.get("config", {}).get("t5_filters", {})
        self._t5_filter_skin_cb = QCheckBox("Skin Removal (MONAI)")
        self._t5_filter_skin_cb.setChecked(_t5f_cfg.get("skin", True))
        self._t5_filter_pixel_cb = QCheckBox("Pixel Classifier segmentation")
        self._t5_filter_pixel_cb.setChecked(_t5f_cfg.get("pixel", True))
        self._t5_filter_cellpose_cb = QCheckBox("Cellpose-SAM segmentation")
        self._t5_filter_cellpose_cb.setChecked(_t5f_cfg.get("cellpose", True))
        self._t5_filter_general_cb = QCheckBox("General (any pipeline)")
        self._t5_filter_general_cb.setChecked(_t5f_cfg.get("general", True))
        for cb in (self._t5_filter_skin_cb, self._t5_filter_pixel_cb,
                   self._t5_filter_cellpose_cb, self._t5_filter_general_cb):
            t5fl.addWidget(cb)
        t5fg.setLayout(t5fl)
        t5.addWidget(t5fg)
        t5.addWidget(_sep())

        # Sliders that the plugin's own GT-verification sweeps can now
        # auto-apply their recommendation to (per explicit instruction:
        # "report but set the values for the next use" -- otherwise
        # what are the sweeps for). Persisted like any other setting, so
        # a sweep's finding survives a napari restart, not just the rest
        # of this session.
        _root_cfg = self._state.get("config", {})

        # ── GT Toolkit Tuning Tool — one button tunes the whole plugin ── #
        # Every GT-driven auto-apply in this plugin (Min/Max volume, Min
        # hole size, MONAI Threshold/Erosion, BG Threshold/Signal
        # Erosion, Sigma XY/Z, Cellprob + Krendl merge params,
        # branch_radius) now happens ONLY from here -- explicit
        # principle: "anything that changes recommended values should be
        # included in the tuning tool." Every one of the 5 folded-in
        # tools below still works standalone for a quick preview/report
        # (informational, all fields/buttons/reports fully usable), but
        # each one's own "This is verified ground truth" checkbox is
        # hidden (see _add_gt_checkbox(..., visible=False)) so nothing
        # except this orchestrator can actually move a shared
        # recommended value -- same reasoning Tab 3 Statistics's own GT
        # checkbox was removed outright for. "Verify Best Epoch" is
        # deliberately NOT folded in here at all -- it selects a
        # training checkpoint, not a GT-tuned pipeline parameter, a
        # different category of tool entirely, explicitly kept separate.
        # Reuses every folded-in tool's own tested worker code
        # unchanged: this only auto-discovers one fish's files from its
        # canonical folder (_gt_toolkit.py), fills them in, and runs
        # each applicable step in sequence, one at a time.
        gtkg = QGroupBox("GT Toolkit Tuning Tool")
        gtkl = QVBoxLayout()
        gtkl.setSpacing(6)

        gtk_note = QLabel(
            "One button: point this at a fish's original file (or its "
            "already-created output folder) and it tunes the whole plugin "
            "against that fish's GROUND_TRUTH in one shot -- Min/Max volume "
            "+ Min hole size (instant), then whichever of MONAI Threshold/"
            "Erosion, BG Threshold/Signal Erosion, Sigma XY/Z, Cellprob + "
            "Krendl merge params, and branch_radius this fish's discovered "
            "files support, run in that order. Skips any step whose "
            "required file isn't found -- see the discovery report after "
            "scanning."
        )
        gtk_note.setWordWrap(True)
        gtk_note.setStyleSheet("color: #888; font-size: 10px;")
        gtkl.addWidget(gtk_note)

        gtk_path_row = QHBoxLayout()
        gtk_path_row.addWidget(QLabel("Fish original file or folder:"))
        self._gtk_path_edit = QLineEdit(_root_cfg.get("gtk_last_path", ""))
        gtk_path_row.addWidget(self._gtk_path_edit)
        self._gtk_browse_file_btn = QPushButton("File...")
        self._gtk_browse_file_btn.setFixedWidth(52)
        gtk_path_row.addWidget(self._gtk_browse_file_btn)
        self._gtk_browse_folder_btn = QPushButton("Folder...")
        self._gtk_browse_folder_btn.setFixedWidth(60)
        gtk_path_row.addWidget(self._gtk_browse_folder_btn)
        gtkl.addLayout(gtk_path_row)
        gtk_path_note = QLabel(
            "  Established convention: <parent>/<stem>/<stem>_<artifact>.tif "
            "for every derived file (ExtRm/NoBG/RndFill, brain_mask, "
            "cp_masks, cp_masks_corrected, GROUND_TRUTH, statistics.csv) -- "
            "the same folder \"Build GT-Correction Package\" below delivers "
            "into too. Point here at either the raw source file (its "
            "sibling folder is found automatically) or the folder itself."
        )
        gtk_path_note.setStyleSheet("color: #aaa; font-size: 10px;")
        gtk_path_note.setWordWrap(True)
        gtkl.addWidget(gtk_path_note)

        self._gtk_scan_btn = QPushButton("Scan Folder")
        gtkl.addWidget(self._gtk_scan_btn)

        self._gtk_discovery_view = QTextEdit()
        self._gtk_discovery_view.setReadOnly(True)
        self._gtk_discovery_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._gtk_discovery_view.setFixedHeight(140)
        gtkl.addWidget(self._gtk_discovery_view)

        self._gtk_notify_cb = _make_notify_checkbox()
        gtkl.addWidget(self._gtk_notify_cb)

        self._gtk_is_gt_cb = QCheckBox("This is verified ground truth")
        self._gtk_is_gt_cb.setChecked(False)
        gtkl.addWidget(self._gtk_is_gt_cb)
        gtk_is_gt_note = QLabel(
            "  Off by default. Only tick when <stem>_GROUND_TRUTH.tif (and "
            "brain_mask.tif, for the MONAI step) have actually been "
            "manually corrected/verified -- this is the only switch left "
            "anywhere in the plugin that can move a shared recommended "
            "value; every step below still runs and reports without it, "
            "it just can't apply anything. Un-ticks itself once the whole "
            "run finishes."
        )
        gtk_is_gt_note.setWordWrap(True)
        gtk_is_gt_note.setStyleSheet("color: #888; font-size: 10px;")
        gtkl.addWidget(gtk_is_gt_note)

        gtk_btn_row = QHBoxLayout()
        self._gtk_run_btn = QPushButton("Run Full GT Toolkit Tuning")
        self._gtk_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        gtk_btn_row.addWidget(self._gtk_run_btn)
        self._gtk_stop_btn = QPushButton("Stop")
        self._gtk_stop_btn.setEnabled(False)
        gtk_btn_row.addWidget(self._gtk_stop_btn)
        gtkl.addLayout(gtk_btn_row)

        self._gtk_status_lbl = QLabel("")
        self._gtk_status_lbl.setWordWrap(True)
        gtkl.addWidget(self._gtk_status_lbl)

        self._gtk_report_view = QTextEdit()
        self._gtk_report_view.setReadOnly(True)
        self._gtk_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._gtk_report_view.setFixedHeight(220)
        gtkl.addWidget(self._gtk_report_view)

        gtkg.setLayout(gtkl)
        gtkg = _make_collapsible(gtkg)
        t5.addWidget(gtkg)
        t5.addWidget(_sep())
        self._t5_category_groups.setdefault("general", []).append(gtkg)

        self._gtk_job = {"queue": [], "step_index": 0, "timer": None, "running": False}

        # ============================================================ #
        # TAB 1 — Skin Remover
        # ============================================================ #
        tab1 = QWidget()
        t1 = QVBoxLayout()
        t1.setSpacing(6)

        t1_note = QLabel(
            "Runs a trained MONAI 3D U-Net to detect and remove everything "
            "outside the brain from a raw confocal stack, producing a clean "
            "brain_only image for Tab 2 to label. Open a file below, adjust "
            "the threshold/background options if needed, then Run."
        )
        t1_note.setWordWrap(True)
        t1_note.setStyleSheet("color: #888; font-size: 10px;")
        t1.addWidget(t1_note)
        t1.addWidget(_sep())

        self._open_btn = QPushButton("Open TIF / IMS file")
        t1.addWidget(self._open_btn)

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Model (.pth):"))
        model_row = QHBoxLayout()
        self._model_lbl = QLabel(
            str(self._state["model_path"]) if self._state["model_path"] else "— no model selected —"
        )
        self._model_lbl.setWordWrap(True)
        self._model_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._model_browse_btn = QPushButton("...")
        self._model_browse_btn.setFixedWidth(32)
        model_row.addWidget(self._model_lbl)
        model_row.addWidget(self._model_browse_btn)
        t1.addLayout(model_row)

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Input: active (selected) layer"))
        self._layer_info = QLabel("  — no layers yet —")
        self._layer_info.setWordWrap(True)
        t1.addWidget(self._layer_info)

        t1.addWidget(_sep())

        self._meta_lbl = QLabel("  — voxel info unavailable —")
        self._meta_lbl.setWordWrap(True)
        self._meta_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(self._meta_lbl)

        t1.addWidget(_sep())

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("MONAI Threshold:"))
        self._thresh_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._thresh_slider.setDecimals(2)
        self._thresh_slider.setMinimum(0.01)
        self._thresh_slider.setMaximum(0.99)
        self._thresh_slider.setSingleStep(0.01)
        self._thresh_slider.setValue(_root_cfg.get("monai_threshold", 0.25))
        thresh_row.addWidget(self._thresh_slider)
        self._thresh_spin = _add_reliable_spinbox(
            thresh_row, self._thresh_slider, 0.01, 0.99, 0.01, decimals=2
        )
        t1.addLayout(thresh_row)
        self._thresh_recommended_lbl = _add_recommended_label(
            t1, _root_cfg.get("monai_threshold_recommended"), noun="MONAI Threshold"
        )

        erosion_row = QHBoxLayout()
        erosion_row.addWidget(QLabel("MONAI Erosion (vox):"))
        self._erosion_slider = QLabeledSlider(Qt.Horizontal)
        self._erosion_slider.setMinimum(0)
        self._erosion_slider.setMaximum(15)
        self._erosion_slider.setValue(_root_cfg.get("erosion_voxels", 0))
        erosion_row.addWidget(self._erosion_slider)
        self._erosion_spin = _add_reliable_spinbox(
            erosion_row, self._erosion_slider, 0, 15, 1
        )
        t1.addLayout(erosion_row)
        erosion_note = QLabel(
            "  Erodes the raw MONAI brain mask (Skin-Remover step)\n"
            "  (raw brain_mask is always saved un-eroded)"
        )
        erosion_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(erosion_note)
        self._erosion_recommended_lbl = _add_recommended_label(
            t1, _root_cfg.get("erosion_voxels_recommended"), unit=" vox", noun="MONAI Erosion"
        )

        t1.addWidget(_sep())

        t1.addWidget(QLabel("Background (brain mode):"))
        self._bg_group = QButtonGroup(self)
        self._bg_off_rb    = QRadioButton("Off")
        self._bg_mode1_rb  = QRadioButton("1 — Remove background outside brain (inference)")
        self._bg_mode2_rb  = QRadioButton("2 — Remove background globally (full stack)")
        self._bg_mode3_rb  = QRadioButton("3 — Fill removed with random background")
        self._bg_group.addButton(self._bg_off_rb,   0)
        self._bg_group.addButton(self._bg_mode1_rb, 1)
        self._bg_group.addButton(self._bg_mode2_rb, 2)
        self._bg_group.addButton(self._bg_mode3_rb, 3)
        self._bg_off_rb.setChecked(True)
        t1.addWidget(self._bg_off_rb)
        t1.addWidget(self._bg_mode1_rb)
        t1.addWidget(self._bg_mode2_rb)
        t1.addWidget(self._bg_mode3_rb)

        tol_row = QHBoxLayout()
        self._tol_lbl = QLabel("  BG Threshold:")
        tol_row.addWidget(self._tol_lbl)
        self._tol_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._tol_slider.setDecimals(2)
        self._tol_slider.setMinimum(0.00)
        self._tol_slider.setMaximum(2.00)
        self._tol_slider.setSingleStep(0.01)
        self._tol_slider.setValue(_root_cfg.get("bg_tolerance", 1.40))
        tol_row.addWidget(self._tol_slider)
        self._tol_spin = _add_reliable_spinbox(
            tol_row, self._tol_slider, 0.00, 2.00, 0.01, decimals=2
        )
        t1.addLayout(tol_row)
        self._tol_recommended_lbl = _add_recommended_label(
            t1, _root_cfg.get("bg_tolerance_recommended"), noun="BG Threshold"
        )

        self._sig_erosion_row = QHBoxLayout()
        self._sig_erosion_lbl = QLabel("  Signal Erosion (vox):")
        self._sig_erosion_row.addWidget(self._sig_erosion_lbl)
        self._sig_erosion_slider = QLabeledSlider(Qt.Horizontal)
        self._sig_erosion_slider.setMinimum(0)
        self._sig_erosion_slider.setMaximum(15)
        self._sig_erosion_slider.setValue(_root_cfg.get("signal_erosion_voxels", 0))
        self._sig_erosion_row.addWidget(self._sig_erosion_slider)
        self._sig_erosion_spin = _add_reliable_spinbox(
            self._sig_erosion_row, self._sig_erosion_slider, 0, 15, 1
        )
        t1.addLayout(self._sig_erosion_row)
        self._sig_erosion_recommended_lbl = _add_recommended_label(
            t1, _root_cfg.get("signal_erosion_voxels_recommended"), unit=" vox", noun="Signal Erosion"
        )
        # Only meaningful for Background mode 2 (noBG stack prep) -- shown/
        # hidden, not just enabled/disabled, so it doesn't sit there
        # looking relevant when it does nothing in modes Off/1/3.
        _set_layout_widgets_visible(self._sig_erosion_row, False)
        self._sig_erosion_recommended_lbl.setVisible(False)

        bg_note = QLabel(
            "  Probe: inside-brain mode (post-inference)\n"
            "  Mode 1 & 2 use BG Threshold  |  Mode 3: no threshold\n"
            "  Signal Erosion applies only to Mode 2 (noBG stack prep)"
        )
        bg_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(bg_note)

        t1.addWidget(_sep())

        self._save_only_cb = QCheckBox("Save brain_only.tif")
        self._save_only_cb.setChecked(True)
        self._save_mask_cb = QCheckBox("Save brain_mask.tif")
        self._save_mask_cb.setChecked(True)
        t1.addWidget(self._save_only_cb)
        t1.addWidget(self._save_mask_cb)

        t1.addWidget(_sep())

        self._run_notify_cb = _make_notify_checkbox()
        t1.addWidget(self._run_notify_cb)

        self._run_btn = QPushButton("Run Skin-Remover")
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t1.addWidget(self._run_btn)

        self._status_lbl = QLabel("Status: Ready")
        self._status_lbl.setWordWrap(True)
        t1.addWidget(self._status_lbl)

        # Live output -- MONAI's own sliding-window progress (tqdm, one
        # line per window) forwarded here instead of vanishing into a
        # terminal that may not exist; see _live_progress.py.
        self._run_log_view = QTextEdit()
        self._run_log_view.setReadOnly(True)
        self._run_log_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._run_log_view.setFixedHeight(120)
        t1.addWidget(self._run_log_view)

        t1.addWidget(_sep())

        # ── Verify MONAI Threshold / Erosion (GT sweep) — always visible, ──
        # third of the plugin's three GT-sweep tools (see Tab 2's BG
        # Threshold/Erosion sweep, Tab 4's Cellpose-SAM epoch sweep). Scores
        # the brain MASK itself against a hand-corrected GT mask (from GT
        # Annotation) rather than per-cell labels -- no GPU hard-gate, since
        # inference already falls back to CPU/MPS elsewhere in this tab.
        bsg = QGroupBox("Verify MONAI Threshold / Erosion (GT Sweep)")
        bsl = QVBoxLayout()
        bsl.setSpacing(6)

        bs_note = QLabel(
            "Runs MONAI inference once, then cheaply sweeps MONAI Threshold x "
            "Erosion on the resulting probability map, scoring the whole brain "
            "mask against a hand-corrected GT brain mask (e.g. from GT "
            "Annotation in Tab 4) — Dice/IoU on the mask itself, not per-cell."
        )
        bs_note.setWordWrap(True)
        bs_note.setStyleSheet("color: #888; font-size: 10px;")
        bsl.addWidget(bs_note)

        bs_img_row = QHBoxLayout()
        bs_img_row.addWidget(QLabel("Image:"))
        self._bs_img_edit = QLineEdit("")
        bs_img_row.addWidget(self._bs_img_edit)
        self._bs_img_browse_btn = QPushButton("...")
        self._bs_img_browse_btn.setFixedWidth(32)
        bs_img_row.addWidget(self._bs_img_browse_btn)
        bsl.addLayout(bs_img_row)
        bs_img_note = QLabel(
            "  Must be the RAW, pre-Tab 1 image — no _ExtRm / _NoBG / "
            "_RndFill suffix. Feeding an already brain-masked image would "
            "bias the very segmentation this tool is scoring."
        )
        bs_img_note.setStyleSheet("color: #aaa; font-size: 10px;")
        bs_img_note.setWordWrap(True)
        bsl.addWidget(bs_img_note)

        bs_gt_row = QHBoxLayout()
        bs_gt_row.addWidget(QLabel("GT brain mask:"))
        self._bs_gt_edit = QLineEdit("")
        bs_gt_row.addWidget(self._bs_gt_edit)
        self._bs_gt_browse_btn = QPushButton("...")
        self._bs_gt_browse_btn.setFixedWidth(32)
        bs_gt_row.addWidget(self._bs_gt_browse_btn)
        bsl.addLayout(bs_gt_row)
        bs_gt_note = QLabel(
            "  GT brain mask = a hand-corrected brain_mask.tif (e.g. from GT "
            "Annotation, Tab 4) — not a MONAI prediction."
        )
        bs_gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        bs_gt_note.setWordWrap(True)
        bsl.addWidget(bs_gt_note)

        bs_th_row = QHBoxLayout()
        bs_th_row.addWidget(QLabel("Threshold min:"))
        self._bs_thmin_spin = QDoubleSpinBox()
        self._bs_thmin_spin.setDecimals(2)
        self._bs_thmin_spin.setRange(0.01, 0.99)
        self._bs_thmin_spin.setValue(0.15)
        bs_th_row.addWidget(self._bs_thmin_spin)
        bsl.addLayout(bs_th_row)
        bs_th_row2 = QHBoxLayout()
        bs_th_row2.addWidget(QLabel("max:"))
        self._bs_thmax_spin = QDoubleSpinBox()
        self._bs_thmax_spin.setDecimals(2)
        self._bs_thmax_spin.setRange(0.01, 0.99)
        self._bs_thmax_spin.setValue(0.35)
        bs_th_row2.addWidget(self._bs_thmax_spin)
        bs_th_row2.addWidget(QLabel("step:"))
        self._bs_thstep_spin = QDoubleSpinBox()
        self._bs_thstep_spin.setDecimals(2)
        self._bs_thstep_spin.setRange(0.01, 0.99)
        self._bs_thstep_spin.setValue(0.05)
        bs_th_row2.addWidget(self._bs_thstep_spin)
        bsl.addLayout(bs_th_row2)

        bs_er_row = QHBoxLayout()
        bs_er_row.addWidget(QLabel("Erosion min:"))
        self._bs_ermin_spin = QSpinBox()
        self._bs_ermin_spin.setRange(0, 15)
        self._bs_ermin_spin.setValue(0)
        bs_er_row.addWidget(self._bs_ermin_spin)
        bsl.addLayout(bs_er_row)
        bs_er_row2 = QHBoxLayout()
        bs_er_row2.addWidget(QLabel("max:"))
        self._bs_ermax_spin = QSpinBox()
        self._bs_ermax_spin.setRange(0, 15)
        self._bs_ermax_spin.setValue(4)
        bs_er_row2.addWidget(self._bs_ermax_spin)
        bs_er_row2.addWidget(QLabel("step:"))
        self._bs_erstep_spin = QSpinBox()
        self._bs_erstep_spin.setRange(1, 15)
        self._bs_erstep_spin.setValue(1)
        bs_er_row2.addWidget(self._bs_erstep_spin)
        bsl.addLayout(bs_er_row2)

        (self._bs_sieve_cb, self._bs_sieve2_w_spin, self._bs_sieve2_s_spin,
         self._bs_sieve3_w_spin, self._bs_sieve3_s_spin) = _add_sieve_controls(
            bsl, "MONAI Threshold", 0.05, 0.01, 0.02, 0.005
        )

        self._bs_is_gt_cb = _add_gt_checkbox(bsl, "this sweep", visible=False)

        bs_btn_row = QHBoxLayout()
        self._bs_run_btn = QPushButton("Run Threshold/Erosion Sweep")
        self._bs_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        bs_btn_row.addWidget(self._bs_run_btn)
        self._bs_stop_btn = QPushButton("Stop Sweep")
        self._bs_stop_btn.setEnabled(False)
        bs_btn_row.addWidget(self._bs_stop_btn)
        bsl.addLayout(bs_btn_row)

        self._bs_status_lbl = QLabel("")
        self._bs_status_lbl.setWordWrap(True)
        bsl.addWidget(self._bs_status_lbl)

        self._bs_report_view = QTextEdit()
        self._bs_report_view.setReadOnly(True)
        self._bs_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._bs_report_view.setFixedHeight(160)
        bsl.addWidget(self._bs_report_view)

        bsg.setLayout(bsl)
        bsg = _make_collapsible(bsg)
        t5.addWidget(bsg)
        self._t5_category_groups.setdefault("skin", []).append(bsg)

        self._brain_sweep_job = {"thread": None, "cancel_event": None, "timer": None}

        t1.addStretch()
        tab1.setLayout(t1)
        tabs.addTab(_wrap_scroll(tab1), "Skin Remover")

        # ============================================================ #
        # TAB 2 — Create Labels
        # ============================================================ #
        tab2 = QWidget()
        t2 = QVBoxLayout()
        t2.setSpacing(6)

        t2_note = QLabel(
            "Detects and labels individual microglia in 3D from Tab 1's "
            "brain_only output. Shows exactly one of two interchangeable "
            "methods below, chosen automatically from your active layer's "
            "filename suffix — Pixel Classifier (_NoBG) or Cellpose-SAM "
            "Segmentation (_ExtRm, recommended if you have a GPU)."
        )
        t2_note.setWordWrap(True)
        t2_note.setStyleSheet("color: #888; font-size: 10px;")
        t2.addWidget(t2_note)
        t2.addWidget(_sep())

        self._labels_mode_hint = QLabel("")
        self._labels_mode_hint.setWordWrap(True)
        self._labels_mode_hint.setStyleSheet("color: #8ab; font-size: 10px; font-style: italic;")
        t2.addWidget(self._labels_mode_hint)

        # ── Common Settings — shared, always visible regardless of which ───
        # route (Pixel Classifier or Cellpose-SAM) is currently shown below.
        # Min hole size and Final min-size fraction are genuinely shared
        # values/histories used identically in spirit by both routes (Min
        # hole size: Pixel Classifier per-slice, Cellpose-SAM in full 3D
        # via a monkey-patch -- see _make_capped_fill_holes() in
        # _cellpose_seg.py. Final min-size fraction: Pixel Classifier's own
        # volume filter, Cellpose-SAM's last-stage safety net -- see
        # _labeling.py / _cellpose_seg.py's final_min_size_cleanup()).
        # Min volume is shown here too (informative, not editable -- see
        # its own note below) since both routes read it. Cellpose-SAM's
        # min_size is NOT one of these -- it lives in the Cellpose-SAM
        # group instead, since it's a deliberately tiny early noise filter
        # unrelated to Min volume's role (real fragments below Min
        # volume's threshold still need to survive to reach GMM cleanup
        # and Krendl safe-merge, which decide reattach-vs-discard far more
        # precisely than a single blunt cutoff could -- sharing one number
        # between them would silently break that reattachment path).
        common_group = QGroupBox("Common Settings (both Pixel Classifier + Cellpose-SAM)")
        common_layout = QVBoxLayout()

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Min volume (vox):"))
        init_min_volume = _root_cfg.get("min_volume_vox", 7500)
        self._area_value_lbl = QLabel(str(init_min_volume))
        self._area_value_lbl.setStyleSheet("font-weight: bold;")
        area_row.addWidget(self._area_value_lbl)
        area_row.addStretch()
        common_layout.addLayout(area_row)
        area_note = QLabel(
            "  Informative only, not directly editable — the smallest "
            "real cell volume ever confirmed by GT, shared by the Pixel "
            "Classifier's own volume filter and Cellpose-SAM's Safe-merge "
            "\"already a whole cell\" floor. It only ever shrinks, and "
            "only from a Tab 5 sweep or a GT-verified Generate Statistics "
            "run (Tab 3) — never guessed or hand-tuned. Final min-size "
            "fraction below is the actual tunable control: how far under "
            "this floor a blob can sit before either route deletes it."
        )
        area_note.setWordWrap(True)
        area_note.setStyleSheet("color: #888; font-size: 10px;")
        common_layout.addWidget(area_note)

        maxvol_row = QHBoxLayout()
        maxvol_row.addWidget(QLabel("Max volume (vox):"))
        init_max_volume = _root_cfg.get("max_volume_recommended_vox")
        self._maxvol_value_lbl = QLabel(
            str(init_max_volume) if init_max_volume is not None else "not yet measured"
        )
        self._maxvol_value_lbl.setStyleSheet("font-weight: bold;")
        maxvol_row.addWidget(self._maxvol_value_lbl)
        maxvol_row.addStretch()
        common_layout.addLayout(maxvol_row)
        maxvol_note = QLabel(
            "  Informative only, not directly editable — the largest real "
            "cell volume ever confirmed by GT. It only ever grows, and "
            "only from a GT-verified Generate Statistics run (Tab 3) — "
            "never guessed or hand-tuned. Doesn't drive any pipeline "
            "stage; only used to flag is_volume_outlier in the "
            "Statistics CSV (Tab 3) for anything bigger."
        )
        maxvol_note.setWordWrap(True)
        maxvol_note.setStyleSheet("color: #888; font-size: 10px;")
        common_layout.addWidget(maxvol_note)

        hole_row = QHBoxLayout()
        hole_row.addWidget(QLabel("Min hole size (vox) — shared:"))
        self._hole_slider = QLabeledSlider(Qt.Horizontal)
        init_min_hole = _root_cfg.get("min_hole_size_vox", 0)
        init_min_hole_recommended = _root_cfg.get("min_hole_size_recommended_vox")
        hole_slider_max = max(500, init_min_hole, init_min_hole_recommended or 500)
        self._hole_slider.setMinimum(0)
        self._hole_slider.setMaximum(hole_slider_max)
        self._hole_slider.setValue(init_min_hole)
        hole_row.addWidget(self._hole_slider)
        self._hole_spin = _add_reliable_spinbox(
            hole_row, self._hole_slider, 0, hole_slider_max, 10
        )
        common_layout.addLayout(hole_row)
        hole_note = QLabel(
            "  A background region fully enclosed by signal survives as "
            "real background only if it's at or above this size; smaller "
            "enclosed gaps are filled in as noise. 0 = fill every enclosed "
            "gap regardless of size (the old, unconditional behavior). "
            "Applies to whichever labelling route is active below — per "
            "2D slice for Pixel Classifier, per full 3D mask for "
            "Cellpose-SAM."
        )
        hole_note.setStyleSheet("color: #888; font-size: 10px;")
        hole_note.setWordWrap(True)
        common_layout.addWidget(hole_note)
        self._hole_recommended_lbl = QLabel(
            f"  Recommended floor (from GT sweeps so far): {init_min_hole_recommended} vox"
            if init_min_hole_recommended is not None else
            "  Recommended floor: not yet measured — run Verify BG Threshold / Erosion (GT Sweep) below."
        )
        self._hole_recommended_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        self._hole_recommended_lbl.setWordWrap(True)
        common_layout.addWidget(self._hole_recommended_lbl)

        finalfrac_row = QHBoxLayout()
        finalfrac_row.addWidget(QLabel("Final min-size fraction — both routes:"))
        self._finalfrac_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._finalfrac_slider.setDecimals(3)
        self._finalfrac_slider.setMinimum(0.0)
        self._finalfrac_slider.setMaximum(1.0)
        self._finalfrac_slider.setSingleStep(0.01)
        self._finalfrac_slider.setValue(
            _root_cfg.get("final_min_fraction", _root_cfg.get("cellpose_final_min_fraction", 0.618))
        )
        finalfrac_row.addWidget(self._finalfrac_slider)
        self._finalfrac_spin = _add_reliable_spinbox(
            finalfrac_row, self._finalfrac_slider, 0.0, 1.0, 0.01, decimals=3
        )
        common_layout.addLayout(finalfrac_row)
        finalfrac_note = QLabel(
            "  The actual small-object deletion cutoff both routes use is "
            "this fraction of the Min volume floor above (the smallest "
            "real cell ever confirmed in GT), not Min volume itself: "
            "Cellpose-SAM applies it as a final safety-net stage after "
            "large-contact merge; the Pixel Classifier applies it directly "
            "as its own 3D-blob volume filter, since that route has no "
            "merge/reattach stage to leave a smaller object standing for "
            "later reconsideration. Default is the golden ratio, ~0.618 -- "
            "strict enough to catch clear debris, lenient enough not to "
            "reject a legitimately smaller-than-average real cell. 1.0 "
            "disables the relaxation entirely (cutoff == Min volume)."
        )
        finalfrac_note.setStyleSheet("color: #888; font-size: 10px;")
        finalfrac_note.setWordWrap(True)
        common_layout.addWidget(finalfrac_note)

        common_group.setLayout(common_layout)
        t2.addWidget(common_group)

        # ── Pixel Classifier (union-find labels) — shown for _NoBG layers ── #
        self._pixel_classifier_group = QGroupBox("Pixel Classifier — Union-Find Labels")
        pcg = QVBoxLayout()
        pcg.setSpacing(6)

        lbl_note = QLabel(
            "Classical, GPU-optional labelling: Gaussian smooth → threshold "
            "→ per-slice 2D connected components → overlap-based union-find "
            "into 3D objects → volume filter. No trained model needed — the "
            "fallback for machines with no usable GPU (see Cellpose-SAM "
            "Segmentation below for the recommended path).\n\n"
            "Run option 2 (Remove globally) first to get a\n"
            "brain_only (_NoBG) layer, then select it and click below."
        )
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color: #aaa; font-size: 10px;")
        pcg.addWidget(lbl_note)

        sxy_row = QHBoxLayout()
        sxy_row.addWidget(QLabel("Smooth σ XY:"))
        self._sxy_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._sxy_slider.setDecimals(1)
        self._sxy_slider.setMinimum(0.0)
        self._sxy_slider.setMaximum(5.0)
        self._sxy_slider.setSingleStep(0.1)
        self._sxy_slider.setValue(_root_cfg.get("sigma_xy", 1.5))
        sxy_row.addWidget(self._sxy_slider)
        self._sxy_spin = _add_reliable_spinbox(
            sxy_row, self._sxy_slider, 0.0, 5.0, 0.1, decimals=1
        )
        pcg.addLayout(sxy_row)
        self._sxy_recommended_lbl = _add_recommended_label(
            pcg, _root_cfg.get("sigma_xy_recommended"), noun="Smooth σ XY"
        )

        sz_row = QHBoxLayout()
        sz_row.addWidget(QLabel("Smooth σ Z:"))
        self._sz_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._sz_slider.setDecimals(1)
        self._sz_slider.setMinimum(0.0)
        self._sz_slider.setMaximum(5.0)
        self._sz_slider.setSingleStep(0.1)
        self._sz_slider.setValue(_root_cfg.get("sigma_z", 3.0))
        sz_row.addWidget(self._sz_slider)
        self._sz_spin = _add_reliable_spinbox(
            sz_row, self._sz_slider, 0.0, 5.0, 0.1, decimals=1
        )
        pcg.addLayout(sz_row)
        self._sz_recommended_lbl = _add_recommended_label(
            pcg, _root_cfg.get("sigma_z_recommended"), noun="Smooth σ Z"
        )

        self._labels_btn = QPushButton("Create Labels")
        self._labels_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        pcg.addWidget(self._labels_btn)

        self._labels_status_lbl = QLabel("")
        self._labels_status_lbl.setWordWrap(True)
        pcg.addWidget(self._labels_status_lbl)

        self._labels_log_view = QTextEdit()
        self._labels_log_view.setReadOnly(True)
        self._labels_log_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._labels_log_view.setFixedHeight(100)
        pcg.addWidget(self._labels_log_view)

        self._pixel_classifier_group.setLayout(pcg)
        self._pixel_classifier_group = _make_collapsible(self._pixel_classifier_group)
        t2.addWidget(self._pixel_classifier_group)

        # ── Verify BG Threshold / Erosion (GT sweep) — always visible, ──
        # not tied to the active-layer auto-switch above: it works from
        # explicit file paths, not the current viewer selection, and no
        # GPU is required (Create Labels already has a CPU fallback), so
        # it isn't grouped with Tab 4's GPU-gated tools either.
        psg = QGroupBox("Verify BG Threshold / Signal Erosion (GT Sweep)")
        psl = QVBoxLayout()
        psl.setSpacing(6)

        ps_note = QLabel(
            "Sweeps Tab 1's BG Threshold x Signal Erosion (Background mode 2 — "
            "\"Remove globally\", the noBG stack used by the Pixel Classifier) "
            "against the N most morphologically complex "
            "cells in a ground-truth-annotated fish, using this section's own "
            "σ XY / σ Z / Min volume above. Doesn't re-run MONAI inference — "
            "needs a pre-computed brain_mask.tif from a normal Tab 1 run."
        )
        ps_note.setWordWrap(True)
        ps_note.setStyleSheet("color: #888; font-size: 10px;")
        psl.addWidget(ps_note)

        ps_img_row = QHBoxLayout()
        ps_img_row.addWidget(QLabel("GT image:"))
        self._ps_img_edit = QLineEdit("")
        ps_img_row.addWidget(self._ps_img_edit)
        self._ps_img_browse_btn = QPushButton("...")
        self._ps_img_browse_btn.setFixedWidth(32)
        ps_img_row.addWidget(self._ps_img_browse_btn)
        psl.addLayout(ps_img_row)
        ps_img_note = QLabel(
            "  Raw image or _ExtRm — both give identical results here, "
            "since this tool only ever reads pixels inside the brain mask. "
            "Do not use _NoBG/_RndFill — those already had a different "
            "background step applied."
        )
        ps_img_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ps_img_note.setWordWrap(True)
        psl.addWidget(ps_img_note)

        ps_mask_row = QHBoxLayout()
        ps_mask_row.addWidget(QLabel("brain_mask.tif:"))
        self._ps_mask_edit = QLineEdit("")
        ps_mask_row.addWidget(self._ps_mask_edit)
        self._ps_mask_browse_btn = QPushButton("...")
        self._ps_mask_browse_btn.setFixedWidth(32)
        ps_mask_row.addWidget(self._ps_mask_browse_btn)
        psl.addLayout(ps_mask_row)

        ps_lbl_row = QHBoxLayout()
        ps_lbl_row.addWidget(QLabel("GT labels:"))
        self._ps_lbl_edit = QLineEdit("")
        ps_lbl_row.addWidget(self._ps_lbl_edit)
        self._ps_lbl_browse_btn = QPushButton("...")
        self._ps_lbl_browse_btn.setFixedWidth(32)
        ps_lbl_row.addWidget(self._ps_lbl_browse_btn)
        psl.addLayout(ps_lbl_row)
        ps_gt_note = QLabel(
            "  brain_mask.tif = the RAW (un-eroded) mask Tab 1 saves — same "
            "fish as GT image/labels. GT labels = a hand-corrected "
            "microglia instance-label volume, typically named "
            "_GROUND_TRUTH.tif, one integer ID per cell — not the brain "
            "mask."
        )
        ps_gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ps_gt_note.setWordWrap(True)
        psl.addWidget(ps_gt_note)

        ps_bg_row = QHBoxLayout()
        ps_bg_row.addWidget(QLabel("BG Threshold min:"))
        self._ps_bgmin_spin = QDoubleSpinBox()
        self._ps_bgmin_spin.setDecimals(2)
        self._ps_bgmin_spin.setRange(0.0, 2.0)
        self._ps_bgmin_spin.setValue(1.0)
        ps_bg_row.addWidget(self._ps_bgmin_spin)
        psl.addLayout(ps_bg_row)
        ps_bg_row2 = QHBoxLayout()
        ps_bg_row2.addWidget(QLabel("max:"))
        self._ps_bgmax_spin = QDoubleSpinBox()
        self._ps_bgmax_spin.setDecimals(2)
        self._ps_bgmax_spin.setRange(0.0, 2.0)
        self._ps_bgmax_spin.setValue(1.8)
        ps_bg_row2.addWidget(self._ps_bgmax_spin)
        ps_bg_row2.addWidget(QLabel("step:"))
        self._ps_bgstep_spin = QDoubleSpinBox()
        self._ps_bgstep_spin.setDecimals(2)
        self._ps_bgstep_spin.setRange(0.01, 2.0)
        self._ps_bgstep_spin.setValue(0.2)
        ps_bg_row2.addWidget(self._ps_bgstep_spin)
        psl.addLayout(ps_bg_row2)

        (self._ps_sieve_cb, self._ps_sieve2_w_spin, self._ps_sieve2_s_spin,
         self._ps_sieve3_w_spin, self._ps_sieve3_s_spin) = _add_sieve_controls(
            psl, "BG Threshold", 0.2, 0.05, 0.05, 0.01
        )

        ps_er_row = QHBoxLayout()
        ps_er_row.addWidget(QLabel("Signal Erosion min:"))
        self._ps_ermin_spin = QSpinBox()
        self._ps_ermin_spin.setRange(0, 15)
        self._ps_ermin_spin.setValue(0)
        ps_er_row.addWidget(self._ps_ermin_spin)
        psl.addLayout(ps_er_row)
        ps_er_row2 = QHBoxLayout()
        ps_er_row2.addWidget(QLabel("max:"))
        self._ps_ermax_spin = QSpinBox()
        self._ps_ermax_spin.setRange(0, 15)
        self._ps_ermax_spin.setValue(4)
        ps_er_row2.addWidget(self._ps_ermax_spin)
        ps_er_row2.addWidget(QLabel("step:"))
        self._ps_erstep_spin = QSpinBox()
        self._ps_erstep_spin.setRange(1, 15)
        self._ps_erstep_spin.setValue(1)
        ps_er_row2.addWidget(self._ps_erstep_spin)
        psl.addLayout(ps_er_row2)

        ps_cells_row = QHBoxLayout()
        ps_cells_row.addWidget(QLabel("Complex cells to test:"))
        self._ps_ncells_spin = QSpinBox()
        self._ps_ncells_spin.setRange(1, 50)
        self._ps_ncells_spin.setValue(5)
        ps_cells_row.addWidget(self._ps_ncells_spin)
        psl.addLayout(ps_cells_row)
        ps_cells_row2 = QHBoxLayout()
        ps_cells_row2.addWidget(QLabel("Pad Z:"))
        self._ps_padz_spin = QSpinBox()
        self._ps_padz_spin.setRange(0, 200)
        self._ps_padz_spin.setValue(15)
        ps_cells_row2.addWidget(self._ps_padz_spin)
        ps_cells_row2.addWidget(QLabel("Pad XY:"))
        self._ps_padxy_spin = QSpinBox()
        self._ps_padxy_spin.setRange(0, 500)
        self._ps_padxy_spin.setValue(40)
        ps_cells_row2.addWidget(self._ps_padxy_spin)
        psl.addLayout(ps_cells_row2)

        ps_scale_row = QHBoxLayout()
        ps_scale_row.addWidget(QLabel("Z (µm):"))
        self._ps_scalez_spin = QDoubleSpinBox()
        self._ps_scalez_spin.setDecimals(4)
        self._ps_scalez_spin.setRange(0.0001, 100.0)
        self._ps_scalez_spin.setValue(1.0)
        ps_scale_row.addWidget(self._ps_scalez_spin)
        ps_scale_row.addWidget(QLabel("XY (µm):"))
        self._ps_scalexy_spin = QDoubleSpinBox()
        self._ps_scalexy_spin.setDecimals(4)
        self._ps_scalexy_spin.setRange(0.0001, 100.0)
        self._ps_scalexy_spin.setValue(0.174)
        ps_scale_row.addWidget(self._ps_scalexy_spin)
        psl.addLayout(ps_scale_row)

        self._ps_is_gt_cb = _add_gt_checkbox(psl, "this sweep", visible=False)

        ps_btn_row = QHBoxLayout()
        self._ps_run_btn = QPushButton("Run BG Threshold/Erosion Sweep")
        self._ps_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        ps_btn_row.addWidget(self._ps_run_btn)
        self._ps_stop_btn = QPushButton("Stop Sweep")
        self._ps_stop_btn.setEnabled(False)
        ps_btn_row.addWidget(self._ps_stop_btn)
        psl.addLayout(ps_btn_row)

        self._ps_status_lbl = QLabel("")
        self._ps_status_lbl.setWordWrap(True)
        psl.addWidget(self._ps_status_lbl)

        self._ps_report_view = QTextEdit()
        self._ps_report_view.setReadOnly(True)
        self._ps_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._ps_report_view.setFixedHeight(160)
        psl.addWidget(self._ps_report_view)

        psg.setLayout(psl)
        psg = _make_collapsible(psg)
        t5.addWidget(psg)
        self._t5_category_groups.setdefault("pixel", []).append(psg)

        self._pixel_sweep_job = {"thread": None, "cancel_event": None, "timer": None}

        # ── Verify Smooth sigma XY / Z (GT Sweep) ─────────────────────── #
        sgg = QGroupBox("Verify Smooth σ XY / σ Z (GT Sweep)")
        sgl = QVBoxLayout()
        sgl.setSpacing(6)

        sg_note = QLabel(
            "Sweeps Smooth sigma XY x sigma Z (the Pixel Classifier's "
            "pre-threshold Gaussian smoothing) against the N most complex "
            "cells in a ground-truth-annotated fish, holding BG Threshold "
            "and Signal Erosion fixed at Tab 1's current values. These sigma "
            "values have never been swept before in this project -- "
            "1.5/3.0 has been a guessed starting point since Tab 2 was "
            "first built, never verified against real GT."
        )
        sg_note.setWordWrap(True)
        sg_note.setStyleSheet("color: #888; font-size: 10px;")
        sgl.addWidget(sg_note)

        sg_img_row = QHBoxLayout()
        sg_img_row.addWidget(QLabel("GT image:"))
        self._sg_img_edit = QLineEdit("")
        sg_img_row.addWidget(self._sg_img_edit)
        self._sg_img_browse_btn = QPushButton("...")
        self._sg_img_browse_btn.setFixedWidth(32)
        sg_img_row.addWidget(self._sg_img_browse_btn)
        sgl.addLayout(sg_img_row)
        sg_img_note = QLabel(
            "  Raw image or _ExtRm — both give identical results here, "
            "since this tool only ever reads pixels inside the brain mask. "
            "Do not use _NoBG/_RndFill — those already had a different "
            "background step applied."
        )
        sg_img_note.setStyleSheet("color: #aaa; font-size: 10px;")
        sg_img_note.setWordWrap(True)
        sgl.addWidget(sg_img_note)

        sg_mask_row = QHBoxLayout()
        sg_mask_row.addWidget(QLabel("brain_mask.tif:"))
        self._sg_mask_edit = QLineEdit("")
        sg_mask_row.addWidget(self._sg_mask_edit)
        self._sg_mask_browse_btn = QPushButton("...")
        self._sg_mask_browse_btn.setFixedWidth(32)
        sg_mask_row.addWidget(self._sg_mask_browse_btn)
        sgl.addLayout(sg_mask_row)

        sg_lbl_row = QHBoxLayout()
        sg_lbl_row.addWidget(QLabel("GT labels:"))
        self._sg_lbl_edit = QLineEdit("")
        sg_lbl_row.addWidget(self._sg_lbl_edit)
        self._sg_lbl_browse_btn = QPushButton("...")
        self._sg_lbl_browse_btn.setFixedWidth(32)
        sg_lbl_row.addWidget(self._sg_lbl_browse_btn)
        sgl.addLayout(sg_lbl_row)
        sg_gt_note = QLabel(
            "  brain_mask.tif = the RAW (un-eroded) mask Tab 1 saves — "
            "same fish as GT image/labels. GT labels = a hand-corrected "
            "microglia instance-label volume, typically named "
            "_GROUND_TRUTH.tif, one integer ID per cell — not the brain "
            "mask. BG Threshold/Erosion are read from Tab 1's current "
            "sliders when you click below, not swept."
        )
        sg_gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        sg_gt_note.setWordWrap(True)
        sgl.addWidget(sg_gt_note)

        sg_scale_row = QHBoxLayout()
        sg_scale_row.addWidget(QLabel("Z (µm):"))
        self._sg_scalez_spin = QDoubleSpinBox()
        self._sg_scalez_spin.setDecimals(4)
        self._sg_scalez_spin.setRange(0.0001, 100.0)
        self._sg_scalez_spin.setValue(1.0)
        sg_scale_row.addWidget(self._sg_scalez_spin)
        sg_scale_row.addWidget(QLabel("XY (µm):"))
        self._sg_scalexy_spin = QDoubleSpinBox()
        self._sg_scalexy_spin.setDecimals(4)
        self._sg_scalexy_spin.setRange(0.0001, 100.0)
        self._sg_scalexy_spin.setValue(0.174)
        sg_scale_row.addWidget(self._sg_scalexy_spin)
        sgl.addLayout(sg_scale_row)

        sg_sxy_row = QHBoxLayout()
        sg_sxy_row.addWidget(QLabel("sigma XY min:"))
        self._sg_sxymin_spin = QDoubleSpinBox()
        self._sg_sxymin_spin.setDecimals(1)
        self._sg_sxymin_spin.setRange(0.0, 5.0)
        self._sg_sxymin_spin.setValue(0.5)
        sg_sxy_row.addWidget(self._sg_sxymin_spin)
        sgl.addLayout(sg_sxy_row)
        sg_sxy_row2 = QHBoxLayout()
        sg_sxy_row2.addWidget(QLabel("max:"))
        self._sg_sxymax_spin = QDoubleSpinBox()
        self._sg_sxymax_spin.setDecimals(1)
        self._sg_sxymax_spin.setRange(0.0, 5.0)
        self._sg_sxymax_spin.setValue(2.5)
        sg_sxy_row2.addWidget(self._sg_sxymax_spin)
        sg_sxy_row2.addWidget(QLabel("step:"))
        self._sg_sxystep_spin = QDoubleSpinBox()
        self._sg_sxystep_spin.setDecimals(1)
        self._sg_sxystep_spin.setRange(0.1, 5.0)
        self._sg_sxystep_spin.setValue(0.5)
        sg_sxy_row2.addWidget(self._sg_sxystep_spin)
        sgl.addLayout(sg_sxy_row2)

        (self._sg_sieve_cb, self._sg_sieve2_w_spin, self._sg_sieve2_s_spin,
         self._sg_sieve3_w_spin, self._sg_sieve3_s_spin) = _add_sieve_controls(
            sgl, "sigma XY", 0.5, 0.1, 0.2, 0.05
        )

        sg_sz_row = QHBoxLayout()
        sg_sz_row.addWidget(QLabel("sigma Z min:"))
        self._sg_szmin_spin = QDoubleSpinBox()
        self._sg_szmin_spin.setDecimals(1)
        self._sg_szmin_spin.setRange(0.0, 5.0)
        self._sg_szmin_spin.setValue(1.0)
        sg_sz_row.addWidget(self._sg_szmin_spin)
        sgl.addLayout(sg_sz_row)
        sg_sz_row2 = QHBoxLayout()
        sg_sz_row2.addWidget(QLabel("max:"))
        self._sg_szmax_spin = QDoubleSpinBox()
        self._sg_szmax_spin.setDecimals(1)
        self._sg_szmax_spin.setRange(0.0, 5.0)
        self._sg_szmax_spin.setValue(5.0)
        sg_sz_row2.addWidget(self._sg_szmax_spin)
        sg_sz_row2.addWidget(QLabel("step:"))
        self._sg_szstep_spin = QDoubleSpinBox()
        self._sg_szstep_spin.setDecimals(1)
        self._sg_szstep_spin.setRange(0.1, 5.0)
        self._sg_szstep_spin.setValue(1.0)
        sg_sz_row2.addWidget(self._sg_szstep_spin)
        sgl.addLayout(sg_sz_row2)

        sg_cells_row = QHBoxLayout()
        sg_cells_row.addWidget(QLabel("Complex cells to test:"))
        self._sg_ncells_spin = QSpinBox()
        self._sg_ncells_spin.setRange(1, 50)
        self._sg_ncells_spin.setValue(5)
        sg_cells_row.addWidget(self._sg_ncells_spin)
        sgl.addLayout(sg_cells_row)
        sg_cells_row2 = QHBoxLayout()
        sg_cells_row2.addWidget(QLabel("Pad Z:"))
        self._sg_padz_spin = QSpinBox()
        self._sg_padz_spin.setRange(0, 200)
        self._sg_padz_spin.setValue(15)
        sg_cells_row2.addWidget(self._sg_padz_spin)
        sg_cells_row2.addWidget(QLabel("Pad XY:"))
        self._sg_padxy_spin = QSpinBox()
        self._sg_padxy_spin.setRange(0, 500)
        self._sg_padxy_spin.setValue(40)
        sg_cells_row2.addWidget(self._sg_padxy_spin)
        sgl.addLayout(sg_cells_row2)

        self._sg_is_gt_cb = _add_gt_checkbox(sgl, "this sweep", visible=False)

        sg_btn_row = QHBoxLayout()
        self._sg_run_btn = QPushButton("Run Sigma Sweep")
        self._sg_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        sg_btn_row.addWidget(self._sg_run_btn)
        self._sg_stop_btn = QPushButton("Stop Sweep")
        self._sg_stop_btn.setEnabled(False)
        sg_btn_row.addWidget(self._sg_stop_btn)
        sgl.addLayout(sg_btn_row)

        self._sg_status_lbl = QLabel("Status: Ready")
        self._sg_status_lbl.setWordWrap(True)
        sgl.addWidget(self._sg_status_lbl)

        self._sg_report_view = QTextEdit()
        self._sg_report_view.setReadOnly(True)
        self._sg_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._sg_report_view.setFixedHeight(160)
        sgl.addWidget(self._sg_report_view)

        sgg.setLayout(sgl)
        sgg = _make_collapsible(sgg)
        t5.addWidget(sgg)
        self._t5_category_groups.setdefault("pixel", []).append(sgg)

        self._sigma_sweep_job = {"thread": None, "cancel_event": None, "timer": None}

        # ── Cellpose-SAM segmentation (do_3D + Krendl corrections) — shown for _ExtRm layers ── #
        self._cellpose_group = QGroupBox("Cellpose-SAM Segmentation")
        cpg = QVBoxLayout()
        cpg.setSpacing(6)

        cp_note = QLabel(
            "  The recommended labelling method: a fine-tuned Cellpose-SAM "
            "foundation model, run in 3D (do_3D inference), then cleaned up "
            "with a 3-component-GMM pass, a Krendl safe-merge pass, and a "
            "large-contact merge pass — handles branching/overlapping cells "
            "far better than classical thresholding.\n\n"
            "  Select a brain_only layer, pick a Cellpose-SAM checkpoint,\n"
            "  then click below. do_3D inference is slow (can be hours\n"
            "  for a full fish) — this runs in the background."
        )
        cp_note.setWordWrap(True)
        cp_note.setStyleSheet("color: #aaa; font-size: 10px;")
        cpg.addWidget(cp_note)

        cp_model_row = QHBoxLayout()
        self._cp_model_lbl = QLabel(
            str(self._state["cellpose_model_path"]) if self._state["cellpose_model_path"] else "— no model selected —"
        )
        self._cp_model_lbl.setWordWrap(True)
        self._cp_model_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._cp_model_browse_btn = QPushButton("...")
        self._cp_model_browse_btn.setFixedWidth(32)
        cp_model_row.addWidget(self._cp_model_lbl)
        cp_model_row.addWidget(self._cp_model_browse_btn)
        cpg.addLayout(cp_model_row)

        cpminsize_row = QHBoxLayout()
        cpminsize_row.addWidget(QLabel("Min size (vox):"))
        self._cp_minsize_spin = QSpinBox()
        self._cp_minsize_spin.setRange(1, 5000)
        self._cp_minsize_spin.setValue(_root_cfg.get("cellpose_min_size", 15))
        cpminsize_row.addWidget(self._cp_minsize_spin)
        cpg.addLayout(cpminsize_row)
        cpminsize_note = QLabel(
            "  Not shared with the Pixel Classifier's Min volume above --"
            " deliberately a different, much smaller value: this only "
            "discards genuine prediction noise before GMM cleanup and "
            "Krendl safe-merge run — those stages, not this field, decide "
            "whether a small real fragment gets reattached or discarded "
            "as debris. Raising this toward Min volume's range would "
            "discard fragments before they ever reach that decision."
        )
        cpminsize_note.setStyleSheet("color: #888; font-size: 10px;")
        cpminsize_note.setWordWrap(True)
        cpg.addWidget(cpminsize_note)

        cpniter_row = QHBoxLayout()
        cpniter_row.addWidget(QLabel("Flow iterations (niter):"))
        self._cp_niter_spin = QSpinBox()
        self._cp_niter_spin.setRange(50, 2000)
        self._cp_niter_spin.setSingleStep(50)
        self._cp_niter_spin.setValue(_root_cfg.get("cellpose_niter", 200))
        cpniter_row.addWidget(self._cp_niter_spin)
        cpg.addLayout(cpniter_row)
        cpniter_note = QLabel(
            "  Euler-integration steps each voxel's flow trajectory gets "
            "before instances are formed. 200 is Cellpose's own default. "
            "If a cell comes out looking porous/\"pumice stone\" in 3D "
            "(parallel banding per 2D slice) -- see the Solidity warning "
            "below after a run -- try raising Cellprob first (e.g. "
            "toward -0.3): a faint cell's raw probability map is noisy "
            "near a permissive threshold, letting marginal, unstable "
            "voxels flicker in and out slice-by-slice before flow-"
            "following even starts, which is the actual usual cause. "
            "Raising Flow iterations alone rarely fixes it by itself "
            "(confirmed empirically, 2026-08) -- worth trying together "
            "with Cellprob on a Re-run This Cell Only pass, not instead "
            "of it."
        )
        cpniter_note.setStyleSheet("color: #888; font-size: 10px;")
        cpniter_note.setWordWrap(True)
        cpg.addWidget(cpniter_note)

        cp_cellprob_row = QHBoxLayout()
        cp_cellprob_row.addWidget(QLabel("Cellprob threshold:"))
        self._cp_cellprob_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._cp_cellprob_slider.setDecimals(2)
        self._cp_cellprob_slider.setMinimum(-6.0)
        self._cp_cellprob_slider.setMaximum(6.0)
        self._cp_cellprob_slider.setSingleStep(0.1)
        self._cp_cellprob_slider.setValue(_root_cfg.get("cellpose_cellprob", -2.5))
        cp_cellprob_row.addWidget(self._cp_cellprob_slider)
        self._cp_cellprob_spin = _add_reliable_spinbox(
            cp_cellprob_row, self._cp_cellprob_slider, -6.0, 6.0, 0.1, decimals=2
        )
        cpg.addLayout(cp_cellprob_row)
        self._cp_cellprob_recommended_lbl = _add_recommended_label(
            cpg, _root_cfg.get("cellpose_cellprob_recommended"), noun="Cellprob threshold"
        )

        # Flow threshold deliberately has no UI control: Cellpose only
        # applies its QC filter in 2D/stitch mode, never under do_3D (this
        # pipeline's only mode) -- see _FLOW_THRESHOLD_FIXED above. A live,
        # editable slider that silently did nothing was a real user-facing
        # trap and was removed rather than merely documented.

        cp_maxgap_row = QHBoxLayout()
        cp_maxgap_row.addWidget(QLabel("Safe-merge max gap (µm):"))
        self._cp_maxgap_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._cp_maxgap_slider.setDecimals(2)
        self._cp_maxgap_slider.setMinimum(0.0)
        self._cp_maxgap_slider.setMaximum(5.0)
        self._cp_maxgap_slider.setSingleStep(0.05)
        self._cp_maxgap_slider.setValue(_root_cfg.get("cellpose_max_gap_um", 1.0))
        cp_maxgap_row.addWidget(self._cp_maxgap_slider)
        self._cp_maxgap_spin = _add_reliable_spinbox(
            cp_maxgap_row, self._cp_maxgap_slider, 0.0, 5.0, 0.05, decimals=2
        )
        cpg.addLayout(cp_maxgap_row)
        cp_maxgap_note = QLabel(
            "  Physical distance now (not voxels) -- previously this "
            "slider's value was compared against a distance transform "
            "computed with no voxel-scale correction, so \"2\" meant "
            "2.0µm along Z but only 0.35µm in-plane (Z=1.0µm/vox, "
            "XY=0.174µm/vox), a ~5.7x inconsistency purely from which "
            "direction a fragment happened to break. Old voxel-based "
            "values don't carry over -- needs recalibrating."
        )
        cp_maxgap_note.setWordWrap(True)
        cp_maxgap_note.setStyleSheet("color: #888; font-size: 10px;")
        cpg.addWidget(cp_maxgap_note)
        self._cp_maxgap_recommended_lbl = _add_recommended_label(
            cpg, _root_cfg.get("cellpose_max_gap_um_recommended"), unit="µm", noun="Safe-merge max gap"
        )

        cp_mincontact_row = QHBoxLayout()
        cp_mincontact_row.addWidget(QLabel("Safe-merge min contact (vox):"))
        self._cp_mincontact_slider = QLabeledSlider(Qt.Horizontal)
        self._cp_mincontact_slider.setMinimum(0)
        self._cp_mincontact_slider.setMaximum(200)
        self._cp_mincontact_slider.setValue(_root_cfg.get("cellpose_min_contact_vox", 10))
        cp_mincontact_row.addWidget(self._cp_mincontact_slider)
        self._cp_mincontact_spin = _add_reliable_spinbox(
            cp_mincontact_row, self._cp_mincontact_slider, 0, 200, 1
        )
        cpg.addLayout(cp_mincontact_row)
        self._cp_mincontact_recommended_lbl = _add_recommended_label(
            cpg, _root_cfg.get("cellpose_min_contact_vox_recommended"), unit=" vox", noun="Safe-merge min contact"
        )

        cp_gtmin_note = QLabel(
            "Safe-merge's \"already a whole cell\" floor is no longer a "
            "separate field here -- it now reads the shared Min volume "
            "field in Common Settings above, since both are the exact "
            "same measurement (smallest true GT cell volume). Recalibrated "
            "automatically from real GT statistics by the BG Threshold/"
            "Erosion sweep, the Sigma sweep, the Cellprob/Large-contact "
            "sweep below, or Tab 3 Statistics when marked as verified GT."
        )
        cp_gtmin_note.setWordWrap(True)
        cp_gtmin_note.setStyleSheet("color: #888; font-size: 10px;")
        cpg.addWidget(cp_gtmin_note)

        cp_largecontact_row = QHBoxLayout()
        cp_largecontact_row.addWidget(QLabel("Large-contact merge (vox):"))
        self._cp_largecontact_slider = QLabeledSlider(Qt.Horizontal)
        self._cp_largecontact_slider.setMinimum(1)
        self._cp_largecontact_slider.setMaximum(2000)
        self._cp_largecontact_slider.setValue(_root_cfg.get("cellpose_large_contact", 20))
        cp_largecontact_row.addWidget(self._cp_largecontact_slider)
        self._cp_largecontact_spin = _add_reliable_spinbox(
            cp_largecontact_row, self._cp_largecontact_slider, 1, 2000, 10
        )
        cpg.addLayout(cp_largecontact_row)
        self._cp_largecontact_recommended_lbl = _add_recommended_label(
            cpg, _root_cfg.get("cellpose_large_contact_recommended"), unit=" vox", noun="Large-contact merge"
        )

        self._cp_run_notify_cb = _make_notify_checkbox()
        cpg.addWidget(self._cp_run_notify_cb)

        self._cp_run_btn = QPushButton("Run Cellpose-SAM Segmentation")
        self._cp_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        cpg.addWidget(self._cp_run_btn)

        self._cp_status_lbl = QLabel("")
        self._cp_status_lbl.setWordWrap(True)
        cpg.addWidget(self._cp_status_lbl)

        # Live output -- do_3D's own progress is entirely logging-based
        # (cellpose attaches a NullHandler to its own logger by default,
        # so it's normally invisible even in a terminal unless something
        # calls cellpose.io.logger_setup()); forwarded here instead, see
        # _live_progress.py.
        self._cp_log_view = QTextEdit()
        self._cp_log_view.setReadOnly(True)
        self._cp_log_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._cp_log_view.setFixedHeight(120)
        cpg.addWidget(self._cp_log_view)

        cpg.addWidget(_sep())
        cp_relabel_note = QLabel(
            "Re-run one cell instead of the whole fish -- crops to that "
            "label's own bounding box (+ padding), re-runs do_3D + the "
            "same GMM/Krendl/large-contact/final-min-size cleanup on "
            "just the crop (using this section's current settings, "
            "including Flow iterations above), then splices the result "
            "back in place of the old label. Only pieces that actually "
            "overlap the original label survive the splice -- a "
            "neighbouring cell caught by the padding is discarded, not "
            "duplicated. If the fix reveals more than one real cell, "
            "all of them are kept as new label IDs. Requires the "
            "matching '<image>_cellpose_labels' layer to already exist "
            "(run the full segmentation once first)."
        )
        cp_relabel_note.setWordWrap(True)
        cp_relabel_note.setStyleSheet("color: #888; font-size: 10px;")
        cpg.addWidget(cp_relabel_note)

        cp_relabel_row = QHBoxLayout()
        cp_relabel_row.addWidget(QLabel("Label ID to re-run:"))
        self._cp_relabel_id_spin = QSpinBox()
        self._cp_relabel_id_spin.setRange(1, 1_000_000)
        cp_relabel_row.addWidget(self._cp_relabel_id_spin)
        self._cp_relabel_use_sel_btn = QPushButton("Use selected")
        self._cp_relabel_use_sel_btn.setFixedWidth(90)
        cp_relabel_row.addWidget(self._cp_relabel_use_sel_btn)
        cpg.addLayout(cp_relabel_row)

        self._cp_relabel_btn = QPushButton("Re-run This Cell Only")
        cpg.addWidget(self._cp_relabel_btn)

        self._cellpose_group.setLayout(cpg)
        self._cellpose_group = _make_collapsible(self._cellpose_group)
        t2.addWidget(self._cellpose_group)

        # ── Calibrate Cellprob + Krendl Merge Parameters (GT sweep) ──────── #
        # always visible, not gated by the active-layer auto-switch: works
        # from explicit file paths.
        #
        # Deliberately does NOT sweep Large-contact (or max_gap/min_contact)
        # alongside Cellprob -- scoring Cellprob against the FULLY corrected
        # result (post GMM/Krendl/large-contact) entangles "best Cellprob"
        # with whatever those merge parameters happen to be set to, circular
        # if they haven't themselves been calibrated yet. Instead, one
        # button does the whole correct-order chain in sequence, mirroring
        # exactly what the 3-fish production validation run (2026-08-18)
        # did by hand: (1) find the best Cellprob on raw voxel-level signal
        # quality alone (binarized foreground vs. GT, instance identity
        # ignored -- mirrors Tab 1's Verify MONAI Threshold/Erosion doing
        # the identical thing for the brain mask), (2) generate raw cp_masks
        # AT that winning Cellprob (reusing the same cached do_3D flows --
        # zero extra network passes), (3) compare those cp_masks against GT
        # to calibrate max_gap/min_contact (measure_merge_params_from_
        # prediction()/recommend_merge_params()) -- GT labels which raw
        # fragments are/aren't the same real cell, so the calibration comes
        # from the network's own real fragmentation behaviour, not GT
        # geometry alone (which has a real ambiguity: is the smallest gap
        # between two GT cells actual biology, or just how the boundary
        # happened to be drawn?).
        krg = QGroupBox("Calibrate Cellprob + Krendl Merge Parameters (GT Sweep)")
        krl = QVBoxLayout()
        krl.setSpacing(6)

        kr_note = QLabel(
            "One button, three steps in the correct order: (1) sweeps "
            "Cellprob against GT on raw voxel-level Dice/IoU (no GMM/"
            "Krendl/large-contact run at all here -- avoids entangling "
            "the Cellprob pick with not-yet-calibrated merge parameters); "
            "(2) forms raw cp_masks at the winning Cellprob, reusing the "
            "same cached do_3D flows from step 1 -- no extra network pass; "
            "(3) compares those cp_masks against GT to jointly calibrate "
            "max_gap/min_contact against Krendl safe-merge's real OR-"
            "combined rule (gap<=max_gap OR contact>=min_contact), "
            "minimizing total corrections (missed + false merges) rather "
            "than avoiding one error type at any cost -- an over-merge is "
            "fixable with Split Label, an under-merge with Join Labels. "
            "do_3D's network pass runs exactly ONCE for the whole thing "
            "(~3h on a full-size fish). Does NOT calibrate large_contact "
            "(a separate, later merge stage operating on already-Krendl-"
            "corrected data, not raw cp_masks) -- needs its own "
            "measurement, not built yet."
        )
        kr_note.setWordWrap(True)
        kr_note.setStyleSheet("color: #888; font-size: 10px;")
        krl.addWidget(kr_note)

        kr_img_row = QHBoxLayout()
        kr_img_row.addWidget(QLabel("Image:"))
        self._kr_img_edit = QLineEdit("")
        kr_img_row.addWidget(self._kr_img_edit)
        self._kr_img_browse_btn = QPushButton("...")
        self._kr_img_browse_btn.setFixedWidth(32)
        kr_img_row.addWidget(self._kr_img_browse_btn)
        krl.addLayout(kr_img_row)

        kr_gt_row = QHBoxLayout()
        kr_gt_row.addWidget(QLabel("GT labels:"))
        self._kr_gt_edit = QLineEdit("")
        kr_gt_row.addWidget(self._kr_gt_edit)
        self._kr_gt_browse_btn = QPushButton("...")
        self._kr_gt_browse_btn.setFixedWidth(32)
        kr_gt_row.addWidget(self._kr_gt_browse_btn)
        krl.addLayout(kr_gt_row)
        kr_gt_note = QLabel(
            "  Image = a full-fish _ExtRm brain_only volume (the Cellpose-"
            "SAM route's input — not _NoBG/_RndFill, and not the raw "
            "pre-Tab-1 image). GT labels = the corresponding hand-"
            "corrected microglia instance-label volume, typically named "
            "_GROUND_TRUTH.tif, one integer ID per cell — not a brain mask."
        )
        kr_gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        kr_gt_note.setWordWrap(True)
        krl.addWidget(kr_gt_note)

        kr_scale_row = QHBoxLayout()
        kr_scale_row.addWidget(QLabel("Z (µm):"))
        self._kr_scalez_spin = QDoubleSpinBox()
        self._kr_scalez_spin.setDecimals(4)
        self._kr_scalez_spin.setRange(0.0001, 100.0)
        self._kr_scalez_spin.setValue(1.0)
        kr_scale_row.addWidget(self._kr_scalez_spin)
        kr_scale_row.addWidget(QLabel("XY (µm):"))
        self._kr_scalexy_spin = QDoubleSpinBox()
        self._kr_scalexy_spin.setDecimals(4)
        self._kr_scalexy_spin.setRange(0.0001, 100.0)
        self._kr_scalexy_spin.setValue(0.174)
        kr_scale_row.addWidget(self._kr_scalexy_spin)
        krl.addLayout(kr_scale_row)

        kr_cp_row = QHBoxLayout()
        kr_cp_row.addWidget(QLabel("Cellprob min:"))
        self._kr_cpmin_spin = QDoubleSpinBox()
        self._kr_cpmin_spin.setDecimals(2)
        self._kr_cpmin_spin.setRange(-6.0, 6.0)
        self._kr_cpmin_spin.setValue(-2.0)
        kr_cp_row.addWidget(self._kr_cpmin_spin)
        krl.addLayout(kr_cp_row)
        kr_cp_row2 = QHBoxLayout()
        kr_cp_row2.addWidget(QLabel("max:"))
        self._kr_cpmax_spin = QDoubleSpinBox()
        self._kr_cpmax_spin.setDecimals(2)
        self._kr_cpmax_spin.setRange(-6.0, 6.0)
        self._kr_cpmax_spin.setValue(2.0)
        kr_cp_row2.addWidget(self._kr_cpmax_spin)
        kr_cp_row2.addWidget(QLabel("step:"))
        self._kr_cpstep_spin = QDoubleSpinBox()
        self._kr_cpstep_spin.setDecimals(2)
        self._kr_cpstep_spin.setRange(0.05, 6.0)
        self._kr_cpstep_spin.setValue(0.25)
        kr_cp_row2.addWidget(self._kr_cpstep_spin)
        krl.addLayout(kr_cp_row2)

        (self._kr_sieve_cb, self._kr_sieve2_w_spin, self._kr_sieve2_s_spin,
         self._kr_sieve3_w_spin, self._kr_sieve3_s_spin) = _add_sieve_controls(
            krl, "Cellprob", 1.0, 0.25, 0.5, 0.1
        )

        self._kr_notify_cb = _make_notify_checkbox()
        krl.addWidget(self._kr_notify_cb)

        self._kr_is_gt_cb = _add_gt_checkbox(krl, "this sweep", visible=False)

        kr_btn_row = QHBoxLayout()
        self._kr_run_btn = QPushButton("Run Full Calibration")
        self._kr_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        kr_btn_row.addWidget(self._kr_run_btn)
        self._kr_stop_btn = QPushButton("Stop Sweep")
        self._kr_stop_btn.setEnabled(False)
        kr_btn_row.addWidget(self._kr_stop_btn)
        krl.addLayout(kr_btn_row)

        self._kr_status_lbl = QLabel("")
        self._kr_status_lbl.setWordWrap(True)
        krl.addWidget(self._kr_status_lbl)

        self._kr_report_view = QTextEdit()
        self._kr_report_view.setReadOnly(True)
        self._kr_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._kr_report_view.setFixedHeight(160)
        krl.addWidget(self._kr_report_view)

        krg.setLayout(krl)
        krg = _make_collapsible(krg)
        t5.addWidget(krg)
        self._t5_category_groups.setdefault("cellpose", []).append(krg)

        self._krendl_sweep_job = {"thread": None, "cancel_event": None, "timer": None}

        # ── Build GT-Correction Package — always visible, no GPU needed ── #
        # Packages a Krendl segmentation result for external manual
        # correction, matching the exact file layout this project has
        # produced by hand for every fish sent out (D1F1, D1F2, D1F4x2):
        # GROUND_TRUTH_CREATION_GUIDE.md + masks_corrected.tif (the Krendl
        # output, correction starting point) + cp_masks_3D.tif (raw,
        # reference) + a per-cell statistics CSV + the source image,
        # zipped together.
        gtpg = QGroupBox("Build GT-Correction Package")
        gtpl = QVBoxLayout()
        gtpl.setSpacing(6)

        gtp_note = QLabel(
            "Packages a Krendl segmentation result for external manual "
            "correction — same layout this project has hand-assembled for "
            "every fish sent out so far. The corrected result becomes future "
            "training/GT data."
        )
        gtp_note.setWordWrap(True)
        gtp_note.setStyleSheet("color: #888; font-size: 10px;")
        gtpl.addWidget(gtp_note)

        gtp_stem_row = QHBoxLayout()
        gtp_stem_row.addWidget(QLabel("Fish stem:"))
        self._gtp_stem_edit = QLineEdit("")
        gtp_stem_row.addWidget(self._gtp_stem_edit)
        gtpl.addLayout(gtp_stem_row)
        gtp_stem_note = QLabel("  Used to name every file in the package, e.g. NT39-3dpf-D1F4_2024-...")
        gtp_stem_note.setStyleSheet("color: #aaa; font-size: 10px;")
        gtp_stem_note.setWordWrap(True)
        gtpl.addWidget(gtp_stem_note)

        gtp_img_row = QHBoxLayout()
        gtp_img_row.addWidget(QLabel("Source image:"))
        self._gtp_img_edit = QLineEdit("")
        gtp_img_row.addWidget(self._gtp_img_edit)
        self._gtp_img_browse_btn = QPushButton("...")
        self._gtp_img_browse_btn.setFixedWidth(32)
        gtp_img_row.addWidget(self._gtp_img_browse_btn)
        gtpl.addLayout(gtp_img_row)

        gtp_masks_row = QHBoxLayout()
        gtp_masks_row.addWidget(QLabel("Krendl masks:"))
        self._gtp_masks_edit = QLineEdit("")
        gtp_masks_row.addWidget(self._gtp_masks_edit)
        self._gtp_masks_browse_btn = QPushButton("...")
        self._gtp_masks_browse_btn.setFixedWidth(32)
        gtp_masks_row.addWidget(self._gtp_masks_browse_btn)
        gtpl.addLayout(gtp_masks_row)
        gtp_masks_note = QLabel("  The Run Cellpose-SAM Segmentation output — becomes masks_corrected.tif (\"start here\").")
        gtp_masks_note.setStyleSheet("color: #aaa; font-size: 10px;")
        gtp_masks_note.setWordWrap(True)
        gtpl.addWidget(gtp_masks_note)

        gtp_raw_row = QHBoxLayout()
        gtp_raw_row.addWidget(QLabel("Raw Cellpose masks (optional):"))
        self._gtp_raw_edit = QLineEdit("")
        gtp_raw_row.addWidget(self._gtp_raw_edit)
        self._gtp_raw_browse_btn = QPushButton("...")
        self._gtp_raw_browse_btn.setFixedWidth(32)
        gtp_raw_row.addWidget(self._gtp_raw_browse_btn)
        gtpl.addLayout(gtp_raw_row)

        gtp_guide_row = QHBoxLayout()
        gtp_guide_row.addWidget(QLabel("Creation guide (optional override):"))
        self._gtp_guide_edit = QLineEdit("")
        self._gtp_guide_edit.setPlaceholderText(str(_gtp.DEFAULT_GT_GUIDE_PATH))
        gtp_guide_row.addWidget(self._gtp_guide_edit)
        self._gtp_guide_browse_btn = QPushButton("...")
        self._gtp_guide_browse_btn.setFixedWidth(32)
        gtp_guide_row.addWidget(self._gtp_guide_browse_btn)
        gtpl.addLayout(gtp_guide_row)

        gtp_out_row = QHBoxLayout()
        gtp_out_row.addWidget(QLabel("Output folder:"))
        self._gtp_out_edit = QLineEdit("")
        gtp_out_row.addWidget(self._gtp_out_edit)
        self._gtp_out_browse_btn = QPushButton("...")
        self._gtp_out_browse_btn.setFixedWidth(32)
        gtp_out_row.addWidget(self._gtp_out_browse_btn)
        gtpl.addLayout(gtp_out_row)

        self._gtp_run_btn = QPushButton("Build GT-Correction Package")
        self._gtp_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        gtpl.addWidget(self._gtp_run_btn)

        self._gtp_status_lbl = QLabel("")
        self._gtp_status_lbl.setWordWrap(True)
        gtpl.addWidget(self._gtp_status_lbl)

        gtpg.setLayout(gtpl)
        gtpg = _make_collapsible(gtpg)
        t5.addWidget(gtpg)
        self._t5_category_groups.setdefault("cellpose", []).append(gtpg)

        self._gt_package_job = {"thread": None, "timer": None}

        t2.addWidget(_sep())

        # ── Downstream label tools (Resort / Split / Save) — shown only
        #    when a label-creation option is applicable to the active layer ── #
        self._downstream_label_tools = QWidget()
        dlt = QVBoxLayout()
        dlt.setContentsMargins(0, 0, 0, 0)
        dlt.setSpacing(6)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort by:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("Size",       "size")
        self._sort_combo.addItem("Centroid Z", "centroid_z")
        self._sort_combo.addItem("Centroid Y", "centroid_y")
        self._sort_combo.addItem("Centroid X", "centroid_x")
        self._sort_combo.addItem("Complexity", "complexity")
        sort_row.addWidget(self._sort_combo)
        dlt.addLayout(sort_row)

        self._sort_reverse_cb = QCheckBox("Reverse order")
        dlt.addWidget(self._sort_reverse_cb)

        self._resort_btn = QPushButton("Resort Labels")
        self._resort_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._resort_btn)

        self._resort_status_lbl = QLabel("")
        self._resort_status_lbl.setWordWrap(True)
        dlt.addWidget(self._resort_status_lbl)

        dlt.addWidget(_sep())

        self._debris_btn = QPushButton("Remove Debris")
        self._debris_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._debris_btn)
        debris_note = QLabel(
            "  Removes any object below Final min-size fraction x Min "
            "volume (Common Settings, 6a) from the active Labels layer "
            "as it currently stands -- for debris a manual edit left "
            "behind (e.g. deleting a whole label that was actually skin, "
            "or splitting one) that never went through Create Labels' "
            "own volume filter, since that filter only ran once, before "
            "the edit. Works on labels from either route."
        )
        debris_note.setWordWrap(True)
        debris_note.setStyleSheet("color: #888; font-size: 10px;")
        dlt.addWidget(debris_note)
        self._debris_status_lbl = QLabel("")
        self._debris_status_lbl.setWordWrap(True)
        dlt.addWidget(self._debris_status_lbl)

        dlt.addWidget(_sep())

        split_lbl_row = QHBoxLayout()
        split_lbl_row.addWidget(QLabel("Target label:"))
        self._split_label_spin = QSpinBox()
        self._split_label_spin.setMinimum(1)
        self._split_label_spin.setMaximum(99999)
        self._split_label_spin.setValue(1)
        split_lbl_row.addWidget(self._split_label_spin)
        self._split_use_sel_btn = QPushButton("Use selected")
        self._split_use_sel_btn.setFixedWidth(90)
        split_lbl_row.addWidget(self._split_use_sel_btn)
        dlt.addLayout(split_lbl_row)

        split_mode_row = QHBoxLayout()
        split_mode_row.addWidget(QLabel("Split mode:"))
        self._split_mode_combo = QComboBox()
        self._split_mode_combo.addItem("3D (whole label)", "3d")
        self._split_mode_combo.addItem("2D (current slice only)", "2d")
        split_mode_row.addWidget(self._split_mode_combo)
        dlt.addLayout(split_mode_row)

        split_mode_note = QLabel(
            "  2D splits only the CURRENT slice, watersheding on the "
            "SIGNAL layer's raw intensity instead of the mask's shape -- "
            "seeds sit at local brightness peaks, the cut runs along the "
            "dimmest ridge between them. Use it when two things only "
            "touch on one cross-section (e.g. real signal grazing a "
            "skin-residue fragment right at that slice) with no real "
            "geometric neck for a 3D/shape-based split to find, since "
            "the mask can be wide right through a point the real signal "
            "already dips low at. Every other slice of this label is "
            "left completely untouched."
        )
        split_mode_note.setWordWrap(True)
        split_mode_note.setStyleSheet("color: #888; font-size: 10px;")
        dlt.addWidget(split_mode_note)

        split_signal_row = QHBoxLayout()
        split_signal_row.addWidget(QLabel("Signal layer (2D mode only):"))
        self._split_signal_combo = QComboBox()
        split_signal_row.addWidget(self._split_signal_combo)
        dlt.addLayout(split_signal_row)

        split_n_row = QHBoxLayout()
        split_n_row.addWidget(QLabel("Split into:"))
        self._split_n_spin = QSpinBox()
        self._split_n_spin.setMinimum(2)
        self._split_n_spin.setMaximum(10)
        self._split_n_spin.setValue(2)
        split_n_row.addWidget(self._split_n_spin)
        split_n_row.addWidget(QLabel("parts"))
        split_n_row.addStretch()
        dlt.addLayout(split_n_row)

        split_sigma_row = QHBoxLayout()
        split_sigma_row.addWidget(QLabel("Smooth σ:"))
        self._split_sigma_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._split_sigma_slider.setDecimals(1)
        self._split_sigma_slider.setMinimum(0.0)
        self._split_sigma_slider.setMaximum(3.0)
        self._split_sigma_slider.setSingleStep(0.1)
        self._split_sigma_slider.setValue(1.0)
        split_sigma_row.addWidget(self._split_sigma_slider)
        self._split_sigma_spin = _add_reliable_spinbox(
            split_sigma_row, self._split_sigma_slider, 0.0, 3.0, 0.1, decimals=1
        )
        dlt.addLayout(split_sigma_row)

        split_dist_row = QHBoxLayout()
        split_dist_row.addWidget(QLabel("Min distance:"))
        self._split_dist_slider = QLabeledSlider(Qt.Horizontal)
        self._split_dist_slider.setMinimum(1)
        self._split_dist_slider.setMaximum(30)
        self._split_dist_slider.setValue(5)
        split_dist_row.addWidget(self._split_dist_slider)
        self._split_dist_spin = _add_reliable_spinbox(
            split_dist_row, self._split_dist_slider, 1, 30, 1
        )
        dlt.addLayout(split_dist_row)

        self._split_btn = QPushButton("Split Label")
        self._split_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._split_btn)

        self._split_status_lbl = QLabel("")
        self._split_status_lbl.setWordWrap(True)
        dlt.addWidget(self._split_status_lbl)

        dlt.addWidget(_sep())

        join_note = QLabel(
            "  Join Labels — the inverse of Split: merges Label B into "
            "Label A when one cell got wrongly cut into two pieces "
            "(e.g. a thin process fooled the segmenter into treating it "
            "as a neck). Label A's ID survives; Label B disappears."
        )
        join_note.setWordWrap(True)
        join_note.setStyleSheet("color: #888; font-size: 10px;")
        dlt.addWidget(join_note)

        join_a_row = QHBoxLayout()
        join_a_row.addWidget(QLabel("Label A (keep):"))
        self._join_a_spin = QSpinBox()
        self._join_a_spin.setMinimum(1)
        self._join_a_spin.setMaximum(99999)
        self._join_a_spin.setValue(1)
        join_a_row.addWidget(self._join_a_spin)
        self._join_a_use_sel_btn = QPushButton("Use selected")
        self._join_a_use_sel_btn.setFixedWidth(90)
        join_a_row.addWidget(self._join_a_use_sel_btn)
        dlt.addLayout(join_a_row)

        join_b_row = QHBoxLayout()
        join_b_row.addWidget(QLabel("Label B (merge into A):"))
        self._join_b_spin = QSpinBox()
        self._join_b_spin.setMinimum(1)
        self._join_b_spin.setMaximum(99999)
        self._join_b_spin.setValue(2)
        join_b_row.addWidget(self._join_b_spin)
        self._join_b_use_sel_btn = QPushButton("Use selected")
        self._join_b_use_sel_btn.setFixedWidth(90)
        join_b_row.addWidget(self._join_b_use_sel_btn)
        dlt.addLayout(join_b_row)

        self._join_btn = QPushButton("Join Labels")
        self._join_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._join_btn)

        self._join_status_lbl = QLabel("")
        self._join_status_lbl.setWordWrap(True)
        dlt.addWidget(self._join_status_lbl)

        dlt.addWidget(_sep())

        correct_note = QLabel(
            "  Correct Label — regenerates one label's shape on the "
            "CURRENT slice only, from the signal layer's own current "
            "contrast limits (whatever intensity window it's displayed "
            "with right now). A neighboring label's own pixels are "
            "never touched, even if they fall inside the padded box."
        )
        correct_note.setWordWrap(True)
        correct_note.setStyleSheet("color: #888; font-size: 10px;")
        dlt.addWidget(correct_note)

        correct_signal_row = QHBoxLayout()
        correct_signal_row.addWidget(QLabel("Signal layer:"))
        self._correct_signal_combo = QComboBox()
        correct_signal_row.addWidget(self._correct_signal_combo)
        dlt.addLayout(correct_signal_row)

        correct_label_row = QHBoxLayout()
        correct_label_row.addWidget(QLabel("Label to correct:"))
        self._correct_label_spin = QSpinBox()
        self._correct_label_spin.setMinimum(1)
        self._correct_label_spin.setMaximum(99999)
        self._correct_label_spin.setValue(1)
        correct_label_row.addWidget(self._correct_label_spin)
        self._correct_use_sel_btn = QPushButton("Use selected")
        self._correct_use_sel_btn.setFixedWidth(90)
        correct_label_row.addWidget(self._correct_use_sel_btn)
        dlt.addLayout(correct_label_row)

        correct_pad_row = QHBoxLayout()
        correct_pad_row.addWidget(QLabel("Bbox padding (px):"))
        self._correct_pad_spin = QSpinBox()
        self._correct_pad_spin.setMinimum(0)
        self._correct_pad_spin.setMaximum(500)
        self._correct_pad_spin.setValue(15)
        correct_pad_row.addWidget(self._correct_pad_spin)
        dlt.addLayout(correct_pad_row)

        self._correct_btn = QPushButton("Correct Label")
        self._correct_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._correct_btn)

        self._correct_status_lbl = QLabel("")
        self._correct_status_lbl.setWordWrap(True)
        dlt.addWidget(self._correct_status_lbl)

        dlt.addWidget(_sep())

        copyslice_note = QLabel(
            "  Copy Label to Adjacent Slice — copies this label's shape "
            "from the CURRENT slice onto the next or previous slice, "
            "e.g. to patch a slice where its cross-section is missing "
            "or broken. The label's own old shape on the target slice "
            "is replaced; a neighboring label's pixels are never "
            "touched, even where the copied shape would otherwise land "
            "on top of them."
        )
        copyslice_note.setWordWrap(True)
        copyslice_note.setStyleSheet("color: #888; font-size: 10px;")
        dlt.addWidget(copyslice_note)

        copyslice_label_row = QHBoxLayout()
        copyslice_label_row.addWidget(QLabel("Label to copy:"))
        self._copyslice_label_spin = QSpinBox()
        self._copyslice_label_spin.setMinimum(1)
        self._copyslice_label_spin.setMaximum(99999)
        self._copyslice_label_spin.setValue(1)
        copyslice_label_row.addWidget(self._copyslice_label_spin)
        self._copyslice_use_sel_btn = QPushButton("Use selected")
        self._copyslice_use_sel_btn.setFixedWidth(90)
        copyslice_label_row.addWidget(self._copyslice_use_sel_btn)
        dlt.addLayout(copyslice_label_row)

        copyslice_dir_row = QHBoxLayout()
        copyslice_dir_row.addWidget(QLabel("Copy to:"))
        self._copyslice_dir_combo = QComboBox()
        self._copyslice_dir_combo.addItem("Next slice (Z+1)", 1)
        self._copyslice_dir_combo.addItem("Previous slice (Z-1)", -1)
        copyslice_dir_row.addWidget(self._copyslice_dir_combo)
        dlt.addLayout(copyslice_dir_row)

        self._copyslice_btn = QPushButton("Copy Label to Adjacent Slice")
        self._copyslice_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._copyslice_btn)

        self._copyslice_status_lbl = QLabel("")
        self._copyslice_status_lbl.setWordWrap(True)
        dlt.addWidget(self._copyslice_status_lbl)

        dlt.addWidget(_sep())

        self._save_labels_btn = QPushButton("Save Labels")
        self._save_labels_btn.setStyleSheet("QPushButton { padding: 5px; }")
        dlt.addWidget(self._save_labels_btn)

        self._save_labels_status_lbl = QLabel("")
        self._save_labels_status_lbl.setWordWrap(True)
        dlt.addWidget(self._save_labels_status_lbl)

        self._downstream_label_tools.setLayout(dlt)
        t2.addWidget(self._downstream_label_tools)

        t2.addStretch()
        tab2.setLayout(t2)
        tabs.addTab(_wrap_scroll(tab2), "Create Labels")

        # ============================================================ #
        # TAB 3 — Statistics
        # ============================================================ #
        tab3 = QWidget()
        t3 = QVBoxLayout()
        t3.setSpacing(6)

        cfg = self._state.get("config", {})

        t3_note = QLabel(
            "Computes morphological, spatial, and intensity statistics per "
            "labelled cell (volume, shape, branching, nearest neighbours, "
            "optional intensity/brain-region features) and exports a CSV. "
            "Select a Labels layer, then choose a description backend and "
            "click Generate Statistics."
        )
        t3_note.setWordWrap(True)
        t3_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t3.addWidget(t3_note)

        # Statistics needs an actual Labels layer to operate on -- this tab
        # used to hide itself entirely until one existed, which meant a
        # first-time user had no way to discover Tab 3 was there to fill in
        # once they had labels. Instead, the tab stays visible and shows an
        # explanatory hint in place of the (temporarily irrelevant) controls
        # below -- same "explain instead of hide" pattern already used for
        # Tab 2's Pixel Classifier/Cellpose-SAM sections.
        self._stats_no_labels_hint = QLabel("")
        self._stats_no_labels_hint.setWordWrap(True)
        self._stats_no_labels_hint.setStyleSheet("color: #8ab; font-size: 10px; font-style: italic;")
        t3.addWidget(self._stats_no_labels_hint)

        t3.addWidget(_sep())

        _stats_content_start = t3.count()

        def _update_stats_tab_content(has_labels):
            if has_labels:
                self._stats_no_labels_hint.setVisible(False)
            else:
                self._stats_no_labels_hint.setText(
                    "No Labels layer yet — create one first via Tab 2's Create "
                    "Labels (Pixel Classifier or Cellpose-SAM Segmentation), or "
                    "load/create one another way, then come back here to "
                    "compute statistics."
                )
                self._stats_no_labels_hint.setVisible(True)
            for i in range(_stats_content_start, t3.count()):
                item = t3.itemAt(i)
                w_ = item.widget()
                if w_ is not None:
                    w_.setVisible(has_labels)
                else:
                    lay = item.layout()
                    if lay is not None:
                        _set_layout_widgets_visible(lay, has_labels)

        self._update_stats_tab_content = _update_stats_tab_content

        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("Description:"))
        self._stats_backend_combo = QComboBox()
        self._stats_backend_combo.addItem("Rule-based (offline)",    "rule")
        self._stats_backend_combo.addItem("Ollama (local, free)",    "ollama")
        self._stats_backend_combo.addItem("OpenAI API (paid)",       "openai")
        self._stats_backend_combo.addItem("Claude API (paid)",       "claude")
        # Restore saved backend selection
        saved_backend = cfg.get("stats_backend", "rule")
        for i in range(self._stats_backend_combo.count()):
            if self._stats_backend_combo.itemData(i) == saved_backend:
                self._stats_backend_combo.setCurrentIndex(i)
                break
        desc_row.addWidget(self._stats_backend_combo)
        t3.addLayout(desc_row)

        stats_note = QLabel(
            "  Rule-based: no internet, no key needed.\n"
            "  Ollama: install from ollama.com, then: ollama pull llama3\n"
            "  Paid APIs: provide your own key below."
        )
        stats_note.setStyleSheet("color: #aaa; font-size: 10px;")
        stats_note.setWordWrap(True)
        t3.addWidget(stats_note)

        # ── Ollama sub-panel ──────────────────────────────────────────── #
        self._ollama_panel = QWidget()
        op = QVBoxLayout()
        op.setContentsMargins(0, 0, 0, 0)
        op.setSpacing(3)
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("  Endpoint:"))
        self._ollama_endpoint_edit = QLineEdit(cfg.get("ollama_endpoint", "http://localhost:11434"))
        ep_row.addWidget(self._ollama_endpoint_edit)
        op.addLayout(ep_row)
        om_row = QHBoxLayout()
        om_row.addWidget(QLabel("  Model:"))
        self._ollama_model_edit = QLineEdit(cfg.get("ollama_model", "llama3"))
        om_row.addWidget(self._ollama_model_edit)
        op.addLayout(om_row)
        self._ollama_panel.setLayout(op)
        t3.addWidget(self._ollama_panel)

        # ── Remote API sub-panel ──────────────────────────────────────── #
        self._api_panel = QWidget()
        ap = QVBoxLayout()
        ap.setContentsMargins(0, 0, 0, 0)
        ap.setSpacing(3)
        ak_row = QHBoxLayout()
        ak_row.addWidget(QLabel("  API Key:"))
        self._api_key_edit = QLineEdit(_secrets.get_secret("api_key"))
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setPlaceholderText("sk-… or ant-… (saved encrypted, OS credential store)")
        ak_row.addWidget(self._api_key_edit)
        ap.addLayout(ak_row)
        am_row = QHBoxLayout()
        am_row.addWidget(QLabel("  Model:"))
        self._api_model_edit = QLineEdit(cfg.get("api_model", ""))
        self._api_model_edit.setPlaceholderText("e.g. gpt-4o-mini or claude-haiku-4-5-20251001")
        am_row.addWidget(self._api_model_edit)
        ap.addLayout(am_row)
        au_row = QHBoxLayout()
        au_row.addWidget(QLabel("  Base URL:"))
        self._api_url_edit = QLineEdit(cfg.get("api_url", ""))
        self._api_url_edit.setPlaceholderText("optional override (OpenAI-compat proxies)")
        au_row.addWidget(self._api_url_edit)
        ap.addLayout(au_row)
        self._api_panel.setLayout(ap)
        t3.addWidget(self._api_panel)

        t3.addWidget(_sep())

        # ── Intensity statistics ──────────────────────────────────────── #
        t3.addWidget(QLabel("Intensity statistics (optional):"))
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("  Image layer:"))
        self._stats_image_combo = QComboBox()
        self._stats_image_combo.addItem("None", None)
        img_row.addWidget(self._stats_image_combo)
        t3.addLayout(img_row)
        img_note = QLabel(
            "  Adds mean_intensity, integrated_intensity, intensity_cv per label."
        )
        img_note.setStyleSheet("color: #aaa; font-size: 10px;")
        img_note.setWordWrap(True)
        t3.addWidget(img_note)

        t3.addWidget(_sep())

        # ── Brain regions ─────────────────────────────────────────────── #
        t3.addWidget(QLabel("Brain regions (optional):"))
        shapes_row = QHBoxLayout()
        shapes_row.addWidget(QLabel("  Boundary lines:"))
        self._stats_shapes_combo = QComboBox()
        self._stats_shapes_combo.addItem("None", None)
        shapes_row.addWidget(self._stats_shapes_combo)
        t3.addLayout(shapes_row)
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("  Region names:"))
        self._stats_region_names_edit = QLineEdit()
        self._stats_region_names_edit.setPlaceholderText(
            "e.g. Optic tectum, Hindbrain  (comma-sep., anterior→posterior)"
        )
        region_row.addWidget(self._stats_region_names_edit)
        t3.addLayout(region_row)
        regions_note = QLabel(
            "  Draw 'line' shapes in a Shapes layer to mark region boundaries\n"
            "  (sorted anterior→posterior). N lines → N+1 region names."
        )
        regions_note.setStyleSheet("color: #aaa; font-size: 10px;")
        regions_note.setWordWrap(True)
        t3.addWidget(regions_note)

        t3.addWidget(_sep())

        # ── Output column selector ────────────────────────────────────────── #
        col_hdr = QHBoxLayout()
        col_hdr.addWidget(QLabel("Output columns:"))
        _col_all_btn   = QPushButton("All")
        _col_all_btn.setFixedWidth(36)
        _col_reset_btn = QPushButton("Reset")
        _col_reset_btn.setFixedWidth(44)
        col_hdr.addWidget(_col_all_btn)
        col_hdr.addWidget(_col_reset_btn)
        col_hdr.addStretch()
        t3.addLayout(col_hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(155)
        col_inner = QWidget()
        col_vbox  = QVBoxLayout()
        col_vbox.setSpacing(1)
        col_vbox.setContentsMargins(4, 2, 4, 2)
        self._col_checkboxes = {}
        for _key, _lbl, _on in _STATS_COLUMNS:
            cb = QCheckBox(_lbl)
            cb.setChecked(_on)
            if _key == "label":
                cb.setEnabled(False)
            col_vbox.addWidget(cb)
            self._col_checkboxes[_key] = cb
        col_inner.setLayout(col_vbox)
        scroll.setWidget(col_inner)
        t3.addWidget(scroll)

        def _select_all_cols():
            for cb in self._col_checkboxes.values():
                if cb.isEnabled():
                    cb.setChecked(True)

        def _reset_cols():
            for (_key, _lbl, _on) in _STATS_COLUMNS:
                cb = self._col_checkboxes[_key]
                if cb.isEnabled():
                    cb.setChecked(_on)

        _col_all_btn.clicked.connect(_select_all_cols)
        _col_reset_btn.clicked.connect(_reset_cols)

        t3.addWidget(_sep())

        stats_is_gt_note = QLabel(
            "  is_volume_outlier below is checked against the never-rising "
            "Min volume floor and never-falling largest-cell ceiling -- "
            "both are tuned exclusively by the GT Toolkit Tuning Tool "
            "(Tab 5), never from here. Statistics only ever reads them, "
            "purely for reporting."
        )
        stats_is_gt_note.setWordWrap(True)
        stats_is_gt_note.setStyleSheet("color: #888; font-size: 10px;")
        t3.addWidget(stats_is_gt_note)
        init_max_volume = _root_cfg.get("max_volume_recommended_vox")
        self._maxvol_ceiling_lbl = QLabel(
            f"  is_volume_outlier ceiling (largest confirmed GT cell so far): {init_max_volume} vox "
            "-- the floor side reuses Min volume above."
            if init_max_volume is not None else
            "  is_volume_outlier ceiling: not yet measured -- run the GT "
            "Toolkit Tuning Tool (Tab 5) against a verified GT fish at "
            "least once. The floor side reuses Min volume above."
        )
        self._maxvol_ceiling_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        self._maxvol_ceiling_lbl.setWordWrap(True)
        t3.addWidget(self._maxvol_ceiling_lbl)

        self._stats_btn = QPushButton("Generate Statistics")
        self._stats_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t3.addWidget(self._stats_btn)

        self._stats_status_lbl = QLabel("")
        self._stats_status_lbl.setWordWrap(True)
        t3.addWidget(self._stats_status_lbl)

        t3.addWidget(_sep())

        # ── Score Against GT — whole-fish Hungarian-matched scoring ────── #
        # The compare_pred_gt.py methodology this project has used to
        # validate essentially every real modeling decision, ported as a
        # reusable scorer instead of remaining a CLI-only script. No GPU
        # needed -- pure Hungarian matching (scipy) between two already-
        # computed Labels layers.
        gtg = QGroupBox("Score Against GT")
        gtl = QVBoxLayout()
        gtl.setSpacing(6)

        gt_note = QLabel(
            "Whole-fish, Hungarian-matched instance scoring (TP/FP/FN/Score + "
            "mean IoU/Dice on matched pairs) between any predicted Labels layer "
            "and a ground-truth Labels layer — same methodology used throughout "
            "this project's own model comparisons, not an approximation."
        )
        gt_note.setWordWrap(True)
        gt_note.setStyleSheet("color: #888; font-size: 10px;")
        gtl.addWidget(gt_note)

        gt_pred_row = QHBoxLayout()
        gt_pred_row.addWidget(QLabel("Predicted labels:"))
        self._gtscore_pred_combo = QComboBox()
        self._gtscore_pred_combo.addItem("None", None)
        gt_pred_row.addWidget(self._gtscore_pred_combo)
        gtl.addLayout(gt_pred_row)

        gt_gt_row = QHBoxLayout()
        gt_gt_row.addWidget(QLabel("GT labels:"))
        self._gtscore_gt_combo = QComboBox()
        self._gtscore_gt_combo.addItem("None", None)
        gt_gt_row.addWidget(self._gtscore_gt_combo)
        gtl.addLayout(gt_gt_row)

        gt_thresh_row = QHBoxLayout()
        gt_thresh_row.addWidget(QLabel("IoU threshold for a match:"))
        self._gtscore_thresh_spin = QDoubleSpinBox()
        self._gtscore_thresh_spin.setDecimals(2)
        self._gtscore_thresh_spin.setRange(0.05, 0.95)
        self._gtscore_thresh_spin.setValue(0.5)
        gt_thresh_row.addWidget(self._gtscore_thresh_spin)
        gtl.addLayout(gt_thresh_row)

        self._gtscore_btn = QPushButton("Score Against GT")
        self._gtscore_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        gtl.addWidget(self._gtscore_btn)

        self._gtscore_status_lbl = QLabel("")
        self._gtscore_status_lbl.setWordWrap(True)
        gtl.addWidget(self._gtscore_status_lbl)

        self._gtscore_report_view = QTextEdit()
        self._gtscore_report_view.setReadOnly(True)
        self._gtscore_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._gtscore_report_view.setFixedHeight(160)
        gtl.addWidget(self._gtscore_report_view)

        gtg.setLayout(gtl)
        gtg = _make_collapsible(gtg)
        t5.addWidget(gtg)
        self._t5_category_groups.setdefault("general", []).append(gtg)

        # ── Email notification (optional, shared by both Tab 4 training ── #
        # groups) -- wraps the launched training command in a small
        # self-contained supervisor script (see _training_jobs.launch_
        # detached's `notify` param) that IS the detached process, so the
        # completion email still gets sent even if napari (and this whole
        # plugin process) isn't running when the job finishes, same
        # guarantee as the training itself surviving napari closing.
        notify_cfg = self._state.get("config", {})
        ng = QGroupBox("Email notification (optional)")
        nl = QVBoxLayout()
        nl.setSpacing(6)

        ng_note = QLabel(
            "Sends one email when a Tab 4 training run stops (finishes, "
            "crashes, or is early-stopped) — even if napari is closed at the "
            "time. Leave Notify email blank to disable (the default). Shared "
            "by both MONAI and Cellpose-SAM training, whichever is active."
        )
        ng_note.setWordWrap(True)
        ng_note.setStyleSheet("color: #888; font-size: 10px;")
        nl.addWidget(ng_note)

        nto_row = QHBoxLayout()
        nto_row.addWidget(QLabel("Notify email:"))
        self._notify_to_edit = QLineEdit(notify_cfg.get("notify_email_to", ""))
        self._notify_to_edit.setPlaceholderText("leave blank to disable — e.g. you@example.com")
        nto_row.addWidget(self._notify_to_edit)
        nl.addLayout(nto_row)

        nsmtp_row = QHBoxLayout()
        nsmtp_row.addWidget(QLabel("SMTP server:"))
        self._notify_smtp_host_edit = QLineEdit(notify_cfg.get("notify_smtp_host", "smtp.gmail.com"))
        nsmtp_row.addWidget(self._notify_smtp_host_edit)
        nsmtp_row.addWidget(QLabel("port:"))
        self._notify_smtp_port_spin = QSpinBox()
        self._notify_smtp_port_spin.setRange(1, 65535)
        self._notify_smtp_port_spin.setValue(int(notify_cfg.get("notify_smtp_port", 465)))
        nsmtp_row.addWidget(self._notify_smtp_port_spin)
        nl.addLayout(nsmtp_row)

        nuser_row = QHBoxLayout()
        nuser_row.addWidget(QLabel("SMTP username:"))
        self._notify_smtp_user_edit = QLineEdit(notify_cfg.get("notify_smtp_user", ""))
        nuser_row.addWidget(self._notify_smtp_user_edit)
        nl.addLayout(nuser_row)

        npass_row = QHBoxLayout()
        npass_row.addWidget(QLabel("SMTP password:"))
        self._notify_smtp_password_edit = QLineEdit(_secrets.get_secret("notify_smtp_password"))
        self._notify_smtp_password_edit.setEchoMode(QLineEdit.Password)
        self._notify_smtp_password_edit.setPlaceholderText("saved encrypted (OS credential store) — use a Gmail App Password, never your real password")
        npass_row.addWidget(self._notify_smtp_password_edit)
        nl.addLayout(npass_row)

        notify_note = QLabel(
            "Configure this once, opt in per tool with each 'Email me when done' checkbox "
            "elsewhere in the plugin. Free with any Gmail account: smtp.gmail.com, port 465, "
            "username = your Gmail address, password = a Google App Password "
            "(myaccount.google.com/apppasswords — requires 2-Step Verification), not your normal "
            "Gmail password. The password is saved encrypted in your OS's credential store "
            "(Windows Credential Manager / macOS Keychain / Linux Secret Service), not the "
            "plugin's own config file, so you only need to set this up once — safe because an "
            "App Password is a separate, revocable credential Google issues specifically for "
            "this kind of unattended use, never your real account password. On Linux without an "
            "unlocked Secret Service session (common over SSH), it falls back automatically to a "
            "local encrypted file instead — still not plaintext, just a weaker guarantee than the "
            "OS store since the key lives alongside it."
        )
        notify_note.setWordWrap(True)
        notify_note.setStyleSheet("color: #888; font-size: 10px;")
        nl.addWidget(notify_note)

        self._notify_test_btn = QPushButton("Send Test Email")
        nl.addWidget(self._notify_test_btn)
        self._notify_test_status_lbl = QLabel("")
        self._notify_test_status_lbl.setWordWrap(True)
        nl.addWidget(self._notify_test_status_lbl)

        ng.setLayout(nl)
        ng = _make_collapsible(ng)
        t5.addWidget(ng)
        self._t5_category_groups.setdefault("general", []).append(ng)

        t3.addStretch()
        tab3.setLayout(t3)
        tabs.addTab(_wrap_scroll(tab3), "Statistics")
        self._update_stats_tab_content(
            any(isinstance(l, napari.layers.Labels) for l in self._viewer.layers)
        )

        # ============================================================ #
        # TAB 4 — AI Tools
        # ============================================================ #
        # Always visible regardless of GPU -- deliberately NOT gated on
        # VRAM (see _gpu_check.py). CPU-only training/inference still
        # works, just far slower (days-months instead of hours for a
        # full run); a smaller GPU may also work with a reduced
        # batch_size. Hiding the tab entirely would block a user who
        # could still get real value out of it, just more slowly, or by
        # tuning batch_size down. A prominent disclaimer below stands in
        # for the old hard gate.
        tab4 = QWidget()
        t4 = QVBoxLayout()
        t4.setSpacing(6)

        gpu_banner = QLabel()
        gpu_banner.setWordWrap(True)
        if not GPU_HAS_CUDA:
            gpu_banner.setText(
                "⚠ No CUDA-capable GPU detected. Training and inference below will run "
                "on CPU, which can take days to months for a full training run instead "
                "of hours — fine for small experiments, but plan accordingly for "
                "anything larger. GT Annotation itself needs no GPU either way."
            )
            gpu_banner.setStyleSheet(
                "background-color: #5a1a1a; color: #ffcccc; padding: 8px; "
                "border-radius: 4px; font-size: 11px; font-weight: bold;"
            )
        elif not GPU_MEETS_RECOMMENDED:
            gpu_banner.setText(
                f"⚠ GPU detected ({GPU_NAME}, {GPU_VRAM_GB:.1f} GB VRAM) — below the "
                "8 GB recommended for full-size batches. Training may still work by "
                "reducing batch_size (try 2, or even 1) but could be slow or hit "
                "out-of-memory errors. If a run crashes, lower batch_size first before "
                "assuming something else is wrong."
            )
            gpu_banner.setStyleSheet(
                "background-color: #5a4a1a; color: #ffe6b3; padding: 8px; "
                "border-radius: 4px; font-size: 11px; font-weight: bold;"
            )
        else:
            gpu_banner.setText(f"{GPU_MSG} — meets the recommended minimum for full-size batches.")
            gpu_banner.setStyleSheet(
                "background-color: #1a4a1a; color: #ccffcc; padding: 6px; "
                "border-radius: 4px; font-size: 10px;"
            )
        t4.addWidget(gpu_banner)

        t4_note = QLabel(
            "Builds and trains the two AI models the rest of the plugin "
            "depends on: MONAI (Tab 1's skin/brain segmentation) and "
            "Cellpose-SAM (Tab 2's microglia segmentation). Everything here "
            "is either ground-truth creation or a training-launcher — for "
            "GT-verification sweeps and related utilities, see Tab 5."
        )
        t4_note.setWordWrap(True)
        t4_note.setStyleSheet("color: #888; font-size: 10px;")
        t4.addWidget(t4_note)

        t4.addWidget(_sep())

        # Email notification (optional, shared by both MONAI and
        # Cellpose-SAM training below) now lives in Tab 5 -- Sweeps &
        # Utilities, General category -- not gated behind either group,
        # nothing else here needs to change to reach it.

        # ── Group switch ──────────────────────────────────────────────── #
        switch_row = QHBoxLayout()
        self._ai_monai_radio = QRadioButton("MONAI Training")
        self._ai_cellpose_radio = QRadioButton("Cellpose-SAM Training")
        self._ai_mode_group = QButtonGroup(self)
        self._ai_mode_group.addButton(self._ai_monai_radio, 0)
        self._ai_mode_group.addButton(self._ai_cellpose_radio, 1)
        saved_ai_mode = self._state.get("config", {}).get("ai_tools_mode", "monai")
        if saved_ai_mode == "cellpose":
            self._ai_cellpose_radio.setChecked(True)
        else:
            self._ai_monai_radio.setChecked(True)
        switch_row.addWidget(self._ai_monai_radio)
        switch_row.addWidget(self._ai_cellpose_radio)
        t4.addLayout(switch_row)

        t4.addWidget(_sep())

        # ── Group 1 — GT annotation + MONAI training ────────────────────── #
        self._ai_monai_group = QGroupBox("MONAI Training")
        self._ai_monai_group_layout = QVBoxLayout()
        self._ai_monai_group_layout.setSpacing(6)

        ai_monai_group_note = QLabel(
            "The MONAI pipeline, in order: annotate ground truth (below), "
            "prepare the training dataset from it, then train the brain-"
            "segmentation model Tab 1 uses."
        )
        ai_monai_group_note.setWordWrap(True)
        ai_monai_group_note.setStyleSheet("color: #888; font-size: 10px;")
        self._ai_monai_group_layout.addWidget(ai_monai_group_note)

        gtg = QGroupBox("GT Annotation")
        gtl = QVBoxLayout()
        gtl.setSpacing(6)

        gt_note = QLabel(
            "Creates hand-drawn ground-truth brain/skin masks by interpolating "
            "polygons between key slices — the source of truth MONAI training, "
            "and the GT-sweep tools in Tab 5, are checked against.\n\n"
            "1. Pick the Image layer to annotate.\n"
            "2. Select the 'brain_polygons' layer (yellow) and draw\n"
            "   polygons every ~10 slices with the polygon tool.\n"
            "3. Click Interpolate, review the cyan result, then Generate Masks."
        )
        gt_note.setWordWrap(True)
        gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        gtl.addWidget(gt_note)

        gt_img_row = QHBoxLayout()
        gt_img_row.addWidget(QLabel("Image layer:"))
        self._gt_image_combo = QComboBox()
        self._gt_image_combo.addItem("None", None)
        gt_img_row.addWidget(self._gt_image_combo)
        gtl.addLayout(gt_img_row)

        self._gt_interpolate_btn = QPushButton("1. Interpolate Polygons")
        gtl.addWidget(self._gt_interpolate_btn)
        self._gt_generate_btn = QPushButton("2. Generate Masks")
        gtl.addWidget(self._gt_generate_btn)

        self._gt_status_lbl = QLabel("")
        self._gt_status_lbl.setWordWrap(True)
        gtl.addWidget(self._gt_status_lbl)

        gtg.setLayout(gtl)
        gtg = _make_collapsible(gtg)
        self._ai_monai_group_layout.addWidget(gtg)
        self._ai_monai_group_layout.addWidget(_sep())

        # ── Prepare Training Data (prepare_data.py) ─────────────────── #
        cfg = self._state.get("config", {})
        pdg = QGroupBox("Prepare Training Data")
        pdl = QVBoxLayout()
        pdl.setSpacing(6)

        pd_note = QLabel(
            "Converts raw+GT fish folders into the HDF5 dataset\n"
            "train.py needs. Leave brain/skin dirs blank to use the\n"
            "script's own built-in defaults."
        )
        pd_note.setWordWrap(True)
        pd_note.setStyleSheet("color: #aaa; font-size: 10px;")
        pdl.addWidget(pd_note)

        bd_row = QHBoxLayout()
        bd_row.addWidget(QLabel("Brain dirs:"))
        self._pd_brain_dirs_edit = QLineEdit(cfg.get("monai_brain_dirs", ""))
        self._pd_brain_dirs_edit.setPlaceholderText("comma-separated paths (optional)")
        bd_row.addWidget(self._pd_brain_dirs_edit)
        pdl.addLayout(bd_row)

        sd_row = QHBoxLayout()
        sd_row.addWidget(QLabel("Skin dirs:"))
        self._pd_skin_dirs_edit = QLineEdit(cfg.get("monai_skin_dirs", ""))
        self._pd_skin_dirs_edit.setPlaceholderText("comma-separated paths (optional)")
        sd_row.addWidget(self._pd_skin_dirs_edit)
        pdl.addLayout(sd_row)

        pd_out_row = QHBoxLayout()
        pd_out_row.addWidget(QLabel("Output dir:"))
        self._pd_output_dir_edit = QLineEdit(cfg.get("monai_data_dir", "training_data_v2"))
        pd_out_row.addWidget(self._pd_output_dir_edit)
        self._pd_output_browse_btn = QPushButton("...")
        self._pd_output_browse_btn.setFixedWidth(32)
        pd_out_row.addWidget(self._pd_output_browse_btn)
        pdl.addLayout(pd_out_row)

        pd_tissue_row = QHBoxLayout()
        pd_tissue_row.addWidget(QLabel("Tissue:"))
        self._pd_tissue_combo = QComboBox()
        self._pd_tissue_combo.addItems(["brain", "skin"])
        pd_tissue_row.addWidget(self._pd_tissue_combo)
        pdl.addLayout(pd_tissue_row)

        pd_nums_row = QHBoxLayout()
        pd_nums_row.addWidget(QLabel("n_val:"))
        self._pd_nval_spin = QSpinBox()
        self._pd_nval_spin.setRange(1, 100)
        self._pd_nval_spin.setValue(5)
        pd_nums_row.addWidget(self._pd_nval_spin)
        pd_nums_row.addWidget(QLabel("n_test:"))
        self._pd_ntest_spin = QSpinBox()
        self._pd_ntest_spin.setRange(1, 100)
        self._pd_ntest_spin.setValue(5)
        pd_nums_row.addWidget(self._pd_ntest_spin)
        pdl.addLayout(pd_nums_row)

        pd_seed_row = QHBoxLayout()
        pd_seed_row.addWidget(QLabel("split_seed:"))
        self._pd_seed_spin = QSpinBox()
        self._pd_seed_spin.setRange(0, 999999)
        self._pd_seed_spin.setValue(16)
        pd_seed_row.addWidget(self._pd_seed_spin)
        pd_seed_row.addWidget(QLabel("num_workers:"))
        self._pd_workers_spin = QSpinBox()
        self._pd_workers_spin.setRange(1, 128)
        self._pd_workers_spin.setValue(max(1, int((os.cpu_count() or 4) * 0.75)))
        pd_seed_row.addWidget(self._pd_workers_spin)
        pdl.addLayout(pd_seed_row)

        self._pd_run_btn = QPushButton("Prepare Training Data")
        self._pd_run_btn.setStyleSheet("QPushButton { padding: 5px; }")
        pdl.addWidget(self._pd_run_btn)

        self._pd_status_lbl = QLabel("")
        self._pd_status_lbl.setWordWrap(True)
        pdl.addWidget(self._pd_status_lbl)

        pdg.setLayout(pdl)
        pdg = _make_collapsible(pdg)
        self._ai_monai_group_layout.addWidget(pdg)
        self._ai_monai_group_layout.addWidget(_sep())

        # ── Train MONAI (train.py) ──────────────────────────────────── #
        mtg = QGroupBox("Train MONAI U-Net")
        mtl = QVBoxLayout()
        mtl.setSpacing(6)

        mt_note = QLabel(
            "Launches the actual brain-segmentation training run on the "
            "dataset from Prepare Training Data above (hours to multiple "
            "days). Runs as a detached process — closing napari doesn't stop "
            "it, and reopening reconnects automatically. See \"How training "
            "launches work\" below."
        )
        mt_note.setWordWrap(True)
        mt_note.setStyleSheet("color: #888; font-size: 10px;")
        mtl.addWidget(mt_note)

        mt_data_row = QHBoxLayout()
        mt_data_row.addWidget(QLabel("Data dir:"))
        self._mt_data_dir_edit = QLineEdit(cfg.get("monai_data_dir", "training_data_v2"))
        mt_data_row.addWidget(self._mt_data_dir_edit)
        mtl.addLayout(mt_data_row)

        mt_model_row = QHBoxLayout()
        mt_model_row.addWidget(QLabel("Model dir:"))
        self._mt_model_dir_edit = QLineEdit(cfg.get("monai_model_dir", "models_v2"))
        mt_model_row.addWidget(self._mt_model_dir_edit)
        self._mt_model_browse_btn = QPushButton("...")
        self._mt_model_browse_btn.setFixedWidth(32)
        mt_model_row.addWidget(self._mt_model_browse_btn)
        mtl.addLayout(mt_model_row)

        mt_ep_row = QHBoxLayout()
        mt_ep_row.addWidget(QLabel("epochs:"))
        self._mt_epochs_spin = QSpinBox()
        self._mt_epochs_spin.setRange(1, 100000)
        self._mt_epochs_spin.setValue(1500)
        mt_ep_row.addWidget(self._mt_epochs_spin)
        mt_ep_row.addWidget(QLabel("batch_size:"))
        self._mt_batch_spin = QSpinBox()
        self._mt_batch_spin.setRange(1, 64)
        self._mt_batch_spin.setValue(2)
        mt_ep_row.addWidget(self._mt_batch_spin)
        mtl.addLayout(mt_ep_row)

        mt_lr_row = QHBoxLayout()
        mt_lr_row.addWidget(QLabel("lr:"))
        self._mt_lr_spin = QDoubleSpinBox()
        self._mt_lr_spin.setDecimals(6)
        self._mt_lr_spin.setRange(1e-6, 1.0)
        self._mt_lr_spin.setSingleStep(1e-5)
        self._mt_lr_spin.setValue(1e-4)
        mt_lr_row.addWidget(self._mt_lr_spin)
        mt_lr_row.addWidget(QLabel("gpu idx:"))
        self._mt_gpu_spin = QSpinBox()
        self._mt_gpu_spin.setRange(0, 15)
        mt_lr_row.addWidget(self._mt_gpu_spin)
        mtl.addLayout(mt_lr_row)

        mt_resume_row = QHBoxLayout()
        mt_resume_row.addWidget(QLabel("resume:"))
        self._mt_resume_edit = QLineEdit("")
        self._mt_resume_edit.setPlaceholderText("optional checkpoint — blank = fresh start")
        mt_resume_row.addWidget(self._mt_resume_edit)
        self._mt_resume_browse_btn = QPushButton("...")
        self._mt_resume_browse_btn.setFixedWidth(32)
        mt_resume_row.addWidget(self._mt_resume_browse_btn)
        mtl.addLayout(mt_resume_row)

        mt_sched_row = QHBoxLayout()
        mt_sched_row.addWidget(QLabel("val_every:"))
        self._mt_valevery_spin = QSpinBox()
        self._mt_valevery_spin.setRange(1, 1000)
        self._mt_valevery_spin.setValue(5)
        mt_sched_row.addWidget(self._mt_valevery_spin)
        mt_sched_row.addWidget(QLabel("ckpt_every:"))
        self._mt_ckptevery_spin = QSpinBox()
        self._mt_ckptevery_spin.setRange(1, 1000)
        self._mt_ckptevery_spin.setValue(50)
        mt_sched_row.addWidget(self._mt_ckptevery_spin)
        mtl.addLayout(mt_sched_row)

        mt_pat_row = QHBoxLayout()
        mt_pat_row.addWidget(QLabel("Patience (checkpoints):"))
        self._mt_patience_early_spin = QSpinBox()
        self._mt_patience_early_spin.setRange(0, 1000)
        self._mt_patience_early_spin.setValue(5)
        mt_pat_row.addWidget(self._mt_patience_early_spin)
        mtl.addLayout(mt_pat_row)
        mt_pat_note = QLabel(
            "  Stop after N checkpoints with no improvement in the model-selection\n"
            "  metric (Full-brain Dice). Same rule as Cellpose-SAM's patience field\n"
            "  below — only the metric differs. 0 disables early stopping."
        )
        mt_pat_note.setStyleSheet("color: #aaa; font-size: 10px;")
        mt_pat_note.setWordWrap(True)
        mtl.addWidget(mt_pat_note)

        self._mt_notify_cb = _make_notify_checkbox()
        self._mt_notify_cb.setText("Email me when this training run stops")
        mtl.addWidget(self._mt_notify_cb)

        mt_btn_row = QHBoxLayout()
        self._mt_launch_btn = QPushButton("Launch Training")
        self._mt_launch_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        mt_btn_row.addWidget(self._mt_launch_btn)
        self._mt_stop_btn = QPushButton("Stop Training")
        self._mt_stop_btn.setEnabled(False)
        mt_btn_row.addWidget(self._mt_stop_btn)
        mtl.addLayout(mt_btn_row)

        self._mt_status_lbl = QLabel("")
        self._mt_status_lbl.setWordWrap(True)
        mtl.addWidget(self._mt_status_lbl)

        self._mt_log_view = QTextEdit()
        self._mt_log_view.setReadOnly(True)
        self._mt_log_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._mt_log_view.setFixedHeight(120)
        mtl.addWidget(self._mt_log_view)

        mtg.setLayout(mtl)
        mtg = _make_collapsible(mtg)
        self._ai_monai_group_layout.addWidget(mtg)

        # In-memory job state (PID/log path/timer) — mirrors the config
        # keys persisted for resume-after-restart (see _load_monai_job_state)
        self._monai_job = {"pid": None, "log_path": None, "timer": None}

        self._ai_monai_group.setLayout(self._ai_monai_group_layout)
        self._ai_monai_group = _make_collapsible(self._ai_monai_group)
        t4.addWidget(self._ai_monai_group)

        # ── Group 2 — Cellpose-SAM crop extraction + training ───────────── #
        self._ai_cellpose_group = QGroupBox("Cellpose-SAM Training")
        self._ai_cellpose_group_layout = QVBoxLayout()
        self._ai_cellpose_group_layout.setSpacing(6)

        ai_cellpose_group_note = QLabel(
            "The Cellpose-SAM pipeline, in order: extract training crops "
            "(below), then fine-tune the microglia-segmentation model Tab 2 "
            "uses."
        )
        ai_cellpose_group_note.setWordWrap(True)
        ai_cellpose_group_note.setStyleSheet("color: #888; font-size: 10px;")
        self._ai_cellpose_group_layout.addWidget(ai_cellpose_group_note)

        cfg = self._state.get("config", {})

        # ── Extract XZYZ Patches (generate_xzyz_patches.py) ─────────── #
        # The crop-generation method every real Cellpose-SAM training
        # run has actually used since May 2026 (train_cellpose_512,
        # _multi, _multi3 -- including the branch-weighted-loss runs).
        # Cleanup is on by default, not a separate manual step, per
        # explicit instruction: crops should only ever train on cells
        # that are substantially complete, going forward.
        xzg = QGroupBox("Extract XZYZ Patches")
        xzl = QVBoxLayout()
        xzl.setSpacing(6)

        xz_note = QLabel(
            "Generates 2D training crops in all 3 orientations (XY native, "
            "XZ/YZ Z-stretched by anisotropy) from a full-fish image + GT "
            "labels pair — the method actually used for every real "
            "Cellpose-SAM training run in this project."
        )
        xz_note.setWordWrap(True)
        xz_note.setStyleSheet("color: #888; font-size: 10px;")
        xzl.addWidget(xz_note)

        xz_img_row = QHBoxLayout()
        xz_img_row.addWidget(QLabel("Image:"))
        self._xz_img_edit = QLineEdit("")
        xz_img_row.addWidget(self._xz_img_edit)
        self._xz_img_browse_btn = QPushButton("...")
        self._xz_img_browse_btn.setFixedWidth(32)
        xz_img_row.addWidget(self._xz_img_browse_btn)
        xzl.addLayout(xz_img_row)

        xz_gt_row = QHBoxLayout()
        xz_gt_row.addWidget(QLabel("GT labels:"))
        self._xz_gt_edit = QLineEdit("")
        xz_gt_row.addWidget(self._xz_gt_edit)
        self._xz_gt_browse_btn = QPushButton("...")
        self._xz_gt_browse_btn.setFixedWidth(32)
        xz_gt_row.addWidget(self._xz_gt_browse_btn)
        xzl.addLayout(xz_gt_row)

        xz_out_row = QHBoxLayout()
        xz_out_row.addWidget(QLabel("Output folder:"))
        self._xz_out_edit = QLineEdit("")
        xz_out_row.addWidget(self._xz_out_edit)
        self._xz_out_browse_btn = QPushButton("...")
        self._xz_out_browse_btn.setFixedWidth(32)
        xz_out_row.addWidget(self._xz_out_browse_btn)
        xzl.addLayout(xz_out_row)

        xz_scale_row = QHBoxLayout()
        xz_scale_row.addWidget(QLabel("Z (µm):"))
        self._xz_scalez_spin = QDoubleSpinBox()
        self._xz_scalez_spin.setDecimals(4)
        self._xz_scalez_spin.setRange(0.0001, 100.0)
        self._xz_scalez_spin.setValue(1.0)
        xz_scale_row.addWidget(self._xz_scalez_spin)
        xz_scale_row.addWidget(QLabel("XY (µm):"))
        self._xz_scalexy_spin = QDoubleSpinBox()
        self._xz_scalexy_spin.setDecimals(4)
        self._xz_scalexy_spin.setRange(0.0001, 100.0)
        self._xz_scalexy_spin.setValue(0.174)
        xz_scale_row.addWidget(self._xz_scalexy_spin)
        xzl.addLayout(xz_scale_row)

        xz_opts_row = QHBoxLayout()
        xz_opts_row.addWidget(QLabel("crop_size:"))
        self._xz_cropsize_spin = QSpinBox()
        self._xz_cropsize_spin.setRange(32, 4096)
        self._xz_cropsize_spin.setValue(512)
        xz_opts_row.addWidget(self._xz_cropsize_spin)
        xz_opts_row.addWidget(QLabel("crops/slice:"))
        self._xz_ncrops_spin = QSpinBox()
        self._xz_ncrops_spin.setRange(1, 100)
        self._xz_ncrops_spin.setValue(5)
        xz_opts_row.addWidget(self._xz_ncrops_spin)
        xzl.addLayout(xz_opts_row)
        xz_opts_row2 = QHBoxLayout()
        xz_opts_row2.addWidget(QLabel("max/orientation:"))
        self._xz_maxn_spin = QSpinBox()
        self._xz_maxn_spin.setRange(1, 100000)
        self._xz_maxn_spin.setValue(320)
        xz_opts_row2.addWidget(self._xz_maxn_spin)
        xzl.addLayout(xz_opts_row2)

        xz_opts2_row = QHBoxLayout()
        xz_opts2_row.addWidget(QLabel("min_gt_pixels:"))
        self._xz_mingt_spin = QSpinBox()
        self._xz_mingt_spin.setRange(1, 100000)
        self._xz_mingt_spin.setValue(10)
        xz_opts2_row.addWidget(self._xz_mingt_spin)
        xz_opts2_row.addWidget(QLabel("seed:"))
        self._xz_seed_spin = QSpinBox()
        self._xz_seed_spin.setRange(0, 999999)
        self._xz_seed_spin.setValue(42)
        xz_opts2_row.addWidget(self._xz_seed_spin)
        xzl.addLayout(xz_opts2_row)

        self._xz_clean_cb = QCheckBox("Clean truncated labels")
        self._xz_clean_cb.setChecked(True)
        xzl.addWidget(self._xz_clean_cb)

        xz_thresh_row = QHBoxLayout()
        xz_thresh_row.addWidget(QLabel("Min visible fraction:"))
        self._xz_threshold_spin = QDoubleSpinBox()
        self._xz_threshold_spin.setDecimals(2)
        self._xz_threshold_spin.setRange(0.05, 1.0)
        self._xz_threshold_spin.setValue(0.9)
        xz_thresh_row.addWidget(self._xz_threshold_spin)
        xzl.addLayout(xz_thresh_row)
        xz_clean_note = QLabel(
            "  Any label whose crop-visible pixel count is below this fraction of "
            "its true full-slice size gets zeroed out — an incidental neighbor "
            "grazed by the crop edge, not the crop's own intended cell (which is "
            "essentially always well above this already). On by default: crops "
            "should only ever train on substantially complete cells going forward."
        )
        xz_clean_note.setStyleSheet("color: #aaa; font-size: 10px;")
        xz_clean_note.setWordWrap(True)
        xzl.addWidget(xz_clean_note)

        self._xz_run_btn = QPushButton("Extract XZYZ Patches")
        self._xz_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        xzl.addWidget(self._xz_run_btn)

        self._xz_status_lbl = QLabel("")
        self._xz_status_lbl.setWordWrap(True)
        xzl.addWidget(self._xz_status_lbl)

        xzg.setLayout(xzl)
        xzg = _make_collapsible(xzg)
        self._ai_cellpose_group_layout.addWidget(xzg)
        self._ai_cellpose_group_layout.addWidget(_sep())

        self._xz_patches_job = {"thread": None, "timer": None}

        # ── Train Cellpose-SAM (train_xzyz.py) ──────────────────────── #
        ctg = QGroupBox("Train Cellpose-SAM")
        ctl = QVBoxLayout()
        ctl.setSpacing(6)

        ct_note = QLabel(
            "Launches Cellpose-SAM fine-tuning on the crops from Extract "
            "XZYZ Patches above (~20h for 200 epochs on this project's usual "
            "dataset sizes). Runs as a detached process — closing napari "
            "doesn't stop it, and reopening reconnects automatically."
        )
        ct_note.setWordWrap(True)
        ct_note.setStyleSheet("color: #888; font-size: 10px;")
        ctl.addWidget(ct_note)

        ct_data_row = QHBoxLayout()
        ct_data_row.addWidget(QLabel("Data dir:"))
        self._ct_data_dir_edit = QLineEdit(cfg.get("cellpose_crops_data_dir", ""))
        ct_data_row.addWidget(self._ct_data_dir_edit)
        ctl.addLayout(ct_data_row)

        ct_pre_row = QHBoxLayout()
        ct_pre_row.addWidget(QLabel("Pretrained:"))
        init_pretrained = str(self._state["cellpose_model_path"]) if self._state.get("cellpose_model_path") else "cpsam"
        self._ct_pretrained_edit = QLineEdit(init_pretrained)
        ct_pre_row.addWidget(self._ct_pretrained_edit)
        self._ct_pretrained_browse_btn = QPushButton("...")
        self._ct_pretrained_browse_btn.setFixedWidth(32)
        ct_pre_row.addWidget(self._ct_pretrained_browse_btn)
        ctl.addLayout(ct_pre_row)
        ct_pre_note = QLabel("  Defaults to the checkpoint already loaded in Tab 2 — \"continue training\".")
        ct_pre_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ct_pre_note.setWordWrap(True)
        ctl.addWidget(ct_pre_note)

        ct_name_row = QHBoxLayout()
        ct_name_row.addWidget(QLabel("model_name:"))
        self._ct_modelname_edit = QLineEdit("cpsam_microglia_xzyz")
        ct_name_row.addWidget(self._ct_modelname_edit)
        ctl.addLayout(ct_name_row)

        ct_ep_row = QHBoxLayout()
        ct_ep_row.addWidget(QLabel("n_epochs:"))
        self._ct_epochs_spin = QSpinBox()
        self._ct_epochs_spin.setRange(1, 100000)
        self._ct_epochs_spin.setValue(200)
        ct_ep_row.addWidget(self._ct_epochs_spin)
        ct_ep_row.addWidget(QLabel("batch_size:"))
        self._ct_batch_spin = QSpinBox()
        self._ct_batch_spin.setRange(1, 64)
        self._ct_batch_spin.setValue(4)
        ct_ep_row.addWidget(self._ct_batch_spin)
        ctl.addLayout(ct_ep_row)

        ct_save_row = QHBoxLayout()
        ct_save_row.addWidget(QLabel("save_every:"))
        self._ct_saveevery_spin = QSpinBox()
        self._ct_saveevery_spin.setRange(1, 1000)
        self._ct_saveevery_spin.setValue(10)
        ct_save_row.addWidget(self._ct_saveevery_spin)
        ct_save_row.addWidget(QLabel("log_every:"))
        self._ct_logevery_spin = QSpinBox()
        self._ct_logevery_spin.setRange(1, 1000)
        self._ct_logevery_spin.setValue(5)
        ct_save_row.addWidget(self._ct_logevery_spin)
        ctl.addLayout(ct_save_row)

        ct_lr_row = QHBoxLayout()
        ct_lr_row.addWidget(QLabel("lr:"))
        self._ct_lr_spin = QDoubleSpinBox()
        self._ct_lr_spin.setDecimals(6)
        self._ct_lr_spin.setRange(1e-6, 1.0)
        self._ct_lr_spin.setSingleStep(1e-5)
        self._ct_lr_spin.setValue(1e-4)
        ct_lr_row.addWidget(self._ct_lr_spin)
        ctl.addLayout(ct_lr_row)

        ct_bw_row = QHBoxLayout()
        ct_bw_row.addWidget(QLabel("branch_weight:"))
        self._ct_branchweight_spin = QDoubleSpinBox()
        self._ct_branchweight_spin.setDecimals(1)
        self._ct_branchweight_spin.setRange(0.0, 20.0)
        self._ct_branchweight_spin.setSingleStep(0.5)
        self._ct_branchweight_spin.setValue(0.0)
        ct_bw_row.addWidget(self._ct_branchweight_spin)
        ct_bw_row.addWidget(QLabel("branch_radius:"))
        self._ct_branchradius_spin = QSpinBox()
        self._ct_branchradius_spin.setRange(1, 20)
        self._ct_branchradius_spin.setValue(_root_cfg.get("cellpose_branch_radius", 3))
        ct_bw_row.addWidget(self._ct_branchradius_spin)
        ctl.addLayout(ct_bw_row)
        ct_bw_note = QLabel("  branch_weight=0 disables the branch-weighted loss (standard Cellpose loss).")
        ct_bw_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ct_bw_note.setWordWrap(True)
        ctl.addWidget(ct_bw_note)
        self._ct_branchradius_recommended_lbl = _add_recommended_label(
            ctl, _root_cfg.get("cellpose_branch_radius_recommended"), unit=" px", noun="branch_radius"
        )

        ct_calib_row = QHBoxLayout()
        ct_calib_row.addWidget(QLabel("Calibrate from GT:"))
        self._ct_calib_gt_edit = QLineEdit("")
        ct_calib_row.addWidget(self._ct_calib_gt_edit)
        self._ct_calib_browse_btn = QPushButton("...")
        self._ct_calib_browse_btn.setFixedWidth(32)
        ct_calib_row.addWidget(self._ct_calib_browse_btn)
        ctl.addLayout(ct_calib_row)

        ct_calib_scale_row = QHBoxLayout()
        ct_calib_scale_row.addWidget(QLabel("scale Z:"))
        self._ct_calib_scalez_spin = QDoubleSpinBox()
        self._ct_calib_scalez_spin.setDecimals(4)
        self._ct_calib_scalez_spin.setRange(0.0001, 100.0)
        self._ct_calib_scalez_spin.setValue(1.0)
        ct_calib_scale_row.addWidget(self._ct_calib_scalez_spin)
        ct_calib_scale_row.addWidget(QLabel("scale XY:"))
        self._ct_calib_scalexy_spin = QDoubleSpinBox()
        self._ct_calib_scalexy_spin.setDecimals(4)
        self._ct_calib_scalexy_spin.setRange(0.0001, 100.0)
        self._ct_calib_scalexy_spin.setValue(0.174)
        ct_calib_scale_row.addWidget(self._ct_calib_scalexy_spin)
        ctl.addLayout(ct_calib_scale_row)
        self._ct_calib_is_gt_cb = _add_gt_checkbox(ctl, "this calibration", visible=False)
        self._ct_calib_run_btn = QPushButton("Calibrate branch_radius")
        ctl.addWidget(self._ct_calib_run_btn)
        ct_calib_note = QLabel(
            "  Measures real branch thickness (3D skeleton + distance transform) from a "
            "GT labels volume and sets branch_radius above to the thinnest-quartile "
            "(distal branch tip) radius in pixels — recalibrated from actual morphology "
            "instead of a guessed value. GT-verified auto-apply now only happens via the "
            "GT Toolkit Tuning Tool (Tab 5, top); running it here directly is preview-only."
        )
        ct_calib_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ct_calib_note.setWordWrap(True)
        ctl.addWidget(ct_calib_note)
        self._ct_calib_status_lbl = QLabel("")
        self._ct_calib_status_lbl.setWordWrap(True)
        ctl.addWidget(self._ct_calib_status_lbl)

        ct_pat_row = QHBoxLayout()
        ct_pat_row.addWidget(QLabel("Patience (checkpoints):"))
        self._ct_patience_early_spin = QSpinBox()
        self._ct_patience_early_spin.setRange(0, 1000)
        self._ct_patience_early_spin.setValue(5)
        ct_pat_row.addWidget(self._ct_patience_early_spin)
        ctl.addLayout(ct_pat_row)
        ct_pat_note = QLabel(
            "  Stop after N checkpoints with no improvement in test_loss. Same\n"
            "  rule as MONAI's patience field above — only the metric differs.\n"
            "  0 disables early stopping."
        )
        ct_pat_note.setStyleSheet("color: #aaa; font-size: 10px;")
        ct_pat_note.setWordWrap(True)
        ctl.addWidget(ct_pat_note)

        self._ct_notify_cb = _make_notify_checkbox()
        self._ct_notify_cb.setText("Email me when this training run stops")
        ctl.addWidget(self._ct_notify_cb)

        ct_btn_row = QHBoxLayout()
        self._ct_launch_btn = QPushButton("Launch Training")
        self._ct_launch_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        ct_btn_row.addWidget(self._ct_launch_btn)
        self._ct_stop_btn = QPushButton("Stop Training")
        self._ct_stop_btn.setEnabled(False)
        ct_btn_row.addWidget(self._ct_stop_btn)
        ctl.addLayout(ct_btn_row)

        self._ct_status_lbl = QLabel("")
        self._ct_status_lbl.setWordWrap(True)
        ctl.addWidget(self._ct_status_lbl)

        self._ct_log_view = QTextEdit()
        self._ct_log_view.setReadOnly(True)
        self._ct_log_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._ct_log_view.setFixedHeight(120)
        ctl.addWidget(self._ct_log_view)

        ctg.setLayout(ctl)
        ctg = _make_collapsible(ctg)
        self._ai_cellpose_group_layout.addWidget(ctg)
        self._ai_cellpose_group_layout.addWidget(_sep())

        # ── Verify Best Epoch (GT sweep) ────────────────────────────── #
        esg = QGroupBox("Verify Best Epoch (GT Sweep)")
        esl = QVBoxLayout()
        esl.setSpacing(6)

        es_note = QLabel(
            "Automates the manual GT-verification sweep: finds the N most "
            "morphologically complex cells in a ground-truth-annotated fish "
            "(most skeleton branches, not simply the largest), crops each to "
            "its bounding box, and runs do_3D inference at the recommended "
            "epoch plus checkpoints below/above it to confirm which epoch "
            "actually scores best against real GT — test_loss is only a proxy."
        )
        es_note.setWordWrap(True)
        es_note.setStyleSheet("color: #888; font-size: 10px;")
        esl.addWidget(es_note)

        es_img_row = QHBoxLayout()
        es_img_row.addWidget(QLabel("GT image:"))
        self._es_img_edit = QLineEdit("")
        es_img_row.addWidget(self._es_img_edit)
        self._es_img_browse_btn = QPushButton("...")
        self._es_img_browse_btn.setFixedWidth(32)
        es_img_row.addWidget(self._es_img_browse_btn)
        esl.addLayout(es_img_row)

        es_lbl_row = QHBoxLayout()
        es_lbl_row.addWidget(QLabel("GT labels:"))
        self._es_lbl_edit = QLineEdit("")
        es_lbl_row.addWidget(self._es_lbl_edit)
        self._es_lbl_browse_btn = QPushButton("...")
        self._es_lbl_browse_btn.setFixedWidth(32)
        es_lbl_row.addWidget(self._es_lbl_browse_btn)
        esl.addLayout(es_lbl_row)
        es_gt_note = QLabel(
            "  GT image = a full-fish _ExtRm brain_only volume (same input "
            "do_3D inference uses in production — not _NoBG/_RndFill, and "
            "not the raw pre-Tab-1 image). GT labels = the corresponding "
            "hand-corrected microglia instance-label volume, typically "
            "named _GROUND_TRUTH.tif, one integer ID per cell — the same "
            "pair used to create training crops."
        )
        es_gt_note.setStyleSheet("color: #aaa; font-size: 10px;")
        es_gt_note.setWordWrap(True)
        esl.addWidget(es_gt_note)

        es_rec_row = QHBoxLayout()
        es_rec_row.addWidget(QLabel("Recommended epoch:"))
        self._es_recepoch_spin = QSpinBox()
        self._es_recepoch_spin.setRange(0, 1000000)
        es_rec_row.addWidget(self._es_recepoch_spin)
        self._es_readrec_btn = QPushButton("From pointer file")
        es_rec_row.addWidget(self._es_readrec_btn)
        esl.addLayout(es_rec_row)
        es_rec_note = QLabel(
            "  Reads <model_name>_best_recommended.txt (see Train Cellpose-SAM "
            "above) using this section's Data dir/model_name — or set manually."
        )
        es_rec_note.setStyleSheet("color: #aaa; font-size: 10px;")
        es_rec_note.setWordWrap(True)
        esl.addWidget(es_rec_note)

        es_span_row = QHBoxLayout()
        es_span_row.addWidget(QLabel("Checkpoints below:"))
        self._es_below_spin = QSpinBox()
        self._es_below_spin.setRange(0, 20)
        self._es_below_spin.setValue(2)
        es_span_row.addWidget(self._es_below_spin)
        es_span_row.addWidget(QLabel("above:"))
        self._es_above_spin = QSpinBox()
        self._es_above_spin.setRange(0, 20)
        self._es_above_spin.setValue(2)
        es_span_row.addWidget(self._es_above_spin)
        esl.addLayout(es_span_row)
        es_span_row2 = QHBoxLayout()
        es_span_row2.addWidget(QLabel("save_every:"))
        self._es_saveevery_spin = QSpinBox()
        self._es_saveevery_spin.setRange(1, 1000)
        self._es_saveevery_spin.setValue(10)
        es_span_row2.addWidget(self._es_saveevery_spin)
        esl.addLayout(es_span_row2)

        es_cells_row = QHBoxLayout()
        es_cells_row.addWidget(QLabel("Complex cells to test:"))
        self._es_ncells_spin = QSpinBox()
        self._es_ncells_spin.setRange(1, 50)
        self._es_ncells_spin.setValue(5)
        es_cells_row.addWidget(self._es_ncells_spin)
        esl.addLayout(es_cells_row)
        es_cells_row2 = QHBoxLayout()
        es_cells_row2.addWidget(QLabel("Pad Z:"))
        self._es_padz_spin = QSpinBox()
        self._es_padz_spin.setRange(0, 200)
        self._es_padz_spin.setValue(15)
        es_cells_row2.addWidget(self._es_padz_spin)
        es_cells_row2.addWidget(QLabel("Pad XY:"))
        self._es_padxy_spin = QSpinBox()
        self._es_padxy_spin.setRange(0, 500)
        self._es_padxy_spin.setValue(40)
        es_cells_row2.addWidget(self._es_padxy_spin)
        esl.addLayout(es_cells_row2)

        es_scale_row = QHBoxLayout()
        es_scale_row.addWidget(QLabel("Z (µm):"))
        self._es_scalez_spin = QDoubleSpinBox()
        self._es_scalez_spin.setDecimals(4)
        self._es_scalez_spin.setRange(0.0001, 100.0)
        self._es_scalez_spin.setValue(1.0)
        es_scale_row.addWidget(self._es_scalez_spin)
        es_scale_row.addWidget(QLabel("XY (µm):"))
        self._es_scalexy_spin = QDoubleSpinBox()
        self._es_scalexy_spin.setDecimals(4)
        self._es_scalexy_spin.setRange(0.0001, 100.0)
        self._es_scalexy_spin.setValue(0.174)
        es_scale_row.addWidget(self._es_scalexy_spin)
        esl.addLayout(es_scale_row)
        es_scale_note = QLabel(
            "  Independent of whatever's currently open in the viewer — set explicitly "
            "for the GT fish above (defaults match this project's standard 0.174µm/1.0µm)."
        )
        es_scale_note.setStyleSheet("color: #aaa; font-size: 10px;")
        es_scale_note.setWordWrap(True)
        esl.addWidget(es_scale_note)

        es_inf_row = QHBoxLayout()
        es_inf_row.addWidget(QLabel("Cellprob threshold:"))
        self._es_cellprob_spin = QDoubleSpinBox()
        self._es_cellprob_spin.setDecimals(2)
        self._es_cellprob_spin.setRange(-6.0, 6.0)
        self._es_cellprob_spin.setSingleStep(0.1)
        self._es_cellprob_spin.setValue(self._cp_cellprob_spin.value())
        es_inf_row.addWidget(self._es_cellprob_spin)
        esl.addLayout(es_inf_row)

        self._es_notify_cb = _make_notify_checkbox()
        esl.addWidget(self._es_notify_cb)

        self._es_is_gt_cb = _add_gt_checkbox(
            esl, "this sweep -- overriding the active checkpoint is a consequential change"
        )

        es_btn_row = QHBoxLayout()
        self._es_run_btn = QPushButton("Run Epoch Sweep")
        self._es_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 5px; }")
        es_btn_row.addWidget(self._es_run_btn)
        self._es_stop_btn = QPushButton("Stop Sweep")
        self._es_stop_btn.setEnabled(False)
        es_btn_row.addWidget(self._es_stop_btn)
        esl.addLayout(es_btn_row)

        self._es_status_lbl = QLabel("")
        self._es_status_lbl.setWordWrap(True)
        esl.addWidget(self._es_status_lbl)

        self._es_report_view = QTextEdit()
        self._es_report_view.setReadOnly(True)
        self._es_report_view.setStyleSheet("font-family: monospace; font-size: 9px;")
        self._es_report_view.setFixedHeight(160)
        esl.addWidget(self._es_report_view)

        esg.setLayout(esl)
        esg = _make_collapsible(esg)
        t5.addWidget(esg)
        self._t5_category_groups.setdefault("cellpose", []).append(esg)

        # Now that every Tab 5 tool has registered its category, wire the
        # filter checkboxes built at the top of this tab and apply their
        # initial (persisted) state.
        def _on_t5_filter_changed(*_):
            state = {
                "skin": self._t5_filter_skin_cb.isChecked(),
                "pixel": self._t5_filter_pixel_cb.isChecked(),
                "cellpose": self._t5_filter_cellpose_cb.isChecked(),
                "general": self._t5_filter_general_cb.isChecked(),
            }
            for category, visible in state.items():
                for group in self._t5_category_groups.get(category, []):
                    group.setVisible(visible)
            self._save_cfg(t5_filters=state)

        self._on_t5_filter_changed = _on_t5_filter_changed
        for cb in (self._t5_filter_skin_cb, self._t5_filter_pixel_cb,
                   self._t5_filter_cellpose_cb, self._t5_filter_general_cb):
            cb.stateChanged.connect(self._on_t5_filter_changed)
        self._on_t5_filter_changed()

        self._epoch_sweep_job = {"thread": None, "cancel_event": None, "timer": None}
        self._branch_calib_job = {"thread": None, "timer": None}

        self._cellpose_job = {"pid": None, "log_path": None, "timer": None}

        self._ai_cellpose_group.setLayout(self._ai_cellpose_group_layout)
        self._ai_cellpose_group = _make_collapsible(self._ai_cellpose_group)
        t4.addWidget(self._ai_cellpose_group)

        t4.addStretch()
        tab4.setLayout(t4)
        tabs.addTab(_wrap_scroll(tab4), "AI Tools")
        self._tabs.setTabVisible(3, True)

        # ============================================================ #
        # TAB 5 — Sweeps & Utilities
        # ============================================================ #
        # Content (bsg, psg, krg, gtpg, gtg, esg) was built inline within
        # Tabs 1-4's own sections above, targeting t5.addWidget(...)
        # directly at each site -- see the t5 setup note near the top of
        # _build_ui for why moving *where* a group displays doesn't
        # require moving *how* it's built.
        t5.addStretch()
        tab5.setLayout(t5)
        tabs.addTab(_wrap_scroll(tab5), "Sweeps && Utilities")  # && -- Qt treats a lone & as a mnemonic marker and swallows it

        # ── outer layout ────────────────────────────────────────────── #
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(tabs)
        self.setLayout(outer)

    # ------------------------------------------------------------------ #
    # Signal connections
    # ------------------------------------------------------------------ #

    def _connect_signals(self):
        self._open_btn.clicked.connect(self._on_open)
        self._model_browse_btn.clicked.connect(self._on_browse_model)
        self._bg_group.buttonClicked.connect(self._on_bg_mode_changed)
        self._run_btn.clicked.connect(self._on_run)
        self._gtk_browse_file_btn.clicked.connect(self._on_gtk_browse_file)
        self._gtk_browse_folder_btn.clicked.connect(self._on_gtk_browse_folder)
        self._gtk_scan_btn.clicked.connect(self._on_gtk_scan)
        self._gtk_run_btn.clicked.connect(self._on_gtk_run)
        self._gtk_stop_btn.clicked.connect(self._on_gtk_stop)
        self._bs_img_browse_btn.clicked.connect(self._on_bs_browse_img)
        self._bs_gt_browse_btn.clicked.connect(self._on_bs_browse_gt)
        self._bs_run_btn.clicked.connect(self._on_bs_run_sweep)
        self._bs_stop_btn.clicked.connect(self._on_bs_stop_sweep)
        self._labels_btn.clicked.connect(self._on_create_labels)
        self._ps_img_browse_btn.clicked.connect(self._on_ps_browse_img)
        self._ps_mask_browse_btn.clicked.connect(self._on_ps_browse_mask)
        self._ps_lbl_browse_btn.clicked.connect(self._on_ps_browse_lbl)
        self._ps_run_btn.clicked.connect(self._on_ps_run_sweep)
        self._ps_stop_btn.clicked.connect(self._on_ps_stop_sweep)
        self._sg_img_browse_btn.clicked.connect(self._on_sg_browse_img)
        self._sg_mask_browse_btn.clicked.connect(self._on_sg_browse_mask)
        self._sg_lbl_browse_btn.clicked.connect(self._on_sg_browse_lbl)
        self._sg_run_btn.clicked.connect(self._on_sg_run_sweep)
        self._sg_stop_btn.clicked.connect(self._on_sg_stop_sweep)
        self._cp_model_browse_btn.clicked.connect(self._on_browse_cp_model)
        self._cp_run_btn.clicked.connect(self._on_run_cellpose_seg)
        self._cp_relabel_btn.clicked.connect(self._on_rerun_single_cell)
        self._cp_relabel_use_sel_btn.clicked.connect(self._on_use_selected_label_rerun)
        self._kr_img_browse_btn.clicked.connect(self._on_kr_browse_img)
        self._kr_gt_browse_btn.clicked.connect(self._on_kr_browse_gt)
        self._kr_run_btn.clicked.connect(self._on_kr_run_sweep)
        self._kr_stop_btn.clicked.connect(self._on_kr_stop_sweep)
        self._gtp_img_browse_btn.clicked.connect(self._on_gtp_browse_img)
        self._gtp_masks_browse_btn.clicked.connect(self._on_gtp_browse_masks)
        self._gtp_raw_browse_btn.clicked.connect(self._on_gtp_browse_raw)
        self._gtp_guide_browse_btn.clicked.connect(self._on_gtp_browse_guide)
        self._gtp_out_browse_btn.clicked.connect(self._on_gtp_browse_out)
        self._gtp_run_btn.clicked.connect(self._on_gtp_run)
        self._notify_test_btn.clicked.connect(self._on_send_test_email)
        self._resort_btn.clicked.connect(self._on_resort_labels)
        self._debris_btn.clicked.connect(self._on_remove_debris)
        self._split_use_sel_btn.clicked.connect(self._on_use_selected_label)
        self._split_btn.clicked.connect(self._on_split_label)
        self._join_a_use_sel_btn.clicked.connect(self._on_use_selected_label_join_a)
        self._join_b_use_sel_btn.clicked.connect(self._on_use_selected_label_join_b)
        self._join_btn.clicked.connect(self._on_join_labels)
        self._correct_use_sel_btn.clicked.connect(self._on_use_selected_label_correct)
        self._correct_btn.clicked.connect(self._on_correct_label)
        self._copyslice_use_sel_btn.clicked.connect(self._on_use_selected_label_copyslice)
        self._copyslice_btn.clicked.connect(self._on_copy_label_to_adjacent_slice)
        self._save_labels_btn.clicked.connect(self._on_save_labels)
        self._stats_backend_combo.currentIndexChanged.connect(self._on_stats_backend_changed)
        self._stats_btn.clicked.connect(self._on_generate_stats)
        self._gtscore_btn.clicked.connect(self._on_gtscore_run)
        self._ai_mode_group.buttonClicked.connect(self._on_ai_tools_mode_changed)
        self._gt_image_combo.currentIndexChanged.connect(self._on_gt_image_changed)
        self._gt_interpolate_btn.clicked.connect(self._on_gt_interpolate)
        self._gt_generate_btn.clicked.connect(self._on_gt_generate_masks)
        self._viewer.layers.events.inserted.connect(self._refresh_gt_layers)
        self._viewer.layers.events.removed.connect(self._refresh_gt_layers)
        self._pd_output_browse_btn.clicked.connect(self._on_pd_browse_output)
        self._pd_run_btn.clicked.connect(self._on_prepare_monai_data)
        self._mt_model_browse_btn.clicked.connect(self._on_mt_browse_model_dir)
        self._mt_resume_browse_btn.clicked.connect(self._on_mt_browse_resume)
        self._mt_launch_btn.clicked.connect(self._on_mt_launch_training)
        self._mt_stop_btn.clicked.connect(self._on_mt_stop_training)
        self._xz_img_browse_btn.clicked.connect(self._on_xz_browse_img)
        self._xz_gt_browse_btn.clicked.connect(self._on_xz_browse_gt)
        self._xz_out_browse_btn.clicked.connect(self._on_xz_browse_out)
        self._xz_run_btn.clicked.connect(self._on_xz_run)
        self._ct_pretrained_browse_btn.clicked.connect(self._on_ct_browse_pretrained)
        self._ct_calib_browse_btn.clicked.connect(self._on_ct_calib_browse_gt)
        self._ct_calib_run_btn.clicked.connect(self._on_ct_calib_run)
        self._ct_launch_btn.clicked.connect(self._on_ct_launch_training)
        self._ct_stop_btn.clicked.connect(self._on_ct_stop_training)
        self._es_img_browse_btn.clicked.connect(self._on_es_browse_img)
        self._es_lbl_browse_btn.clicked.connect(self._on_es_browse_lbl)
        self._es_readrec_btn.clicked.connect(self._on_es_read_recommended)
        self._es_run_btn.clicked.connect(self._on_es_run_sweep)
        self._es_stop_btn.clicked.connect(self._on_es_stop_sweep)
        self._viewer.layers.events.inserted.connect(self._refresh_layer_info)
        self._viewer.layers.events.removed.connect(self._refresh_layer_info)
        self._viewer.layers.selection.events.changed.connect(self._refresh_layer_info)
        self._viewer.layers.events.inserted.connect(self._refresh_stats_layers)
        self._viewer.layers.events.removed.connect(self._refresh_stats_layers)
        # Apply initial panel visibility
        self._on_stats_backend_changed()
        self._on_ai_tools_mode_changed()
        self._refresh_gt_layers()
        self._resume_monai_job_if_active()
        self._resume_cellpose_job_if_active()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _status(self, msg):
        self._status_lbl.setText(f"Status: {msg}")

    def _on_bg_mode_changed(self, btn):
        """Enable/disable the tolerance slider, and show/hide Signal Erosion
        (mode 2 only -- it's meaningless in every other mode), depending
        on the selected Background mode."""
        mode = self._bg_group.checkedId()
        has_tol = mode in (1, 2)
        self._tol_slider.setEnabled(has_tol)
        self._tol_spin.setEnabled(has_tol)
        self._tol_lbl.setEnabled(has_tol)
        has_sig_erosion = mode == 2
        _set_layout_widgets_visible(self._sig_erosion_row, has_sig_erosion)
        self._sig_erosion_recommended_lbl.setVisible(has_sig_erosion)

    def _on_ai_tools_mode_changed(self, *_):
        """Show only the training group matching the switch position."""
        mode = "cellpose" if self._ai_cellpose_radio.isChecked() else "monai"
        self._ai_monai_group.setVisible(mode == "monai")
        self._ai_cellpose_group.setVisible(mode == "cellpose")
        self._save_cfg(ai_tools_mode=mode)

    def _refresh_gt_layers(self, *_):
        """Repopulate the GT annotation Image-layer combo."""
        cur = self._gt_image_combo.currentData()
        self._gt_image_combo.blockSignals(True)
        self._gt_image_combo.clear()
        self._gt_image_combo.addItem("None", None)
        for lyr in self._viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                self._gt_image_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur:
                    self._gt_image_combo.setCurrentIndex(self._gt_image_combo.count() - 1)
        self._gt_image_combo.blockSignals(False)

    def _gt_selected_layer(self):
        """Return the Image layer currently selected in the GT combo, or None."""
        name = self._gt_image_combo.currentData()
        if name is None or name not in self._viewer.layers:
            return None
        lyr = self._viewer.layers[name]
        return lyr if isinstance(lyr, napari.layers.Image) else None

    def _on_gt_image_changed(self, *_):
        """Auto-create the 'brain_polygons' Shapes layer once a real Image
        layer is picked, so the user never hits the original script's
        KeyError-on-missing-layer footgun."""
        lyr = self._gt_selected_layer()
        if lyr is None:
            return
        scale = tuple(float(v) for v in lyr.scale) if len(lyr.scale) == 3 else (1.0, 1.0, 1.0)
        _gt.ensure_brain_polygons_layer(self._viewer, scale=scale)
        self._gt_status_lbl.setText(f"Draw polygons on the '{_gt.KEY_SHAPES_LAYER}' layer, then Interpolate.")

    def _on_gt_interpolate(self):
        lyr = self._gt_selected_layer()
        if lyr is None:
            self._gt_status_lbl.setText("ERROR: select an Image layer first.")
            return
        try:
            result = _gt.interpolate_shapes(self._viewer, lyr)
        except ValueError as exc:
            self._gt_status_lbl.setText(f"ERROR: {exc}")
            return
        self._gt_status_lbl.setText(
            f"Interpolated {len(result.data)} polygons — review 'brain_polygons_interpolated' (cyan), "
            f"then Generate Masks."
        )

    def _on_gt_generate_masks(self):
        lyr = self._gt_selected_layer()
        if lyr is None:
            self._gt_status_lbl.setText("ERROR: select an Image layer first.")
            return
        if not self._state.get("last_file_path"):
            self._gt_status_lbl.setText(
                "ERROR: no source file path known — open the file via "
                "'Open TIF / IMS file' first (needed to name the output folder)."
            )
            return
        try:
            out_dir = _gt.generate_masks(self._viewer, lyr, self._state["last_file_path"])
        except ValueError as exc:
            self._gt_status_lbl.setText(f"ERROR: {exc}")
            return
        self._gt_status_lbl.setText(f"Masks saved to {out_dir}")

    def _on_pd_browse_output(self):
        path_str = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path_str:
            self._pd_output_dir_edit.setText(path_str)

    def _on_prepare_monai_data(self):
        brain_dirs = [d.strip() for d in self._pd_brain_dirs_edit.text().split(",") if d.strip()]
        skin_dirs = [d.strip() for d in self._pd_skin_dirs_edit.text().split(",") if d.strip()]
        output_dir = self._pd_output_dir_edit.text().strip() or "training_data_v2"
        tissue = self._pd_tissue_combo.currentText()
        n_val = self._pd_nval_spin.value()
        n_test = self._pd_ntest_spin.value()
        split_seed = self._pd_seed_spin.value()
        num_workers = self._pd_workers_spin.value()

        script_path = self._state["config"].get("monai_prepare_script_path") or str(_ait.DEFAULT_PREPARE_DATA_SCRIPT)
        if not Path(script_path).exists():
            self._pd_status_lbl.setText(f"ERROR: prepare_data.py not found at {script_path}")
            return
        conda_env = self._state["config"].get("monai_conda_env", "zf-microglia-ai")

        argv = _ait.build_prepare_data_argv(
            script_path, brain_dirs, skin_dirs, output_dir, tissue,
            n_val, n_test, split_seed, num_workers,
        )
        cwd = Path(script_path).parent

        self._pd_run_btn.setEnabled(False)
        self._pd_status_lbl.setText("Running (this can take ~15 min)...")

        result = {}

        def _worker():
            try:
                proc = _tj.run_subprocess_job(argv, cwd, conda_env)
                result["returncode"] = proc.returncode
                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            self._pd_run_btn.setEnabled(True)
            if "error" in result:
                self._pd_status_lbl.setText(f"ERROR: {result['error']}")
                return
            if result["returncode"] != 0:
                tail = (result.get("stderr") or "")[-500:]
                self._pd_status_lbl.setText(f"ERROR (exit {result['returncode']}): {tail}")
                return
            self._pd_status_lbl.setText(f"Done — dataset prepared at {output_dir}")
            self._mt_data_dir_edit.setText(output_dir)
            self._save_cfg(
                monai_data_dir=output_dir,
                monai_brain_dirs=self._pd_brain_dirs_edit.text(),
                monai_skin_dirs=self._pd_skin_dirs_edit.text(),
            )

        timer.timeout.connect(_poll)
        timer.start(1000)

    def _on_mt_browse_model_dir(self):
        path_str = QFileDialog.getExistingDirectory(self, "Select model output directory")
        if path_str:
            self._mt_model_dir_edit.setText(path_str)

    def _on_mt_browse_resume(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select checkpoint to resume from", "", "All files (*)")
        if path_str:
            self._mt_resume_edit.setText(path_str)

    def _start_monai_polling(self):
        """(Re)start the coarse-interval log-tail/status poll for the active
        MONAI training job. 8s interval, deliberately coarser than every
        other job in the plugin — each tick does a psutil PID check plus a
        bounded log-file read, and there's zero UX benefit to sub-second
        polling on an hours-to-days job."""
        timer = QTimer(self)

        def _poll():
            pid = self._monai_job.get("pid")
            log_path = self._monai_job.get("log_path")
            if log_path:
                self._mt_log_view.setPlainText(_tj.tail_log(log_path))
                sb = self._mt_log_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            patience = self._monai_job.get("patience", 0)
            if pid and log_path and _tj.is_running(pid) and patience > 0:
                chk = _tj.patience_exceeded(log_path, _tj.MONAI_METRIC, patience)
                if chk["exceeded"]:
                    _tj.kill_process_tree(pid)
                    timer.stop()
                    self._monai_job["timer"] = None
                    self._mt_status_lbl.setText(
                        f"Early-stopped (PID {pid}): {chk['checkpoints_since_best']} checkpoints "
                        f"without improvement (best {_tj.MONAI_METRIC['label']}={chk['best_value']:.4f} "
                        f"at epoch {chk['best_epoch']} — already saved as best_model_fullstack.pth)."
                    )
                    self._mt_launch_btn.setEnabled(True)
                    self._mt_stop_btn.setEnabled(False)
                    return

            if pid and not _tj.is_running(pid):
                timer.stop()
                self._monai_job["timer"] = None
                # train.py auto-saves its own best_model_fullstack.pth whenever
                # a new best Full-brain Dice lands, so there's nothing for the
                # GUI to copy here (unlike Cellpose-SAM) -- just report which
                # epoch that already-saved file corresponds to.
                if log_path:
                    best = _tj.patience_exceeded(log_path, _tj.MONAI_METRIC, patience=0)
                else:
                    best = None
                if best and best["best_epoch"] is not None:
                    self._mt_status_lbl.setText(
                        f"Training process (PID {pid}) has stopped. Best "
                        f"{_tj.MONAI_METRIC['label']}={best['best_value']:.4f} at epoch "
                        f"{best['best_epoch']} — saved as best_model_fullstack.pth."
                    )
                else:
                    self._mt_status_lbl.setText(f"Training process (PID {pid}) has stopped.")
                self._mt_launch_btn.setEnabled(True)
                self._mt_stop_btn.setEnabled(False)

        timer.timeout.connect(_poll)
        timer.start(8000)
        self._monai_job["timer"] = timer
        _poll()  # immediate first tick so the log view isn't empty for 8s

    def _on_mt_launch_training(self):
        if self._monai_job.get("pid") and _tj.is_running(self._monai_job["pid"]):
            self._mt_status_lbl.setText("A training job is already running.")
            return

        script_path = self._state["config"].get("monai_train_script_path") or str(_ait.DEFAULT_MONAI_TRAIN_SCRIPT)
        if not Path(script_path).exists():
            self._mt_status_lbl.setText(f"ERROR: train.py not found at {script_path}")
            return

        data_dir = self._mt_data_dir_edit.text().strip() or "training_data_v2"
        model_dir = self._mt_model_dir_edit.text().strip() or "models_v2"
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        resume = self._mt_resume_edit.text().strip() or None

        # train.py has its own internal --patience early-stop; we override it
        # with an effectively-infinite value so it never preempts the GUI's
        # own external patience check below (_mt_patience_early_spin) --
        # that's the single source of early-stopping truth for both MONAI
        # and Cellpose-SAM, not two different mechanisms that happen to
        # look similar in the UI.
        argv = _ait.build_monai_train_argv(
            script_path, data_dir, model_dir,
            self._mt_epochs_spin.value(), self._mt_batch_spin.value(), self._mt_lr_spin.value(),
            resume, 999999, self._mt_valevery_spin.value(),
            self._mt_ckptevery_spin.value(), self._mt_gpu_spin.value(),
        )
        conda_env = self._state["config"].get("monai_conda_env", "zf-microglia-ai")
        cwd = Path(script_path).parent
        session = f"monai_train_{datetime.now():%Y%m%d_%H%M%S}"
        log_path = Path(model_dir) / f"gui_launch_{session}.log"

        notify = None
        if self._mt_notify_cb.isChecked():
            notify, notify_err = self._build_notify_cfg("MONAI U-Net", _tj.MONAI_METRIC)
            if notify_err:
                self._mt_status_lbl.setText(notify_err)
                return

        try:
            pid = _tj.launch_detached(argv, cwd, log_path, conda_env, notify=notify)
        except Exception as exc:
            self._mt_status_lbl.setText(f"ERROR launching: {exc}")
            return

        self._monai_job["pid"] = pid
        self._monai_job["log_path"] = str(log_path)
        self._monai_job["patience"] = self._mt_patience_early_spin.value()
        self._save_cfg(
            monai_active_pid=pid, monai_active_log=str(log_path),
            monai_data_dir=data_dir, monai_model_dir=model_dir,
            monai_patience=self._mt_patience_early_spin.value(),
        )
        self._mt_launch_btn.setEnabled(False)
        self._mt_stop_btn.setEnabled(True)
        self._mt_status_lbl.setText(f"Launched (PID {pid}) — log: {log_path}")
        self._start_monai_polling()

    def _on_mt_stop_training(self):
        pid = self._monai_job.get("pid")
        if not pid:
            return
        _tj.kill_process_tree(pid)
        self._mt_status_lbl.setText(f"Stopped (PID {pid}).")
        self._mt_launch_btn.setEnabled(True)
        self._mt_stop_btn.setEnabled(False)
        self._save_cfg(monai_active_pid=None, monai_active_log="")
        if self._monai_job.get("timer"):
            self._monai_job["timer"].stop()
            self._monai_job["timer"] = None

    def _resume_monai_job_if_active(self):
        """If a MONAI training job launched in a previous plugin/napari
        session is still running, reconnect the GUI to it instead of
        silently showing no active job — these are hours-to-days jobs
        that regularly outlive a single napari session.

        If the job already finished (or crashed) while napari was closed,
        is_running(pid) is now False -- report that outcome here instead
        of just discarding the stale PID, since this is the only chance
        the GUI ever gets to tell the user a job it doesn't currently see
        running actually completed."""
        cfg = self._state.get("config", {})
        pid = cfg.get("monai_active_pid")
        log_path = cfg.get("monai_active_log")
        if not pid:
            return
        if _tj.is_running(pid):
            self._monai_job["pid"] = pid
            self._monai_job["log_path"] = log_path
            self._monai_job["patience"] = cfg.get("monai_patience", 0)
            self._mt_launch_btn.setEnabled(False)
            self._mt_stop_btn.setEnabled(True)
            self._mt_status_lbl.setText(f"Resumed monitoring PID {pid} (already running).")
            self._start_monai_polling()
        else:
            self._save_cfg(monai_active_pid=None, monai_active_log="")
            best = _tj.patience_exceeded(log_path, _tj.MONAI_METRIC, patience=0) if log_path else None
            if best and best["best_epoch"] is not None:
                self._mt_status_lbl.setText(
                    f"Training (PID {pid}) finished while napari was closed. Best "
                    f"{_tj.MONAI_METRIC['label']}={best['best_value']:.4f} at epoch "
                    f"{best['best_epoch']} — saved as best_model_fullstack.pth."
                )
            else:
                self._mt_status_lbl.setText(f"Training (PID {pid}) is no longer running.")

    def _on_ct_browse_pretrained(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select Cellpose-SAM checkpoint", "", "All files (*)")
        if path_str:
            self._ct_pretrained_edit.setText(path_str)

    def _on_ct_calib_browse_gt(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._ct_calib_gt_edit.setText(path_str)

    def _on_ct_calib_run(self):
        if self._branch_calib_job.get("thread") and self._branch_calib_job["thread"].is_alive():
            self._ct_calib_status_lbl.setText("A calibration is already running.")
            return

        gt_path = self._ct_calib_gt_edit.text().strip()
        if not gt_path:
            self._ct_calib_status_lbl.setText("ERROR: set a GT labels path first.")
            return
        if not Path(gt_path).exists():
            self._ct_calib_status_lbl.setText(f"ERROR: GT labels not found: {gt_path}")
            return

        scale_zyx = (
            self._ct_calib_scalez_spin.value(),
            self._ct_calib_scalexy_spin.value(),
            self._ct_calib_scalexy_spin.value(),
        )
        current_radius = self._ct_branchradius_spin.value()

        result = {}

        def _worker():
            try:
                gt_labels = tifffile.imread(gt_path).astype(np.int32)
                stats = _bcal.recommend_branch_radius(gt_labels, scale_zyx=scale_zyx)
                result["stats"] = stats
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._branch_calib_job["thread"] = thread
        self._branch_calib_job["result"] = result
        self._branch_calib_job["current_radius"] = current_radius
        self._branch_calib_job["fish_key"] = Path(gt_path).stem
        self._ct_calib_run_btn.setEnabled(False)
        self._ct_calib_status_lbl.setText("Measuring branch morphology from GT...")
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            self._branch_calib_job["timer"] = None
            self._ct_calib_run_btn.setEnabled(True)

            # One-shot, same as Tab 3 Statistics.
            is_gt = self._ct_calib_is_gt_cb.isChecked()
            self._ct_calib_is_gt_cb.setChecked(False)

            if "error" in result:
                self._ct_calib_status_lbl.setText(f"ERROR: {result['error'].splitlines()[0]}")
                return

            stats = result["stats"]
            report = _bcal.format_branch_calibration_report(stats, self._branch_calib_job["current_radius"])
            measured_this_fish = stats["recommended_branch_radius_px"]

            if is_gt:
                # Never-falling ceiling, opposite direction from min_volume/
                # min_hole_size/gt_min -- see _update_gt_history's mode="max"
                # docstring. Previously this just overwrote branch_radius
                # with whatever this one run measured, with no cross-fish
                # memory at all -- a real bug, fixed here alongside adding
                # the aggregation.
                recommended = self._update_gt_history(
                    "cellpose_branch_radius", self._branch_calib_job["fish_key"],
                    measured_this_fish, mode="max",
                )
                self._ct_branchradius_spin.setValue(recommended)
                self._save_cfg(cellpose_branch_radius=recommended,
                                cellpose_branch_radius_recommended=recommended)
                self._ct_branchradius_recommended_lbl.setText(f"  Recommended branch_radius: {recommended} px")
                applied_note = (
                    f" Applied ceiling across all fish calibrated so far: "
                    f"branch_radius={recommended}. Saved."
                )
            else:
                applied_note = (
                    " Not GT-verified — recommended value NOT updated "
                    "(tick \"This is verified ground truth\" before running to apply)."
                )
            self._ct_calib_status_lbl.setText(
                f"{report.splitlines()[-1] if report else ''} This fish measured "
                f"branch_radius={measured_this_fish}.{applied_note}"
            )

        timer.timeout.connect(_poll)
        timer.start(500)
        self._branch_calib_job["timer"] = timer

    def _write_cellpose_best_pointer(self, job, best_epoch):
        """Write the best-checkpoint pointer file (see
        _tj.write_best_checkpoint_pointer) for `job` (a dict with
        'data_dir'/'model_name' keys — either self._cellpose_job during
        live polling, or a config snapshot when finalizing a job that
        finished while napari was closed) and return a status suffix
        describing the outcome. Never raises — a missing/racing
        checkpoint file shouldn't crash the poll loop or block reopening
        napari on a stale job."""
        data_dir = job.get("data_dir")
        model_name = job.get("model_name")
        if not (data_dir and model_name and best_epoch is not None):
            return ""
        models_dir = Path(data_dir) / "models"
        try:
            pointer = _tj.write_best_checkpoint_pointer(models_dir, model_name, best_epoch)
            return f"  Recommended-model pointer: {pointer.name}"
        except FileNotFoundError as exc:
            return f"  (could not write recommended-model pointer: {exc})"

    def _start_cellpose_polling(self):
        """Same coarse (8s) log-tail/status poll as MONAI (_start_monai_polling) —
        see that method's docstring for why the interval is this coarse."""
        timer = QTimer(self)

        def _poll():
            pid = self._cellpose_job.get("pid")
            log_path = self._cellpose_job.get("log_path")
            if log_path:
                self._ct_log_view.setPlainText(_tj.tail_log(log_path))
                sb = self._ct_log_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            patience = self._cellpose_job.get("patience", 0)
            if pid and log_path and _tj.is_running(pid) and patience > 0:
                chk = _tj.patience_exceeded(log_path, _tj.CELLPOSE_METRIC, patience)
                if chk["exceeded"]:
                    _tj.kill_process_tree(pid)
                    timer.stop()
                    self._cellpose_job["timer"] = None
                    pointer_msg = self._write_cellpose_best_pointer(self._cellpose_job, chk["best_epoch"])
                    self._ct_status_lbl.setText(
                        f"Early-stopped (PID {pid}): {chk['checkpoints_since_best']} checkpoints "
                        f"without improvement (best {_tj.CELLPOSE_METRIC['label']}={chk['best_value']:.4f} "
                        f"at epoch {chk['best_epoch']})." + pointer_msg
                    )
                    self._ct_launch_btn.setEnabled(True)
                    self._ct_stop_btn.setEnabled(False)
                    return

            if pid and not _tj.is_running(pid):
                timer.stop()
                self._cellpose_job["timer"] = None
                if log_path:
                    best = _tj.patience_exceeded(log_path, _tj.CELLPOSE_METRIC, patience=0)
                else:
                    best = None
                if best and best["best_epoch"] is not None:
                    pointer_msg = self._write_cellpose_best_pointer(self._cellpose_job, best["best_epoch"])
                    self._ct_status_lbl.setText(
                        f"Training process (PID {pid}) has stopped. Best "
                        f"{_tj.CELLPOSE_METRIC['label']}={best['best_value']:.4f} at epoch "
                        f"{best['best_epoch']}." + pointer_msg
                    )
                else:
                    self._ct_status_lbl.setText(f"Training process (PID {pid}) has stopped.")
                self._ct_launch_btn.setEnabled(True)
                self._ct_stop_btn.setEnabled(False)

        timer.timeout.connect(_poll)
        timer.start(8000)
        self._cellpose_job["timer"] = timer
        _poll()

    def _on_ct_launch_training(self):
        if self._cellpose_job.get("pid") and _tj.is_running(self._cellpose_job["pid"]):
            self._ct_status_lbl.setText("A training job is already running.")
            return

        script_path = self._state["config"].get("cellpose_train_script_path") or str(_ait.DEFAULT_CELLPOSE_TRAIN_SCRIPT)
        if not Path(script_path).exists():
            self._ct_status_lbl.setText(f"ERROR: train_xzyz.py not found at {script_path}")
            return

        data_dir = self._ct_data_dir_edit.text().strip()
        if not data_dir:
            self._ct_status_lbl.setText("ERROR: data dir required — run Extract Crops first or set it manually.")
            return
        pretrained = self._ct_pretrained_edit.text().strip() or "cpsam"
        model_name = self._ct_modelname_edit.text().strip() or "cpsam_microglia_xzyz"

        argv = _ait.build_cellpose_train_argv(
            script_path, data_dir, model_name, pretrained,
            self._ct_epochs_spin.value(), self._ct_batch_spin.value(),
            self._ct_saveevery_spin.value(), self._ct_logevery_spin.value(),
            self._ct_lr_spin.value(), self._ct_branchweight_spin.value(),
            self._ct_branchradius_spin.value(),
        )
        conda_env = self._state["config"].get("cellpose_conda_env", "cellpose")
        cwd = Path(script_path).parent
        session = f"cellpose_train_{datetime.now():%Y%m%d_%H%M%S}"
        log_dir = Path(data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gui_launch_{session}.log"

        notify = None
        if self._ct_notify_cb.isChecked():
            notify, notify_err = self._build_notify_cfg("Cellpose-SAM", _tj.CELLPOSE_METRIC)
            if notify_err:
                self._ct_status_lbl.setText(notify_err)
                return

        try:
            pid = _tj.launch_detached(argv, cwd, log_path, conda_env, notify=notify)
        except Exception as exc:
            self._ct_status_lbl.setText(f"ERROR launching: {exc}")
            return

        self._cellpose_job["pid"] = pid
        self._cellpose_job["log_path"] = str(log_path)
        self._cellpose_job["patience"] = self._ct_patience_early_spin.value()
        self._cellpose_job["data_dir"] = data_dir
        self._cellpose_job["model_name"] = model_name
        self._save_cfg(
            cellpose_active_pid=pid, cellpose_active_log=str(log_path),
            cellpose_crops_data_dir=data_dir,
            cellpose_patience=self._ct_patience_early_spin.value(),
            cellpose_model_name=model_name,
        )
        self._ct_launch_btn.setEnabled(False)
        self._ct_stop_btn.setEnabled(True)
        self._ct_status_lbl.setText(f"Launched (PID {pid}) — log: {log_path}")
        self._start_cellpose_polling()

    def _on_ct_stop_training(self):
        pid = self._cellpose_job.get("pid")
        if not pid:
            return
        _tj.kill_process_tree(pid)
        self._ct_status_lbl.setText(f"Stopped (PID {pid}).")
        self._ct_launch_btn.setEnabled(True)
        self._ct_stop_btn.setEnabled(False)
        self._save_cfg(cellpose_active_pid=None, cellpose_active_log="")
        if self._cellpose_job.get("timer"):
            self._cellpose_job["timer"].stop()
            self._cellpose_job["timer"] = None

    def _resume_cellpose_job_if_active(self):
        """Mirrors _resume_monai_job_if_active — see its docstring, including
        the finalize-on-reopen branch for a job that finished while napari
        was closed (the only place that job's pointer file can still get
        written, since the live _poll() loop that normally does it never
        ran while napari was shut)."""
        cfg = self._state.get("config", {})
        pid = cfg.get("cellpose_active_pid")
        log_path = cfg.get("cellpose_active_log")
        if not pid:
            return
        data_dir = cfg.get("cellpose_crops_data_dir", "")
        model_name = cfg.get("cellpose_model_name", "")
        if _tj.is_running(pid):
            self._cellpose_job["pid"] = pid
            self._cellpose_job["log_path"] = log_path
            self._cellpose_job["patience"] = cfg.get("cellpose_patience", 0)
            self._cellpose_job["data_dir"] = data_dir
            self._cellpose_job["model_name"] = model_name
            self._ct_launch_btn.setEnabled(False)
            self._ct_stop_btn.setEnabled(True)
            self._ct_status_lbl.setText(f"Resumed monitoring PID {pid} (already running).")
            self._start_cellpose_polling()
        else:
            self._save_cfg(cellpose_active_pid=None, cellpose_active_log="")
            best = _tj.patience_exceeded(log_path, _tj.CELLPOSE_METRIC, patience=0) if log_path else None
            if best and best["best_epoch"] is not None:
                pointer_msg = self._write_cellpose_best_pointer(
                    dict(data_dir=data_dir, model_name=model_name), best["best_epoch"]
                )
                self._ct_status_lbl.setText(
                    f"Training (PID {pid}) finished while napari was closed. Best "
                    f"{_tj.CELLPOSE_METRIC['label']}={best['best_value']:.4f} at epoch "
                    f"{best['best_epoch']}." + pointer_msg
                )
            else:
                self._ct_status_lbl.setText(f"Training (PID {pid}) is no longer running.")

    def _on_es_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT fish's image", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._es_img_edit.setText(path_str)

    def _on_es_browse_lbl(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._es_lbl_edit.setText(path_str)

    def _on_es_read_recommended(self):
        """Reads <model_name>_best_recommended.txt (written by the Train
        Cellpose-SAM section on stop -- see _write_cellpose_best_pointer)
        and fills the Recommended epoch spinbox from it."""
        data_dir = self._ct_data_dir_edit.text().strip()
        model_name = self._ct_modelname_edit.text().strip()
        if not (data_dir and model_name):
            self._es_status_lbl.setText("ERROR: set Data dir/model_name in Train Cellpose-SAM above first.")
            return
        models_dir = Path(data_dir) / "models"
        target = _tj.read_best_checkpoint_pointer(models_dir, model_name)
        if target is None:
            self._es_status_lbl.setText(
                f"No recommended-checkpoint pointer found yet for '{model_name}' in {models_dir} "
                "— it's written once a training run stops. Set the epoch manually instead."
            )
            return
        try:
            epoch = int(target.name.rsplit("_epoch_", 1)[1])
        except (IndexError, ValueError):
            self._es_status_lbl.setText(f"Could not parse an epoch number from pointer target: {target.name}")
            return
        self._es_recepoch_spin.setValue(epoch)
        self._es_status_lbl.setText(f"Read epoch {epoch} from {target.name}.")

    def _on_es_run_sweep(self):
        if self._epoch_sweep_job.get("thread") and self._epoch_sweep_job["thread"].is_alive():
            self._es_status_lbl.setText("A sweep is already running.")
            return

        img_path = self._es_img_edit.text().strip()
        lbl_path = self._es_lbl_edit.text().strip()
        if not (img_path and lbl_path):
            self._es_status_lbl.setText("ERROR: set GT image and GT labels paths first.")
            return
        if not Path(img_path).exists():
            self._es_status_lbl.setText(f"ERROR: GT image not found: {img_path}")
            return
        if not Path(lbl_path).exists():
            self._es_status_lbl.setText(f"ERROR: GT labels not found: {lbl_path}")
            return

        data_dir = self._ct_data_dir_edit.text().strip()
        model_name = self._ct_modelname_edit.text().strip()
        if not (data_dir and model_name):
            self._es_status_lbl.setText("ERROR: set Data dir/model_name in Train Cellpose-SAM above first.")
            return
        models_dir = Path(data_dir) / "models"

        recommended = self._es_recepoch_spin.value()
        save_every = self._es_saveevery_spin.value()
        available = []
        prefix = f"{model_name}_epoch_"
        if models_dir.exists():
            for p in models_dir.iterdir():
                if p.name.startswith(prefix) and p.is_file():
                    try:
                        available.append(int(p.name[len(prefix):]))
                    except ValueError:
                        pass
        if recommended not in available:
            self._es_status_lbl.setText(
                f"ERROR: no checkpoint '{model_name}_epoch_{recommended:04d}' in {models_dir} "
                "— set Recommended epoch to a value that actually has a checkpoint on disk."
            )
            return

        try:
            epochs = _esw.pick_sweep_epochs(
                recommended, save_every, available,
                n_below=self._es_below_spin.value(), n_above=self._es_above_spin.value(),
            )
        except ValueError as exc:
            self._es_status_lbl.setText(f"ERROR: {exc}")
            return

        scale_zyx = (self._es_scalez_spin.value(), self._es_scalexy_spin.value(), self._es_scalexy_spin.value())
        cellprob = self._es_cellprob_spin.value()
        flow = _FLOW_THRESHOLD_FIXED
        n_cells = self._es_ncells_spin.value()
        pad_z = self._es_padz_spin.value()
        pad_xy = self._es_padxy_spin.value()

        cancel_event = threading.Event()
        result = {}
        progress = {"lines": []}
        progress_lock = threading.Lock()

        def _progress_cb(msg):
            with progress_lock:
                progress["lines"].append(msg)

        def _worker():
            try:
                sweep = _esw.run_epoch_sweep(
                    img_path, lbl_path, models_dir, model_name, epochs, scale_zyx,
                    cellprob=cellprob, flow=flow, n_cells=n_cells,
                    pad_z=pad_z, pad_xy=pad_xy, gpu=torch.cuda.is_available(),
                    progress_cb=_progress_cb, cancel_event=cancel_event,
                )
                result["sweep"] = sweep
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._epoch_sweep_job["thread"] = thread
        self._epoch_sweep_job["cancel_event"] = cancel_event
        self._epoch_sweep_job["result"] = result
        self._epoch_sweep_job["progress"] = progress
        self._epoch_sweep_job["progress_lock"] = progress_lock
        self._epoch_sweep_job["recommended"] = recommended
        self._epoch_sweep_job["models_dir"] = models_dir
        self._epoch_sweep_job["model_name"] = model_name
        self._es_run_btn.setEnabled(False)
        self._es_stop_btn.setEnabled(True)
        self._es_report_view.clear()
        self._es_status_lbl.setText(f"Sweeping epochs {epochs} across up to {n_cells} cells ...")
        thread.start()
        self._start_epoch_sweep_polling()

    def _on_es_stop_sweep(self):
        cancel_event = self._epoch_sweep_job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
        self._es_status_lbl.setText("Cancelling — finishing the current checkpoint's inferences, then stopping...")
        self._es_stop_btn.setEnabled(False)

    def _start_epoch_sweep_polling(self):
        """Fast (500ms) poll matching other bounded, minutes-to-an-hour-class
        background jobs in this tab (Extract Crops, Prepare Training Data) --
        this one is not launched as a detached process, so it does not
        survive napari closing; unlike Launch Training, it's meant to be
        watched, and is short enough not to need that guarantee."""
        timer = QTimer(self)
        job = self._epoch_sweep_job

        def _poll():
            with job["progress_lock"]:
                lines = list(job["progress"]["lines"])
                job["progress"]["lines"].clear()
            if lines:
                self._es_report_view.append("\n".join(lines))
                sb = self._es_report_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if job["thread"].is_alive():
                return
            timer.stop()
            job["timer"] = None
            self._es_run_btn.setEnabled(True)
            self._es_stop_btn.setEnabled(False)

            result = job["result"]
            if "error" in result:
                self._es_is_gt_cb.setChecked(False)
                self._es_status_lbl.setText(f"ERROR during sweep: {result['error'].splitlines()[0]}")
                self._es_report_view.append("\n" + result["error"])
                self._maybe_send_notify(
                    self._es_notify_cb,
                    "[ZF-Microglia-AI] Verify Best Epoch sweep failed",
                    f"Verify Best Epoch (GT Sweep) failed:\n\n{result['error']}",
                )
                return

            sweep = result["sweep"]
            report = _esw.format_sweep_report(sweep, job["recommended"])
            self._es_report_view.setPlainText(report)
            # One-shot, same as Tab 3 Statistics: only a GT-confirmed run
            # may rewrite the checkpoint pointer / swap Tab 2's active
            # model, and the box never stays armed for the next run.
            is_gt = self._es_is_gt_cb.isChecked()
            self._es_is_gt_cb.setChecked(False)
            if sweep.get("cancelled"):
                self._es_status_lbl.setText("Sweep cancelled — partial results above.")
            elif sweep["best_epoch"] == job["recommended"]:
                self._es_status_lbl.setText(
                    f"Confirmed: epoch {job['recommended']} is the sweep's best "
                    f"(avg IoU={sweep['per_epoch_avg'][sweep['best_epoch']]['iou']:.1f}%)."
                )
            elif not is_gt:
                best_epoch = sweep["best_epoch"]
                self._es_status_lbl.setText(
                    f"Sweep's best is epoch {best_epoch} "
                    f"(avg IoU={sweep['per_epoch_avg'][best_epoch]['iou']:.1f}%), "
                    f"not the recommended {job['recommended']}. Not GT-verified — checkpoint "
                    f"pointer / active model NOT changed (tick \"This is verified ground "
                    f"truth\" before running to apply)."
                )
            else:
                best_epoch = sweep["best_epoch"]
                applied_msg = ""
                # Rewrite the recommended-checkpoint pointer to the sweep-
                # confirmed epoch (models_dir already resolved at launch
                # time, so use _tj directly rather than the data_dir-based
                # helper) and auto-load that checkpoint as Tab 2's active
                # model -- same update sequence as _on_browse_cp_model.
                try:
                    pointer = _tj.write_best_checkpoint_pointer(job["models_dir"], job["model_name"], best_epoch)
                    checkpoint_path = job["models_dir"] / f"{job['model_name']}_epoch_{best_epoch:04d}"
                    self._state["cellpose_model_path"] = checkpoint_path
                    self._cp_model_lbl.setText(str(checkpoint_path))
                    self._save_cfg(cellpose_model_path=str(checkpoint_path))
                    applied_msg = f" Pointer updated ({pointer.name}) and applied to Tab 2 as the active model."
                except FileNotFoundError as exc:
                    applied_msg = f" (could not update pointer/active model: {exc})"
                self._es_status_lbl.setText(
                    f"Sweep's best is epoch {best_epoch} "
                    f"(avg IoU={sweep['per_epoch_avg'][best_epoch]['iou']:.1f}%), "
                    f"not the recommended {job['recommended']}.{applied_msg}"
                )

            self._maybe_send_notify(
                self._es_notify_cb,
                "[ZF-Microglia-AI] Verify Best Epoch sweep done",
                f"Verify Best Epoch (GT Sweep) finished.\n\n{self._es_status_lbl.text()}",
            )

        timer.timeout.connect(_poll)
        timer.start(500)
        job["timer"] = timer

    def _get_layer_scale(self):
        """
        Return (z, y, x) scale in µm.

        Priority:
          1. metadata from a file loaded via Open button
          2. scale of the active layer (respects per-layer scale set by reader)
          3. default (1.0, 1.0, 1.0)
        """
        meta = self._state.get("metadata")
        if meta:
            return meta["scale"]
        lyr = self._active_layer()
        if lyr is not None:
            sc = lyr.scale
            if len(sc) == 3:
                return tuple(float(v) for v in sc)
        return (1.0, 1.0, 1.0)

    def _refresh_meta_lbl(self):
        scale = self._get_layer_scale()
        z, y, x = scale
        meta = self._state.get("metadata")
        if meta:
            source     = meta.get("source", "Unknown")
            anisotropy = meta.get("anisotropy", 1.0)
        else:
            xy         = (x + y) / 2.0
            anisotropy = z / xy if xy > 0 else 1.0
            source     = (
                "from layer scale"
                if scale != (1.0, 1.0, 1.0)
                else "default (1, 1, 1)"
            )
        line1 = f"Z={z:.4f}  Y={y:.4f}  X={x:.4f} \u00b5m"
        line2 = f"Anisotropy {anisotropy:.2f}:1  |  {source}"
        self._meta_lbl.setText(f"{line1}\n{line2}")

    def _active_layer(self):
        """Return the active (selected) Image layer, or None."""
        active = self._viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Image):
            return active
        # fall back to topmost Image layer
        for lyr in reversed(self._viewer.layers):
            if isinstance(lyr, napari.layers.Image):
                return lyr
        return None

    def _refresh_stats_layers(self, *_):
        """Repopulate image and shapes layer combos in the Statistics tab."""
        # Image layers
        cur_img = self._stats_image_combo.currentData()
        cur_correct_img = self._correct_signal_combo.currentData()
        cur_split_img = self._split_signal_combo.currentData()
        self._stats_image_combo.blockSignals(True)
        self._stats_image_combo.clear()
        self._stats_image_combo.addItem("None", None)
        self._correct_signal_combo.blockSignals(True)
        self._correct_signal_combo.clear()
        self._correct_signal_combo.addItem("None", None)
        self._split_signal_combo.blockSignals(True)
        self._split_signal_combo.clear()
        self._split_signal_combo.addItem("None", None)
        for lyr in self._viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                self._stats_image_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur_img:
                    self._stats_image_combo.setCurrentIndex(
                        self._stats_image_combo.count() - 1
                    )
                self._correct_signal_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur_correct_img:
                    self._correct_signal_combo.setCurrentIndex(
                        self._correct_signal_combo.count() - 1
                    )
                self._split_signal_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur_split_img:
                    self._split_signal_combo.setCurrentIndex(
                        self._split_signal_combo.count() - 1
                    )
        self._stats_image_combo.blockSignals(False)
        self._correct_signal_combo.blockSignals(False)
        self._split_signal_combo.blockSignals(False)

        # Shapes layers
        cur_shp = self._stats_shapes_combo.currentData()
        self._stats_shapes_combo.blockSignals(True)
        self._stats_shapes_combo.clear()
        self._stats_shapes_combo.addItem("None", None)
        for lyr in self._viewer.layers:
            if isinstance(lyr, napari.layers.Shapes):
                self._stats_shapes_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur_shp:
                    self._stats_shapes_combo.setCurrentIndex(
                        self._stats_shapes_combo.count() - 1
                    )
        self._stats_shapes_combo.blockSignals(False)

        # Labels layers (Score Against GT)
        for combo in (self._gtscore_pred_combo, self._gtscore_gt_combo):
            cur = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("None", None)
            for lyr in self._viewer.layers:
                if isinstance(lyr, napari.layers.Labels):
                    combo.addItem(lyr.name, lyr.name)
                    if lyr.name == cur:
                        combo.setCurrentIndex(combo.count() - 1)
            combo.blockSignals(False)

    def _refresh_layer_info(self, *_):
        lyr = self._active_layer()
        if lyr is None:
            self._layer_info.setText("  — no image layers yet —")
            self._meta_lbl.setText("  — voxel info unavailable —")
            self._update_labels_section_visibility()
            return
        d = lyr.data
        self._layer_info.setText(f'  "{lyr.name}"\n  {d.shape}  {d.dtype}')
        self._refresh_meta_lbl()
        self._update_labels_section_visibility()

    def _update_labels_section_visibility(self):
        """
        Show each route's labeling tool based on whether a layer with that
        route's background-removal suffix exists anywhere in the viewer —
        not just whether it's the single currently-active layer (see
        _BG_SUFFIX):
          _ExtRm present -> show Cellpose-SAM Segmentation
          _NoBG  present -> show Pixel Classifier
          both present   -> both shown together (e.g. comparing routes on
                             the same fish)
          neither        -> neither, with a hint explaining what's needed

        Existence-based rather than active-layer-based per explicit
        instruction: the old logic only ever looked at _active_layer()
        (which only returns Image layers, falling back to the topmost one),
        so selecting the Labels layer itself -- or simply having both an
        _ExtRm and a _NoBG layer loaded at once -- could hide a route's
        tools even though a matching layer was right there in the viewer.
        """
        img_names = [
            lyr.name for lyr in self._viewer.layers
            if isinstance(lyr, napari.layers.Image)
        ]
        has_extrm = any(n.endswith("_ExtRm") for n in img_names)
        has_nobg  = any(n.endswith("_NoBG") for n in img_names)

        self._cellpose_group.setVisible(has_extrm)
        self._pixel_classifier_group.setVisible(has_nobg)

        if has_extrm and has_nobg:
            self._labels_mode_hint.setText(
                "Both _ExtRm and _NoBG layers are present → showing "
                "Cellpose-SAM Segmentation and Pixel Classifier."
            )
        elif has_extrm:
            self._labels_mode_hint.setText(
                "An _ExtRm layer is present → showing Cellpose-SAM Segmentation."
            )
        elif has_nobg:
            self._labels_mode_hint.setText(
                "A _NoBG layer is present → showing Pixel Classifier."
            )
        else:
            self._labels_mode_hint.setText(
                "Select a brain_only layer from Tab 1 (_ExtRm or _NoBG) to see "
                "labeling options here."
            )

        # Downstream label tools (Resort/Remove Debris/Split/Join/Save) work
        # directly on whatever Labels layer exists — none of them are
        # route-specific (unlike the Cellpose-SAM-only "Re-run This Cell
        # Only", which stays inside _cellpose_group and is correctly gated
        # by route). Gating this group on _active_layer()'s route suffix was
        # wrong: _active_layer() only ever returns an Image layer (it falls
        # back to the topmost one), so selecting the Labels layer itself —
        # exactly what "Use selected" requires for Split/Join — could hide
        # this whole group if the fallback Image layer's suffix didn't
        # match either route. Same has_labels condition Statistics already
        # uses below, since labels can persist even after switching the
        # active layer.
        has_labels = any(isinstance(l, napari.layers.Labels) for l in self._viewer.layers)
        self._downstream_label_tools.setVisible(has_labels)

        # Statistics needs an actual Labels layer to operate on. The tab
        # itself always stays visible (explain instead of hide, per explicit
        # instruction) -- _update_stats_tab_content swaps its content for an
        # explanatory hint instead.
        self._update_stats_tab_content(has_labels)

    # ------------------------------------------------------------------ #
    # Public helper (used by __main__.py for CLI pre-loading)
    # ------------------------------------------------------------------ #

    def _add_channels(self, path, channels):
        """Add a list of (volume, name, metadata) channel tuples as image layers."""
        colormaps = ["gray", "green", "magenta", "cyan"]
        self._state["last_file_path"] = path
        self._state["metadata"]       = channels[0][2]   # shared metadata
        for i, (volume, name, metadata) in enumerate(channels):
            cmap = colormaps[i % len(colormaps)]
            self._viewer.add_image(
                volume, name=name, colormap=cmap, scale=metadata["scale"]
            )
        self._refresh_layer_info()

    def preload(self, path):
        """Load a file programmatically (CLI / __main__.py use)."""
        path = Path(path)
        self._status(f"Loading {path.name}...")
        try:
            channels = load_file(path)
            self._add_channels(path, channels)
            n = len(channels)
            shape = channels[0][0].shape
            self._status(f"Loaded: {path.name}  {n} ch  {shape}")
        except Exception as exc:
            self._status(f"ERROR: {exc}")
            print(f"ERROR loading {path.name}: {exc}")

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _on_open(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open confocal stack",
            "",
            "Confocal stacks (*.tif *.tiff *.ims);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        self._status(f"Loading {path.name}… (IMS files may take ~1 min)")
        self._open_btn.setEnabled(False)

        result = {}

        def _worker():
            try:
                result["channels"] = load_file(path)
            except Exception as exc:
                result["error"] = str(exc)
                import traceback as _tb
                _tb.print_exc()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            self._open_btn.setEnabled(True)
            if "error" in result:
                self._status(f"ERROR: {result['error']}")
                return
            channels = result["channels"]
            self._add_channels(path, channels)
            n     = len(channels)
            shape = channels[0][0].shape
            self._status(f"Loaded: {path.name}  {n} ch  {shape}")

        timer.timeout.connect(_poll)
        timer.start(500)

    def _output_dir(self) -> Path:
        """
        Return (and create) the output folder for all saved files.
        Folder = <original_file_parent> / <original_file_stem>
        Falls back to current working directory if no file has been opened.
        """
        fp = self._state.get("last_file_path")
        if fp:
            out = fp.parent / fp.stem
        else:
            out = Path(".")
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _get_notify_creds(self):
        """Read the shared Email notification fields (Tab 5 -- Sweeps &
        Utilities, General category) and return (creds_dict_or_None,
        error_or_None). creds is None when the feature is disabled
        (recipient left blank) -- not an error. Persists the SMTP
        password to the OS's encrypted credential store (Windows
        Credential Manager / macOS Keychain / Linux Secret Service via
        `keyring` -- see _secrets.py), not the plugin's own plaintext
        config.json, so notification works out of the box every session
        without re-entering it. Deliberately different from the
        Statistics tab's LLM API key policy: this password is meant to
        always be a Gmail App Password (or equivalent scoped credential
        for another provider), not a real account password -- Google
        issues it specifically for unattended third-party use and it's
        revocable independently of the main account password.

        If the OS credential store is unavailable (e.g. Linux with no
        unlocked Secret Service session -- confirmed to actually happen
        on this project's own workstation, not just a hypothetical),
        _secrets.set_secret() falls back to a local Fernet-encrypted
        file automatically (see _secrets.py) rather than losing the
        value or writing it in plaintext -- this only prints a warning
        (never returned as `err`, never blocks the operation) if *both*
        the OS store and that fallback fail.

        Shared by every "Email me when done" checkbox in the plugin (Tab 1
        Run, Tab 2 Cellpose-SAM Segmentation, Tab 4's two training
        launchers via _build_notify_cfg below, Tab 5's Cellprob/Large-
        contact and Best-Epoch sweeps) -- one set of credentials, opted
        into per tool."""
        to_addr = self._notify_to_edit.text().strip()
        if not to_addr:
            return None, None
        smtp_host = self._notify_smtp_host_edit.text().strip() or "smtp.gmail.com"
        smtp_port = self._notify_smtp_port_spin.value()
        smtp_user = self._notify_smtp_user_edit.text().strip()
        smtp_password = self._notify_smtp_password_edit.text()
        if not smtp_user or not smtp_password:
            return None, (
                "ERROR: Notify email is set but SMTP username/password is missing "
                "— fill both in, or clear the Notify email field in Tab 5's Email "
                "notification panel to proceed without notification."
            )
        self._save_cfg(
            notify_email_to=to_addr, notify_smtp_host=smtp_host,
            notify_smtp_port=smtp_port, notify_smtp_user=smtp_user,
        )
        secret_err = _secrets.set_secret("notify_smtp_password", smtp_password)
        if secret_err:
            print(f"SMTP password not saved for next session: {secret_err}")
        return dict(
            to_addr=to_addr, smtp_host=smtp_host, smtp_port=smtp_port,
            smtp_user=smtp_user, smtp_password=smtp_password,
        ), None

    def _build_notify_cfg(self, job_label, metric_cfg):
        """Wraps _get_notify_creds() with the extra job_label/metric_cfg
        fields _tj.launch_detached's supervisor-script `notify` param
        needs (Tab 4's two detached training launchers only)."""
        creds, err = self._get_notify_creds()
        if creds is None:
            return None, err
        return dict(**creds, job_label=job_label, metric_cfg=metric_cfg), None

    def _maybe_send_notify(self, checkbox, subject, body):
        """For the in-process "Email me when done" checkboxes (Tab 1 Run,
        Tab 2 Cellpose-SAM Segmentation, Tab 5's Cellprob/Large-contact and
        Best-Epoch sweeps) -- call from the worker thread right before it
        finishes (not the GUI/poll thread, so the SMTP round-trip doesn't
        block the UI). No-op if the checkbox is unchecked. Errors (bad
        creds, network failure) are swallowed into a print() rather than
        raised, matching send_notification_email's own contract -- a
        broken email config must never take down the operation whose
        result it was reporting."""
        if checkbox is None or not checkbox.isChecked():
            return
        creds, err = self._get_notify_creds()
        if err:
            print(f"Email notification skipped: {err}")
            return
        if creds is None:
            print("Email notification checkbox is checked but Notify email "
                  "(Tab 5, General) is blank -- skipped.")
            return
        send_err = _tj.send_notification_email(
            creds["to_addr"], creds["smtp_host"], creds["smtp_port"],
            creds["smtp_user"], creds["smtp_password"], subject, body,
        )
        if send_err:
            print(f"Email notification failed: {send_err}")

    def _on_send_test_email(self):
        """Verify SMTP credentials actually work without waiting on any
        real 30+min operation -- sends one email immediately using
        exactly the same _get_notify_creds()/send_notification_email()
        path every "Email me when done" checkbox uses, just with a fixed
        test subject/body instead of a real result. Runs in a background
        thread since a misconfigured host/port can hang for the full
        SMTP timeout (30s) rather than fail instantly, and that shouldn't
        freeze napari."""
        creds, err = self._get_notify_creds()
        if err:
            self._notify_test_status_lbl.setText(err)
            return
        if creds is None:
            self._notify_test_status_lbl.setText(
                "ERROR: Notify email is blank -- fill it in above first."
            )
            return

        self._notify_test_btn.setEnabled(False)
        self._notify_test_status_lbl.setText(f"Sending to {creds['to_addr']}...")

        result = {}

        def _worker():
            result["error"] = _tj.send_notification_email(
                creds["to_addr"], creds["smtp_host"], creds["smtp_port"],
                creds["smtp_user"], creds["smtp_password"],
                "[ZF-Microglia-AI] Test email",
                "This is a test email from the ZF-Microglia-AI napari plugin's "
                "Email notification panel (Tab 5, General category).\n\n"
                "If you received this, your SMTP settings are correct and any "
                "'Email me when done' checkbox in the plugin will work.",
            )

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            self._notify_test_btn.setEnabled(True)
            if result["error"]:
                self._notify_test_status_lbl.setText(f"ERROR: {result['error']}")
            else:
                self._notify_test_status_lbl.setText(f"Sent to {creds['to_addr']} -- check your inbox.")

        timer.timeout.connect(_poll)
        timer.start(200)

    def _save_cfg(self, **kwargs) -> None:
        """Merge kwargs into the config and persist."""
        cfg = self._state.get("config", {})
        cfg.update(kwargs)
        self._state["config"] = cfg
        _save_config(cfg)

    def _current_min_volume(self) -> int:
        """Min volume (vox) -- informative only in the UI (see the Common
        Settings note by self._area_value_lbl), not a slider: only ever
        moved by a Tab 5 GT sweep or a GT-verified Generate Statistics
        run, never hand-tuned. Reads straight from config rather than a
        widget so every route/sweep launcher shares one source of truth."""
        return self._state.get("config", {}).get("min_volume_vox", 7500)

    def _update_gt_history(self, config_key: str, fish_key: str, value, mode: str = "mean"):
        """Persist this fish's contribution to config_key's cross-fish GT
        sweep history and return the aggregated recommendation.

        Every GT-sweep tool used to either (a) blindly overwrite its
        target config value with whatever this one run found best, with
        no memory of any other fish ever swept, or (b) for a handful of
        floor-style values, track only a single running scalar compared
        against the newest measurement -- which quietly keeps a stale
        value forever if the same fish is re-swept after its GT is
        corrected, since there is no per-fish record to update. Both are
        fixed by keeping a real {fish_key: value} history per config
        key: re-running against the same fish_key updates that entry in
        place instead of duplicating or being unable to revise it, and
        the recommendation is always recomputed fresh from the complete
        history rather than carried forward as a single running number.

        mode="min"  : never-rising floor (min_volume, min_hole_size,
                      gt_min -- "smallest real X any fish has proven,
                      never go higher than that" -- a must-not-exceed
                      constraint, so the tightest bound across fish is
                      the smallest measurement).
        mode="max"  : never-falling ceiling (branch_radius -- the radius
                      needed to fully cover a real branch's thinnest
                      cross-section is a must-reach-at-least constraint;
                      a thicker "thin branch" measured in any fish sets
                      a higher bar that must still be met, so the
                      tightest bound across fish is the largest
                      measurement -- the opposite direction from
                      mode="min", not a copy-paste of it).
        mode="mean" : average across every fish swept so far (BG
                      Threshold, Erosion, Sigma XY/Z, Cellprob,
                      Large-contact, MONAI Threshold -- each fish's
                      sweep is just that fish's own local optimum, with
                      no safe direction to bias toward, so the average
                      is the representative value across the fish
                      actually validated so far).
        """
        history_key = f"{config_key}_history"
        history = dict(self._state.get("config", {}).get(history_key, {}))
        history[fish_key] = value
        self._save_cfg(**{history_key: history})
        values = list(history.values())
        if mode == "min":
            return min(values)
        if mode == "max":
            return max(values)
        return sum(values) / len(values)

    def _update_merge_stats_history(self, fish_key: str, merge_stats: dict) -> dict:
        """Persist this fish's raw cp_masks-vs-GT merge_stats (the 4 gap/
        contact sample lists from measure_merge_params_from_prediction)
        and return a freshly-pooled recommend_merge_params() across every
        fish calibrated so far.

        Unlike _update_gt_history, each entry here is a dict of sample
        lists, not a scalar -- max_gap/min_contact are jointly optimized
        against the pooled OR-combined-rule sample set (see
        recommend_merge_params), not averaged per-fish then combined.
        """
        history_key = "cellpose_merge_stats_history"
        history = dict(self._state.get("config", {}).get(history_key, {}))
        history[fish_key] = merge_stats
        self._save_cfg(**{history_key: history})
        return _ksw.recommend_merge_params(list(history.values()))

    # ---------------------------------------------------------------- #
    # GT Toolkit Tuning Tool
    # ---------------------------------------------------------------- #

    def _on_gtk_browse_file(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select fish original file", "", "All files (*)")
        if path_str:
            self._gtk_path_edit.setText(path_str)

    def _on_gtk_browse_folder(self):
        path_str = QFileDialog.getExistingDirectory(self, "Select fish folder")
        if path_str:
            self._gtk_path_edit.setText(path_str)

    def _on_gtk_scan(self):
        """Discover this fish's files and fill every folded-in tool's own
        path fields from them -- a preview step on its own (safe to run
        repeatedly), and also the first thing _on_gtk_run() does."""
        path_str = self._gtk_path_edit.text().strip()
        if not path_str:
            self._gtk_status_lbl.setText("ERROR: set a fish original file or folder path first.")
            return None
        p = Path(path_str)
        if not p.exists():
            self._gtk_status_lbl.setText(f"ERROR: path not found: {path_str}")
            return None
        folder, stem, found = _gtk.discover_fish_files(p)
        self._gtk_discovery_view.setPlainText(_gtk.format_discovery_report(folder, stem, found))
        self._gtk_autofill(folder, stem, found)
        self._save_cfg(gtk_last_path=path_str)
        return folder, stem, found

    def _gtk_autofill(self, folder, stem, found):
        """Fill every folded-in tool's own path fields from one
        discovery pass -- keeps each usable individually afterward too
        (e.g. to re-run just one step's grid with a tweaked range),
        already pointed at the right files."""
        original = found["original"]
        ext_rm = found["ext_rm"]
        brain_mask = found["brain_mask"]
        gt = found["ground_truth"]
        raw_or_ext = original or ext_rm

        if original is not None:
            self._bs_img_edit.setText(str(original))
        if brain_mask is not None:
            self._bs_gt_edit.setText(str(brain_mask))

        if raw_or_ext is not None:
            self._ps_img_edit.setText(str(raw_or_ext))
            self._sg_img_edit.setText(str(raw_or_ext))
        if brain_mask is not None:
            self._ps_mask_edit.setText(str(brain_mask))
            self._sg_mask_edit.setText(str(brain_mask))
        if gt is not None:
            self._ps_lbl_edit.setText(str(gt))
            self._sg_lbl_edit.setText(str(gt))

        if ext_rm is not None:
            self._kr_img_edit.setText(str(ext_rm))
        if gt is not None:
            self._kr_gt_edit.setText(str(gt))

        if gt is not None:
            self._ct_calib_gt_edit.setText(str(gt))

        self._gtp_stem_edit.setText(stem)
        if ext_rm is not None:
            self._gtp_img_edit.setText(str(ext_rm))
        if found["cp_masks_corrected"] is not None:
            self._gtp_masks_edit.setText(str(found["cp_masks_corrected"]))
        if found["cp_masks"] is not None:
            self._gtp_raw_edit.setText(str(found["cp_masks"]))
        self._gtp_out_edit.setText(str(folder))

    def _on_gtk_run(self):
        if self._gtk_job.get("running"):
            self._gtk_status_lbl.setText("The GT Toolkit is already running.")
            return
        scan = self._on_gtk_scan()
        if scan is None:
            return
        folder, stem, found = scan
        reasons = _gtk.step_preconditions(found)
        is_gt = self._gtk_is_gt_cb.isChecked()
        self._gtk_is_gt_cb.setChecked(False)

        report_lines = [_gtk.format_discovery_report(folder, stem, found), ""]

        # Step 0: Min/Max volume + Min hole size, straight from GT labels
        # -- instant, no thread needed. Sole surviving source of these
        # three values now that Tab 3 Statistics's own GT checkbox has
        # been removed -- see [[feedback_plugin_ux]] "only the tuning
        # tool tunes values".
        if found["ground_truth"] is not None:
            try:
                gt_labels = tifffile.imread(str(found["ground_truth"])).astype(np.int32)
                _, counts = np.unique(gt_labels[gt_labels > 0], return_counts=True)
                if len(counts):
                    measured_min = int(counts.min())
                    measured_max = int(counts.max())
                    measured_hole = _psw.min_hole_size_from_gt(gt_labels)
                    if is_gt:
                        recommended_min = self._update_gt_history("min_volume_vox", stem, measured_min, mode="min")
                        self._save_cfg(min_volume_recommended_vox=recommended_min, min_volume_vox=recommended_min)
                        self._area_value_lbl.setText(str(recommended_min))

                        recommended_max = self._update_gt_history("max_volume_vox", stem, measured_max, mode="max")
                        self._save_cfg(max_volume_recommended_vox=recommended_max)
                        self._maxvol_ceiling_lbl.setText(
                            f"  is_volume_outlier ceiling (largest confirmed GT cell so far): "
                            f"{recommended_max} vox -- the floor side reuses Min volume above."
                        )
                        self._maxvol_value_lbl.setText(str(recommended_max))

                        recommended_hole = self._update_gt_history("min_hole_size_vox", stem, measured_hole, mode="min")
                        self._save_cfg(min_hole_size_recommended_vox=recommended_hole, min_hole_size_vox=recommended_hole)
                        if recommended_hole > self._hole_slider.maximum():
                            self._hole_slider.setMaximum(recommended_hole)
                            self._hole_spin.setMaximum(recommended_hole)
                        self._hole_slider.setValue(recommended_hole)
                        self._hole_recommended_lbl.setText(
                            f"  Recommended floor (from GT sweeps so far): {recommended_hole} vox"
                        )
                        report_lines.append(
                            f"[Min/Max volume + Min hole size] this fish measured "
                            f"min={measured_min}, max={measured_max}, hole={measured_hole} vox -- "
                            f"applied: Min volume floor={recommended_min}, outlier ceiling={recommended_max}, "
                            f"Min hole size floor={recommended_hole}."
                        )
                    else:
                        report_lines.append(
                            f"[Min/Max volume + Min hole size] this fish measured "
                            f"min={measured_min}, max={measured_max}, hole={measured_hole} vox -- "
                            f"not applied (not GT-verified)."
                        )
            except Exception as exc:
                report_lines.append(f"[Min/Max volume + Min hole size] ERROR: {exc}")
        else:
            report_lines.append("[Min/Max volume + Min hole size] skipped: no GROUND_TRUTH.tif found.")

        self._gtk_report_view.setPlainText("\n".join(report_lines))

        steps = [
            ("monai", "MONAI Threshold/Erosion", self._bs_is_gt_cb, self._on_bs_run_sweep,
             self._brain_sweep_job, self._bs_status_lbl, self._on_bs_stop_sweep),
            ("bg", "BG Threshold/Signal Erosion", self._ps_is_gt_cb, self._on_ps_run_sweep,
             self._pixel_sweep_job, self._ps_status_lbl, self._on_ps_stop_sweep),
            ("sigma", "Smooth sigma XY/Z", self._sg_is_gt_cb, self._on_sg_run_sweep,
             self._sigma_sweep_job, self._sg_status_lbl, self._on_sg_stop_sweep),
            ("cellprob", "Cellprob + Krendl merge params", self._kr_is_gt_cb, self._on_kr_run_sweep,
             self._krendl_sweep_job, self._kr_status_lbl, self._on_kr_stop_sweep),
            # No Stop button/handler exists for this one -- it's a fast
            # CPU-only skeleton+distance-transform measurement, not a
            # multi-hour GPU sweep, so cancellation was never built for
            # it; a no-op stop_fn is fine since it finishes almost
            # immediately regardless.
            ("branch_radius", "Calibrate branch_radius", self._ct_calib_is_gt_cb, self._on_ct_calib_run,
             self._branch_calib_job, self._ct_calib_status_lbl, lambda: None),
        ]
        queue = []
        for key, label, cb, trigger, job, status_lbl, stop_fn in steps:
            ok, reason = reasons[key]
            if ok:
                queue.append((key, label, cb, trigger, job, status_lbl, stop_fn))
            else:
                self._gtk_report_view.append(f"\n[{label}] skipped: {reason}")

        if not queue:
            self._gtk_status_lbl.setText(
                "Nothing more to run -- no folded-in sweep's required files were found."
            )
            self._maybe_send_notify(
                self._gtk_notify_cb, "[ZF-Microglia-AI] GT Toolkit Tuning finished",
                f"GT Toolkit Tuning finished for {stem}.\n\n{self._gtk_report_view.toPlainText()}",
            )
            return

        self._gtk_job["queue"] = queue
        self._gtk_job["step_index"] = 0
        self._gtk_job["is_gt"] = is_gt
        self._gtk_job["stem"] = stem
        self._gtk_job["running"] = True
        self._gtk_run_btn.setEnabled(False)
        self._gtk_stop_btn.setEnabled(True)
        self._gtk_status_lbl.setText(f"Running step 1/{len(queue)}: {queue[0][1]}...")
        self._gtk_launch_step(queue[0], is_gt)

        timer = QTimer(self)
        timer.timeout.connect(self._gtk_poll)
        timer.start(500)
        self._gtk_job["timer"] = timer

    def _gtk_launch_step(self, step, is_gt):
        _key, _label, cb, trigger, _job, _status_lbl, _stop_fn = step
        # The only place in the plugin allowed to tick one of these
        # hidden checkboxes -- see _add_gt_checkbox(..., visible=False).
        cb.setChecked(is_gt)
        trigger()

    def _gtk_poll(self):
        job = self._gtk_job
        queue = job["queue"]
        idx = job["step_index"]
        _key, label, _cb, _trigger, step_job, status_lbl, _stop_fn = queue[idx]
        thread = step_job.get("thread")
        if thread is not None and thread.is_alive():
            return
        # thread has ended, but this step's OWN poll timer (500ms, same
        # cadence) still needs one more tick to finish applying/un-
        # ticking before its status label reflects the final outcome.
        if step_job.get("timer") is not None:
            return

        self._gtk_report_view.append(f"\n===== {label} =====\n{status_lbl.text()}\n")

        idx += 1
        if idx >= len(queue):
            job["timer"].stop()
            job["timer"] = None
            job["running"] = False
            self._gtk_run_btn.setEnabled(True)
            self._gtk_stop_btn.setEnabled(False)
            self._gtk_status_lbl.setText(
                f"GT Toolkit Tuning finished for {job['stem']} -- {len(queue)} step(s) run."
            )
            self._maybe_send_notify(
                self._gtk_notify_cb, "[ZF-Microglia-AI] GT Toolkit Tuning finished",
                f"GT Toolkit Tuning finished for {job['stem']}.\n\n{self._gtk_report_view.toPlainText()}",
            )
            return

        job["step_index"] = idx
        self._gtk_status_lbl.setText(f"Running step {idx + 1}/{len(queue)}: {queue[idx][1]}...")
        self._gtk_launch_step(queue[idx], job["is_gt"])

    def _on_gtk_stop(self):
        job = self._gtk_job
        if not job.get("running"):
            return
        idx = job["step_index"]
        _key, label, _cb, _trigger, _step_job, _status_lbl, stop_fn = job["queue"][idx]
        stop_fn()
        self._gtk_status_lbl.setText(
            f"Cancelling {label} -- will stop after this step, not run the rest of the queue..."
        )
        # Truncate the queue so _gtk_poll's completion path fires once
        # this current step's own cancellation finishes, instead of
        # moving on to the next step.
        job["queue"] = job["queue"][: idx + 1]
        self._gtk_stop_btn.setEnabled(False)

    def _on_browse_model(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            str(_SKIN_SEG_DIR / "models"),
            "PyTorch checkpoints (*.pth)",
        )
        if not path_str:
            return
        p = Path(path_str)
        self._state["model_path"] = p
        self._model_lbl.setText(path_str)
        self._save_cfg(model_path=str(p))
        self._status(f"Model: {p.name}")

    def _on_browse_cp_model(self):
        # Cellpose-SAM checkpoints are typically extensionless files
        # (e.g. "cpsam_microglia_512_multi3_bw_epoch_0150") — no filter.
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Select Cellpose-SAM checkpoint", "", "All files (*)",
        )
        if not path_str:
            return
        p = Path(path_str)
        if not _is_valid_cellpose_checkpoint(p):
            self._labels_status_lbl.setText(
                f"ERROR: {p.name} doesn't look like a valid PyTorch checkpoint "
                f"(not a zip archive) -- selected the wrong file? Not applied."
            )
            return
        self._state["cellpose_model_path"] = p
        self._cp_model_lbl.setText(path_str)
        self._save_cfg(cellpose_model_path=str(p))
        self._labels_status_lbl.setText(f"Cellpose-SAM model: {p.name}")

    def _on_run(self):
        if not self._state["model_path"] or not Path(self._state["model_path"]).is_file():
            self._status("ERROR: model file not found — browse to a .pth file.")
            return
        target = self._active_layer()
        if target is None:
            self._status("ERROR: no image layer selected — open a file and click a layer.")
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._status(f"ERROR: 3D volume required, got {volume.ndim}D {volume.shape}.")
            return

        threshold        = self._thresh_slider.value()
        erosion_voxels   = self._erosion_slider.value()
        bg_mode          = self._bg_group.checkedId()  # 0=off, 1=remove, 2=fill
        bg_tolerance_pct = self._tol_slider.value()
        signal_erosion_voxels = self._sig_erosion_slider.value()
        model_path       = Path(self._state["model_path"])
        stem             = target.name
        file_path        = self._state.get("last_file_path")
        # Prefer scale directly from the target layer (set by reader or Open btn)
        sc = target.scale
        scale = tuple(float(v) for v in sc) if len(sc) == 3 else self._get_layer_scale()
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self._run_btn.setEnabled(False)
        self._status(f"Running on {device} (threshold={threshold:.2f})...")
        self._run_log_view.clear()

        print(f"\n{'='*70}")
        print(f"SKIN-REMOVER — {stem}  shape={volume.shape}")
        print(f"Model    : {model_path.name}")
        print(f"Threshold: {threshold}   Device: {device}")
        print(f"MONAI Erosion : {erosion_voxels} voxel(s)")
        bg_mode_str = {
            0: "Off",
            1: f"Remove outside-brain (BG threshold={bg_tolerance_pct:.2f})",
            2: f"Remove globally (BG threshold={bg_tolerance_pct:.2f}, Signal Erosion={signal_erosion_voxels} vox)",
            3: "Fill sub-background with random noise",
        }[bg_mode]
        print(f"BG mode  : {bg_mode_str}")
        print(f"Scale    : Z={scale[0]:.4f}  Y={scale[1]:.4f}  X={scale[2]:.4f} µm")
        print(f"{'='*70}")

        result = {}
        log_lines = []
        log_lock = threading.Lock()

        def _push_log(line):
            with log_lock:
                log_lines.append(line)

        def _worker():
            try:
                # Both steps share one capture_live_output block -- Step 2
                # (background/skin removal) has its own real print()
                # statements in _background.py (threshold computed, voxels
                # removed/filled) that were previously invisible in the
                # GUI: the old code only wrapped Step 1, so once MONAI's
                # inference progress bar finished, the log view went
                # completely silent for the rest of the run even though
                # background removal on a full-stack volume is itself a
                # real, sometimes slow, whole-volume pass.
                with capture_live_output(_push_log):
                    # Step 1: inference on original volume (never bg-removed)
                    brain_mask, brain_only, eroded_mask = run_inference(
                        volume, model_path, threshold, device, erosion_voxels
                    )

                    # Step 2: optional background processing -- all three
                    # modes use eroded_mask (MONAI Erosion, from Step 1) as
                    # the brain boundary, so that slider still takes effect
                    # everywhere. (Using brain_mask -- the always-un-eroded
                    # mask meant only for saving brain_mask.tif -- would
                    # silently discard whatever the MONAI Erosion slider is
                    # set to.) Mode 2 (noBG stack prep for the Pixel
                    # Classifier) additionally passes Signal Erosion into
                    # remove_global(), which erodes the *retained signal*
                    # after thresholding, not the brain boundary -- a
                    # conceptually different operation (how much background
                    # halo survives immediately around each cell) from
                    # MONAI Erosion's semantic brain-segmentation boundary,
                    # so it must not share MONAI Erosion's slider/value.
                    if bg_mode == 1:
                        vol_proc, *_ = remove_outside_brain(
                            volume, eroded_mask, tolerance_pct=bg_tolerance_pct
                        )
                        brain_only = (vol_proc * eroded_mask).astype(volume.dtype)
                    elif bg_mode == 2:
                        vol_proc, *_ = remove_global(
                            volume, eroded_mask, tolerance_pct=bg_tolerance_pct,
                            signal_erosion_voxels=signal_erosion_voxels,
                        )
                        brain_only = (vol_proc * eroded_mask).astype(volume.dtype)
                    elif bg_mode == 3:
                        brain_only, _ = fill_outside_brain_random(
                            volume, eroded_mask
                        )

                result["brain_mask"] = brain_mask
                result["brain_only"] = brain_only
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            with log_lock:
                new_lines = list(log_lines)
                log_lines.clear()
            if new_lines:
                self._run_log_view.append("\n".join(new_lines))
                sb = self._run_log_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if thread.is_alive():
                return

            timer.stop()

            if "error" in result:
                self._status(f"ERROR: {result['error']}")
                self._run_btn.setEnabled(True)
                self._maybe_send_notify(
                    self._run_notify_cb,
                    f"[ZF-Microglia-AI] Run Skin-Remover failed — {stem}",
                    f"Run Skin-Remover on {stem} failed:\n\n{result['error']}",
                )
                return

            brain_mask  = result["brain_mask"]
            brain_only  = result["brain_only"]
            nonzero_pct = 100.0 * brain_mask.sum() / brain_mask.size
            print(f"Brain mask: {brain_mask.sum():,} voxels ({nonzero_pct:.1f}%)")

            bg_suffix  = _BG_SUFFIX.get(bg_mode, "")
            only_name  = f"{stem}_brain_only{bg_suffix}"

            # Replace stale output layers if present
            for lname in (f"{stem}_brain_mask", only_name):
                if lname in self._viewer.layers:
                    self._viewer.layers.remove(lname)

            mask_layer = self._viewer.add_labels(
                brain_mask,
                name=f"{stem}_brain_mask",
                opacity=0.4,
                scale=scale,
            )
            try:
                mask_layer.color = {1: "cyan"}
            except Exception:
                pass  # older napari — default label color is fine
            self._viewer.add_image(
                brain_only,
                name=only_name,
                colormap="gray",
                scale=scale,
            )
            self._refresh_layer_info()

            out_dir    = self._output_dir()
            bg_suffix  = _BG_SUFFIX.get(bg_mode, "")
            if self._save_only_cb.isChecked():
                out = out_dir / f"{stem}_brain_only{bg_suffix}.tif"
                tifffile.imwrite(str(out), brain_only, compression="zlib")
                print(f"Saved: {out}")
            if self._save_mask_cb.isChecked():
                out = out_dir / f"{stem}_brain_mask.tif"
                tifffile.imwrite(
                    str(out),
                    (brain_mask * 255).astype(np.uint8),
                    compression="zlib",
                )
                print(f"Saved: {out}")

            self._status(f"Done — brain={nonzero_pct:.1f}% of volume.")
            self._run_btn.setEnabled(True)
            self._maybe_send_notify(
                self._run_notify_cb,
                f"[ZF-Microglia-AI] Run Skin-Remover done — {stem}",
                f"Run Skin-Remover on {stem} finished.\n\nBrain: {nonzero_pct:.1f}% of volume.",
            )

            print(f"{'='*70}")
            print("SKIN-REMOVER COMPLETE")
            print(f"{'='*70}\n")

        timer.timeout.connect(_poll)
        timer.start(500)

    def _on_bs_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select image to sweep", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._bs_img_edit.setText(path_str)

    def _on_bs_browse_gt(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select hand-corrected GT brain mask", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._bs_gt_edit.setText(path_str)

    def _on_bs_run_sweep(self):
        if self._brain_sweep_job.get("thread") and self._brain_sweep_job["thread"].is_alive():
            self._bs_status_lbl.setText("A sweep is already running.")
            return

        if not self._state["model_path"] or not Path(self._state["model_path"]).is_file():
            self._bs_status_lbl.setText("ERROR: model file not found — browse to a .pth file above first.")
            return
        img_path = self._bs_img_edit.text().strip()
        gt_path = self._bs_gt_edit.text().strip()
        if not (img_path and gt_path):
            self._bs_status_lbl.setText("ERROR: set Image and GT brain mask paths first.")
            return
        for label_str, p in (("Image", img_path), ("GT brain mask", gt_path)):
            if not Path(p).exists():
                self._bs_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return

        th_min = self._bs_thmin_spin.value()
        th_max = self._bs_thmax_spin.value()
        th_step = self._bs_thstep_spin.value()
        if th_max < th_min:
            self._bs_status_lbl.setText("ERROR: Threshold max must be >= min.")
            return
        thresholds = list(np.round(np.arange(th_min, th_max + th_step / 2, th_step), 4))

        er_min = self._bs_ermin_spin.value()
        er_max = self._bs_ermax_spin.value()
        er_step = self._bs_erstep_spin.value()
        if er_max < er_min:
            self._bs_status_lbl.setText("ERROR: Erosion max must be >= min.")
            return
        erosions = list(range(er_min, er_max + 1, er_step))

        model_path = Path(self._state["model_path"])
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        current_threshold = self._thresh_slider.value()
        current_erosion = self._erosion_slider.value()

        sieve_on = self._bs_sieve_cb.isChecked()
        sieve_refine_steps = [
            (self._bs_sieve2_w_spin.value(), self._bs_sieve2_s_spin.value()),
            (self._bs_sieve3_w_spin.value(), self._bs_sieve3_s_spin.value()),
        ]

        cancel_event = threading.Event()
        result = {}
        progress = {"lines": []}
        progress_lock = threading.Lock()

        def _progress_cb(msg):
            with progress_lock:
                progress["lines"].append(msg)

        def _worker():
            try:
                # MONAI's sliding-window progress is otherwise invisible --
                # see _live_progress.py.
                with capture_live_output(_progress_cb):
                    if sieve_on:
                        precomputed_holder = [None]

                        def _run_stage(th_values):
                            sw = _bsw.run_brain_sweep(
                                img_path, gt_path, model_path, device, th_values, erosions,
                                progress_cb=_progress_cb, cancel_event=cancel_event,
                                precomputed=precomputed_holder[0],
                            )
                            precomputed_holder[0] = sw["precomputed"]
                            best_th, _ = sw["best_point"] if sw["best_point"] else (None, None)
                            return best_th, sw

                        sieve = _sieve.run_sieve(
                            _run_stage, lo=th_min, hi=th_max, coarse_step=th_step,
                            refine_steps=sieve_refine_steps, progress_cb=_progress_cb,
                            cancel_check=cancel_event.is_set,
                        )
                        result["sweep"] = sieve["final_result"]
                        result["sieve_stages"] = sieve["stages"]
                    else:
                        result["sweep"] = _bsw.run_brain_sweep(
                            img_path, gt_path, model_path, device, thresholds, erosions,
                            progress_cb=_progress_cb, cancel_event=cancel_event,
                        )
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._brain_sweep_job["thread"] = thread
        self._brain_sweep_job["cancel_event"] = cancel_event
        self._brain_sweep_job["result"] = result
        self._brain_sweep_job["progress"] = progress
        self._brain_sweep_job["progress_lock"] = progress_lock
        self._brain_sweep_job["current_threshold"] = current_threshold
        self._brain_sweep_job["current_erosion"] = current_erosion
        self._brain_sweep_job["fish_key"] = Path(gt_path).stem
        self._bs_run_btn.setEnabled(False)
        self._bs_stop_btn.setEnabled(True)
        self._bs_report_view.clear()
        if sieve_on:
            self._bs_status_lbl.setText(
                f"Sieve mode: stage 1 sweeps {len(thresholds)} Threshold x "
                f"{len(erosions)} Erosion values, then 2 more narrowing "
                f"stages -- one inference pass reused throughout..."
            )
        else:
            self._bs_status_lbl.setText(
                f"Running MONAI inference once, then sweeping {len(thresholds)} Threshold x "
                f"{len(erosions)} Erosion values on {device} ..."
            )
        thread.start()
        self._start_brain_sweep_polling()

    def _on_bs_stop_sweep(self):
        cancel_event = self._brain_sweep_job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
        self._bs_status_lbl.setText("Cancelling — finishing the current threshold value, then stopping...")
        self._bs_stop_btn.setEnabled(False)

    def _start_brain_sweep_polling(self):
        """Same fast (500ms) poll and non-detached, doesn't-survive-napari-
        closing contract as the other two sweep tools' polling loops."""
        timer = QTimer(self)
        job = self._brain_sweep_job

        def _poll():
            with job["progress_lock"]:
                lines = list(job["progress"]["lines"])
                job["progress"]["lines"].clear()
            if lines:
                self._bs_report_view.append("\n".join(lines))
                sb = self._bs_report_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if job["thread"].is_alive():
                return
            timer.stop()
            job["timer"] = None
            self._bs_run_btn.setEnabled(True)
            self._bs_stop_btn.setEnabled(False)

            result = job["result"]
            if "error" in result:
                self._bs_is_gt_cb.setChecked(False)
                self._bs_status_lbl.setText(f"ERROR during sweep: {result['error'].splitlines()[0]}")
                self._bs_report_view.append("\n" + result["error"])
                return

            sweep = result["sweep"]
            if "sieve_stages" in result:
                parts = []
                for i, st in enumerate(result["sieve_stages"], start=1):
                    parts.append(f"===== Sieve stage {i}/{len(result['sieve_stages'])} "
                                 f"(winner: threshold={st['best_value']}) =====")
                    parts.append(_bsw.format_brain_sweep_report(
                        st["result"], job["current_threshold"], job["current_erosion"]
                    ))
                report = "\n\n".join(parts)
            else:
                report = _bsw.format_brain_sweep_report(sweep, job["current_threshold"], job["current_erosion"])
            self._bs_report_view.setPlainText(report)
            if sweep.get("cancelled"):
                self._bs_is_gt_cb.setChecked(False)
                self._bs_status_lbl.setText("Sweep cancelled — partial results above.")
            elif sweep["best_point"] is not None:
                best_th, best_er = sweep["best_point"]
                fish_key = job["fish_key"]
                # One-shot, same as Tab 3 Statistics: only a GT-confirmed
                # run may move the shared recommended values, and the box
                # never stays armed for the next run.
                is_gt = self._bs_is_gt_cb.isChecked()
                self._bs_is_gt_cb.setChecked(False)
                if is_gt:
                    avg_th = self._update_gt_history("monai_threshold", fish_key, best_th, mode="mean")
                    avg_er = self._update_gt_history("erosion_voxels", fish_key, best_er, mode="mean")
                    self._thresh_slider.setValue(avg_th)
                    self._erosion_slider.setValue(round(avg_er))
                    self._save_cfg(
                        monai_threshold=avg_th, monai_threshold_recommended=avg_th,
                        erosion_voxels=round(avg_er), erosion_voxels_recommended=round(avg_er),
                    )
                    self._thresh_recommended_lbl.setText(f"  Recommended MONAI Threshold: {avg_th:.3f}")
                    self._erosion_recommended_lbl.setText(f"  Recommended MONAI Erosion: {round(avg_er)} vox")
                    applied_note = (
                        f" Applied average across all fish swept so far: "
                        f"Threshold={avg_th:.3f}, Erosion={avg_er:.1f}. Saved."
                    )
                else:
                    applied_note = (
                        " Not GT-verified — recommended values NOT updated "
                        "(tick \"This is verified ground truth\" before running to apply)."
                    )
                self._bs_status_lbl.setText(
                    f"This fish's best: MONAI Threshold={best_th}, Erosion={best_er} "
                    f"(Dice={sweep['results'][sweep['best_point']]['dice']:.1f}%)."
                    f"{applied_note}"
                )
            else:
                self._bs_is_gt_cb.setChecked(False)
                self._bs_status_lbl.setText("Sweep finished but no grid points could be scored.")

        timer.timeout.connect(_poll)
        timer.start(500)
        job["timer"] = timer

    def _active_labels_layer(self):
        """Return the active Labels layer, or the topmost one, or None."""
        active = self._viewer.layers.selection.active
        if active is not None and isinstance(active, napari.layers.Labels):
            return active
        for lyr in reversed(self._viewer.layers):
            if isinstance(lyr, napari.layers.Labels):
                return lyr
        return None

    def _on_resort_labels(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._resort_status_lbl.setText("No Labels layer selected.")
            return

        sort_by = self._sort_combo.currentData()
        reverse = self._sort_reverse_cb.isChecked()

        self._resort_btn.setEnabled(False)
        self._resort_status_lbl.setText("Resorting...")

        import numpy as np
        labels = np.asarray(lyr.data)
        result = {}

        def _worker():
            try:
                result["labels"] = resort_labels(labels, sort_by=sort_by, reverse=reverse)
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._resort_status_lbl.setText(f"ERROR: {result['error']}")
                self._resort_btn.setEnabled(True)
                return
            # In-place mutation + refresh(), not a wholesale lyr.data =
            # reassignment -- see copy_label_to_adjacent_slice's memory
            # entry: reassigning .data on a large, already-displayed
            # Labels layer can race napari's own async thumbnail update
            # (napari/layers/labels/labels.py's _update_thumbnail(), a
            # confirmed napari-internal bug, own "ASYNC_TODO" comment)
            # and OOM-crash napari. Same array identity -> no data-changed
            # event, no race; refresh() with async slicing off (this
            # env's default) is fully synchronous.
            lyr.data[:] = result["labels"]
            lyr.refresh()
            n = int(result["labels"].max())
            sort_label = self._sort_combo.currentText()
            rev_str    = " (reversed)" if reverse else ""
            self._resort_status_lbl.setText(
                f"Done — {n} labels, sorted by {sort_label}{rev_str}."
            )
            self._resort_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_remove_debris(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._debris_status_lbl.setText("No Labels layer selected.")
            return

        min_volume = self._current_min_volume()
        fraction = self._finalfrac_spin.value()
        threshold = max(1, round(fraction * min_volume))

        self._debris_btn.setEnabled(False)
        self._debris_status_lbl.setText(f"Removing debris (< {threshold} vox)...")

        labels = np.asarray(lyr.data)
        result = {}

        def _worker():
            try:
                result["labels"], result["n_removed"] = remove_debris(labels, threshold)
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._debris_status_lbl.setText(f"ERROR: {result['error']}")
                self._debris_btn.setEnabled(True)
                return
            lyr.data[:] = result["labels"]  # in-place -- see Resort Labels above for why
            lyr.refresh()
            n = result["n_removed"]
            self._debris_status_lbl.setText(
                f"Done — removed {n} object(s) below {threshold} vox "
                f"({fraction:.3f} x Min volume {min_volume})."
            )
            self._debris_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_use_selected_label(self):
        """Copy the currently selected label from the active Labels layer."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._split_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._split_status_lbl.setText("Selected label is 0 (background).")
            return
        self._split_label_spin.setValue(sel)
        self._split_status_lbl.setText(f"Target set to label {sel}.")

    def _on_use_selected_label_rerun(self):
        """Copy the currently selected label from the active Labels layer
        into the Re-run This Cell Only target field."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._cp_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._cp_status_lbl.setText("Selected label is 0 (background).")
            return
        self._cp_relabel_id_spin.setValue(sel)
        self._cp_status_lbl.setText(f"Re-run target set to label {sel}.")

    def _on_split_label(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._split_status_lbl.setText("No Labels layer selected.")
            return

        target_label = self._split_label_spin.value()
        n_splits     = self._split_n_spin.value()
        sigma        = self._split_sigma_slider.value()
        min_dist     = self._split_dist_slider.value()
        mode         = self._split_mode_combo.currentData()
        z            = int(self._viewer.dims.current_step[0]) if mode == "2d" else None

        image = None
        if mode == "2d":
            signal_name = self._split_signal_combo.currentData()
            if not signal_name or signal_name not in self._viewer.layers:
                self._split_status_lbl.setText("ERROR: pick a signal layer first (required for 2D mode).")
                return
            image = np.asarray(self._viewer.layers[signal_name].data)
            if image.shape != lyr.data.shape:
                self._split_status_lbl.setText(
                    f"ERROR: signal layer shape {image.shape} != labels shape "
                    f"{lyr.data.shape} -- pick the matching signal layer."
                )
                return

        self._split_btn.setEnabled(False)
        self._split_status_lbl.setText(
            f"Splitting (2D, slice {z})…" if mode == "2d" else "Splitting (3D)…"
        )

        labels = np.asarray(lyr.data)
        result = {}

        def _worker():
            try:
                result["labels"], result["new_ids"] = split_label(
                    labels,
                    target_label=target_label,
                    n_splits=n_splits,
                    sigma=sigma,
                    min_distance=min_dist,
                    mode=mode,
                    z=z,
                    image=image,
                )
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._split_status_lbl.setText(f"ERROR: {result['error']}")
                self._split_btn.setEnabled(True)
                return
            lyr.data[:] = result["labels"]  # in-place -- see Resort Labels above for why
            lyr.refresh()
            new_ids   = result["new_ids"]
            n_total   = int(result["labels"].max())
            all_ids   = [target_label] + new_ids
            mode_note = f" (2D, slice {z} only)" if mode == "2d" else ""
            self._split_status_lbl.setText(
                f"Done{mode_note} — {n_splits} parts: {all_ids}. Total labels: {n_total}."
            )
            self._split_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_use_selected_label_join_a(self):
        """Copy the currently selected label from the active Labels layer
        into Join's Label A field."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._join_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._join_status_lbl.setText("Selected label is 0 (background).")
            return
        self._join_a_spin.setValue(sel)
        self._join_status_lbl.setText(f"Label A set to {sel}.")

    def _on_use_selected_label_join_b(self):
        """Copy the currently selected label from the active Labels layer
        into Join's Label B field."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._join_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._join_status_lbl.setText("Selected label is 0 (background).")
            return
        self._join_b_spin.setValue(sel)
        self._join_status_lbl.setText(f"Label B set to {sel}.")

    def _on_join_labels(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._join_status_lbl.setText("No Labels layer selected.")
            return

        label_a = self._join_a_spin.value()
        label_b = self._join_b_spin.value()
        if label_a == label_b:
            self._join_status_lbl.setText("ERROR: Label A and Label B must be different.")
            return

        self._join_btn.setEnabled(False)
        self._join_status_lbl.setText(f"Joining label {label_b} into {label_a}…")

        labels = np.asarray(lyr.data)
        result = {}

        def _worker():
            try:
                result["labels"] = join_labels(labels, label_a, label_b)
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._join_status_lbl.setText(f"ERROR: {result['error']}")
                self._join_btn.setEnabled(True)
                return
            lyr.data[:] = result["labels"]  # in-place -- see Resort Labels above for why
            lyr.refresh()
            self._join_status_lbl.setText(
                f"Done — label {label_b} merged into {label_a}. "
                f"Label {label_b} no longer exists."
            )
            self._join_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_use_selected_label_correct(self):
        """Copy the currently selected label from the active Labels layer
        into Correct Label's target field."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._correct_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._correct_status_lbl.setText("Selected label is 0 (background).")
            return
        self._correct_label_spin.setValue(sel)
        self._correct_status_lbl.setText(f"Label to correct set to {sel}.")

    def _on_correct_label(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._correct_status_lbl.setText("No Labels layer selected.")
            return

        signal_name = self._correct_signal_combo.currentData()
        if not signal_name or signal_name not in self._viewer.layers:
            self._correct_status_lbl.setText("ERROR: pick a signal layer first.")
            return
        signal_lyr = self._viewer.layers[signal_name]

        labels = np.asarray(lyr.data)
        image = np.asarray(signal_lyr.data)
        if labels.shape != image.shape:
            self._correct_status_lbl.setText(
                f"ERROR: labels shape {labels.shape} != signal shape "
                f"{image.shape} -- pick the matching signal layer for "
                f"this labels layer."
            )
            return

        label_id = self._correct_label_spin.value()
        pad = self._correct_pad_spin.value()
        # Read the current on-screen contrast lower limit directly --
        # the whole point is that whatever value the user has dialed
        # in by eye becomes the correction threshold. Only lo is used
        # as the foreground cutoff (signal = image >= lo); hi is
        # napari's display saturation ceiling, not an upper bound on
        # what still counts as real signal -- see
        # correct_label_from_intensity()'s docstring for why a band
        # threshold got this backwards (emptied the label, kept the
        # surrounding background instead).
        lo, hi = (float(v) for v in signal_lyr.contrast_limits)
        z = int(self._viewer.dims.current_step[0])

        self._correct_btn.setEnabled(False)
        self._correct_status_lbl.setText(
            f"Correcting label {label_id} on slice {z} from '{signal_name}' "
            f"-- signal = intensity >= {lo:.3g}…"
        )

        result = {}

        def _worker():
            try:
                result["labels"] = correct_label_from_intensity(
                    labels, image, label_id, z, lo, hi, pad=pad
                )
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._correct_status_lbl.setText(f"ERROR: {result['error']}")
                self._correct_btn.setEnabled(True)
                return
            lyr.data[:] = result["labels"]  # in-place -- see Resort Labels above for why
            lyr.refresh()
            self._correct_status_lbl.setText(
                f"Done — label {label_id} regenerated on slice {z} "
                f"(signal = intensity >= {lo:.3g})."
            )
            self._correct_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_use_selected_label_copyslice(self):
        """Copy the currently selected label from the active Labels layer
        into Copy Label to Adjacent Slice's target field."""
        lyr = self._active_labels_layer()
        if lyr is None:
            self._copyslice_status_lbl.setText("No Labels layer selected.")
            return
        sel = int(lyr.selected_label)
        if sel == 0:
            self._copyslice_status_lbl.setText("Selected label is 0 (background).")
            return
        self._copyslice_label_spin.setValue(sel)
        self._copyslice_status_lbl.setText(f"Label to copy set to {sel}.")

    def _on_copy_label_to_adjacent_slice(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._copyslice_status_lbl.setText("No Labels layer selected.")
            return

        label_id = self._copyslice_label_spin.value()
        direction = self._copyslice_dir_combo.currentData()
        z_src = int(self._viewer.dims.current_step[0])
        z_dst = z_src + direction
        dir_word = "next" if direction > 0 else "previous"

        labels = np.asarray(lyr.data)

        self._copyslice_btn.setEnabled(False)
        self._copyslice_status_lbl.setText(
            f"Copying label {label_id} from slice {z_src} to the {dir_word} "
            f"slice ({z_dst})…"
        )

        result = {}

        def _worker():
            try:
                result["labels"], result["n_excluded"] = copy_label_to_adjacent_slice(
                    labels, label_id, z_src, direction
                )
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._copyslice_status_lbl.setText(f"ERROR: {result['error']}")
                self._copyslice_btn.setEnabled(True)
                return
            lyr.data[:] = result["labels"]  # in-place -- see Resort Labels above for why
            lyr.refresh()
            n_excluded = result["n_excluded"]
            excl_note = (
                f" ({n_excluded} px skipped -- landed on a different label)"
                if n_excluded > 0 else ""
            )
            self._copyslice_status_lbl.setText(
                f"Done — label {label_id} copied from slice {z_src} to "
                f"slice {z_dst}{excl_note}."
            )
            self._copyslice_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(200)

    def _on_save_labels(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._save_labels_status_lbl.setText("No Labels layer selected.")
            return
        default = str(self._output_dir() / f"{lyr.name}.tif")
        out_str, _ = QFileDialog.getSaveFileName(
            self, "Save Labels", default, "TIFF (*.tif *.tiff)"
        )
        if not out_str:
            return
        try:
            tifffile.imwrite(out_str, np.asarray(lyr.data).astype(np.int32), compression="zlib")
            self._save_labels_status_lbl.setText(f"Saved: {Path(out_str).name}")
            print(f"Labels saved: {out_str}")
        except Exception as exc:
            self._save_labels_status_lbl.setText(f"ERROR: {exc}")

    def _on_stats_backend_changed(self, *_):
        backend = self._stats_backend_combo.currentData()
        self._ollama_panel.setVisible(backend == "ollama")
        self._api_panel.setVisible(backend in ("openai", "claude"))
        self._save_cfg(stats_backend=backend)

    def _on_generate_stats(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._stats_status_lbl.setText("No Labels layer selected.")
            return

        # Scale priority: 1) file metadata (most reliable, set by Open button),
        # 2) Labels layer scale, 3) active image layer scale, 4) default (1,1,1).
        # Using metadata avoids the case where a Labels layer loaded from a TIF
        # has the napari default scale (1,1,1), which makes centroid_um == centroid_vox
        # and volume_um3 == volume_vox (wrong — Z=1.0 µm, Y=X=0.174 µm per voxel).
        meta = self._state.get("metadata")
        if meta and "scale" in meta:
            scale_zyx = tuple(float(v) for v in meta["scale"])
        else:
            sc = lyr.scale
            if len(sc) != 3 or all(v == 1.0 for v in sc):
                img = self._active_layer()
                sc = img.scale if img is not None and len(img.scale) == 3 else sc
            scale_zyx = tuple(float(v) for v in sc)
        print(f"Statistics scale: Z={scale_zyx[0]:.4f}  Y={scale_zyx[1]:.4f}  X={scale_zyx[2]:.4f} µm/vox")

        backend = self._stats_backend_combo.currentData()

        # Build backend_config and persist API settings (key stored locally only)
        backend_config = {"backend": backend}
        if backend == "ollama":
            ep = self._ollama_endpoint_edit.text().strip()
            mo = self._ollama_model_edit.text().strip()
            backend_config.update(ollama_endpoint=ep, ollama_model=mo)
            self._save_cfg(ollama_endpoint=ep, ollama_model=mo)
        elif backend in ("openai", "claude"):
            ak  = self._api_key_edit.text().strip()
            mo  = self._api_model_edit.text().strip()
            url = self._api_url_edit.text().strip()
            backend_config.update(api_key=ak, api_model=mo, api_url=url)
            self._save_cfg(api_model=mo, api_url=url)
            secret_err = _secrets.set_secret("api_key", ak)
            if secret_err:
                print(f"API key not saved for next session: {secret_err}")

        # Intensity image (optional)
        image = None
        img_name = self._stats_image_combo.currentData()
        if img_name is not None and img_name in self._viewer.layers:
            image = np.asarray(self._viewer.layers[img_name].data)

        # Brain region lines (optional)
        region_lines = None
        region_names = None
        shp_name = self._stats_shapes_combo.currentData()
        if shp_name is not None and shp_name in self._viewer.layers:
            shp_lyr = self._viewer.layers[shp_name]
            region_lines = _extract_region_lines_um(shp_lyr)
            if region_lines:
                names_text = self._stats_region_names_edit.text().strip()
                if names_text:
                    region_names = [n.strip() for n in names_text.split(",") if n.strip()]
                if not region_names:
                    # Auto-generate names if user left field blank
                    region_names = [f"Region {i+1}" for i in range(len(region_lines) + 1)]
            else:
                region_lines = None  # layer had no line shapes

        # Validate: warn if a column is checked but its required input is missing
        warnings = []
        region_cols = {"brain_region", "region_boundary_dist_um"}
        intensity_cols = {"mean_intensity", "integrated_intensity", "intensity_cv"}
        checked = {k for k, cb in self._col_checkboxes.items() if cb.isChecked()}
        if checked & region_cols and region_lines is None:
            warnings.append(
                "brain_region / region_boundary_dist_um checked but no Shapes layer "
                "with line/path shapes is selected — those columns will be skipped."
            )
        if checked & intensity_cols and image is None:
            warnings.append(
                "Intensity columns checked but no Image layer is selected — "
                "those columns will be skipped."
            )
        if warnings:
            self._stats_status_lbl.setText("Warning: " + "  |  ".join(warnings))

        labels    = np.asarray(lyr.data)
        out_dir   = self._output_dir()
        stem      = self._state["last_file_path"].stem if self._state.get("last_file_path") else lyr.name
        out_csv   = out_dir / f"{stem}_statistics.csv"

        self._stats_btn.setEnabled(False)
        self._stats_status_lbl.setText("Computing statistics…")

        result = {}

        def _worker():
            try:
                df = compute_stats(
                    labels, scale_zyx,
                    image=image,
                    region_lines=region_lines,
                    region_names=region_names,
                    backend_config=backend_config,
                )
                result["df"] = df
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            if thread.is_alive():
                return
            timer.stop()
            if "error" in result:
                self._stats_status_lbl.setText(f"ERROR: {result['error']}")
                self._stats_btn.setEnabled(True)
                return
            df = result["df"]

            # Outlier bounds. Purely read here -- Statistics used to also
            # WRITE min_volume_vox/max_volume_vox/min_hole_size_vox from
            # its own "This is verified ground truth" checkbox, but every
            # GT-tuned value in this plugin is now applied exclusively by
            # the GT Toolkit Tuning Tool (Tab 5), never from individual
            # tools like this one -- see _on_gtk_run().
            max_ceiling = self._state.get("config", {}).get("max_volume_recommended_vox")
            min_floor = self._state.get("config", {}).get("min_volume_recommended_vox")

            if "volume_vox" in df.columns:
                too_big = df["volume_vox"] > max_ceiling if max_ceiling is not None else False
                too_small = df["volume_vox"] < min_floor if min_floor is not None else False
                df["is_volume_outlier"] = too_big | too_small

            # Filter to selected columns; label is always kept.
            # Group keys (bbox_vox, bbox_um) expand to their constituent columns.
            selected = {"label"}
            for k, cb in self._col_checkboxes.items():
                if cb.isChecked():
                    selected.update(_COL_GROUPS.get(k, [k]))
            df = df[[c for c in df.columns if c in selected]]
            df.to_csv(str(out_csv), index=False)
            self._stats_status_lbl.setText(
                f"Done — {len(df)} labels. Saved: {out_csv.name}."
            )
            print(f"Statistics saved: {out_csv}")
            self._stats_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(500)

    def _on_gtscore_run(self):
        """Whole-fish Hungarian-matched scoring -- pure CPU (scipy), fast
        enough at typical whole-fish object counts (tens to low hundreds)
        to run synchronously without a background thread."""
        pred_name = self._gtscore_pred_combo.currentData()
        gt_name = self._gtscore_gt_combo.currentData()
        if not pred_name or not gt_name:
            self._gtscore_status_lbl.setText("ERROR: select both a Predicted labels and a GT labels layer.")
            return
        if pred_name not in self._viewer.layers or gt_name not in self._viewer.layers:
            self._gtscore_status_lbl.setText("ERROR: selected layer no longer exists — refresh the dropdowns.")
            return

        pred_labels = np.asarray(self._viewer.layers[pred_name].data)
        gt_labels = np.asarray(self._viewer.layers[gt_name].data)
        if pred_labels.shape != gt_labels.shape:
            self._gtscore_status_lbl.setText(
                f"ERROR: shape mismatch — {pred_name} is {pred_labels.shape}, {gt_name} is {gt_labels.shape}."
            )
            return

        threshold = self._gtscore_thresh_spin.value()
        try:
            result = _gts.score_against_gt(pred_labels, gt_labels, iou_threshold=threshold)
        except Exception as exc:
            self._gtscore_status_lbl.setText(f"ERROR: {exc}")
            traceback.print_exc()
            return

        report = _gts.format_gt_score_report(result, pred_name=pred_name, gt_name=gt_name)
        self._gtscore_report_view.setPlainText(report)
        self._gtscore_status_lbl.setText(
            f"TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  Score={result['score']:+.1f}  "
            f"MeanIoU={result['mean_iou']:.1f}%  MeanDice={result['mean_dice']:.1f}%"
        )

    def _on_create_labels(self):
        # Read active layer
        target = self._active_layer()
        if target is None:
            self._labels_status_lbl.setText("Select a brain_only layer first.")
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._labels_status_lbl.setText(
                f"ERROR: 3D volume required, got {volume.ndim}D."
            )
            return

        sigma_xy      = self._sxy_slider.value()
        sigma_z       = self._sz_slider.value()
        min_volume    = self._current_min_volume()
        min_hole_size = self._hole_slider.value()
        final_min_fraction = self._finalfrac_spin.value()
        stem            = target.name
        scale           = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1., 1., 1.)

        self._labels_btn.setEnabled(False)
        self._labels_status_lbl.setText("Running...")
        self._labels_log_view.clear()

        print(f"\n{'='*70}")
        print(f"CREATE LABELS — {stem}  shape={volume.shape}")
        print(f"σ_xy={sigma_xy}  σ_z={sigma_z}  min_volume={min_volume} vox  min_hole_size={min_hole_size} vox  "
              f"final_min_fraction={final_min_fraction}")
        print(f"{'='*70}")

        result = {}
        log_lines = []
        log_lock = threading.Lock()

        def _push_log(line):
            with log_lock:
                log_lines.append(line)

        def _worker():
            try:
                with capture_live_output(_push_log):
                    labels = create_labels(
                        volume,
                        sigma_xy=sigma_xy,
                        sigma_z=sigma_z,
                        min_volume=min_volume,
                        min_hole_size=min_hole_size,
                        final_min_fraction=final_min_fraction,
                    )
                result["labels"] = labels
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer = QTimer(self)

        def _poll():
            with log_lock:
                new_lines = list(log_lines)
                log_lines.clear()
            if new_lines:
                self._labels_log_view.append("\n".join(new_lines))
                sb = self._labels_log_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if thread.is_alive():
                return
            timer.stop()

            if "error" in result:
                self._labels_status_lbl.setText(f"ERROR: {result['error']}")
                self._labels_btn.setEnabled(True)
                return

            labels     = result["labels"]
            n_labels   = int(labels.max())
            lname      = f"{stem}_labels"

            if lname in self._viewer.layers:
                self._viewer.layers.remove(lname)

            self._viewer.add_labels(labels, name=lname, scale=scale)

            self._labels_status_lbl.setText(f"Done — {n_labels} labels.")
            self._labels_btn.setEnabled(True)

            print(f"{'='*70}")
            print(f"CREATE LABELS COMPLETE — {n_labels} objects")
            print(f"{'='*70}\n")

        timer.timeout.connect(_poll)
        timer.start(500)

    def _on_ps_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT fish's image", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._ps_img_edit.setText(path_str)

    def _on_ps_browse_mask(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select brain_mask.tif (raw, un-eroded)", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._ps_mask_edit.setText(path_str)

    def _on_ps_browse_lbl(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._ps_lbl_edit.setText(path_str)

    def _on_ps_run_sweep(self):
        if self._pixel_sweep_job.get("thread") and self._pixel_sweep_job["thread"].is_alive():
            self._ps_status_lbl.setText("A sweep is already running.")
            return

        img_path = self._ps_img_edit.text().strip()
        mask_path = self._ps_mask_edit.text().strip()
        lbl_path = self._ps_lbl_edit.text().strip()
        if not (img_path and mask_path and lbl_path):
            self._ps_status_lbl.setText("ERROR: set GT image, brain_mask.tif, and GT labels paths first.")
            return
        for label_str, p in (("GT image", img_path), ("brain_mask.tif", mask_path), ("GT labels", lbl_path)):
            if not Path(p).exists():
                self._ps_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return

        bg_min = self._ps_bgmin_spin.value()
        bg_max = self._ps_bgmax_spin.value()
        bg_step = self._ps_bgstep_spin.value()
        if bg_max < bg_min:
            self._ps_status_lbl.setText("ERROR: BG Threshold max must be >= min.")
            return
        bg_thresholds = list(np.round(np.arange(bg_min, bg_max + bg_step / 2, bg_step), 4))

        er_min = self._ps_ermin_spin.value()
        er_max = self._ps_ermax_spin.value()
        er_step = self._ps_erstep_spin.value()
        if er_max < er_min:
            self._ps_status_lbl.setText("ERROR: Erosion max must be >= min.")
            return
        erosions = list(range(er_min, er_max + 1, er_step))

        scale_zyx = (self._ps_scalez_spin.value(), self._ps_scalexy_spin.value(), self._ps_scalexy_spin.value())
        sigma_xy = self._sxy_slider.value()
        sigma_z = self._sz_slider.value()
        n_cells = self._ps_ncells_spin.value()
        pad_z = self._ps_padz_spin.value()
        pad_xy = self._ps_padxy_spin.value()
        current_bg = self._tol_slider.value()
        current_erosion = self._sig_erosion_slider.value()
        final_min_fraction = self._finalfrac_spin.value()

        sieve_on = self._ps_sieve_cb.isChecked()
        sieve_refine_steps = [
            (self._ps_sieve2_w_spin.value(), self._ps_sieve2_s_spin.value()),
            (self._ps_sieve3_w_spin.value(), self._ps_sieve3_s_spin.value()),
        ]

        cancel_event = threading.Event()
        result = {}
        progress = {"lines": []}
        progress_lock = threading.Lock()

        def _progress_cb(msg):
            with progress_lock:
                progress["lines"].append(msg)

        def _worker():
            try:
                if sieve_on:
                    def _run_stage(bg_values):
                        sw = _psw.run_pixel_sweep(
                            img_path, mask_path, lbl_path, bg_values, erosions, scale_zyx,
                            sigma_xy=sigma_xy, sigma_z=sigma_z, min_volume=None, min_hole_size=None,
                            final_min_fraction=final_min_fraction,
                            n_cells=n_cells, pad_z=pad_z, pad_xy=pad_xy,
                            progress_cb=_progress_cb, cancel_event=cancel_event,
                        )
                        best_bg, _ = sw["best_point"] if sw["best_point"] else (None, None)
                        return best_bg, sw

                    sieve = _sieve.run_sieve(
                        _run_stage, lo=bg_min, hi=bg_max, coarse_step=bg_step,
                        refine_steps=sieve_refine_steps, progress_cb=_progress_cb,
                        cancel_check=cancel_event.is_set,
                    )
                    result["sweep"] = sieve["final_result"]
                    result["sieve_stages"] = sieve["stages"]
                else:
                    result["sweep"] = _psw.run_pixel_sweep(
                        img_path, mask_path, lbl_path, bg_thresholds, erosions, scale_zyx,
                        sigma_xy=sigma_xy, sigma_z=sigma_z, min_volume=None, min_hole_size=None,
                        final_min_fraction=final_min_fraction,
                        n_cells=n_cells, pad_z=pad_z, pad_xy=pad_xy,
                        progress_cb=_progress_cb, cancel_event=cancel_event,
                    )
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._pixel_sweep_job["thread"] = thread
        self._pixel_sweep_job["cancel_event"] = cancel_event
        self._pixel_sweep_job["result"] = result
        self._pixel_sweep_job["progress"] = progress
        self._pixel_sweep_job["progress_lock"] = progress_lock
        self._pixel_sweep_job["current_bg"] = current_bg
        self._pixel_sweep_job["current_erosion"] = current_erosion
        self._pixel_sweep_job["fish_key"] = Path(lbl_path).stem
        self._ps_run_btn.setEnabled(False)
        self._ps_stop_btn.setEnabled(True)
        self._ps_report_view.clear()
        if sieve_on:
            self._ps_status_lbl.setText(
                f"Sieve mode: stage 1 sweeps {len(bg_thresholds)} BG Threshold x "
                f"{len(erosions)} Signal Erosion values across up to {n_cells} cells, "
                f"then 2 more narrowing stages..."
            )
        else:
            self._ps_status_lbl.setText(
                f"Sweeping {len(bg_thresholds)} BG Threshold x {len(erosions)} Signal Erosion "
                f"values across up to {n_cells} cells ..."
            )
        thread.start()
        self._start_pixel_sweep_polling()

    def _on_ps_stop_sweep(self):
        cancel_event = self._pixel_sweep_job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
        self._ps_status_lbl.setText("Cancelling — finishing the current grid point, then stopping...")
        self._ps_stop_btn.setEnabled(False)

    def _start_pixel_sweep_polling(self):
        """Same fast (500ms) poll and non-detached, doesn't-survive-napari-
        closing contract as _start_epoch_sweep_polling — see its docstring."""
        timer = QTimer(self)
        job = self._pixel_sweep_job

        def _poll():
            with job["progress_lock"]:
                lines = list(job["progress"]["lines"])
                job["progress"]["lines"].clear()
            if lines:
                self._ps_report_view.append("\n".join(lines))
                sb = self._ps_report_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if job["thread"].is_alive():
                return
            timer.stop()
            job["timer"] = None
            self._ps_run_btn.setEnabled(True)
            self._ps_stop_btn.setEnabled(False)

            result = job["result"]
            if "error" in result:
                self._ps_is_gt_cb.setChecked(False)
                self._ps_status_lbl.setText(f"ERROR during sweep: {result['error'].splitlines()[0]}")
                self._ps_report_view.append("\n" + result["error"])
                return

            sweep = result["sweep"]
            if "sieve_stages" in result:
                parts = []
                for i, st in enumerate(result["sieve_stages"], start=1):
                    parts.append(f"===== Sieve stage {i}/{len(result['sieve_stages'])} "
                                 f"(winner: BG threshold={st['best_value']}) =====")
                    parts.append(_psw.format_pixel_sweep_report(
                        st["result"], job["current_bg"], job["current_erosion"]
                    ))
                report = "\n\n".join(parts)
            else:
                report = _psw.format_pixel_sweep_report(sweep, job["current_bg"], job["current_erosion"])
            self._ps_report_view.setPlainText(report)
            if sweep.get("cancelled"):
                self._ps_is_gt_cb.setChecked(False)
                self._ps_status_lbl.setText("Sweep cancelled — partial results above.")
            elif sweep["best_point"] is not None:
                best_bt, best_er = sweep["best_point"]
                fish_key = job["fish_key"]
                # One-shot, same as Tab 3 Statistics: only a GT-confirmed
                # run may move any recommended value below, and the box
                # never stays armed for the next run.
                is_gt = self._ps_is_gt_cb.isChecked()
                self._ps_is_gt_cb.setChecked(False)

                if is_gt:
                    # min_volume/min_hole_size are never-rising floors:
                    # once one fish's GT proves a cell (or hole) of size N
                    # is real, no other fish's sweep (which may simply
                    # lack any cell/hole that small) should raise the
                    # recommendation back above N. Tracked per-fish so
                    # re-sweeping the same fish after a GT correction
                    # properly updates its own entry rather than being
                    # stuck with a stale value forever -- see
                    # _update_gt_history.
                    recommended = self._update_gt_history(
                        "min_volume_vox", fish_key, sweep["min_volume_used"], mode="min"
                    )
                    recommended_hole = self._update_gt_history(
                        "min_hole_size_vox", fish_key, sweep["min_hole_size_used"], mode="min"
                    )
                    self._save_cfg(min_volume_recommended_vox=recommended,
                                    min_hole_size_recommended_vox=recommended_hole)
                    self._area_value_lbl.setText(str(recommended))
                    self._hole_recommended_lbl.setText(
                        f"  Recommended floor (from GT sweeps so far): {recommended_hole} vox"
                    )

                    # BG Threshold and Signal Erosion aren't safety floors -- each
                    # fish's sweep just finds that fish's own local optimum,
                    # with no safe direction to bias toward -- so these are
                    # averaged across every fish swept so far instead of
                    # floored. Signal Erosion has its own history, separate from
                    # the MONAI Threshold/Erosion sweep's -- it tunes the
                    # noBG-stack-prep erosion (Tab 1's Signal Erosion slider),
                    # not the semantic brain-mask erosion.
                    avg_bt = self._update_gt_history("bg_tolerance", fish_key, best_bt, mode="mean")
                    avg_er = self._update_gt_history("signal_erosion_voxels", fish_key, best_er, mode="mean")

                    # Widen the Min hole size slider's (and its paired
                    # spinbox's -- kept in sync only via signals, see
                    # _add_reliable_spinbox) range first if a recommendation
                    # falls outside its fixed default, tuned around the old
                    # guessed constant. Without widening both, setValue() on
                    # the slider would trigger the synced spinbox to clamp
                    # back to its own stale bounds, which then forces the
                    # slider back too -- silently wrong instead of the real
                    # recommendation. Min volume needs no such treatment
                    # anymore -- it's an informative label now, not a slider.
                    if recommended_hole > self._hole_slider.maximum():
                        self._hole_slider.setMaximum(recommended_hole)
                        self._hole_spin.setMaximum(recommended_hole)
                    self._hole_slider.setValue(recommended_hole)
                    self._tol_slider.setValue(avg_bt)
                    self._sig_erosion_slider.setValue(round(avg_er))
                    self._tol_recommended_lbl.setText(f"  Recommended BG Threshold: {avg_bt:.3f}")
                    self._sig_erosion_recommended_lbl.setText(f"  Recommended Signal Erosion: {round(avg_er)} vox")

                    self._save_cfg(
                        bg_tolerance=avg_bt, bg_tolerance_recommended=avg_bt,
                        signal_erosion_voxels=round(avg_er), signal_erosion_voxels_recommended=round(avg_er),
                        min_volume_vox=recommended, min_hole_size_vox=recommended_hole,
                    )
                    applied_note = (
                        f" Applied average across all fish swept so far: BG Threshold="
                        f"{avg_bt:.3f}, Signal Erosion={avg_er:.1f}. Min volume floor={recommended} "
                        f"vox, min hole size floor={recommended_hole} vox. All saved."
                    )
                else:
                    applied_note = (
                        " Not GT-verified — recommended values NOT updated "
                        "(tick \"This is verified ground truth\" before running to apply)."
                    )
                self._ps_status_lbl.setText(
                    f"This fish's best: BG Threshold={best_bt}, Signal Erosion={best_er} "
                    f"(avg IoU={sweep['per_point_avg'][sweep['best_point']]['iou']:.1f}%)."
                    f"{applied_note}"
                )
            else:
                self._ps_is_gt_cb.setChecked(False)
                self._ps_status_lbl.setText("Sweep finished but no grid points could be scored.")

        timer.timeout.connect(_poll)
        timer.start(500)
        job["timer"] = timer

    def _on_sg_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT fish's image", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._sg_img_edit.setText(path_str)

    def _on_sg_browse_mask(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select brain_mask.tif (raw, un-eroded)", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._sg_mask_edit.setText(path_str)

    def _on_sg_browse_lbl(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._sg_lbl_edit.setText(path_str)

    def _on_sg_run_sweep(self):
        if self._sigma_sweep_job.get("thread") and self._sigma_sweep_job["thread"].is_alive():
            self._sg_status_lbl.setText("A sweep is already running.")
            return

        img_path = self._sg_img_edit.text().strip()
        mask_path = self._sg_mask_edit.text().strip()
        lbl_path = self._sg_lbl_edit.text().strip()
        if not (img_path and mask_path and lbl_path):
            self._sg_status_lbl.setText("ERROR: set GT image, brain_mask.tif, and GT labels paths first.")
            return
        for label_str, p in (("GT image", img_path), ("brain_mask.tif", mask_path), ("GT labels", lbl_path)):
            if not Path(p).exists():
                self._sg_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return

        sxy_min = self._sg_sxymin_spin.value()
        sxy_max = self._sg_sxymax_spin.value()
        sxy_step = self._sg_sxystep_spin.value()
        if sxy_max < sxy_min:
            self._sg_status_lbl.setText("ERROR: sigma XY max must be >= min.")
            return
        sigma_xy_values = list(np.round(np.arange(sxy_min, sxy_max + sxy_step / 2, sxy_step), 2))

        sz_min = self._sg_szmin_spin.value()
        sz_max = self._sg_szmax_spin.value()
        sz_step = self._sg_szstep_spin.value()
        if sz_max < sz_min:
            self._sg_status_lbl.setText("ERROR: sigma Z max must be >= min.")
            return
        sigma_z_values = list(np.round(np.arange(sz_min, sz_max + sz_step / 2, sz_step), 2))

        scale_zyx = (self._sg_scalez_spin.value(), self._sg_scalexy_spin.value(), self._sg_scalexy_spin.value())
        # Held fixed at Tab 1's current values -- this sweep varies sigma,
        # not BG Threshold/Signal Erosion (see run_pixel_sweep above for that
        # one). Signal Erosion, not MONAI Erosion, since this pipeline is the
        # same noBG-stack-prep path the BG Threshold/Erosion sweep tunes.
        bg_threshold = self._tol_slider.value()
        erosion = self._sig_erosion_slider.value()
        n_cells = self._sg_ncells_spin.value()
        pad_z = self._sg_padz_spin.value()
        pad_xy = self._sg_padxy_spin.value()
        current_sxy = self._sxy_slider.value()
        current_sz = self._sz_slider.value()
        final_min_fraction = self._finalfrac_spin.value()

        sieve_on = self._sg_sieve_cb.isChecked()
        sieve_refine_steps = [
            (self._sg_sieve2_w_spin.value(), self._sg_sieve2_s_spin.value()),
            (self._sg_sieve3_w_spin.value(), self._sg_sieve3_s_spin.value()),
        ]

        cancel_event = threading.Event()
        result = {}
        progress = {"lines": []}
        progress_lock = threading.Lock()

        def _progress_cb(msg):
            with progress_lock:
                progress["lines"].append(msg)

        def _worker():
            try:
                if sieve_on:
                    def _run_stage(sxy_values):
                        sw = _psw.run_sigma_sweep(
                            img_path, mask_path, lbl_path, sxy_values, sigma_z_values, scale_zyx,
                            bg_threshold, erosion, min_volume=None, min_hole_size=None,
                            final_min_fraction=final_min_fraction,
                            n_cells=n_cells, pad_z=pad_z, pad_xy=pad_xy,
                            progress_cb=_progress_cb, cancel_event=cancel_event,
                        )
                        best_sxy, _ = sw["best_point"] if sw["best_point"] else (None, None)
                        return best_sxy, sw

                    sieve = _sieve.run_sieve(
                        _run_stage, lo=sxy_min, hi=sxy_max, coarse_step=sxy_step,
                        refine_steps=sieve_refine_steps, progress_cb=_progress_cb,
                        cancel_check=cancel_event.is_set,
                    )
                    result["sweep"] = sieve["final_result"]
                    result["sieve_stages"] = sieve["stages"]
                else:
                    result["sweep"] = _psw.run_sigma_sweep(
                        img_path, mask_path, lbl_path, sigma_xy_values, sigma_z_values, scale_zyx,
                        bg_threshold, erosion, min_volume=None, min_hole_size=None,
                        final_min_fraction=final_min_fraction,
                        n_cells=n_cells, pad_z=pad_z, pad_xy=pad_xy,
                        progress_cb=_progress_cb, cancel_event=cancel_event,
                    )
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._sigma_sweep_job["thread"] = thread
        self._sigma_sweep_job["cancel_event"] = cancel_event
        self._sigma_sweep_job["result"] = result
        self._sigma_sweep_job["progress"] = progress
        self._sigma_sweep_job["progress_lock"] = progress_lock
        self._sigma_sweep_job["current_sxy"] = current_sxy
        self._sigma_sweep_job["current_sz"] = current_sz
        self._sigma_sweep_job["fish_key"] = Path(lbl_path).stem
        self._sg_run_btn.setEnabled(False)
        self._sg_stop_btn.setEnabled(True)
        self._sg_report_view.clear()
        if sieve_on:
            self._sg_status_lbl.setText(
                f"Sieve mode: stage 1 sweeps {len(sigma_xy_values)} sigma XY x "
                f"{len(sigma_z_values)} sigma Z values across up to {n_cells} cells, "
                f"then 2 more narrowing stages..."
            )
        else:
            self._sg_status_lbl.setText(
                f"Sweeping {len(sigma_xy_values)} sigma XY x {len(sigma_z_values)} sigma Z "
                f"values across up to {n_cells} cells (BG Threshold={bg_threshold}, "
                f"Signal Erosion={erosion} held fixed) ..."
            )
        thread.start()
        self._start_sigma_sweep_polling()

    def _on_sg_stop_sweep(self):
        cancel_event = self._sigma_sweep_job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
        self._sg_status_lbl.setText("Cancelling — finishing the current grid point, then stopping...")
        self._sg_stop_btn.setEnabled(False)

    def _start_sigma_sweep_polling(self):
        """Same fast (500ms) poll and non-detached, doesn't-survive-napari-
        closing contract as _start_pixel_sweep_polling — see its docstring."""
        timer = QTimer(self)
        job = self._sigma_sweep_job

        def _poll():
            with job["progress_lock"]:
                lines = list(job["progress"]["lines"])
                job["progress"]["lines"].clear()
            if lines:
                self._sg_report_view.append("\n".join(lines))
                sb = self._sg_report_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if job["thread"].is_alive():
                return
            timer.stop()
            job["timer"] = None
            self._sg_run_btn.setEnabled(True)
            self._sg_stop_btn.setEnabled(False)

            result = job["result"]
            if "error" in result:
                self._sg_is_gt_cb.setChecked(False)
                self._sg_status_lbl.setText(f"ERROR during sweep: {result['error'].splitlines()[0]}")
                self._sg_report_view.append("\n" + result["error"])
                return

            sweep = result["sweep"]
            if "sieve_stages" in result:
                parts = []
                for i, st in enumerate(result["sieve_stages"], start=1):
                    parts.append(f"===== Sieve stage {i}/{len(result['sieve_stages'])} "
                                 f"(winner: sigma XY={st['best_value']}) =====")
                    parts.append(_psw.format_sigma_sweep_report(
                        st["result"], job["current_sxy"], job["current_sz"]
                    ))
                report = "\n\n".join(parts)
            else:
                report = _psw.format_sigma_sweep_report(sweep, job["current_sxy"], job["current_sz"])
            self._sg_report_view.setPlainText(report)
            if sweep.get("cancelled"):
                self._sg_is_gt_cb.setChecked(False)
                self._sg_status_lbl.setText("Sweep cancelled — partial results above.")
            elif sweep["best_point"] is not None:
                best_sxy, best_sz = sweep["best_point"]
                fish_key = job["fish_key"]
                # One-shot, same as Tab 3 Statistics.
                is_gt = self._sg_is_gt_cb.isChecked()
                self._sg_is_gt_cb.setChecked(False)

                if is_gt:
                    # Same never-rises-floor treatment as the BG Threshold/
                    # Erosion sweep -- see that handler for the full reasoning.
                    recommended = self._update_gt_history(
                        "min_volume_vox", fish_key, sweep["min_volume_used"], mode="min"
                    )
                    recommended_hole = self._update_gt_history(
                        "min_hole_size_vox", fish_key, sweep["min_hole_size_used"], mode="min"
                    )
                    self._save_cfg(min_volume_recommended_vox=recommended,
                                    min_hole_size_recommended_vox=recommended_hole)
                    self._area_value_lbl.setText(str(recommended))
                    self._hole_recommended_lbl.setText(
                        f"  Recommended floor (from GT sweeps so far): {recommended_hole} vox"
                    )
                    if recommended_hole > self._hole_slider.maximum():
                        self._hole_slider.setMaximum(recommended_hole)
                        self._hole_spin.setMaximum(recommended_hole)
                    self._hole_slider.setValue(recommended_hole)

                    # Sigma XY/Z aren't safety floors either -- averaged across
                    # every fish swept so far, same reasoning as BG Threshold/
                    # Erosion above.
                    avg_sxy = self._update_gt_history("sigma_xy", fish_key, best_sxy, mode="mean")
                    avg_sz = self._update_gt_history("sigma_z", fish_key, best_sz, mode="mean")
                    self._sxy_slider.setValue(avg_sxy)
                    self._sz_slider.setValue(avg_sz)
                    self._sxy_recommended_lbl.setText(f"  Recommended Smooth σ XY: {avg_sxy:.2f}")
                    self._sz_recommended_lbl.setText(f"  Recommended Smooth σ Z: {avg_sz:.2f}")
                    self._save_cfg(
                        sigma_xy=avg_sxy, sigma_xy_recommended=avg_sxy,
                        sigma_z=avg_sz, sigma_z_recommended=avg_sz,
                        min_volume_vox=recommended, min_hole_size_vox=recommended_hole,
                    )
                    applied_note = (
                        f" Applied average across all fish swept so far: sigma XY={avg_sxy:.2f}, "
                        f"sigma Z={avg_sz:.2f}. Min volume floor={recommended} vox, min hole "
                        f"size floor={recommended_hole} vox. All saved."
                    )
                else:
                    applied_note = (
                        " Not GT-verified — recommended values NOT updated "
                        "(tick \"This is verified ground truth\" before running to apply)."
                    )
                self._sg_status_lbl.setText(
                    f"This fish's best: Smooth sigma XY={best_sxy}, sigma Z={best_sz} "
                    f"(avg IoU={sweep['per_point_avg'][sweep['best_point']]['iou']:.1f}%)."
                    f"{applied_note}"
                )
            else:
                self._sg_is_gt_cb.setChecked(False)
                self._sg_status_lbl.setText("Sweep finished but no grid points could be scored.")

        timer.timeout.connect(_poll)
        timer.start(500)
        job["timer"] = timer

    def _on_run_cellpose_seg(self):
        target = self._active_layer()
        if target is None:
            self._cp_status_lbl.setText("Select a brain_only layer first.")
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._cp_status_lbl.setText(f"ERROR: 3D volume required, got {volume.ndim}D.")
            return
        model_path = self._state.get("cellpose_model_path")
        if not model_path or not Path(model_path).is_file():
            self._cp_status_lbl.setText("ERROR: no Cellpose-SAM model selected — browse to a checkpoint.")
            return
        if not _is_valid_cellpose_checkpoint(model_path):
            self._cp_status_lbl.setText(
                f"ERROR: {Path(model_path).name} doesn't look like a valid PyTorch "
                f"checkpoint (not a zip archive) -- re-browse to the correct file."
            )
            return

        cellprob      = self._cp_cellprob_slider.value()
        flow          = _FLOW_THRESHOLD_FIXED
        max_gap       = self._cp_maxgap_slider.value()
        min_contact   = self._cp_mincontact_slider.value()
        # Unified with the Pixel Classifier's Min volume -- see the
        # Common Settings note in _build_ui().
        gt_min        = self._current_min_volume()
        large_contact = self._cp_largecontact_slider.value()
        # Shared with the Pixel Classifier route -- same underlying idea
        # (real vs. noise-sized enclosed gaps) applies to Cellpose-SAM's
        # own predicted masks too, see _make_capped_fill_holes() in
        # _cellpose_seg.py.
        min_hole_size = self._hole_slider.value()
        # Deliberately NOT the same value as Min volume -- see the
        # "Common Settings" note above self._hole_slider's construction.
        min_size = self._cp_minsize_spin.value()
        final_min_fraction = self._finalfrac_spin.value()
        niter = self._cp_niter_spin.value()

        stem  = target.name
        scale = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1.0, 1.0, 1.0)
        z, y, x = scale
        xy = (x + y) / 2.0
        anisotropy = z / xy if xy > 0 else 5.747

        gpu = torch.cuda.is_available()

        # Recoverable-crash cache -- see save_flow_cache()/load_flow_cache()
        # in _cellpose_seg.py. Lives in this fish's own output folder, one
        # cache file per stem; only ever survives on disk if mask-formation
        # (the step right after the expensive network pass) itself crashes
        # (e.g. a GPU OOM from another concurrent job) -- a normal run
        # deletes it automatically once that step succeeds.
        flow_cache_path = self._output_dir() / f"{stem}_cp_flow_cache.npz"

        self._cp_run_btn.setEnabled(False)
        self._cp_status_lbl.setText(f"Starting (device={'cuda' if gpu else 'cpu'})...")
        self._cp_log_view.clear()

        print(f"\n{'='*70}")
        print(f"CELLPOSE-SAM SEGMENTATION — {stem}  shape={volume.shape}")
        print(f"Model        : {Path(model_path).name}")
        print(f"Cellprob     : {cellprob}   Flow: {flow}   Anisotropy: {anisotropy:.3f}")
        print(f"Safe merge   : max_gap={max_gap}  min_contact={min_contact}")
        print(f"Large contact: {large_contact}")
        print(f"Flow iters   : {niter}")
        print(f"{'='*70}")

        result = {}
        log_lines = []
        log_lock = threading.Lock()

        def _push_log(line):
            with log_lock:
                log_lines.append(line)

        def _worker():
            try:
                def _progress(msg):
                    result["_progress"] = msg
                with capture_live_output(_push_log):
                    labels, stats = _run_cellpose_pipeline(
                        volume, model_path,
                        cellprob=cellprob, flow=flow, anisotropy=anisotropy,
                        max_gap=max_gap, min_contact=min_contact, gt_min=gt_min,
                        large_contact=large_contact, min_hole_size=min_hole_size, min_size=min_size,
                        final_min_fraction=final_min_fraction, niter=niter, scale_zyx=scale,
                        gpu=gpu, progress_cb=_progress, flow_cache_path=flow_cache_path,
                    )
                result["labels"] = labels
                result["stats"]  = stats
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer2 = QTimer(self)

        def _poll2():
            with log_lock:
                new_lines = list(log_lines)
                log_lines.clear()
            if new_lines:
                self._cp_log_view.append("\n".join(new_lines))
                sb = self._cp_log_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if thread.is_alive():
                if "_progress" in result:
                    self._cp_status_lbl.setText(result["_progress"])
                return
            timer2.stop()

            if "error" in result:
                self._cp_status_lbl.setText(f"ERROR: {result['error']}")
                self._cp_run_btn.setEnabled(True)
                self._maybe_send_notify(
                    self._cp_run_notify_cb,
                    f"[ZF-Microglia-AI] Cellpose-SAM Segmentation failed — {stem}",
                    f"Cellpose-SAM Segmentation on {stem} failed:\n\n{result['error']}",
                )
                return

            labels = result["labels"]
            stats  = result["stats"]
            lname  = f"{stem}_cellpose_labels"

            # Always saved, no checkbox -- raw pre-GMM/pre-Krendl cp_masks
            # and the final Krendl-corrected result, every single run, not
            # just when manually building a GT-correction package (see
            # _gt_package.py, which used to be the only place these two
            # ever got written to disk). Same naming convention as that
            # package's own cp_masks_3D.tif/masks_corrected.tif so a fish
            # processed through either path looks the same on disk.
            out_dir = self._output_dir()
            cp_masks_path = out_dir / f"{stem}_cp_masks.tif"
            cp_masks_corrected_path = out_dir / f"{stem}_cp_masks_corrected.tif"
            tifffile.imwrite(str(cp_masks_path), stats["raw_masks"].astype(np.int32))
            tifffile.imwrite(str(cp_masks_corrected_path), labels.astype(np.int32))

            if lname in self._viewer.layers:
                self._viewer.layers.remove(lname)
            self._viewer.add_labels(labels, name=lname, scale=scale)

            porous = stats.get("porous_cells") or {}
            porous_note = ""
            if porous:
                sol_thr = stats.get("solidity_threshold", 0.5)
                ids_str = ", ".join(
                    f"#{lid} (solidity={sol:.2f})"
                    for lid, sol in sorted(porous.items(), key=lambda kv: kv[1])
                )
                porous_note = (
                    f" WARNING: {len(porous)} cell(s) look porous/skeletonized "
                    f"(solidity < {sol_thr:.2f}): {ids_str} -- select these label "
                    f"IDs in the layer to inspect. Try Re-run This Cell Only with "
                    f"a stricter Cellprob (e.g. -0.3) first -- usually a faint "
                    f"cell's noisy probability map, not a flow-convergence issue; "
                    f"raising Flow iterations alone rarely fixes it by itself."
                )

            self._cp_status_lbl.setText(
                f"Done — {stats['n_final']} cells "
                f"(raw={stats['n_raw']} -> gmm={stats['n_after_gmm']} -> "
                f"safe={stats['n_after_safe_merge']} -> large={stats['n_after_large_contact']}). "
                f"Saved {cp_masks_path.name} and {cp_masks_corrected_path.name}."
                f"{porous_note}"
            )
            self._cp_run_btn.setEnabled(True)
            self._maybe_send_notify(
                self._cp_run_notify_cb,
                f"[ZF-Microglia-AI] Cellpose-SAM Segmentation done — {stem}",
                f"Cellpose-SAM Segmentation on {stem} finished.\n\n"
                f"{stats['n_final']} cells (raw={stats['n_raw']} -> gmm={stats['n_after_gmm']} "
                f"-> safe={stats['n_after_safe_merge']} -> large={stats['n_after_large_contact']})."
                f"{porous_note}",
            )

            print(f"{'='*70}")
            print(f"CELLPOSE-SAM SEGMENTATION COMPLETE — {stats['n_final']} cells")
            print(f"  raw={stats['n_raw']}  after_gmm={stats['n_after_gmm']}"
                  f"  after_safe_merge={stats['n_after_safe_merge']}"
                  f"  after_large_contact={stats['n_after_large_contact']}")
            if porous:
                print(f"  {porous_note.strip()}")
            print(f"{'='*70}\n")

        timer2.timeout.connect(_poll2)
        timer2.start(500)

    def _on_rerun_single_cell(self):
        # _active_layer() only ever returns an Image layer -- if the
        # currently *selected* layer is instead the segmented Labels
        # layer itself (the normal state right after a full run, since
        # napari auto-selects the layer add_labels() just created), it
        # silently falls back to "topmost Image layer", which is wrong
        # whenever more than one Image layer is loaded (e.g. ch0+ch1 from
        # multi-channel loading) -- previously surfaced as a false
        # "no '<stem>_cellpose_labels' layer found" error even though the
        # user was looking straight at it. Resolve the stem from the
        # active Labels layer's own name first, when that's what's
        # actually selected, instead of ignoring the selection outright.
        active = self._viewer.layers.selection.active
        target = None
        lname = None
        if (isinstance(active, napari.layers.Labels)
                and active.name.endswith("_cellpose_labels")):
            stem_candidate = active.name[: -len("_cellpose_labels")]
            if stem_candidate in self._viewer.layers:
                cand_layer = self._viewer.layers[stem_candidate]
                if isinstance(cand_layer, napari.layers.Image):
                    target = cand_layer
                    lname = active.name

        if target is None:
            target = self._active_layer()
        if target is None:
            self._cp_status_lbl.setText(
                "Select the brain_only layer first (the same one Cellpose-SAM "
                "Segmentation ran on)."
            )
            return
        volume = np.asarray(target.data)
        if volume.ndim != 3:
            self._cp_status_lbl.setText(f"ERROR: 3D volume required, got {volume.ndim}D.")
            return

        stem = target.name
        if lname is None:
            lname = f"{stem}_cellpose_labels"
        if lname not in self._viewer.layers:
            self._cp_status_lbl.setText(
                f"ERROR: no '{lname}' layer found — run Cellpose-SAM Segmentation "
                f"on this volume first."
            )
            return
        labels_layer = self._viewer.layers[lname]
        labels = np.asarray(labels_layer.data)
        if labels.ndim != 3:
            # A stray singleton axis (e.g. from a save/reload round-trip
            # through a TIFF writer/reader that adds a unit-length T or C
            # dimension) is common enough to auto-recover from rather than
            # crash on -- squeeze first, only error if that doesn't get to
            # exactly 3D.
            squeezed = np.squeeze(labels)
            if squeezed.ndim == 3:
                labels = squeezed
            else:
                self._cp_status_lbl.setText(
                    f"ERROR: '{lname}' is {labels.ndim}D (shape={labels.shape}), "
                    f"expected 3D even after squeezing singleton axes -- "
                    f"re-run the full segmentation on this volume, or fix "
                    f"the layer's shape before re-running a single cell."
                )
                return
        if labels.shape != volume.shape:
            self._cp_status_lbl.setText(
                f"ERROR: '{lname}' shape {labels.shape} doesn't match the "
                f"volume's shape {volume.shape} -- make sure this is really "
                f"the labels layer produced from this exact volume."
            )
            return

        label_id = self._cp_relabel_id_spin.value()
        if not np.any(labels == label_id):
            self._cp_status_lbl.setText(f"ERROR: label {label_id} not found in '{lname}'.")
            return

        model_path = self._state.get("cellpose_model_path")
        if not model_path or not Path(model_path).is_file():
            self._cp_status_lbl.setText("ERROR: no Cellpose-SAM model selected — browse to a checkpoint.")
            return
        if not _is_valid_cellpose_checkpoint(model_path):
            self._cp_status_lbl.setText(
                f"ERROR: {Path(model_path).name} doesn't look like a valid PyTorch "
                f"checkpoint (not a zip archive) -- re-browse to the correct file."
            )
            return

        cellprob      = self._cp_cellprob_slider.value()
        flow          = _FLOW_THRESHOLD_FIXED
        max_gap       = self._cp_maxgap_slider.value()
        min_contact   = self._cp_mincontact_slider.value()
        gt_min        = self._current_min_volume()
        large_contact = self._cp_largecontact_slider.value()
        min_hole_size = self._hole_slider.value()
        min_size      = self._cp_minsize_spin.value()
        final_min_fraction = self._finalfrac_spin.value()
        niter         = self._cp_niter_spin.value()

        scale = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1.0, 1.0, 1.0)
        z, y, x = scale
        xy = (x + y) / 2.0
        anisotropy = z / xy if xy > 0 else 5.747

        gpu = torch.cuda.is_available()

        # Disable both buttons -- this splices into the same labels
        # array a full run would overwrite outright, so the two must
        # not run concurrently.
        self._cp_run_btn.setEnabled(False)
        self._cp_relabel_btn.setEnabled(False)
        self._cp_status_lbl.setText(f"Re-running label {label_id} only (device={'cuda' if gpu else 'cpu'})...")

        result = {}

        def _progress(msg):
            result["_progress"] = msg

        def _worker():
            try:
                new_labels, info = _rerun_single_cell(
                    volume, labels, label_id, model_path,
                    cellprob=cellprob, flow=flow, anisotropy=anisotropy,
                    max_gap=max_gap, min_contact=min_contact, large_contact=large_contact,
                    gt_min=gt_min, gpu=gpu, min_hole_size=min_hole_size, min_size=min_size,
                    final_min_fraction=final_min_fraction, niter=niter, scale_zyx=scale,
                    progress_cb=_progress,
                )
                result["new_labels"] = new_labels
                result["info"] = info
            except Exception as exc:
                traceback.print_exc()
                result["error"] = str(exc)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        timer3 = QTimer(self)

        def _poll3():
            if thread.is_alive():
                if "_progress" in result:
                    self._cp_status_lbl.setText(result["_progress"])
                return
            timer3.stop()
            self._cp_run_btn.setEnabled(True)
            self._cp_relabel_btn.setEnabled(True)

            if "error" in result:
                self._cp_status_lbl.setText(f"ERROR: {result['error']}")
                return

            info = result["info"]
            labels_layer.data[:] = result["new_labels"]  # in-place -- see Resort Labels above for why
            labels_layer.refresh()

            new_ids_str = ", ".join(str(i) for i in info["new_labels"]) or "(none -- discarded, no overlap survived)"
            porous_note = ""
            if info["porous_cells"]:
                porous_note = (
                    f" Still porous after re-run: "
                    + ", ".join(f"#{lid} (solidity={sol:.2f})" for lid, sol in info["porous_cells"].items())
                    + " -- try a stricter Cellprob (e.g. -0.3) instead of/alongside "
                    "a higher niter; usually the more effective lever for a faint cell."
                )
            self._cp_status_lbl.setText(
                f"Done — label {label_id} replaced with {info['n_new']} new label(s): "
                f"{new_ids_str}.{porous_note}"
            )
            print(f"RE-RUN SINGLE CELL — label {label_id} -> {info['n_new']} new label(s): {new_ids_str}")
            if porous_note:
                print(f"  {porous_note.strip()}")

        timer3.timeout.connect(_poll3)
        timer3.start(500)

    def _on_kr_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select image to sweep", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._kr_img_edit.setText(path_str)

    def _on_kr_browse_gt(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._kr_gt_edit.setText(path_str)

    def _on_kr_run_sweep(self):
        if self._krendl_sweep_job.get("thread") and self._krendl_sweep_job["thread"].is_alive():
            self._kr_status_lbl.setText("A sweep is already running.")
            return

        model_path = self._state.get("cellpose_model_path")
        if not model_path or not Path(model_path).is_file():
            self._kr_status_lbl.setText("ERROR: no Cellpose-SAM model selected — browse to a checkpoint above.")
            return
        if not _is_valid_cellpose_checkpoint(model_path):
            self._kr_status_lbl.setText(
                f"ERROR: {Path(model_path).name} doesn't look like a valid PyTorch "
                f"checkpoint (not a zip archive) -- re-browse to the correct file."
            )
            return
        img_path = self._kr_img_edit.text().strip()
        gt_path = self._kr_gt_edit.text().strip()
        if not (img_path and gt_path):
            self._kr_status_lbl.setText("ERROR: set Image and GT labels paths first.")
            return
        for label_str, p in (("Image", img_path), ("GT labels", gt_path)):
            if not Path(p).exists():
                self._kr_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return

        cp_min = self._kr_cpmin_spin.value()
        cp_max = self._kr_cpmax_spin.value()
        cp_step = self._kr_cpstep_spin.value()
        if cp_max < cp_min:
            self._kr_status_lbl.setText("ERROR: Cellprob max must be >= min.")
            return
        cellprobs = list(np.round(np.arange(cp_min, cp_max + cp_step / 2, cp_step), 4))

        anisotropy = self._kr_scalez_spin.value() / self._kr_scalexy_spin.value()
        scale_zyx = (self._kr_scalez_spin.value(), self._kr_scalexy_spin.value(), self._kr_scalexy_spin.value())
        min_size = self._cp_minsize_spin.value()
        gpu = torch.cuda.is_available()
        current_cellprob = self._cp_cellprob_slider.value()

        sieve_on = self._kr_sieve_cb.isChecked()
        sieve_refine_steps = [
            (self._kr_sieve2_w_spin.value(), self._kr_sieve2_s_spin.value()),
            (self._kr_sieve3_w_spin.value(), self._kr_sieve3_s_spin.value()),
        ]

        cancel_event = threading.Event()
        result = {}
        progress = {"lines": []}
        progress_lock = threading.Lock()

        def _progress_cb(msg):
            with progress_lock:
                progress["lines"].append(msg)

        def _worker():
            try:
                volume = tifffile.imread(img_path)
                gt_labels = tifffile.imread(gt_path).astype(np.int32)
                # min_hole_size measured from this GT's own real holes,
                # same convention every other GT-sweep tool in this plugin
                # uses -- not guessed, not left at masks_from_flows()'s own
                # bare default.
                min_hole_size = _psw.min_hole_size_from_gt(gt_labels)
                _progress_cb(f"min_hole_size computed from this GT's own real holes: {min_hole_size} vox")
                # do_3D's own progress is otherwise invisible for the ~3h
                # this can take -- see _live_progress.py.
                with capture_live_output(_progress_cb):
                    if sieve_on:
                        precomputed_holder = [None]

                        def _run_stage(cp_values):
                            sw = _ksw.run_cellprob_voxel_sweep(
                                volume, gt_labels, model_path, cp_values,
                                anisotropy=anisotropy, gpu=gpu, min_size=min_size,
                                min_hole_size=min_hole_size,
                                progress_cb=_progress_cb, cancel_event=cancel_event,
                                precomputed=precomputed_holder[0],
                            )
                            precomputed_holder[0] = sw["precomputed"]
                            return sw["best_cellprob"], sw

                        sieve = _sieve.run_sieve(
                            _run_stage, lo=cp_min, hi=cp_max, coarse_step=cp_step,
                            refine_steps=sieve_refine_steps, progress_cb=_progress_cb,
                            cancel_check=cancel_event.is_set,
                        )
                        result["sweep"] = sieve["final_result"]
                        result["sieve_stages"] = sieve["stages"]
                    else:
                        result["sweep"] = _ksw.run_cellprob_voxel_sweep(
                            volume, gt_labels, model_path, cellprobs,
                            anisotropy=anisotropy, gpu=gpu, min_size=min_size,
                            min_hole_size=min_hole_size,
                            progress_cb=_progress_cb, cancel_event=cancel_event,
                        )

                    # Steps 2-3: at the now-fixed winning Cellprob, form raw
                    # cp_masks reusing the SAME cached flows (zero extra
                    # do_3D), then calibrate max_gap/min_contact against GT
                    # from the network's own real fragmentation behaviour.
                    # Skipped on cancellation/no-winner -- nothing to
                    # calibrate against.
                    sw = result["sweep"]
                    if not sw.get("cancelled") and sw.get("best_cellprob") is not None:
                        best_cp = sw["best_cellprob"]
                        model, dP, cellprob_map, shape = sw["precomputed"]
                        _progress_cb(f"Forming raw cp_masks at the winning cellprob={best_cp} "
                                     f"(reusing cached flows, no re-inference)...")
                        raw_cp_masks = _masks_from_flows(
                            model, dP, cellprob_map, shape, best_cp, flow=0.4,
                            min_size=min_size, min_hole_size=min_hole_size,
                        )
                        _progress_cb("Comparing raw cp_masks against GT to calibrate "
                                     "max_gap/min_contact...")
                        merge_stats = _ksw.measure_merge_params_from_prediction(
                            raw_cp_masks, gt_labels, scale_zyx=scale_zyx,
                        )
                        result["merge_stats"] = merge_stats
                        result["merge_recommendation"] = _ksw.recommend_merge_params([merge_stats])
                        _progress_cb(
                            f"Merge calibration: {merge_stats['n_gt_cells_fragmented']} GT cells "
                            f"fragmented, {len(merge_stats['should_merge_gaps_um'])} should-merge / "
                            f"{len(merge_stats['should_not_merge_gaps_um'])} should-not-merge samples."
                        )
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._krendl_sweep_job["thread"] = thread
        self._krendl_sweep_job["cancel_event"] = cancel_event
        self._krendl_sweep_job["result"] = result
        self._krendl_sweep_job["progress"] = progress
        self._krendl_sweep_job["progress_lock"] = progress_lock
        self._krendl_sweep_job["current_cellprob"] = current_cellprob
        self._krendl_sweep_job["fish_key"] = Path(gt_path).stem
        self._kr_run_btn.setEnabled(False)
        self._kr_stop_btn.setEnabled(True)
        self._kr_report_view.clear()
        if sieve_on:
            self._kr_status_lbl.setText(
                f"Sieve mode: stage 1 sweeps {len(cellprobs)} Cellprob values "
                f"(voxel-level Dice/IoU vs GT), then 2 more narrowing stages, "
                f"then calibrates max_gap/min_contact from cp_masks at the "
                f"winner -- one do_3D pass reused throughout..."
            )
        else:
            self._kr_status_lbl.setText(
                f"Predicting flows once (do_3D, {'cuda' if gpu else 'cpu'}), sweeping "
                f"{len(cellprobs)} Cellprob values (voxel-level Dice/IoU vs GT), then "
                f"calibrating max_gap/min_contact from cp_masks at the winner..."
            )
        thread.start()
        self._start_krendl_sweep_polling()

    def _on_kr_stop_sweep(self):
        cancel_event = self._krendl_sweep_job.get("cancel_event")
        if cancel_event:
            cancel_event.set()
        self._kr_status_lbl.setText("Cancelling — finishing the current cellprob value, then stopping...")
        self._kr_stop_btn.setEnabled(False)

    def _start_krendl_sweep_polling(self):
        """Same fast (500ms) poll and non-detached, doesn't-survive-napari-
        closing contract as the plugin's other sweep tools."""
        timer = QTimer(self)
        job = self._krendl_sweep_job

        def _poll():
            with job["progress_lock"]:
                lines = list(job["progress"]["lines"])
                job["progress"]["lines"].clear()
            if lines:
                self._kr_report_view.append("\n".join(lines))
                sb = self._kr_report_view.verticalScrollBar()
                sb.setValue(sb.maximum())

            if job["thread"].is_alive():
                return
            timer.stop()
            job["timer"] = None
            self._kr_run_btn.setEnabled(True)
            self._kr_stop_btn.setEnabled(False)

            result = job["result"]
            if "error" in result:
                self._kr_is_gt_cb.setChecked(False)
                self._kr_status_lbl.setText(f"ERROR during sweep: {result['error'].splitlines()[0]}")
                self._kr_report_view.append("\n" + result["error"])
                self._maybe_send_notify(
                    self._kr_notify_cb,
                    "[ZF-Microglia-AI] Cellprob sweep failed",
                    f"Verify Cellprob (GT Sweep) failed:\n\n{result['error']}",
                )
                return

            sweep = result["sweep"]
            if "sieve_stages" in result:
                parts = []
                for i, st in enumerate(result["sieve_stages"], start=1):
                    parts.append(f"===== Sieve stage {i}/{len(result['sieve_stages'])} "
                                 f"(winner: cellprob={st['best_value']}) =====")
                    parts.append(_ksw.format_cellprob_voxel_sweep_report(
                        st["result"], job["current_cellprob"]
                    ))
                report = "\n\n".join(parts)
            else:
                report = _ksw.format_cellprob_voxel_sweep_report(sweep, job["current_cellprob"])
            merge_stats = result.get("merge_stats")
            merge_rec = result.get("merge_recommendation")
            if merge_stats is not None:
                if merge_rec["max_gap_um"] is None:
                    rec_line = "This fish alone has no fragmented GT cells to calibrate from."
                else:
                    rec_line = (
                        f"This fish's own joint-optimal recommendation: "
                        f"max_gap={merge_rec['max_gap_um']:.2f}µm, contact={merge_rec['contact_vox']}vox "
                        f"(missed_merges={merge_rec['missed_merges']}, false_merges={merge_rec['false_merges']})"
                    )
                report += (
                    "\n\n===== Krendl merge-parameter calibration (raw cp_masks vs GT) =====\n"
                    f"GT cells fragmented by the network: {merge_stats['n_gt_cells_fragmented']}\n"
                    f"Should-merge samples (real same-cell fragments): {len(merge_stats['should_merge_gaps_um'])}\n"
                    f"Should-NOT-merge samples (different real cells / noise): {len(merge_stats['should_not_merge_gaps_um'])}\n"
                    f"Fragments dropped as noise (too small to match any GT cell): {merge_stats['n_fragments_dropped_as_noise']}\n"
                    f"{rec_line}"
                )
            self._kr_report_view.setPlainText(report)
            if sweep.get("cancelled"):
                self._kr_is_gt_cb.setChecked(False)
                self._kr_status_lbl.setText("Sweep cancelled — partial results above.")
            elif sweep["best_cellprob"] is not None:
                best_cp = sweep["best_cellprob"]
                fish_key = job["fish_key"]
                # One-shot, same as Tab 3 Statistics.
                is_gt = self._kr_is_gt_cb.isChecked()
                self._kr_is_gt_cb.setChecked(False)

                if is_gt:
                    # Cellprob isn't a safety floor -- each fish's sweep
                    # just finds that fish's own local optimum, so it's
                    # averaged across every fish swept so far, same
                    # reasoning as BG Threshold/Erosion and Sigma XY/Z.
                    avg_cp = self._update_gt_history("cellpose_cellprob", fish_key, best_cp, mode="mean")
                    self._cp_cellprob_slider.setValue(avg_cp)
                    self._cp_cellprob_recommended_lbl.setText(f"  Recommended Cellprob threshold: {avg_cp:.3f}")
                    self._save_cfg(cellpose_cellprob=avg_cp, cellpose_cellprob_recommended=avg_cp)
                    applied_note = f" Applied average across all fish swept so far: cellprob={avg_cp:.3f}."

                    if merge_stats is not None:
                        # max_gap/min_contact are jointly optimized against
                        # the pooled sample set across every fish
                        # calibrated so far (recommend_merge_params), not
                        # averaged per-fish then combined -- see
                        # _update_merge_stats_history / [[feedback_plugin_ux]].
                        pooled_rec = self._update_merge_stats_history(fish_key, merge_stats)
                        if pooled_rec["max_gap_um"] is None:
                            applied_note += (
                                " No fragmented GT cells seen yet across any fish calibrated "
                                "so far -- max_gap/min_contact left unchanged."
                            )
                        else:
                            gap_val = pooled_rec["max_gap_um"]
                            contact_val = int(round(pooled_rec["contact_vox"]))
                            self._cp_maxgap_slider.setValue(gap_val)
                            self._cp_maxgap_recommended_lbl.setText(f"  Recommended Safe-merge max gap: {gap_val:.2f}µm")
                            self._save_cfg(cellpose_max_gap_um=gap_val, cellpose_max_gap_um_recommended=gap_val)
                            self._cp_mincontact_slider.setValue(contact_val)
                            self._cp_mincontact_recommended_lbl.setText(f"  Recommended Safe-merge min contact: {contact_val} vox")
                            self._save_cfg(cellpose_min_contact_vox=contact_val, cellpose_min_contact_vox_recommended=contact_val)
                            applied_note += (
                                f" Merge params pooled across all fish calibrated so far: "
                                f"max_gap={gap_val:.2f}µm, contact={contact_val}vox "
                                f"(missed_merges={pooled_rec['missed_merges']}, false_merges={pooled_rec['false_merges']})."
                            )
                    applied_note += " Saved."
                else:
                    applied_note = (
                        " Not GT-verified — recommended value(s) NOT updated "
                        "(tick \"This is verified ground truth\" before running to apply)."
                    )
                self._kr_status_lbl.setText(
                    f"This fish's best: cellprob={best_cp} "
                    f"(Dice={sweep['results'][best_cp]['dice']:.1f}%)."
                    f"{applied_note}"
                )
            else:
                self._kr_is_gt_cb.setChecked(False)
                self._kr_status_lbl.setText("Sweep finished but no grid points could be scored.")

            self._maybe_send_notify(
                self._kr_notify_cb,
                "[ZF-Microglia-AI] Cellprob sweep done",
                f"Verify Cellprob (GT Sweep) finished.\n\n{self._kr_status_lbl.text()}",
            )

        timer.timeout.connect(_poll)
        timer.start(500)
        job["timer"] = timer

    def _on_gtp_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select source image", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._gtp_img_edit.setText(path_str)

    def _on_gtp_browse_masks(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select Krendl segmentation output", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._gtp_masks_edit.setText(path_str)

    def _on_gtp_browse_raw(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select raw pre-merge Cellpose masks (optional)", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._gtp_raw_edit.setText(path_str)

    def _on_gtp_browse_guide(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GROUND_TRUTH_CREATION_GUIDE.md", "", "Markdown files (*.md)")
        if path_str:
            self._gtp_guide_edit.setText(path_str)

    def _on_gtp_browse_out(self):
        path_str = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path_str:
            self._gtp_out_edit.setText(path_str)

    def _on_gtp_run(self):
        if self._gt_package_job.get("thread") and self._gt_package_job["thread"].is_alive():
            self._gtp_status_lbl.setText("A package build is already running.")
            return

        stem = self._gtp_stem_edit.text().strip()
        img_path = self._gtp_img_edit.text().strip()
        masks_path = self._gtp_masks_edit.text().strip()
        out_dir = self._gtp_out_edit.text().strip()
        if not (stem and img_path and masks_path and out_dir):
            self._gtp_status_lbl.setText("ERROR: set Fish stem, Source image, Krendl masks, and Output folder.")
            return
        for label_str, p in (("Source image", img_path), ("Krendl masks", masks_path)):
            if not Path(p).exists():
                self._gtp_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return
        raw_path = self._gtp_raw_edit.text().strip() or None
        if raw_path and not Path(raw_path).exists():
            self._gtp_status_lbl.setText(f"ERROR: Raw Cellpose masks not found: {raw_path}")
            return
        guide_path = self._gtp_guide_edit.text().strip() or None

        scale = self._get_layer_scale()

        self._gtp_run_btn.setEnabled(False)
        self._gtp_status_lbl.setText("Building GT-correction package...")

        result = {}

        def _worker():
            try:
                package_dir, zip_path, n_cells = _gtp.build_gt_package(
                    stem, out_dir, img_path, masks_path,
                    raw_cellpose_masks_path=raw_path, scale_zyx=scale, guide_path=guide_path,
                    progress_cb=lambda msg: result.update(_progress=msg),
                )
                result["package_dir"] = package_dir
                result["zip_path"] = zip_path
                result["n_cells"] = n_cells
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._gt_package_job["thread"] = thread
        self._gt_package_job["result"] = result
        thread.start()

        timer = QTimer(self)

        def _poll():
            if "_progress" in result:
                self._gtp_status_lbl.setText(result["_progress"])
            if thread.is_alive():
                return
            timer.stop()
            self._gt_package_job["timer"] = None
            self._gtp_run_btn.setEnabled(True)

            if "error" in result:
                self._gtp_status_lbl.setText(f"ERROR: {result['error'].splitlines()[0]}")
                print(result["error"])
                return

            self._gtp_status_lbl.setText(
                f"Done — {result['n_cells']} cells. Package: {result['zip_path']}"
            )

        timer.timeout.connect(_poll)
        timer.start(500)
        self._gt_package_job["timer"] = timer

    def _on_xz_browse_img(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select source image", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._xz_img_edit.setText(path_str)

    def _on_xz_browse_gt(self):
        path_str, _ = QFileDialog.getOpenFileName(self, "Select GT label volume", "", "TIFF files (*.tif *.tiff)")
        if path_str:
            self._xz_gt_edit.setText(path_str)

    def _on_xz_browse_out(self):
        path_str = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path_str:
            self._xz_out_edit.setText(path_str)

    def _on_xz_run(self):
        """Generates XZYZ patches, then (by default -- see _xz_clean_cb)
        immediately cleans truncated incidental-neighbor labels in the
        same background thread, so a fresh extraction is clean by
        construction rather than needing a separate remembered step."""
        if self._xz_patches_job.get("thread") and self._xz_patches_job["thread"].is_alive():
            self._xz_status_lbl.setText("An extraction is already running.")
            return

        img_path = self._xz_img_edit.text().strip()
        gt_path = self._xz_gt_edit.text().strip()
        out_dir = self._xz_out_edit.text().strip()
        if not (img_path and gt_path and out_dir):
            self._xz_status_lbl.setText("ERROR: set Image, GT labels, and Output folder.")
            return
        for label_str, p in (("Image", img_path), ("GT labels", gt_path)):
            if not Path(p).exists():
                self._xz_status_lbl.setText(f"ERROR: {label_str} not found: {p}")
                return

        anisotropy = self._xz_scalez_spin.value() / self._xz_scalexy_spin.value()
        crop_size = self._xz_cropsize_spin.value()
        ncrops_per_slice = self._xz_ncrops_spin.value()
        max_per_orientation = self._xz_maxn_spin.value()
        min_gt_pixels = self._xz_mingt_spin.value()
        seed = self._xz_seed_spin.value()
        do_clean = self._xz_clean_cb.isChecked()
        threshold = self._xz_threshold_spin.value()

        self._xz_run_btn.setEnabled(False)
        self._xz_status_lbl.setText("Extracting XZYZ patches...")

        result = {}

        def _worker():
            try:
                gen = _xzp.generate_xzyz_patches(
                    img_path, gt_path, out_dir, anisotropy,
                    crop_size=crop_size, ncrops_per_slice=ncrops_per_slice,
                    max_per_orientation=max_per_orientation, min_gt_pixels=min_gt_pixels,
                    seed=seed, progress_cb=lambda msg: result.update(_progress=msg),
                )
                result["gen"] = gen
                if do_clean and not gen.get("cancelled"):
                    clean = _ctr.clean_crop_truncation(
                        gen["out_dir"], gt_path, anisotropy, threshold=threshold,
                        progress_cb=lambda msg: result.update(_progress=f"cleanup: {msg}"),
                    )
                    result["clean"] = clean
            except Exception as exc:
                result["error"] = f"{exc}\n{traceback.format_exc()}"

        thread = threading.Thread(target=_worker, daemon=True)
        self._xz_patches_job["thread"] = thread
        self._xz_patches_job["result"] = result
        thread.start()

        timer = QTimer(self)

        def _poll():
            if "_progress" in result:
                self._xz_status_lbl.setText(result["_progress"])
            if thread.is_alive():
                return
            timer.stop()
            self._xz_patches_job["timer"] = None
            self._xz_run_btn.setEnabled(True)

            if "error" in result:
                self._xz_status_lbl.setText(f"ERROR: {result['error'].splitlines()[0]}")
                print(result["error"])
                return

            gen = result["gen"]
            n_total = gen["n_xy"] + gen["n_xz"] + gen["n_yz"]
            msg = f"Extracted {n_total} crops (xy={gen['n_xy']}, xz={gen['n_xz']}, yz={gen['n_yz']})."
            if "clean" in result:
                c = result["clean"]
                msg += (f" Cleanup: {c['n_files_modified']}/{c['n_files_scanned']} files modified, "
                        f"{c['n_labels_zeroed']} truncated labels zeroed.")
            self._xz_status_lbl.setText(msg)

        timer.timeout.connect(_poll)
        timer.start(500)
        self._xz_patches_job["timer"] = timer

