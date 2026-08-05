"""
_widget.py — SkinRemoverWidget napari dock panel.
"""

import json
import threading
import traceback
from pathlib import Path

import numpy as np
import tifffile
import torch
import napari

from qtpy.QtWidgets import (
    QPushButton, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QFileDialog, QSizePolicy, QButtonGroup, QRadioButton,
    QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QScrollArea, QGroupBox,
)
from qtpy.QtCore import Qt, QTimer
from superqt import QLabeledSlider, QLabeledDoubleSlider

from ._io import load_file
from ._inference import DEFAULT_MODEL, _SKIN_SEG_DIR, run_inference
from ._background import remove_outside_brain, remove_global, fill_outside_brain_random
from ._labeling import create_labels, resort_labels, split_label
from ._statistics import compute_stats
from ._cellpose_seg import run_full_pipeline as _run_cellpose_pipeline

_CONFIG_PATH = Path.home() / ".config" / "napari-skin-remover" / "config.json"

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
                           decimals=None, slider_max_width=110):
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


def _sep():
    """Thin horizontal separator line."""
    w = QWidget()
    w.setFixedHeight(1)
    w.setStyleSheet("background-color: #666;")
    return w


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


class SkinRemoverWidget(QWidget):
    """
    Napari dock panel for MONAI skin removal.

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
        # Model path priority: saved config > hardcoded default > None
        saved_model = Path(cfg.get("model_path", ""))
        if saved_model.exists():
            initial_model = saved_model
        elif DEFAULT_MODEL.exists():
            initial_model = DEFAULT_MODEL
        else:
            initial_model = None
        # Cellpose-SAM checkpoint: saved config only (no bundled default —
        # this is a project-specific fine-tuned model, not shipped with the plugin)
        saved_cp_model = Path(cfg.get("cellpose_model_path", ""))
        initial_cp_model = saved_cp_model if saved_cp_model.exists() else None
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

        # ============================================================ #
        # TAB 1 — Skin Remover
        # ============================================================ #
        tab1 = QWidget()
        t1 = QVBoxLayout()
        t1.setSpacing(6)

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
        self._thresh_slider.setValue(0.25)
        thresh_row.addWidget(self._thresh_slider)
        self._thresh_spin = _add_reliable_spinbox(
            thresh_row, self._thresh_slider, 0.01, 0.99, 0.01, decimals=2
        )
        t1.addLayout(thresh_row)

        erosion_row = QHBoxLayout()
        erosion_row.addWidget(QLabel("Erosion (vox):"))
        self._erosion_slider = QLabeledSlider(Qt.Horizontal)
        self._erosion_slider.setMinimum(0)
        self._erosion_slider.setMaximum(15)
        self._erosion_slider.setValue(0)
        erosion_row.addWidget(self._erosion_slider)
        self._erosion_spin = _add_reliable_spinbox(
            erosion_row, self._erosion_slider, 0, 15, 1
        )
        t1.addLayout(erosion_row)
        erosion_note = QLabel(
            "  Erodes mask before applying to brain_only\n"
            "  (raw brain_mask is always saved un-eroded)"
        )
        erosion_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t1.addWidget(erosion_note)

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
        self._tol_slider.setValue(1.40)
        tol_row.addWidget(self._tol_slider)
        self._tol_spin = _add_reliable_spinbox(
            tol_row, self._tol_slider, 0.00, 2.00, 0.01, decimals=2
        )
        t1.addLayout(tol_row)

        bg_note = QLabel(
            "  Probe: inside-brain mode (post-inference)\n"
            "  Mode 1 & 2 use BG Threshold  |  Mode 3: no threshold"
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

        self._run_btn = QPushButton("Run Skin-Remover")
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t1.addWidget(self._run_btn)

        self._status_lbl = QLabel("Status: Ready")
        self._status_lbl.setWordWrap(True)
        t1.addWidget(self._status_lbl)

        t1.addStretch()
        tab1.setLayout(t1)
        tabs.addTab(tab1, "Skin Remover")

        # ============================================================ #
        # TAB 2 — Create Labels
        # ============================================================ #
        tab2 = QWidget()
        t2 = QVBoxLayout()
        t2.setSpacing(6)

        self._labels_mode_hint = QLabel("")
        self._labels_mode_hint.setWordWrap(True)
        self._labels_mode_hint.setStyleSheet("color: #8ab; font-size: 10px; font-style: italic;")
        t2.addWidget(self._labels_mode_hint)

        # ── Pixel Classifier (union-find labels) — shown for _NoBG layers ── #
        self._pixel_classifier_group = QGroupBox("Pixel Classifier — Union-Find Labels")
        pcg = QVBoxLayout()
        pcg.setSpacing(6)

        lbl_note = QLabel(
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
        self._sxy_slider.setValue(1.5)
        sxy_row.addWidget(self._sxy_slider)
        self._sxy_spin = _add_reliable_spinbox(
            sxy_row, self._sxy_slider, 0.0, 5.0, 0.1, decimals=1
        )
        pcg.addLayout(sxy_row)

        sz_row = QHBoxLayout()
        sz_row.addWidget(QLabel("Smooth σ Z:"))
        self._sz_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._sz_slider.setDecimals(1)
        self._sz_slider.setMinimum(0.0)
        self._sz_slider.setMaximum(5.0)
        self._sz_slider.setSingleStep(0.1)
        self._sz_slider.setValue(3.0)
        sz_row.addWidget(self._sz_slider)
        self._sz_spin = _add_reliable_spinbox(
            sz_row, self._sz_slider, 0.0, 5.0, 0.1, decimals=1
        )
        pcg.addLayout(sz_row)

        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Min volume (vox):"))
        self._area_slider = QLabeledSlider(Qt.Horizontal)
        self._area_slider.setMinimum(5000)
        self._area_slider.setMaximum(10000)
        self._area_slider.setValue(7500)
        area_row.addWidget(self._area_slider)
        self._area_spin = _add_reliable_spinbox(
            area_row, self._area_slider, 5000, 10000, 100
        )
        pcg.addLayout(area_row)

        self._labels_btn = QPushButton("Create Labels")
        self._labels_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        pcg.addWidget(self._labels_btn)

        self._labels_status_lbl = QLabel("")
        self._labels_status_lbl.setWordWrap(True)
        pcg.addWidget(self._labels_status_lbl)

        self._pixel_classifier_group.setLayout(pcg)
        t2.addWidget(self._pixel_classifier_group)

        # ── Cellpose-SAM segmentation (do_3D + Krendl corrections) — shown for _ExtRm layers ── #
        self._cellpose_group = QGroupBox("Cellpose-SAM Segmentation")
        cpg = QVBoxLayout()
        cpg.setSpacing(6)

        cp_note = QLabel(
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

        cp_cellprob_row = QHBoxLayout()
        cp_cellprob_row.addWidget(QLabel("Cellprob threshold:"))
        self._cp_cellprob_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._cp_cellprob_slider.setDecimals(2)
        self._cp_cellprob_slider.setMinimum(-6.0)
        self._cp_cellprob_slider.setMaximum(6.0)
        self._cp_cellprob_slider.setSingleStep(0.1)
        self._cp_cellprob_slider.setValue(-2.5)
        cp_cellprob_row.addWidget(self._cp_cellprob_slider)
        self._cp_cellprob_spin = _add_reliable_spinbox(
            cp_cellprob_row, self._cp_cellprob_slider, -6.0, 6.0, 0.1, decimals=2
        )
        cpg.addLayout(cp_cellprob_row)

        cp_flow_row = QHBoxLayout()
        cp_flow_row.addWidget(QLabel("Flow threshold:"))
        self._cp_flow_slider = QLabeledDoubleSlider(Qt.Horizontal)
        self._cp_flow_slider.setDecimals(2)
        self._cp_flow_slider.setMinimum(0.0)
        self._cp_flow_slider.setMaximum(1.0)
        self._cp_flow_slider.setSingleStep(0.05)
        self._cp_flow_slider.setValue(0.4)
        cp_flow_row.addWidget(self._cp_flow_slider)
        self._cp_flow_spin = _add_reliable_spinbox(
            cp_flow_row, self._cp_flow_slider, 0.0, 1.0, 0.05, decimals=2
        )
        cpg.addLayout(cp_flow_row)

        cp_maxgap_row = QHBoxLayout()
        cp_maxgap_row.addWidget(QLabel("Safe-merge max gap (vox):"))
        self._cp_maxgap_slider = QLabeledSlider(Qt.Horizontal)
        self._cp_maxgap_slider.setMinimum(0)
        self._cp_maxgap_slider.setMaximum(20)
        self._cp_maxgap_slider.setValue(2)
        cp_maxgap_row.addWidget(self._cp_maxgap_slider)
        self._cp_maxgap_spin = _add_reliable_spinbox(
            cp_maxgap_row, self._cp_maxgap_slider, 0, 20, 1
        )
        cpg.addLayout(cp_maxgap_row)

        cp_mincontact_row = QHBoxLayout()
        cp_mincontact_row.addWidget(QLabel("Safe-merge min contact (vox):"))
        self._cp_mincontact_slider = QLabeledSlider(Qt.Horizontal)
        self._cp_mincontact_slider.setMinimum(0)
        self._cp_mincontact_slider.setMaximum(200)
        self._cp_mincontact_slider.setValue(10)
        cp_mincontact_row.addWidget(self._cp_mincontact_slider)
        self._cp_mincontact_spin = _add_reliable_spinbox(
            cp_mincontact_row, self._cp_mincontact_slider, 0, 200, 1
        )
        cpg.addLayout(cp_mincontact_row)

        cp_largecontact_row = QHBoxLayout()
        cp_largecontact_row.addWidget(QLabel("Large-contact merge (vox):"))
        self._cp_largecontact_slider = QLabeledSlider(Qt.Horizontal)
        self._cp_largecontact_slider.setMinimum(1)
        self._cp_largecontact_slider.setMaximum(2000)
        self._cp_largecontact_slider.setValue(20)
        cp_largecontact_row.addWidget(self._cp_largecontact_slider)
        self._cp_largecontact_spin = _add_reliable_spinbox(
            cp_largecontact_row, self._cp_largecontact_slider, 1, 2000, 10
        )
        cpg.addLayout(cp_largecontact_row)

        self._cp_run_btn = QPushButton("Run Cellpose-SAM Segmentation")
        self._cp_run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        cpg.addWidget(self._cp_run_btn)

        self._cp_status_lbl = QLabel("")
        self._cp_status_lbl.setWordWrap(True)
        cpg.addWidget(self._cp_status_lbl)

        self._cellpose_group.setLayout(cpg)
        t2.addWidget(self._cellpose_group)

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
        tabs.addTab(tab2, "Create Labels")

        # ============================================================ #
        # TAB 3 — Statistics
        # ============================================================ #
        tab3 = QWidget()
        t3 = QVBoxLayout()
        t3.setSpacing(6)

        cfg = self._state.get("config", {})

        t3_note = QLabel(
            "Select a Labels layer, then choose a description\n"
            "backend and click Generate Statistics."
        )
        t3_note.setWordWrap(True)
        t3_note.setStyleSheet("color: #aaa; font-size: 10px;")
        t3.addWidget(t3_note)

        t3.addWidget(_sep())

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
        self._api_key_edit = QLineEdit(cfg.get("api_key", ""))
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setPlaceholderText("sk-… or ant-…")
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

        self._stats_btn = QPushButton("Generate Statistics")
        self._stats_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 6px; }")
        t3.addWidget(self._stats_btn)

        self._stats_status_lbl = QLabel("")
        self._stats_status_lbl.setWordWrap(True)
        t3.addWidget(self._stats_status_lbl)

        t3.addStretch()
        tab3.setLayout(t3)
        tabs.addTab(tab3, "Statistics")

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
        self._labels_btn.clicked.connect(self._on_create_labels)
        self._cp_model_browse_btn.clicked.connect(self._on_browse_cp_model)
        self._cp_run_btn.clicked.connect(self._on_run_cellpose_seg)
        self._resort_btn.clicked.connect(self._on_resort_labels)
        self._split_use_sel_btn.clicked.connect(self._on_use_selected_label)
        self._split_btn.clicked.connect(self._on_split_label)
        self._save_labels_btn.clicked.connect(self._on_save_labels)
        self._stats_backend_combo.currentIndexChanged.connect(self._on_stats_backend_changed)
        self._stats_btn.clicked.connect(self._on_generate_stats)
        self._viewer.layers.events.inserted.connect(self._refresh_layer_info)
        self._viewer.layers.events.removed.connect(self._refresh_layer_info)
        self._viewer.layers.selection.events.changed.connect(self._refresh_layer_info)
        self._viewer.layers.events.inserted.connect(self._refresh_stats_layers)
        self._viewer.layers.events.removed.connect(self._refresh_stats_layers)
        # Apply initial panel visibility
        self._on_stats_backend_changed()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _status(self, msg):
        self._status_lbl.setText(f"Status: {msg}")

    def _on_bg_mode_changed(self, btn):
        """Enable/disable tolerance slider depending on mode."""
        mode = self._bg_group.checkedId()
        has_tol = mode in (1, 2)
        self._tol_slider.setEnabled(has_tol)
        self._tol_spin.setEnabled(has_tol)
        self._tol_lbl.setEnabled(has_tol)

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
        self._stats_image_combo.blockSignals(True)
        self._stats_image_combo.clear()
        self._stats_image_combo.addItem("None", None)
        for lyr in self._viewer.layers:
            if isinstance(lyr, napari.layers.Image):
                self._stats_image_combo.addItem(lyr.name, lyr.name)
                if lyr.name == cur_img:
                    self._stats_image_combo.setCurrentIndex(
                        self._stats_image_combo.count() - 1
                    )
        self._stats_image_combo.blockSignals(False)

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
        Show only the labeling tool that matches the active layer's
        background-removal mode (by filename suffix, see _BG_SUFFIX):
          _ExtRm    -> Cellpose-SAM Segmentation (this pipeline was always
                       run on outside-brain-only-removed layers all session)
          _NoBG     -> Pixel Classifier (globally background-removed —
                       what the union-find tool was designed for)
          _RndFill  -> neither (presentation/visualization output only,
                       not meant to be labeled)
          anything else (raw layer, no recognized suffix) -> neither,
                       with a hint explaining what's needed
        """
        lyr = self._active_layer()
        name = lyr.name if lyr is not None else ""

        if name.endswith("_ExtRm"):
            self._cellpose_group.setVisible(True)
            self._pixel_classifier_group.setVisible(False)
            self._labels_mode_hint.setText(
                f'Active layer "{name}" ends in _ExtRm → showing Cellpose-SAM Segmentation.'
            )
        elif name.endswith("_NoBG"):
            self._cellpose_group.setVisible(False)
            self._pixel_classifier_group.setVisible(True)
            self._labels_mode_hint.setText(
                f'Active layer "{name}" ends in _NoBG → showing Pixel Classifier.'
            )
        elif name.endswith("_RndFill"):
            self._cellpose_group.setVisible(False)
            self._pixel_classifier_group.setVisible(False)
            self._labels_mode_hint.setText(
                f'Active layer "{name}" ends in _RndFill (presentation/visualization '
                f"output only) — no labeling tool applies here."
            )
        else:
            self._cellpose_group.setVisible(False)
            self._pixel_classifier_group.setVisible(False)
            self._labels_mode_hint.setText(
                "Select a brain_only layer from Tab 1 (_ExtRm or _NoBG) to see "
                "labeling options here."
            )

        # Downstream label tools (Resort/Split/Save) only make sense once a
        # creation option is applicable to the current selection — same
        # "based on what the user selects" logic, cascaded one step further.
        creation_available = self._cellpose_group.isVisible() or self._pixel_classifier_group.isVisible()
        self._downstream_label_tools.setVisible(creation_available)

        # Statistics tab needs an actual Labels layer to operate on — a
        # different, more direct condition than "is a creation tool shown",
        # since labels can persist even after switching the active layer.
        has_labels = any(isinstance(l, napari.layers.Labels) for l in self._viewer.layers)
        self._tabs.setTabVisible(2, has_labels)

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

    def _save_cfg(self, **kwargs) -> None:
        """Merge kwargs into the config and persist."""
        cfg = self._state.get("config", {})
        cfg.update(kwargs)
        self._state["config"] = cfg
        _save_config(cfg)

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
        self._state["cellpose_model_path"] = p
        self._cp_model_lbl.setText(path_str)
        self._save_cfg(cellpose_model_path=str(p))
        self._labels_status_lbl.setText(f"Cellpose-SAM model: {p.name}")

    def _on_run(self):
        if not self._state["model_path"] or not Path(self._state["model_path"]).exists():
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

        print(f"\n{'='*70}")
        print(f"SKIN-REMOVER — {stem}  shape={volume.shape}")
        print(f"Model    : {model_path.name}")
        print(f"Threshold: {threshold}   Device: {device}")
        print(f"Erosion  : {erosion_voxels} voxel(s)")
        bg_mode_str = {
            0: "Off",
            1: f"Remove outside-brain (BG threshold={bg_tolerance_pct:.2f})",
            2: f"Remove globally (BG threshold={bg_tolerance_pct:.2f})",
            3: "Fill sub-background with random noise",
        }[bg_mode]
        print(f"BG mode  : {bg_mode_str}")
        print(f"Scale    : Z={scale[0]:.4f}  Y={scale[1]:.4f}  X={scale[2]:.4f} µm")
        print(f"{'='*70}")

        result = {}

        def _worker():
            try:
                # Step 1: inference on original volume (never bg-removed)
                brain_mask, brain_only = run_inference(
                    volume, model_path, threshold, device, erosion_voxels
                )

                # Step 2: optional background processing
                if bg_mode == 1:
                    vol_proc, *_ = remove_outside_brain(
                        volume, brain_mask, tolerance_pct=bg_tolerance_pct
                    )
                    brain_only = (vol_proc * brain_mask).astype(volume.dtype)
                elif bg_mode == 2:
                    vol_proc, *_ = remove_global(
                        volume, brain_mask, tolerance_pct=bg_tolerance_pct
                    )
                    brain_only = (vol_proc * brain_mask).astype(volume.dtype)
                elif bg_mode == 3:
                    brain_only, _ = fill_outside_brain_random(
                        volume, brain_mask
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
            if thread.is_alive():
                return

            timer.stop()

            if "error" in result:
                self._status(f"ERROR: {result['error']}")
                self._run_btn.setEnabled(True)
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

            print(f"{'='*70}")
            print("SKIN-REMOVER COMPLETE")
            print(f"{'='*70}\n")

        timer.timeout.connect(_poll)
        timer.start(500)

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
            lyr.data = result["labels"]
            n = int(result["labels"].max())
            sort_label = self._sort_combo.currentText()
            rev_str    = " (reversed)" if reverse else ""
            self._resort_status_lbl.setText(
                f"Done — {n} labels, sorted by {sort_label}{rev_str}."
            )
            self._resort_btn.setEnabled(True)

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

    def _on_split_label(self):
        lyr = self._active_labels_layer()
        if lyr is None:
            self._split_status_lbl.setText("No Labels layer selected.")
            return

        target_label = self._split_label_spin.value()
        n_splits     = self._split_n_spin.value()
        sigma        = self._split_sigma_slider.value()
        min_dist     = self._split_dist_slider.value()

        self._split_btn.setEnabled(False)
        self._split_status_lbl.setText("Splitting…")

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
            lyr.data  = result["labels"]
            new_ids   = result["new_ids"]
            n_total   = int(result["labels"].max())
            all_ids   = [target_label] + new_ids
            self._split_status_lbl.setText(
                f"Done — {n_splits} parts: {all_ids}. Total labels: {n_total}."
            )
            self._split_btn.setEnabled(True)

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
            # Persist model + URL but NOT the API key for security
            self._save_cfg(api_model=mo, api_url=url)

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
            # Filter to selected columns; label is always kept.
            # Group keys (bbox_vox, bbox_um) expand to their constituent columns.
            selected = {"label"}
            for k, cb in self._col_checkboxes.items():
                if cb.isChecked():
                    selected.update(_COL_GROUPS.get(k, [k]))
            df = df[[c for c in df.columns if c in selected]]
            df.to_csv(str(out_csv), index=False)
            self._stats_status_lbl.setText(
                f"Done — {len(df)} labels. Saved: {out_csv.name}"
            )
            print(f"Statistics saved: {out_csv}")
            self._stats_btn.setEnabled(True)

        timer.timeout.connect(_poll)
        timer.start(500)

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

        sigma_xy   = self._sxy_slider.value()
        sigma_z    = self._sz_slider.value()
        min_volume = self._area_slider.value()
        stem            = target.name
        scale           = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1., 1., 1.)

        self._labels_btn.setEnabled(False)
        self._labels_status_lbl.setText("Running...")

        print(f"\n{'='*70}")
        print(f"CREATE LABELS — {stem}  shape={volume.shape}")
        print(f"σ_xy={sigma_xy}  σ_z={sigma_z}  min_volume={min_volume} vox")
        print(f"{'='*70}")

        result = {}

        def _worker():
            try:
                labels = create_labels(
                    volume,
                    sigma_xy=sigma_xy,
                    sigma_z=sigma_z,
                    min_volume=min_volume,
                )
                result["labels"] = labels
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
        if not model_path or not Path(model_path).exists():
            self._cp_status_lbl.setText("ERROR: no Cellpose-SAM model selected — browse to a checkpoint.")
            return

        cellprob      = self._cp_cellprob_slider.value()
        flow          = self._cp_flow_slider.value()
        max_gap       = self._cp_maxgap_slider.value()
        min_contact   = self._cp_mincontact_slider.value()
        large_contact = self._cp_largecontact_slider.value()

        stem  = target.name
        scale = tuple(float(v) for v in target.scale) if len(target.scale) == 3 else (1.0, 1.0, 1.0)
        z, y, x = scale
        xy = (x + y) / 2.0
        anisotropy = z / xy if xy > 0 else 5.747

        gpu = torch.cuda.is_available()

        self._cp_run_btn.setEnabled(False)
        self._cp_status_lbl.setText(f"Starting (device={'cuda' if gpu else 'cpu'})...")

        print(f"\n{'='*70}")
        print(f"CELLPOSE-SAM SEGMENTATION — {stem}  shape={volume.shape}")
        print(f"Model        : {Path(model_path).name}")
        print(f"Cellprob     : {cellprob}   Flow: {flow}   Anisotropy: {anisotropy:.3f}")
        print(f"Safe merge   : max_gap={max_gap}  min_contact={min_contact}")
        print(f"Large contact: {large_contact}")
        print(f"{'='*70}")

        result = {}

        def _worker():
            try:
                def _progress(msg):
                    result["_progress"] = msg
                labels, stats = _run_cellpose_pipeline(
                    volume, model_path,
                    cellprob=cellprob, flow=flow, anisotropy=anisotropy,
                    max_gap=max_gap, min_contact=min_contact, large_contact=large_contact,
                    gpu=gpu, progress_cb=_progress,
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
            if thread.is_alive():
                if "_progress" in result:
                    self._cp_status_lbl.setText(result["_progress"])
                return
            timer2.stop()

            if "error" in result:
                self._cp_status_lbl.setText(f"ERROR: {result['error']}")
                self._cp_run_btn.setEnabled(True)
                return

            labels = result["labels"]
            stats  = result["stats"]
            lname  = f"{stem}_cellpose_labels"

            if lname in self._viewer.layers:
                self._viewer.layers.remove(lname)
            self._viewer.add_labels(labels, name=lname, scale=scale)

            self._cp_status_lbl.setText(
                f"Done — {stats['n_final']} cells "
                f"(raw={stats['n_raw']} -> gmm={stats['n_after_gmm']} -> "
                f"safe={stats['n_after_safe_merge']} -> large={stats['n_after_large_contact']})."
            )
            self._cp_run_btn.setEnabled(True)

            print(f"{'='*70}")
            print(f"CELLPOSE-SAM SEGMENTATION COMPLETE — {stats['n_final']} cells")
            print(f"  raw={stats['n_raw']}  after_gmm={stats['n_after_gmm']}"
                  f"  after_safe_merge={stats['n_after_safe_merge']}"
                  f"  after_large_contact={stats['n_after_large_contact']}")
            print(f"{'='*70}\n")

        timer2.timeout.connect(_poll2)
        timer2.start(1000)  # 1s poll — this run is long, no need for finer granularity
