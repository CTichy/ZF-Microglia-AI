# ZF-Microglia-AI — Software Guide

**A developer/technical reference: every module, every function, what algorithm it implements, and how the pieces connect.** Companion to [GUIDE.md](GUIDE.md) (user-facing, how to operate the plugin) and [STATISTICS_GUIDE.md](STATISTICS_GUIDE.md) (the statistics formulas in detail — this guide summarizes them, that one derives them). Written for someone reading the source for the first time, or citing the pipeline's algorithms in a thesis/paper.

Pseudocode blocks use a plain, language-agnostic notation (not literal Python) — `for`/`if`/`while`, indentation for scope, `←` for assignment where it helps distinguish from equality, and function calls written `name(args)`. Real function/parameter names are used throughout so a reader can go from pseudocode straight to `grep` in the real source.

---

## Table of Contents

1. [Architecture overview](#1-architecture-overview)
2. [Common infrastructure](#2-common-infrastructure) — file I/O, secrets, GPU detection, live-progress capture, generic coarse-to-fine sweeping, `_widget.py`'s own shared helpers
3. [Stage 1 — Skin removal](#3-stage-1--skin-removal) — `_inference.py`, `_background.py`
4. [Stage 2 — Label creation](#4-stage-2--label-creation) — `_labeling.py` (Pixel Classifier), `_cellpose_seg.py` (Cellpose-SAM), `_sanding.py`
5. [Stage 3 — Label editing & correction](#5-stage-3--label-editing--correction) — `_labeling.py` (editing tools), `_grow_correct.py`, `_contrast_sweep.py`, `_auto_correction.py`
6. [Stage 4 — Statistics](#6-stage-4--statistics) — `_statistics.py`
7. [Stage 5 — AI Tools](#7-stage-5--ai-tools) — `_gt_annotation.py`, `_xzyz_patches.py`, `_crop_truncation.py`, `_branch_calibration.py`, `_training_jobs.py`
8. [Stage 6 — Sweeps & Utilities](#8-stage-6--sweeps--utilities) — `_sieve.py`, `_brain_sweep.py`, `_pixel_sweep.py`, `_krendl_sweep.py`, `_epoch_sweep.py`, `_gt_score.py`, `_gt_package.py`, `_gt_toolkit.py`
9. [`_widget.py` — application shell](#9-_widgetpy--application-shell)
10. [Module reference (quick index)](#10-module-reference-quick-index)

---

## 1. Architecture overview

### What the plugin does, as a pipeline

```
raw confocal stack (.tif/.ims)
        │  Stage 1 — Skin Removal (_inference.py, _background.py)
        ▼
brain_only volume (_ExtRm / _NoBG / _RndFill)
        │  Stage 2 — Label Creation (_labeling.py or _cellpose_seg.py)
        ▼
Labels layer (one integer ID per cell)
        │  Stage 3 — Label Editing & Correction (_labeling.py, _grow_correct.py,
        │            _contrast_sweep.py, _auto_correction.py)
        ▼
corrected Labels layer
        │  Stage 4 — Statistics (_statistics.py)
        ▼
per-cell CSV (shape, branching, spatial, intensity, optional LLM description)
```

Two more subsystems sit alongside this straight-line pipeline, not inside it:

- **Stage 5 — AI Tools**: builds and trains the two models Stages 1 and 2 depend on (MONAI for skin removal, Cellpose-SAM for label creation), plus the ground-truth annotation tool that produces the training data in the first place.
- **Stage 6 — Sweeps & Utilities**: GT-verification tools that check a pipeline parameter against real ground truth and (when confirmed) recalibrate it — this is how the "recommended" defaults quoted throughout GUIDE.md were actually found, not guessed.

### Where state lives

There is no persistent application database. Three places hold state:

1. **napari's own layer list** (`self._viewer.layers`) — the volumes and label arrays themselves, as `napari.layers.Image`/`napari.layers.Labels` objects. Every pipeline stage reads its input from a layer and writes its output as a new (or in-place-mutated) layer. This is the plugin's real "shared memory" between stages.
2. **`~/.config/napari-zf-microglia-ai/config.json`** (`self._state["config"]`, read/written via `_load_config()`/`_save_cfg()` in `_widget.py`) — every slider/checkbox default, every "recommended" GT-swept value (see `_update_gt_history()`), file-browse paths, and email/API settings (except the secrets themselves — see `_secrets.py` in §2).
3. **The OS credential store or a local encrypted file** (`_secrets.py`) — the two values deliberately kept out of `config.json`: the SMTP app password and the LLM API key.

### The threading pattern used by almost every "Run"/"Sweep" button

Every long operation in this plugin — MONAI inference, Cellpose-SAM's `do_3D`, a GT sweep, a training launch — follows the same shape, to keep napari's own Qt event loop responsive:

```
def _on_run_button_click():
    validate inputs; if invalid → show error, return
    disable the button; set status text to "Running..."
    result ← {}                              # plain dict, mutated by the worker
    def _worker():
        try:
            result["value"] ← do_the_actual_work(...)   # runs on a background thread
        except Exception as exc:
            result["error"] ← str(exc)
    thread ← Thread(target=_worker); thread.start()

    timer ← QTimer()                          # polls on the GUI thread
    def _poll():
        if thread.is_alive():
            if "_progress" in result: update status text
            return                            # check again next tick
        timer.stop()
        if "error" in result: show it; re-enable button; return
        apply result["value"] to the relevant napari layer(s)   # GUI-thread-only work
        re-enable button
    timer.timeout.connect(_poll); timer.start(200)   # re-check every 200 ms
```

`result` is the hand-off point between the background thread (which must never touch Qt widgets or `viewer.layers` directly — napari/Qt are not thread-safe) and the GUI thread (which owns every widget and layer mutation). A `progress_cb` callback threaded through the actual algorithm lets the worker report intermediate status without touching the GUI itself — it just writes into `result["_progress"]`, which the next poll tick picks up.

Two further patterns build on this:

- **Detached processes** (`_training_jobs.py`, Stage 5): a training run is a genuinely separate OS process (`conda run ...`), not a background thread inside napari — it survives napari closing entirely. The same poll-a-shared-state loop just polls a PID + log file on disk instead of an in-process `Thread`.
- **The hang watchdog** (`_hang_watch_start`/`_hang_watch_stop` in `_widget.py`, §9): wraps the thread-start of the slowest operations (Protect Skin as Label, Auto-correct) with `faulthandler.dump_traceback_later()`, so a run that's still going after 30 minutes prints every thread's real stack to the terminal — turning "is it stuck?" into "here's exactly which line it's on."

### Layer-selection philosophy

Early in this project every editing tool read its input from whatever layer happened to be "active" in napari's own layer list (`_active_labels_layer()`: active selection, else topmost Labels layer, else `None`) — implicit, and silently wrong whenever the wrong layer happened to be selected. Every tool in Stage 3 (Edit MG Labels) now instead reads from one **explicit, shared combo-box selector** (`_edit_labels_layer()`, `self._edit_signal_combo`/`_edit_labels_combo`/`_edit_mask_combo`) that the user sets once, not per tool — see §9.

---

## 2. Common infrastructure

Modules and helpers used across more than one pipeline stage.

### `_io.py` — file loading with metadata-aware voxel scale

Every downstream algorithm that measures a physical distance (µm) needs the true voxel size (Z spacing vs. XY pixel size — these stacks are anisotropic, typically ~5.75:1). This module's only job is finding that scale and loading the raw pixel data consistently for `.tif`/`.tiff`/`.ims`.

**`_calc_anisotropy(voxel_z_um, voxel_x_um, voxel_y_um)`**
```
xy ← (voxel_x_um + voxel_y_um) / 2
return voxel_z_um / xy   (or 1.0 if xy ≤ 0)
```

**`extract_tif_metadata(tif_path)`** — reads ImageJ-embedded metadata from a TIFF's own tags (`spacing`/`finterval` for Z, `XResolution`/`YResolution` for XY, which ImageJ stores as *pixels per µm*, hence the reciprocal). Returns `None` on any failure so the caller can fall through to the next metadata source.

**`parse_metadata(metadata_path)`** — parses a Leica-format `*_metadata.txt` sidecar file with regexes for camera pixel width/height, Z step size, and every `TotalConsolidatedOpticalMagnification` value present (there are usually two: the objective's, ≥10×, and a digital zoom factor, <10×):
```
cam_x, cam_y ← regex-extracted "Pixel Width/Height (µm)"
voxel_z      ← regex-extracted "StepSize"
mags         ← every "TotalConsolidatedOpticalMagnification" value found
obj_mag      ← first mag ≥ 10        (the objective)
zoom_mag     ← first mag < 10        (digital zoom)
total_mag    ← obj_mag × zoom_mag
voxel_x, voxel_y ← cam_x / total_mag, cam_y / total_mag
```
Returns `None` if any of the four regex groups is missing.

**`find_best_metadata_match(file_path)`** — a stack's own metadata file isn't always named exactly `<stem>_metadata.txt`; this looks for an exact match first, then the closest name (`difflib.SequenceMatcher` string-similarity ratio, threshold 0.3) among every `*_metadata.txt` in the same folder, then repeats one directory up if nothing was found (metadata sometimes lives one level above the stack).

**`load_file(path)`** — the public entry point:
```
if suffix is .tif/.tiff:
    raw ← tifffile.imread(path)
    metadata ← find_best_metadata_match(path) → parse_metadata(...)
               or extract_tif_metadata(path)
               or the hardcoded default (1,1,1) µm/vox
    if raw is 3D:  channels ← [(raw, stem, metadata)]
    if raw is 4D:  channels ← [(raw[c], f"{stem}_ch{c}", metadata) for each channel c]
elif suffix is .ims:
    f ← ImsReader(path)
    metadata ← find_best_metadata_match(...) → parse_metadata(...)
               or f's own embedded resolution
    channels ← [(f.get_Volume_At_Specific_Resolution(...), name, metadata)
                for each channel]
return channels    # list of (volume, layer_name, metadata_dict)
```
Every channel is returned separately so the caller (`_reader.py`, or `_widget.py`'s own Open-file handler) can add each as its own napari `Image` layer, letting the user pick which channel is the actual microglia signal.

### `_reader.py` — napari File-menu integration

Registers this plugin as a napari reader (`nape2`/`napari.yaml` "readers" contribution) so `.tif`/`.ims` files opened via napari's own File → Open work correctly, not just via this plugin's own Open button.

```
get_reader(path):
    if path's suffix in {.tif, .tiff, .ims}: return _read_file
    else: return None                          # tells napari "not mine"

_read_file(path):
    channels ← _io.load_file(path)
    for each (volume, name, metadata) in channels, with index i:
        colormap ← _CH_COLORMAPS[i mod 4]       # gray, green, magenta, cyan, repeating
        yield (volume, {name, colormap, scale: metadata["scale"]}, "image")
```

### `_secrets.py` — layered credential storage

Two values need to persist between sessions without ever sitting in the plugin's own plaintext `config.json`: the SMTP app password (email notifications) and the LLM API key (Statistics' description backend). Two-tier storage:

```
get_secret(key):
    try: value ← keyring.get_password(SERVICE_NAME, key)   # Tier 1: OS credential store
         if value: return value
    except: pass
    return _fallback_get(key)                              # Tier 2: local Fernet-encrypted file

set_secret(key, value):
    try:
        keyring.set_password/delete_password(...)           # Tier 1
        _fallback_set(key, "")     # clear any stale Tier-2 copy now that Tier 1 works
        return None                # success, nothing to report
    except Exception as tier1_exc:
        try:
            _fallback_set(key, value)                        # Tier 2
            return None (but print a warning naming the weaker guarantee)
        except Exception as tier2_exc:
            return an error string (both tiers failed -- caller shows it, doesn't crash)
```

`_get_fernet()` generates a machine-local AES key on first use (`Fernet.generate_key()`), stores it `chmod 600` next to the encrypted blob — a real but deliberately weaker guarantee than Tier 1 (the key lives beside what it protects), kept because Tier 1 (`keyring`'s Secret Service backend) reliably fails on SSH-only Linux sessions with no PAM-driven keyring unlock, confirmed directly on this project's own workstation.

`migrate_plaintext_secrets(cfg)` — a one-time upgrade path: if an old `config.json` still has `api_key`/`notify_smtp_password` in plaintext (from before this module existed), move them into `set_secret()` and strip them from the dict that gets written back.

### `_gpu_check.py` — GPU/VRAM classification

Checked once at import time (hardware doesn't change mid-session) and cached into module-level constants (`GPU_HAS_CUDA`, `GPU_VRAM_GB`, `GPU_MEETS_RECOMMENDED`, `GPU_NAME`, `GPU_MSG`):
```
check_gpu():
    try:
        if not torch.cuda.is_available(): return {has_cuda: False, ...}
        props ← torch.cuda.get_device_properties(0)
        vram_gb ← props.total_memory / 1024³
        meets ← vram_gb ≥ 8
        return {has_cuda: True, vram_gb, meets_recommended: meets, name: props.name}
    except: return {has_cuda: False, ...}     # never raises -- worst case, a cautious banner
```
Informational only — Tab 5 (AI Tools) is always shown regardless of the result; this only decides which color/text its disclaimer banner uses.

### `_live_progress.py` — surfacing libraries' own hidden progress output

MONAI's `sliding_window_inference(progress=True)` draws a `tqdm` bar to raw `sys.stderr`; Cellpose routes everything through Python's `logging` module but never calls its own `logger_setup()` unless its CLI is used, so by default its log records go nowhere — not even a terminal. `capture_live_output(push)` is a context manager that makes both visible in the GUI for the duration of one call:
```
capture_live_output(push):
    if another capture is already active: yield (no-op passthrough); return
    save real sys.stdout/sys.stderr and the root logger's level
    sys.stdout ← sys.stderr ← a _LineSink(push)   # buffers partial writes,
                                                    # pushes a complete line on \n OR \r
                                                    # (tqdm redraws with \r; GUI can't
                                                    #  overwrite in place, so \r → new line)
    attach a logging.Handler to the root logger that calls push(formatted record)
    set root logger level to INFO if it was less verbose
    yield                                          # caller's real work runs here
    finally: restore stdout/stderr, remove the handler, restore the log level
```
`push` must be thread-safe (the caller always passes the same `result["_progress"] = msg` pattern every other worker uses).

### `_sieve.py` — generic coarse-to-fine sweep narrowing

Shared by every GT-sweep tool that has one continuous "primary axis" (MONAI Threshold, BG Threshold, Smooth σ XY, Cellprob) — born from a real manual workflow (sweep a wide range coarsely, narrow around the winner, narrow again) generalized into one reusable engine that knows nothing about what it's sweeping.

```
sieve_grid(lo, hi, center, half_width, step):
    stage_lo ← max(lo, center - half_width)
    stage_hi ← min(hi, center + half_width)
    if stage_hi ≤ stage_lo: return [clip(center, lo, hi)]     # degenerate range → 1 point
    return every value from stage_lo to stage_hi, step apart

run_sieve(run_stage_fn, lo, hi, coarse_step, refine_steps, progress_cb, cancel_check):
    stage_specs ← [(None, None)] + refine_steps    # stage 1 = full range, no narrowing
    values ← grid from lo to hi, step = coarse_step
    for each (half_width, step) in stage_specs, with index i:
        if cancel_check(): mark cancelled; break
        if i > 0: values ← sieve_grid(lo, hi, best_value, half_width, step)
        best_value, stage_result ← run_stage_fn(values)   # caller runs ONE 1D grid,
                                                             # everything else held fixed
        record this stage (values, best_value, stage_result)
        if stage_result says cancelled: mark cancelled; break
    return {stages: [...], final_best_value, final_result, cancelled}
```
`run_stage_fn` is the only thing that actually knows about MONAI/Cellpose/etc. — it's a closure the caller builds, e.g. "run the BG-Threshold sweep at these candidate values, holding Erosion fixed at its current slider value, return the winning threshold and the sweep's own report dict."

### `_widget.py`'s own shared helpers

A handful of small functions used throughout `_build_ui()` (the ~10,000-line method that constructs every tab — see §9), not tied to any one pipeline stage:

- **`_add_reliable_spinbox(row_layout, slider, min, max, step, decimals=None)`** — every numeric slider in this plugin is paired with a real `QSpinBox`/`QDoubleSpinBox` synced bidirectionally (`slider.valueChanged → spin.setValue`, `spin.valueChanged → slider.setValue`), replacing `superqt`'s own built-in numeric label whose width reset unpredictably on first show. Returns the spinbox.
- **`_set_layout_widgets_visible(layout, visible)`** — recursively walks a `QVBoxLayout`/`QHBoxLayout` tree (rows mix `addWidget` and nested `addLayout` calls) and calls `setVisible()` on every real widget found, since only widgets (not layouts) support it directly.
- **`_make_collapsible(groupbox, start_expanded=True)`** — turns any already-built `QGroupBox` into a collapsible section: makes its own title bar checkable, and toggling it calls `_set_layout_widgets_visible` on the group's contents. Every group box in every tab goes through this.
- **`_wrap_scroll(widget)`** — wraps a fully-built tab page in a vertical-only `QScrollArea` (`setWidgetResizable(True)`, horizontal scrollbar forced off) so a tab taller than the napari window stays fully reachable.
- **`_add_recommended_label(layout, initial, unit, noun)`** — the read-only "Recommended: X" line under a GT-sweepable slider, kept separate from the slider's own editable value so a GT-confirmed sweep's finding is never silently overwritten by the user nudging the slider to test something else.
- **`_add_gt_checkbox(layout, hint, visible=True)`** — the "This is verified ground truth" one-shot opt-in every sweep tool has: off by default, un-checks itself after every run, and is the only thing that lets a sweep move a shared recommended value (Min volume, a slider's own "Recommended" line, etc.) rather than just reporting a result.
- **`_make_notify_checkbox()`** — builds one "Email me when done" `QCheckBox` with a shared tooltip; every long-running, non-detached tool (Run Skin-Remover, Cellpose-SAM Segmentation, the Cellprob/Best-Epoch sweeps) gets its own instance of this, all reading the same shared SMTP credentials configured once in Tab 6.

---

## 3. Stage 1 — Skin removal

**UI**: Tab 1 — Skin Remover. **Modules**: `_inference.py` (MONAI brain segmentation), `_background.py` (three background-removal modes). **Goal**: given a raw confocal stack, produce `brain_only` — everything but the brain zeroed — for Stage 2 to label.

### `_inference.py` — MONAI 3D U-Net sliding-window inference

**`_normalize(volume)`** — percentile normalization, the same transform the model was trained under:
```
p01, p99 ← 1st and 99th percentile of volume
return clip((volume - p01) / max(p99 - p01, 1), 0, 1)
```

**`predict_probability(volume, model_path, device)`** — the expensive half of inference, split out on its own so a GT-sweep tool can run it once and cheaply re-threshold many candidate values afterward (see §8):
```
model ← UNet(3D, 1→1 channel, channels=(32,64,128,256,512), 2 res-units/level)
        load weights from model_path checkpoint; eval mode
volume_t ← normalize(volume) as a (1,1,Z,Y,X) tensor on device
pred_logits ← sliding_window_inference(volume_t, roi=(64,192,192),
                  batch=4, overlap=0.25, mode="gaussian", predictor=model)
              # slides a 64×192×192 window across the whole volume, blending
              # overlapping predictions with a Gaussian-weighted average
              # (softer seams than a hard-edge tile stitch)
return sigmoid(pred_logits)     # per-voxel probability, 0..1
```

**`postprocess_probability(pred_prob, threshold)`** — the cheap half, safe to call repeatedly per candidate threshold:
```
raw_mask ← pred_prob > threshold
labeled, n_components ← connected_components(raw_mask)
clean ← the single largest component            # drops isolated blobs / mislabeled tissue
clean ← fill_holes(clean)                         # solid, contiguous brain
return clean as uint8   # "brain_mask", always un-eroded
```

**`run_inference(volume, model_path, threshold, device, erosion_voxels)`** — glues the two together and applies the optional erosion:
```
pred_prob ← predict_probability(...)
brain_mask ← postprocess_probability(pred_prob, threshold)     # always un-eroded — this is
                                                                  # what gets saved as brain_mask.tif
if erosion_voxels > 0:
    eroded_mask ← binary_erosion(brain_mask, iterations=erosion_voxels)
else:
    eroded_mask ← brain_mask
brain_only ← volume × eroded_mask       # zero everything the (possibly eroded) mask excludes
return brain_mask, brain_only, eroded_mask
```
`eroded_mask` is the one downstream background-removal steps must use as "the brain boundary" — using `brain_mask` there instead would silently discard the Erosion slider (a real bug fixed earlier in this project's history).

### `_background.py` — background estimation and removal, 3 modes

All three modes share one background estimate and one threshold rule:

**`_estimate_background(volume, brain_mask)`**:
```
brain_pixels ← volume[brain_mask]                    # only INSIDE the (possibly eroded) brain
hist ← 1000-bin histogram of brain_pixels
bg_mode ← the histogram bin center with the highest count   # the population mode, robust
                                                                # to a skewed intensity tail
bg_values ← every brain_pixel ≤ bg_mode
return bg_values, bg_mode, bg_mode, bg_mode          # (median/min/max all collapse to bg_mode
                                                        #  in the current implementation)
```

**`_threshold(volume, brain_mask, tolerance_pct)`**:
```
bg_max ← _estimate_background(...)'s bg_mode
tol ← (max(volume) - min(volume)) × tolerance_pct / 100
threshold ← bg_max + tol
bg_mask ← volume ≤ threshold             # single-sided: everything at/below threshold = background
return bg_values, bg_max, threshold, bg_mask
```

**Mode 1 — `remove_outside_brain`** (`_ExtRm` suffix): zero background pixels *outside* the brain only; everything inside the brain mask is kept untouched regardless of intensity.
```
bg_mask ← _threshold(...)
to_zero ← (~brain_mask) AND bg_mask
result ← volume.copy(); result[to_zero] ← 0
```
*(Note: as actually wired in `_widget.py`'s `_on_run`, Mode 1's saved `brain_only` is `volume × eroded_mask` directly — the entire exterior zeroed, not `remove_outside_brain`'s more selective "background-only" result — see §9.)*

**Mode 2 — `remove_global`** (`_NoBG` suffix): zero every background-threshold pixel in the *whole* stack, inside the brain included — this is the input the Pixel Classifier route needs (background must be zero everywhere for its own threshold/union-find pipeline to find clean isolated blobs).
```
bg_mask ← _threshold(...)
if signal_erosion_voxels > 0:
    signal_mask ← erode_signal_2d(~bg_mask, signal_erosion_voxels)   # shrink what SURVIVES
    bg_mask ← ~signal_mask                                            # thresholding, not the mask
result ← volume.copy(); result[bg_mask] ← 0
```
**`erode_signal_2d(mask, iterations)`** erodes one Z-slice at a time (`binary_erosion` per 2D slice), not a single 3D erosion — a plain 3D erosion also eats `iterations` slices off the top *and* bottom of a blob's own Z-extent, which for a thin blob can delete it outright; per-slice erosion only ever shrinks each slice's in-plane contour.

**Mode 3 — `fill_outside_brain_random`** (`_RndFill` suffix, presentation-only, not fed into any labeling route):
```
bg_mode ← _estimate_background(...)
tol ← 0.001 × (max(volume) - min(volume))          # ±0.10% fill window around the mode
result ← volume.copy()
result[outside brain] ← uniform_random(bg_mode - tol, bg_mode + tol), one value per outside voxel
```

### `_widget.py` — Tab 1 handlers

**`_on_open()`** — background-thread wrapper around `_io.load_file(path)`; on completion, adds every returned channel as its own `Image` layer (`_add_channels`) and reports channel count + shape.

**`_on_load_labels()`** — loads a saved `.tif` directly as a `Labels` layer: reads with `tifffile.imread(...).astype(int32)`, scales it to match whatever stack is already open (`self._state["metadata"]["scale"]`, warning if none is open yet), and names it `<active Image layer>_labels` — the same convention Stage 2's own label-creation tools use, so a reloaded file fits the rest of the plugin's naming scheme.

**`_on_run()`** — the main button:
```
validate: model file exists, an Image layer is selected and 3D
read Threshold/Erosion/Background-mode/BG-Threshold/Signal-Erosion from their sliders
device ← cuda, else mps, else cpu
background thread, wrapped in capture_live_output (surfaces MONAI's tqdm bar +
    _background.py's own print() lines into the GUI's log view, not just a terminal):
    brain_mask, brain_only, eroded_mask ← run_inference(volume, model_path, threshold,
                                                          device, erosion_voxels)
    if bg_mode == 1 (_ExtRm):  brain_only ← volume × eroded_mask
    if bg_mode == 2 (_NoBG):   brain_only ← remove_global(volume, eroded_mask, ...) × eroded_mask
    if bg_mode == 3 (_RndFill): brain_only ← fill_outside_brain_random(volume, eroded_mask)
    # bg_mode == 0 (Off): brain_only stays exactly run_inference()'s own (volume × eroded_mask)
on completion: add brain_mask/brain_only as new Labels/Image layers (suffix per _BG_SUFFIX),
    save the ticked output files, send email notification if configured
```

---

## 4. Stage 2 — Label creation

**UI**: Tab 2 — Create MG Labels. Two independent routes, auto-shown/hidden by the active layer's background-mode suffix (`_NoBG` → Pixel Classifier, `_ExtRm` → Cellpose-SAM): **Pixel Classifier** (`_labeling.py`'s `create_labels`, a plain 3D connected-components engine) and **Cellpose-SAM** (`_cellpose_seg.py`'s `run_full_pipeline`, a trained-network `do_3D` segmentation + several correction stages). Both finish with an optional **Sanding** pass (`_sanding.py`). **Goal**: turn `brain_only` into a `Labels` layer, one integer ID per cell.

### `_labeling.py` — Pixel Classifier: true 3D connected components

Despite older project history calling this a "union-find" engine, the current implementation is a plain 3D connected-components labeling, not a per-slice-then-stitch union-find — `create_labels()` dispatches to whichever of 3 backends is fastest (detected once at import time: **CUDA** via CuPy/cupyx > **Apple MPS** → falls back to threaded CPU, since MPS itself has no `ndimage` ops > **CPU**, `scipy.ndimage` + a `ThreadPoolExecutor` at 50% of cores). All three backends implement the identical 6-step algorithm; only the array library and slice-loop parallelism strategy differ.

**`create_labels(volume, sigma_xy, sigma_z, min_volume, min_hole_size, final_min_fraction)`**:
```
1. binary ← volume > 0
2. blurred ← gaussian_filter(binary, sigma=(sigma_z, sigma_xy, sigma_xy))
   smooth_mask ← blurred > 0.5          # re-thresholded smooth binary mask —
                                          # this is what actually connects/
                                          # disconnects nearby blobs, not the
                                          # raw thresholded signal
3. for each Z slice:
       fill any enclosed background hole whose area < min_hole_size
       (min_hole_size <= 0 → fill every hole unconditionally, the
        original behaviour)
4. labeled, n_objects ← connected_components(smooth_mask, structure=ones(3,3,3))
                          # 26-connectivity: any of the 26 neighbors counts
5. threshold ← max(1, round(final_min_fraction × min_volume))
                          # NOT min_volume itself — same golden-ratio
                          # safety-net relaxation _cellpose_seg.py's
                          # final_min_size_cleanup() uses, since this route
                          # has no later merge/reattach stage to give a
                          # gray-zone object a second chance
   remove every blob smaller than threshold
6. renumber surviving blobs 1..N by DESCENDING volume (label 1 = largest)
return int32 labels, 0=background
```

The CUDA path (`_create_labels_cuda`) runs steps 1–2 and the volume/threshold math as vectorized CuPy array ops (`cp.bincount` for per-label voxel counts instead of a Python loop), loops step 3 per Z slice on the GPU, and frees the CuPy + PyTorch memory pools (`_free_gpu_cache()`) before returning — including on the exception path, so a mid-run CUDA OOM still falls back to CPU cleanly (`create_labels()`'s own `try/except` around the CUDA call). The threaded-CPU path (`_create_labels_threaded`, also used for Apple MPS) runs step 3's per-slice hole fill via a `ThreadPoolExecutor`, everything else as a single `scipy.ndimage`/`numpy` call over the whole volume.

**`min_hole_size` mechanism** (`_fill_holes_capped_gpu` / the CPU per-slice closure): unconditional `binary_fill_holes` fills every enclosed background region no matter its size — much like Cellpose's own default hole-filling (see §4's Cellpose-SAM section), a single-voxel prediction artifact and a genuine internal structural void look identical to it. A positive `min_hole_size` switches to an area-limited fill (`skimage.morphology.remove_small_holes` on CPU; a manual `label()` + `bincount()`-based size filter on GPU, since cupyx has no equivalent), leaving any hole at or above the floor standing as real background.

### `_cellpose_seg.py` — Cellpose-SAM: `do_3D` + 4-stage cleanup pipeline

Same math as the project's earlier standalone `krendl_do3d.py` CLI script, refactored into importable functions. No GT-based relabeling/scoring here (that stays a CLI/research workflow, see §8) — this produces a clean, sequentially-labeled instance mask ready for manual correction.

**`predict_flows(volume, model_path, anisotropy, gpu)`** — the one genuinely GPU-bound, expensive step: a single Cellpose-SAM network forward pass producing a per-voxel flow field (`dP`) and cell-probability map (`cellprob`), with neither `cellprob_threshold` nor `flow_threshold` applied yet — both are cheap, CPU/light-GPU post-processing steps applied afterward. Split out on its own (mirroring `_inference.py`'s `predict_probability`/`postprocess_probability` split for MONAI) so a cellprob sweep can call this once and cheaply re-threshold many candidate values without repeating the expensive network pass:
```
model ← CellposeModel(pretrained_model=model_path, gpu=gpu)
_, flows, _ ← model.eval(volume, do_3D=True, anisotropy=anisotropy, z_axis=0,
                          channel_axis=None, diameter=None, normalize=True,
                          augment=False, compute_masks=False)  # compute_masks=False
                                                                 # skips mask formation
dP, cellprob ← flows[1], flows[2]
return model, dP, cellprob, volume.shape
```

**`save_flow_cache` / `load_flow_cache`** — persist `predict_flows()`'s expensive output to disk (`np.savez_compressed`) plus a fingerprint (model checkpoint path, anisotropy, volume shape/dtype). Exists because mask formation (`masks_from_flows`'s own internal `follow_flows()` re-run) needs its own separate chunk of GPU memory and can CUDA-OOM *even after* a multi-hour network pass already succeeded — without a cache, that OOM silently destroys the entire network pass (Python's own except-block variable cleanup, confirmed by reading `_widget.py`'s own handler, keeps only `str(exc)`, nothing else). `run_full_pipeline()`'s `flow_cache_path` saves the cache right before mask formation starts and deletes it automatically the moment mask formation succeeds — it only survives on disk if that specific step crashes, and a stale/mismatched cache (different fish/model/anisotropy) is detected via the fingerprint and ignored, never silently reused.

**`masks_from_flows(model, dP, cellprob, shape, cellprob_threshold, flow_threshold, min_size, max_size_fraction, niter, min_hole_size, progress_cb)`** — cheap: forms instance masks from an already-computed flow field. `do_3D=True` is baked in. `flow_threshold` is accepted only to match `do_3D`'s call signature — Cellpose's own `compute_masks()` only applies its flow-error QC filter when `do_3D=False` (confirmed directly in `cellpose/dynamics.py`; a call-count spy test found 0 calls under `do_3D=True`), so it's a documented no-op here. `niter=None` is resolved to Cellpose's own default of 200 (calling `_compute_masks()` directly, as this function does, skips `eval()`'s own resolution of that default, which would otherwise crash with a `TypeError` on `range(None)`).
```
if niter is None: niter ← 200
report progress (no further progress prints until this whole step finishes —
                  neither this function nor Cellpose's own follow_flows()/
                  compute_masks() log anything while it runs)
monkey-patch cellpose.utils.fill_holes_and_remove_small_masks →
    _make_capped_fill_holes(min_hole_size)   # see below
try:
    masks ← model._compute_masks(shape, dP, cellprob, flow_threshold, cellprob_threshold,
                                  min_size, max_size_fraction, niter, do_3D=True)
finally:
    restore the original fill_holes_and_remove_small_masks
return masks as int32
```

**`_make_capped_fill_holes(min_hole_size)`** — builds a drop-in, size-aware replacement for Cellpose's own hole-filling (which unconditionally calls `fill_voids.fill()` over each predicted mask's full 3D crop, no size threshold at all — the same category of bug `create_labels()`'s own `min_hole_size` was built to fix). `min_hole_size<=0` keeps Cellpose's exact original unconditional behavior. A positive value switches to a **per-Z-slice** `skimage.morphology.remove_small_holes` loop, not one 3D call over the whole crop — a real per-slice-small hole that persists across many Z-slices is one 3D-connected void whose *total* volume can dwarf the threshold, so a plain 3D `remove_small_holes()` judges it "too big to be noise" and leaves a visible ring on every slice it spans; looping per-slice avoids that (mirrors `_labeling.py`'s own per-slice treatment).

**`gmm_cleanup(masks)`** — 3-component Gaussian Mixture Model on `log1p(volume)` of every raw predicted object, separating a noise / gray-zone / real-cell population and dropping everything below the auto-detected gray→cell intersection point:
```
if fewer than 3 objects: return unchanged (not enough data to fit 3 components)
x ← log1p(each object's voxel volume)
gmm ← GaussianMixture(n_components=3).fit(x)
sort the 3 fitted components by mean: small, mid, large
cutoff ← the point where the mid and large components' fitted Gaussian
          curves cross (closed-form solution of two Gaussians' log-density
          equality; falls back to the midpoint of the two means if the
          quadratic has no real root in range)
remove every object smaller than cutoff
return masks, cutoff, n_removed
```

**`krendl_safe_merge(masks, max_gap, min_contact, gt_min, scale_zyx)`** — merges only genuinely small fragments (< `gt_min`, the smallest true GT-labeled cell volume ever confirmed) into their nearest larger neighbor, when either close enough or touching with enough contact area:
```
repeat up to 200 passes:
    candidates ← every fragment below gt_min, smallest first
    for each candidate fragment f:
        find the nearest larger-volume object t (bbox-proximity-prefiltered
        in voxel space, using max_gap converted to a voxel margin via the
        FINEST axis — safe over-inclusion, never wrongly rejects a close pair)
        gap ← min distance from f's mask to t's mask, via a physically-scaled
              EDT (distance_transform_edt(~f_mask, sampling=scale_zyx)) —
              NOT plain voxel-index distance: this project's voxels are
              anisotropic (Z=1.0µm, XY=0.174µm typical), so an unscaled EDT
              would treat a "2 voxel" gap as 2.0µm along Z but only 0.35µm
              in-plane, a ~5.7x orientation-dependent inconsistency (a real
              bug found and fixed in this project's history)
        do_merge ← gap <= max_gap (µm)
        if not do_merge: fall back to contact-area check
            (dilate f by 1 voxel, count overlap with t) >= min_contact
        if do_merge: absorb f into t, update t's cached vol/centroid/bbox
    if nothing merged this pass: stop
return masks, total_merges
```
`min_contact`'s fallback branch is largely dead code in practice at any reasonable `max_gap`: the smallest possible EDT gap between two literally-touching-but-disjoint fragments is exactly one voxel spacing, always ≤ a positive `max_gap` — confirmed both by this structural argument and empirically against real production GT data (see the project's session history).

**`large_contact_merge(masks, large_contact)`** — a separate, later, size-agnostic merge: any two objects (regardless of size) sharing a contact area ≥ `large_contact` voxels get merged, catching blobs split through a thick junction rather than a thin neck. Same repeat-until-stable structure as safe-merge, contact measured the same way (1-voxel dilation + overlap count); the larger of the two objects' ID always survives.

**`final_min_size_cleanup(masks, gt_min, fraction=0.618)`** — the last-resort safety net, run after GMM/safe-merge/large-contact: removes anything still below `fraction × gt_min` (golden ratio by default — strict enough to reject debris, lenient enough not to reject a legitimately smaller-than-average real cell). Nothing upstream is guaranteed to remove every possible debris object (GMM judges by population statistics; safe-merge/large-contact only act when a mergeable neighbor exists), so this stage is a floor under all of them, not a replacement.

**`relabel_sequential(masks)`** — renumbers surviving labels 1..N with no gaps.

**`run_full_pipeline(volume, model_path, cellprob, flow, anisotropy, max_gap, min_contact, large_contact, gt_min, ..., flow_cache_path)`** — glues every step above together:
```
1. dP, cellprob_map, shape ← predict_flows(...) or reuse precomputed_flows /
   a matching on-disk flow_cache_path (see save/load_flow_cache above)
2. masks ← masks_from_flows(...)                              # raw prediction
   raw_masks ← masks.copy()  # kept in stats['raw_masks'] — this is what
                              # _widget.py always auto-saves as <stem>_cp_masks.tif
3. masks, gmm_cutoff, gmm_removed ← gmm_cleanup(masks)
4. masks, safe_merges ← krendl_safe_merge(masks, max_gap, min_contact, gt_min, scale_zyx)
                         # this is what _widget.py always auto-saves as
                         # <stem>_cp_masks_corrected.tif
5. masks, lc_merges ← large_contact_merge(masks, large_contact)
6. masks, final_removed ← final_min_size_cleanup(masks, gt_min, final_min_fraction)
7. masks, n_final ← relabel_sequential(masks)
return masks, stats{n_raw, n_after_gmm, n_after_safe_merge, n_after_large_contact,
                     n_after_final_min_size, n_final, gmm_cutoff_vox, gmm_removed,
                     safe_merges, large_contact_merges, final_min_threshold_vox,
                     final_min_removed, raw_masks}
```

**`rerun_single_cell(volume, labels, label_id, model_path, ..., pad_z=15, pad_xy=40)`** — "fix one mis-segmented cell" without a full-fish re-run: crops to that label's own padded bounding box, re-runs the *entire* `run_full_pipeline()` on just the crop (safe to reuse unmodified — GMM cleanup already no-ops below 3 objects, and the merge/cleanup stages use a fixed `gt_min` floor, not crop-local population statistics), then splices back only the crop-result objects that actually overlap the original label's own footprint — anything else the crop's own `do_3D` pass happens to also detect (e.g. a neighboring cell caught by the padding) is discarded, never duplicated into the full volume. If the crop genuinely reveals more than one real cell where there was one label before, all of them are kept as brand-new label IDs appended after the current max.

### `_sanding.py` — sigma-softening pass, chained after any label-regenerating tool

Runs *after* a label's shape has already been rebuilt (auto-correction, Correct Label, etc.) — purely geometric, no image/intensity involved: each label's own binary mask is Gaussian-blurred and re-thresholded at 0.5, rounding off small jagged/blocky voxel-scale steps without meaningfully reshaping the cell. Same foreign-label protection as every Correct Label tool in this plugin — a neighbor's already-claimed voxels can never be grown into.

**`sanding_pad(sigma_xy, sigma_z)`**: `max(10, round(3 × max(sigma_xy, sigma_z)) + 3)` — enough bbox padding for the blur to round a boundary without being clipped by the crop edge.

**`sand_labels_stack(labels, sigma_xy=0.7, sigma_z=0.7, pad, min_volume, final_min_fraction=0.618, skin_label_id, progress_cb)`**:
```
for each positive label ID in the volume:
    new_labels, info ← sand_label(new_labels, lid, sigma_xy, sigma_z, pad)
                        # per-label Gaussian blur + re-threshold + foreign-
                        # label exclusion (see _labeling.sand_label, §5)
    record applied/skipped (+ reason) per label
if min_volume given:
    threshold ← round(final_min_fraction × min_volume)   # same golden-ratio
                                                            # safety net as
                                                            # every other
                                                            # final stage
    remove_debris(new_labels, threshold, skin_label_id)   # skin's own small
                                                            # fragments swept
                                                            # too, if present
return new_labels, report{sigma_xy, sigma_z, n_cells_total, n_cells_sanded,
                           skipped_cells, n_debris_fragments_removed}
```
Deliberately kept as its own small default (0.7/0.7 voxels), much smaller than the Pixel Classifier's own Smooth σXY/σZ (1.5/3.0) — that pair decides whether raw blobs *merge* into one 3D instance before labels even exist; sanding runs after labels already exist and is foreign-protected (structurally can't merge two cells), so its only job is polishing edges, not reshaping.

### `_widget.py` — Tab 2 handlers

**Route auto-detection**: the Pixel Classifier and Cellpose-SAM sections (plus downstream tools — Resort/Split/Save) show or hide based on whether a `_NoBG` or `_ExtRm`-suffixed layer exists anywhere in the viewer, not just whichever layer is currently active — both can be shown together (e.g. comparing routes on the same fish).

**`_on_create_labels()`** (Pixel Classifier): background-thread wrapper reading Smooth σXY/σZ, Min overlap (historical UI label — no longer used by the current connected-components algorithm, kept for compatibility), Min volume, Min hole size sliders, calling `create_labels()`, adding the result as a new `Labels` layer named `<image>_labels`.

**`_on_run_cellpose_seg()`** (Cellpose-SAM Segmentation): validates the checkpoint file (`_is_valid_cellpose_checkpoint()` — checks it's a zip archive, the format `torch.save()` uses, to catch a wrongly-selected non-checkpoint file before a multi-hour run rather than deep inside `torch.load()`), auto-selects the matching `<stem>_brain_mask` layer if the "Auto-correct" checkbox needs skin protection, then runs `run_full_pipeline()` in a background thread wrapped in `capture_live_output` (surfaces Cellpose's own otherwise-silent flow-following/hole-fill/small-object-removal step into the GUI log). On completion: adds the result as a `Labels` layer, **unconditionally** (no checkbox) writes `<stem>_cp_masks.tif` (`raw_masks`) and `<stem>_cp_masks_corrected.tif` (post-safe-merge) alongside the fish's other outputs, and — if "Auto-correct labels via contrast sweep after segmentation" is ticked — chains into the Stage 3 auto-correction pipeline (see §5) as a further background stage.

**`_on_rerun_single_cell()`**: resolves the correct source Image layer and target Labels layer via `_resolve_rerun_layers()` (selected Labels layer wins if any name; else `<image>_cellpose_labels` → `<image>_labels` → a single same-shape Labels layer; ambiguity is reported, never guessed — fixes a real earlier bug where a labels layer not literally named `<stem>_cellpose_labels` was rejected), then calls `rerun_single_cell()` in a background thread and splices the result back into the live layer.

**Sanding UI**: one shared checkbox + Sigma XY/Z pair (Common Settings) drives sanding at every call site that regenerates a label's shape (Correct Label 2D/3D, Correct Adjacent Labels, and — chained as its own stage — the Cellpose-SAM auto-correct pipeline), not a separate control per tool.

---

## 5. Stage 3 — Label editing & correction

**UI**: Tab 3 — Edit MG Labels, working against a shared explicit Signal/Labels/Brain-mask layer selection (`_edit_labels_layer()`, see §1 "Layer-selection philosophy") plus a Label A / Label B selector pair most tools reuse. **Modules**: `_labeling.py` (the editing primitives), `_grow_correct.py` (auto-grow/until-stable orchestration), `_contrast_sweep.py` (calibrating the intensity threshold these tools use), `_auto_correction.py` (the full unattended per-cell pipeline chained after Cellpose-SAM Segmentation). This is the largest, most-iterated part of the plugin.

### Shared building blocks

Every tool in this section that regenerates a shape from intensity shares one convention: **`candidate = image >= lo`** — a one-sided threshold, not a band. An earlier band version (`lo <= image <= hi`) got this backwards for exactly the narrow contrast window that makes the tool useful: a window like `[100, 101]` chosen to make the display saturate into a clean silhouette excludes the cell's true bright interior (well above `hi`, just shown as saturated white) while pulling in unrelated background pixels that happen to fall inside the narrow band. `hi` (the signal layer's own display ceiling) is shown for context only, never used to restrict the mask.

Every tool also shares **foreign-label protection**: any pixel already claimed by a *different* existing label is excluded from the candidate mask before anything (connected components, watershed) runs — a foreign label can never be grown into, merged into, or even appear in a corrected shape, regardless of how the threshold or connectivity falls.

### `_labeling.py` — editing primitives

**`resort_labels(labels, sort_by, reverse)`**: renumber every label 1..N by a chosen key — `"size"` (voxel count via `bincount`, natural = descending, largest→1), `"centroid_z/y/x"` (via `label_centroids_zyx`, natural = ascending), or `"complexity"` (skeleton branch count via `_statistics._skeleton_stats`, natural = descending, computed per-label in a thread pool). Only positive labels are ever touched by the final LUT remap — a sentinel like skin's `-1` would otherwise corrupt via numpy's negative-index wraparound (`lut[-1]` = last element), a real bug found and fixed in this project's history.

**`label_centroids_zyx(labels, label_ids)`**: bit-identical to `scipy.ndimage.center_of_mass`, but computed from each label's own bounding box (`find_objects` + a per-label crop) instead of one whole-volume call — ~10-100x faster when a large negative sentinel (skin) is present, since scipy's own implementation takes a much slower path for negative-containing arrays.

**`remove_debris(labels, threshold, skin_label_id=None)`**: zeros every *spatially connected fragment* smaller than `threshold` voxels — evaluated per connected component (26-connectivity), not per raw label ID, so several small disconnected fragments still sharing one old ID (e.g. after a manual edit) are each judged on their own size, not their summed total. Does **not** renumber survivors. `skin_label_id`, when given and present, is swept too (skin's own bounding box computed manually since `find_objects()` never indexes negative values). The per-label sweep (`_remove_debris_from_crop`) clears every small fragment of a label in one `bincount`+boolean-index pass, not one array scan per fragment — a real historical bug (per-fragment `crop[cc==id]=0` scanning the whole crop each time) made this "hang" on a fragmented skin label with thousands of small pieces.

**`split_label(labels, target_label, n_splits, sigma, min_distance, mode, z, image)`** — watershed split into N parts:
```
mode="3d": surface = the mask's own EDT (splits at the geometrically
           narrowest neck, regardless of what the underlying signal looks like)
mode="2d": surface = raw signal intensity on slice z (seeds at brightness
           peaks; cut follows the dimmest ridge — for when two things only
           merge on ONE cross-section, where the 3D mask has no real neck
           but the signal already dips)
1. crop to bbox(+pad)
2. Gaussian-smooth the surface (GPU if available)
3. h_maxima(surface, h) — TOPOLOGICAL prominence, not peak_local_max's
   Euclidean-radius search: finds peaks separated by the thinnest saddle
   regardless of how physically close their centers are. Auto-reduces h
   from 50% of max down to 0.5% until >= n_splits peaks are found.
4. within each h_maxima region, seed = its own EDT/intensity maximum;
   drop seeds closer than min_distance to an already-chosen one
5. watershed(-surface, markers=seeds, mask=mask_crop)
6. clear the 1-voxel interface where two parts directly touch
   (outer surface of each part untouched)
```
Part 1 keeps `target_label`; parts 2..N get fresh IDs above the current max.

**`join_labels(labels, label_a, label_b)`**: the inverse — every `label_b` voxel becomes `label_a`. One vectorized boolean assignment, no crop/GPU needed.

**`correct_label_from_intensity(labels, image, label_id, z, lo, hi, pad)`** — Correct Label's 2D engine: within `label_id`'s own padded bbox on slice `z`, `candidate = image >= lo`, foreign-excluded, painted in wholesale (not connectivity-restricted to the label's existing footprint — a human reading the same contrast window would count a visible blob as part of the cell whether or not it happens to already touch). Raises rather than silently emptying the label if nothing survives.

**`correct_label_from_intensity_3d(labels, image, label_id, lo, pad, ..., auto_grow, growth_step, max_iterations, until_stable, max_stability_passes, resolve_adjacent, z_extent_pad)`** — Correct Label's 3D engine, the most complex function in the plugin. NOT a bounding box over the whole cell — loops one Z-slice at a time, each with its own locally re-derived neighborhood:
```
zmid ← z_orig_min + ceil((z_orig_max - z_orig_min) / 2)
walk z from zmid down to z_orig_min, then from zmid+1 up to z_orig_max:
    _or_seed(z):  # before correcting z, OR-union z-1 and z+1's footprints
                  # (within the label's fixed original range) into z's own
                  # current footprint as watershed seed context — an
                  # already-walked neighbor contributes its JUST-corrected
                  # shape, a not-yet-walked one its ORIGINAL shape. OR, not
                  # AND (the source pseudocode's own literal "AND" was a
                  # stated user error — AND would erase real signal any
                  # time a neighbor's shape isn't already identical).
                  # Pasted only into empty background or skin (-1), never
                  # a different real label's territory.
    _correct_slice_full(z):
        if another real label is present in z's own local area:
            jointly re-derive via the same marker-seeded watershed
            Correct Adjacent Labels uses (_correct_label_group_2d_core,
            focus_ids=[label_id]) — only label_id's OWN resulting side
            is ever written back
        else:
            plain single-label 2D correction (_intensity_correct_2d)
        if auto_grow and result touches its own local window's edge:
            retry this ONE slice at pad += growth_step, up to max_iterations
        if until_stable:
            keep re-running THIS SLICE (reseeded from its own last result)
            until it matches the previous attempt, or max_stability_passes
extend outward past the original [zmin, zmax] one slice at a time (up to
    z_extent_pad, a FIXED value never inflated by growth): copy the
    current footprint forward (foreign-protected), then run the same
    local correction there; undo and stop the moment nothing supports it
    — the true edge has been found
remove_debris_for_label() cleans up any small disconnected leftover
```
Every slice's own correction is *locally anchored* — re-derived fresh from wherever the label actually sits one slice ago, so it can only ever reach `pad` beyond its own verified position, never balloon into unrelated distant signal the way one whole-cell-sized window would (a real design flaw an earlier version had). `resolve_adjacent=False` treats every other label as purely excluded territory instead of jointly resolving against it — used for skin's own sentinel, which borders essentially every real cell and shouldn't be watershed-split against each one.

**`sand_label(labels, label_id, sigma_xy, sigma_z, pad)`** — the per-label engine behind §4's sanding pass: 3D Gaussian-blur the label's own binary mask, re-threshold at 0.5, foreign-excluded; skips (reports why) if the result collapses to nothing or loses contact with the original footprint.

**`_intensity_correct_2d` / `_intensity_grow_2d`** — the shared single-slice threshold core every 2D tool above calls into: crop to `seed_mask`'s bbox+pad, `candidate = image >= lo` minus foreign territory, fill the *whole* crop (not connectivity-restricted to the seed).

**`correct_adjacent_labels_2d(labels, image, label_a, label_b, z, lo, pad, sigma)`** — corrects two touching labels on one slice *simultaneously*, boundary placed by a **marker-seeded** watershed (`markers = each label's own existing footprint`, not auto-detected peaks — Split Label's blind peak-finding was tried and rejected here, since a combined region can have more than one real intensity dip and lock onto the wrong one). The working rectangle is sized from `label_a` alone (+pad), not the union — `label_b` only participates via whatever falls inside that rectangle.

**`correct_label_group_2d` / `_correct_label_group_2d_core`** — the N-label generalization of the above, used by Correct Adjacent Labels' own group-correction and (crop-only, no full-volume copy) by `correct_label_from_intensity_3d`'s per-slice joint branch. `focus_ids` optionally scopes the working rectangle to a subset of the group (mirrors `label_a`-only scoping); markers are always each label's own existing footprint, watershed runs unmasked over the whole crop then intersects with the real candidate signal so a diagonally-touching island still gets an owner.

**`touching_groups_for_stack`** (general-purpose utility, not on the current pipeline's own critical path — replaced there by the per-cell engine's own adjacency resolution): per-Z-slice union-find of mutually-touching label clusters (`_touching_pairs_on_slice` + `_union_find_groups`).

**`copy_label_to_adjacent_slice(labels, label_id, z_src, direction)`**: copies one label's 2D footprint onto the next/previous slice, clearing its own old footprint there first, foreign-excluded (reports how many pixels were dropped to foreign territory rather than pretending an exact copy).

**Skin protection**: `skin_voxel_count(labels, skin_label_id=-1)` — the single, array-content-only definition of "is skin already protected" every path checks (never widget state). `seed_skin_label(labels, brain_mask, skin_label_id=-1)` — bulk-fills every background voxel outside the brain mask with the sentinel `-1` (never overwrites a real label, never touches inside the brain); this is only a coarse seed. `trim_skin_label(...)` — a thin wrapper around `correct_label_2d_stack` (below) trimming that bulk seed down to real signal-supported territory, per-slice only (skin has no cross-slice-continuity problem to solve and spans nearly the whole Z range, so the 3D walk's machinery would be pure waste — the same reason 3D correction of an *existing* skin label is structurally blocked in the interactive UI). No brain-mask clamp (an earlier version had one; removed per the user's own root-cause diagnosis that it blocked legitimate inner-boundary correction — the debris problem it existed to prevent is instead handled by a `remove_debris(..., skin_label_id=...)` pass run right after). `remove_label(labels, label_id)` — plain bulk clear, not connectivity-restricted, used to drop skin's sentinel when no longer needed.

**`correct_label_2d_stack(labels, image, label_id, lo, pad, ..., n_workers)`** — corrects one label across its *whole* Z range, slice by slice, **independently and in parallel** (`ThreadPoolExecutor`, 75% of cores by default, `n_workers` overridable so a caller already parallel across cells — the auto-correct waves — doesn't multiply thread pools), each slice via `grow_correct_label_2d()` (the exact function the interactive Correct Label 2D button calls) on a one-slice *view*, not a full-volume copy per slice (a real fix for a 2026-09-18 OOM: the old version's `labels.copy()` per slice attempt, times 36 concurrent slice threads, exhausted 119 GB RAM). Wherever another real label is present, the boundary is jointly resolved via the same marker-seeded watershed, but only `label_id`'s own resulting side is ever painted back — guaranteed by watershed's own marker semantics (a label's own existing pixels are always its own marker, which watershed can never reassign), not a separately-enforced rule. `trim_skin_label()` is a thin wrapper calling this with `label_id = skin_label_id`.

### `_grow_correct.py` — auto-grow/until-stable orchestration

**`grow_correct_label_2d(labels, image, label_ids, z, lo, initial_pad, growth_step, max_iterations, sigma, until_stable, max_stability_passes)`** — wraps `correct_label_group_2d` in two nested loops:
```
group ← {label_ids}  (a single int for Correct Label; [A, B] for Correct
                       Adjacent Labels — the rectangle stays scoped to
                       the FIRST id even as the group grows)
for each stability pass (1 if until_stable=False, else up to max_stability_passes):
    for attempt in 1..max_iterations:
        correct the current group at the current pad
        new_neighbors ← labels the ORIGINAL group's own footprint (not a
                         folded-in neighbor's) GENUINELY touches, that
                         aren't in the group yet
        if new_neighbors: group |= new_neighbors; redo at the SAME pad
        elif the focus id(s) still touch their own rectangle's edge:
            pad += growth_step; retry
        else: converged, break
    if until_stable and this pass's result on every group member matches
       the immediately preceding pass's: stop (stable)
    else: reseed the next pass from this pass's own result
```
Neighbor discovery is deliberately narrow: only the *originally requested* label's own genuine touching-adjacency counts (never "merely present in the box," which would keep finding something as the box grows; never an already-folded-in neighbor's own further touches, which could cascade the group outward indefinitely).

**`grow_correct_label_3d(labels, image, label_ids, lo, ...)`** — now just a thin pass-through to `correct_label_from_intensity_3d(auto_grow=True, ...)`: all growth and stability happens *inside* that function's own per-slice walk (an earlier version redid the whole cell at a bigger global pad on every retry; removed once the per-slice engine made that redundant and much more expensive).

**`format_grow_report(report, mode)`**: renders either shape into the same human-readable style (which slices grew, which took multiple stability passes, whether it converged) used throughout the plugin's status/report boxes.

### `_contrast_sweep.py` — calibrating Correct Label's own `lo`

Answers "which `lo` best *reproduces* what Cellpose-SAM already segmented" — self-referential calibration against the model's own output, not independent GT (a deliberately different target from every GT-sweep tool in §8).

**`select_calibration_samples(labels, scale_zyx, n_cells, slices_per_cell, edge_margin_um)`**: picks the `n_cells` most morphologically complex cells (skeleton branch count) whose centroid sits ≥ `edge_margin_um` from the volume's own edge (a proxy for "not already touching skin"), each axis's margin capped at 30% of that axis's own extent so an un-capped physical-µm margin doesn't collapse Z's own thin window to nothing. For each, draws `slices_per_cell` Z-slices spread across the middle 60% of its own Z-extent (avoiding thin top/bottom cross-sections).

**`default_lo_candidates(image, samples, pad, n_steps)`**: auto-scales the candidate range from the 1st–99.5th percentile of intensity *around the actual samples*, not a hardcoded guess — robust to a different channel/bit-depth/normalization.

**`sweep_contrast_lower_value(labels, image, samples, lo_candidates, pad)`**:
```
for each candidate lo:
    for each (label_id, z) sample:
        corrected ← _intensity_correct_2d(..., lo) result (IoU=0 on error,
                     doesn't abort the whole sweep)
        iou[sample] ← IoU(corrected, the sample's EXISTING footprint)
    mean_iou[lo] ← mean(iou over all samples)   # jointly, not each sample's
                                                  # own independent best —
                                                  # stops one outlier sample
                                                  # from swinging the result
best_lo ← argmax(mean_iou)
```
Any non-positive label (skin `-1`) is treated as plain background during this sweep — otherwise the correction-under-test would see skin as foreign territory and exclude it, making the calibrated `lo` depend on whether skin happened to already be protected.

### `_auto_correction.py` — the full unattended per-cell pipeline

`auto_contrast_correct_stack(labels, image, scale_zyx, brain_mask, skin_image, ...)` — chained onto the end of Cellpose-SAM Segmentation (§4) when its "Auto-correct" checkbox is ticked, or run standalone via "Auto-correct Existing Labels" against any Labels layer. Six steps:

```
1. best_lo ← sweep_contrast_lower_value(...) via select_calibration_samples/
             default_lo_candidates (or lo_override, skipping the sweep
             entirely, minus an optional lo_adjustment subtracted after)

2. if skin (-1) already present in `labels`: REUSE it exactly as-is,
       never re-seed/re-trim (protects any prior manual correction)
   else:
       seeded ← seed_skin_label(labels, brain_mask)
       labels_with_skin, skin_report ← trim_skin_label(seeded, skin_image
           or image, -1, best_lo, pad=skin_pad, ...)
       # same best_lo real cells get — no longer offset, since skin's
       # own joint-resolution against a touching cell (§ above) made the
       # old +1 margin unnecessary

3. remove_debris(labels_with_skin, threshold, skin_label_id=-1)
   # sweeps whatever small stray blob skin's own unclamped trim absorbed

   GATE: skin_voxel_count(...) == 0 after steps 2-3 → raise. No cell is
   EVER corrected without verifiably-present skin protection.

4. resort_labels(sort_by="centroid_z")
   touching_skin_ids ← _labels_touching_skin(...)   # GEOMETRIC test on
       skin's FINAL territory (dilate each cell by skin_touch_px, check
       overlap) — NOT read from trim_skin_label()'s own foreign_nearby
       report, which lists every label ever transiently folded into a
       growth attempt (found to wrongly flag 33/33 cells on a real fish)
   waves, boxes ← _compute_correction_waves(...)   # partitions every
       cell into provably non-conflicting groups by each cell's own
       worst-case reach (pad + growth_step*max_iterations); same-wave
       cells run concurrently (75%-of-cores ThreadPoolExecutor), later
       waves see all earlier waves' already-corrected state

5. for each wave, for each cell (in parallel):
       if touching skin: correct_label_2d_stack(..., label_id=cell)
                          # same per-slice mechanism trim_skin_label()
                          # itself uses — a cell meeting skin has to do
                          # so on skin's own 2D-only terms
       else:              grow_correct_label_3d(..., cell)
       merge only this cell's own expanded box back into the shared array
       (wave members' boxes never overlap, so this is race-free)

6. remove_debris(new_labels, threshold, skin_label_id=-1)   # final
       whole-layer safety net

   (report-only) _flag_possible_non_microglia(...): flags any cell with
       >= non_microglia_fraction (default 40%) of its volume outside the
       brain mask, OR of its own boundary SURFACE within skin_touch_px of
       skin — macrophages or MONAI-missed skin that Cellpose segmented as
       a cell. Nothing is removed or changed; report only.
```
`format_auto_correction_report()` renders one consolidated report: the calibration, skin protection outcome, wave/parallelism summary, per-cell detail (reusing `format_grow_report()` for 3D cells, a matching `_format_2d_vs_skin_report()` for skin-touching ones), and the non-microglia flag list.

### `_widget.py` — Tab 3 (Edit MG Labels) handlers

**Shared layer selection**: `_edit_labels_layer()` / `_edit_signal_combo` / `_edit_mask_combo` — one explicit combo triple the user sets once, replacing the old implicit "whatever's active" fallback (see §1).

**`_on_resort_labels()`**: reads sort-by/reverse from the UI, calls `resort_labels()` in a background thread, writes the result back in place (`layer.data[:] = result; layer.refresh()` — not `layer.data = result`, which was found to race with napari's own async thumbnail update and crash; see §9).

**`_on_remove_debris()`**: reads Min volume/Min hole size from Common Settings, calls `remove_debris()`.

**`_on_split_label()`**: reads target label, N splits, σ, min distance, mode (2D/3D), calls `split_label()`, adds the new IDs to the layer.

**`_on_join_labels()`**: reads Label A/B (each with a "Use selected" button reading `layer.selected_label`), calls `join_labels()`.

**`_on_correct_label()`**: reads the signal layer's live `contrast_limits` for `lo`, target label, pad, mode (2D uses the current slice; 3D disabled outright for label `-1`, since a 3D correction of skin would walk nearly the whole fish — see `correct_label_from_intensity_3d`'s own note), auto-grow/until-stable checkboxes; calls `grow_correct_label_2d`/`3d` accordingly; sands the result afterward if Sanding is enabled (§4).

**`_on_correct_adjacent_labels()`**: reads Label A/B + current slice, calls `grow_correct_label_2d(label_ids=[A, B], ...)`, sands both labels afterward if enabled.

**`_on_copy_label_to_adjacent_slice()`**: reads label + direction, calls `copy_label_to_adjacent_slice()`.

**`_on_protect_skin()`**: runs the contrast sweep (§ above, "Calibrate Correct-Label Contrast" is the canonical source — every other autosweep tool in the plugin reuses its result rather than re-sweeping) to get `best_lo`, then `seed_skin_label` + `trim_skin_label`; refuses ("Skin is ALREADY protected…") if `skin_voxel_count() > 0` already. Populates the "Reduce best_lo by" reduction slider's own section.

**`_on_remove_skin_label()`**: calls `remove_label(labels, -1)`.

**`_on_auto_correct_existing_labels()`**: background-thread wrapper around `auto_contrast_correct_stack()` against whatever Labels/Image/brain-mask layers are selected — the same pipeline Cellpose-SAM Segmentation's own "Auto-correct" checkbox chains into, exposed as its own standalone button so it can be re-run against any existing Labels layer, not just fresh segmentation output. Wrapped in the hang watchdog (`_hang_watch_start`/`stop`, 30-minute `faulthandler.dump_traceback_later()`) given how long a full-fish run can take.

**`_on_sand_label()`** (interactive single-label sanding, when not chained automatically): reads σXY/σZ from the shared Sanding section, calls `sand_label()`.

**"Hide skin label" toggle**: a purely visual `DirectLabelColormap` swap (never touches `layer.data`) making skin render transparent — a `_FastDirectLabelColormap` subclass bypasses napari's own slower numba-dict backend selection for this one colormap (a real ~2x measured fix for 3D-display slowness), verified to produce byte-identical output to the base class.

**"Drift View in 3D"**: Start/Stop + speed slider driving a `QTimer`-ticked camera rotation (`viewer.camera.angles`, composed via a real `scipy.Rotation` each tick rather than incrementing 3 Euler angles independently — napari's own camera API restricts its middle axis to ±90°, which an increment-based approach visibly stalls against). Camera-only; never touches any layer's data.

---

## 6. Stage 4 — Statistics

**UI**: Tab 4 — Statistics, with a scrollable per-column output checklist (defaults matching core microglia morphology; several columns Nathalie's review flagged as marginal/unvalidated default OFF — see §9). **Module**: `_statistics.py`, one function (`compute_stats`) producing a `pandas.DataFrame`, one row per label, up to 51 columns. **Goal**: turn a corrected Labels layer into a per-cell CSV.

### Three-phase pipeline (`compute_stats`)

**Phase 1 — batch regionprops, GPU or CPU** (`_batch_regionprops`): one vectorized pass over every label at once — `cucim.skimage.measure.regionprops_table` on GPU (CuPy) if available, else `skimage.measure.regionprops_table` on CPU; both requesting the same property list (`label, area, centroid, bbox, inertia_tensor, inertia_tensor_eigvals, axis_major_length, axis_minor_length, solidity, extent`). From this table, vectorized post-processing derives:
```
IT ← the 3x3 inertia tensor per label, stacked
eigvals, eigvecs ← eigh(IT)                    # symmetric eigensolve, all labels at once
principal_axis_dir[i] ← "Z"/"Y"/"X", whichever axis the LONGEST eigenvector
                          (column 0) has its largest |component| along
axis1_um ← axis_major_length × (that axis's own physical scale)
axis3_um ← axis_minor_length × mean(sz, sy, sx)
axis2_um ← (axis1_um + axis3_um) / 2            # not independently measured
elongation ← axis1_um / axis3_um                # 1 = sphere, >1 = elongated
bbox_dz/dy/dx_um ← bbox extent x physical scale
volume_um3 ← voxel count x (sz x sy x sx)
eq_diam_um ← (6 x volume_um3 / pi) ^ (1/3)       # equivalent-sphere diameter
```

**Phase 2a — per-label marching cubes + skeleton, threaded** (`_slow_stats_worker`, one call per label in a `ThreadPoolExecutor` at 50% of cores): for each label's own bbox crop,
```
_surface_area(binary, scale_zyx):
    pad the crop by 1 voxel (avoids a boundary artifact)
    verts, faces ← marching_cubes(padded, level=0.5, spacing=scale_zyx)
    return mesh_surface_area(verts, faces)        # µm², 0.0 on any failure

_skeleton_stats(binary, scale_zyx):
    skeleton ← skeletonize(binary)                 # skimage, topological thinning
    sk ← skan.Skeleton(skeleton, spacing=scale_zyx)
    bd ← skan.summarize(sk)                        # one row per branch segment
    n_branches ← len(bd)
    n_endpoints ← count of branch-type == 1 (free ends, not junction-junction)
    mean/max_branch_len_um ← mean/max of bd's euclidean-distance column
    tortuosity ← mean(path_length / euclidean_distance) over branches with
                 nonzero euclidean distance — 1.0 = perfectly straight
    returns (0,0,0.0,0.0,1.0) on any failure (e.g. skan not installed, empty skeleton)
```

**Phase 2b — intensity stats, optional, also threaded** (`_intensity_stats_worker`, only if an Image layer was given): per-label mean/sum/coefficient-of-variation of the raw signal intensity within the label's own mask.

**Phase 3 — assembly + derived metrics + descriptions**:
```
for each label:
    sphericity ← min(1.0, pi^(1/3) x (6 x volume)^(2/3) / surface_area)
    surface_to_volume_ratio ← surface_area / volume
    branch_density ← n_branches / volume x 1e6          # per-µm³, scaled for readability
    endpoint_density ← n_endpoints / volume x 1e6
    process_complexity ← n_endpoints x mean_branch_len / volume
    morphotype ← _classify_morphotype(sphericity, solidity, elongation, n_branches, sav)
    description ← desc_fn(row)                            # rule-based or an LLM backend
```

**`_classify_morphotype(spher, solid, elong, n_br, sav)`** — rule-based, priority order:
```
if elong > 3.5 and n_br <= 3:                          → "Rod-shaped"
elif spher > 0.70 and solid > 0.80 and n_br <= 2:       → "Amoeboid"        (activated)
elif n_br >= 6 and sav > 2.0 and spher < 0.55:          → "Ramified"        (resting)
elif n_br >= 4 and spher < 0.65:                        → "Intermediate-ramified"
else:                                                    → "Intermediate"
```

**Post-assembly, whole-dataset spatial statistics** (`_spatial_stats`, `scipy.spatial.cKDTree` over all centroids at once):
```
tree ← cKDTree(centroids_um)
k=3 query → self + 1st NN + 2nd NN (k=2 fallback if only 2 cells exist)
nearest_neighbor_dist_um, nearest_neighbor_2_dist_um ← those distances
Clark-Evans 3D index:
    density ← N / bbox_volume(all centroids)
    expected_NND ← Gamma(4/3) x (3 / (4*pi*density))^(1/3)   # Poisson-process expectation
    nearest_neighbor_ratio ← observed_NND / expected_NND      # <1 clustered, >1 dispersed
local_density_100um ← count of other centroids within a 100µm radius (tree.query_ball_point)
depth_normalized ← centroid_z_um / max(centroid_z_um), clipped to [0,1]
```

**Post-assembly, optional brain-region assignment** (`_assign_brain_regions` + `_polyline_side_and_dist`), only if a Shapes layer with `"line"`/`"path"` shapes and region names were given:
```
boundaries ← the drawn polylines, in µm, sorted by mean X of their own vertices
             (ascending = most anterior first) -- the fish lies along X,
             head at X=0; each boundary is meant to run top-to-bottom (Y),
             dividing anterior (smaller X, left of the path) from posterior
for each cell's centroid:
    for each boundary, nearest-to-first:
        find the closest point on ANY of the boundary's segments (point-to-
        segment projection, not just to its two endpoints -- supports a
        curved multi-vertex polyline, not just a straight 2-point line)
        side ← sign of the cross-product at that nearest segment
        if side says "posterior of this boundary": region_index += 1
    region ← names[region_index]                    # N boundaries -> N+1 regions
    region_boundary_dist_um ← distance to the closest boundary overall
```

### Description backends (`_make_desc_fn`)

Four interchangeable backends, selected by `backend_config["backend"]`:
- **`rule`** (default, always available, offline): `_rule_based_description` — a template assembling shape (from sphericity/elongation), surface texture (from solidity), branching, and the morphotype tag into one sentence.
- **`ollama`**: POSTs the same statistics, formatted into `_STATS_PROMPT`, to a local Ollama server's `/api/generate`.
- **`openai`** / **`claude`**: same prompt to the OpenAI Chat Completions or Anthropic Messages API respectively, using a key from `_secrets.py` (§2).
All three network backends catch their own request errors and return an inline `"[Backend error: ...]"` string rather than raising — one label's description failure never aborts the whole run.

### `_widget.py` — Tab 4 handlers

**Per-column checkboxes**: `_COL_GROUPS` maps one checkbox to multiple CSV columns where they're conceptually one unit (e.g. `centroid_vox` → all three `centroid_z/y/x_vox` columns; `bbox_vox` → all six bbox corner columns; `nn_1st`/`nn_2nd` → the label+distance pair each). Validation warns (doesn't crash) if brain-region or intensity columns are checked but no Shapes/Image layer is selected, and just skips those columns.

**`_refresh_stats_layers()`**: repopulates the Image/Shapes layer combos on every napari layer add/remove event, preserving the current selection across the refresh (`blockSignals()` guard).

**`_on_generate_stats()`**: reads the active Labels layer's own scale from `self._state["metadata"]` first (the value Tab 1's Open button set), falling back to the Labels layer's own napari `scale` only if no stack metadata is on record — a real historical bug had this backwards (defaulting to the Labels layer's own possibly-unset scale), silently making `centroid_um`/`volume_um3` identical to their voxel-unit counterparts whenever a labels `.tif` was loaded without its own embedded scale. Extracts region-boundary polylines from the selected Shapes layer via `_extract_region_lines_um()` (accepts both `"line"` and multi-vertex `"path"` shape types), calls `compute_stats()` in a background thread, filters the returned DataFrame down to only the checked columns before writing the CSV.

**GT-verification checkbox** ("This is verified ground truth"): when ticked, also measures the active layer's own smallest/largest real cell volume and smallest real hole directly from this GT, updating the shared cross-fish Min/Max-volume and Min-hole-size histories (`_update_gt_history`) other tools throughout the plugin read from — the same one-shot, self-unticking pattern every GT-sweep tool in §5/§8 uses.

---

## 7. Stage 5 — AI Tools

**UI**: Tab 5 — AI Tools, a manual MONAI-Training/Cellpose-SAM-Training switch (not auto-detected — no per-layer signal to key off here), gated informationally (not blocked) on GPU/VRAM (§2's `_gpu_check.py`). **Goal**: build and train the two models Stages 1–2 depend on, and produce the ground-truth data that trains them.

### `_gt_annotation.py` — polygon-based GT annotation

Ported from an earlier standalone tool (`polygon_annotation_tool.py`) into plain viewer-taking functions on the plugin's own shared viewer, rather than a second `napari.Viewer()` — three deliberate deviations from the original noted in the module docstring: the target Image layer is passed explicitly (not "first Image layer in the list," a real risk once Tabs 1–3 have added other layers); the `brain_polygons` Shapes layer is auto-created on demand; `generate_masks()` no longer hides every other layer in the viewer.

```
resample_polygon_preserve_order(pts, n_points=96):
    walk the closed polygon's own perimeter by cumulative arc length,
    resample to exactly 96 evenly-arc-spaced points — gives every
    hand-drawn polygon, regardless of how many vertices were clicked,
    the same point COUNT and roughly consistent point CORRESPONDENCE
    for the interpolation step below

is_polygon_clockwise(yx) / standardize_polygon_direction(...):
    shoelace-formula signed area — every key-slice polygon is forced to
    the SAME winding direction as the first (reference) one, since
    point-to-point interpolation between two oppositely-wound polygons
    would twist through the middle instead of sweeping smoothly

interpolate_shapes(viewer, image_layer):
    by_z ← {slice: resampled 96-point polygon} for every hand-drawn key polygon
    standardize every polygon's winding to match by_z's own first slice
    for every Z slice in the volume:
        if a key polygon exists exactly here: use it as-is
        elif slice > PROPAGATION_SLICE (default 90) and propagation is on:
            reuse the PROPAGATION_SLICE polygon UNCHANGED (just relabel its
            own Z) — for the common case where the brain silhouette stops
            changing meaningfully past a certain depth, so no further
            hand-drawn key slices are needed there
        else:
            find the nearest key slice below and above this one
            linearly interpolate every one of the 96 point-pairs between
            them by the fractional Z position (point i of the lower
            polygon -> point i of the upper polygon)
    → a new "brain_polygons_interpolated" Shapes layer, one polygon per slice

generate_masks(viewer, image_layer, input_path):
    brain_mask ← rasterize every interpolated polygon via napari's own
                  Shapes.to_labels(), thresholded > 0
    optional median_filter along Z (window 3) to smooth slice-to-slice
        rasterization jitter
    skin_mask ← 1 - brain_mask
    brain_only / skin_only ← image x each mask
    save brain_mask/skin_mask/original/brain_only/skin_only .tif + both
        polygon .npz files under <input>/<stem>/ (the plugin's own
        output-folder convention)
```

### `_xzyz_patches.py` — 2D orientation-slice crop generation (current training methodology)

Generates the actual crop set every real Cellpose-SAM training run since May 2026 has used — **not** the older bbox-based single/double/triple/quadruple crop approach a separate, still-available tool ports from an April-2026 script that was abandoned after failing (catastrophic forgetting on too little data).

```
stretch_z(slice_2d, anisotropy): bilinear zoom of the Z axis by
    anisotropy — brings a Z-cross-section (XZ or YZ) to roughly the same
    pixel scale as a native XY slice, since this project's voxels are
    strongly anisotropic
stretch_z_mask(...): the same zoom, nearest-neighbor (order=0) — integer
    labels must never be blended/interpolated

_random_crops(img_slice, gt_slice, crop_size, n_crops, rng, min_gt_pixels):
    repeat (up to 30x n_crops attempts):
        pick a random GT-positive pixel as an anchor
        place a crop_size x crop_size window with a random offset so the
            anchor lands somewhere inside it (not necessarily centered)
        keep it only if it contains >= min_gt_pixels of real GT signal,
            and its own top-left corner hasn't already been used

generate_xzyz_patches(image_path, gt_path, out_dir, anisotropy, crop_size=512,
                       ncrops_per_slice=5, max_per_orientation=320, ...):
    for each orientation in (XY native, XZ stretched, YZ stretched):
        shuffle every GT-containing slice of that orientation
        for each slice (until max_per_orientation total crops collected):
            draw up to ncrops_per_slice random crops from it
        cap the COLLECTED list to max_per_orientation, THEN write files
            (not an early-exit mid-slice that could overshoot the cap)
        save <orientation>_<slice:03d/04d>_<i:02d>.tif + _masks.tif pairs
```

### `_crop_truncation.py` — cleaning incidental-neighbor truncation

A crop framed around one target cell often also grazes the corner of a different, nearby cell purely by chance, keeping a valid-looking but tiny sliver whose visible centroid is nowhere near that cell's true center — a genuinely wrong training target for Cellpose's flow-vector loss (which points every voxel toward its own object's true centroid). This is the exact fix applied by hand to this project's real training data before being ported into the plugin.

```
clean_crop_truncation(crop_dir, gt_full_path, anisotropy, threshold=0.9):
    back up crop_dir first (skip if a backup already exists — never
        clobbers a still-valid backup with already-modified files)
    for each <stem>.tif/<stem>_masks.tif pair (xy/xz/yz_NNN_NN naming):
        true_slice ← the SAME stretch_z_mask() transform generate_xzyz_
            patches() used at generation time, applied to the fish's own
            full GT volume at this crop's source slice index (cached —
            many crops share a source slice)
        for each label present in the crop's mask:
            visible ← pixel count in the crop
            true_count ← pixel count in the true full-slice cross-section
            if visible < threshold x true_count: zero this label in the
                crop (an incidental neighbor caught mid-truncation) —
                the crop's own INTENDED target cell is essentially never
                affected, since it's already near-complete by construction
        overwrite the mask file only if something was actually zeroed
```

### `_branch_calibration.py` — measuring real branch radius from GT

Recalibrates `train_xzyz.py`'s `branch_radius` parameter (the erosion-survival-distance threshold the branch-weighted training loss uses to decide "this pixel is part of a thin branch") from actually-measured GT morphology instead of a frozen guess.

```
measure_branch_diameters(gt_labels, scale_zyx, pad, min_volume):
    for each labeled cell (cropped to its own padded bbox):
        edt ← anisotropic distance_transform_edt(binary_mask, sampling=scale_zyx)
        skel ← skeletonize(binary_mask)                    # topological thinning
        skel_valued ← edt VALUES baked onto the skeleton's own nonzero
            voxels (a real skan API gotcha: Skeleton(..., source_image=...)
            does NOT feed skan.summarize()'s own mean-pixel-value column —
            that column reads the skeleton_image array's own values, so
            the EDT has to be baked IN, not passed as a separate argument)
        bd ← skan.summarize(Skeleton(skel_valued, spacing=scale_zyx))
        diam_um ← 2 x bd's mean-pixel-value per branch segment
    → pooled diam_um / length_um across every cell

recommend_branch_radius(gt_labels, scale_zyx, xy_scale=None):
    stats ← measure_branch_diameters(...)
    tip_diam_um ← the thinnest QUARTILE of all measured diameters
                   (distal branch tips — what branch_radius exists to protect)
    tip_radius_um ← mean(tip_diam_um) / 2
    recommended_px ← round(tip_radius_um / xy_scale)   # µm -> pixels at the
                       crop's own XY resolution, directly comparable to
                       train_xzyz.py's own branch_radius parameter
```
An approximation, not an exact reconstruction — the training-time threshold is a Chebyshev (chessboard) erosion distance in pixels; this measures a true Euclidean physical radius and converts. Close enough to calibrate a threshold, acknowledged in the docstring as not identical.

### `_training_jobs.py` — cross-platform detached process management

Both MONAI (`train.py`, hours–days) and Cellpose-SAM (`train_xzyz.py`, ~20h) training runs must survive napari closing and be re-discoverable after a full restart. Built on primitives that behave identically on Linux/Mac/**Windows** (this plugin ships a native Windows install path, so a tmux-only design was rejected): a detached `subprocess.Popen` + PID tracking via `psutil`, stdout/stderr to a plugin-controlled log file the GUI tails.

```
launch_detached(argv, cwd, log_path, conda_env, notify=None):
    if notify is given:
        write a small STANDALONE supervisor .py file (stdlib-only —
            subprocess/smtplib/re, no dependency on this package being
            importable) that: runs the real "conda run -n <env> <argv>"
            command (same log_path the GUI already tails live) -> waits
            for it to exit -> parses the log for the best-checkpoint
            metric -> emails a completion report via SMTP_SSL
        the SUPERVISOR itself is what's launched detached, not argv
            directly — so the email still fires even if napari closes
            and never reopens before the job finishes. kill_process_tree
            needs no special case: the real training process is a
            descendant of the supervisor, so killing the whole tree (a
            manual Stop click) kills the supervisor before it reaches
            its own email step, correctly sending no "stopped" email.
    else:
        Popen(["conda","run","-n",env,"--no-capture-output",*argv],
              stdout/stderr -> log_path,
              start_new_session=True on POSIX (setsid) /
              CREATE_NEW_PROCESS_GROUP|DETACHED_PROCESS on Windows)
    return pid

is_running(pid) / kill_process_tree(pid): psutil-based liveness check /
    recursive terminate-then-force-kill of pid + every descendant (conda
    run spawns a child python process, so killing only the top PID would
    leave training running)

tail_log(log_path, n_bytes=8192): last N bytes only, bounded regardless
    of how many days/GB the log has grown to

patience_exceeded(log_path, metric_cfg, patience):
    parse every (epoch, metric_value) the log's own regex matches
        (MONAI_METRIC: "Epoch N Summary:" ... "Full-brain Dice: X
         [MODEL SELECTION]", DOTALL, higher-is-better; CELLPOSE_METRIC:
         "N, train_loss=..., test_loss=X" on one line, lower-is-better —
         the ONLY thing that differs between the two scripts' early-
         stopping is this metric config, the counting logic is shared)
    best ← the best-value checkpoint logged so far
    checkpoints_since_best ← how many logged checkpoints have passed
        without a new best
    exceeded ← patience > 0 and checkpoints_since_best >= patience
    # patience counts CHECKPOINTS (the plugin's own observable cadence
    # from the log), not epochs — each script's own internal epoch/val
    # cadence differs and doesn't matter here

write_best_checkpoint_pointer(models_dir, model_name, best_epoch):
    writes "<model_name>_best_recommended.txt" naming the checkpoint file
    — a plain TEXT pointer, not a copy (checkpoints are 100s of MB) and
    not an OS symlink (needs elevated privileges/Developer Mode on
    Windows) — resolves identically cross-platform with zero special
    permissions. Only needed for Cellpose-SAM, which has no built-in
    best-tracking of its own (MONAI's train.py already auto-saves its
    own best_model_fullstack.pth).
```

### `_ai_tools.py` — argv builders

Pure argument-list construction (`build_prepare_data_argv` / `build_monai_train_argv` / `build_cellpose_train_argv`) for the three training scripts, which ship *inside the installed package* (`training_scripts/`, declared as package data) rather than as sibling folders in this project's private monorepo — a real historical bug: a monorepo-sibling path only ever existed on the original dev machine, so anyone else installing via `pip install git+https://...` got "script not found" at launch. Package placement means they travel into `site-packages` on both editable and normal installs. Always overridable via a Browse button.

### `_widget.py` — Tab 5 (AI Tools) handlers

**GT Annotation section**: buttons wrapping `interpolate_shapes()`/`generate_masks()` directly — synchronous (both are fast: pure numpy/shape rasterization, no model inference), errors shown in the status label rather than crashing.

**Extract Training Crops** (two tools, side by side): the older bbox-based single/double/triple/quadruple extractor (kept for anyone still using that approach) and `generate_xzyz_patches()` (the current default), both background-thread wrapped with a cancel button (`cancel_event`) since a full-fish run can take a while. "Clean crop truncation" is on by default as part of the XZYZ extractor's own run (not a separate button to remember), per explicit instruction that Cellpose-SAM should train only on near-complete cells (≥90%) going forward.

**Calibrate branch_radius (from GT)**: background-thread wrapper around `recommend_branch_radius()`, with the same one-shot "This is verified ground truth" gating pattern (§2/§5/§8) before it's allowed to update the shared `branch_radius` config value.

**Train MONAI U-Net / Train Cellpose-SAM** buttons: build the argv via `_ai_tools.py`, call `launch_detached()`; poll `is_running`/`tail_log`/`patience_exceeded` on a `QTimer` while napari stays open, and **reconnect automatically on napari reopen** by reading the persisted PID + log path from config — a dead PID found on reopen finalizes immediately (best checkpoint reported, pointer written, stale PID cleared), closing a real gap where a job that finished while napari was closed previously showed nothing until manually reopened and re-triggered. "Stop Training" calls `kill_process_tree`. An optional Email-notification panel (SMTP fields, password via `_secrets.py`) threads a `notify` dict through to `launch_detached`.

---

## 8. Stage 6 — Sweeps & Utilities

**UI**: Tab 6 — Sweeps & Utilities. **Modules**: `_sieve.py` (generic coarse-to-fine engine, already covered in §2), `_brain_sweep.py`, `_pixel_sweep.py`, `_krendl_sweep.py`, `_epoch_sweep.py` (the four GT-verified parameter sweeps), `_gt_score.py` (the whole-fish scorer they build on), `_gt_package.py` (packaging for external manual correction), `_gt_toolkit.py` (single-folder auto-discovery tying them together). **Goal**: check a pipeline parameter against real ground truth and, once confirmed, recalibrate the "recommended" defaults quoted throughout §3–§5 — this is how those defaults were actually found, not guessed.

### `_gt_score.py` — the whole-fish scoring methodology

`score_against_gt(pred_labels, gt_labels, iou_threshold=0.5)` is the project's single, validated instance-matching methodology (used, in different guises, by every sweep in this section):
```
pred_info, gt_info ← per-object volume/centroid/bbox (_get_info from _cellpose_seg.py)
build a (n_gt x n_pred) IoU cost matrix, but only for pairs whose BBOXES
    actually intersect (IoU is necessarily 0 otherwise — keeps this fast
    even for dozens of objects, without ever missing a real match)
row_ind, col_ind ← linear_sum_assignment(-iou_matrix)   # Hungarian algorithm,
                     the GLOBALLY optimal one-to-one matching, not a greedy
                     nearest-first assignment
for each assigned (gt, pred) pair:
    if IoU >= iou_threshold: count as TP, record IoU/Dice/size_delta
every unmatched GT object → FN;  every unmatched pred object → FP
score ← TP - 0.5 x (FP + FN)                    # this project's own established metric
mean_iou / mean_dice ← averaged over TP matches only
```
Verified against this project's own historical result tables (e.g. `TP=34/FP=8/FN=5 → Score=+27.5`).

### `_brain_sweep.py` — MONAI Threshold x Erosion (Tab 1's own mask)

The cheapest of the four sweeps: `predict_probability()` (the expensive network pass) runs **once**; every candidate `(threshold, erosion)` combination is then a cheap re-threshold via `postprocess_probability()` + a `binary_erosion`, scored as a whole-volume Dice/IoU/precision/recall against a hand-corrected GT brain mask (from GT Annotation, §7) — no per-cell matching needed, since brain segmentation is one global object.
```
pred_prob ← predict_probability(volume, model_path, device)   # once, or
             reused via precomputed= across sieve stages
for each threshold:
    raw_mask ← postprocess_probability(pred_prob, threshold)
    for each erosion:
        mask ← binary_erosion(raw_mask, erosion) if erosion > 0 else raw_mask
        score whole-volume Dice/IoU/precision/recall vs GT
best_point ← the (threshold, erosion) with highest Dice
```

### `_pixel_sweep.py` — the Pixel Classifier route's two sweeps + GT-measured floors

Deliberately does **not** re-run MONAI inference per grid point (neither Signal Erosion nor BG Threshold change what MONAI predicted) — takes a pre-computed, un-eroded `brain_mask.tif` as input, making the whole sweep union-find-bound (minutes, no GPU needed) rather than GPU-inference-bound.

**`run_pixel_sweep(image_path, brain_mask_path, gt_labels_path, bg_thresholds, erosions, ...)`**:
```
cells ← find_complex_cells(gt_labels, n_cells)     # most skeleton branches
        first, NOT by volume — a large cell is often simple/amoeboid,
        while a genuinely branchy cell stresses the pipeline more
for each cell: crop image+brain_mask+GT to its own padded bbox once
bg_max ← the GLOBAL background estimate (histogram over the whole
          volume's in-brain pixels) — computed once, since it's a
          genuinely global quantity, independent of Signal Erosion
for each bg_threshold:
    thresh ← bg_max + data_range x (bg_threshold / 100)
    for each erosion:
        for each cell's own crop:
            signal_mask ← erode_signal_2d((image > thresh) & brain_mask, erosion)
                           # erodes what SURVIVED the threshold, not the
                           # probe mask itself — matches remove_global()
            pred_labels ← create_labels(image x signal_mask, sigma_xy, sigma_z,
                                         min_volume, min_hole_size)
            score ← best-IoU match of any predicted object against this cell's GT
    average IoU/Dice across all N cells at this grid point
best_point ← highest average IoU
```

**`run_sigma_sweep(...)`** — the Smooth σXY/σZ counterpart, roles reversed: BG Threshold/Signal Erosion held fixed, σXY/σZ swept. Cheaper per point than `run_pixel_sweep`, since σ only affects `create_labels()`, not the background-threshold crop-building step (computed once per cell, reused across every σ combination).

**`min_volume_from_gt(gt_labels)`**: the smallest true voxel volume among GT-labeled cells — replaces a single hardcoded constant that used to be shared across every fish regardless of whether its real microglia ran smaller/larger. **`min_hole_size_from_gt(gt_labels, min_hole_size_to_trust=5)`**: the smallest *real* internal hole any hand-corrected GT cell actually has (a human deliberately left it unlabeled) — holes below a trust threshold are discarded first, since real GT showed a sharp bimodal split (1–2 voxel annotation slips vs. a cluster of 400+ voxel real structural gaps, nothing in between); trusting every reported hole collapsed the recommendation to 0. **`min_intercell_gap_um(gt_labels, scale_zyx, pad_um=15)`**: smallest real physical surface-to-surface gap between any two distinct GT cells — a safety ceiling for Krendl safe-merge's `max_gap` (never bridge a gap that could be two real cells), computed per-cell-bbox rather than one full-volume distance transform (which timed out past 100s on a real fish).

### `_epoch_sweep.py` — GT-verified Cellpose-SAM checkpoint sweep

Automates the bbox-restricted GT-IoU methodology developed by hand across this project's training rounds to confirm a checkpoint pick (from `test_loss`, a proxy metric) against real GT.
```
find_complex_cells(gt_labels, n_cells):
    rank every GT cell by (n_branches, -sphericity) — same skeleton/
    surface-area code as Statistics (§6), so this ranking matches what's
    reported elsewhere in the plugin
bbox_crop(image, gt_labels, label_id, pad_z=15, pad_xy=40): crop to one
    cell's own padded bbox — do_3D cost scales with crop size, so this
    turns a full-fish sweep (~hours per checkpoint) into a small handful
    of seconds-scale crop inferences

run_epoch_sweep(image_path, gt_labels_path, models_dir, model_name, epochs, ...):
    crop each of the N complex cells once
    for each candidate epoch:
        load that checkpoint ONCE (not once per cell)
        for each cell's crop: raw do_3D inference (no GMM/Krendl — just
            best-IoU-match the single predicted object against this cell's GT)
    average IoU/Dice per epoch across all N cells
best_epoch ← highest average IoU
```
`pick_sweep_epochs(recommended_epoch, save_every, available_epochs, n_below=2, n_above=2)`: centers the candidate epoch list on the `test_loss`-recommended epoch, stepping by the training run's own checkpoint interval, clipped to whatever actually exists on disk.

### `_krendl_sweep.py` — Cellprob x Large-contact + real-data merge-parameter calibration

The most involved sweep module — three related but distinct calibration tools, reflecting a real methodological evolution documented in the module's own history (see `skin_segmentation.md`'s 2026-08-17/18 entries).

**`run_krendl_sweep(volume, gt_labels, model_path, cellprobs, large_contacts, ...)`** — the original instance-matched sweep, scored via `score_against_gt()` on the *fully corrected* result:
```
model, dP, cellprob_map, shape ← predict_flows(...)   # ONE network pass —
     cellprob_threshold only feeds the cheap masks_from_flows() step
     (confirmed directly from cellpose/models.py's own eval()/_compute_masks()
     split), so re-running the expensive GPU pass per cellprob value would
     be needless; a multi-stage sieve narrowing can reuse this same pass
     across every stage via precomputed=
for each cellprob:
    masks ← masks_from_flows(..., cellprob)
    masks ← gmm_cleanup(masks); masks ← krendl_safe_merge(masks, max_gap,
             min_contact, gt_min)                       # once per cellprob
    for each large_contact:                              # cheap on top of
             the same intermediate result
        merged ← large_contact_merge(masks, large_contact)
        merged ← final_min_size_cleanup(merged, gt_min, final_min_fraction)
        labels ← relabel_sequential(merged)
        results[(cellprob, large_contact)] ← score_against_gt(labels, gt_labels)
best_point ← highest Score, ties broken by mean_iou then mean_dice, EXCLUDING
             any point with tp==0 (a point that detects nothing can still
             "tie" a genuinely-detecting-but-noisy point on Score — never a
             usable winner) unless literally every point detected nothing
```
`gt_min` (Krendl safe-merge's "already a whole cell" floor) and `min_hole_size`, if not given, are each measured from the real `gt_labels` passed in (`min_volume_from_gt`/`min_hole_size_from_gt`, imported from `_pixel_sweep.py` — the same measurement, since `gt_min` and the Pixel Classifier's `min_volume` turned out to be literally the same quantity, unified after being tracked as two separate config histories by historical accident).

**`run_cellprob_voxel_sweep(volume, gt_labels, model_path, cellprobs, ...)`** — breaks a real circularity in the tool above: scoring cellprob against the *fully corrected* result entangles "best cellprob" with whatever `max_gap`/`min_contact`/`large_contact` happen to already be set to. This sweep scores cellprob on **pure voxel-level Dice/IoU** against binarized GT (instance identity ignored entirely) — mirrors `_brain_sweep.run_brain_sweep()`'s identical mask-level approach for MONAI Threshold. The documented correct calibration order: this sweep picks cellprob on signal quality alone → generate real `cp_masks` at that cellprob → the two functions below calibrate the merge parameters from that real prediction → optionally cross-check with `run_krendl_sweep`'s instance-matched Score using the now-non-circular calibrated parameters.

**`measure_merge_params_from_prediction(cp_masks, gt_labels, scale_zyx, min_overlap_vox=5, search_pad_um=3.0)`** — measures `max_gap`/`min_contact` from a real raw (pre-GMM, pre-Krendl) prediction rather than pure-GT geometry (which has a real ambiguity: is the smallest gap between two GT cells real biology, or just how an annotator drew the shared boundary?). Sidesteps that by using GT only to *label* which raw fragments belong to which real cell, then measuring the network's own actual fragmentation:
```
assign every raw fragment to whichever GT cell it overlaps MOST (drop as
    noise if the best overlap is < min_overlap_vox)
"should_merge" samples: for every GT cell with >= 2 assigned fragments,
    each fragment's gap/contact to its NEAREST same-cell sibling only —
    not all pairwise combinations (krendl_safe_merge() only ever needs to
    bridge to the nearest neighbor, iteratively; all-pairs would badly
    inflate the "should merge" gap distribution with meaningless far-apart
    pairs within one large sprawling real cell)
"should_not_merge" samples: fragment pairs assigned to DIFFERENT GT cells
    that end up close (bbox-proximity pre-filtered) — an unambiguous
    safety-ceiling data point, since GT already confirms these are
    genuinely separate cells
```

**`recommend_merge_params(merge_stats_list)`** — turns those pooled samples into one recommended `(max_gap, min_contact)` **pair**, jointly optimized against the REAL OR-combined decision rule (`krendl_safe_merge` merges if `gap <= max_gap` OR `contact >= min_contact`), not two independently-optimized single-criterion thresholds:
```
every OBSERVED (gap, contact) value becomes a candidate threshold (exact,
    not a coarse grid)
for every candidate (max_gap, min_contact) pair:
    missed ← should-merge samples where BOTH gap > max_gap AND contact < min_contact
              (computed for the whole grid in one matrix multiply,
               _count_gap_gt_and_contact_lt, not a triple-nested loop)
    false ← should-not-merge samples where NEITHER criterion would have
             excluded them (i.e. the pair WOULD wrongly merge them)
    total_error ← missed + false
best ← the (max_gap, min_contact) pair minimizing total_error
```
Optimizes for **fewest total manual corrections**, not zero risk of one error direction — an explicit, user-confirmed design choice: an over-merge is trivially fixable with Split Label, an under-merge with Join Labels (§5), so neither failure is silent or catastrophic; a defensively-clamped earlier version that minimized false-merges alone caught close to none of the real should-merge cases on real test data.

### `_gt_package.py` — packaging for external manual correction

`build_gt_package(stem, out_dir, image_path, corrected_masks_path, raw_cellpose_masks_path=None, brain_mask_path=None, ...)` — automates a manual packaging step repeated by hand for every fish sent out for GT correction: copies the source image, the most-advanced correction stage available (as the correction starting point), the raw pre-merge masks (reference only), an optional brain mask (needed if the reviewer wants to use Protect Skin as Label themselves), and `GROUND_TRUTH_CREATION_GUIDE.md`, plus a lightweight per-cell CSV (`_cell_statistics_csv` — label/volume/centroid/bbox only, deliberately not the full ~51-column Tab 4 output), all zipped into `<stem>_GT_package.zip`.

### `_gt_toolkit.py` — single-folder auto-discovery ("GT Toolkit Tuning Tool")

Given one fish's original source file (or its already-created output folder), locates every artifact this project's established naming convention produces there and decides which of the above sweep/calibration steps can actually run without the user hunting down and browsing to each file by hand.
```
SUFFIXES ← the fixed map of {key: filename suffix} this plugin's own
           _output_dir() convention produces (_original.tif, _brain_mask.tif,
           _brain_only_ExtRm/NoBG/RndFill.tif, _cp.tif, _cp_krendl.tif,
           _cp_krendl_ac.tif, _cp_krendl_ac_snd.tif, _GROUND_TRUTH.tif,
           _statistics.csv) — Cellpose-SAM's own naming is cumulative and
           self-documenting: _cp_krendl_ac_snd.tif's name itself says it
           went through Krendl, then auto-correct, then sanding

discover_fish_files(source_path):
    resolve_fish_folder(...) ← accepts the folder itself, any file inside
        it, or the original raw source file sitting beside it
    for each SUFFIXES key: an exact <stem><suffix> match, else the single
        result of a same-suffix glob if exactly one exists (tolerates the
        folder name and a file's own stem drifting apart slightly), else None

best_corrected_masks(found): the most-advanced stage present, priority
    sanded > auto-corrected > Krendl-only (never the raw "cp" stage,
    which is always kept as a separate reference file)

step_preconditions(found): {step_key: (can_run: bool, reason: str)} for
    each of the 5 folded-in steps (monai/bg/sigma/cellprob/branch_radius)
    — e.g. the MONAI sweep needs BOTH _original.tif AND _brain_mask.tif to
    exist, since their joint presence is what signals this fish went
    through GT Annotation (a raw MONAI-predicted brain_mask.tif alone
    isn't real ground truth to sweep against)
```

### `_widget.py` — Tab 6 (Sweeps & Utilities) handlers

Every sweep tool shares: a **Sieve checkbox** (§2's `_add_sieve_controls()` — coarse full-range pass, then 1–2 narrowing stages around the winner) wired to `run_sieve()`; a one-shot **"This is verified ground truth"** checkbox (off by default, self-unticking after every run) gating whether the result is allowed to auto-apply to a live slider/config value or just reported; a **cancel button** wired to a `threading.Event` checked between grid points; and, once a sweep confirms a value, a call to `_update_gt_history(config_key, fish_key, value, mode)` — the shared never-rising/never-falling cross-fish floor/ceiling mechanism every "recommended" default in this plugin is tracked through (§9).

**GT Toolkit Tuning Tool**: a single file/folder picker driving `discover_fish_files()` + `step_preconditions()`, pre-filling every other sweep tool's own file fields with whatever it found and greying out (with a reason shown) any step this fish's files can't support.

**Build GT-Correction Package**: background-thread wrapper around `build_gt_package()`, auto-selecting `best_corrected_masks()` and the matching `_cp.tif`/`_brain_mask.tif` when a fish folder is picked via the GT Toolkit discovery above.

---

## 9. `_widget.py` — application shell

At 10,810 lines and 145 functions, `_widget.py` is both the biggest file in the plugin and the least algorithmically interesting one — the vast majority of it is straightforward Qt widget construction (`_build_ui()`, ~4,000 lines) and thin handler functions that read a few widgets, call into one of the modules covered in §3–§8, and write the result back to a napari layer via the threading pattern in §1. Rather than re-describe every handler a second time, this section covers the parts of the file that are genuinely its own infrastructure — construction, config persistence, cross-fish history, layer resolution, the hang watchdog, and email notification — and then indexes every handler by which pipeline-stage section already documents it.

### Class structure

```
_FastDirectLabelColormap(DirectLabelColormap)   — module-level, see §5
                                                    ("Hide skin label")
_load_config() / _save_config(data)             — module-level, plain
                                                    JSON read/write of
                                                    ~/.config/napari-zf-
                                                    microglia-ai/config.json
_add_reliable_spinbox / _add_gt_checkbox /
_add_sieve_controls / _is_valid_cellpose_
checkpoint / _add_recommended_label /
_set_layout_widgets_visible / _make_collapsible /
_wrap_scroll / _sep / _make_notify_checkbox /
_extract_region_lines_um                        — module-level shared
                                                    UI helpers, see §2/§6

class ZFMicrogliaAIWidget(QWidget):
    __init__(napari_viewer)
    _build_ui()            — constructs every tab's every widget
    _connect_signals()      — wires every button/combo/layer-event to its handler
    <145 methods total>     — handlers, one section per tab, plus shell helpers
```

### `__init__` and startup sequence

```
cfg ← _load_config()
cfg ← _secrets.migrate_plaintext_secrets(cfg)   # one-time upgrade, see §2
model_path ← saved config path, else the bundled default checkpoint, else
             None — resolved via is_file(), not exists(): Path("") silently
             normalizes to Path(".") under pathlib, and exists() is True
             for a directory too, so an unconfigured path used to be
             wrongly treated as "a valid model is loaded" (a real bug,
             surfacing much later as a confusing torch.load() permission error)
cellpose_model_path ← saved config only — no bundled default (project-specific,
             not shipped with the plugin)
self._state ← {model_path, cellpose_model_path, last_file_path, metadata, config}
self._skin_hidden_state ← None   # see _on_toggle_skin_visibility, §5

_build_ui()
_connect_signals()             # also fires the initial-visibility calls and
                                # _resume_monai_job_if_active() / _resume_
                                # cellpose_job_if_active() (§7's "reconnect
                                # on reopen" mechanism)
_refresh_layer_info()
_refresh_stats_layers()
_start_recovery_timer()
```

### `_build_ui()` — organization, not algorithm

One long method building six `QTabWidget` pages in order (Skin Remover / Create MG Labels / Edit MG Labels / Statistics / AI Tools / Sweeps & Utilities), each wrapped in `_wrap_scroll()` and built from `_make_collapsible()` group boxes. There's no real control flow here worth pseudocode — every group follows the same shape: a `QGroupBox`, a `QVBoxLayout`, a slider+`_add_reliable_spinbox()` pair per numeric parameter, a status label, a run button. The one thing worth naming explicitly is the **Common Settings** group (Tab 2): Min volume and Min hole size are deliberately *read-only labels*, not editable sliders — both are empirical GT-measured facts (§8's `min_volume_from_gt`/`min_hole_size_from_gt`), not tunable knobs a user should be able to accidentally drift away from a measured value.

### Config persistence

```
_load_config() / _save_config(data):  plain JSON read/write, empty dict / no-op
                                        on any failure — never raises, so a
                                        corrupted or missing config file never
                                        blocks the plugin from opening
_save_cfg(self, **kwargs):  self._state["config"].update(kwargs); _save_config(...)
                              — the one method every handler actually calls;
                                merges rather than replaces, so one handler's
                                save never clobbers another's unrelated keys
```

### Cross-fish GT-sweep history (`_update_gt_history`, `_update_merge_stats_history`)

The mechanism behind every "Recommended: X" value quoted throughout §3–§8 — a real per-fish history, not a single overwritten scalar (an earlier design either blindly overwrote the target value with whatever one run found, or tracked a single running scalar with no way to revise a re-swept fish's own contribution):
```
_update_gt_history(config_key, fish_key, value, mode):
    history ← config["<config_key>_history"]           # {fish_key: value}
    history[fish_key] ← value                            # update in place if
                                                            # this fish was
                                                            # already swept
    save history back to config
    return  min(history.values())  if mode == "min"       # never-rising floor:
                                                            # min_volume, min_
                                                            # hole_size, gt_min
            max(history.values())  if mode == "max"       # never-falling ceiling:
                                                            # branch_radius
            mean(history.values()) otherwise               # BG Threshold, Erosion,
                                                            # Sigma XY/Z, Cellprob,
                                                            # Large-contact, MONAI
                                                            # Threshold — each
                                                            # fish's own local
                                                            # optimum, no safe
                                                            # direction to bias
```
`_update_merge_stats_history(fish_key, merge_stats)` is the one exception, storing a whole *sample-list dict* per fish (not a scalar) and re-running `recommend_merge_params()` (§8) over the freshly-pooled set from every fish calibrated so far, rather than averaging a per-fish scalar.

### Layer resolution

Three different "which layer" resolvers, each deliberately scoped to a different part of the plugin (see §1's "Layer-selection philosophy"):
- **`_active_layer()`**: active selection if it's an `Image` layer, else the topmost `Image` layer, else `None` — the plugin's original, still-used-for-Tab-1 fallback.
- **`_active_labels_layer()`**: same pattern for `Labels` layers — used by a few route-agnostic tools (Split/Join/Save) that predate the Edit MG Labels tab's shared selector.
- **`_edit_labels_layer()`**: reads the *explicit* combo the user set in Edit MG Labels' own "Layers and label(s) being edited" section — every Stage-3 tool (§5) uses this one, never a fallback, so the layer being edited is never guessed.

`_get_layer_scale()`: metadata from the last Open-button load, else the active layer's own napari `scale`, else `(1,1,1)` — the one function nearly every physical-unit calculation in the plugin ultimately calls. `_output_dir()`: `<original file's parent>/<original file's stem>`, created on first use — the one folder convention every saved output file in the plugin follows (and that §8's `_gt_toolkit.py` reverse-engineers to auto-discover a fish's files). `_current_min_volume()` / `_lo_sweep_sample_params()`: thin config readers so every route/sweep launcher shares one source of truth instead of each hardcoding its own possibly-diverging copy.

### The hang watchdog

```
_hang_watch_start(what, after_s=600):
    faulthandler.dump_traceback_later(after_s, repeat=True, file=sys.stderr)
    # if the operation is STILL running after after_s seconds (and every
    # after_s after that), every thread's real stack prints to the
    # terminal napari was launched from — turns "is it stuck?" into
    # "here's exactly which line it's on", instead of only the last
    # status message (which can stay unchanged through several later steps)
_hang_watch_stop():
    faulthandler.cancel_dump_traceback_later()
```
Wrapped around Protect Skin as Label, Auto-correct, and the Cellpose-SAM-chained auto-correct stage — the slowest operations in the plugin, extended to 30 minutes (from an original 10) after real full-fish runs were found to legitimately take that long.

### Email notification

Two independent paths sharing one credential source (`_get_notify_creds()`, reading Tab 6's shared SMTP fields and persisting the password via `_secrets.py`, never plaintext):
- **In-process** (`_maybe_send_notify(checkbox, subject, body)`): called from a background *worker* thread right before it finishes (not the GUI/poll thread, so the SMTP round-trip never blocks the UI) — used by Tab 1 Run, Tab 2 Cellpose-SAM Segmentation, and Tab 6's Cellprob/Large-contact and Best-Epoch sweeps, none of which survive napari closing anyway.
- **Detached** (`_build_notify_cfg()` → `_training_jobs.launch_detached(..., notify=...)`, §7): used only by Tab 5's two training launchers, which must keep running (and therefore be able to notify) even after napari itself has closed — hence the standalone supervisor-script approach instead of this same in-process function.
Both paths swallow their own errors into a `print()` rather than raising — a broken email config must never take down the operation whose completion it was reporting.

### The 10-minute recovery timer

`_start_recovery_timer()` — the one `QTimer` in the plugin meant to live for the whole session (every other background-work `QTimer` calls `deleteLater()` once its job finishes; a real historical OOM leak came from `QTimer`s that didn't, keeping their connected closures — often a full label-array copy — pinned in memory for the rest of the session). Every 10 minutes, saves every current `Labels` layer to `<output_dir>/<layer_name>_recovery.tif` in a background thread, overwriting the previous save — a crash loses at most ~10 minutes of manual label editing, not the whole session.

### `_refresh_*` family — keeping the UI in sync with the viewer

Each wired to `viewer.layers.events.inserted`/`.removed` (and, for the active-selection-dependent ones, `.selection.events.changed`), so the UI never goes stale as layers are added/removed/renamed elsewhere:
- **`_refresh_layer_info()`** / **`_update_labels_section_visibility()`**: shows/hides the Pixel Classifier vs. Cellpose-SAM Segmentation groups based on whether a matching `_NoBG`/`_ExtRm`-suffixed Image layer *exists anywhere in the viewer* — not just whether it's the currently active layer (a real historical bug: the old active-layer-only check could hide a route's tools the moment the user selected the Labels layer itself, exactly what "Use selected" requires).
- **`_refresh_stats_layers()`**: repopulates Tab 4's Image/Shapes combos (§6).
- **`_refresh_gt_layers()`**: repopulates Tab 5's GT Annotation Image combo (§7).
- **`_refresh_meta_lbl()`**: updates the Z/Y/X µm + anisotropy display under Tab 1's Open button.

### Handler index — every `_widget.py` function, by home section

Every button/combo handler in the file is documented, with real behavior, in the pipeline-stage section for the tab it belongs to — listed here so nothing in the file is left unaccounted for:

| Tab | Section | Representative handlers |
|---|---|---|
| 1 — Skin Remover | §3 | `_on_open`, `_on_load_labels`, `_on_run`, `_on_browse_model`, `_on_bg_mode_changed` |
| 2 — Create MG Labels | §4 | `_on_create_labels`, `_on_run_cellpose_seg`, `_on_rerun_single_cell`, `_resolve_rerun_layers`, `_on_browse_cp_model`, `_run_auto_correction_stage`, `_run_sanding_stage` |
| 3 — Edit MG Labels | §5 | `_on_resort_labels`, `_on_remove_debris`, `_on_split_label`, `_on_join_labels`, `_on_correct_label`, `_on_correct_adjacent_labels`, `_on_copy_label_to_adjacent_slice`, `_on_protect_skin`, `_on_remove_skin_label`, `_on_toggle_skin_visibility`, `_on_autocorrect_labels`, `_on_save_labels`, `_on_run_contrast_sweep`, `_on_toggle_drift`, `_on_use_selected_label_a/b/rerun`, `_on_correct_label_id_changed` |
| 4 — Statistics | §6 | `_on_generate_stats`, `_on_stats_backend_changed`, `_on_gtscore_run` |
| 5 — AI Tools | §7 | `_on_gt_interpolate`, `_on_gt_generate_masks`, `_on_gt_image_changed`, `_on_prepare_monai_data`, `_on_mt_launch_training`, `_on_mt_stop_training`, `_resume_monai_job_if_active`, `_start_monai_polling`, `_on_ct_launch_training`, `_on_ct_stop_training`, `_resume_cellpose_job_if_active`, `_start_cellpose_polling`, `_on_ct_calib_run` (branch_radius), `_on_es_run_sweep` (epoch sweep), `_on_xz_run` (XZYZ patches), `_write_cellpose_best_pointer` |
| 6 — Sweeps & Utilities | §8 | `_on_gtk_scan`/`_on_gtk_run` (GT Toolkit), `_on_bs_run_sweep` (brain/MONAI), `_on_ps_run_sweep` (BG Threshold/Erosion), `_on_sg_run_sweep` (Sigma), `_on_kr_run_sweep` (Cellprob/Large-contact), `_on_gtp_run` (GT package) |
| Shell (this section) | §9 | `__init__`, `_build_ui`, `_connect_signals`, `_status`, `_save_cfg`, `_update_gt_history`, `_update_merge_stats_history`, `_hang_watch_start/stop`, `_get_notify_creds`, `_build_notify_cfg`, `_maybe_send_notify`, `_on_send_test_email`, `_start_recovery_timer`, `_output_dir`, `_get_layer_scale`, `_active_layer`, `_active_labels_layer`, `_edit_labels_layer`, `_refresh_*`, `_current_min_volume`, `_lo_sweep_sample_params`, `_on_ai_tools_mode_changed`, `_update_labels_section_visibility` |

---

## 10. Module reference (quick index)

| File | Lines | Purpose | Guide section |
|---|---|---|---|
| `_widget.py` | 10,810 | Application shell — all 6 tabs, every button handler, config, layer resolution | §3–§9 |
| `_labeling.py` | 2,883 | Pixel Classifier connected-components engine + every label-editing primitive (resort, debris, split, join, correct, copy, skin protection) | §4, §5 |
| `_auto_correction.py` | 993 | 6-step unattended per-cell auto-correction pipeline, chained after Cellpose-SAM Segmentation | §5 |
| `_cellpose_seg.py` | 808 | Cellpose-SAM `do_3D` inference + GMM cleanup + Krendl safe-merge + large-contact merge + final safety net | §4 |
| `_statistics.py` | 764 | Per-label morphology/spatial/intensity statistics + natural-language descriptions | §6 |
| `_krendl_sweep.py` | 719 | Cellprob x Large-contact GT sweep + real-prediction merge-parameter calibration | §8 |
| `_pixel_sweep.py` | 609 | BG Threshold x Signal Erosion / Smooth σXY x σZ GT sweeps + GT-measured floors (`min_volume_from_gt`, `min_hole_size_from_gt`, `min_intercell_gap_um`) | §8 |
| `_io.py` | 284 | Metadata-aware `.tif`/`.ims` loading, voxel-scale resolution | §2 |
| `_grow_correct.py` | 493 | Auto-grow / until-stable orchestration around Correct Label's 2D and 3D engines | §5 |
| `_training_jobs.py` | 378 | Cross-platform detached-process training job management (launch, poll, kill, patience early-stop, email) | §7 |
| `_gt_annotation.py` | 322 | Polygon-based hand-drawn GT annotation (brain/skin masks) | §7 |
| `_epoch_sweep.py` | 275 | GT-verified Cellpose-SAM checkpoint (epoch) sweep, bbox-restricted | §8 |
| `_contrast_sweep.py` | 275 | Self-referential Correct-Label contrast (`lo`) calibration against Cellpose-SAM's own output | §5 |
| `_background.py` | 193 | Population-mode background estimation + 3 background-removal modes | §3 |
| `_xzyz_patches.py` | 187 | Orientation-slice (XY/XZ/YZ) 2D training-crop generation, the current Cellpose-SAM training methodology | §7 |
| `_gt_toolkit.py` | 180 | Single-folder auto-discovery of a fish's files + step preconditions ("GT Toolkit Tuning Tool") | §8 |
| `_gt_package.py` | 167 | Zips a GT-correction package (source image, correction starting point, guide, stats CSV) for external manual review | §8 |
| `_brain_sweep.py` | 165 | GT-verified MONAI Threshold x Erosion sweep for the brain mask | §8 |
| `_secrets.py` | 188 | Two-tier (OS keyring → local Fernet-encrypted file) secret storage | §2 |
| `_crop_truncation.py` | 146 | Zeros out incidentally-truncated neighbor labels in training crops | §7 |
| `_branch_calibration.py` | 134 | Measures real branch radius from GT to recalibrate the branch-weighted training loss | §7 |
| `_gt_score.py` | 126 | Whole-fish Hungarian-matched TP/FP/FN/Score/IoU/Dice scoring — the shared methodology every sweep builds on | §8 |
| `_live_progress.py` | 116 | Captures MONAI's/Cellpose's own otherwise-invisible progress output into the GUI log | §2 |
| `_sieve.py` | 105 | Generic coarse-to-fine sweep-narrowing engine, shared by every continuous-axis GT sweep | §2 |
| `_ai_tools.py` | 89 | argv builders + default script-path resolution for the 3 training scripts | §7 |
| `_sanding.py` | 134 | Sigma-softening (contour polishing) pass, chained after any label-regenerating correction | §4 |
| `_reader.py` | 58 | napari File-menu integration (`.tif`/`.ims` reader registration) | §2 |
| `_gpu_check.py` | 55 | GPU/VRAM classification (informational banner, never a hard gate) | §2 |
| `__main__.py` | 53 | CLI entry point / pre-loading a file at launch | — |
| `_inference.py` | 169 | MONAI 3D U-Net sliding-window brain inference | §3 |
| `__init__.py` | 22 | Package version + widget export | — |

*(`training_scripts/` — `prepare_data.py`, `train.py`, `train_xzyz.py` — ships inside the package as launchable subprocess scripts, not imported; see `_ai_tools.py` and §7.)*

---

*End of Software Guide.*
