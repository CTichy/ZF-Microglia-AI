# ZF-Microglia-AI

A [napari](https://napari.org) plugin for automated 3D brain extraction and AI-assisted microglia segmentation and analysis from *Danio rerio* (zebrafish) confocal stacks.

Developed at **FH Technikum Wien** — Artificial Intelligence & Data Science.

**Documentation:** this README (quick reference) · [GUIDE.md](docs/GUIDE.md) (full user guide, [GUIDE.pdf](docs/GUIDE.pdf) also available) · [STATISTICS_GUIDE.md](docs/STATISTICS_GUIDE.md) (Tab 4 statistics — the algorithm/formula behind every CSV column, [STATISTICS_GUIDE.pdf](docs/STATISTICS_GUIDE.pdf) also available) · [SOFTWARE_GUIDE.md](docs/SOFTWARE_GUIDE.md) (developer reference — every module and function explained in pseudocode, [SOFTWARE_GUIDE.pdf](docs/SOFTWARE_GUIDE.pdf) also available) · [GROUND_TRUTH_CREATION_GUIDE.md](docs/GROUND_TRUTH_CREATION_GUIDE.md) (how to correct a GT-Correction Package in napari; shipped inside every package). All guides live in [`docs/`](docs/).

---

## What it does

Given a 3D confocal volume (TIF or IMS), the plugin provides six tabs:

- **Tab 1 — Skin Remover:** runs a trained MONAI U-Net to predict the brain mask, removes the skin, and saves `brain_mask.tif` + `brain_only.tif`
- **Tab 2 — Create MG Labels:** two ways to detect and label individual microglia in 3D. **Cellpose-SAM Segmentation** (`do_3D` inference → 3-component GMM cleanup → Krendl safe-merge → large-contact merge → final min-size safety net) is the recommended path — a fine-tuned foundation model that handles branching/overlapping cells far better than classical thresholding. The **Pixel Classifier** (3D Gaussian smooth → threshold → true 3D connected components → volume filter) is a lighter, older-technology fallback for machines with no GPU at all. Each section appears automatically once a layer with its matching background-removal suffix is open in the viewer (see below) — no manual switching needed. Only creation tools live here, plus the checkboxes that chain auto-correction and sanding onto a fresh Cellpose-SAM run.
- **Tab 3 — Edit MG Labels:** every edit/correct/modify tool for an existing Labels layer (Resort, Remove Debris, Split, Join, Correct Label, Copy to Adjacent Slice, Correct Adjacent Labels, Protect Skin as Label, Auto-correct Labels, Save Labels), all reading their layers and label ID(s) from **one shared selector** at the top of the tab instead of each tool having its own copy.
- **Tab 4 — Statistics:** computes up to 52 CSV columns per labelled cell (morphological, spatial, intensity and region features, plus a description) and exports a CSV. Always visible — shows an explanatory hint in place of the controls until at least one Labels layer exists, the same "explain instead of hide" pattern as Tab 2's method sections.
- **Tab 5 — AI Tools:** ground-truth polygon annotation plus launchers for MONAI and Cellpose-SAM training (dataset prep/crop extraction, hours-to-days training runs that survive napari closing). Always available — shows a disclaimer banner instead of hiding if your GPU is missing or under the recommended 8GB — see below.
- **Tab 6 — Sweeps & Utilities:** the **GT Toolkit Tuning Tool** (one button that runs every applicable GT sweep for one fish), every GT-verification sweep tool (MONAI segmentation, Pixel Classifier background/smoothing, Cellpose-SAM labelling, Cellpose-SAM checkpoint selection), the Correct-Label contrast calibration, two GT utilities (Score Against GT, Build GT-Correction Package) and the session utilities (Autosave, Email notification, Drift View). Each tool still operates on its own tab's parameters — this tab doesn't introduce a separate workflow, it just keeps the day-to-day tabs focused on running the pipeline rather than tuning it.

---

## Environment setup (first time)

### 1. Clone the repository

```bash
git clone https://github.com/CTichy/ZF-Microglia-AI.git
cd ZF-Microglia-AI
```

### 2. Create the environment

**Windows / Linux (CUDA GPU):**
```bash
conda env create -f environment.yml
```

**Mac (CPU / Apple MPS):**
```bash
conda env create -f environment-mac.yml
```

### 3. Activate and launch

```bash
conda activate zf-microglia-ai
napari
```

Then: **Plugins → ZF-Microglia-ToolKit (ZF-Microglia-AI)**

---

## Updating (subsequent runs)

```bash
cd ZF-Microglia-AI
git pull
conda env update --name zf-microglia-ai -f environment.yml --prune   # or environment-mac.yml on Mac
```

> **Windows / Linux (CUDA GPU):** after every `conda env update`, restore the correct torch:
> ```bash
> pip install "torch==2.7.0+cu126" "torchvision==0.22.0+cu126" \
>   --index-url https://download.pytorch.org/whl/cu126
> ```
> (run this with the `zf-microglia-ai` env active — `conda env update` ignores `--extra-index-url` in environment.yml and resets torch to the wrong version. Not needed on Mac, which doesn't use CUDA torch.)

---

## Developing the plugin (editable install)

The steps above install a fixed snapshot from GitHub — fine for using the plugin, but source edits won't take effect until you reinstall. For active development, install the local clone in editable mode instead:

```bash
cd ZF-Microglia-AI
conda activate zf-microglia-ai
pip install -e .
```

Source edits now take effect the next time napari launches — no reinstall needed. Verify it's picking up the local clone (not a stale `site-packages` copy):

```bash
python -c "import napari_zf_microglia_ai; print(napari_zf_microglia_ai.__file__)"
```

This should print a path inside your cloned `ZF-Microglia-AI/` folder, not inside `site-packages`.

---

## Model files

This plugin needs **two** trained checkpoints — neither is bundled in the repo. Quick setup:

```
YourPluginFolder/            <- anywhere convenient, e.g. next to your cloned repo
├── models/
│   ├── MONAI/
│   │   └── best_model_fullstack.pth
│   └── Cellpose/
│       └── <your checkpoint>
```

This exact layout isn't required — the plugin remembers whatever path you browse to via the **Browse (...)** buttons in Tabs 1/2, so any location works. `models/MONAI/` and `models/Cellpose/` is just a tidy default if you'd rather not decide.

### MONAI skin-removal model (required, Tab 1)

A trained `.pth` checkpoint — **not included in this repo** (~220 MB).

1. Download [best_model_fullstack.pth](https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY).
2. Move it into `models/MONAI/` (or anywhere else you prefer).
3. In the plugin, Tab 1 → **Browse (...)** → select the file. The path is remembered across sessions.

### Cellpose-SAM checkpoint (recommended, Tab 2)

Needed for **Cellpose-SAM Segmentation**, the recommended labelling method (see above). This is a project-specific fine-tuned Cellpose-SAM model (~580 MB), branch-weighted 3-fish checkpoint (`multi3_bw`, epoch 150).

1. Download [cpsam_microglia_512_multi3_bw_epoch_0150](https://cloud.technikum-wien.at/s/eFBJepk9DakDxyb).
2. Move it into `models/Cellpose/`.
3. In the plugin, Tab 2 → Cellpose-SAM Segmentation section → **Browse (...)** → select the file. The path is remembered across sessions.

If your machine has no GPU at all, use the **Pixel Classifier** instead — it needs no additional model file and no GPU, but is a lighter, older-technology approach; only fall back to it when a GPU genuinely isn't available.

### Training scripts (Tab 5 — AI Tools only)

Tab 5 launches three project-specific research scripts (`prepare_data.py`, `train.py`, `train_xzyz.py`) as separate subprocesses. They ship with the plugin — bundled under `napari_zf_microglia_ai/training_scripts/` and installed as package data — so they're present for every install method (plain `git+https://` install or editable). If you want to point at a locally modified copy instead, the config file (`~/.config/napari-zf-microglia-ai/config.json`) has `monai_prepare_script_path`/`monai_train_script_path`/`cellpose_train_script_path` keys to override.

---

## Tab 1 — Skin Remover

### Workflow

1. **Open a file** — click "Open TIF / IMS file". All channels load as separate layers. **Load Labels layer (.tif)**, right below it, loads a saved labels file directly as a Labels layer instead, named `<active Image layer>_labels` (same convention Create Labels uses) and scaled to match whatever stack is already open.
2. **Select the channel** to process by clicking its layer in the Layers panel.
3. **Browse to the model** `.pth` file if not auto-detected.
4. **Adjust MONAI Threshold** (default 0.38) and **MONAI Erosion** (default 0).
5. **Choose Background mode** — pick **Option 1 (Remove outside brain only, `_ExtRm`)** if you plan to segment with **Cellpose-SAM** in Tab 2, or **Option 2 (Remove globally, `_NoBG`)** if you plan to use the **Pixel Classifier**. Tab 2 auto-detects which one you produced and shows the matching tool.
6. Click **Run Skin-Remover**.

All numeric sliders in this plugin are directly editable — click the number box next to any slider and type an exact value instead of dragging.

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| MONAI Threshold | 0.38 | Sigmoid cutoff. A read-only "Recommended MONAI Threshold" line under the slider shows the GT-verified recommendation once one has been measured. |
| MONAI Erosion | 0 vox | Strips skin rim from `brain_only`. `brain_mask` always saved un-eroded. Every Background mode uses the eroded mask. |
| Background mode | Off | 1 for Cellpose-SAM, 2 for Pixel Classifier (see Tab 2), 3 = random-noise fill (presentation only) |
| BG Threshold | 1.025 | Only active in Background mode 2. Percent of the stack's data range added above the in-brain background mode. |
| Signal Erosion | 0 vox | Only shown in Background mode 2. Erodes the signal that survives BG Threshold, per Z slice. |

**Verify MONAI Threshold / Erosion (GT Sweep)** — in **Tab 6 — Sweeps & Utilities**; scores Threshold/Erosion against a hand-corrected GT brain mask (applied to the sliders above only when run through the GT Toolkit Tuning Tool with GT verified).

### Output files

Saved in `<source_folder>/<source_stem>/`:

| File | Content | Feeds into (Tab 2) |
|------|---------|---------------------|
| `*_brain_only.tif` | Volume with everything outside the brain zeroed | — |
| `*_brain_only_ExtRm.tif` | Background removed outside the brain only (Mode 1) | Cellpose-SAM Segmentation |
| `*_brain_only_NoBG.tif` | Background also removed inside the brain (Mode 2) | Pixel Classifier |
| `*_brain_mask.tif` | Binary mask (0/255 uint8), un-eroded | — |
| `*_original.tif` | The raw, unmasked input channel — saved whenever Background mode is not Off | Skin protection / auto-correct (the raw signal) |

`*` is the selected layer's name (e.g. `<stem>_ch1`); the folder is named after the opened file.

---

## Tab 2 — Create MG Labels

Each section is shown whenever a layer with its matching suffix exists anywhere in the viewer (not only the selected one), so both can be shown at once:

| An Image layer ending in … is open | Section shown |
|---|---|
| `_ExtRm` | Cellpose-SAM Segmentation |
| `_NoBG` | Pixel Classifier |
| both | both |
| neither (e.g. only `_RndFill` or the raw channel) | neither, with a hint on what to select |

The **Common Settings** box above both sections is always visible. Only creation tools live here — every edit/correct/modify tool (Resort Labels, Remove Debris, Split, Join, Correct Label, Copy to Adjacent Slice, Correct Adjacent Labels, Protect Skin as Label, Auto-correct Labels, Save Labels) is in **Tab 3 — Edit MG Labels**, below. **Tab 4 — Statistics** stays visible regardless, showing an explanatory hint instead of its controls until at least one Labels layer exists in the viewer.

### Common Settings (both routes)

| Field | Default | Notes |
|-------|---------|-------|
| Min volume (vox) | 7069 | **Read-only.** The smallest real cell volume confirmed by GT; only the GT Toolkit Tuning Tool (Tab 6) lowers it. |
| Max volume (vox) | not yet measured | **Read-only.** The largest real cell confirmed by GT; only used for the `is_volume_outlier` CSV column. |
| Min hole size (vox) | 5 | Enclosed gaps smaller than this are filled (per Z slice). |
| Final min-size fraction | 0.618 | The real deletion cutoff is this × Min volume. |
| Soften label contours (sanding) after any label correction | on | Sigma XY/Z sliders live in Tab 3. |
| Fill interior cavities (vesicles) before sanding | on | |

### Pixel Classifier (on screen: "Pixel Classifier — 3D Connected-Component Labels")

Runs on the selected Image layer (else the topmost): 3D Gaussian smooth of the `volume > 0` mask → re-threshold at 0.5 → per-slice hole-fill (Min hole size) → true 3D connected components (26-connectivity) → volume filter (`Final min-size fraction × Min volume`) → renumber 1…N, largest first. Dispatches to CUDA, Apple MPS (threaded CPU), or plain threaded CPU automatically. Needs a `_NoBG` layer (background removed everywhere, not just outside the brain). Output layer: `<layer>_labels`.

| Parameter | Default |
|-----------|---------|
| Smooth σ XY | 2.6 |
| Smooth σ Z | 1.0 |

**Verify BG Threshold / Signal Erosion (GT Sweep)** and **Verify Smooth σ XY / σ Z (GT Sweep)** — in **Tab 6 — Sweeps & Utilities**; both also measure Min volume and Min hole size from GT.

### Cellpose-SAM Segmentation

Runs `do_3D` inference with a Cellpose-SAM checkpoint on the layer picked in its own **Signal layer** field (default: the first `_ExtRm` layer), then 3-component-GMM cleanup, a Krendl safe-merge pass (only fragments below Min volume, gap/contact-based), a large-contact merge pass (catches blobs split through a thick junction) and the final min-size safety net. `do_3D` inference is slow — can take hours for a full-size fish — and runs in a background thread so the UI stays responsive. Output layer: `<signal layer>_cellpose_labels`; always saved as `<signal layer>_cp.tif` (raw, before GMM/merges) and `<signal layer>_cp_krendl.tif` (final). The network output is cached to `<signal layer>_cp_flow_cache.npz` while masks are formed, so a crash there can resume without re-running the network; the cache is deleted once mask formation succeeds.

Requires a **Cellpose-SAM checkpoint** (see "Model files" above). The browse dialog accepts any file, and the plugin refuses one that isn't a PyTorch checkpoint (zip archive). The path is remembered across sessions (like the MONAI model path).

| Parameter | Default |
|-----------|---------|
| Min size (vox) | 15 (early noise filter, not shared with Min volume) |
| Flow iterations (niter) | 200 |
| Cellprob threshold | 2.84 |
| Safe-merge max gap (µm) | 3.52 |
| Safe-merge min contact (vox) | 1 |
| Large-contact merge (vox) | 20 |

Flow threshold is not a field in this plugin — Cellpose only applies its flow-error filter outside `do_3D` mode, which this plugin always uses, so the parameter has no effect here and isn't exposed.

**Auto-correct labels via contrast sweep after segmentation** (checkbox, on by default) — requires the **raw channel layer** (the signal layer's name without its `_brain_only…` suffix) to be open; the run refuses to start otherwise, and every step below reads the raw channel, not `_ExtRm`. After a run finishes: calibrates the contrast threshold using the same **Auto-sweep**/**Reduce best_lo by** controls as Tab 3 — Edit MG Labels' own Contrast low (best_lo) calibration section (physically in Tab 3, read directly regardless of which tab is visible), then protects skin as its own label — always ID **-1** — at that threshold (or **reuses it if the labels already carry skin** — it never re-protects, and it stops with an error rather than correct any cell without skin) (Protect Skin as Label, using the **Brain mask layer** field below the checkbox only to *seed* skin, never to clamp its result — auto-filled from `<stem>_brain_mask`, since Tab 1 already creates and loads it alongside every `_ExtRm`/`_NoBG`/`_RndFill` layer — with the same Correction Settings (Bbox padding/auto-grow/until-stable) and joint-adjacent-resolution behavior Protect Skin as Label itself has, adapting to whatever the still-raw cells currently look like without ever modifying them), **removes debris skin just absorbed** right there (this sweeps up any small stray blob by size alone before any real cell's own turn), then resorts every cell by Centroid Z and figures out which cells **touch skin**, measured from skin's final geometry (a cell counts if, on any slice, it is within **3 px** of skin). Each cell is then corrected at that threshold, mode chosen by whether it touches skin: a **touching** cell is corrected in **2D, slice by slice, jointly against skin** (skin's own per-slice engine, writing back the cell's own side this time — skin can only ever be corrected in 2D, so a cell meeting it has to do so on skin's own terms); every other cell gets the same auto-grow + until-stable **3D** engine as Correct Label, reading Tab 3's shared **Correction Settings** (Bbox padding/Auto-grow/Growth step/Max growth iterations/Keep-until-stable/Max stability passes — the same values Correct Label, Correct Adjacent Labels, and Protect Skin as Label use, read live at the moment this stage actually starts, not when the segmentation run was launched) — both modes batched into the same parallel-safe waves (up to 75% of CPU cores) instead of strictly one cell at a time, since two cells whose maximum-possible working areas can never overlap can't interact and are corrected concurrently regardless of mode. Then runs Remove Debris once more over the whole result and screens for **possible non-microglia blobs** (≥ 40% of a cell's volume outside the brain mask, or of its surface within 3 px of skin — macrophages or leftover skin; **report only**, nothing is removed). Calibration and Centroid-Z resorting count microglia only (skin is ignored), and the CPU budget is shared between cells and slices, never multiplied. Produces one consolidated report covering every cell (not one report per cell). Required brain mask field means the run refuses to start if it's missing, checked before the segmentation itself begins. On completion, the raw channel's and the segmented `_ExtRm` layer's contrast limits move to the calibrated threshold (low) and an adaptive high limit measured from this fish's own real signal (the 99th percentile of intensity among the corrected labels, floored at threshold + 50) so the display shows real signal's own dynamic range instead of whatever unrelated ceiling the layer came in with, and Tab 3's Protect Skin as Label section (touching/nearby-labels report, status) is filled in as if that button had been clicked directly. The result is saved as `<stem>_cp_krendl_ac.tif` and the report as `<stem>_autocorrect_report.txt`. If **Soften label contours (sanding)** (Tab 2 Common Settings) is also on, a further stage chains onto this one afterward: every corrected cell first has any interior cavity/vesicle filled in (if that checkbox is also on), then its contour is softened — saved as `<stem>_cp_krendl_ac_snd.tif`, with the sanding report appended to the same `.txt`.

**Calibrate Cellprob + Krendl Merge Parameters + Min size + Large-contact (GT Sweep)** and **Build GT-Correction Package** — both in **Tab 6 — Sweeps & Utilities**.

**Re-run This Cell Only** — fixes one label without redoing the whole fish: crops to that label's own bounding box (+15 Z / +40 XY voxels), re-runs `do_3D` + the same GMM/Krendl/large-contact/final-min-size cleanup on just the crop, then replaces the old label with the crop result. Only crop pieces that overlap the original label survive the splice, each as a new label ID after the current maximum, and they are written only into background: a neighbouring label or skin inside the crop is never overwritten (the status line reports how many voxels were left to a neighbour). Works on the selected labels layer (any name), or finds `<image>_cellpose_labels` / `<image>_labels` (or a single same-shape labels layer) for the selected volume; asks instead of guessing if several qualify.

---

## Tab 3 — Edit MG Labels

Every edit/correct/modify tool for an existing Labels layer, gathered here instead of scattered through Tab 2. **One shared selector at the top of the tab** — Signal layer / Labels layer / Brain mask layer, plus **Label A** (single-label tools) and **Label B** (the two tools that need a second label) — drives every tool below; pick your layers and label(s) once, not per tool. Label A/B take a positive cell ID or **-1** (skin); a tool refuses to run on 0 or any other negative ID. Skin operations (Protect/Remove/Hide Skin Label) always act on the fixed sentinel ID **-1**, never on Label A/B. **Signal layer should be the raw, un-processed channel** (not `_ExtRm`/`_NoBG`) — it is what the contrast sweep and every correction read, and the source of skin protection's real outside-brain signal; a background-processed layer has already been zeroed everywhere outside the brain (skin protection then looks for the raw channel behind it). All three layer combos **auto-populate together** the moment a matching `_brain_only_*` layer exists (e.g. right after Tab 1 runs) — Signal to the raw channel (the layer named `<stem>`), Brain mask to `<stem>_brain_mask`, Labels to `<brain_only layer>_cellpose_labels` or `_labels` (else the selected/topmost Labels layer) — only filling a combo that is still empty. The tools below the selector only appear once at least one Labels layer exists.

On screen, top to bottom: the selector; Split Label, Join Labels, Copy Label to Adjacent Slice; Resort Labels, Remove Debris; Correction Settings, Sanding, Contrast low (best_lo) calibration; Correct Label, Correct Adjacent Labels, Protect Skin as Label (+ Remove Skin Label, Hide skin label), Auto-correct Labels; Save Labels.

- **Correction Settings (Bbox / Auto-grow / Stability)** — Bbox padding (default 15px), Auto-grow (default on) + Growth step (default 5px) + Max growth iterations (default 20), and Keep re-running until the shape stabilizes (default on) + Max stability passes (default 50): one shared set of controls for **Correct Label**, **Correct Adjacent Labels**, **Protect Skin as Label**, and **Auto-correct Labels** (standalone or chained onto Cellpose-SAM Segmentation) — changing a value here changes it for all four at once, read live at the moment each one's own computation actually starts. (Correct Label's 2D mode uses Auto-grow but not Keep-until-stable.)
- **Sanding sigma XY / Z** (default 0.7/0.7 vox) — the sigma sliders for the "Soften label contours (sanding)" checkbox (Tab 2 Common Settings) live here, not there; that checkbox stays the only sanding control in Tab 2.
- **Fill interior cavities (vesicles) before sanding** (Tab 2 Common Settings, default on) — runs right before the blur: a real intracellular vesicle/vacuole shows up dark in the fluorescence channel, so a plain intensity threshold correctly excludes it as "not signal" even though it's genuine cell material, leaving a literal 3D-enclosed cavity. This fills it back in, fully in 3D (not per Z/Y/X slice, not a convex hull — a cell's own concave/branching shape is left exactly as it is; only background voxels fully enclosed by the label are ever touched). Cells only, never the skin sentinel — skin's own "cavities" are real cells sitting inside the brain, not vesicles.
- **Contrast low (best_lo) calibration** — **Auto-sweep for best contrast low (best_lo)** (default on) + **Reduce best_lo by** (default 2, range 0-6): the calibration Protect Skin as Label and Auto-correct Labels (both entry points) use. With Auto-sweep on, sweeps for best_lo then subtracts the reduction; with it off, uses the Signal layer's own current contrast low directly (no sweep). The sweep's sample selection — Cells, Slices/cell, Edge margin, Sweep steps — is read from **Calibrate Correct-Label Contrast** (Tab 6), including its Bbox padding for the sample crops.
- **Resort Labels** — renumber 1…N by size, centroid Z/Y/X, or complexity (skeleton branch count); optional reverse order
- **Remove Debris** — sweeps the selected layer for anything below `Final min-size fraction × Min volume`, per connected fragment (not per label ID); sweeps skin (-1) too, whenever present
- **Split Label** — watershed split of Label A into N parts: 3D (whole label, cut at the mask's thinnest neck) or 2D (current slice only, cut along the dimmest signal ridge)
- **Join Labels** — the inverse of Split: merges Label B into Label A
- **Correct Label** — regenerates Label A's shape (2D current slice, or 3D whole cell) from the shared Signal layer's own live contrast window (signal = intensity at/above the lower contrast limit); Label A is a positive ID or **-1** for skin, though **3D is disabled for skin specifically** (a 3D correction walks the WHOLE outside-brain territory slice by slice — real computation cost for no benefit over Protect Skin as Label's own dedicated, far cheaper trim; 2D skin correction still works normally) — 3D walks the label's own Z range mid-outward (not plain Z order), topping up each slice's watershed seed with whichever already-corrected or original neighboring slice(s) are in range before correcting it, and jointly resolves against a real neighboring label wherever one's actually nearby — Bbox padding/Auto-grow/Growth step/Max growth iterations/Keep-until-stable/Max stability passes all come from the shared **Correction Settings** above: Auto-grow retries with a bigger padded area if the result touches its own edge, auto-folding in any neighboring label the growing area reaches instead of encroaching on it; Keep-until-stable (3D only) keeps re-running each slice, reseeded from its own previous result, until it stops changing
- **Copy Label to Adjacent Slice** — copies Label A's shape from the current slice onto the next/previous slice, e.g. to patch a broken/missing cross-section
- **Correct Adjacent Labels** — jointly regenerates Label A and Label B on the current slice (working area scoped to Label A alone), cut placed by watershed seeded at each label's own existing footprint — same shared Correction Settings as Correct Label, seeded with both labels from the start, plus Keep-until-stable: keeps re-running until every label in the group matches its previous pass
- **Protect Skin as Label** — always label **-1**, never Label A/B. Gets `lo` from the shared Contrast low (best_lo) calibration section above, then fills everything outside the shared Brain mask layer with skin and trims it to real signal, **slice by slice independently by calling Correct Label's own single-slice engine directly**, once per slice — the exact same function the button itself calls, byte-for-byte identical to selecting skin by hand on a slice and clicking Correct Label; not a cross-slice 3D whole-cell walk — skin's candidate territory is already fully known per slice from the brain mask, and spans nearly the whole stack, so walking that whole-object machinery per slice would be pure wasted computation here. Each slice still gets its own **per-slice auto-grow + until-stable**, reading the same shared Correction Settings every other correction tool uses, run **in parallel across up to 75% of CPU cores** (every slice is fully independent), and wherever a real cell is present, the boundary is **jointly resolved** against it via the same watershed Correct Adjacent Labels uses (not just excluded) — but only skin's own resulting territory is ever written back, the touching cell's label is never modified (a cell's own existing pixels are always its own watershed marker, which structurally can't be reassigned). **Does NOT clamp against the brain mask** — a **Remove Debris** pass runs automatically right after protection (on this button and in both auto-correct entry points) to sweep up whatever small stray blob skin absorbed, by size alone — reports which real labels end up touching/nearby skin, worth a closer look. The skin trim reads the raw channel behind the Signal layer when it can find it (an open layer with the suffix-stripped name, or `<output folder>/<raw>_original.tif`). **Any slice that comes back with no skin at all is automatically repaired** — re-seeded from scratch and re-corrected independently, up to 2 tries, before being reported as a genuine gap (a cell corrected in 2D against skin gets the identical check); a companion **Remove Skin Label** button clears -1 again whenever it's no longer needed, and a **Hide skin label in the viewer** checkbox makes -1 fully transparent in the display only (no data change, every other label keeps its own real distinct color both while hidden and after toggling back off) whenever it's visually cluttering a correction session. **Refuses to protect twice** — if the layer already carries the skin label it says so and does nothing (click Remove Skin Label first to start over)
- **Auto-correct Labels** — runs the exact same 6-step auto-correct pipeline the Cellpose-SAM Segmentation section's own checkbox chains onto a fresh run (contrast sweep per the shared calibration controls above, skin protection + debris cleanup, per-cell correction — 2D against skin for a touching cell, 3D otherwise — then a final debris cleanup), but against the selected Labels layer as it currently stands instead of a just-produced one, so a labels layer segmented earlier (any route, any session, or one loaded via Load Labels layer) can get the same treatment without re-running Cellpose-SAM. Uses the shared Signal/Labels/Brain mask layer selector, the shared calibration controls, and the shared Correction Settings — for every real cell's own correction, and for skin's own trim too. If skin is already protected (e.g. you clicked Protect Skin as Label first) it is reused as is; if skin cannot be protected the run stops before touching any cell. The report ends with the report-only list of possible non-microglia blobs. **If sanding is on, chains the same fill-cavities + soften-contours stage afterward** as the Cellpose-SAM route does — this button never saves the labels to a file on its own (use Save Labels); it does save its report as `<output folder>/<labels layer>_autocorrect_report.txt`
- **Save Labels** — explicit file dialog, pre-filled with `<output folder>/<layer name>.tif`; saves int32, zlib-compressed

> **Soften label contours (sanding)** itself (the on/off checkbox) is in Tab 2's Common Settings — shared by Correct Label, Correct Adjacent Labels, Auto-correct Labels, and Cellpose-SAM's own auto-correct stage: after any of these, the touched label(s) first have any interior cavity/vesicle filled in (if that checkbox is also on), then get their contours smoothed using this tab's Sigma XY/Z sliders (foreign-protected, purely geometric — rounds off blocky voxel edges without reshaping the cell).

---

## Tab 4 — Statistics

Computes up to 52 CSV columns per labelled cell and exports a CSV. For what each column means, see [GUIDE.md §12](docs/GUIDE.md#12-statistics-csv--all-columns-explained); for the algorithm/formula behind each one, see [STATISTICS_GUIDE.md](docs/STATISTICS_GUIDE.md).

- Select the Labels layer in napari's layer list (the tab uses the selected Labels layer, else the topmost one); optionally pick an Image layer (intensity stats) and a Shapes layer with line/path shapes (brain region assignment)
- Choose output columns via the per-column checklist (**All** / **Reset** buttons)
- Select a description backend (Rule-based / Ollama / OpenAI / Claude API)
- Click **Generate Statistics**

CSV saved as `<fish stem>_statistics.csv` in the fish's output folder. The `is_volume_outlier` column is checked against the GT-tracked Min/Max volume bounds; Statistics only reads them, never changes them.

**Score Against GT** — in **Tab 6 — Sweeps & Utilities**; scores any two Labels layers already in the viewer against each other.

---

## Tab 5 — AI Tools

Always visible, regardless of GPU (checked once at plugin startup). A banner at the top adjusts to your GPU situation instead of hiding the tab: red/bold with no CUDA GPU ("CPU fallback — days to months for a full training run instead of hours"), amber/bold with a GPU under the recommended 8GB ("may still work with a reduced `batch_size`"), or a quiet green confirmation once the recommendation is met. GT Annotation needs no GPU at all.

An **Email notification (optional)** panel — in **Tab 6 — Sweeps & Utilities**, General category — holds the plugin's **one email switch**: **Email me when a job finishes or fails** (default on) plus **only if it ran longer than** (default 10 min). Any background job that runs past the threshold — segmentation, corrections, statistics, sweeps, packaging — sends one email when it finishes or fails (a chained run like Cellpose-SAM → auto-correct → sanding sends one email for the whole chain); training runs always email when they stop. There are no per-tool email checkboxes. Leave the recipient blank to send nothing (the default); a missing or wrong SMTP setup never stops a job, it only prints a note. The SMTP password (and, in Tab 4, the OpenAI/Claude API key) is saved **encrypted in your OS's credential store** (Windows Credential Manager / macOS Keychain / Linux Secret Service, via `keyring`) rather than the plugin's own plaintext `config.json` — a Gmail App Password is a separate, revocable credential meant for exactly this kind of unattended use, so persisting it doesn't expose your real account password. On Linux the OS store needs an unlocked Secret Service session (GNOME Keyring/KWallet), which SSH-only sessions typically don't have; if unavailable, the plugin automatically falls back to a local Fernet-encrypted file (machine-local key, no password prompt) rather than writing anything in plaintext — a weaker guarantee than the OS store since the key sits next to the file it protects, but still not plaintext. See GUIDE.md Section 10h for the full explanation. See [How training launches work](#how-training-launches-work) below for why training emails still arrive even if napari is never reopened (every other job needs napari to stay open, same as the job itself).

**Quickest setup, with a Gmail account (free, no other signup):**

1. Turn on [2-Step Verification](https://myaccount.google.com/security) on your Google account, if it isn't already.
2. Generate a [Google App Password](https://myaccount.google.com/apppasswords) — name it anything (e.g. `napari-zf-microglia-ai`), copy the 16-character code shown. This is **not** your normal Gmail password.
3. In the panel: **Notify email** = where you want the report sent; **SMTP server** = `smtp.gmail.com` (default); **port** = `465` (default); **SMTP username** = your Gmail address; **SMTP password** = the App Password from step 2.
4. Click **Send Test Email** to confirm it works — instant, no GPU or waiting needed, reports success or the specific SMTP error right there.
5. Leave **Email me when a job finishes or fails** ticked and adjust **only if it ran longer than** if 10 minutes doesn't suit you.

Any other SMTP-over-SSL (implicit TLS, not STARTTLS) provider on a fixed port works the same way, just with a different server/port — see **GUIDE.md Section 13a** for the full walkthrough, more provider examples, and a note on why STARTTLS-only providers (e.g. Office 365 on port 587) aren't currently supported.

A switch below that selects one of two mutually-exclusive groups:

### MONAI Training

- **GT Annotation** — hand-draw polygon boundaries on key slices (every ~10 slices) of a chosen Image layer, interpolate along Z (point-to-point, with propagation past a reference slice), then rasterize to `brain_mask`/`skin_mask`/`original`/`brain_only`/`skin_only` TIFFs (plus the polygon `.npz` files), saved in the output folder of the file last opened with **Open TIF / IMS file** (same `<parent>/<stem>/` convention as Tabs 1-4).
- **Prepare Training Data** — converts raw+GT fish folders into the HDF5 dataset the MONAI trainer needs.
- **Train MONAI U-Net** — launches the actual training run (hours to multiple days).

### Cellpose-SAM Training

- **Extract X/Y/Z Patches** — generates 2D crops in all three orientations (XY native, XZ/YZ Z-stretched to match XY's pixel scale) from a full-fish image + GT labels pair. When the GT carries the skin label (-1), it also adds cell-skin boundary crops (a cell within 3 px of skin, default 20% of the cell crops) and skin-only crops (empty mask, default 10%), so the model learns that bright skin is background; skin is written as 0 in every mask crop. **Cleans truncated incidental-neighbor labels by default**: a crop framed around one cell can graze the corner of a different nearby cell, keeping only a tiny sliver as a valid-looking (but wildly wrong-centered) training label — any label below a configurable visible-fraction threshold (default 90%) of its true full-slice size gets zeroed out automatically right after generation, backing up the crop folder first.
- **Combine Crops into Shared Training Folder** — symlinks one fish's crop folder (from Extract X/Y/Z Patches above) into a single shared folder every fish's crops get pooled into for a multi-fish training run. Each file is prefixed with the crop folder's parent-folder name — normally the fish's own full data-folder name, never a short Dish/Fish tag (that shorthand repeats across different experiments/ages in this project). Safe to re-run for a new fish: an already-linked fish's crops are left untouched.
- **Train Cellpose-SAM** — launches fine-tuning (~20h for 200 epochs), defaulting the pretrained-checkpoint field to whatever's already loaded in Tab 2 — "continue training from where Tab 2 left off." Includes the project's branch-weighted loss option (`branch_weight`/`branch_radius`; `branch_weight=0` disables it, using the standard Cellpose loss).
- **Calibrate branch_radius (from GT)** — measures real branch thickness from a GT labels volume (3D skeleton + anisotropic distance transform, thinnest-quartile segment radius) instead of guessing `branch_radius` by hand. Run here, it only reports; the recommendation is applied to the field and saved only when the GT Toolkit Tuning Tool (Tab 6) runs it with GT verified. (Kept here rather than in Tab 6 — it's a direct input to the training run right below it.)
- **Verify Best Epoch (GT Sweep)** — in **Tab 6 — Sweeps & Utilities**; confirms or corrects the recommended-checkpoint pointer against real GT.

### How training launches work

Both "Launch Training" buttons start a **detached background process** — `conda run -n <env> --no-capture-output <script> ...` (env `zf-microglia-ai` for MONAI and `cellpose` for Cellpose-SAM by default, overridable with the `monai_conda_env`/`cellpose_conda_env` config keys), launched so it survives napari closing (POSIX `setsid()` / Windows `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`), with its own log file the GUI tails every 8 seconds. Progress and status persist to config, so **reopening napari automatically reconnects to a still-running job** instead of losing visibility into it. If the job already finished (or crashed) while napari was closed, reopening reports that outcome immediately too — "Training (PID ...) finished while napari was closed. Best ... at epoch ..." — so you're never left checking a blank status for a job that's actually long done. "Stop Training" kills the whole process tree (`conda run` spawns a child `python` process). This works identically on Windows, Linux, and Mac — no `tmux` dependency, since tmux doesn't exist natively on Windows.

**Patience (checkpoints)** — both groups have this field, and it's the *same* early-stopping rule for both: stop once N checkpoints in a row show no improvement in the model-selection metric (Full-brain Dice for MONAI, `test_loss` for Cellpose-SAM — direction handled automatically, higher-better vs. lower-better). `0` disables it. This is enforced externally by the GUI itself (parsing each script's log as checkpoints land), not by `train.py`'s own built-in `--patience` flag — the plugin always overrides that to an effectively-infinite value so there's exactly one early-stopping mechanism in play, not two different ones that happen to look similar in the UI. Also persists and resumes correctly across a napari restart, same as the rest of the job state.

**Recommended checkpoint (Cellpose-SAM only)** — MONAI's `train.py` already saves its own best checkpoint as `best_model_fullstack.pth`, so nothing extra is needed there. `train_xzyz.py` only saves periodic epoch checkpoints with no best-tracking, so whenever a Cellpose-SAM run stops (naturally or via early-stop), the GUI writes `<model_name>_best_recommended.txt` into the run's `models/` folder — a one-line pointer naming the best-`test_loss` checkpoint (e.g. `cpsam_microglia_xzyz_epoch_0150`), not a copy of the checkpoint itself and not an OS symlink (those need elevated privileges on Windows) — so it works identically cross-platform with no special permissions.

**Email notification** — unlike the log-tail/patience/recommended-checkpoint features above, this one doesn't depend on the GUI polling loop at all, and so doesn't require ever reopening napari. When the email switch is on and SMTP is set up in Tab 6 at launch time, the launched process is actually a small self-contained supervisor script (stdlib-only: `subprocess`/`smtplib`/`re`) that runs the real training command, waits for it to exit, parses the same log for the best checkpoint, and emails a completion report — *then* exits. Since the supervisor itself is what's detached, the email fires on its own schedule regardless of whether napari is running at that moment. Clicking "Stop Training" kills the whole process tree (supervisor included) before it reaches the email step, so a manual stop correctly sends no notification — only unattended completions/crashes do.

---

## Tab 6 — Sweeps & Utilities

Twelve tools, each individually collapsible (click a section's title checkbox to hide its contents), with a **Show tools for...** filter at the top (Skin Removal / Pixel Classifier / Cellpose-SAM / General, all on by default). They are gathered here so the other tabs stay focused on running the pipeline rather than tuning it. Every sweep still reads from (and, when applied, writes back to) its *original* tab's own sliders/fields — moving where a tool is displayed doesn't change what it operates on.

**How GT sweeps apply their results.** Each sweep can be run on its own, but then it only reports: its "This is verified ground truth" checkbox is hidden. Shared values (sliders, "Recommended" lines, Min volume/Min hole size floors, branch_radius) are only moved when the **GT Toolkit Tuning Tool** runs that sweep with its own "This is verified ground truth" box ticked. The exception is **Verify Best Epoch**, which keeps its own visible checkbox. Every applied value is pooled across every fish swept so far (average, never-rising floor, or never-falling ceiling).

**GT Toolkit Tuning Tool** — point it at one fish's original file or output folder; **Scan Folder** finds that fish's files by the `<parent>/<stem>/<stem>_<artifact>` convention and fills every sweep's path fields; **Run Full GT Toolkit Tuning** measures Min/Max volume + Min hole size from `<stem>_GROUND_TRUTH.tif`, then runs MONAI Threshold/Erosion, BG Threshold/Signal Erosion, Smooth σ XY/Z, Cellprob + Krendl, and branch_radius in that order, skipping any step whose files are missing. The MONAI step only runs when the brain mask is provably GT Annotation's hand-drawn one (`<stem>_brain_polygons_interpolated.npz` beside it, mask not rewritten afterwards). Sends one email for the whole run.

**Verify MONAI Threshold / Erosion (GT Sweep)** — the cheapest of the five GT-sweep tools here: scores the brain *mask itself* (Dice/IoU/precision/recall) against a hand-corrected GT brain mask (e.g. from GT Annotation in Tab 5) — not a MONAI prediction. MONAI's sliding-window inference runs exactly once regardless of grid size; every threshold/erosion combination is a cheap re-threshold + post-process on the same probability map, so a 5×5 grid finishes in well under a minute on GPU. A **Sieve** checkbox (default on) auto-narrows Threshold coarse → fine across 3 stages instead of one flat grid, each stage re-centering on the previous stage's winner. Needs a raw/pre-MONAI image (TIFF, not `.ims`) — feeding it an already brain-masked image would bias the very segmentation being scored.

**Verify BG Threshold / Signal Erosion (GT Sweep)** — scores the Pixel Classifier path: sweeps Tab 1's BG Threshold x Signal Erosion (Background mode 2) against the N most complex cells in a GT-annotated fish (default 5, cropped with Pad Z 15 / Pad XY 40), scoring each grid point's resulting labels against GT. Doesn't re-run MONAI inference (takes an already-computed `brain_mask.tif` as input), so a full grid finishes in minutes and works without a GPU. A **Sieve** checkbox (default on) auto-narrows BG Threshold coarse → fine across 3 stages the same way. Also measures **Min volume** and **Min hole size** directly from the GT rather than using guessed constants. Both are **floors that only ever decrease**: applying a sweep's result takes the smallest value measured across every fish so far, so one fish's sweep can never undo what an earlier fish already proved. Min volume is a read-only value; Min hole size is a slider with its own read-only "Recommended floor" line.

**Verify Smooth σ XY / σ Z (GT Sweep)** — sweeps the Pixel Classifier's pre-threshold Gaussian smoothing (sigma XY × sigma Z) against the N most complex cells with BG Threshold/Signal Erosion held fixed at Tab 1's current values — isolates sigma specifically. Cheaper per grid point than the BG Threshold sweep, since each cell's thresholded crop is computed once and reused across every sigma combination. A **Sieve** checkbox (default on) auto-narrows sigma XY coarse → fine across 3 stages. Same Min volume / Min hole size floor measurement as the BG Threshold sweep.

**Calibrate Cellprob + Krendl Merge Parameters + Min size + Large-contact (GT Sweep)** — one button, one `do_3D` network pass, five calibration steps against a full-fish GT labels volume: (1) sweeps Cellprob on raw voxel-level Dice/IoU; (2) forms raw cp_masks at the winner, reusing the same cached flows; (3) compares those against GT to jointly calibrate Safe-merge max gap/min contact against Krendl's real OR-combined merge rule, minimizing total corrections; (4) forms the SAME masks fully unfiltered (min_size=0) and compares those against GT to calibrate Min size — weighted to all-but-never discard a real fragment, since (unlike an over/under-merge) a fragment min_size discards has no "undo"; (5) runs step 2's masks through GMM cleanup + Krendl safe-merge at step 3's result, then calibrates Large-contact merge from whatever's still split, same fewest-total-corrections objective as step 3. Every Cellpose-SAM Segmentation parameter this plugin has, from one network pass; Min hole size for its mask formation is measured from this GT. A **Sieve** checkbox (default on) auto-narrows Cellprob coarse → fine across 3 stages, still reusing that one cached pass throughout. When applied (GT Toolkit, GT verified): Cellprob is averaged across fish; max gap/min contact, Min size and Large-contact are re-optimized over the samples pooled from every fish.

**Calibrate Correct-Label Contrast** — finds the lower-contrast value Correct Label (Tab 3) should start from, by reproducing whatever an active Labels layer already segmented (not independent GT — the one sweep here that isn't). Every other autosweep-driven tool (Protect Skin as Label, Auto-correct Labels, the Cellpose-SAM chained auto-correct stage) runs this exact same sweep whenever its own Auto-sweep checkbox is on, so changing **Cells**/**Slices/cell**/**Edge margin**/**Sweep steps** here (default 20/20/50.0/20, up to 400 samples) changes what every one of those tools does too — this is the one place to tune it. The candidate range is the 1st–99.5th percentile of intensity in the samples' own padded boxes; after the coarse grid, every whole-number value within one step of the winner is tested too. Reads its Signal/Labels layers from Tab 3's shared selector. On success, sets the signal layer's contrast limits to the winning value (low) and an adaptive high (99th percentile of signal inside the labels, at least low + 50).

**Verify Best Epoch (GT Sweep)** — for Cellpose-SAM training: `test_loss` (what the recommended-checkpoint pointer is based on) is only a proxy for real segmentation quality, so this finds the N most morphologically complex cells in a GT-annotated fish (ranked by skeleton branch count, not size), crops each to its bounding box, and runs `do_3D` inference at the recommended epoch plus N checkpoints below/above it (default 5 cells × 5 epochs = 25 inferences), best-IoU-matching each prediction against GT. If the sweep disagrees with the `test_loss`-based recommendation **and** its own "This is verified ground truth" box was ticked, the recommended-checkpoint pointer is rewritten to the sweep-confirmed epoch and that checkpoint is loaded as Tab 2's active model. Takes roughly 30 minutes to a couple of hours; runs as a plain background thread (not detached), so unlike Launch Training it does **not** survive closing napari.

**Score Against GT** — whole-fish, Hungarian-matched instance scoring (TP/FP/FN/Score + mean IoU/Dice over matched pairs, IoU threshold default 0.5) between any two Labels layers already in the viewer. `Score = TP − 0.5×(FP + FN)`. Runs synchronously — pure CPU, no GPU needed. Distinct from the sweep tools above: those test parameter grids against a handful of proxy cells; this scores one specific pair of label volumes completely.

**Drift View in 3D** — General category, nothing to do with GT. Start/Stop Drift button + a Speed slider: slowly, continuously tumbles the 3D camera around whatever's currently in view (switches to 3D display automatically), like drifting past it in space. Purely a camera effect (`viewer.camera.angles`) — never touches any layer's own data. Rotates all three axes (pitch/roll/yaw) freely and continuously, never stalling on one, by composing a proper 3D rotation (`scipy`'s `Rotation`) each tick instead of incrementing napari's three Euler-angle numbers independently.

**Autosave Labels Layer** — General category. Checkbox (default on) + an interval spinbox/unit dropdown (minutes or hours, default **30 minutes**): while checked, periodically saves ONLY the Labels layer currently picked in Tab 3's own shared layer selector to `<output>/<name>_recovery.tif`, overwriting the previous save each time — a crash loses at most one interval's worth of manual editing.

**Build GT-Correction Package** — packages the most-advanced Cellpose-SAM correction stage available (sanded > auto-corrected > Krendl-only, auto-picked when the source image is browsed or the GT Toolkit scans the fish) into a folder + zip (source image + `cp_corrected.tif` + optional raw pre-merge masks as `cp_masks_3D.tif` + `brain_mask.tif` and the raw signal channel (`<stem>_original.tif`) — both needed on the reviewer's end to use Protect Skin as Label — + the auto-correct report (`<stem>_autocorrect_report.txt`) + a lightweight per-cell CSV + [`GROUND_TRUTH_CREATION_GUIDE.md`](docs/GROUND_TRUTH_CREATION_GUIDE.md)). The corrected result becomes future training data via Tab 5's Extract X/Y/Z Patches. On-disk naming for every stage is cumulative: `cp.tif` (raw) → `cp_krendl.tif` (+Krendl) → `cp_krendl_ac.tif` (+auto-correct) → `cp_krendl_ac_snd.tif` (+sanding).

> **Calibrate branch_radius (from GT)** stayed in Tab 5 rather than moving here — it's a direct input to the Train Cellpose-SAM parameters right below it, not a standalone verification tool.

---

## Typical voxel dimensions (zebrafish 4 dpf, 25× objective)

| Axis | Size |
|------|------|
| Z | 1.0 µm |
| X, Y | 0.174 µm |
| Anisotropy | ~5.75:1 |

---

## File format support

| Format | Channels | Metadata source |
|--------|----------|----------------|
| `.tif` / `.tiff` | single or multi-channel (C,Z,Y,X) | ImageJ tags or `*_metadata.txt` |
| `.ims` (Imaris) | all channels | embedded or `*_metadata.txt` |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "no default model found" | Use Browse to select your `.pth` file |
| CUDA out of memory | Pixel Classifier labelling falls back to CPU automatically; for a Cellpose-SAM run that crashed while forming masks, re-run it — the cached network output (`*_cp_flow_cache.npz`) is reused |
| `conda env create` fails on Windows with `Didn't find wheel for cucim-cu12` | `cucim-cu12` (Linux/WSL2-only, no Windows wheels) is Linux-only in `environment.yml`; only Tab 4's GPU-batch stats path needs it, and it falls back to CPU cleanly without it |
| `.ims` files fail to open | `pip install imaris_ims_file_reader` |
| `EnvironmentFileNotFound` on `conda env update` | You must `cd` into the repo folder first |
| "Run Cellpose-SAM Segmentation" errors with `No module named 'cellpose'` | `pip install cellpose` in the `zf-microglia-ai` env (already in `environment.yml` for fresh installs) |
| Neither Tab 2 section shows up | An Image layer whose name ends in `_ExtRm` or `_NoBG` must be open in the viewer — run or open the right Tab 1 output |
| Source edits to the plugin don't take effect | You have a non-editable install — see "Developing the plugin" above |
| Tab 5 (AI Tools) shows a red/amber banner | Expected, not an error — the tab is always available regardless of GPU; the banner just sets expectations (CPU fallback, or try a lower `batch_size`) — see Tab 5 section above |
| Tab 5's "Launch Training" buttons error with script-not-found | The plugin's guessed script path is wrong for your layout — override it in `~/.config/napari-zf-microglia-ai/config.json` (see "Training scripts" above) |

---

## Contact

Carlos Tichy — ai24m016@technikum-wien.at  
FH Technikum Wien — Artificial Intelligence & Data Science
