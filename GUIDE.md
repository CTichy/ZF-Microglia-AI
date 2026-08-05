# MONAI Skin-Remover — Complete User Guide

**For zebrafish confocal microscopy — step by step, from zero to microglia labels.**

---

## Table of Contents

1. [What this plugin does](#1-what-this-plugin-does)
2. [Installation](#2-installation)
3. [Getting the model files](#3-getting-the-model-files)
4. [Opening the plugin in napari](#4-opening-the-plugin-in-napari)
5. [Tab 1 — Skin Remover](#5-tab-1--skin-remover)
6. [Tab 2 — Create Labels](#6-tab-2--create-labels)
   - [6a. Which tool is active — Pixel Classifier or Cellpose-SAM?](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam)
   - [6b. Pixel Classifier — Union-Find Labels](#6b-pixel-classifier--union-find-labels)
   - [6c. Cellpose-SAM Segmentation](#6c-cellpose-sam-segmentation)
7. [Tab 3 — Statistics](#7-tab-3--statistics)
   - [7a. Analysing cells by brain region (optic tectum / hindbrain)](#brain-regions-optional)
   - [7b. Intensity statistics per label](#intensity-statistics-optional)
8. [Output files and folder structure](#8-output-files-and-folder-structure)
9. [Statistics CSV — all columns explained](#9-statistics-csv--all-columns-explained)
10. [Setting up description backends](#10-setting-up-description-backends)
11. [Full workflow: from raw stack to labelled cells](#11-full-workflow-from-raw-stack-to-labelled-cells)
    - [Step 8a. Assign cells to optic tectum / hindbrain](#step-8a--assign-cells-to-brain-regions-optional)
12. [Reinstalling after an update](#12-reinstalling-after-an-update)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. What this plugin does

You have a confocal microscopy stack of a zebrafish brain. The image contains the brain you care about plus skin, tissue, and background surrounding it.

This plugin does two things, in order:

**Step A — Skin Removal (Tab 1):** Uses a trained AI model (MONAI 3D U-Net) to automatically detect and remove everything outside the brain, producing a clean `brain_only` image where only the cells of interest remain visible.

**Step B — Label (Tab 2):** From the cleaned image, automatically finds and labels each individual cell as a separately numbered 3D region, using **one of two interchangeable methods** — a Pixel Classifier or a Cellpose-SAM segmentation pipeline. The tab shows whichever one matches your Tab 1 output automatically; see [Section 6a](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam). Lets you sort, split, and edit labels before saving.

**Step C — Analyse (Tab 3):** Computes a comprehensive set of shape statistics for each labelled cell and exports them to a CSV file, with an optional AI-generated plain-language description per cell.

> **Every numeric field with a slider next to it (in all three tabs) is directly editable** — click the number box and type an exact value instead of dragging the slider. Both stay in sync.

---

## 2. Installation

You need Python with napari already installed. Open a terminal and run:

```bash
pip install git+https://github.com/CTichy/napari-skin-remover.git
```

All dependencies (PyTorch, MONAI, scikit-image, etc.) are installed automatically.

> **Mac with Apple Silicon (M1/M2/M3):** The plugin automatically uses your GPU via Metal (MPS). No extra steps needed.

> **Windows / Linux with NVIDIA GPU:** CUDA is detected and used automatically for Tab 1 inference and Tab 2 GPU-accelerated labelling (`cupy-cuda12x`, which has wheels for both platforms). Tab 3's *fastest* statistics path additionally uses `cucim` for GPU batch regionprops — this only has Linux wheels (RAPIDS/cuCIM has no native Windows build), so on Windows Tab 3 automatically falls back to a CPU-threaded path instead; nothing breaks, statistics just run somewhat slower. See Section 13 for details.

> **No GPU:** Works on CPU too, just slower (~30–60 minutes per stack for inference).

---

## 3. Getting the model files

The plugin needs up to **two** trained checkpoints, depending on which labelling method you plan to use. Neither is bundled in the plugin.

### MONAI skin-removal model (required — Tab 1)

The AI model (~220 MB) that powers skin removal.

**Download:**

```
https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY
```

Save the file `best_model_fullstack.pth` anywhere on your computer that is easy to find, for example:

```
Documents/
└── brain-peel-model/
    └── best_model_fullstack.pth
```

You will point the plugin to this file the first time you use it. **The plugin remembers the path** — you only need to do this once per installation.

### Cellpose-SAM checkpoint (optional — Tab 2, only if using Cellpose-SAM Segmentation)

Only needed if you plan to label cells with **Cellpose-SAM Segmentation** rather than the **Pixel Classifier** (see [Section 6a](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam)). This is a project-specific fine-tuned Cellpose-SAM model — there is no fixed public download, since every lab trains its own on its own ground truth. Browse to whatever checkpoint you have using the **Browse `[...]`** button in the Cellpose-SAM Segmentation section of Tab 2. The path is remembered the same way as the MONAI model path.

If you don't have a checkpoint yet, use the **Pixel Classifier** instead — it needs no additional model file.

---

## 4. Opening the plugin in napari

1. Open a terminal and type `napari` to launch it.
2. In the napari menu bar, click **Plugins**.
3. Click **MONAI Skin-Remover**.
4. A panel appears on the right side with tabs: **Skin Remover**, **Create Labels**, and **Statistics** (Statistics only appears once at least one Labels layer exists in the viewer).

---

## 5. Tab 1 — Skin Remover

### Open TIF / IMS file

Click this button to open your confocal stack (`.tif`, `.tiff`, or `.ims` format).

- All channels in the file are loaded as separate napari layers, each coloured differently:
  - Channel 0 → gray
  - Channel 1 → green
  - Channel 2 → magenta
  - Channel 3 → cyan
- Voxel size (physical scale in µm) is read automatically from the file metadata and applied to all layers.
- The folder and filename are remembered for automatic output file naming (see Section 7).

> **Important:** After loading, **click on the channel you want to process** in the Layers panel on the left. The plugin always runs on whichever image layer is currently selected (highlighted). For microglia, this is usually the green channel (ch1).

---

### Model (.pth) — Browse button `[...]`

Shows the path to the AI model file. If it says "— no model selected —":

1. Click the `[...]` button.
2. Navigate to where you saved the model file.
3. Select `best_model_fullstack.pth` and click Open.

The path is saved automatically to `~/.config/napari-skin-remover/config.json`. Next time you open the plugin, the model is already loaded.

---

### Input info display

Below the model path, the plugin shows:

- The name and shape of the currently selected layer (e.g. `"NT54_ch1"  (300, 1024, 1024)  uint16`)
- The voxel dimensions: `Z=1.0000  Y=0.1740  X=0.1740 µm`
- The anisotropy ratio and the source of the scale information (from file metadata, from the layer scale, or default 1,1,1)

This is read-only — it updates automatically when you click a different layer.

---

### MONAI Threshold

**Range:** 0.01 to 0.99 — **Default: 0.30**

The AI model outputs a probability map (0 = definitely not brain, 1 = definitely brain). This slider sets the cutoff: voxels above the threshold are classified as brain.

| Value | Effect |
|-------|--------|
| 0.20 | More generous — includes uncertain areas; may keep some skin |
| **0.25** | **Recommended — validated best results (Nathalie)** |
| 0.30 | Previously documented default — superseded by 0.25 |
| 0.50 | Stricter — may cut into brain edges |

> Post-processing (largest connected component + hole filling) cleans up most artefacts regardless of threshold. Keep it at 0.25 unless results look obviously wrong.

---

### Erosion (vox)

**Range:** 0 to 15 voxels — **Default: 0**

After the brain mask is computed, this many voxels are stripped inward from the mask edge before applying it to `brain_only`. This removes a thin skin rim.

- **0:** No erosion — use the mask exactly as computed.
- **2–3:** Typical for zebrafish — removes a ~0.3–0.5 µm rim in XY or 2–3 µm in Z.

> The `brain_mask.tif` saved to disk is **always the un-eroded mask**. Erosion only affects `brain_only.tif`.

---

### Background (brain mode)

Four radio buttons controlling how background signal is handled after inference.

The background level is estimated automatically using the **mode** (most common intensity) of pixels inside the brain, computed from the result of inference. The mode represents the baseline scanner noise because background pixels vastly outnumber bright cell pixels.

#### Off

No background processing. `brain_only` = original volume × brain mask. Everything outside the brain is zeroed; everything inside is the original signal unchanged.

#### 1 — Remove background outside brain (inference)

Removes background-level pixels only in the region **outside** the brain boundary. The brain interior is fully protected — nothing inside changes. Useful for cleaning up outer tissue while leaving the brain completely untouched.

- Requires BG Threshold (see below).

#### 2 — Remove background globally (full stack) ⭐ Recommended before labelling

Removes all pixels across the **entire stack** (including inside the brain) whose intensity falls at or below the background threshold.

**Result:** Only the actual signal (bright microglia, stained cells) survives. Background becomes zero everywhere, leaving clean isolated blobs with empty space between them — exactly what the Create Labels algorithm needs.

**Use this option before creating labels.**

The saved filename gets the suffix `_NoBG` (e.g. `NT54_ch1_brain_only_NoBG.tif`).

#### 3 — Fill removed with random background

After skin removal, the region outside the brain is filled with **random noise** sampled from the actual background pixel distribution. The result looks like the original stack but with skin replaced by natural scanner noise — no hard black boundary at the brain edge.

- Uses Gaussian-filtered corner pixels as the noise pool (±2σ outlier removal) so the noise matches the real scanner texture.
- BG Threshold is not used in this mode.

The saved filename gets the suffix `_RndFill` (e.g. `NT54_ch1_brain_only_RndFill.tif`).

---

### BG Threshold

**Range:** 0.00 to 2.00 — **Default: 0.50**

*(Only active for background modes 1 and 2)*

Fine-tunes the background removal threshold:

```
threshold = background_mode_value + BG_Threshold_offset
pixels ≤ threshold → removed (treated as background)
pixels  > threshold → kept (treated as signal)
```

| Value | Effect |
|-------|--------|
| 0.00 | Threshold = exactly the mode — removes only confirmed background |
| 0.50 | Previously documented default |
| 0.60 | Previously documented "recommended for microglia" — superseded by 1.40 |
| **1.40** | **Recommended for microglia labelling — validated best results (Nathalie)** |
| 2.00 (max) | Aggressive — may remove dim signal from thin cell protrusions |

> For microglia labelling, **1.40** typically produces the cleanest isolated blobs with good gaps between cells. If microglia are losing thin protrusions, lower the value.

---

### Save checkboxes

- **Save brain\_only.tif** (checked by default) — saves the brain-only volume with background removed
- **Save brain\_mask.tif** (checked by default) — saves the binary mask as 0/255 uint8

Both files are saved in the output folder (see Section 7). The `brain_only` filename includes a background-mode suffix:

| Mode | Suffix | Example filename |
|------|--------|-----------------|
| Off | (none) | `NT54_ch1_brain_only.tif` |
| 1 — Exterior Removed | `_ExtRm` | `NT54_ch1_brain_only_ExtRm.tif` |
| 2 — No Background | `_NoBG` | `NT54_ch1_brain_only_NoBG.tif` |
| 3 — Random Fill | `_RndFill` | `NT54_ch1_brain_only_RndFill.tif` |

---

### Run Skin-Remover

Click to start processing. The button is greyed out while running; the status bar shows progress.

When complete, two new layers appear in napari:

- `*_brain_mask` — binary mask in cyan, semi-transparent
- `*_brain_only[suffix]` — the cleaned volume

Processing time:
- NVIDIA GPU: ~30 seconds
- Apple Silicon (MPS): ~5–10 minutes
- CPU only: ~30–60 minutes

---

## 6. Tab 2 — Create Labels

> Before using this tab, run Tab 1 first, then click the resulting `brain_only` layer in the Layers panel to select it. Which section of Tab 2 appears depends on which background mode you used — see 6a below.

---

### 6a. Which tool is active — Pixel Classifier or Cellpose-SAM?

Tab 2 shows **exactly one** of the two labelling methods below, chosen automatically from the active layer's filename suffix. Select a different layer in the Layers panel and Tab 2 switches live — no manual toggle needed.

| Active layer ends in | Section shown | Produced by Tab 1 option |
|---|---|---|
| `_ExtRm` | **Cellpose-SAM Segmentation** (6c) | Option 1 — Remove background outside brain |
| `_NoBG` | **Pixel Classifier** (6b) | Option 2 — Remove background globally |
| `_RndFill` | *Neither* — this output is for presentation/visualisation only | Option 3 — Fill removed with random background |
| anything else (e.g. the raw channel) | *Neither*, with a hint on what to select | — |

So the choice is really made back in **Tab 1, Step 5**: pick **Option 1** if you plan to segment with Cellpose-SAM, or **Option 2** if you plan to use the Pixel Classifier.

The **Sort by / Resort Labels**, **Split Label**, and **Save Labels** tools (Section 6, further down) only appear once one of the two sections above is showing — with no `_ExtRm`/`_NoBG` layer selected, there's nothing yet to sort, split, or save. Likewise, **Tab 3 — Statistics** only appears once at least one Labels layer exists in the viewer.

---

### 6b. Pixel Classifier — Union-Find Labels

Fully self-contained: Gaussian smooth → threshold → per-slice 2D connected components → overlap-based union-find into 3D objects → volume filter → sequential renumber. Shown when the active layer ends in `_NoBG` (background removed everywhere, not just outside the brain) — needs no additional model file.

---

### Smooth σ XY

**Range:** 0.0 to 5.0 — **Default: 1.0** — **Recommended: 1.5**

Controls the softness of blob contours **within each 2D slice** (the XY plane).

Gaussian smoothing is applied before thresholding each slice. This rounds jagged pixel edges and fills tiny holes within the same cross-section.

| Value | Effect |
|-------|--------|
| 0.0 | No smoothing — raw pixel edges |
| **1.5** | **Recommended — solid, rounded blobs with preserved shape** |
| 3.0+ | Heavy — risk of merging nearby cells within the same slice |

> Do not confuse with Smooth σ Z. They serve completely different purposes.

---

### Smooth σ Z

**Range:** 0.0 to 5.0 — **Default: 0.5** — **Recommended: 3.0**

Controls **cross-slice connectivity** — how easily the algorithm links blobs in neighbouring Z slices into a single 3D object.

A microglia that disappears for 1–2 slices (due to low signal or a thin neck) and reappears will be correctly merged into one 3D object when σ Z is high enough.

> **Why σ Z = 3.0 while σ XY = 1.5?**
>
> Zebrafish confocal stacks are highly anisotropic: each Z slice is ~1 µm thick while each XY pixel is ~0.17 µm. So σ Z = 3.0 spans ~3 µm physically, while σ XY = 1.5 spans only ~0.26 µm.
>
> A microglia is typically 10–20 µm in diameter. Two microglia need to be closer than ~3 µm in Z for σ Z = 3.0 to risk merging them — which is uncommon in practice. This has been validated safe for zebrafish 4dpf microglia.

| Value | Effect |
|-------|--------|
| 0.0 | No cross-slice smoothing — each slice fully independent |
| 0.5 | Minimal — only adjacent slices with strong overlap connected |
| **3.0** | **Recommended for zebrafish — bridges 1–3 slice gaps** |
| 5.0+ | Very aggressive — may link cells at different Z depths |

---

### Min overlap (%)

**Range:** 1 to 100 — **Default: 10%**

Two blobs in adjacent slices are recognised as the **same 3D cell** only if they share at least this fraction of the smaller blob's area:

```
overlap_ratio = shared_pixel_count / area_of_smaller_blob
if overlap_ratio ≥ min_overlap% → same object (linked)
```

- **Lower (5%):** Permissive — small touching fragments are linked.
- **Higher (30%):** Strict — only well-aligned blobs linked; isolated particles stay separate.
- **Start at 10%** and increase if too many fragments are joined, or decrease if cells are being cut across slices.

---

### Min volume (vox)

**Range:** 5000 to 10000 — **Default: 7500**

After all 2D blobs are linked into 3D objects, any object smaller than this voxel count is deleted as noise.

| Value | When to use |
|-------|-------------|
| 5000 | Keep smaller objects — may include noise |
| **7500** | **Default — validated for adult zebrafish microglia** |
| 10000 | Keep only large objects — use if many small debris remain |

> Zebrafish microglia at 4dpf typically occupy 15,000–50,000 voxels at standard resolution.

---

### Create Labels

Click to run the 3D labelling algorithm. Processing runs in a background thread — the button is disabled until complete.

When done, a `*_labels` layer appears in napari with each detected cell shown in a different colour. The console prints how many labels were found.

---

### 6c. Cellpose-SAM Segmentation

Shown when the active layer ends in `_ExtRm` (background removed only outside the brain — the interior is left intact for Cellpose-SAM to see). Runs `do_3D` Cellpose-SAM inference, then a 3-component-GMM cleanup pass, a Krendl safe-merge pass (rejoins sub-threshold fragments based on gap size and contact area), and a large-contact merge pass (catches cells accidentally split through a thick junction).

**Requires a Cellpose-SAM checkpoint** — see [Section 3](#3-getting-the-model-files). This is a project-specific fine-tuned model, not shipped with the plugin.

#### Model (.pt/checkpoint) — Browse button `[...]`

Browse to your trained Cellpose-SAM checkpoint file. The path is remembered across sessions, the same way the MONAI model path is remembered in Tab 1.

#### Cellprob threshold

**Range:** -6.0 to 6.0 — **Default: -2.5**

Cellpose-SAM's own confidence cutoff for what counts as foreground (cell) vs. background. Lower (more negative) values are more permissive — they recover more of a cell's thin, dim protrusions but can also let in more noise.

#### Flow threshold

**Range:** 0.0 to 1.0 — **Default: 0.4**

Rejects predicted objects whose internal flow field doesn't self-consistently point back to a single centre (a sign of a malformed or noisy detection). Lower values are stricter — fewer, more confident objects.

#### Safe-merge max gap (vox)

**Range:** 0 to 20 — **Default: 2**

During the Krendl safe-merge pass, two fragments separated by a gap up to this many voxels are considered for merging into one cell (in addition to the contact-area check below).

#### Safe-merge min contact (vox)

**Range:** 0 to 200 — **Default: 10**

Minimum shared-boundary voxel count required between two fragments before the safe-merge pass will join them. Higher values require a more substantial touching surface before merging.

#### Large-contact merge (vox)

**Range:** 1 to 2000 — **Default: 20**

A second, separate merge pass for large blobs that got split apart through a thick junction (more contact area than the safe-merge pass alone would normally join). Raise this if large cells are still coming out fragmented; lower it if separate cells are being wrongly joined.

#### Run Cellpose-SAM Segmentation (button)

Click to start. `do_3D` inference is slow — it can take **hours** for a full-size fish — and runs in a background thread with live status updates, so napari itself stays responsive while it works. When complete, a `*_labels` layer appears, exactly as with the Pixel Classifier.

> If this button errors with `No module named 'cellpose'`, install it in your environment: `pip install cellpose` (already listed in `environment.yml`/`environment-mac.yml` for fresh installs — see Section 13).

---

### Sort by / Reverse order / Resort Labels

After creating (or loading) labels, you can renumber them by a criterion of your choice.

**Sort by** dropdown:

| Option | Meaning | Default order |
|--------|---------|---------------|
| Size | Number of voxels | Largest = label 1 |
| Centroid Z | Z coordinate of centre | Smallest Z = label 1 |
| Centroid Y | Y coordinate of centre | Smallest Y = label 1 |
| Centroid X | X coordinate of centre | Smallest X = label 1 |

**Reverse order** checkbox — inverts the ordering (e.g. smallest = label 1 for Size).

Click **Resort Labels** to apply. The active Labels layer is renumbered 1…N in the chosen order, in place. This is useful for consistent numbering across samples or for matching cells to a reference atlas.

---

### Split Label

Splits a single merged label (a blob where two or more cells are stuck together) into separate parts using a 3D watershed algorithm.

The watershed approach finds the **thinnest neck** connecting two large volumes and cuts there — it does not use a simple distance threshold or Euclidean splitting.

#### Target label

The label number of the blob you want to split. You can type it directly, or:

1. In the napari viewer, hover over the blob and read the label number shown in the status bar.
2. Click the blob to select it in the Labels layer.
3. Click **Use selected** — the label number is filled in automatically.

#### Use selected

Reads the currently selected label from the active napari Labels layer and fills it into the Target label spinner. Click the blob in napari first, then click this button.

#### Split into N parts

**Range:** 2 to 10 — **Default: 2**

How many separate pieces the blob should be divided into. The algorithm searches for the N largest sub-volumes (separated at their thinnest necks) and cuts between them.

> If the blob genuinely has only one major volume (no neck), splitting may fail or produce uneven results. Increase Smooth σ or use a lower Min distance if that happens.

#### Smooth σ (Split)

**Range:** 0.0 to 3.0 — **Default: 1.0**

Gaussian smoothing applied to the distance transform before searching for peaks. Higher values smooth out the distance map, making the algorithm more robust to surface noise but less sensitive to subtle necks.

- **0.5–1.0:** Suitable for most cases.
- **1.5–2.0:** Use if the split point jumps around — smoother distance field = more stable result.
- **0.0:** No smoothing — very sensitive to surface texture.

#### Min distance

**Range:** 1 to 30 voxels — **Default: 5**

Minimum voxel distance required between accepted seed peaks. If two candidate peaks are closer than this, only the stronger one is kept.

- **Too high:** The two centres of a closely-packed double-blob may be rejected as "too close" → fewer than N peaks found → error.
- **Too low:** Surface noise peaks may be accepted as separate centres → wrong split point.
- **5 voxels** works well for microglia-sized cells.

#### Split Label (button)

Click to run. The original blob is replaced in-place:

- The original label number is kept for the **first** part (the largest sub-volume).
- New label numbers (`max_existing + 1`, `max_existing + 2`, …) are assigned to the remaining parts.

The cut is **interface-only**: exactly the voxels at the boundary between parts are removed, creating a 1-voxel gap. The outer surface of each part is not touched — thin protrusions are preserved.

If the algorithm cannot find N distinct sub-volumes, an error message is shown. Try reducing Smooth σ or Min distance.

---

### Save Labels

Opens a file-save dialog pre-filled with the output folder (see Section 7) and the current layer name as the filename. Choose a location and filename, then click Save.

Labels are saved as `int32` TIFF. Each voxel value = label number (0 = background).

> **Save Labels is separate from Create Labels by design.** This lets you edit labels in napari (split, delete, merge) before saving the final result.

> After saving labels, switch to **Tab 3 — Statistics** to compute measurements for each cell.

---

## 7. Tab 3 — Statistics

This tab computes a comprehensive set of morphological, spatial, intensity, and brain-region measurements for every label and saves them to a CSV file. It is intentionally separate from Tab 2 so there is room to configure all options comfortably before clicking Generate.

> Make sure a Labels layer is selected in napari before using this tab.

---

### Description backend

Selects the engine used to generate the plain-language `description` column in the CSV.

| Option | Internet | Cost | Notes |
|--------|----------|------|-------|
| **Rule-based (offline)** | No | Free | Always available; template-based sentences |
| **Ollama (local, free)** | No | Free | Runs a local LLM on your machine |
| **OpenAI API (paid)** | Yes | Pay-per-token | GPT-4o-mini recommended for low cost |
| **Claude API (paid)** | Yes | Pay-per-token | claude-haiku-4-5 recommended for low cost |

See Section 10 for detailed setup instructions for each backend.

---

### Ollama sub-panel (shown when Ollama is selected)

- **Endpoint:** URL where Ollama is running. Default: `http://localhost:11434`. Change this if Ollama runs on a different machine or port.
- **Model:** The Ollama model name to use (e.g. `llama3`, `mistral`, `phi3`). Must be pulled first (`ollama pull llama3`).

---

### API sub-panel (shown for OpenAI or Claude)

- **API Key:** Your secret API key. Shown as dots (password field). **Not saved to disk** — you must re-enter it each session.
- **Model:** The model identifier (e.g. `gpt-4o-mini` for OpenAI, `claude-haiku-4-5-20251001` for Claude).
- **Base URL:** Optional. Leave blank unless you use an OpenAI-compatible proxy or self-hosted endpoint.

---

### Intensity statistics (optional)

**Image layer** dropdown — select an Image layer from your napari session (or leave as "None" to skip).

When an image layer is selected, three additional columns are computed per label using the raw intensity values inside each cell's mask:

- `mean_intensity` — average pixel intensity inside the label
- `integrated_intensity` — total sum of all pixel values (proportional to total fluorescent material)
- `intensity_cv` — coefficient of variation (std / mean) — a measure of how uniform the signal is; 0 = perfectly uniform, high values = heterogeneous staining

> Select the microglia channel (usually the green channel, ch1) for biologically meaningful results.

---

### Brain regions (optional)

Assigns each cell to a named anatomical brain region and computes its distance to the nearest region boundary.

**Boundary lines** dropdown — select a Shapes layer containing one or more `line` shapes, or leave as "None" to skip.

**Region names** text field — enter the region names separated by commas, listed anterior to posterior. For N boundary lines, provide exactly N+1 names.

Example: If you draw one line separating the optic tectum from the hindbrain, enter:
```
Optic tectum, Hindbrain
```

**Fish orientation and axis convention:**

In these stacks, the fish lies along the **X axis** with the head pointing toward X = 0 (anterior = small X, posterior = large X). Y runs from 0 to 2048 top-to-bottom. The optic tectum / hindbrain boundary therefore runs roughly **top to bottom along Y**, separating the left part of the image (optic tectum, small X) from the right part (hindbrain, large X).

**How to draw region boundaries:**

1. In the napari toolbar, click **New shapes layer** (or add via Layers → Add shapes layer).
2. Select the **path** tool in the toolbar (a polyline — click once per vertex, double-click to finish). Use **path** rather than **line** so you can follow the curved anatomy of the optic tectum / hindbrain boundary.
3. Click along the boundary curve **from top to bottom** (Y = 0 toward Y = max), following the anatomical contour. The optic tectum will be on the left of your drawn path (smaller X), the hindbrain on the right (larger X). Double-click on the last point to finish.
4. For multiple regions, draw one path per boundary, each running top to bottom.
5. Select the Shapes layer in the **Boundary lines** dropdown and type your region names.

The boundaries are sorted automatically by mean X position of their vertices (leftmost = most anterior). For each cell, the plugin finds the nearest segment on the boundary curve and uses its orientation to determine which side the cell falls on (left = more anterior region, right = more posterior region).

Two additional columns are added to the CSV:

- `brain_region` — name of the region this cell belongs to
- `region_boundary_dist_um` — distance in µm to the nearest region boundary line

---

### Generate Statistics (button)

Click to compute. Runs in a background thread. When complete:

- A CSV file is saved to the output folder (see Section 8), named `{source_file_stem}_statistics.csv`.
- The status line shows how many labels were processed.

The CSV contains one row per label with up to 45 columns depending on which optional features are enabled. See Section 9 for a full description of every column.


---

## 8. Output files and folder structure

All files saved by the plugin go into a dedicated folder named after your original input file:

```
/path/to/your/data/
├── NT54_ch1.ims                        ← original input file
└── NT54_ch1/                           ← output folder (created automatically)
    ├── NT54_ch1_brain_mask.tif         ← binary brain mask (0/255, uint8)
    ├── NT54_ch1_brain_only_NoBG.tif    ← brain only, background removed globally
    ├── NT54_ch1_labels.tif             ← cell labels (int32)
    └── NT54_ch1_statistics.csv         ← per-label statistics
```

The folder is created the first time a file is saved. If no input file has been opened (e.g. you loaded a layer directly in napari), files are saved in the current working directory.

**Brain-only suffixes** depending on background mode:

| Mode | Suffix |
|------|--------|
| Off | *(none)* |
| 1 — Exterior Removed | `_ExtRm` |
| 2 — No Background | `_NoBG` |
| 3 — Random Fill | `_RndFill` |

---

## 9. Statistics CSV — all columns explained

The CSV produced by Generate Statistics has one row per label, with up to 45 columns. The first 39 are always present; the remaining columns appear only when the corresponding optional feature is enabled.

---

### Identification

| Column | Type | Description |
|--------|------|-------------|
| `label` | integer | Label number matching the napari Labels layer (1, 2, 3, …) |

---

### Volume

| Column | Type | Description |
|--------|------|-------------|
| `volume_vox` | integer | Number of voxels belonging to this label |
| `volume_um3` | float (µm³) | Physical volume in cubic micrometres. Computed as `volume_vox × Z_size × Y_size × X_size`. A typical zebrafish microglia is 1,000–10,000 µm³. |

---

### Position (centroid)

The centroid is the 3D centre of mass of the label — the average position of all its voxels.

| Column | Type | Description |
|--------|------|-------------|
| `centroid_z_vox` | float | Z position in voxel units |
| `centroid_y_vox` | float | Y position in voxel units |
| `centroid_x_vox` | float | X position in voxel units |
| `centroid_z_um` | float (µm) | Z position in micrometres |
| `centroid_y_um` | float (µm) | Y position in micrometres |
| `centroid_x_um` | float (µm) | X position in micrometres |

---

### Bounding box

The smallest rectangular box (aligned with the axes) that completely contains the label.

| Column | Type | Description |
|--------|------|-------------|
| `bbox_dz_um` | float (µm) | Height of the bounding box in Z (depth of the cell in the axial direction) |
| `bbox_dy_um` | float (µm) | Height of the bounding box in Y |
| `bbox_dx_um` | float (µm) | Width of the bounding box in X |

---

### Size and shape

| Column | Type | Description |
|--------|------|-------------|
| `eq_diam_um` | float (µm) | **Equivalent sphere diameter** — the diameter of a perfect sphere with the same volume as this label. Formula: `(6V/π)^(1/3)`. Useful as a single "size" number regardless of shape. |
| `axis1_um` | float (µm) | **Longest principal axis** — the maximum extent of the label along its longest geometric direction. Derived from the inertia tensor eigenvectors. |
| `axis2_um` | float (µm) | **Middle principal axis** — approximated as the average of axis1 and axis3. |
| `axis3_um` | float (µm) | **Shortest principal axis** — the minimum extent perpendicular to the longest axis. |
| `elongation` | float | **Elongation ratio** = `axis1 / axis3`. A perfect sphere = 1.0. A cigar-shaped cell = 3.0 or more. The higher the number, the more stretched out the cell is. |
| `principal_axis_dir` | string | The anatomical direction of the longest axis: `"Z"` (axial), `"Y"` (coronal), or `"X"` (sagittal). Tells you which direction the cell is elongated in. |

---

### Surface and compactness

| Column | Type | Description |
|--------|------|-------------|
| `solidity` | float (0–1) | **Solidity** = `volume / convex_hull_volume`. The convex hull is the smallest convex shape enclosing the label (like shrink-wrap). A solid, convex cell = 1.0. A lobulated or branchy cell with lots of indentations < 1.0. Typical range for microglia: 0.5–0.9. |
| `extent` | float (0–1) | **Extent** = `volume / bounding_box_volume`. How much of the bounding box is actually filled. A cube = 1.0. A sphere ≈ 0.52. Highly branched cells = much lower. |
| `surface_area_um2` | float (µm²) | **Surface area** in square micrometres, computed using marching cubes — a 3D mesh is generated from the label boundary and the triangle areas summed. A cell with long thin branches has a much larger surface area than a smooth sphere of the same volume. |
| `sphericity` | float (0–1) | **Sphericity** = `π^(1/3) × (6V)^(2/3) / A` where V = volume and A = surface area. A perfect sphere = 1.0. Anything less than 1.0 is less spherical. Microglia: typically 0.3–0.8 depending on branch complexity. |
| `surface_to_volume_ratio` | float (µm⁻¹) | **Surface-to-volume ratio** = surface_area / volume. Higher values indicate more complex, surface-rich morphology relative to cell size. Branches and protrusions increase this dramatically. |

---

### Skeleton (branching structure)

These columns require the `skan` package. If `skan` is not installed, they will be 0.

The algorithm skeletonizes the label (reduces it to a 1-voxel-wide skeleton) and analyses the resulting graph of branches.

| Column | Type | Description |
|--------|------|-------------|
| `n_branches` | integer | Number of skeleton branches. A sphere = 1 branch. A microglia with 4 protrusions = roughly 4–8 branches depending on how they connect. |
| `n_endpoints` | integer | Number of free-end branch tips (branches that don't loop back). Corresponds roughly to the number of protrusion tips. |
| `mean_branch_len_um` | float (µm) | Average path length of all skeleton branches in micrometres. |
| `max_branch_len_um` | float (µm) | Length of the longest individual branch — an indicator of maximum protrusion reach. |
| `branch_tortuosity` | float (≥1) | Average ratio of path length to straight-line distance per branch. A value of 1.0 = perfectly straight branches. Higher values = winding, curved protrusions. |
| `branch_density` | float (per 10⁶ µm³) | Number of branches per million cubic micrometres of cell volume. Allows fair comparison between cells of different sizes. |
| `endpoint_density` | float (per 10⁶ µm³) | Number of branch tips per million cubic micrometres. A proxy for protrusion count normalised by cell volume. |
| `process_complexity` | float | Combined measure of branching complexity: `n_branches × mean_branch_len / eq_diam`. High values = many long branches relative to cell diameter. |

---

### Morphotype classification

| Column | Type | Description |
|--------|------|-------------|
| `morphotype` | string | Automatic shape classification based on elongation, sphericity, solidity, branch count, and surface-to-volume ratio. Categories: **Rod-shaped** (elongated, few branches), **Amoeboid** (round, compact, few branches), **Ramified** (many long branches, low sphericity), **Intermediate-ramified** (moderate branching), **Intermediate** (doesn't fit the above). |

---

### Spatial relationships

These columns use all cell centroids together to compute neighbourhood statistics.

| Column | Type | Description |
|--------|------|-------------|
| `nearest_neighbor_dist_um` | float (µm) | Distance to the closest other cell centroid. Small values = cells are tightly packed; large values = isolated cells. |
| `nearest_neighbor_ratio` | float | **Clark-Evans 3D index** for this cell: the ratio of its nearest-neighbour distance to the expected distance if cells were randomly distributed at the same density. Values < 1 = clustering; > 1 = regularity/dispersion. |
| `local_density_100um` | float (cells/10⁶ µm³) | Number of other cells within a 100 µm radius sphere, normalised by sphere volume. A measure of local neighbourhood crowding. |
| `depth_normalized` | float (0–1) | Z position normalised to the full depth range of all cells: 0 = shallowest cell, 1 = deepest. Useful for comparing dorsal vs. ventral distribution across samples. |

---

### Intensity statistics *(optional — requires Image layer selection)*

| Column | Type | Description |
|--------|------|-------------|
| `mean_intensity` | float | Mean pixel intensity inside the label mask. Reflects overall fluorescence brightness of the cell. |
| `integrated_intensity` | float | Sum of all pixel values inside the label (mean × voxel count). Proportional to total fluorescent material in the cell regardless of size. |
| `intensity_cv` | float (0–∞) | Coefficient of variation of pixel intensities = std / mean. 0 = perfectly uniform. High values = heterogeneous staining, possibly indicating internal structure or imaging artefacts. |

---

### Brain region assignment *(optional — requires Shapes layer with boundary lines)*

| Column | Type | Description |
|--------|------|-------------|
| `brain_region` | string | Name of the anatomical region this cell belongs to (as defined by the boundary lines and region names you provided). |
| `region_boundary_dist_um` | float (µm) | Distance from this cell's centroid to the nearest region boundary line, in micrometres. Cells near boundaries may have mixed characteristics. |

---

### Description

| Column | Type | Description |
|--------|------|-------------|
| `description` | string | A plain-language sentence summarising the cell's shape, generated by the selected description backend. Example (rule-based): *"Label 3: Elongated along Y-axis (2.8:1), volume 4,521 µm³, centroid Z=87.3 Y=142.1 X=203.5 µm. Lobulated/irregular surface, sphericity 0.41, solidity 0.72. Morphotype: Intermediate-ramified. 6 branches, 4 endpoints (mean 8.3 µm), tortuosity 1.4."* |

---

## 10. Setting up description backends

### Rule-based (offline) — no setup needed

The default. Descriptions are generated using built-in templates based on the numeric values. No internet connection, no API key, no external software. Always available.

---

### Ollama (local, free)

Ollama runs a large language model locally on your machine. No data is sent to external servers, and there is no ongoing cost after the initial download.

**Step 1 — Install Ollama**

Go to [https://ollama.com/download](https://ollama.com/download) and download the installer for your operating system. Run it.

- On Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- On Mac: Download the `.dmg` and drag to Applications.
- On Windows: Download the `.exe` installer.

**Step 2 — Download a model**

Open a terminal and run:

```bash
ollama pull llama3
```

This downloads the Llama 3 model (~4.7 GB). You only need to do this once. Other models you can use:

```bash
ollama pull mistral      # ~4 GB, fast
ollama pull phi3         # ~2 GB, smaller and faster
ollama pull llama3:70b   # ~40 GB, highest quality — needs 64 GB+ RAM
```

**Step 3 — Verify Ollama is running**

Ollama starts automatically in the background after installation. You can confirm it is running:

```bash
ollama list   # should show your downloaded models
```

**Step 4 — Configure in the plugin**

In **Tab 3 — Statistics**:

1. Select **Ollama (local, free)** from the Description dropdown.
2. **Endpoint:** leave as `http://localhost:11434` (default). Only change this if Ollama runs on a different machine on your network.
3. **Model:** type the model name you downloaded, e.g. `llama3`.
4. Click **Generate Statistics**.

> If you get an `[Ollama error: ...]` in the CSV description column, check that Ollama is running (`ollama list`) and that the model name matches exactly what you downloaded.

---

### OpenAI API (paid)

OpenAI's GPT models run on OpenAI's servers. You pay per token processed. For statistics descriptions (short prompts, short responses), the cost is very low — roughly $0.001–0.01 per 100 cells with `gpt-4o-mini`.

**Step 1 — Create an OpenAI account**

Go to [https://platform.openai.com](https://platform.openai.com) and sign up. You will need to provide a credit card for billing.

**Step 2 — Generate an API key**

1. Log in to [https://platform.openai.com](https://platform.openai.com).
2. Click your profile icon (top right) → **API keys**.
3. Click **+ Create new secret key**.
4. Give it a name (e.g. "napari-skin-remover").
5. Copy the key immediately — it starts with `sk-` and you can only see it once.

**Step 3 — Configure in the plugin**

In **Tab 3 — Statistics**:

1. Select **OpenAI API (paid)** from the Description dropdown.
2. **API Key:** paste your `sk-...` key. It is stored only in memory — not saved to disk.
3. **Model:** `gpt-4o-mini` (recommended — low cost, good quality). Other options:
   - `gpt-4o` — highest quality, higher cost
   - `gpt-3.5-turbo` — fastest, cheapest, lower quality
4. **Base URL:** leave blank unless you use an OpenAI-compatible proxy.
5. Click **Generate Statistics**.

> The API key is **not saved to disk** for security. You must paste it again each time you open napari.

---

### Claude API (paid)

Anthropic's Claude models. Similar pricing model to OpenAI. Claude Haiku is very fast and inexpensive.

**Step 1 — Create an Anthropic account**

Go to [https://console.anthropic.com](https://console.anthropic.com) and sign up with a credit card.

**Step 2 — Generate an API key**

1. Log in to [https://console.anthropic.com](https://console.anthropic.com).
2. Click **API Keys** in the left sidebar.
3. Click **+ Create Key**.
4. Give it a name and copy the key (starts with `sk-ant-`).

**Step 3 — Configure in the plugin**

In **Tab 3 — Statistics**:

1. Select **Claude API (paid)** from the Description dropdown.
2. **API Key:** paste your `sk-ant-...` key.
3. **Model:** `claude-haiku-4-5-20251001` (recommended — fast and cheap). Other options:
   - `claude-sonnet-4-6` — higher quality, moderate cost
   - `claude-opus-4-6` — highest quality, highest cost
4. **Base URL:** leave blank (not used for Claude).
5. Click **Generate Statistics**.

---

## 11. Full workflow: from raw stack to labelled cells

### Step 1 — Open your file

1. Open the plugin (Plugins → MONAI Skin-Remover) in napari.
2. Click **Open TIF / IMS file** and select your confocal stack.
3. All channels appear as layers.
4. **Click the microglia channel** (usually ch1, green) in the Layers panel.

---

### Step 2 — Run skin removal

Set these values in Tab 1:

| Setting | Value |
|---------|-------|
| MONAI Threshold | **0.25** |
| Erosion | 0 (default) |
| Background | **Option 1** if you plan to use Cellpose-SAM in Step 3, **Option 2** if you plan to use the Pixel Classifier |
| BG Threshold | **1.40** |

Click **Run Skin-Remover** and wait.

**What you should see:** A `brain_only` layer. With Option 2, microglia appear as bright isolated blobs on a black background, with clear space between cells (needed for the Pixel Classifier). With Option 1, the brain interior is left intact — only the tissue outside the brain is removed (needed for Cellpose-SAM).

**If blobs look hollow or have large halos (Option 2):** Lower BG Threshold (e.g. 0.40).

**If too much dim signal remains between cells (Option 2):** Raise BG Threshold (e.g. 0.80).

---

### Step 3 — Create labels

Click the `brain_only` layer Tab 1 just produced (`_ExtRm` or `_NoBG`, matching your choice above) in the Layers panel, then switch to the **Create Labels** tab. It automatically shows the matching section — see [Section 6a](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam) for the full logic.

#### Option A — Pixel Classifier (active layer ends in `_NoBG`)

Set these values:

| Setting | Value |
|---------|-------|
| Smooth σ XY | **1.5** |
| Smooth σ Z | **3.0** |
| Min overlap | 10% (default) |
| Min volume | 7500 (default) |

Click **Create Labels**.

**What you should see:** A labels layer where each cell is a different colour. The console prints how many were found.

**Tuning:**
- Too many tiny fragments → increase Min volume or increase both σ values
- Two cells merged together → try Split Label (see below)
- Cells cut across slices → decrease Min overlap or increase σ Z

#### Option B — Cellpose-SAM Segmentation (active layer ends in `_ExtRm`)

1. Browse to your Cellpose-SAM checkpoint (Section 3) if not already set.
2. Leave the defaults (Cellprob -2.5, Flow 0.4, Safe-merge max gap 2, Safe-merge min contact 10, Large-contact merge 20) unless you know you need to adjust them — see [Section 6c](#6c-cellpose-sam-segmentation) for what each one does.
3. Click **Run Cellpose-SAM Segmentation** and wait — this can take hours for a full-size fish. Progress is shown in the status bar; napari stays usable while it runs.

**What you should see:** A labels layer where each cell is a different colour, exactly as with the Pixel Classifier.

---

### Step 4 — Review and edit labels in napari

- Toggle the labels layer on/off to compare with the original
- Hover over cells to see their label number
- Zoom through Z slices to verify cells are correctly separated

---

### Step 5 — Split merged cells (if needed)

If two cells were labelled as one because they touch:

1. Hover over the merged blob and note its label number (shown in the napari status bar at the bottom).
2. In Tab 2, under **Split Label**:
   - **Target label:** enter the label number (or click the blob and click **Use selected**).
   - **Split into:** 2 (or however many cells are merged).
   - **Smooth σ:** 1.0 (default).
   - **Min distance:** 5 (default).
3. Click **Split Label**.
4. The two (or more) cells are separated at their thinnest connection point.

---

### Step 6 — Sort labels (optional)

Click **Resort Labels** to renumber cells by size or position. This is helpful for consistent reporting:

- By **Size** (largest = label 1) — most common
- By **Centroid Z/Y/X** — for atlas alignment

---

### Step 7 — Save labels

Click **Save Labels**. A file dialog opens pre-filled with the output folder. Accept or change the name and click Save.

---

### Step 8 — Generate statistics

1. Click the **Statistics** tab (Tab 3).
2. Make sure the Labels layer is selected in napari.
3. Choose your description backend.
4. *(Optional)* Select a fluorescence channel under **Intensity statistics** to add mean/integrated/CV columns.
5. *(Optional — see Step 8a below)* Draw region boundary lines in a Shapes layer, then select it under **Brain regions** and enter the region names (e.g. `Optic tectum, Hindbrain`).
6. Click **Generate Statistics**.
7. The CSV is saved automatically to the output folder.

---

### Step 8a — Assign cells to brain regions (optional)

This lets you label each cell as belonging to the **optic tectum**, the **hindbrain**, or any other anatomical region you define by drawing a dividing line across the image.

**Orientation reminder:** The fish lies along the **X axis** — head at X = 0, tail at X = max. Y runs top to bottom (0 → 2048). The optic tectum / hindbrain boundary therefore appears as a roughly vertical curve when you look at the XY plane — it runs from the top of the brain (small Y) to the bottom (large Y), at some X position. Everything to the **left** of the curve (smaller X) is the optic tectum; everything to the **right** (larger X) is the hindbrain.

**Step-by-step:**

1. **Scroll to a representative Z slice** where the optic tectum / hindbrain boundary is most clearly visible. In zebrafish 4dpf, this boundary is typically a recognisable change in cell density roughly at the mid-point of the anterior–posterior (X) axis.

2. **Add a Shapes layer:** In napari, click the **+** icon in the toolbar and choose **Shapes**, or go to **Layers → Add shapes layer**.

3. **Select the path tool:** In the shapes toolbar (appears when the Shapes layer is active), click the **path** icon (a polyline with multiple vertices). Do **not** use the straight line tool — the optic tectum / hindbrain boundary is curved.

4. **Draw the boundary curve from top to bottom:** Start clicking at the top of the brain (small Y, Y ≈ 0 side) and work downward to the bottom of the brain (large Y). Follow the curved anatomical boundary as you click. Double-click on the last point to finish. You typically need 4–10 click points.

   - The optic tectum (anterior, smaller X) will be on the **left** of your drawn path.
   - The hindbrain (posterior, larger X) will be on the **right**.

   > Tip: zoom in on the XY view where the boundary is clearest. If unsure of the exact position, trace the curve slightly anterior (more to the left). You can always select the path layer, press Delete to remove it, and redraw.

5. *(For three or more regions)* Draw one additional path per boundary, each running top to bottom.

6. **Switch to Tab 3 — Statistics** in the plugin.

7. Under **Brain regions**:
   - **Boundary lines:** select your Shapes layer from the dropdown.
   - **Region names:** type the names separated by commas, anterior to posterior:
     ```
     Optic tectum, Hindbrain
     ```
     For three regions, e.g.: `Forebrain, Optic tectum, Hindbrain`

8. Click **Generate Statistics**.

**Result:** The CSV will include two extra columns:

| Column | Description |
|--------|-------------|
| `brain_region` | Name of the region each cell belongs to |
| `region_boundary_dist_um` | Distance in µm from the cell to the nearest boundary line |

You can then filter the CSV in Excel or Python by `brain_region` to compare microglia density, morphology, or intensity between the optic tectum and the hindbrain.

---

## 12. Reinstalling after an update

```bash
pip uninstall napari-skin-remover -y
pip install git+https://github.com/CTichy/napari-skin-remover.git
```

Then **fully close and reopen napari**. If napari is running when you reinstall, it uses the old version until restarted.

> **Your model path and settings are preserved** across reinstalls. The config is stored in `~/.config/napari-skin-remover/config.json`.

---

## 13. Troubleshooting

### `conda env create -f environment.yml` fails on Windows with `Didn't find wheel for cucim-cu12`

Fixed in the current `environment.yml` (`cucim-cu12` is now Linux-only there — it has no Windows wheels at all, since it's a Linux/WSL2-only RAPIDS package, and isn't something a different pip flag or index fixes on native Windows). It only accelerates Tab 3 statistics, which fall back to CPU cleanly without it; Tab 1 and Tab 2 are unaffected.

If you still hit this error:

- You're on an older clone — `git pull` in the repo folder, then retry.
- If the environment partially exists from the earlier failed attempt, remove it first: `conda env remove -n skin-seg`, then `conda env create -f environment.yml` again.

---

### The plugin does not appear in Plugins menu

- Make sure napari is fully closed and reopened after installation.
- Verify installation: `pip show napari-skin-remover`

---

### "No model selected" after reinstalling

- Click `[...]` and browse to your `.pth` file.
- Config path: `~/.config/napari-skin-remover/config.json`

---

### Processing runs on CPU (very slow)

**NVIDIA GPU:** Check that PyTorch sees CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`. If `False`, reinstall PyTorch with CUDA support from [https://pytorch.org](https://pytorch.org).

**Apple Silicon:** Check MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Should print `True` on M1/M2/M3.

---

### Statistics are slow (CPU only, no GPU batch)

**Linux:** Install CuPy and cuCIM for GPU-accelerated regionprops:

```bash
pip install cupy-cuda12x cucim-cu12
```

Replace `cuda12x`/`cu12` with your actual CUDA version if different (e.g. `cuda11x` for CUDA 11). After installing, reopen napari — the console will print `regionprops: cuCIM GPU` when statistics are computed.

**Windows:** `cucim-cu12` has no native Windows build (RAPIDS/cuCIM is Linux/WSL2-only) — this is expected and not fixable with a different pip command on native Windows. Tab 3 statistics run on a CPU-threaded path instead; Tab 1 inference and Tab 2 GPU labelling are unaffected, since those only need `cupy-cuda12x`, which does support Windows. If GPU-accelerated statistics specifically matter to you, run the plugin inside WSL2 (Windows Subsystem for Linux) instead, where the Linux install path applies.

---

### `brain_only` layer looks mostly empty (all black)

BG Threshold is too high — lower it (e.g. from 1.40 toward 0.50-0.60).

---

### Create Labels finds 0 or too few objects

- Lower **Min volume** (try 5000).
- Increase **σ XY** and **σ Z** slightly.
- Make sure you selected the `brain_only` layer (not the raw channel) before clicking Create Labels.

---

### Create Labels finds hundreds of tiny fragments

- Increase **Min volume** to 10000.
- Ensure Option 2 with sufficient BG Threshold was used — the brain_only layer must have clean gaps between cells.

---

### Two cells appear as one label (merged)

- Use **Split Label** (Section 6 above) to separate them at the thinnest neck.
- Or decrease σ XY and rerun Create Labels.

---

### Split Label error: "Only N sub-volume(s) found"

The blob doesn't have a clear separation into the requested number of parts.

- Reduce **Smooth σ** (the distance field is over-smoothed and the saddle disappears).
- Reduce **Min distance** (the two centres are being rejected as too close).
- Check the blob is genuinely two distinct cells — zoom in and inspect it slice by slice.

---

### Ollama description shows `[Ollama error: ...]`

- Verify Ollama is running: open a terminal and run `ollama list`.
- If not running, start it: `ollama serve`
- Check the model name matches exactly: `ollama list` shows available models.
- Default endpoint `http://localhost:11434` — change only if Ollama is on a different machine.

---

### OpenAI/Claude API returns an error

- The API key must be pasted fresh each session — it is not saved to disk.
- Check your account has billing set up and enough credit.
- The model name must match exactly (e.g. `gpt-4o-mini`, `claude-haiku-4-5-20251001`).

---

### "Run Cellpose-SAM Segmentation" errors with `No module named 'cellpose'`

Install it in your environment: `pip install cellpose`. Already included in `environment.yml`/`environment-mac.yml` for fresh installs — if you set up your environment before this feature was added, run `conda env update --name skin-seg -f environment.yml --prune` (or `environment-mac.yml`) and reinstall.

---

### Neither Pixel Classifier nor Cellpose-SAM section shows up in Tab 2

The active layer's name must end in `_ExtRm` (Cellpose-SAM) or `_NoBG` (Pixel Classifier) — reselect the correct Tab 1 output layer in the Layers panel. See [Section 6a](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam).

---

## Quick Reference Card

### Tab 1 — Skin Remover

| Control | Recommended | What it does |
|---------|-------------|--------------|
| MONAI Threshold | 0.25 | AI confidence cutoff |
| Erosion | 0 | Strips voxels from mask edge |
| Background | Option 2 | Removes background globally (best for labels) |
| BG Threshold | 1.40 | Fine-tunes background removal level |

### Tab 2 — Create Labels

Shown automatically based on active layer suffix — `_ExtRm` → Cellpose-SAM, `_NoBG` → Pixel Classifier (see [6a](#6a-which-tool-is-active--pixel-classifier-or-cellpose-sam)).

**Pixel Classifier**

| Control | Recommended | What it does |
|---------|-------------|--------------|
| Smooth σ XY | 1.5 | Contour softness within each slice |
| Smooth σ Z | 3.0 | Cross-slice blob connectivity |
| Min overlap | 10% | Overlap needed to link blobs across slices |
| Min volume | 7500 | Minimum voxels to keep a 3D object |

**Cellpose-SAM Segmentation**

| Control | Recommended | What it does |
|---------|-------------|--------------|
| Cellprob threshold | -2.5 | Confidence cutoff for foreground vs. background |
| Flow threshold | 0.4 | Rejects self-inconsistent flow predictions |
| Safe-merge max gap | 2 vox | Max gap allowed when merging fragments |
| Safe-merge min contact | 10 vox | Min touching surface required to merge |
| Large-contact merge | 20 vox | Second merge pass for thick-junction splits |

**Both methods (once labels exist)**

| Control | Recommended | What it does |
|---------|-------------|--------------|
| Split σ | 1.0 | Smoothness for watershed split |
| Min distance | 5 | Peak separation for split detection |

### Tab 3 — Statistics

| Control | Options | What it does |
|---------|---------|--------------|
| Description | Rule-based / Ollama / OpenAI / Claude | Engine for the description column |
| Image layer | Any Image layer / None | Adds intensity statistics (mean, integrated, CV) |
| Boundary lines | Any Shapes layer / None | Assigns cells to named brain regions |
| Region names | Comma-separated text | Names for each region (N lines → N+1 names) |
| Generate Statistics | — | Computes up to 45 metrics per label, saves CSV |

---

*Plugin developed at FH Technikum Wien — Artificial Intelligence & Data Science*
*Contact: carlos.tichy@gmail.com*
