# ZF-Microglia-AI

A [napari](https://napari.org) plugin for automated 3D brain extraction and AI-assisted microglia segmentation and analysis from *Danio rerio* (zebrafish) confocal stacks.

Developed at **FH Technikum Wien** — Artificial Intelligence & Data Science.

---

## What it does

Given a 3D confocal volume (TIF or IMS), the plugin provides three tabs:

- **Tab 1 — Skin Remover:** runs a trained MONAI U-Net to predict the brain mask, removes the skin, and saves `brain_mask.tif` + `brain_only.tif`
- **Tab 2 — Create Labels:** two interchangeable ways to detect and label individual microglia in 3D — a **Pixel Classifier** (Gaussian smooth → threshold → overlap-based union-find 3D stitching → volume filter) and a **Cellpose-SAM Segmentation** pipeline (`do_3D` inference → 3-component GMM cleanup → Krendl safe-merge → large-contact merge). The tab automatically shows whichever one matches your active layer's background-removal mode (see below) — no manual switching needed.
- **Tab 3 — Statistics:** computes up to 51 morphological, spatial, and intensity features per labelled cell and exports a CSV. Only shown once at least one Labels layer exists.

---

## Environment setup (first time)

### 1. Clone the repository

```bash
git clone https://github.com/CTichy/ZF-Microglia-AI.git
cd ZF-Microglia-AI
```

### 2. Create the environment

**Linux (CUDA GPU):**
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

Then: **Plugins → Main Panel (ZF-Microglia-AI)**

---

## Updating (subsequent runs)

```bash
cd ZF-Microglia-AI
git pull
conda env update --name zf-microglia-ai -f environment.yml --prune   # or environment-mac.yml on Mac
```

> **Linux only:** after every `conda env update`, restore the correct torch:
> ```bash
> /home/carlos-eduardo-tichy/anaconda3/envs/zf-microglia-ai/bin/pip install \
>   "torch==2.7.0+cu126" "torchvision==0.22.0+cu126" \
>   --index-url https://download.pytorch.org/whl/cu126
> ```
> (`conda env update` ignores `--extra-index-url` in environment.yml and resets torch to the wrong version.)

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

This plugin needs **two** trained checkpoints — neither is bundled in the repo.

### MONAI skin-removal model (required, Tab 1)

A trained `.pth` checkpoint — **not included in this repo** (~220 MB).

**Download:**
[best_model_fullstack_v1_epoch460_dice9573.pth](https://cloud.technikum-wien.at/s/kYQ4qq3Jsn4xEyY)

Save it anywhere and point the plugin to it using the **Browse (...)** button in Tab 1. The path is remembered across sessions.

### Cellpose-SAM checkpoint (optional, Tab 2)

Only needed if you plan to use **Cellpose-SAM Segmentation** rather than the Pixel Classifier. This is a project-specific fine-tuned Cellpose-SAM model — there's no fixed public download, since every lab/dataset will train its own. Browse to whatever checkpoint you've trained using the **Browse (...)** button in the Cellpose-SAM Segmentation section of Tab 2. The path is remembered across sessions.

---

## Tab 1 — Skin Remover

### Workflow

1. **Open a file** — click "Open TIF / IMS file". All channels load as separate layers.
2. **Select the channel** to process by clicking its layer in the Layers panel.
3. **Browse to the model** `.pth` file if not auto-detected.
4. **Adjust MONAI Threshold** (default 0.25).
5. **Choose Background mode** — pick **Option 1 (Remove outside brain only, `_ExtRm`)** if you plan to segment with **Cellpose-SAM** in Tab 2, or **Option 2 (Remove globally, `_NoBG`)** if you plan to use the **Pixel Classifier**. Tab 2 auto-detects which one you produced and shows the matching tool.
6. Click **Run Skin-Remover**.

All numeric sliders in this plugin are directly editable — click the number box next to any slider and type an exact value instead of dragging.

### Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| MONAI Threshold | 0.25 | Sigmoid cutoff. Keep low — post-processing cleans the rest. |
| Erosion | 0 vox | Strips skin rim from `brain_only`. `brain_mask` always saved un-eroded. |
| Background mode | Off | 1 for Cellpose-SAM, 2 for Pixel Classifier (see Tab 2) |
| BG Threshold | 1.40 | Validated for microglia stacks |

### Output files

Saved in `<source_folder>/<source_stem>/`:

| File | Content | Feeds into (Tab 2) |
|------|---------|---------------------|
| `*_brain_only.tif` | Volume with everything outside the brain zeroed | — |
| `*_brain_only_ExtRm.tif` | Background removed outside the brain only (Mode 1) | Cellpose-SAM Segmentation |
| `*_brain_only_NoBG.tif` | Background also removed inside the brain (Mode 2) | Pixel Classifier |
| `*_brain_mask.tif` | Binary mask (0/255 uint8), un-eroded | — |

---

## Tab 2 — Create Labels

Select a `brain_only` layer produced by Tab 1. The tab shows exactly one of the two sections below, chosen automatically by the layer's filename suffix — select a different layer and it switches live:

| Active layer ends in | Section shown |
|---|---|
| `_ExtRm` | Cellpose-SAM Segmentation |
| `_NoBG` | Pixel Classifier |
| `_RndFill` | neither (presentation/visualization output only) |
| anything else | neither, with a hint on what to select |

The **Resort Labels / Split Label / Save Labels** tools below only appear once one of the two sections above is showing, and **Tab 3 — Statistics** only appears once at least one Labels layer exists in the viewer.

### Pixel Classifier — Union-Find Labels

Fully self-contained: Gaussian smooth → threshold → per-slice 2D connected components → overlap-based union-find into 3D objects → volume filter → sequential renumber. Best on `_NoBG` layers (background removed everywhere, not just outside the brain).

| Parameter | Default |
|-----------|---------|
| Smooth σ XY | 1.5 |
| Smooth σ Z | 3.0 |
| Min overlap (%) | 10 |
| Min volume (vox) | 7500 |

### Cellpose-SAM Segmentation

Runs `do_3D` inference with a Cellpose-SAM checkpoint, then 3-component-GMM cleanup, a Krendl safe-merge pass (only sub-threshold fragments, gap/contact-based), and a large-contact merge pass (catches blobs split through a thick junction). Best on `_ExtRm` layers. `do_3D` inference is slow — can take hours for a full-size fish — and runs in a background thread so the UI stays responsive.

Requires a **Cellpose-SAM checkpoint** — this is a project-specific fine-tuned model, not shipped with the plugin or downloadable from a fixed URL; browse to your own trained checkpoint. The path is remembered across sessions (like the MONAI model path).

| Parameter | Default |
|-----------|---------|
| Cellprob threshold | -2.5 |
| Flow threshold | 0.4 |
| Safe-merge max gap (vox) | 2 |
| Safe-merge min contact (vox) | 10 |
| Large-contact merge (vox) | 20 |

### Additional tools

- **Resort Labels** — renumber 1…N by size, centroid Z/Y/X
- **Split Label** — watershed split of a merged blob into N parts
- **Save Labels** — explicit file dialog (edit labels in napari before saving)

---

## Tab 3 — Statistics

Computes up to 51 features per labelled cell and exports a CSV.

- Select a Labels layer, optionally an Image layer (intensity stats) and a Shapes layer (brain region assignment)
- Choose output columns via the per-column checklist
- Select a description backend (Rule-based / Ollama / OpenAI / Claude API)
- Click **Generate Statistics**

CSV saved as `<stem>_statistics.csv` in the output folder.

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
| CUDA out of memory | Plugin falls back to CPU automatically |
| `conda env create` fails on Windows with `Didn't find wheel for cucim-cu12` | Fixed as of this `environment.yml` — `cucim-cu12` (Linux/WSL2-only, no Windows wheels) is now Linux-only. `git pull` if you're on an older clone; only Tab 3's GPU-batch stats path needs it, and it falls back to CPU cleanly without it |
| `.ims` files fail to open | `pip install imaris_ims_file_reader` |
| `EnvironmentFileNotFound` on `conda env update` | You must `cd` into the repo folder first |
| "Run Cellpose-SAM Segmentation" errors with `No module named 'cellpose'` | `pip install cellpose` in the `zf-microglia-ai` env (already in `environment.yml` for fresh installs) |
| Neither Tab 2 section shows up | Active layer name must end in `_ExtRm` or `_NoBG` — reselect the correct Tab 1 output layer |
| Source edits to the plugin don't take effect | You have a non-editable install — see "Developing the plugin" above |

---

## Contact

Carlos Tichy — ai24m016@technikum-wien.at  
FH Technikum Wien — Artificial Intelligence & Data Science
