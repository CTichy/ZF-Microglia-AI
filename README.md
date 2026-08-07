# ZF-Microglia-AI

A [napari](https://napari.org) plugin for automated 3D brain extraction and AI-assisted microglia segmentation and analysis from *Danio rerio* (zebrafish) confocal stacks.

Developed at **FH Technikum Wien** — Artificial Intelligence & Data Science.

---

## What it does

Given a 3D confocal volume (TIF or IMS), the plugin provides four tabs:

- **Tab 1 — Skin Remover:** runs a trained MONAI U-Net to predict the brain mask, removes the skin, and saves `brain_mask.tif` + `brain_only.tif`
- **Tab 2 — Create Labels:** two interchangeable ways to detect and label individual microglia in 3D — a **Pixel Classifier** (Gaussian smooth → threshold → overlap-based union-find 3D stitching → volume filter) and a **Cellpose-SAM Segmentation** pipeline (`do_3D` inference → 3-component GMM cleanup → Krendl safe-merge → large-contact merge). The tab automatically shows whichever one matches your active layer's background-removal mode (see below) — no manual switching needed.
- **Tab 3 — Statistics:** computes up to 51 morphological, spatial, and intensity features per labelled cell and exports a CSV. Only shown once at least one Labels layer exists.
- **Tab 4 — AI Tools:** ground-truth polygon annotation plus launchers for MONAI and Cellpose-SAM training (dataset prep/crop extraction, hours-to-days training runs that survive napari closing). Always available — shows a disclaimer banner instead of hiding if your GPU is missing or under the recommended 8GB — see below.

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

### Training scripts (Tab 4 — AI Tools only)

Tab 4 launches three external, project-specific research scripts (`prepare_data.py`, `train.py`, `train_xzyz.py`) that aren't part of the plugin package or distributed with it. The plugin guesses their location based on this project's own directory layout; if that guess is wrong for your setup, the config file (`~/.config/napari-zf-microglia-ai/config.json`) has `monai_prepare_script_path`/`monai_train_script_path`/`cellpose_train_script_path` keys to override.

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
| Erosion | 0 vox | Strips skin rim from `brain_only`. `brain_mask` always saved un-eroded. Composes correctly with any Background mode — a prior version silently discarded Erosion whenever a Background mode was active, fixed as of this version. |
| Background mode | Off | 1 for Cellpose-SAM, 2 for Pixel Classifier (see Tab 2) |
| BG Threshold | 1.40 | Validated for microglia stacks |

**Verify MONAI Threshold / Erosion (GT Sweep)** — third of the plugin's three GT-sweep tools (alongside Tab 2's BG Threshold/Erosion sweep and Tab 4's Cellpose-SAM epoch sweep), and the cheapest: scores the brain *mask itself* (Dice/IoU/precision/recall) against a hand-corrected GT brain mask (e.g. from GT Annotation in Tab 4). MONAI's sliding-window inference runs exactly once regardless of grid size — every threshold/erosion combination is a cheap re-threshold + post-process on the same probability map, so a 5×5 grid finishes in well under a minute on GPU.

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

**Verify BG Threshold / Erosion (GT Sweep)** — same idea as Tab 4's "Verify Best Epoch," for this path instead: sweeps Tab 1's BG Threshold x Erosion (Background mode 2) against the N most complex cells in a GT-annotated fish, scoring each grid point's resulting Pixel Classifier labels against GT. Doesn't re-run MONAI inference (takes an already-computed `brain_mask.tif` as input), so a full grid finishes in minutes and works without a GPU (Create Labels already has a CPU fallback). Reports a 2D grid (Erosion x BG Threshold) with your current slider values marked. Depends on Erosion and BG Threshold actually composing in Tab 1's pipeline (see the Erosion row above) — a sweep run on an older build would show Erosion having no effect.

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

**Verify Cellprob / Large-contact (GT Sweep)** — sweeps Cellprob x Large-contact against a full-fish GT labels volume, scored with the same whole-fish Hungarian-matched methodology as Tab 3's "Score Against GT" — how the current defaults were actually found historically, now automated. Cellprob needs a real `do_3D` re-inference per value (GPU-preferred, same fallback as Run Cellpose-SAM Segmentation); Large-contact is a cheap post-processing merge threshold swept on top of one `do_3D` result per Cellprob value (reusing this project's own `--skip_inference` shortcut), so total time scales with the Cellprob axis only, not the full grid size.

**Build GT-Correction Package** — packages a Krendl segmentation result (source image + `masks_corrected.tif` + optional raw pre-merge masks + a lightweight per-cell CSV + `GROUND_TRUTH_CREATION_GUIDE.md`) into a folder + zip, matching the exact layout this project has hand-assembled for every fish sent out for manual GT correction (D1F1, D1F2, two D1F4 fish so far). The corrected result becomes future training data via Tab 4's Extract Training Crops.

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

**Score Against GT** — whole-fish, Hungarian-matched instance scoring (TP/FP/FN/Score + mean IoU/Dice over matched pairs) between any two Labels layers already in the viewer. This is the `compare_pred_gt.py` methodology this project has used to validate essentially every real modeling decision, ported as a reusable tool instead of staying CLI-only. `Score = TP − 0.5×(FP + FN)`, matching this project's established convention. Runs synchronously — pure CPU (`scipy.optimize.linear_sum_assignment`), no GPU needed, fast at typical whole-fish object counts. Distinct from the three GT-*sweep* tools elsewhere (Tab 1/2/4): those test parameter grids against a handful of proxy cells; this scores one specific pair of label volumes completely.

---

## Tab 4 — AI Tools

Always visible, regardless of GPU (checked once at plugin startup). A banner at the top adjusts to your GPU situation instead of hiding the tab: red/bold with no CUDA GPU ("CPU fallback — days to months for a full training run instead of hours"), amber/bold with a GPU under the recommended 8GB ("may still work with a reduced `batch_size`"), or a quiet green confirmation once the recommendation is met. GT Annotation has never needed a GPU either way. This used to be a hard gate that hid the whole tab below 8GB VRAM — changed deliberately, since a smaller or absent GPU doesn't make the tools useless, just slower or in need of a smaller `batch_size`.

An **Email notification (optional)** panel sits above the switch, shared by both groups: fill in a recipient address + SMTP server/port/username/password to get one email whenever a training run stops (finishes, crashes, or gets early-stopped). Leave the address blank to disable it (the default). The password is never persisted to disk, same policy as the LLM API key in Tab 3 — you'll re-enter it once per napari session. See [How training launches work](#how-training-launches-work) below for why this still works even if napari is never reopened.

**Quickest setup, with a Gmail account (free, no other signup):**

1. Turn on [2-Step Verification](https://myaccount.google.com/security) on your Google account, if it isn't already.
2. Generate a [Google App Password](https://myaccount.google.com/apppasswords) — name it anything (e.g. `napari-zf-microglia-ai`), copy the 16-character code shown. This is **not** your normal Gmail password.
3. In the panel: **Notify email** = where you want the report sent; **SMTP server** = `smtp.gmail.com` (default); **port** = `465` (default); **SMTP username** = your Gmail address; **SMTP password** = the App Password from step 2.

Any other SMTP-over-SSL (implicit TLS, not STARTTLS) provider on a fixed port works the same way, just with a different server/port — see **GUIDE.md Section 11a** for the full walkthrough, more provider examples, and a note on why STARTTLS-only providers (e.g. Office 365 on port 587) aren't currently supported.

A switch below that selects one of two mutually-exclusive groups:

### MONAI Training

- **GT Annotation** — hand-draw polygon boundaries on key slices (every ~10 slices) of a chosen Image layer, interpolate along Z (point-to-point, with propagation past a reference slice), then rasterize to `brain_mask`/`skin_mask`/`brain_only`/`skin_only` TIFFs, saved next to the source file (same `<parent>/<stem>/` convention as Tabs 1-3).
- **Prepare Training Data** — converts raw+GT fish folders into the HDF5 dataset the MONAI trainer needs.
- **Train MONAI U-Net** — launches the actual training run (hours to multiple days).

### Cellpose-SAM Training

- **Extract Training Crops** — from a `_statistics.csv` + brain_only image + labels triple, extracts single/double/triple/quadruple bbox crops per cell (cell + nearest neighbours) for fine-tuning. Older extraction method.
- **Extract XZYZ Patches (current production method)** — generates 2D crops in all three orientations (XY native, XZ/YZ Z-stretched to match XY's pixel scale) from a full-fish image + GT labels pair — the method every real Cellpose-SAM training dataset in this project has actually used since May 2026. **Cleans truncated incidental-neighbor labels by default**: a crop framed around one cell can graze the corner of a different nearby cell, keeping only a tiny sliver as a valid-looking (but wildly wrong-centered) training label — any label below a configurable visible-fraction threshold (default 90%) of its true full-slice size gets zeroed out automatically right after generation, backing up the crop folder first. This is the fix already applied by hand to this project's real training data on 2026-08-05, now the standing default rather than a separate step.
- **Train Cellpose-SAM** — launches fine-tuning (~20h for 200 epochs), defaulting the pretrained-checkpoint field to whatever's already loaded in Tab 2 — "continue training from where Tab 2 left off." Includes the project's branch-weighted loss option (`branch_weight`/`branch_radius`; `branch_weight=0` disables it, using the standard Cellpose loss).
- **Verify Best Epoch (GT Sweep)** — automates the project's manual GT-verification methodology: `test_loss` (what the recommended-checkpoint pointer is based on) is only a proxy for real segmentation quality, so this finds the N most morphologically complex cells in a GT-annotated fish (ranked by skeleton branch count, not size — a large cell is often simple/amoeboid), crops each to its bounding box, and runs `do_3D` inference at the recommended epoch plus N checkpoints below/above it (default 5 cells × 5 epochs = 25 inferences), best-IoU-matching each prediction against GT. Reports a table plus a confirm/disagree verdict — a disagreement is a signal to look closer, not an automatic override. Takes roughly 30 minutes to a couple of hours depending on cell size and GPU; runs as a plain background thread (not detached), so unlike Launch Training it does **not** survive closing napari.

### How training launches work

Both "Launch Training" buttons start a **detached background process** — `conda run -n <env> --no-capture-output <script> ...`, launched so it survives napari closing (POSIX `setsid()` / Windows `CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`), with its own log file the GUI tails every 8 seconds. Progress and status persist to config, so **reopening napari automatically reconnects to a still-running job** instead of losing visibility into it. If the job already finished (or crashed) while napari was closed, reopening reports that outcome immediately too — "Training (PID ...) finished while napari was closed. Best ... at epoch ..." — so you're never left checking a blank status for a job that's actually long done. "Stop Training" kills the whole process tree (`conda run` spawns a child `python` process). This works identically on Windows, Linux, and Mac — no `tmux` dependency, since tmux doesn't exist natively on Windows.

**Patience (checkpoints)** — both groups have this field, and it's the *same* early-stopping rule for both: stop once N checkpoints in a row show no improvement in the model-selection metric (Full-brain Dice for MONAI, `test_loss` for Cellpose-SAM — direction handled automatically, higher-better vs. lower-better). `0` disables it. This is enforced externally by the GUI itself (parsing each script's log as checkpoints land), not by `train.py`'s own built-in `--patience` flag — the plugin always overrides that to an effectively-infinite value so there's exactly one early-stopping mechanism in play, not two different ones that happen to look similar in the UI. Also persists and resumes correctly across a napari restart, same as the rest of the job state.

**Recommended checkpoint (Cellpose-SAM only)** — MONAI's `train.py` already saves its own best checkpoint as `best_model_fullstack.pth`, so nothing extra is needed there. `train_xzyz.py` only saves periodic epoch checkpoints with no best-tracking, so whenever a Cellpose-SAM run stops (naturally or via early-stop), the GUI writes `<model_name>_best_recommended.txt` into the run's `models/` folder — a one-line pointer naming the best-`test_loss` checkpoint (e.g. `cpsam_microglia_xzyz_epoch_0150`), not a copy of the checkpoint itself and not an OS symlink (those need elevated privileges on Windows) — so it works identically cross-platform with no special permissions.

**Email notification** — unlike the log-tail/patience/recommended-checkpoint features above, this one doesn't depend on the GUI polling loop at all, and so doesn't require ever reopening napari. When a recipient is set, the launched process is actually a small self-contained supervisor script (stdlib-only: `subprocess`/`smtplib`/`re`) that runs the real training command, waits for it to exit, parses the same log for the best checkpoint, and emails a completion report — *then* exits. Since the supervisor itself is what's detached, the email fires on its own schedule regardless of whether napari is running at that moment. Clicking "Stop Training" kills the whole process tree (supervisor included) before it reaches the email step, so a manual stop correctly sends no notification — only unattended completions/crashes do.

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
| Tab 4 (AI Tools) shows a red/amber banner | Expected, not an error — the tab is always available regardless of GPU; the banner just sets expectations (CPU fallback, or try a lower `batch_size`) — see Tab 4 section above |
| Tab 4's "Launch Training" buttons error with script-not-found | The plugin's guessed script path is wrong for your layout — override it in `~/.config/napari-zf-microglia-ai/config.json` (see "Training scripts" above) |

---

## Contact

Carlos Tichy — ai24m016@technikum-wien.at  
FH Technikum Wien — Artificial Intelligence & Data Science
