# vidopt — Offline Windows Guide

End-to-end manual for running **vidopt** with **no network** after the first setup:
install the environment, copy a video corpus, train models, then compress videos.

**Primary path:** extract the production zip → run **`install.bat`** once → use
**`vidopt.bat`**. The package ships its own Python and ffmpeg; the OS does **not**
need either installed.

Related docs: [README.md](README.md) · [USAGE.md](USAGE.md) · [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) · [START_HERE.txt](START_HERE.txt)

---

## Table of contents

1. [What this system does](#1-what-this-system-does)
2. [What you need](#2-what-you-need)
3. [Folder layout](#3-folder-layout)
4. [Build the offline bundle (online PC)](#4-build-the-offline-bundle-online-pc)
5. [On the offline PC — extract and run](#5-on-the-offline-pc--extract-and-run)
6. [Verify the toolchain](#6-verify-the-toolchain)
7. [Prepare and copy the video corpus](#7-prepare-and-copy-the-video-corpus)
8. [Train the ML models](#8-train-the-ml-models)
   - [8.7 Search algorithms](#87-search-algorithms-when-to-change)
   - [8.8 Optional matrix training](#88-optional-train-every-algorithm-matrix)
   - [8.9 Training logs](#89-training-logs)
9. [Inspect trained models](#9-inspect-trained-models)
10. [Compress a video](#10-compress-a-video)
   - [10.5 Offline production deploy](#105-offline-production-deploy-models--runtime)
11. [Day-to-day CLI cheat sheet](#11-day-to-day-cli-cheat-sheet)
12. [Tuning and re-training](#12-tuning-and-re-training)
13. [Repair (damaged or missing files)](#13-repair-damaged-or-missing-files)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What this system does

Two phases:

| Phase | Command | Network? | When | Cost |
|---|---|---|---|---|
| **Train** | `vidopt train <corpus> --resume` | No | Once per corpus/encoder | Hours |
| **Compress (production)** | `vidopt compress IN -o OUT` | No | Every video | Minutes |

Training **measures** VMAF on many encodes, then learns models that map scene features →
encoder settings `(crf, aq-mode, aq-strength)`.

Compression **predicts** those settings and encodes. No VMAF in the hot path unless you
pass `--verify`.

Default VMAF targets: **85, 89, 93**. Default encoder: **libx265** (CPU).

---

## 2. What you need

### Online machine (once)

- Windows 10/11 x64
- Python **3.10+** ([python.org](https://www.python.org/downloads/) — tick â€œAdd to PATHâ€)
- Network (to download wheels + ffmpeg, or use an ffmpeg you already have with **libvmaf**)
- This repository

### Offline / production machine

- Same OS/arch as the online machine (**Windows x64 <-> Windows x64**)
- The **production zip** extracted (ships its own Python + ffmpeg — no OS Python required)
- Disk: ~2 GB for tools, plus **several x corpus size** for training scratch (`runs\`)
- Optional: NVIDIA GPU (use `--encoder hevc_nvenc --gpu-workers 1` only if `vidopt.bat doctor` shows NVENC OK)

Not needed: OS Python, Docker, Git, internet.

---

## 3. Folder layout

Ready-to-run production package (after extract):

```
vidopt-offline-windows-x64\   (or C:\vidopt — shorter paths are safer on Windows)
  START_HERE.txt
  vidopt.bat                  launcher (use this)
  install.bat                 offline repair / reinstall
  REPAIR.txt
  README.md / USAGE.md / OFFLINE_GUIDE.md
  vendor\
    python\                   portable Python 3.11 + installed libraries + vidopt
    ffmpeg\bin\               ffmpeg.exe + ffprobe.exe (WITH libvmaf)
    wheelhouse\               offline pip wheels (repair)
    installers\               Python embed zip / .exe + get-pip.py (repair)
  src\vidopt\                 application source (used by install.bat)
  models\                     trained model bundles (created by `vidopt.bat train`)
    libx265\
      target_85\
      target_89\
      target_93\
  runs\                       scratch: segments, trial cache, datasets
  video\                      YOUR corpus (you create / copy this)
  out\                        compressed outputs (you create this)
  scripts\                    pack / prepare helpers (optional on offline PC)
```

`vendor\python\` is already installed in the production zip. Use `install.bat` only
to repair. `runs\` and `models\` are created locally when you train.

---

## 4. Build the offline bundle (online PC)

Do this **once** on a Windows x64 PC that has network.

```bat
cd E:\video-compression
scripts\prepare_offline_bundle.bat
```

That script:

1. Downloads **portable Python 3.11** into `vendor\python\` (+ `get-pip.py`)
2. Downloads **all wheels** into `vendor\wheelhouse\`
3. Copies **ffmpeg** into `vendor\ffmpeg\bin\` if found on PATH / `VIDOPT_FFMPEG_DIR`

If ffmpeg was not auto-copied, place libvmaf builds yourself:

```bat
copy C:\ffmpeg\bin\ffmpeg.exe  vendor\ffmpeg\bin\
copy C:\ffmpeg\bin\ffprobe.exe vendor\ffmpeg\bin\
```

### Pack the installed environment for upload

After `install.bat` succeeds on the build PC:

```bat
scripts\pack_production.bat
```

Produces `dist\vidopt-offline-windows-x64.zip` — vendor is compressed first, then embedded
in the project zip. On the offline PC: **extract → `install.bat` → train/compress**.

| Included | Excluded |
|---|---|
| `vendor-windows-x64.zip` (compressed runtime) | raw `vendor/` folder |
| `src\`, docs, scripts, `vidopt.bat`, `install.bat` | `.venv\`, `.git\` |
| empty `video\corpus\`, `out\`, `models\` or trained `models\` with `--with-models` | `runs\`, sample videos |

On the offline PC: **extract → `install.bat` → `vidopt.bat doctor`**.  
`install.bat` extracts `vendor-windows-x64.zip` and installs vidopt into bundled Python.

---

## 5. On the offline PC — extract and install

The production zip contains **`vendor-windows-x64.zip`** (compressed runtime), not an
expanded `vendor\` folder. **Run `install.bat` once** after extract — it unpacks vendor
and installs vidopt into bundled Python. No network required.

```bat
cd path\to\extracted\vidopt-offline-windows-x64
install.bat
vidopt.bat doctor
vidopt.bat train video\corpus --encoder libsvtav1 --level 2 --cpu-workers 0 --resume
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder libsvtav1 --level 2 --verify
```

Run `install.bat` again only for **repair** if files are damaged. See **REPAIR.txt**.

---

## 6. Verify the toolchain

```powershell
vidopt doctor
python scripts\setup.py --verify
```

`doctor` must say **All checks passed** and show `libvmaf=True` and `libx265 OK`.

`--verify` runs a real encode + real VMAF. It must print something like:

```
encode: OK
VMAF: 74.15 over 60 frame(s)
VMAF measurement is working and frame-aligned
```

If either fails, see [§13 Repair](#13-repair-damaged-or-missing-files).

---

## 7. Prepare and copy the video corpus

### 7.1 What a good corpus looks like

The model learns from your examples. **Train on content like production content.**

| Rule | Guidance |
|---|---|
| Count | **10+** source videos |
| Resolution | Every resolution you will compress (a 720p-only model is unreliable on 4K) |
| Variety | High/low motion, grain, dark scenes, flat/animation, screen content |
| Length | ~10â€“60 s clips are ideal; longer is OK but slower to train |
| Formats | `.mp4` `.mkv` `.mov` `.webm` `.avi` `.m4v` `.ts` `.y4m` |

### 7.2 Copy corpus onto the machine

```powershell
# Example: put training videos here
New-Item -ItemType Directory -Force -Path E:\video-compression\video\corpus

# From USB
Copy-Item D:\my_videos\*.mp4 E:\video-compression\video\corpus\ -Force

# Or robocopy a whole tree
robocopy D:\corpus E:\video-compression\video\corpus /E
```

Flat folder or nested folders both work — `vidopt train` recurses directories.

**Do not** put only tiny smoke clips if you will compress long 4K films. Match production.

### 7.3 Check a file

```powershell
ffprobe -v error -select_streams v:0 `
  -show_entries stream=width,height,codec_name,duration `
  -of default=nw=1 E:\video-compression\video\corpus\sample.mp4
```

---

## 8. Train the ML models

### 8.1 Plumbing check (optional, ~10â€“30 min)

Not for deployment — only to prove the pipeline works:

```powershell
vidopt train E:\video-compression\video\corpus --limit 2 --config quick --level 2
```

### 8.2 Full training (for real models)

**CPU (recommended):**

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --encoder libsvtav1 --level 2 --cpu-workers 0 `
  --set paths.work_dir=runs/production --resume
```

**GPU (only if doctor shows NVENC OK):**

```powershell
vidopt train E:\video-compression\video\corpus --config gpu `
  --encoder hevc_nvenc --level 2 --gpu-workers 1 --cpu-workers 4 `
  --set paths.work_dir=runs/production --resume
```

**Single VMAF target via `--level`** (level 2 = VMAF 89):

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --encoder libsvtav1 --level 2 --cpu-workers 0 `
  --set paths.work_dir=runs/production --resume
```

Or override targets explicitly:

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --encoder libsvtav1 --set "search.targets=[89]" `
  --set paths.work_dir=runs/production --cpu-workers 0 --resume
```

### 8.3 What happens (4 stages)

```
stage 1/4  segment by scene cuts
stage 2/4  hash segments (trial cache keys)
stage 3/4  search: encode + VMAF per segment (CLI default: boundary)   ← slow
stage 4/4  train models → models\<encoder>\target_<T>\
```

Expect **hours** on 4K. That is normal.

Search strategy is `search.strategy` (CLI default **`boundary`**). Algorithms, samplers, and
CRF solvers: [USAGE.md §2.4](USAGE.md#24-search-algorithms).

```powershell
# AQ neighbour walk
vidopt train E:\video-compression\video\corpus --config cpu `
  --set search.strategy=coordinate

# 3-D Sobol design
vidopt train E:\video-compression\video\corpus --config cpu `
  --set search.strategy=sample --set search.sampler=sobol

# Bayesian optimisation
vidopt train E:\video-compression\video\corpus --config cpu `
  --set search.strategy=bayes --set search.sampler=lhs
```

### 8.4 Interrupt and resume

Safe to stop with `Ctrl+C`. Progress is checkpointed locally:

| Artifact | Purpose |
|---|---|
| `<work_dir>\search_records.jsonl` | One line per finished segment search |
| `runs\cache\trials.sqlite` | Individual encode + VMAF trials |

Re-run the **same command with `--resume`**:

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --encoder libsvtav1 --level 2 --cpu-workers 0 `
  --set paths.work_dir=runs/production `
  --resume
```

With `--resume`, existing scene cuts are reused and segments already in
`search_records.jsonl` for the same encoder + VMAF target are skipped. Without
`--resume`, search runs again for every segment (cached trials still avoid
re-encoding).

### 8.5 Train overnight in the background (PowerShell)

```powershell
Start-Process -FilePath "E:\video-compression\vendor\python\python.exe" `
  -ArgumentList @(
    "-m", "vidopt", "train", "E:\video-compression\video\corpus",
    "--config", "cpu",
    "--encoder", "libsvtav1",
    "--level", "2",
    "--cpu-workers", "0",
    "--set", "paths.work_dir=runs/production",
    "--resume"
  ) `
  -RedirectStandardOutput "E:\video-compression\runs\logs\train.out.log" `
  -RedirectStandardError  "E:\video-compression\runs\logs\train.err.log" `
  -WindowStyle Hidden

Get-Content E:\video-compression\runs\logs\train.err.log -Wait -Tail 30
```

### 8.6 Artifacts after training

| Path | Contents |
|---|---|
| `runs\production\dataset.csv` | labelled rows (segment Ã— target) |
| `runs\production\dev_summary.json` | run summary + metrics |
| `runs\production\search_records.jsonl` | segment search checkpoints (resume) |
| `runs\cache\trials.sqlite` | all encode/VMAF trials (resume cache) |
| `models\libx265\target_85\` … | deployable model bundles |

### 8.7 Search algorithms (when to change)

Train mode **searches** encoder parameters; compress **predicts** from the trained model.
The search algorithm only affects **training quality and time**, not how compress runs.

CLI `vidopt train` defaults to `search.strategy=boundary`. Full catalog:
[USAGE.md §2.4](USAGE.md#24-search-algorithms).

| Strategy | Use when |
|---|---|
| `boundary` | CLI default — threshold-first AQ refinement |
| `aq_then_crf` | Enumerate AQ, then CRF-solve |
| `coordinate` | x265 float AQ-strength interacts with neighbours |
| `sample` | Baseline 3-D design (Sobol/LHS/Halton) |
| `bayes` / `tpe` / `cmaes` | Research; slower, no guarantee vs default |

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --encoder libsvtav1 --cpu-workers 4 `
  --set search.targets=[89] `
  --set paths.work_dir=runs/production `
  --resume
```

### 8.8 Optional: train every algorithm (matrix)

**Not required for production.** Compares search algorithms offline (days on a large corpus):

```powershell
python scripts\train_matrix.py --resume
```

Writes to `models\matrix\<strategy>\<encoder>\target_89\`. Logs under `runs\matrix\logs\`.
See [USAGE.md §2.10](USAGE.md#210-optional-algorithm-matrix).

### 8.9 Training logs

| Log | Path |
|---|---|
| Main train log | `<work_dir>\logs\vidopt.log` |
| Worker logs | `<work_dir>\logs\worker-<pid>.log` |
| Segment checkpoint | `<work_dir>\search_records.jsonl` |
| Trial cache | `runs\cache\trials.sqlite` |

```powershell
Get-Content E:\video-compression\runs\production\logs\vidopt.log -Wait -Tail 30
```

---

## 9. Inspect trained models

```powershell
vidopt inspect
```

Example:

```
models\libx265\target_89
  encoder       libx265   target VMAF 89
  trained on    36 row(s) from 10 source(s)
  crf MAE       1.28
  hit rate      97%        â† watch this; aim for 95%+
```

**Hit rate** = fraction of held-out segments whose predicted params would still meet the
target. Prefer this over RÂ².

If a target is missing, you cannot compress with `--target` for that value until you
train it.

---

## 10. Compress a video

Requires a model for the chosen target (`vidopt inspect`).

### 10.1 Basic compress

```powershell
New-Item -ItemType Directory -Force -Path E:\video-compression\out | Out-Null

vidopt compress E:\video-compression\video\corpus\input.mp4 `
  -o E:\video-compression\out\input_t89.mp4 `
  --target 89
```

### 10.2 Compress with quality verification

```powershell
vidopt compress E:\in\movie.mp4 -o E:\out\movie.mp4 --target 89 --verify
```

Report:

```
size       32.3 MB -> 8.3 MB  (3.87x)
vmaf       88.38  [MISSED]     or  [OK]
score      0.0282
elapsed    151.4s
```

| Flag | Meaning |
|---|---|
| `--target 85\|89\|93` | must match a trained bundle |
| `--verify` | measure final VMAF (slower; use for QA) |
| `--keep-work` | keep segment temps under `runs\` |
| `--json` | machine-readable summary |
| `--config cpu` / `gpu` | encoder overlay (must match how you trained) |

### 10.3 Batch compress

```powershell
Get-ChildItem E:\in\*.mp4 | ForEach-Object {
  $out = Join-Path E:\out ($_.BaseName + "_t89.mp4")
  vidopt compress $_.FullName -o $out --target 89
}
```

### 10.4 What is preserved

- Audio, subtitles, chapters — copied bit-exact from the original
- Duration / frame count — exact
- 10-bit / unusual chroma — kept when the encoder supports them

Production does **not** need the internet and does **not** need VMAF unless `--verify`.

### 10.5 Offline production deploy (compress-only archive)

**Recommended:** one zip with runtime + models, **no corpus**.

Build on the training machine (after `vidopt train` finishes):

```bat
scripts\pack_compress.bat
rem -> dist\vidopt-compress-windows-x64.zip
```

**Excluded:** `video\corpus`, `runs\`, training scripts, trial cache.

**Included:** `vendor\python\`, `vendor\ffmpeg\`, trained `models\<encoder>\target_<T>\`,
`out\` (empty), `COMPRESS_GUIDE.md`, `PACKAGE.json`.

On the production machine:

```bat
vidopt.bat doctor
vidopt.bat inspect
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder libsvtav1 --level 2 --verify
```

Full guide: [COMPRESS_GUIDE.md](COMPRESS_GUIDE.md).

### 10.6 Shipping production-only (minimal copy)

On a machine that only compresses (models already trained), copy:

1. Production zip with `--with-models`, **or** `vendor\` + `vidopt.bat` + `models\`
2. The `models\` directory tree for your encoder/target(s)

Then only run `vidopt compress ...`.

---

## 11. Day-to-day CLI cheat sheet

```bat
vidopt.bat doctor

rem Train (defaults: encoder=libx265, cpu-workers=4, gpu-workers=0)
vidopt.bat train video\corpus --encoder libx265 --level 2 --cpu-workers 4 --resume
vidopt.bat train video\corpus --encoder hevc_nvenc --level 2 --gpu-workers 1 --cpu-workers 4 --resume

rem Compress (must match the encoder used for training)
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder libx265 --level 2 --verify --resume

rem Scale out
vidopt.bat compress in.mp4 -o out\out.mp4 --level 2 --cpu-workers 8
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder hevc_nvenc --level 2 --gpu-workers 2

vidopt.bat inspect
vidopt.bat config
```

| Flag | Default | Meaning |
|---|---|---|
| `--encoder NAME` | `libx265` | Codec: `libx265`, `libx264`, `libsvtav1`, `hevc_nvenc`, `av1_nvenc`, `h264_nvenc` |
| `--cpu-workers N` | `4` | Parallel CPU encodes (`0` = auto from cores) |
| `--gpu-workers N` | `0` | Parallel GPU encodes (`0` = CPU-only) |

Train and compress with the **same** `--encoder` — models live under `models\<encoder>\`.

Common flags on most commands:

| Flag | Meaning |
|---|---|
| `--config cpu\|gpu\|quick\|PATH` | overlay YAML |
| `--set section.key=value` | one-off override |
| `--resume` | Continue interrupted train or compress |

Config layering: **defaults → `--config` overlay(s) → `--encoder` / `--cpu-workers` / `--gpu-workers` / `--set`**.

---

## 12. Tuning and re-training

If search is already complete, re-run train with `--resume` and a new `model.crf_quantile`
— search is skipped, only model fitting runs again:

```powershell
# More conservative CRF (higher chance of hitting VMAF; larger files)
vidopt train E:\video-compression\video\corpus --encoder libx265 --level 2 `
  --set paths.work_dir=runs/production `
  --set model.crf_quantile=0.10 --resume

# More aggressive compression (more risk of MISSED)
vidopt train E:\video-compression\video\corpus --encoder libx265 --level 2 `
  --set paths.work_dir=runs/production `
  --set model.crf_quantile=0.25 --resume
```

Default `crf_quantile` is `0.15`. If `--verify` often reports **MISSED**, lower it
(e.g. `0.10`) and re-run with `--resume`.

If many rows are **infeasible** during search, the target may be too high for that
encoder/preset, or the content is extreme — add more similar corpus clips or lower the
target.

To change **how** parameters are searched (requires a new search pass with `--resume` or
a fresh train without `--resume`):

```powershell
vidopt train E:\video-compression\video\corpus --config cpu `
  --set search.strategy=coordinate
```

`aq_then_crf` (default) enumerates AQ then solves CRF. Also available: `coordinate`,
`sample` (3-D Sobol/LHS/Halton), `bayes`, `tpe`, `cmaes`. CRF solver: `bisect` /
`brent` / `golden`. Compress never re-searches. See [USAGE.md §2.4](USAGE.md#24-search-algorithms).


---

## 13. Repair (damaged or missing files)

If antivirus quarantine, a bad copy, or a partial extract breaks the package, you can
usually repair **without the internet** using files already in the zip.

Also see the short checklist: **[REPAIR.txt](REPAIR.txt)** (ships in the production zip).

### 13.1 Diagnose first

```bat
vidopt.bat doctor
```

You want: `All checks passed`, `libvmaf=True`, and your encoder marked `OK`.

Optional deeper check (encode + VMAF):

```bat
vendor\python\python.exe scripts\setup.py --verify
```

### 13.2 First repair — `install.bat` (rebuild from installables)

You are offline. The runtime may be damaged; repair from pieces **already in the package**.
Close every `vidopt` / `ffmpeg` / `python` window, then:

```bat
install.bat
vidopt.bat doctor
```

`install.bat` rebuilds **without network**:

| Piece | Source |
|---|---|
| Python runtime (if broken/deleted) | `vendor\installers\python-3.11.9-embed-amd64.zip` (or `.exe`) |
| pip bootstrap | `vendor\installers\get-pip.py` |
| All libraries | `vendor\wheelhouse\*.whl` |
| Application code | `src\vidopt\` → `vendor\python\Lib\site-packages\` |
| ffmpeg | `vendor\ffmpeg\bin\` |

If Python itself was deleted, `install.bat` recreates `vendor\python\` from the installers,
then reinstalls every library from the wheelhouse.

### 13.3 Symptom → fix

| Symptom | What to do |
|---|---|
| `vidopt.bat` missing or “bundled Python missing” | Run `install.bat` — rebuilds Python from `vendor\installers\`. |
| `ModuleNotFoundError` / `ImportError` | Run `install.bat` (reinstalls wheels + recopies vidopt). |
| ffmpeg / libvmaf / encode errors | Restore `vendor\ffmpeg\bin\ffmpeg.exe` and `ffprobe.exe`, then `install.bat`. |
| `no model for encoder …` | Restore `models\` from backup, or re-train with the same `--encoder`. |
| Corrupt / partial extract | Delete the folder, re-download the zip, extract again. Do not merge old + new trees. |
| `WinError 32` / file in use during repair | Task Manager → end `ffmpeg.exe` / `python.exe`, retry `install.bat`. |
| `wheelhouse` empty or missing installers | Package incomplete. Get a fresh zip (`scripts\pack_production.bat`). |

### 13.4 Soft reset (libraries only)

```bat
rmdir /s /q vendor\python\Lib\site-packages\vidopt
install.bat
vidopt.bat doctor
```

### 13.5 Hard reset (still offline)

Requires intact `vendor\installers\`, `vendor\wheelhouse\`, and `vendor\ffmpeg\bin\`:

```bat
rmdir /s /q vendor\python
install.bat
vidopt.bat doctor
```

`install.bat` rebuilds Python from the installers, then reinstalls libraries from the wheelhouse.

### 13.6 When the zip itself is incomplete

On a **connected** build PC:

```bat
scripts\prepare_offline_bundle.bat
install.bat
scripts\pack_production.bat
```

Replace the offline folder with the new extract. Back up `models\` first if needed.

---

## 14. Troubleshooting

### `vidopt` not found

```bat
vidopt.bat doctor
```

If that fails, see [§13 Repair](#13-repair-damaged-or-missing-files).

### libvmaf missing / VMAF cannot be measured

```bat
vidopt.bat doctor
```

Confirm bundled `vendor\ffmpeg\bin\ffmpeg.exe` and `libvmaf=True`. Restore ffmpeg from a
good package if needed (§13.3).

### `WinError 32` / file in use

Close players / kill leftover `ffmpeg.exe` processes and retry.

### Out of memory during 4K training

```bat
vidopt.bat train video\corpus --encoder libx265 --cpu-workers 2 --resume
```

### Hit rate low / verify MISSED

1. Train on content like production (same resolution).
2. `vidopt.bat train video\corpus --encoder libx265 --level 2 --set model.crf_quantile=0.10 --resume`
3. Add more corpus clips and re-run `vidopt.bat train ... --resume`.

### Outside training domain warning on compress

Re-run `vidopt.bat train ... --resume` on a corpus that includes that resolution / content type.

### Offline install tries the network

`install.bat` must use `vendor\wheelhouse`. If wheels are missing, get a fresh zip (§13.6).

### Long path errors

Keep the project near the drive root (`C:\vidopt`) or enable Windows long paths (admin).

---

## Quick start (ready-to-run package)

```bat
vidopt.bat doctor
vidopt.bat train video\corpus --encoder libsvtav1 --level 2 --cpu-workers 0 --resume
vidopt.bat inspect
vidopt.bat compress video\some.mp4 -o out\some_t89.mp4 --encoder libsvtav1 --level 2 --verify
```

If something is broken: `install.bat` then `vidopt.bat doctor` — see [§13](#13-repair-damaged-or-missing-files) and **REPAIR.txt**.

---

## Minimal production checklist

- [ ] Extract production zip; `vidopt.bat doctor` passes
- [ ] Corpus under `video\corpus\` (content matches production)
- [ ] `vidopt.bat train ... --encoder libsvtav1 --level 2 --resume` finished; hit rate ≥ ~95%
- [ ] `vidopt.bat compress ... --level 2 --encoder libsvtav1 --verify` OK on a sample
- [ ] Know repair path: `REPAIR.txt` / `install.bat` if files are damaged
- [ ] Back up `models\` after training
