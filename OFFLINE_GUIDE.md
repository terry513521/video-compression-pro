# vidopt — Offline Windows Guide

End-to-end manual for running **vidopt** with **no network** after the first setup:
install the environment, copy a video corpus, train models, then compress videos.

**Primary path:** extract the production zip → run **`install.bat`** once → use
**`vidopt.bat`**. The package ships its own Python and ffmpeg; the OS does **not**
need either installed.

Related docs: [README.md](README.md) · [USAGE.md](USAGE.md) · [HANDOFF.md](HANDOFF.md)

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
9. [Inspect trained models](#9-inspect-trained-models)
10. [Compress a video](#10-compress-a-video)
11. [Day-to-day CLI cheat sheet](#11-day-to-day-cli-cheat-sheet)
12. [Tuning and re-training](#12-tuning-and-re-training)
13. [Repair (damaged or missing files)](#13-repair-damaged-or-missing-files)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. What this system does

Two phases:

| Phase | Command | Network? | When | Cost |
|---|---|---|---|---|
| **Train (dev)** | `vidopt dev <corpus>` | No | Once per corpus/encoder | Hours |
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

Not needed: OS Python, Docker, Git, bash, make, internet.

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
  models\                     trained model bundles (created by `vidopt.bat dev`)
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

Creates `dist\vidopt-offline-windows-x64.zip` — a **ready-to-run** copy of the
installed environment:

| Included | Excluded |
|---|---|
| `vendor\python\` with **all libraries already installed** | `.venv\`, `.git\` |
| `vendor\ffmpeg\bin\` (ffmpeg + ffprobe + libvmaf) | `runs\`, sample `video\` |
| `vendor\installers\python-3.11.9-amd64.exe` (official installer) | `tests\`, caches |
| `vendor\wheelhouse\`, `src\`, `vidopt.bat` | |
| empty `video\corpus\`, `out\`, `models\` | |
| optional trained `models\` with `--with-models` | |

On the download PC: **extract and run** — no `install.bat` required.

---

## 5. On the offline PC — extract and run

The production zip already contains the **installed** environment.
The offline OS needs **no** Python and **no** ffmpeg installed. **Do not run
`install.bat` unless something is broken.**

```bat
cd path\to\extracted\folder
vidopt.bat doctor
vidopt.bat dev video\corpus --encoder libx265 --cpu-workers 4
vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --encoder libx265 --verify
```

`install.bat` is a **repair** tool (reinstalls wheels + recopies vidopt). See
**[REPAIR.txt](REPAIR.txt)** and [§13](#13-repair-damaged-or-missing-files) if files are
damaged. `vendor\installers\python-3.11.9-amd64.exe` is optional and not required to run
vidopt.

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

Flat folder or nested folders both work — `vidopt dev` recurses directories.

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
vidopt dev E:\video-compression\video\corpus --limit 2 --config quick
```

### 8.2 Full training (for real models)

**CPU (default, always works):**

```powershell
# Prefer fewer parallel jobs on 4K — avoids Windows out-of-memory crashes
vidopt dev E:\video-compression\video\corpus --config cpu `
  --set paths.work_dir=runs/production `
  --set jobs.cpu_workers=4
```

**GPU (only if doctor shows NVENC OK):**

```powershell
vidopt dev E:\video-compression\video\corpus --config gpu `
  --set paths.work_dir=runs/production `
  --set jobs.gpu_workers=2
```

**Single VMAF target** (faster; train others later from the same dataset if needed):

```powershell
vidopt dev E:\video-compression\video\corpus --config cpu `
  --set paths.work_dir=runs/production `
  --set "search.targets=[89.0]" `
  --set jobs.cpu_workers=4
```

> PowerShell tip: quote list overrides — `"search.targets=[89.0]"` — so `[ ]` is not
> treated as a wildcard.

### 8.3 What happens (4 stages)

```
stage 1/4  segment by scene cuts
stage 2/4  hash segments (trial cache keys)
stage 3/4  search: encode + VMAF for many (crf, aq) settings   â† slow
stage 4/4  train models → models\<encoder>\target_<T>\
```

Expect **hours** on 4K. That is normal.

### 8.4 Interrupt and resume

Safe to stop with `Ctrl+C`. Trials are cached in SQLite. Re-run the **same** command:

```powershell
vidopt dev E:\video-compression\video\corpus --config cpu `
  --set paths.work_dir=runs/production `
  --set jobs.cpu_workers=4
```

Already-finished trials are skipped.

### 8.5 Train overnight in the background (PowerShell)

```powershell
Start-Process -FilePath "E:\video-compression\.venv\Scripts\vidopt.exe" `
  -ArgumentList @(
    "dev", "E:\video-compression\video\corpus",
    "--config", "cpu",
    "--set", "paths.work_dir=runs/production",
    "--set", "jobs.cpu_workers=4"
  ) `
  -RedirectStandardOutput "E:\video-compression\runs\logs\dev.out.log" `
  -RedirectStandardError  "E:\video-compression\runs\logs\dev.err.log" `
  -WindowStyle Hidden

# Watch progress
Get-Content E:\video-compression\runs\logs\dev.err.log -Wait -Tail 30
```

### 8.6 Artifacts after training

| Path | Contents |
|---|---|
| `runs\production\dataset.csv` | labelled rows (segment Ã— target) |
| `runs\production\dev_summary.json` | run summary + metrics |
| `runs\cache\trials.sqlite` | all encode/VMAF trials (resume cache) |
| `models\libx265\target_85\` â€¦ | deployable model bundles |

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

### 10.5 Shipping production-only

On a machine that only compresses (models already trained), copy:

1. Project + `vendor\` + `.venv` setup (§5), **or** recreate venv from wheelhouse  
2. The `models\` directory  

Then only run `vidopt compress ...`.

---

## 11. Day-to-day CLI cheat sheet

```bat
vidopt.bat doctor

rem Train (defaults: encoder=libx265, cpu-workers=4, gpu-workers=0)
vidopt.bat dev video\corpus --encoder libx265 --cpu-workers 4
vidopt.bat dev video\corpus --encoder hevc_nvenc --gpu-workers 1 --cpu-workers 4
vidopt.bat train runs\production\dataset.csv --encoder libx265

rem Compress (must match the encoder used for training)
vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --encoder libx265 --verify

rem Scale out
vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --cpu-workers 8
vidopt.bat compress in.mp4 -o out\out.mp4 --target 89 --encoder hevc_nvenc --gpu-workers 2

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
| `--log-level DEBUG` | more logging |

Config layering: **defaults → `--config` overlay(s) → `--encoder` / `--cpu-workers` / `--gpu-workers` / `--set`**.

---

## 12. Tuning and re-training

Search is expensive; training from `dataset.csv` is cheap.

```powershell
# More conservative CRF (higher chance of hitting VMAF; larger files)
vidopt train runs\production\dataset.csv --set model.crf_quantile=0.10

# More aggressive compression (more risk of MISSED)
vidopt train runs\production\dataset.csv --set model.crf_quantile=0.25
```

Default `crf_quantile` is `0.15`. If `--verify` often reports **MISSED**, lower it
(e.g. `0.10`) and re-train — no need to re-run `vidopt dev`.

If many rows are **infeasible** during search, the target may be too high for that
encoder/preset, or the content is extreme — add more similar corpus clips or lower the
target.


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
vidopt.bat dev video\corpus --encoder libx265 --cpu-workers 2
```

### Hit rate low / verify MISSED

1. Train on content like production (same resolution).
2. `vidopt.bat train runs\production\dataset.csv --encoder libx265 --set model.crf_quantile=0.10`
3. Add more corpus clips and re-run `vidopt.bat dev`.

### Outside training domain warning on compress

Re-run `vidopt.bat dev` on a corpus that includes that resolution / content type.

### Offline install tries the network

`install.bat` must use `vendor\wheelhouse`. If wheels are missing, get a fresh zip (§13.6).

### Long path errors

Keep the project near the drive root (`C:\vidopt`) or enable Windows long paths (admin).

---

## Quick start (ready-to-run package)

```bat
vidopt.bat doctor
vidopt.bat dev video\corpus --encoder libx265 --cpu-workers 4
vidopt.bat inspect
vidopt.bat compress video\some.mp4 -o out\some_t89.mp4 --target 89 --encoder libx265 --verify
```

If something is broken: `install.bat` then `vidopt.bat doctor` — see [§13](#13-repair-damaged-or-missing-files) and **REPAIR.txt**.

---

## Minimal production checklist

- [ ] Extract production zip; `vidopt.bat doctor` passes
- [ ] Corpus under `video\corpus\` (content matches production)
- [ ] `vidopt.bat dev ... --encoder libx265 --cpu-workers 4` finished; hit rate ≥ ~95%
- [ ] `vidopt.bat compress ... --target 89 --encoder libx265 --verify` OK on a sample
- [ ] Know repair path: `REPAIR.txt` / `install.bat` if files are damaged
- [ ] Back up `models\` after training
