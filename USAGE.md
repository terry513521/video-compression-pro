# vidopt — User Guide

Step-by-step instructions for installing the environment, running dev mode, and running
production mode.

- [README.md](README.md) — overview and reference
- [DESIGN.md](DESIGN.md) — how it works and why
- [REFERENCE_ANALYSIS.md](REFERENCE_ANALYSIS.md) — analysis of the reference projects

Runs on **Windows, Linux and macOS**. No Docker, no bash, no make — just Python and
ffmpeg. Commands are shown for PowerShell where they differ; everything else is
identical across platforms.

---

## Table of contents

1. [Installing the environment](#1-installing-the-environment)
   - [1.1 What you need](#11-what-you-need)
   - [1.2 Install](#12-install)
   - [1.3 Using an ffmpeg you already have](#13-using-an-ffmpeg-you-already-have)
   - [1.4 Verifying](#14-verifying)
   - [1.5 Offline / air-gapped machines](#15-offline--air-gapped-machines)
   - [1.6 Where things live](#16-where-things-live)
   - [1.7 Windows notes](#17-windows-notes)
2. [Dev mode](#2-dev-mode)
3. [Production mode](#3-production-mode)
4. [Tuning](#4-tuning)
5. [Operations](#5-operations)
6. [Command reference](#6-command-reference)

---

# 1. Installing the environment

## 1.1 What you need

| Requirement | Windows | Linux / macOS |
|---|---|---|
| Python 3.10+ | [python.org](https://www.python.org/downloads/) — tick "Add to PATH" | `apt install python3 python3-venv` / `brew install python` |
| Git | [git-scm.com](https://git-scm.com/) | `apt install git` / preinstalled |
| Disk | ~1 GB, plus working space (see [§5.2](#52-disk-usage)) | same |
| NVIDIA GPU | **Optional.** CPU-only is fully supported | same |

Not needed: Docker, bash, make, admin rights, a compiler, or a preinstalled ffmpeg.

## 1.2 Install

```bash
git clone <repo-url> vidopt
cd vidopt
```

Create and activate a virtual environment:

```powershell
# Windows — PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bat
REM Windows — cmd.exe
python -m venv .venv
.venv\Scripts\activate.bat
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

> **PowerShell execution policy.** If `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> Or skip activation entirely and call `.venv\Scripts\vidopt.exe` directly.

Then run the setup script:

```bash
python scripts/setup.py
```

It downloads an ffmpeg build for your platform into `vendor/ffmpeg/`, checks that it has
`libvmaf` and the required encoders, and installs vidopt with its dependencies.

**Why it downloads its own ffmpeg.** Almost every prepackaged ffmpeg — winget, choco,
scoop, Homebrew, apt — is built *without* `libvmaf`. Without it the pipeline cannot
measure quality at all, which is the one thing it exists to do. The build it fetches also
includes `libx264`, `libx265`, SVT-AV1 and NVENC.

| Option | Effect |
|---|---|
| `--check` | Verify an existing setup, change nothing |
| `--verify` | Deeper: run a real encode and a real VMAF measurement |
| `--ffmpeg-only` | Only fetch ffmpeg |
| `--skip-ffmpeg` | Only install the Python package |
| `--force` | Re-download ffmpeg |

## 1.3 Using an ffmpeg you already have

It must have been built with `--enable-libvmaf`. Check with
`ffmpeg -hide_banner -filters | findstr libvmaf` (Windows) or `| grep libvmaf`.

```powershell
$env:VIDOPT_FFMPEG_DIR = "C:\tools\ffmpeg\bin"     # PowerShell, this session
setx VIDOPT_FFMPEG_DIR "C:\tools\ffmpeg\bin"       # persist it
```

```bash
export VIDOPT_FFMPEG_DIR=/opt/ffmpeg/bin            # Linux / macOS
```

```bash
python scripts/setup.py --skip-ffmpeg
```

Or set it in the config instead: `--set ffmpeg.bin_dir=C:/tools/ffmpeg/bin`.

## 1.4 Verifying

```bash
vidopt doctor
```

Reports which ffmpeg was resolved, whether `libvmaf` is present, and — importantly —
tests each encoder by **actually encoding a frame**, so "NVENC is compiled in" is never
confused with "this machine has a working GPU":

```
configured encoder : libx265
vmaf model         : vmaf_v0.6.1neg
cpu workers        : 3

encoder availability:
  av1_nvenc      present but unusable here (no device?)
  hevc_nvenc     present but unusable here (no device?)
  libx265        OK <-- configured

VMAF            : available (CPU)

All checks passed.
```

For a deeper gate that runs a real encode and a real VMAF measurement:

```bash
python scripts/setup.py --verify
```

## 1.5 Offline / air-gapped machines

`scripts/setup.py` needs the network once, to fetch ffmpeg and the Python wheels. To
install somewhere with no network, prepare a bundle on a connected machine **of the same
OS and architecture**:

```powershell
# On the connected Windows machine
python scripts/setup.py --ffmpeg-only          # populates vendor/ffmpeg
# Or, if you already have a libvmaf ffmpeg on PATH / VIDOPT_FFMPEG_DIR:
# python scripts/setup.py --vendor-system-ffmpeg --ffmpeg-only

pip download -d vendor/wheelhouse `
  numpy==2.1.3 scipy==1.14.1 scikit-learn==1.5.2 joblib==1.4.2 `
  opencv-python-headless==4.10.0.84 scenedetect==0.6.5 PyYAML==6.0.2 `
  Click platformdirs tqdm threadpoolctl
```

Copy the whole project directory across (including `vendor/`), then on the offline machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/setup.py --skip-ffmpeg
```

`setup.py` uses `vendor/wheelhouse` with `--no-index` automatically when it is present,
so nothing reaches the network.

## 1.6 Where things live

```
vidopt/
├── .venv/                  your virtual environment
├── vendor/ffmpeg/bin/      ffmpeg + ffprobe (downloaded by setup.py)
├── models/<encoder>/target_<T>/    trained model bundles
├── runs/                   working artifacts (datasets, caches, segments)
├── src/vidopt/             the package
│   └── configs/*.yaml      shipped configuration
├── scripts/setup.py        cross-platform environment setup
└── tests/
```

`vendor/`, `.venv/`, `runs/` and `models/` are all git-ignored — they are rebuilt, not
committed.

ffmpeg is discovered in this order, first match wins:

1. `ffmpeg.bin_dir` in the config
2. `$VIDOPT_FFMPEG_DIR` / `%VIDOPT_FFMPEG_DIR%`
3. the active environment's script directory (`Scripts\` on Windows, `bin/` elsewhere)
4. `vendor/ffmpeg/bin`
5. `PATH`

## 1.7 Windows notes

- **Long paths.** Deeply nested working directories can exceed the 260-character limit.
  Either keep the project near the drive root (`C:\vidopt`) or enable long paths:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
- **Antivirus.** Real-time scanning of thousands of short-lived segment files slows dev
  mode considerably. Consider excluding the `runs/` directory.
- **Paths in commands.** Forward slashes work everywhere:
  `vidopt compress C:/videos/in.mp4 -o C:/videos/out.mp4`. If you use backslashes, quote
  the path.
- **`vidopt` not found?** Either activate the venv, or call `.venv\Scripts\vidopt.exe`
  directly, or use `python -m vidopt`.

---

# 2. Dev mode

Dev mode is the expensive, run-it-once step: it **measures** the best encoder parameters
for many scenes and trains a model to predict them. Production mode then never has to
measure anything.

Run it once per (corpus, encoder). It takes hours — that is expected, it is encoding and
scoring thousands of variants.

## 2.1 What it does

```
corpus ─► segment by scene ─► extract features ─► search parameters ─► train models
```

For every segment and every VMAF target (85/89/93 by default), it answers by direct
measurement:

> Which `(crf, aq-mode, aq-strength)` gives the smallest file that still reaches this VMAF?

## 2.2 Preparing a corpus

The corpus should **look like what you will compress in production**. This matters more
than corpus size: the model interpolates well and extrapolates badly, so if you will
compress 4K, train on 4K.

- Point it at files, directories, or both. Directories recurse.
- Recognised: `.mp4 .mkv .mov .webm .y4m .avi .m4v .ts`
- Aim for **10+ source videos** spanning your real content: high and low motion, grain,
  flat animation, dark scenes, screen content.
- Include every **resolution** you will deploy on.

## 2.3 Running it

```bash
# Full run on the 4K corpus, CPU encoder
vidopt dev path/to/corpus --config cpu

# GPU box: NVENC encoding + CUDA VMAF
vidopt dev path/to/corpus --config gpu --set jobs.gpu_workers=4

# Quick plumbing check (~10 min, NOT for deployment)
vidopt dev path/to/corpus --limit 2 --config quick
```

Useful options:

| Option | Effect |
|---|---|
| `--limit N` | Use at most N source videos |
| `--no-train` | Build the dataset, skip training (train later with `vidopt train`) |
| `--set search.n_explore=20` | More exploration per segment: better models, slower |
| `--set search.targets='[85,89,93]'` | Which VMAF targets to learn |

## 2.4 Reading the output

```
stage 1/4: 36 segment(s) from 6 source(s)
stage 3/4: searching 36 segment(s) x 3 target(s) with 3 worker(s)
clip_01__seg_0000.mkv target 85: crf=38.2,aq_mode=2,aq_strength=0.05 vmaf=86.44 ratio=5.22x
...
stage 3/4 complete: 108 row(s), 7 infeasible (6%)

models:
  target 85 [libx265]  train=36 val=13  crf MAE=1.28 R2=0.69  hit-rate=100%
```

**The metric that matters is `hit-rate`** — the fraction of held-out segments whose
predicted parameters would have met the target. R² can look excellent while the hit rate
is poor, and missing the target scores zero, so hit rate is the one to watch. Aim for
95 %+.

`infeasible` rows are segments that could not reach the target at any CRF (very noisy or
very high-motion content). They are excluded from training rather than teaching the model
to aim at something unreachable. A few percent is normal; 30 % means the target is too
high for your encoder/preset.

## 2.5 Artifacts

```
runs/current/dataset.csv        one row per (segment, target) — inspect this
runs/current/dev_summary.json   what ran, plus model metrics
runs/cache/trials.sqlite        every encode+measure result
models/<encoder>/target_<T>/    the trained bundles
```

## 2.6 Interrupting and resuming

Safe to interrupt with Ctrl-C. Every trial is cached by content hash, so re-running skips
everything already measured:

```bash
vidopt dev path/to/corpus --config cpu     # same command; resumes
```

## 2.7 Re-training without re-searching

The search is the expensive part and is already saved. To change a model setting:

```bash
vidopt train runs/current/dataset.csv --set model.crf_quantile=0.10
```

Takes seconds.

---

# 3. Production mode

Fast: no search, no VMAF measurement. Handles **any resolution and any duration**.

## 3.1 Basic usage

```bash
vidopt compress input.mp4 -o output.mp4 --target 89
```

With verification (re-measures the result — costs an extra VMAF pass):

```bash
vidopt compress input.mp4 -o output.mp4 --target 89 --verify
```

```
input      /data/input.mp4
output     /data/output.mp4
encoder    libx265  target VMAF 89
segments   6
size       32.3 MB -> 8.3 MB  (3.87x)
vmaf       88.38  [MISSED]
score      0.0282
elapsed    151.4s
```

| Option | Effect |
|---|---|
| `--target T` | VMAF target. A model must exist for it (`vidopt inspect`) |
| `--verify` | Measure the final VMAF and report the score |
| `--keep-work` | Keep intermediate segments for inspection |
| `--json` | Also print machine-readable output |

## 3.2 What is preserved

| Property | Behaviour |
|---|---|
| **Audio** | Copied from the original, bit-exact, never re-encoded |
| **Subtitles, chapters** | Copied from the original |
| **Bit depth** | 10-bit and 12-bit sources stay 10/12-bit when the encoder supports it |
| **Chroma subsampling** | 4:2:2 and 4:4:4 preserved when supported |
| **Pixel aspect ratio** | Preserved (anamorphic content stays correct) |
| **Duration, frame count** | Exact |

Audio is grafted back from the *original* at the final mux rather than split across
segments, because audio frame boundaries do not align with video frame boundaries and
splitting them is a reliable source of drift. The video timeline is preserved exactly, so
A/V sync is exact.

If the encoder cannot accept the source pixel format, vidopt steps down to the closest
available one and **logs it**:

```
encoder cannot take yuv422p10le for input.mp4; using yuv420p10le instead
```

To normalise everything to 8-bit 4:2:0 instead, set `--set encoder.pix_fmt=yuv420p`.

## 3.3 Any resolution

The model is only reliable within the range it was trained on. Tree models cannot
extrapolate — given a 4K frame after 720p-only training, they return the nearest trained
leaf with no sign that the question was unanswerable. vidopt detects this and says so:

```
WARNING this input is outside the model's training domain: fps (trained 30..30),
        height (trained 720..720), width (trained 1280..1280). Predictions are
        extrapolations and may miss the VMAF target — verify with --verify, and
        re-run `vidopt dev` on a corpus that includes content like this.
```

It still produces valid output — but treat the result as unverified. **The fix is to
include that resolution in the dev corpus**, not to ignore the warning.

## 3.4 Any duration

Long inputs are handled by construction: the video is split at scene cuts (bounded by
`segment.min/max_segment_seconds`), segments are encoded across the worker pool, and each
cut segment is deleted as soon as its encode finishes so peak disk stays near 1× the
input rather than 2×.

For very long files, tune the segment bounds:

```bash
# Feature-length input: longer segments = fewer of them = less per-segment overhead
vidopt compress movie.mkv -o out.mkv --target 89 \
  --set segment.max_segment_seconds=30
```

Very short inputs (below `min_segment_seconds`) become a single segment automatically.

## 3.5 Batch processing

There is no built-in batch command; a shell loop is clearer and gives you control over
failure handling:

```powershell
# Windows — PowerShell
Get-ChildItem C:\data\in\*.mp4 | ForEach-Object {
    vidopt compress $_.FullName -o "C:\data\out\$($_.Name)" --target 89
    if ($LASTEXITCODE -ne 0) { Add-Content C:\data\failures.txt $_.FullName }
}
```

```bash
# Linux / macOS
for f in /data/in/*.mp4; do
  vidopt compress "$f" -o "/data/out/$(basename "$f")" --target 89 \
    || echo "FAILED: $f" >> /data/failures.txt
done
```

Each invocation already uses every worker, so run them sequentially rather than in
parallel.

---

# 4. Tuning

## 4.1 The knob that matters: `model.crf_quantile`

The scoring function drops to **zero** below `target − 5`, so predicting too aggressively
is far more costly than predicting too conservatively. The CRF model is fit with quantile
loss below the median to build in margin.

Margin is mandatory, not optional, for two compounding reasons:

1. The search returns the *highest* CRF that still meets the target, so the training
   labels sit exactly on the boundary with no margin of their own.
2. Whole-video VMAF is the frame-weighted **harmonic** mean over segments, which
   penalises a segment landing low more than an equal overshoot compensates.

| Value | Behaviour |
|---|---|
| `0.50` | Unbiased. Maximum compression, reliably misses the target |
| `0.15` | **Default.** ~+1 CRF of margin |
| `0.10` | Cautious. For content unlike the training corpus |

Measured on the six-clip corpus: `0.35` gave +0.19 CRF of margin and missed all three
targets by ~2 VMAF; `0.15` gave +1.03 CRF and met two of three.

```bash
# Re-tune without re-searching
vidopt train runs/current/dataset.csv --set model.crf_quantile=0.10
```

## 4.2 Search cost vs. model quality

| Setting | Default | Effect |
|---|---|---|
| `search.n_explore` | 12 | Dominant cost. Higher = better AQ coverage |
| `search.n_refine_configs` | 2 | AQ settings that get a CRF bisection |
| `vmaf.n_subsample_search` | 2 | 2 halves measurement cost for <0.5 VMAF of noise |
| `encoder.preset` | medium | `veryfast` for exploration, `slow` for final quality |

## 4.3 Choosing an encoder

Models are **encoder-specific** — a CRF value means something different to each — so
changing the encoder requires re-running dev mode.

| Encoder | Kind | Notes |
|---|---|---|
| `libx265` | CPU | Default. Richest AQ controls (aq-mode 0–4) |
| `libx264` | CPU | Faster, less efficient |
| `libsvtav1` | CPU | Best compression, coarser AQ knob |
| `hevc_nvenc` | GPU | Much faster, somewhat less efficient per bit |
| `av1_nvenc` | GPU | Needs Ada-generation or newer |

```bash
vidopt dev path/to/corpus --set encoder.name=libsvtav1
```

## 4.4 Parallelism

```yaml
jobs:
  cpu_workers: 0        # 0 = auto (cores / ffmpeg_threads_per_job)
  gpu_workers: 1        # set to roughly the number of GPUs
  ffmpeg_threads_per_job: 2
```

Set `gpu_workers` to about the number of GPUs. NVENC limits **concurrent encode sessions
per device**; exceeding the limit fails the encode rather than queueing, which is why GPU
work is gated separately from the CPU worker count.

---

# 5. Operations

## 5.1 Configuration layering

```
built-in defaults  →  --config overlay(s)  →  --set dotted.key=value
```

```bash
vidopt config --list-overlays        # cpu, gpu, quick — and where they live
vidopt config                        # effective config as JSON (stdout is pure JSON)
vidopt config --config gpu | jq .encoder
```

`--config` accepts a shipped overlay name or a path to your own YAML. Unknown keys are
errors, not silent no-ops.

## 5.2 Disk usage

| Stage | Peak |
|---|---|
| Dev mode | corpus + segments + one trial encode per worker |
| Production | ~1× input (segments are deleted as they are encoded) + output |

`runs/cache/trials.sqlite` grows with the number of trials but stores only numbers — a
few MB. Deleting it is safe; it only costs re-measurement.

Removing artifacts is just deleting directories: `runs/` holds the working data and the
trial cache, `models/` holds the trained bundles. Deleting the cache is always safe — it
only costs re-measurement.

## 5.3 Troubleshooting

**`this ffmpeg has no libvmaf filter`** — the resolved ffmpeg is the system one. Most
prebuilt ffmpeg packages omit libvmaf. Check which binary was picked with
`vidopt doctor`; then run `python scripts/setup.py --force`, or point
`VIDOPT_FFMPEG_DIR` at a build that has it.

**`encoder 'hevc_nvenc' is present but failed a one-frame test encode`** — the build has
NVENC but this machine has no usable NVIDIA device. Check `nvidia-smi`, or use
`--set encoder.name=libx265`.

**`no model for encoder 'libx265' at VMAF target 93`** — run `vidopt inspect` to see what
exists. Models are per encoder *and* per target.

**`--verify` says MISSED** — in order: (1) check for the out-of-domain warning, which
means the corpus does not cover this input; (2) lower `model.crf_quantile`; (3) add
sources to the corpus and re-run dev mode.

**`frame-count mismatch measuring ...`** — a guard, not a bug: the output and reference
did not line up frame-for-frame, so the score would be meaningless. Usually the encoder
dropped or duplicated frames.

**Segment retried at a conservative CRF** — one segment's encode failed and was retried
with safer settings so the job could finish. Occasional retries are fine; frequent ones
point at a resource limit (GPU sessions, disk, memory).

---

# 6. Command reference

| Command | Purpose |
|---|---|
| `vidopt doctor` | Check the toolchain and configuration |
| `vidopt dev CORPUS...` | Phase 1: segment, search, train |
| `vidopt train DATASET` | Re-train from an existing dataset |
| `vidopt compress IN -o OUT` | Phase 2: compress with predicted parameters |
| `vidopt inspect` | List trained models and their metrics |
| `vidopt score --vmaf V --ratio R` | Evaluate the objective function directly |
| `vidopt config` | Print the effective configuration |
| `vidopt config --list-overlays` | List shipped config overlays |

Common to all: `--config`, `--set KEY=VALUE`, `--log-level`.

Setup script:

| Command | Purpose |
|---|---|
| `python scripts/setup.py` | Fetch ffmpeg and install the package |
| `python scripts/setup.py --check` | Verify an existing setup |
| `python scripts/setup.py --verify` | Real encode + real VMAF measurement |
| `python scripts/setup.py --skip-ffmpeg` | Only install the Python package |
| `python scripts/setup.py --force` | Re-download ffmpeg |

## A complete first session

```powershell
# Install (Windows PowerShell; Linux/macOS differs only in the activate line)
git clone <repo-url> vidopt
cd vidopt
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/setup.py

# Check
vidopt doctor
python scripts/setup.py --verify

# Learn (hours)
vidopt dev C:/data/corpus --config cpu

# Inspect what was learned
vidopt inspect

# Compress
vidopt compress C:/data/input.mp4 -o C:/data/output.mp4 --target 89 --verify
```
