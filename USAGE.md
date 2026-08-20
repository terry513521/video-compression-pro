# vidopt — User Guide (Windows)

Step-by-step instructions for installing the environment, running train mode, and running
compress mode on **Windows**. CLI only — no web or desktop UI.

- [README.md](README.md) — overview and reference
- [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) — architecture, search, models, offline deploy (deep dive)
- [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) — offline production zip workflow
- [START_HERE.txt](START_HERE.txt) — shortest quick start
- [DESIGN.md](DESIGN.md) — how it works and why
- [REFERENCE_ANALYSIS.md](REFERENCE_ANALYSIS.md) — analysis of the reference projects

No Docker, no make — bundled or system Python plus ffmpeg. Use `vidopt.bat` in the
production zip, or `vidopt` / `python -m vidopt` from a development install.

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
2. [Train mode](#2-train-mode)
   - [2.4 Search algorithms](#24-search-algorithms)
   - [2.9 Offline training workflow](#29-offline-training-workflow)
   - [2.10 Optional: algorithm matrix](#210-optional-algorithm-matrix)
3. [Compress mode](#3-compress-mode)
   - [3.6 Deploying trained models (offline production)](#36-deploying-trained-models-offline-production)
4. [Tuning](#4-tuning)
5. [Operations](#5-operations)
6. [Command reference](#6-command-reference)

---

# 1. Installing the environment

## 1.1 What you need

| Requirement | Notes |
|---|---|
| Windows 10/11 x64 | Production zip ships its own Python + ffmpeg |
| Python 3.10+ | Only for development install from source ([python.org](https://www.python.org/downloads/) — tick "Add to PATH") |
| Disk | ~1 GB for tools, plus working space (see [§5.2](#52-disk-usage)) |
| NVIDIA GPU | **Optional.** CPU-only is fully supported |

Not needed: Docker, admin rights, a compiler, or a preinstalled ffmpeg (in the production zip).

## 1.2 Install

### Production zip (offline, recommended)

Extract the package. Run `install.bat` only if repair is needed. Then:

```bat
vidopt.bat doctor
```

See [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) and [START_HERE.txt](START_HERE.txt).

### Development install (connected machine)

```powershell
git clone <repo-url> vidopt
cd vidopt
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/setup.py
```

> **PowerShell execution policy.** If `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> Or call `.venv\Scripts\vidopt.exe` directly without activating.

`scripts/setup.py` downloads an ffmpeg build with `libvmaf` into `vendor\ffmpeg\`, checks
encoders, and installs vidopt.

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

Copy the whole project directory across (including `vendor\`), then on the offline machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python scripts/setup.py --skip-ffmpeg
```

Or use the production zip with `install.bat` — see [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md).

## 1.6 Where things live

```
vidopt\
├── vendor\python\          bundled Python (production zip)
├── vendor\ffmpeg\bin\      ffmpeg + ffprobe (libvmaf)
├── models\<encoder>\target_<T>\    trained model bundles
├── runs\                   working artifacts (datasets, caches, segments)
├── src\vidopt\             the package
│   └── configs\*.yaml      shipped configuration
├── scripts\setup.py        environment setup
├── vidopt.bat              launcher (production zip)
└── install.bat             offline install / repair
```

## 1.7 Windows notes

- **Long paths.** Deeply nested working directories can exceed the 260-character limit.
  Either keep the project near the drive root (`C:\vidopt`) or enable long paths:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
- **Antivirus.** Real-time scanning of thousands of short-lived segment files slows train
  mode considerably. Consider excluding the `runs\` directory.
- **Paths in commands.** Forward slashes work everywhere:
  `vidopt compress C:/videos/in.mp4 -o C:/videos/out.mp4`. If you use backslashes, quote
  the path.
- **`vidopt` not found?** Either activate the venv, or call `.venv\Scripts\vidopt.exe`
  directly, or use `python -m vidopt`.

---

# 2. Train mode

Train mode is the expensive, run-it-once step: it **measures** the best encoder parameters
for many scenes and trains a model to predict them. Compress mode then never has to
measure anything.

Run it once per (corpus, encoder, level). It takes hours — that is expected, it is encoding and
scoring thousands of variants.

## 2.1 What it does

```
corpus ─► segment by scene ─► extract features ─► search parameters ─► train models
```

For every segment and every VMAF target (85/89/93 by default), it answers by direct
measurement:

> Which `(crf, aq-mode, aq-strength)` gives the smallest file that still reaches this VMAF?

How those three knobs are searched is `search.strategy`. CLI **`vidopt train` defaults
to `boundary`**. See [§2.4 Search algorithms](#24-search-algorithms).

## 2.2 Preparing a corpus

The corpus should **look like what you will compress in production**. This matters more
than corpus size: the model interpolates well and extrapolates badly, so if you will
compress 4K, train on 4K.

**Copy** the videos onto the machine (USB, external disk, or another folder). There is
no network step:

```powershell
New-Item -ItemType Directory -Force -Path video\corpus
Copy-Item D:\my_videos\*.mp4 video\corpus\ -Force
robocopy D:\corpus video\corpus /E
```

- Point `vidopt train` at files, directories, or both. Directories recurse.
- Recognised: `.mp4 .mkv .mov .webm .y4m .avi .m4v .ts`
- Aim for **10+ source videos** spanning your real content: high and low motion, grain,
  flat animation, dark scenes, screen content.
- Include every **resolution** you will deploy on.

## 2.3 Running it

```bash
# Full run on the 4K corpus, CPU encoder
vidopt train path/to/corpus --config cpu --level 2 --resume

# GPU box: NVENC encoding + CUDA VMAF
vidopt train path/to/corpus --config gpu --level 2 --set jobs.gpu_workers=4 --resume

# Quick plumbing check (~10 min, NOT for deployment)
vidopt train path/to/corpus --limit 2 --config quick --level 2
```

Useful options:

| Option | Effect |
|---|---|
| `--limit N` | Use at most N source videos |
| `--level N` | Quality level shorthand: `1=85`, `2=89`, `3=93` |
| `--set search.strategy=coordinate` | AQ neighbour walk after the AQ screen |
| `--set search.strategy=bayes` | Gaussian-process Bayesian optimisation |
| `--set search.strategy=tpe` | Tree-structured Parzen Estimator |
| `--set search.strategy=cmaes` | CMA-style evolution strategy |
| `--set search.strategy=sample` | 3-D design (`search.sampler=sobol\|lhs\|halton\|random\|grid`) |
| `--set search.crf_solver=brent` | Inverse-quadratic CRF solve instead of secant-bisection |
| `--set search.n_explore=20` | Larger stage-A budget / cap |
| `--set search.targets='[85,89,93]'` | Which VMAF targets to learn |

## 2.4 Search algorithms

Each trial is an encode plus a VMAF measurement. Search runs only in `vidopt train`.
Production (`vidopt compress`) never re-searches: it predicts from the trained model.

CRF is 1-D: **VMAF falls as CRF rises**, so once AQ is fixed the highest feasible CRF
can be found with a 1-D solver. AQ (`aq_mode`, `aq_strength`) is a small discrete set
(for example 8 integer pairs on `libsvtav1`). Every algorithm below is a different way
to spend the stage-A budget on those three knobs, then the same 1-D CRF solve still
runs on the best AQ settings.

Two stages, shared by every VMAF target:

1. **Stage A (explore)** — propose `(crf, aq_mode, aq_strength)` points until
   `search.n_explore` (default 12). Structured strategies may spend about 2× that so
   every AQ can be screened.
2. **Stage B (refine)** — at the best AQ setting(s), run `search.crf_solver` until the
   CRF bracket is narrower than `search.crf_tolerance` (default 0.5).

Trials are cached in SQLite (`runs/cache/trials.sqlite`), so `--resume` and extra VMAF
targets reuse encodes.

### Strategies (`search.strategy`)

| Value | Kind | What it does |
|---|---|---|
| **`boundary`** (CLI default) | Structured | Threshold-first AQ refinement after screening, then CRF solves |
| **`aq_then_crf`** | Structured | Enumerate every AQ pair, screen each at a few CRFs, then 1-D CRF-solve the best |
| **`coordinate`** | Structured | Same AQ screen, then walk 4-neighbours (mode ±1, strength ±1) and re-solve CRF |
| **`sample`** | Space-filling | Draw `n_explore` points in 3-D with `search.sampler`, then CRF-solve the best AQ |
| **`bayes`** | Model-based | Initial design, then a Gaussian process proposes points that look both feasible and compressible |
| **`tpe`** | Model-based | Tree-structured Parzen Estimator: sample where good trials were denser than bad ones |
| **`cmaes`** | Evolutionary | Diagonal CMA-style evolution strategy on the unit cube |

**`aq_then_crf`.** Uses the structure of the problem. AQ is cheap to enumerate (8 pairs
on SVT-AV1, a small float grid on x265). Each pair is probed at `search.n_screen_crfs`
CRFs (default 2). The `search.n_refine_configs` best AQ settings (default 2) then get a
full 1-D CRF solve. This is the right default: it does not waste encodes on a 3-D
random walk when AQ is discrete and CRF is monotone.

**`coordinate`.** Same screen, then a hill-climb on the AQ grid. From the current best
AQ it tries the four neighbours, re-solves CRF at each, and keeps walking for up to
`search.max_coordinate_rounds` (default 4) rounds. Use this when neighbouring AQ
settings interact (typical on x265's float `aq-strength`).

**`sample`.** Ignores AQ structure and covers the 3-D cube with a space-filling design
(`search.sampler`). Then the same CRF solve runs on the best AQ seen. Useful as a
baseline, or when you want a design that is independent of the AQ grid.

**`bayes`.** Fits a Gaussian process (Matern kernel) to VMAF vs. the unit-cube
parameters after an initial design. The next trial maximises
`P(VMAF ≥ target) × normalised CRF + 0.15·σ` — high CRF (smaller files) among points
the model thinks will still hit the target, with a small bonus for uncertainty. The
initial design size is `search.n_init` (0 means `max(4, n_explore/2)`). The GP is
sklearn; if a fit fails, search keeps the trials so far and stops proposing. Adaptive
fitness during explore uses the **first** `search.targets` value; stage B still
refines every target.

**`tpe`.** Bergstra's Tree-structured Parzen Estimator. Rank trials by compression
score, split into a good set (top 25%) and a bad set, fit a diagonal Gaussian to each
in the unit cube, and pick the candidate that maximises `log l(x) − log g(x)` (density
under good vs. bad). Same initial design as `bayes`. Tends to exploit clusters of
good AQ/CRF combinations without assuming VMAF is a smooth GP.

**`cmaes`.** A small diagonal CMA-ES: λ=4 offspring, μ=2 parents, log weights, mean
and per-axis step sizes updated each generation. Samples on the unit cube, then maps
to `(crf, AQ)`. Cheap sequential search when you want an evolutionary method rather
than a surrogate model. Initial design is capped at 4 points so most of the budget
goes to the evolution loop.

### Samplers (`search.sampler`)

Used as the whole of stage A when `strategy=sample`, and as the **initial design**
(and candidate pool) for `bayes` / `tpe` / `cmaes`. All map `[0,1)³` onto
`(crf, aq_mode, aq_strength)`.

| Value | What it does |
|---|---|
| **`sobol`** (default) | Scrambled Sobol sequence. Low discrepancy; good default at tens of trials |
| **`lhs`** | Latin hypercube: one sample per stratum on each axis, then scrambled |
| **`halton`** | Scrambled Halton sequence. Another low-discrepancy construction; similar cover to Sobol |
| **`random`** | Uniform random. Baseline for comparing QMC methods |
| **`grid`** | Regular lattice sized to ≈ `n` points. Deterministic (`search.seed` is ignored) |

At the budgets that matter here (tens of encodes), Sobol / LHS / Halton cover the cube
more evenly than uniform random. `grid` is useful when you want a repeatable lattice
rather than a sequence.

### CRF solvers (`search.crf_solver`)

Stage B, and the CRF re-solve inside `coordinate`. All search for the **highest CRF**
that still meets the VMAF target at fixed AQ. They stop when the bracket is narrower
than `search.crf_tolerance` or after `search.max_bisect_iters` (default 6).

| Value | What it does |
|---|---|
| **`bisect`** (default) | Secant step using the two bracket endpoints, clamped into the open interval. Falls back to the midpoint if VMAF is not monotone in the bracket |
| **`brent`** | Inverse-quadratic interpolation through the last three (CRF, VMAF) points, then the same clamp. Can need fewer encodes when VMAF vs. CRF is smooth |
| **`golden`** | Golden-section split of the bracket (`1/φ`). No VMAF interpolation; slowest to shrink the interval, most robust if the curve is noisy |

### When to use which

| Goal | Setting |
|---|---|
| Production training (recommended) | Default CLI: `boundary`, or `aq_then_crf` via `--set` |
| AQ settings interact (x265 float strength) | `strategy=coordinate` |
| Compare against a 3-D design | `strategy=sample` plus `sampler=sobol` or `lhs` |
| Surrogate model over the cube | `strategy=bayes` (optionally `sampler=lhs`, `n_init=6`) |
| Density-ratio search, no GP | `strategy=tpe` |
| Evolution strategy | `strategy=cmaes` |
| Fewer CRF encodes when VMAF is smooth | `crf_solver=brent` |
| Noisy VMAF (very short segments) | `crf_solver=golden` |

### Examples

```bash
# Default — enumerate AQ, then secant-bisection on CRF
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 --resume

# Neighbour walk after the AQ screen
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 \
  --set search.strategy=coordinate

# 3-D Halton design
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 \
  --set search.strategy=sample --set search.sampler=halton

# Bayesian optimisation with a Latin-hypercube start, Brent CRF solve
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 \
  --set search.strategy=bayes --set search.sampler=lhs --set search.crf_solver=brent

# TPE with a larger initial design
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 \
  --set search.strategy=tpe --set search.n_init=6

# CMA-ES
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 \
  --set search.strategy=cmaes
```

Changing `search.strategy`, `search.sampler`, or `search.crf_solver` needs a new
`vidopt train` run (or `--resume` on an unfinished run).

## 2.5 Reading the output

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

## 2.6 Artifacts

```
runs\current\dataset.csv        one row per (segment, target) — inspect this
runs\current\search_records.jsonl   segment search checkpoints (--resume)
runs\cache\trials.sqlite        every encode+measure result
models\<encoder>\target_<T>\    the trained bundles
```

## 2.7 Interrupting and resuming

Safe to interrupt with Ctrl+C. Individual encode+VMAF trials are cached in
`runs\cache\trials.sqlite`. Finished segments are checkpointed in
`runs\<work_dir>\search_records.jsonl`.

Continue with **`--resume`** and the same encoder, level, and work directory:

```powershell
vidopt train video\corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 --resume
```

With `--resume`:

1. Existing scene cuts are reused from `segments.json`.
2. Segments already in `search_records.jsonl` for this encoder + VMAF target are skipped.
3. Individual trials still hit the SQLite cache even for partially searched segments.

Without `--resume`, search runs again for every segment (cached trials still skip
re-encoding). Always use `--resume` after a crash or Ctrl+C.

## 2.9 Offline training workflow

Training is designed to run **without network access** after the environment is installed.
This is the recommended path for a production deployment — one encoder, one search
algorithm (CLI default **`boundary`**), and the VMAF target(s) you will compress with.

### Phase 1 — Install once (online or air-gapped)

**Windows offline zip:** extract the production package; run `install.bat` only if
repair is needed (see [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md)).

**Air-gapped from source:** copy the whole project including `vendor\wheelhouse\` and
`vendor\ffmpeg\`, then `python scripts/setup.py --skip-ffmpeg` on the offline machine
(see [§1.5](#15-offline--air-gapped-machines)).

### Phase 2 — Copy the corpus (offline)

Copy videos onto the training machine (USB, disk, robocopy). They must **look like production
content** — same resolutions, motion types, grain, animation vs live action. Put them in
`video\corpus\` (or any path you pass to `vidopt train`).

### Phase 3 — Train (offline, hours)

Pick **one encoder** and train with the default search unless you have a reason to change
it (see [§2.4](#24-search-algorithms)):

```bat
vidopt.bat train video\corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 `
  --set paths.work_dir=runs/production --resume
```

What you get:

```
models/<encoder>/target_89/
  metadata.json       metrics, feature ranges, training sources
  crf.joblib          CRF regression head
  aq_mode.joblib      AQ mode classifier
  aq_strength.joblib  AQ strength regression head
runs/<work_dir>/dataset.csv
runs/cache/trials.sqlite   encode+VMAF cache (safe to keep; speeds re-runs)
```

**Interrupt safely:** Ctrl-C, then re-run the **same command with `--resume`**. Finished
segments are checkpointed in `<work_dir>/search_records.jsonl`; trials are in SQLite.

**Logs:** train mode writes to `<work_dir>\logs\vidopt.log` (and `worker-<pid>.log`
with multiple workers). Use `--log-file PATH` to override.

### Phase 4 — Inspect before you deploy

```bash
vidopt inspect
```

Watch **hit-rate** on held-out segments (aim for 95 %+). If `--verify` often misses the
target in production, re-train with a lower `model.crf_quantile` — no re-search needed:

```bash
vidopt train video/corpus --encoder libsvtav1 --level 2 --set model.crf_quantile=0.10 --resume
```

### Phase 5 — Compress in production (offline, minutes)

Use the **same `--encoder`** as training. Production never searches; it loads
`models/<encoder>/target_<T>/`:

```bash
vidopt compress input.mp4 -o output.mp4 --encoder libsvtav1 --level 2 --verify --resume
```

See [§3.6](#36-deploying-trained-models-offline-production) for copying models to another
machine.

## 2.10 Optional: algorithm matrix

For **research or comparison** (not required for production), you can train every search
algorithm × every CPU encoder at one VMAF target. This takes **days** on a large corpus.

```bash
python scripts/train_matrix.py --resume
```

Each combo gets isolated paths so nothing overwrites a production model:

| Artifact | Path |
|---|---|
| Work dir | `runs/matrix/<encoder>/<strategy>/` |
| Model bundle | `models/matrix/<strategy>/<encoder>/target_89/` |
| Driver log | `runs/matrix/logs/matrix.log` |
| Combo log | `runs/matrix/logs/<encoder>__<strategy>.log` |
| Per-segment summary | `runs/matrix/logs/<encoder>__<strategy>.segments.log` |
| Status | `runs/matrix/status.json` |

Order: `libsvtav1` → `libx265` → `libx264`, each with
`aq_then_crf`, `coordinate`, `sample`, `bayes`, `tpe`, `cmaes`. GPU encoders are skipped
when no NVIDIA device is present. On the first error the script **stops** (no partial
model). Fix and re-run with `--resume`.

**For production, skip the matrix.** Train once with default `aq_then_crf`:

```bash
vidopt train video/corpus --config cpu --encoder libsvtav1 --level 2 \
  --set paths.work_dir=runs/production --resume
```

Log index: `runs/matrix/logs/INDEX.txt`.

---

# 3. Compress mode

Fast: no search, no VMAF measurement. Handles **any resolution and any duration**.

## 3.1 Basic usage

```bash
vidopt compress input.mp4 -o output.mp4 --level 2
```

With verification (re-measures the result — costs an extra VMAF pass):

```bash
vidopt compress input.mp4 -o output.mp4 --level 2 --verify
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
| `--level N` | Quality level: `1=85`, `2=89`, `3=93` |
| `--target T` | Explicit VMAF target (optional; `--level` is preferred) |
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
        re-run `vidopt train` on a corpus that includes content like this.
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
vidopt compress movie.mkv -o out.mkv --level 2 \
  --set segment.max_segment_seconds=30
```

Very short inputs (below `min_segment_seconds`) become a single segment automatically.

## 3.5 Batch processing

There is no built-in batch command; a PowerShell loop is clearer:

```powershell
Get-ChildItem C:\data\in\*.mp4 | ForEach-Object {
    vidopt compress $_.FullName -o "C:\data\out\$($_.Name)" --level 2
    if ($LASTEXITCODE -ne 0) { Add-Content C:\data\failures.txt $_.FullName }
}
```

Each invocation already uses every worker, so run them sequentially rather than in
parallel.

## 3.6 Deploying trained models (offline production)

Production compress needs **vidopt + ffmpeg + the model bundle**. It does **not** need
the training corpus, `runs/`, or the trial cache unless you want to re-train.

### What a model bundle contains

```
models/libsvtav1/target_89/
  metadata.json       schema, feature names/ranges, hit-rate, training sources
  crf.joblib
  aq_mode.joblib
  aq_strength.joblib
```

`vidopt compress` discovers bundles under `models/<encoder>/target_<T>/` by default
(`paths.models_dir` in config). Override with `--models-dir` if needed.

### Windows compress package

After `install.bat` and training:

```bat
scripts\pack_compress.bat
rem -> dist\vidopt-compress-windows-x64.zip
```

Extract on the offline PC — no corpus, no training scripts. See [COMPRESS_GUIDE.md](COMPRESS_GUIDE.md).

### Rules that prevent silent mistakes

| Rule | Why |
|---|---|
| Same `--encoder` at train and compress | CRF means different things per codec |
| `--target` must exist in `models/` | `vidopt inspect` lists available targets |
| Corpus should match production resolution/content | Model extrapolates badly outside training domain (see [§3.3](#33-any-resolution)) |
| `--verify` on QA passes | Confirms whole-file VMAF; omit in bulk production for speed |

Matrix-trained models live under `models/matrix/<strategy>/<encoder>/target_89/`. Point
compress at them explicitly:

```bash
vidopt compress in.mp4 -o out.mp4 --encoder libsvtav1 --level 2 \
  --models-dir models/matrix/aq_then_crf
```

For normal deployment use the canonical path: `models/<encoder>/target_<T>/`.

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
vidopt train video/corpus --level 2 --set model.crf_quantile=0.10 --resume
```

## 4.2 Search cost vs. model quality

| Setting | Default | Effect |
|---|---|---|
| `search.strategy` | `aq_then_crf` | How stage A proposes points. See [§2.4](#24-search-algorithms) |
| `search.sampler` | `sobol` | Space-filling design for `sample` and for bayes/tpe/cmaes init (`sobol`, `lhs`, `halton`, `random`, `grid`) |
| `search.crf_solver` | `bisect` | 1-D CRF at fixed AQ: `bisect`, `brent`, or `golden` |
| `search.n_explore` | 12 | Stage-A budget / cap |
| `search.n_init` | 0 | Initial design for bayes/tpe/cmaes. 0 ⇒ `max(4, n_explore/2)` |
| `search.n_refine_configs` | 2 | AQ settings that get a full 1-D CRF solve |
| `search.n_screen_crfs` | 2 | CRF probes per AQ during screening |
| `search.n_strength_steps` | 5 | Float AQ-strength grid (`libx265`); integer-strength encoders ignore this |
| `search.max_coordinate_rounds` | 4 | Neighbour-walk iterations (`coordinate` only) |
| `vmaf.n_subsample_search` | 2 | 2 halves measurement cost for <0.5 VMAF of noise |
| `encoder.preset` | medium | `veryfast` for exploration, `slow` for final quality |

Changing `search.strategy` / `sampler` / `crf_solver` needs a new `vidopt train`
run (or `--resume` on an unfinished run). `vidopt compress` never searches.

## 4.3 Choosing an encoder

Models are **encoder-specific** — a CRF value means something different to each — so
changing the encoder requires re-running train mode.

| Encoder | Kind | Notes |
|---|---|---|
| `libx265` | CPU | Default. Richest AQ controls (aq-mode 0–4) |
| `libx264` | CPU | Faster, less efficient |
| `libsvtav1` | CPU | Best compression, coarser AQ knob |
| `hevc_nvenc` | GPU | Much faster, somewhat less efficient per bit |
| `av1_nvenc` | GPU | Needs Ada-generation or newer |

```bash
vidopt train path/to/corpus --set encoder.name=libsvtav1 --level 2 --resume
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
| Train mode | corpus + segments + one trial encode per worker |
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
sources to the corpus and re-run train mode.

**`frame-count mismatch measuring ...`** — a guard, not a bug: the output produced
*extra* frames (timestamp padding), so the score would compare the wrong pictures.
A few frames short at the end (typical of SVT-AV1 on stream-copied cuts) is allowed.

**Segment retried at a conservative CRF** — one segment's encode failed and was retried
with safer settings so the job could finish. Occasional retries are fine; frequent ones
point at a resource limit (GPU sessions, disk, memory).

---

# 6. Command reference

| Command | Purpose |
|---|---|
| `vidopt doctor` | Check the toolchain and configuration |
| `vidopt train CORPUS...` | Phase 1: segment, search, train (`--resume` continues) |
| `vidopt compress IN -o OUT` | Phase 2: compress (`--resume` continues) |
| `vidopt inspect` | List trained models and their metrics |
| `vidopt score --vmaf V --ratio R` | Evaluate the objective function directly |
| `vidopt config` | Print the effective configuration |
| `vidopt config --list-overlays` | List shipped config overlays |

Common to all: `--config`, `--set KEY=VALUE` (e.g. `search.strategy=bayes`), `--level`, `--log-level`.

Setup script:

| Command | Purpose |
|---|---|
| `python scripts/setup.py` | Fetch ffmpeg and install the package |
| `python scripts/setup.py --check` | Verify an existing setup |
| `python scripts/setup.py --verify` | Real encode + real VMAF measurement |
| `python scripts/setup.py --skip-ffmpeg` | Only install the Python package |
| `python scripts/setup.py --force` | Re-download ffmpeg |

## A complete first session (Windows)

```powershell
# Development install
git clone <repo-url> vidopt
cd vidopt
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/setup.py

vidopt doctor
python scripts/setup.py --verify

# Copy videos into video\corpus\ first, then learn (hours)
vidopt train video\corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 --resume

vidopt inspect

vidopt compress C:\data\input.mp4 -o C:\data\output.mp4 --encoder libsvtav1 --level 2 --verify --resume
```

Production zip users: skip venv setup; use `vidopt.bat` instead of `vidopt`. See
[START_HERE.txt](START_HERE.txt).
