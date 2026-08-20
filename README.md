# vidopt

Scene-adaptive video compression with learned encoder parameters.

`vidopt` compresses video by treating every scene separately: it splits the input at
scene cuts, measures what each segment looks like, and predicts the ffmpeg settings that
hit a target VMAF with the smallest possible file. A slow, offline **train mode** searches
for those settings by direct measurement and trains a model; a fast **compress mode**
applies the model with no measurement at all.

**Windows only.** CLI workflow — no web or desktop UI. No Docker, no make — bundled or
system Python plus ffmpeg.

**[SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) — deep dive: architecture, search, models, offline deploy.**
**[START_HERE.txt](START_HERE.txt) — shortest path: extract, train, compress.**
**[OFFLINE_GUIDE.md](OFFLINE_GUIDE.md) — Windows offline: pack, extract, train, compress, repair.**
**[REPAIR.txt](REPAIR.txt) — short checklist if files are damaged.**
**[USAGE.md](USAGE.md) — step-by-step guide: install, train, compress.**
**[COMPRESS_GUIDE.md](COMPRESS_GUIDE.md) — compress-only production package.**

---

## Quick start (offline production zip)

Extract the package, copy training videos into `video\corpus\`, then:

```bat
vidopt.bat doctor
vidopt.bat train video\corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 --resume
vidopt.bat inspect
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder libsvtav1 --level 2 --verify --resume
```

| Flag | Meaning |
|---|---|
| `--level 1` | VMAF target 85 |
| `--level 2` | VMAF target 89 |
| `--level 3` | VMAF target 93 |
| `--resume` | Continue interrupted train or compress (reuses segments, search checkpoints, trial cache) |
| `--encoder NAME` | Must match at train and compress (`libx265`, `libsvtav1`, `hevc_nvenc`, …) |

If you stop training with Ctrl+C, re-run the **same command with `--resume`**. Finished
segment searches are skipped via `runs\current\search_records.jsonl`; individual encode
trials are reused from `runs\cache\trials.sqlite`.

---

## Build the offline bundle (online PC, once)

Self-contained package: **no system Python or ffmpeg required** on the offline PC.

```bat
scripts\prepare_offline_bundle.bat
install.bat
scripts\pack_production.bat          rem -> dist\vidopt-offline-windows-x64.zip
```

After training, pack for offline deployment:

```bat
scripts\pack_production.bat --with-models
rem -> dist\vidopt-offline-windows-x64.zip  (vendor compressed inside)

rem Or full project backup:
scripts\pack_project.bat --with-models
```

Offline PC: **extract → install.bat → vidopt.bat doctor → train/compress**

See [COMPRESS_GUIDE.md](COMPRESS_GUIDE.md) and [USAGE.md §2.9](USAGE.md#29-offline-training-workflow).

---

## Scalability & encoder

| Flag | Default | Meaning |
|---|---|---|
| `--encoder NAME` | `libx265` | Codec: `libx265`, `libx264`, `libsvtav1`, `hevc_nvenc`, `av1_nvenc`, `h264_nvenc` |
| `--cpu-workers N` | `4` | Parallel CPU encodes (`0` = auto from cores) |
| `--gpu-workers N` | `0` | Parallel GPU encodes (`0` = CPU-only) |

```bat
rem CPU train / compress
vidopt.bat train video\corpus --encoder libx265 --level 2 --cpu-workers 4 --resume
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder libx265 --level 2 --verify --resume

rem GPU (NVENC) — raise --gpu-workers to match devices
vidopt.bat train video\corpus --encoder hevc_nvenc --level 2 --gpu-workers 1 --cpu-workers 4 --resume
vidopt.bat compress in.mp4 -o out\out.mp4 --encoder hevc_nvenc --level 2 --gpu-workers 1 --resume
```

Train and compress with the **same** `--encoder` — models live under `models\<encoder>\`.

---

## Repair (offline — damaged environment)

The production PC is **offline**. When the runtime breaks, repair from the
**installable pieces already in the package** — not from the internet.

```bat
install.bat
vidopt.bat doctor
```

Short checklist: **[REPAIR.txt](REPAIR.txt)**. Full guide:
[OFFLINE_GUIDE.md §13](OFFLINE_GUIDE.md#13-repair-damaged-or-missing-files).

---

## How it works

### Train mode — measure, then learn

```
corpus ──► segment by scene ──► extract features ──► search parameters ──► train
             (PySceneDetect)      (8 per segment)      (per VMAF target)    (per target)
```

For every segment and VMAF target, the search finds the smallest file that still reaches
the target VMAF by direct encode + measurement. CLI `train` defaults to **`boundary`**
search (threshold-first AQ refinement). Override with `--set search.strategy=...`.
Every trial is cached in SQLite. See [USAGE.md §2.4](USAGE.md#24-search-algorithms).

### Compress mode — predict, then encode

```
input ──► segment ──► features ──► predict ──► encode in parallel ──► concat ──► output
```

No search, no VMAF in the hot path unless you pass `--verify`.

---

## Commands

| Command | Purpose |
|---|---|
| `vidopt doctor` | Toolchain report; tests encoders by encoding a frame |
| `vidopt train CORPUS...` | Segment corpus, search parameters, train models (`--resume` continues) |
| `vidopt compress IN -o OUT` | Compress with predicted parameters (`--resume` continues) |
| `vidopt inspect` | Trained models and metrics |
| `vidopt score --vmaf V --ratio R` | Evaluate the objective function |
| `vidopt config [--list-overlays]` | Effective configuration |
| `vidopt.bat` | Launcher for bundled Python in the production zip |
| `install.bat` | Offline install or repair |

Common flags: `--encoder`, `--level`, `--cpu-workers`, `--gpu-workers`, `--config`,
`--set key.path=value`, `--log-level`, `--resume`.

Full details: [USAGE.md](USAGE.md), [OFFLINE_GUIDE.md](OFFLINE_GUIDE.md).

---

## Development install (optional)

On a connected Windows machine with Python 3.10+:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/setup.py
vidopt doctor
python scripts/setup.py --verify
```

Point at an existing libvmaf ffmpeg:

```powershell
$env:VIDOPT_FFMPEG_DIR = "C:\ffmpeg\bin"
python scripts/setup.py --skip-ffmpeg
```

```powershell
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

---

## Related docs

- [DESIGN.md](DESIGN.md) — architecture
- [REFERENCE_ANALYSIS.md](REFERENCE_ANALYSIS.md) — reference project analysis
