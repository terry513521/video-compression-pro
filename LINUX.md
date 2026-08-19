# vidopt on Linux (CPU-only)

This machine has **no usable NVIDIA GPU**. Use the CPU encoder (`libx265`) and the
shipped `--config cpu` overlay. Windows `.bat` launchers are ignored here.

Related: [README.md](README.md) · [USAGE.md](USAGE.md)

Training is **offline**: copy videos onto the machine, then train. No network is
needed after install.

---

## 1. Install (this box)

```bash
./install.sh
```

What it does:

1. Creates `.venv` with the system `python3` (3.10+)
2. Downloads a **libvmaf** ffmpeg build into `vendor/ffmpeg/` (distribution ffmpeg
   packages almost never include VMAF)
3. Installs vidopt editable into the venv
4. Runs a real encode + VMAF measurement, then `vidopt doctor --config cpu`

Repair / reinstall is the same command: `./install.sh`.

```bash
./vidopt.sh doctor --config cpu
source ./activate_vidopt.sh          # optional: put vidopt on PATH in this shell
```

On a 6-core CPU, leave workers on auto:

```bash
./vidopt.sh ... --cpu-workers 0      # auto: ~3 jobs × 2 ffmpeg threads
```

Or pin `--cpu-workers 3`. Do **not** pass `--gpu-workers` / NVENC encoders here.

---

## 2. Copy the training corpus

Copy videos into `video/corpus/` (USB, external disk, or another folder on the box).
The model interpolates well and extrapolates badly, so the files should look like
what you will compress later: mixed motion, grain, dark scenes, animation, screen
content, and every resolution you will deploy on. Aim for **10+** sources.

```bash
mkdir -p video/corpus
cp /media/usb/*.mp4 video/corpus/
# or a whole tree:
cp -a /path/to/your_videos/. video/corpus/
```

Recognised: `.mp4 .mkv .mov .webm .y4m .avi .m4v .ts`. Nested folders are fine —
`vidopt dev` recurses.

---

## 3. Train, then compress

```bash
./vidopt.sh doctor --config cpu

# plumbing check (~10 min, not for deployment)
./vidopt.sh dev video/corpus --limit 2 --config quick --cpu-workers 0

# real CPU training (hours — expected)
./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0

# after Ctrl-C or a crash: same command plus --resume
./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0 --resume

./vidopt.sh inspect
./vidopt.sh compress in.mp4 -o out/demo.mp4 --target 89 --encoder libx265 --verify
```

Train and compress with the **same** `--encoder`. Models land in
`models/<encoder>/target_<VMAF>/` (for example `models/libx265/target_89/`).

Dev-mode search (not used at compress time):

| Flag | Meaning |
|---|---|
| *(default)* | `search.strategy=aq_then_crf` — enumerate AQ, then CRF-solve |
| `--set search.strategy=coordinate` | AQ neighbour walk after the same screen |
| `--set search.strategy=sample` | 3-D design (`search.sampler=sobol\|lhs\|halton\|random\|grid`) |
| `--set search.strategy=bayes` | Gaussian-process Bayesian optimisation |
| `--set search.strategy=tpe` | Tree-structured Parzen Estimator |
| `--set search.strategy=cmaes` | CMA-style evolution strategy |
| `--set search.crf_solver=brent` | Inverse-quadratic CRF solve (`bisect` / `brent` / `golden`) |

`--resume` continues an interrupted `dev` and reuses the trial cache. Full catalog:
[USAGE.md §2.4](USAGE.md#24-search-algorithms).

---

## 6. Offline end-to-end (train → deploy → compress)

This is the workflow for a **CPU-only offline box** like this one.

### Install once

```bash
./install.sh
./vidopt.sh doctor --config cpu
```

### Train offline (copy corpus first)

```bash
# Recommended: one encoder, VMAF 89, default search, resume-safe
./vidopt.sh dev video/corpus --config cpu --encoder libsvtav1 --cpu-workers 0 \
  --set search.targets='[89]' \
  --set paths.work_dir=runs/production \
  --resume
```

Logs: `runs/production/logs/vidopt.log`. Algorithms explained in
[USAGE.md §2.4](USAGE.md#24-search-algorithms). Full offline workflow:
[USAGE.md §2.9](USAGE.md#29-offline-training-workflow).

Optional research (days): `python scripts/train_matrix.py --resume` — see
[USAGE.md §2.10](USAGE.md#210-optional-algorithm-matrix).

### Inspect

```bash
./vidopt.sh inspect
```

### Compress for production (same encoder)

```bash
./vidopt.sh compress in.mp4 -o out/out.mp4 --target 89 --encoder libsvtav1 --verify
```

### Copy to another Linux machine

```bash
./scripts/pack_compress.sh
# upload dist/vidopt-compress-linux-x64.tar.gz
```

Details: [COMPRESS_GUIDE.md](COMPRESS_GUIDE.md) · [USAGE.md §3.6](USAGE.md#36-deploying-trained-models-offline-production).

---

## 4. Layout

```
.venv/                  Linux virtualenv
vendor/ffmpeg/bin/      ffmpeg + ffprobe (libvmaf) — fetched by install.sh
video/corpus/           training videos (you copy these in)
models/                 trained bundles
runs/                   search cache + datasets
./vidopt.sh             launcher
./install.sh            install / repair
./activate_vidopt.sh    source this for PATH
```

---

## 5. If something breaks

| Symptom | Fix |
|---|---|
| `vidopt.sh`: venv missing | `./install.sh` |
| ffmpeg / libvmaf missing | `./install.sh` (or `python scripts/setup.py --force`) |
| `encoder 'hevc_nvenc' ... no device` | this box is CPU-only — use `--config cpu --encoder libx265` |
| empty corpus | copy videos into `video/corpus/` |
