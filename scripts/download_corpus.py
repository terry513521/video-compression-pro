#!/usr/bin/env python3
"""Download a mixed free-video training corpus into video/corpus/.

The model interpolates well and extrapolates badly, so the corpus needs *kinds* of
content more than hours of it: animation, CGI film, live action, talking heads, high
motion, outdoor, dark/grainy, graphics, and more than one resolution.

Sources are Creative Commons, public domain, or explicitly free sample clips. Long
films are not kept whole — short high-quality extracts are written as training clips.

    python scripts/download_corpus.py
    python scripts/download_corpus.py --list
    python scripts/download_corpus.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "video" / "corpus"
SOURCES_DIR = CORPUS / "_sources"
MANIFEST = CORPUS / "MANIFEST.json"

# Browser-like UA: several CDNs (Google sample bucket, Mixkit) 403 a custom agent.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
TESTVIDS = "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/"
PEXELS = "https://videos.pexels.com/video-files/"


@dataclass(frozen=True)
class Clip:
    """One training file cut from a downloaded source (or the whole file)."""

    name: str
    kind: str
    start: float = 0.0
    duration: float | None = 22.0
    """None => keep the whole source (after optional max-duration cap)."""


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    license: str
    title: str
    clips: tuple[Clip, ...] = field(default_factory=tuple)
    optional: bool = False
    remote_extract: bool = False
    """If True, ffmpeg reads the URL directly (for huge files we only need a slice)."""


# Curated mix: many content kinds, short clips, free to use for local training.
# Google's gtv-videos-bucket returns 403 from this network; Blender/Wikimedia/Pexels/IA used instead.
CATALOG: tuple[Source, ...] = (
    Source(
        id="bbb_trailer",
        url="https://download.blender.org/peach/trailer/trailer_720p.mov",
        license="CC-BY 3.0 — Blender Foundation (Big Buck Bunny trailer)",
        title="Big Buck Bunny trailer 720p",
        clips=(
            Clip("01_animation_bbb_establishing.mp4", "animation", 2, 20),
            Clip("02_animation_bbb_action.mp4", "animation_high_motion", 20, 20),
        ),
    ),
    Source(
        id="sintel_trailer",
        url="https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4",
        license="CC-BY 3.0 — Blender Foundation (Sintel trailer)",
        title="Sintel trailer 720p",
        clips=(
            Clip("03_cgi_sintel_portrait.mp4", "cgi_portrait", 5, 20),
            Clip("04_cgi_sintel_action.mp4", "cgi_action", 28, 20),
        ),
    ),
    Source(
        id="elephants",
        url="https://archive.org/download/ElephantsDream/ed_1024_512kb.mp4",
        license="CC-BY 2.5 — Blender Foundation / Netherlands Media Art Institute",
        title="Elephants Dream",
        clips=(
            Clip("05_animation_dark_elephants.mp4", "animation_dark", 70, 22),
            Clip("38_animation_dark_elephants_b.mp4", "animation_dark", 20, 22),
            Clip("39_animation_dark_elephants_c.mp4", "animation_dark", 180, 22),
        ),
        optional=True,
    ),
    Source(
        id="caminandes",
        url="https://upload.wikimedia.org/wikipedia/commons/transcoded/d/d0/Caminandes-_Llama_Drama_-_Short_Movie.ogv/Caminandes-_Llama_Drama_-_Short_Movie.ogv.720p.vp9.webm",
        license="CC-BY 3.0 — Blender Foundation (Caminandes: Llama Drama)",
        title="Caminandes: Llama Drama",
        clips=(Clip("06_animation_caminandes.mp4", "animation_stylized", 15, 22),),
        optional=True,
    ),
    Source(
        id="tears",
        url="https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov",
        license="CC-BY 3.0 — Blender Foundation (Tears of Steel)",
        title="Tears of Steel (remote slice)",
        clips=(
            Clip("07_liveaction_tears_of_steel.mp4", "live_action_cgi", 40, 22),
            Clip("36_liveaction_tos_b.mp4", "live_action_cgi", 120, 22),
            Clip("37_liveaction_tos_c.mp4", "live_action_cgi", 200, 22),
        ),
        optional=True,
        remote_extract=True,
    ),
    Source(
        id="notld",
        url="https://archive.org/download/night_of_the_living_dead_ipod/night_of_the_living_dead.mp4",
        license="Public domain — Night of the Living Dead (1968)",
        title="Night of the Living Dead (grain / dark film)",
        clips=(
            Clip("08_grain_dark_notld.mp4", "grain_dark_film", 120, 22),
            Clip("34_grain_dark_notld_b.mp4", "grain_dark_film", 240, 22),
            Clip("35_grain_dark_notld_c.mp4", "grain_dark_film", 480, 22),
        ),
        optional=True,
        remote_extract=True,
    ),
    Source(
        id="pexels_855564",
        url=PEXELS + "855564/855564-hd_1920_1080_24fps.mp4",
        license="Pexels License — free stock video 855564",
        title="Pexels 855564 (live action 1080p)",
        clips=(Clip("09_live_pexels_855564.mp4", "live_action", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_city",
        url=PEXELS + "3571264/3571264-hd_1920_1080_30fps.mp4",
        license="Pexels License — free stock video 3571264",
        title="Pexels city 1080p",
        clips=(Clip("10_urban_city.mp4", "urban_city", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_nature",
        url=PEXELS + "1093662/1093662-hd_1920_1080_30fps.mp4",
        license="Pexels License — free stock video 1093662",
        title="Pexels nature 1080p",
        clips=(Clip("11_nature_landscape.mp4", "nature_landscape", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_water",
        url=PEXELS + "1409899/1409899-hd_1920_1080_25fps.mp4",
        license="Pexels License — free stock video 1409899",
        title="Pexels water 1080p",
        clips=(Clip("12_nature_water.mp4", "nature_water", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_856027",
        url=PEXELS + "856027/856027-hd_1920_1080_25fps.mp4",
        license="Pexels License — free stock video 856027",
        title="Pexels 856027 (1080p)",
        clips=(Clip("13_outdoor_pexels_856027.mp4", "outdoor", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_people",
        url=PEXELS + "5752729/5752729-hd_1920_1080_30fps.mp4",
        license="Pexels License — free stock video 5752729",
        title="Pexels people 1080p",
        clips=(Clip("14_people_pexels_5752729.mp4", "people", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_urban",
        url=PEXELS + "3195394/3195394-hd_1920_1080_25fps.mp4",
        license="Pexels License — free stock video 3195394",
        title="Pexels urban 1080p",
        clips=(Clip("15_urban_pexels_3195394.mp4", "urban", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_landscape",
        url=PEXELS + "3015510/3015510-hd_1920_1080_24fps.mp4",
        license="Pexels License — free stock video 3015510",
        title="Pexels landscape 1080p",
        clips=(Clip("16_landscape_pexels_3015510.mp4", "landscape", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_motion",
        url=PEXELS + "2098989/2098989-hd_1920_1080_30fps.mp4",
        license="Pexels License — free stock video 2098989",
        title="Pexels motion 1080p",
        clips=(Clip("17_motion_pexels_2098989.mp4", "high_motion", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_street",
        url=PEXELS + "3173312/3173312-hd_1920_1080_30fps.mp4",
        license="Pexels License — free stock video 3173312",
        title="Pexels street 1080p",
        clips=(Clip("18_street_pexels_3173312.mp4", "street", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_cars",
        url=INTEL + "car-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Car detection (real camera)",
        clips=(Clip("19_camera_cars.mp4", "real_camera_vehicles", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_people",
        url=INTEL + "person-bicycle-car-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Person / bicycle / car (real camera)",
        clips=(Clip("20_camera_people.mp4", "real_camera_people", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_face",
        url=INTEL + "head-pose-face-detection-female.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Face / head pose (real camera)",
        clips=(Clip("21_camera_face.mp4", "talking_head_closeup", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_one",
        url=INTEL + "one-by-one-person-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="One-by-one person detection",
        clips=(Clip("22_camera_person_walk.mp4", "real_camera_walk", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_bottle",
        url=INTEL + "bottle-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Bottle detection (indoor objects)",
        clips=(Clip("23_camera_indoor_objects.mp4", "indoor_objects", 0, None),),
        optional=True,
    ),
    Source(
        id="bbb_360",
        url=TESTVIDS + "360/Big_Buck_Bunny_360_10s_1MB.mp4",
        license="CC-BY 3.0 — Blender Foundation (10s 360p extract)",
        title="Big Buck Bunny 360p 10s",
        clips=(Clip("24_res_360p_bbb.mp4", "resolution_360p", 0, None),),
        optional=True,
    ),
    Source(
        id="bbb_720",
        url=TESTVIDS + "720/Big_Buck_Bunny_720_10s_2MB.mp4",
        license="CC-BY 3.0 — Blender Foundation (10s 720p extract)",
        title="Big Buck Bunny 720p 10s",
        clips=(Clip("25_res_720p_bbb.mp4", "resolution_720p", 0, None),),
        optional=True,
    ),
    Source(
        id="bbb_1080",
        url=TESTVIDS + "1080/Big_Buck_Bunny_1080_10s_5MB.mp4",
        license="CC-BY 3.0 — Blender Foundation (10s 1080p extract)",
        title="Big Buck Bunny 1080p 10s",
        clips=(Clip("26_res_1080p_bbb.mp4", "resolution_1080p", 0, None),),
        optional=True,
    ),
    Source(
        id="samplelib20",
        url="https://download.samplelib.com/mp4/sample-20s.mp4",
        license="SampleLib free sample video",
        title="SampleLib 20s",
        clips=(Clip("27_samplelib_20s.mp4", "sample_mixed", 0, None),),
        optional=True,
    ),
    Source(
        id="filesamples_640",
        url="https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
        license="FileSamples free sample video",
        title="FileSamples 640x360",
        clips=(Clip("28_res_640x360_sample.mp4", "resolution_360p", 0, None),),
        optional=True,
    ),
    Source(
        id="filesamples_1920",
        url="https://filesamples.com/samples/video/mp4/sample_1920x1080.mp4",
        license="FileSamples free sample video",
        title="FileSamples 1920x1080",
        clips=(Clip("29_res_1080p_sample.mp4", "resolution_1080p", 0, None),),
        optional=True,
    ),
    Source(
        id="learningcontainer",
        url="https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
        license="LearningContainer free sample MP4",
        title="LearningContainer sample MP4",
        clips=(Clip("30_sample_learningcontainer.mp4", "sample_mixed", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_city_1440",
        url=PEXELS + "3571264/3571264-uhd_2560_1440_30fps.mp4",
        license="Pexels License — free stock video 3571264 (1440p)",
        title="Pexels city 1440p",
        clips=(Clip("40_res_1440p_city.mp4", "resolution_1440p", 0, 22),),
        optional=True,
    ),
    Source(
        id="pexels_night",
        url=PEXELS + "854982/854982-hd_1280_720_25fps.mp4",
        license="Pexels License — free stock video 854982",
        title="Pexels night 720p",
        clips=(Clip("41_night_pexels_854982.mp4", "night", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_6981411",
        url=PEXELS + "6981411/6981411-hd_1920_1080_25fps.mp4",
        license="Pexels License — free stock video 6981411",
        title="Pexels 6981411 1080p",
        clips=(Clip("42_pexels_6981411.mp4", "live_action", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_4114797",
        url=PEXELS + "4114797/4114797-hd_1920_1080_25fps.mp4",
        license="Pexels License — free stock video 4114797",
        title="Pexels 4114797 1080p",
        clips=(Clip("43_pexels_4114797.mp4", "live_action", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_857195",
        url=PEXELS + "857195/857195-hd_1280_720_25fps.mp4",
        license="Pexels License — free stock video 857195",
        title="Pexels 857195 720p",
        clips=(Clip("44_pexels_857195.mp4", "live_action", 0, None),),
        optional=True,
    ),
    Source(
        id="pexels_water_1440",
        url=PEXELS + "1409899/1409899-uhd_2560_1440_25fps.mp4",
        license="Pexels License — free stock video 1409899 (1440p)",
        title="Pexels water 1440p",
        clips=(Clip("45_res_1440p_water.mp4", "resolution_1440p", 0, 22),),
        optional=True,
    ),
    Source(
        id="samplelib5",
        url="https://download.samplelib.com/mp4/sample-5s.mp4",
        license="SampleLib free sample video",
        title="SampleLib 5s",
        clips=(Clip("46_samplelib_5s.mp4", "sample_mixed", 0, None),),
        optional=True,
    ),
    Source(
        id="samplelib10",
        url="https://download.samplelib.com/mp4/sample-10s.mp4",
        license="SampleLib free sample video",
        title="SampleLib 10s",
        clips=(Clip("47_samplelib_10s.mp4", "sample_mixed", 0, None),),
        optional=True,
    ),
    Source(
        id="samplelib15",
        url="https://download.samplelib.com/mp4/sample-15s.mp4",
        license="SampleLib free sample video",
        title="SampleLib 15s",
        clips=(Clip("48_samplelib_15s.mp4", "sample_mixed", 0, None),),
        optional=True,
    ),
    Source(
        id="samplelib30",
        url="https://download.samplelib.com/mp4/sample-30s.mp4",
        license="SampleLib free sample video",
        title="SampleLib 30s",
        clips=(Clip("49_samplelib_30s.mp4", "sample_mixed", 0, 22),),
        optional=True,
    ),
    Source(
        id="sintel_480",
        url="https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4",
        license="CC-BY 3.0 — Blender Foundation (Sintel trailer 480p)",
        title="Sintel trailer 480p",
        clips=(Clip("50_res_480p_sintel.mp4", "resolution_480p", 5, 20),),
        optional=True,
    ),
    Source(
        id="bbb_400p",
        url="https://download.blender.org/peach/trailer/trailer_400p.ogg",
        license="CC-BY 3.0 — Blender Foundation (Big Buck Bunny trailer 400p)",
        title="Big Buck Bunny trailer 400p",
        clips=(Clip("51_res_400p_bbb.mp4", "resolution_400p", 2, 20),),
        optional=True,
    ),
    Source(
        id="sintel_10s",
        url="https://test-videos.co.uk/vids/sintel/mp4/h264/720/Sintel_720_10s_1MB.mp4",
        license="CC-BY 3.0 — Blender Foundation (Sintel 10s 720p)",
        title="Sintel 720p 10s",
        clips=(Clip("52_cgi_sintel_10s.mp4", "cgi_film", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_classroom",
        url=INTEL + "classroom.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Classroom (real camera)",
        clips=(Clip("53_camera_classroom.mp4", "indoor_people", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_store",
        url=INTEL + "store-aisle-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Store aisle (real camera)",
        clips=(Clip("54_camera_store.mp4", "indoor_store", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_face_male",
        url=INTEL + "head-pose-face-detection-male.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Face / head pose male",
        clips=(Clip("55_camera_face_male.mp4", "talking_head_closeup", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_people_det",
        url=INTEL + "people-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="People detection (real camera)",
        clips=(Clip("56_camera_people_walk.mp4", "real_camera_people", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_worker",
        url=INTEL + "worker-zone-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Worker zone (real camera)",
        clips=(Clip("57_camera_workers.mp4", "indoor_industrial", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_fruit",
        url=INTEL + "fruit-and-vegetable-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Fruit and vegetable (real camera)",
        clips=(Clip("58_camera_fruit.mp4", "still_life", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_driver",
        url=INTEL + "driver-action-recognition.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Driver action (real camera)",
        clips=(Clip("59_camera_driver.mp4", "in_car", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_face_walk",
        url=INTEL + "face-demographics-walking.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Face demographics walking",
        clips=(Clip("60_camera_face_walk.mp4", "real_camera_walk", 0, None),),
        optional=True,
    ),
    Source(
        id="filesamples_ocean",
        url="https://filesamples.com/samples/video/mp4/sample_960x400_ocean_with_audio.mp4",
        license="FileSamples free sample video",
        title="FileSamples ocean 960x400",
        clips=(Clip("61_ocean_960.mp4", "nature_water", 0, 22),),
        optional=True,
    ),
    Source(
        id="intel_bolt",
        url=INTEL + "bolt-detection.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Bolt detection (real camera)",
        clips=(Clip("62_camera_bolt.mp4", "indoor_objects", 0, None),),
        optional=True,
    ),
    Source(
        id="intel_face_both",
        url=INTEL + "head-pose-face-detection-female-and-male.mp4",
        license="Intel Open Source Technology Center sample video",
        title="Face / head pose two people",
        clips=(Clip("63_camera_faces.mp4", "talking_head_closeup", 0, None),),
        optional=True,
    ),
)


def info(message: str) -> None:
    print(f"==> {message}", flush=True)


def detail(message: str) -> None:
    print(f"    {message}", flush=True)


def ffmpeg_bin() -> str | None:
    env_dir = os.environ.get("VIDOPT_FFMPEG_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser() / "ffmpeg"
        if candidate.is_file():
            return str(candidate)
    vendored = ROOT / "vendor" / "ffmpeg" / "bin" / "ffmpeg"
    if vendored.is_file():
        return str(vendored)
    found = shutil.which("ffmpeg")
    return found


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        total = int(response.headers.get("Content-Length") or 0)
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        t0 = time.time()
        with tmp.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = min(100, done * 100 // total)
                    elapsed = max(0.1, time.time() - t0)
                    mb_s = (done / 1e6) / elapsed
                    print(f"\r    {pct:3d}%  {done/1e6:7.1f} MB  {mb_s:4.1f} MB/s", end="", flush=True)
        print(flush=True)
        tmp.replace(dest)


def source_path(source: Source) -> Path:
    suffix = Path(source.url.split("?", 1)[0]).suffix or ".bin"
    if suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".ogv", ".avi"}:
        suffix = ".mp4"
    return SOURCES_DIR / f"{source.id}{suffix}"


def probe_duration(ffmpeg: str, path: Path) -> float | None:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if not Path(ffprobe).is_file():
        found = shutil.which("ffprobe")
        ffprobe = found or ffprobe
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def extract_clip(ffmpeg: str, src: str | Path, clip: Clip, dest: Path, max_seconds: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = clip.duration if clip.duration is not None else max_seconds
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{clip.start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", "fast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ]
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=180,
        )
    except subprocess.TimeoutExpired:
        dest.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg extract timed out") from None
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 10_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip()[:400] or "ffmpeg extract failed")


def copy_or_cap(ffmpeg: str, src: Path, clip: Clip, dest: Path, max_seconds: float) -> None:
    """Keep a short source as-is (re-wrapped), or cap a long one."""
    duration = probe_duration(ffmpeg, src)
    if clip.duration is None and duration is not None and duration <= max_seconds + 1.0 and src.suffix.lower() == ".mp4":
        shutil.copy2(src, dest)
        return
    extract_clip(ffmpeg, src, clip, dest, max_seconds)


def make_synthetic_graphics(ffmpeg: str, dest: Path) -> None:
    """Screen/graphics content: test pattern + overlay text (no third-party file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # drawtext needs a font; fall back to testsrc2 alone if fonts are missing.
    graph = (
        "testsrc2=size=1280x720:rate=30:duration=20,"
        "drawtext=text='vidopt screen corpus':fontsize=42:fontcolor=white:"
        "x=36:y=40:box=1:boxcolor=black@0.5,"
        "drawtext=text='frame %{n}':fontsize=28:fontcolor=yellow:x=36:y=100"
    )
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", graph,
        "-c:v", "libx264", "-preset", "fast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode == 0 and dest.is_file():
        return
    dest.unlink(missing_ok=True)
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=20",
        "-c:v", "libx264", "-preset", "fast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest.is_file():
        raise RuntimeError(result.stderr.strip()[:400] or "synthetic graphics encode failed")


def _lavfi_clip(ffmpeg: str, dest: Path, lavfi: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", lavfi,
        "-t", "20",
        "-c:v", "libx264", "-preset", "fast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(dest),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest.is_file():
        dest.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip()[:300] or "lavfi encode failed")


def write_manifest(rows: list[dict]) -> None:
    payload = {
        "created_by": "scripts/download_corpus.py",
        "purpose": "vidopt training corpus — mixed free video kinds",
        "clips": rows,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def list_catalog() -> int:
    print(f"{'id':<18} {'kind(s)':<28} title")
    print("-" * 80)
    for source in CATALOG:
        kinds = ",".join(sorted({c.kind for c in source.clips}))
        flag = " (optional)" if source.optional else ""
        print(f"{source.id:<18} {kinds:<28} {source.title}{flag}")
    print()
    print("Plus synthetic: 31_graphics_testsrc.mp4  (generated, not downloaded)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="Print the catalog and exit.")
    parser.add_argument("--limit", type=int, default=None, help="Download at most N sources.")
    parser.add_argument(
        "--max-clip-seconds", type=float, default=22.0,
        help="Cap extracted / whole clips (default 22).",
    )
    parser.add_argument(
        "--purge-sources", action="store_true",
        help="Delete video/corpus/_sources after extracting clips.",
    )
    parser.add_argument(
        "--skip-synthetic", action="store_true",
        help="Do not generate the screen/graphics clip.",
    )
    args = parser.parse_args()

    if args.list:
        return list_catalog()

    ffmpeg = ffmpeg_bin()
    if not ffmpeg:
        print(
            "ERROR: ffmpeg not found. Run ./install.sh first so vendor/ffmpeg exists.",
            file=sys.stderr,
        )
        return 1

    CORPUS.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    sources = list(CATALOG)
    if args.limit is not None:
        sources = sources[: args.limit]

    rows: list[dict] = []
    failures: list[str] = []

    for source in sources:
        dest_src = source_path(source)
        info(f"{source.id}: {source.title}")
        detail(source.license)

        if source.remote_extract:
            for clip in source.clips:
                out = CORPUS / clip.name
                if out.is_file() and out.stat().st_size > 10_000:
                    detail(f"exists {clip.name}")
                else:
                    try:
                        detail(f"ffmpeg slice from URL @ {clip.start:.0f}s")
                        extract_clip(ffmpeg, source.url, clip, out, args.max_clip_seconds)
                        detail(f"wrote {clip.name} ({out.stat().st_size/1e6:.1f} MB)")
                    except Exception as exc:  # noqa: BLE001
                        msg = f"{clip.name}: remote extract failed ({exc})"
                        if source.optional:
                            detail(f"skipping: {msg}")
                            continue
                        failures.append(msg)
                        detail(msg)
                        continue
                if out.is_file():
                    rows.append(
                        {
                            "file": clip.name,
                            "kind": clip.kind,
                            "license": source.license,
                            "source_id": source.id,
                            "title": source.title,
                            "url": source.url,
                            "bytes": out.stat().st_size,
                        }
                    )
            continue

        try:
            if dest_src.is_file() and dest_src.stat().st_size > 50_000:
                detail(f"already have {dest_src.name} ({dest_src.stat().st_size/1e6:.1f} MB)")
            else:
                detail(source.url)
                download(source.url, dest_src)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            msg = f"{source.id}: download failed ({exc})"
            if source.optional:
                detail(f"skipping optional source: {msg}")
                continue
            failures.append(msg)
            detail(msg)
            continue

        for clip in source.clips:
            out = CORPUS / clip.name
            if out.is_file() and out.stat().st_size > 10_000:
                detail(f"exists {clip.name}")
            else:
                try:
                    copy_or_cap(ffmpeg, dest_src, clip, out, args.max_clip_seconds)
                    detail(f"wrote {clip.name} ({out.stat().st_size/1e6:.1f} MB)")
                except Exception as exc:  # noqa: BLE001
                    msg = f"{clip.name}: extract failed ({exc})"
                    if source.optional:
                        detail(f"skipping: {msg}")
                        continue
                    failures.append(msg)
                    detail(msg)
                    continue
            rows.append(
                {
                    "file": clip.name,
                    "kind": clip.kind,
                    "license": source.license,
                    "source_id": source.id,
                    "title": source.title,
                    "url": source.url,
                    "bytes": out.stat().st_size if out.is_file() else 0,
                }
            )

    if not args.skip_synthetic:
        synth = CORPUS / "31_graphics_testsrc.mp4"
        info("synthetic graphics / screen-like pattern")
        try:
            if synth.is_file() and synth.stat().st_size > 10_000:
                detail(f"exists {synth.name}")
            else:
                make_synthetic_graphics(ffmpeg, synth)
                detail(f"wrote {synth.name} ({synth.stat().st_size/1e6:.1f} MB)")
            rows.append(
                {
                    "file": synth.name,
                    "kind": "graphics_screen",
                    "license": "generated locally by ffmpeg testsrc2 (no third-party video)",
                    "source_id": "synthetic",
                    "title": "testsrc2 graphics",
                    "url": None,
                    "bytes": synth.stat().st_size if synth.is_file() else 0,
                }
            )
        except Exception as exc:  # noqa: BLE001
            detail(f"synthetic clip skipped: {exc}")

        extra_synths = (
            (
                CORPUS / "32_dark_grain.mp4",
                "dark_grain",
                "color=c=0x0a0a12:s=1280x720:d=20:r=24,noise=alls=14:allf=t+u",
                "dark grain (synthetic)",
            ),
            (
                CORPUS / "33_high_motion_life.mp4",
                "high_motion_synthetic",
                "life=s=1280x720:rate=30:ratio=0.12:death_color=black:life_color=white",
                "high-motion cellular automata (synthetic)",
            ),
        )
        for path, kind, lavfi, title in extra_synths:
            info(title)
            try:
                if path.is_file() and path.stat().st_size > 10_000:
                    detail(f"exists {path.name}")
                else:
                    _lavfi_clip(ffmpeg, path, lavfi)
                    detail(f"wrote {path.name} ({path.stat().st_size/1e6:.1f} MB)")
                rows.append(
                    {
                        "file": path.name,
                        "kind": kind,
                        "license": "generated locally by ffmpeg lavfi (no third-party video)",
                        "source_id": "synthetic",
                        "title": title,
                        "url": None,
                        "bytes": path.stat().st_size if path.is_file() else 0,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                detail(f"synthetic clip skipped: {exc}")

    write_manifest(rows)

    if args.purge_sources and SOURCES_DIR.is_dir():
        shutil.rmtree(SOURCES_DIR)
        detail("purged video/corpus/_sources")

    print()
    info(f"{len(rows)} clip(s) in {CORPUS}")
    kinds = sorted({row["kind"] for row in rows})
    detail("kinds: " + ", ".join(kinds))
    if failures:
        print()
        print("WARNINGS:")
        for item in failures:
            print(f"  - {item}")
        return 1 if not rows else 0
    print()
    detail("next:  ./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
