Training corpus
===============

Copy training videos into this folder (USB, disk, or another machine).
No network is required. Nested directories are fine.

  Linux:    cp /media/usb/*.mp4 video/corpus/
            cp -a /path/to/your_videos/. video/corpus/
  Windows:  copy D:\my_videos\*.mp4 video\corpus\

Recognised: .mp4 .mkv .mov .webm .y4m .avi .m4v .ts

Aim for 10+ sources that look like production: high and low motion, grain,
dark scenes, animation, screen content, and every resolution you will compress.

Train (Linux CPU):
  ./vidopt.sh dev video/corpus --config cpu --encoder libx265 --cpu-workers 0

Models after training: models/<encoder>/target_<VMAF>/
