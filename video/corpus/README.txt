Training corpus
===============

Copy training videos into this folder (USB, disk, or another machine).
No network is required. Nested directories are fine.

  copy D:\my_videos\*.mp4 video\corpus\
  robocopy D:\corpus video\corpus /E

Recognised: .mp4 .mkv .mov .webm .y4m .avi .m4v .ts

Aim for 10+ sources that look like production: high and low motion, grain,
dark scenes, animation, screen content, and every resolution you will compress.

Train (offline):
  vidopt.bat train video\corpus --config cpu --encoder libsvtav1 --level 2 --cpu-workers 0 --resume

If interrupted, re-run the same command with --resume.

Models after training: models\<encoder>\target_<VMAF>\
