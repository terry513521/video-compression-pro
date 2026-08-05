Place training / input videos here.
  vidopt.bat dev video\corpus --config cpu --set jobs.cpu_workers=3
  vidopt.bat compress video\corpus\in.mp4 -o out\out.mp4 --target 89 --verify
