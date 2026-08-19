vidopt — compress-only package (offline production)
===================================================

This archive contains everything to **compress videos** with pre-trained models.
It does **not** include a training corpus or dev-mode search tools.

Unpack, then:

Linux
-----
  tar xzf vidopt-compress-linux-x64.tar.gz
  cd vidopt-compress-linux-x64
  ./vidopt.sh doctor --config cpu
  ./vidopt.sh inspect
  ./vidopt.sh compress input.mp4 -o out/output.mp4 --target 89 --encoder libsvtav1 --verify

Windows
-------
  Extract vidopt-compress-windows-x64.zip
  cd vidopt-compress-windows-x64
  vidopt.bat doctor
  vidopt.bat inspect
  vidopt.bat compress in.mp4 -o out\output.mp4 --target 89 --encoder libsvtav1 --verify

Rules
-----
  --encoder must match the folder under models\ (see vidopt inspect)
  --target is the desired VMAF floor (model input; must be within training range)
  Put input videos anywhere; only -o out\ is used by default

Models shipped: see PACKAGE.json

Full guide: COMPRESS_GUIDE.md
Repair:     REPAIR.txt (Linux: ./install.sh  Windows: install.bat)
