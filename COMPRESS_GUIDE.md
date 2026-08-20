vidopt — compress-only package (offline production, Windows)
============================================================

This archive contains everything to **compress videos** with pre-trained models.
It does **not** include a training corpus or search tools.

Unpack, then:

```bat
cd vidopt-compress-windows-x64
install.bat
vidopt.bat doctor
vidopt.bat inspect
vidopt.bat compress in.mp4 -o out\output.mp4 --encoder libsvtav1 --level 2 --verify
```

`install.bat` extracts the bundled `vendor-windows-x64.zip` on first run.

Rules
-----
  --encoder must match the folder under models\ (see vidopt inspect)
  --level sets the VMAF floor (1=85, 2=89, 3=93); or use --target explicitly
  Put input videos anywhere; only -o out\ is used by default
  --resume continues an interrupted compress run

Build this package (on a machine that finished training):

```bat
scripts\pack_compress.bat
rem -> dist\vidopt-compress-windows-x64.zip
```

Models shipped: see PACKAGE.json

Repair: REPAIR.txt (install.bat)

Full guide: USAGE.md section 3.6 · OFFLINE_GUIDE.md section 10.5
