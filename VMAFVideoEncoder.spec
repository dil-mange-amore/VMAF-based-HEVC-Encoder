# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

# Bundle tkinterdnd2 data files (tkdnd tcl/dll resources) so drag & drop works in packaged builds
tkdnd_datas = collect_data_files('tkinterdnd2')

# Project-local binaries to include alongside the exe
local_datas = [
    ('ffmpeg.exe', '.'),
    ('ffprobe.exe', '.'),
    ('avcodec-61.dll', '.'),
    ('avdevice-61.dll', '.'),
    ('avfilter-10.dll', '.'),
    ('avformat-61.dll', '.'),
    ('avutil-59.dll', '.'),
    ('postproc-58.dll', '.'),
    ('swresample-5.dll', '.'),
    ('swscale-8.dll', '.'),
]


a = Analysis(
    ['CubeDrop.py'],
    pathex=[],
    binaries=[],
    datas=tkdnd_datas + local_datas,
    hiddenimports=['tkinterdnd2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VMAFVideoEncoder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # windowed app; no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.png'],
)

# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['CubeDrop.py'],
    pathex=[],
    binaries=[],
    datas=[('ffmpeg.exe', '.'), ('ffprobe.exe', '.'), ('avcodec-61.dll', '.'), ('avdevice-61.dll', '.'), ('avfilter-10.dll', '.'), ('avformat-61.dll', '.'), ('avutil-59.dll', '.'), ('postproc-58.dll', '.'), ('swresample-5.dll', '.'), ('swscale-8.dll', '.')],
    hiddenimports=['tkinterdnd2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='VMAFVideoEncoder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.png'],
)
