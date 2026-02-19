# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Audio Visualizer GUI.
Build with: pyinstaller audio_visualizer_gui.spec --clean
"""

import importlib
import os

# Get the bundled ffmpeg binary from imageio-ffmpeg
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

a = Analysis(
    ['audio_visualizer_gui.py'],
    pathex=[],
    binaries=[(ffmpeg_exe, 'imageio_ffmpeg/binaries')],
    datas=[],
    hiddenimports=[
        'scipy._lib.messagestream',
        'scipy.signal',
        'scipy.ndimage',
        'librosa',
        'librosa.display',
        'librosa.feature',
        'librosa.onset',
        'librosa.beat',
        'librosa.effects',
        'audioread',
        'soundfile',
        'PIL',
        'PIL.Image',
        'sklearn',
        'sklearn.utils',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors',
        'sklearn.neighbors._typedefs',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree',
        'sklearn.tree._utils',
        'imageio_ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioVisualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AudioVisualizer',
)
