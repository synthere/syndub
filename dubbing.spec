# -*- mode: python ; coding: utf-8 -*-

added_files = [
         ( 'res/*', 'res' ),
         ('resource/*', 'resource'),
         ('models/whisper/*', 'models/whisper'),
         ('models/paraformer/*', 'models/paraformer'),
         ]
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],

    datas=added_files,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['cv2', 'sudachidict_core'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SynthereDub',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resource/app.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SynthereDub',
)
