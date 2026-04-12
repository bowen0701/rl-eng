# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/bowenlee/github/rl-eng/rl_eng', 'rl_eng'), ('/Users/bowenlee/github/rl-eng/runs/tic_tac_toe_20260412_0014_s42', 'runs/tic_tac_toe_20260412_0014_s42'), ('/Users/bowenlee/github/rl-eng/games/tic_tac_toe', 'games/tic_tac_toe')],
    hiddenimports=['rl_eng', 'rl_eng.tic_tac_toe', 'pygame'],
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
    [],
    exclude_binaries=True,
    name='TicTacToe',
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TicTacToe',
)
app = BUNDLE(
    coll,
    name='TicTacToe.app',
    icon=None,
    bundle_identifier=None,
)
