# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

added_files = [
    ('assets', 'assets'),
    ('squig_catalog.json', '.'),
    ('presets.json', '.'),
]

# Check for EqualizerAPO-x64-1.4.2.exe (or any EqualizerAPO*.exe) in assets/, installers/, and the repository root
base_dir = os.path.abspath(os.path.dirname(SPEC)) if 'SPEC' in locals() else os.path.abspath('.')

installer_search_dirs = [
    (os.path.join(base_dir, 'assets'), 'assets'),
    (os.path.join(base_dir, 'installers'), 'installers'),
    (base_dir, 'assets'),
]

for s_dir, target_dir in installer_search_dirs:
    if os.path.isdir(s_dir):
        for fname in os.listdir(s_dir):
            if fname.lower().startswith("equalizerapo") and fname.lower().endswith(".exe"):
                src_path = os.path.join(s_dir, fname)
                entry = (src_path, target_dir)
                if entry not in added_files and (target_dir, target_dir) not in added_files:
                    added_files.append(entry)
                break

hidden_imports = [
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.interpolate',
    'scipy.special',
    'scipy.signal',
    'sounddevice',
    'pyaudiowpatch',
    'pyqtgraph',
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'requests',
    'urllib.parse',
    'sqlite3',
    'unittest',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CleanAPO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['EqualizerAPO-x64-1.4.2.exe', 'qwindows.dll'],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/clean_apo.ico',
)
