# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

spec_file = globals().get('SPEC')
base_dir = os.path.abspath(os.path.dirname(spec_file)) if spec_file else os.path.abspath('.')

added_files = [
    ('squig_catalog.json', '.'),
]

# Bundle assets/ cleanly (skipping scratch test_*.svg files and tracking any Equalizer APO installer)
bundled_installer_names = set()
assets_root = os.path.join(base_dir, 'assets')
if os.path.isdir(assets_root):
    for root, _dirs, files in os.walk(assets_root):
        rel_dir = os.path.relpath(root, base_dir)
        for fname in sorted(files):
            low = fname.lower()
            if low.startswith('test_') and low.endswith('.svg'):
                continue
            src_path = os.path.join(root, fname)
            added_files.append((src_path, rel_dir))
            if low.startswith('equalizerapo') and low.endswith('.exe'):
                bundled_installer_names.add(fname)

# Also look for EqualizerAPO*.exe in installers/ or repository root if not already bundled from assets/
if not bundled_installer_names:
    for search_dir in (os.path.join(base_dir, 'installers'), base_dir):
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                low = fname.lower()
                if low.startswith('equalizerapo') and low.endswith('.exe'):
                    src_path = os.path.join(search_dir, fname)
                    added_files.append((src_path, 'assets'))
                    bundled_installer_names.add(fname)
                    break
        if bundled_installer_names:
            break

hidden_imports = [
    'numpy',
    'scipy',
    'scipy.optimize',
    'scipy.interpolate',
    'scipy.special',
    'scipy.signal',
    'scipy.sparse',
    'sounddevice',
    'pyaudiowpatch',
    'pyqtgraph',
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtMultimedia',
    'requests',
    'urllib.parse',
    'sqlite3',
    'hearing_test',
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

upx_exclude_list = sorted({'qwindows.dll', *bundled_installer_names})

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
    # Disabled to prevent Windows Defender false positives and NSIS installer CRC issues
    upx=False,
    upx_exclude=upx_exclude_list,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/clean_apo.ico',
)
