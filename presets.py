"""
Preset management, JSON persistence, Equalizer APO file writer, and universal EQ importer.
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import copy
import shutil
import configparser
from typing import Dict, List, Optional, Tuple, Any
from PyQt6 import QtCore
from biquad import EQBand, FilterType


def get_app_data_dir() -> str:
    """Returns directory for user settings and presets. Supports portable mode."""
    base_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))
    portable_marker = os.path.join(base_dir, "portable.txt")
    local_presets = os.path.join(base_dir, "presets.json")
    if os.path.exists(portable_marker) or (os.path.exists(local_presets) and os.access(base_dir, os.W_OK) and not getattr(sys, 'frozen', False)):
        return base_dir
    appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
    data_dir = os.path.join(appdata, "CleanAPO")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_settings_file() -> str:
    return os.path.join(get_app_data_dir(), "settings.json")


def get_presets_file() -> str:
    return os.path.join(get_app_data_dir(), "presets.json")


def get_fallback_output_path() -> str:
    return os.path.join(get_app_data_dir(), "eq_output.txt")


DEFAULT_APO_PATH = r"C:\Program Files\EqualizerAPO\config\clean_apo_generated.txt"
LEGACY_APO_PATH = r"C:\Program Files\EqualizerAPO\config\peqdb_generated.txt"
LOCAL_FALLBACK_PATH = get_fallback_output_path()
PRESETS_FILE = get_presets_file()
SETTINGS_FILE = get_settings_file()


def clean_apo_device_name(dev_name: Optional[str]) -> str:
    """
    Cleans audio device name for Equalizer APO directive matching.
    Removes PyAudio '(Loopback)' artifact, quotes, and WASAPI prefixes.
    Returns 'all' for auto-follow/default, or an unquoted clean substring for Equalizer APO.
    """
    if not dev_name:
        return "all"
    name = dev_name.strip()
    if not name or "auto-follow" in name.lower() or "default" in name.lower() or name.lower() == "all":
        return "all"

    # 1. Remove (Loopback), [Loopback], (WASAPI), and quotes
    name = re.sub(r"\s*[\(\[]\s*(?:Loopback|WASAPI)\s*[\)\]]", "", name, flags=re.IGNORECASE).strip()
    name = name.strip('"\'')
    if not name or "auto-follow" in name.lower() or "default" in name.lower() or name.lower() == "all":
        return "all"

    # 2. Strip Windows endpoint hardware prefix numbering like "1- ", "2- " from the start of the name or within leading brackets
    name = re.sub(r"^[0-9]+-\s*", "", name)
    name = re.sub(r"([\(\[])\s*[0-9]+-\s*", r"\1", name)

    # 3. Preserve the full friendly device name without truncating at inner closing parentheses
    return name.strip()


def can_write_to_path(path: str) -> bool:
    """Checks if the current process has write permissions for the specified file path or its parent directory."""
    if not path:
        return False
    try:
        if os.path.exists(path):
            if os.path.isdir(path):
                test_file = os.path.join(path, f".clean_apo_perm_{os.getpid()}.tmp")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return True
            else:
                with open(path, "a", encoding="utf-8"):
                    pass
                return True
        else:
            folder = os.path.dirname(path)
            if not folder:
                folder = "."
            os.makedirs(folder, exist_ok=True)
            test_file = os.path.join(folder, f".clean_apo_perm_{os.getpid()}.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
    except Exception:
        return False


def grant_apo_directory_permissions(cfg_dir: Optional[str] = None) -> Tuple[bool, str]:
    """
    Elevates via UAC to grant standard user (Users group) Modify permissions
    to the Equalizer APO config directory using Windows icacls.
    """
    if not cfg_dir:
        cfg_dir = get_apo_config_dir() or r"C:\Program Files\EqualizerAPO\config"
    if not os.path.isdir(cfg_dir):
        return False, "Equalizer APO config directory does not exist."
    try:
        import ctypes
        # *S-1-5-32-545 is the Builtin\Users SID (locale-independent across all Windows languages)
        cmd = f'/c icacls "{cfg_dir}" /grant *S-1-5-32-545:(OI)(CI)M /T'
        res = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe", cmd, None, 0
        )
        if res > 32:
            return True, "Permissions granted successfully via UAC."
        elif res == 5:
            return False, "UAC elevation was cancelled by user."
        return False, f"Elevation failed (Error code {res})."
    except Exception as e:
        return False, f"Failed granting permissions: {e}"


def ensure_apo_root_config_include(apo_file_path: str) -> bool:
    """
    Ensures Equalizer APO's main config.txt includes the target APO file.
    Creates config.txt if it does not exist yet.
    Guarantees proper newline separation and cleans up legacy peqdb directives.
    Supports relative include (if in same config dir) or absolute include (e.g. AppData path).
    Returns True if config.txt contains the include directive.
    """
    try:
        target_dir = os.path.dirname(apo_file_path)
        config_txt = os.path.join(target_dir, "config.txt") if target_dir else ""
        if not (config_txt and os.path.exists(config_txt)):
            cfg_dir = get_apo_config_dir()
            if cfg_dir and os.path.isdir(cfg_dir):
                config_txt = os.path.join(cfg_dir, "config.txt")
            elif os.path.isdir(r"C:\Program Files\EqualizerAPO\config"):
                config_txt = r"C:\Program Files\EqualizerAPO\config\config.txt"
            elif target_dir and os.path.isdir(target_dir):
                config_txt = os.path.join(target_dir, "config.txt")
            else:
                return False

        cfg_folder = os.path.normcase(os.path.abspath(os.path.dirname(config_txt)))
        apo_folder = os.path.normcase(os.path.abspath(os.path.dirname(apo_file_path)))
        base_name = os.path.basename(apo_file_path)

        if cfg_folder == apo_folder:
            directive = f"Include: {base_name}"
        else:
            directive = f"Include: {os.path.abspath(apo_file_path)}"

        # If config.txt does not exist, create it with include directive
        if not os.path.exists(config_txt):
            try:
                os.makedirs(os.path.dirname(config_txt), exist_ok=True)
                with open(config_txt, "w", encoding="utf-8") as f:
                    f.write(f"# Equalizer APO Main Configuration\n{directive}\n")
                return True
            except (PermissionError, OSError):
                ok, _ = grant_apo_directory_permissions(os.path.dirname(config_txt))
                if ok:
                    try:
                        with open(config_txt, "w", encoding="utf-8") as f:
                            f.write(f"# Equalizer APO Main Configuration\n{directive}\n")
                        return True
                    except Exception:
                        pass
                return False

        with open(config_txt, "r", encoding="utf-8", errors="replace") as f:
            existing_lines = [line.strip() for line in f.read().splitlines() if line.strip()]

        changed = False
        # Remove legacy peqdb_generated.txt include so filters aren't processed twice
        cleaned_lines = []
        for line in existing_lines:
            if "peqdb_generated.txt" in line and base_name == "clean_apo_generated.txt":
                changed = True
                continue
            cleaned_lines.append(line)

        # Check case-insensitive match for the directive
        if not any(directive.lower() == line.lower() for line in cleaned_lines):
            cleaned_lines.append(directive)
            changed = True

        if changed:
            try:
                with open(config_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(cleaned_lines) + "\n")
            except (PermissionError, OSError):
                ok, _ = grant_apo_directory_permissions(os.path.dirname(config_txt))
                if ok:
                    try:
                        with open(config_txt, "w", encoding="utf-8") as f:
                            f.write("\n".join(cleaned_lines) + "\n")
                        return True
                    except Exception:
                        pass
                return False
        return True
    except Exception as e:
        print(f"[PresetManager] Notice: Could not update config.txt: {e}")
        return False


def get_apo_install_dir() -> Optional[str]:
    """Finds the installed Equalizer APO directory from registry or filesystem."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\EqualizerAPO") as k:
            val, _ = winreg.QueryValueEx(k, "InstallPath")
            if val and os.path.isdir(val):
                return val
    except Exception:
        pass

    candidates = [
        r"C:\Program Files\EqualizerAPO",
        r"C:\Program Files (x86)\EqualizerAPO",
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "EqualizerAPO.dll"))
            or os.path.exists(os.path.join(c, "Configurator.exe"))
            or os.path.exists(os.path.join(c, "DeviceSelector.exe"))
        ):
            return c
    return None


def get_apo_config_dir() -> Optional[str]:
    """Finds Equalizer APO config directory."""
    inst = get_apo_install_dir()
    if inst:
        cfg = os.path.join(inst, "config")
        if os.path.isdir(cfg):
            return cfg
    if os.path.isdir(r"C:\Program Files\EqualizerAPO\config"):
        return r"C:\Program Files\EqualizerAPO\config"
    return None


def is_equalizer_apo_installed() -> bool:
    """Returns True if Equalizer APO is installed on this machine."""
    return get_apo_install_dir() is not None


def get_bundled_apo_installer_path() -> Optional[str]:
    """Finds the bundled Equalizer APO installer executable if present."""
    base_dirs = [
        getattr(sys, "_MEIPASS", ""),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__)),
    ]
    names = [
        "EqualizerAPO-x64-1.4.2.exe",
        "EqualizerAPO-x64.exe",
        "EqualizerAPO_setup.exe",
    ]
    for b in base_dirs:
        if not b:
            continue
        for n in names:
            p = os.path.join(b, "assets", n)
            if os.path.isfile(p):
                return p
            p_root = os.path.join(b, n)
            if os.path.isfile(p_root):
                return p_root
        assets_dir = os.path.join(b, "assets")
        if os.path.isdir(assets_dir):
            try:
                for f in os.listdir(assets_dir):
                    if f.lower().startswith("equalizerapo") and f.lower().endswith(".exe"):
                        return os.path.join(assets_dir, f)
            except Exception:
                pass
    return None


def launch_apo_installer(installer_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Safely extracts (if running from PyInstaller bundle) and launches the
    Equalizer APO installer with Administrator elevation (UAC prompt).
    Returns (success: bool, message: str).
    """
    import ctypes
    import tempfile
    import shutil

    if not installer_path:
        installer_path = get_bundled_apo_installer_path()

    if not installer_path or not os.path.isfile(installer_path):
        return False, "Equalizer APO installer was not found in application bundle."

    try:
        temp_dir = tempfile.gettempdir()
        target_name = os.path.basename(installer_path)
        target_installer = os.path.join(temp_dir, target_name)

        if os.path.abspath(installer_path) != os.path.abspath(target_installer):
            try:
                shutil.copy2(installer_path, target_installer)
                run_path = target_installer
            except Exception:
                run_path = installer_path
        else:
            run_path = installer_path

        res = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", run_path, None, temp_dir, 1
        )
        if res > 32:
            return True, "Equalizer APO installer launched successfully."
        elif res == 5:
            return False, "Administrator elevation was cancelled by user."
        else:
            res2 = ctypes.windll.shell32.ShellExecuteW(
                None, "open", run_path, None, temp_dir, 1
            )
            if res2 > 32:
                return True, "Equalizer APO installer launched."
            return False, f"Could not launch installer (Error code {res})."
    except Exception as e:
        return False, f"Failed to launch installer: {e}"


def get_apo_configurator_path() -> Optional[str]:
    """Finds Equalizer APO DeviceSelector.exe or Configurator.exe path."""
    inst = get_apo_install_dir()
    if inst:
        for name in ["DeviceSelector.exe", "Configurator.exe"]:
            p = os.path.join(inst, name)
            if os.path.exists(p):
                return p

    candidates = [
        r"C:\Program Files\EqualizerAPO\DeviceSelector.exe",
        r"C:\Program Files\EqualizerAPO\Configurator.exe",
        r"C:\Program Files (x86)\EqualizerAPO\DeviceSelector.exe",
        r"C:\Program Files (x86)\EqualizerAPO\Configurator.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def launch_apo_configurator(config_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Safely launches Equalizer APO's DeviceSelector.exe or Configurator.exe with:
    1. Administrator elevation (via Windows ShellExecute 'runas').
    2. Sanitized environment (removes Python/PyQt6 Qt DLLs from PATH to prevent
       the 'no Qt platform plugin could be initialized' crash).
    3. Explicit QT_PLUGIN_PATH pointing to Equalizer APO's qt directory.
    4. Working directory set to the Equalizer APO install directory.

    Returns (success: bool, message: str).
    """
    import ctypes

    if not config_path:
        config_path = get_apo_configurator_path()

    if not config_path or not os.path.exists(config_path):
        return False, "Equalizer APO Device Selector / Configurator was not found."

    apo_dir = os.path.dirname(os.path.abspath(config_path))

    # Save original environment
    orig_path = os.environ.get("PATH", "")
    orig_qt_plugin = os.environ.get("QT_PLUGIN_PATH")
    orig_qt_qpa = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    orig_pythonpath = os.environ.get("PYTHONPATH")

    try:
        # Sanitize PATH: remove any python/venv/site-packages paths that shadow Qt6 DLLs
        raw_paths = orig_path.split(";")
        clean_paths = []
        for p in raw_paths:
            p_low = p.lower().strip()
            if not p_low:
                continue
            if "pyqt6" in p_low or "site-packages" in p_low or "clean_apo" in p_low:
                continue
            clean_paths.append(p)

        clean_paths.insert(0, apo_dir)
        os.environ["PATH"] = ";".join(clean_paths)

        # Set QT plugin paths explicitly to APO's qt folder
        qt_dir = os.path.join(apo_dir, "qt")
        if os.path.isdir(qt_dir):
            os.environ["QT_PLUGIN_PATH"] = qt_dir
            platforms_dir = os.path.join(qt_dir, "platforms")
            if os.path.isdir(platforms_dir):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms_dir
        else:
            os.environ.pop("QT_PLUGIN_PATH", None)
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        os.environ.pop("PYTHONPATH", None)
        os.environ.pop("PYTHONHOME", None)

        # Attempt ShellExecuteW with verb 'runas' for UAC elevation
        try:
            res = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", config_path, None, apo_dir, 1
            )
            # ShellExecute returns an HINSTANCE > 32 on success
            if res > 32:
                return True, "Launched successfully with elevation."
        except Exception:
            res = 0

        # Fallback to verb 'open'
        try:
            res2 = ctypes.windll.shell32.ShellExecuteW(
                None, "open", config_path, None, apo_dir, 1
            )
            if res2 > 32:
                return True, "Launched successfully."
        except Exception:
            res2 = 0

        # Final fallback: subprocess.Popen with sanitized env and explicit cwd
        import subprocess
        child_env = dict(os.environ)
        subprocess.Popen([config_path], cwd=apo_dir, env=child_env)
        return True, "Launched via subprocess."
    except Exception as e:
        return False, str(e)
    finally:
        # Restore calling process's environment
        os.environ["PATH"] = orig_path
        if orig_qt_plugin is not None:
            os.environ["QT_PLUGIN_PATH"] = orig_qt_plugin
        else:
            os.environ.pop("QT_PLUGIN_PATH", None)

        if orig_qt_qpa is not None:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = orig_qt_qpa
        else:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        if orig_pythonpath is not None:
            os.environ["PYTHONPATH"] = orig_pythonpath
        else:
            os.environ.pop("PYTHONPATH", None)


def create_windows_shortcuts() -> Tuple[bool, str]:
    """Creates or updates Desktop and Start Menu shortcuts for Clean APO."""
    try:
        import subprocess

        exe_path = sys.executable if getattr(sys, "frozen", False) else os.path.abspath(sys.argv[0])
        # If running in development/source mode, prioritize installed or dist CleanAPO.exe
        if not getattr(sys, "frozen", False):
            candidate_installed = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Clean APO\CleanAPO.exe")
            candidate_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "CleanAPO.exe")
            if os.path.isfile(candidate_installed):
                exe_path = candidate_installed
            elif os.path.isfile(candidate_dist):
                exe_path = candidate_dist

        work_dir = os.path.dirname(exe_path)
        icon_path = os.path.join(work_dir, "clean_apo.ico")
        if not os.path.isfile(icon_path):
            alt_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "clean_apo.ico")
            if os.path.isfile(alt_icon):
                icon_path = alt_icon
            else:
                icon_path = exe_path

        ps_script = f'''
        $WshShell = New-Object -ComObject WScript.Shell
        $desktopPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Clean APO.lnk"
        $s1 = $WshShell.CreateShortcut($desktopPath)
        $s1.TargetPath = "{exe_path}"
        $s1.WorkingDirectory = "{work_dir}"
        $s1.IconLocation = "{icon_path},0"
        $s1.Description = "Clean APO - High-Performance Audio Equalizer & AutoEQ Studio"
        $s1.Save()

        $startMenuPath = Join-Path ([Environment]::GetFolderPath("Programs")) "Clean APO.lnk"
        $s2 = $WshShell.CreateShortcut($startMenuPath)
        $s2.TargetPath = "{exe_path}"
        $s2.WorkingDirectory = "{work_dir}"
        $s2.IconLocation = "{icon_path},0"
        $s2.Description = "Clean APO - High-Performance Audio Equalizer & AutoEQ Studio"
        $s2.Save()
        '''
        creation_flags = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            check=True,
            creationflags=creation_flags,
        )
        return True, "Shortcuts created on Desktop and Start Menu!"
    except Exception as e:
        return False, str(e)


# Built-in factory presets
FACTORY_PRESETS: Dict[str, Dict] = {
    "Flat (Default)": {
        "preamp_db": 0.0,
        "bands": [
            {"band_id": 1, "enabled": True, "filter_type": "Low Shelf", "freq": 100.0, "gain": 0.0, "q": 0.707},
            {"band_id": 2, "enabled": True, "filter_type": "Peak", "freq": 250.0, "gain": 0.0, "q": 1.414},
            {"band_id": 3, "enabled": True, "filter_type": "Peak", "freq": 1000.0, "gain": 0.0, "q": 1.414},
            {"band_id": 4, "enabled": True, "filter_type": "Peak", "freq": 4000.0, "gain": 0.0, "q": 1.414},
            {"band_id": 5, "enabled": True, "filter_type": "High Shelf", "freq": 8000.0, "gain": 0.0, "q": 0.707},
        ],
    },
    "Harman Target (Over-Ear)": {
        "preamp_db": -5.5,
        "bands": [
            {"band_id": 1, "enabled": True, "filter_type": "Low Shelf", "freq": 105.0, "gain": 5.5, "q": 0.707},
            {"band_id": 2, "enabled": True, "filter_type": "Peak", "freq": 200.0, "gain": -2.0, "q": 1.40},
            {"band_id": 3, "enabled": True, "filter_type": "Peak", "freq": 1200.0, "gain": -1.5, "q": 2.00},
            {"band_id": 4, "enabled": True, "filter_type": "Peak", "freq": 3000.0, "gain": 3.0, "q": 1.414},
            {"band_id": 5, "enabled": True, "filter_type": "High Shelf", "freq": 8000.0, "gain": 2.0, "q": 0.707},
        ],
    },
    "Bass Boost & Warmth": {
        "preamp_db": -6.0,
        "bands": [
            {"band_id": 1, "enabled": True, "filter_type": "Low Shelf", "freq": 80.0, "gain": 6.0, "q": 0.707},
            {"band_id": 2, "enabled": True, "filter_type": "Peak", "freq": 250.0, "gain": 1.5, "q": 1.00},
            {"band_id": 3, "enabled": True, "filter_type": "Peak", "freq": 3500.0, "gain": 1.0, "q": 2.00},
            {"band_id": 4, "enabled": True, "filter_type": "High Shelf", "freq": 10000.0, "gain": -1.0, "q": 0.707},
        ],
    },
    "V-Shape Punch & Sparkle": {
        "preamp_db": -4.5,
        "bands": [
            {"band_id": 1, "enabled": True, "filter_type": "Low Shelf", "freq": 90.0, "gain": 4.5, "q": 0.707},
            {"band_id": 2, "enabled": True, "filter_type": "Peak", "freq": 800.0, "gain": -2.5, "q": 1.20},
            {"band_id": 3, "enabled": True, "filter_type": "Peak", "freq": 4500.0, "gain": 2.0, "q": 1.50},
            {"band_id": 4, "enabled": True, "filter_type": "High Shelf", "freq": 9000.0, "gain": 3.5, "q": 0.707},
        ],
    },
    "Vocal Presence & Clarity": {
        "preamp_db": -3.0,
        "bands": [
            {"band_id": 1, "enabled": True, "filter_type": "High Pass", "freq": 60.0, "gain": 0.0, "q": 0.707},
            {"band_id": 2, "enabled": True, "filter_type": "Peak", "freq": 300.0, "gain": -1.5, "q": 1.50},
            {"band_id": 3, "enabled": True, "filter_type": "Peak", "freq": 2200.0, "gain": 3.0, "q": 1.414},
            {"band_id": 4, "enabled": True, "filter_type": "Peak", "freq": 4500.0, "gain": 2.5, "q": 1.80},
            {"band_id": 5, "enabled": True, "filter_type": "High Shelf", "freq": 12000.0, "gain": 1.0, "q": 0.707},
        ],
    },
}


class PresetManager(QtCore.QObject):
    """
    Manages user presets, active preset selection, EQ importing,
    and debounced Equalizer APO file exports.
    """
    sigPresetChanged = QtCore.pyqtSignal(str)          # Emits preset name when loaded
    sigPresetsListChanged = QtCore.pyqtSignal()         # Emits when presets added/renamed/deleted
    sigApoSyncStatus = QtCore.pyqtSignal(bool, str)     # success, message

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.presets: Dict[str, Dict] = {}
        self.active_preset_name: str = "Flat (Default)"
        self.apo_target_path: str = DEFAULT_APO_PATH
        self.active_apo_path: str = DEFAULT_APO_PATH
        self.audio_device_index: Optional[int] = None
        self.active_device_name: Optional[str] = None
        self.favorite_databases: List[str] = ["peqdb", "superreview", "superreview_hp", "jaraba", "aden"]
        
        # Debounce timer for Equalizer APO file writes (~50 ms)
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(50)
        self._debounce_timer.timeout.connect(self._do_write_apo_file)

        # Pending data for debounced write
        self._pending_bands: List[EQBand] = []
        self._pending_preamp: float = 0.0
        self._pending_preset_name: str = ""
        self._pending_bypassed: bool = False

        self._load_settings()
        self._load_presets()

    # -------------------------------------------------------------------------
    # Settings & Persistence
    # -------------------------------------------------------------------------
    def is_apo_installed(self) -> bool:
        """Returns True if Equalizer APO is installed on this PC."""
        return is_equalizer_apo_installed()

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        self.settings[key] = value
        self.save_settings()

    def _load_settings(self) -> None:
        self.settings = {}
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    self.settings = json.load(f)
                    self.apo_target_path = self.settings.get("apo_path", DEFAULT_APO_PATH)
                    self.active_preset_name = self.settings.get("active_preset", "Flat (Default)")
                    self.audio_device_index = self.settings.get("audio_device_index", None)
                    self.active_device_name = self.settings.get("active_device_name", None)
                    self.favorite_databases = self.settings.get(
                        "favorite_databases",
                        ["peqdb", "superreview", "superreview_hp", "jaraba", "aden"],
                    )
            except Exception as e:
                print(f"[PresetManager] Error loading settings: {e}")
                self.settings = {}
        else:
            self.apo_target_path = DEFAULT_APO_PATH

        # If Equalizer APO is detected and target was set to fallback, restore default APO path
        if is_equalizer_apo_installed() and (not self.apo_target_path or self.apo_target_path == LOCAL_FALLBACK_PATH):
            self.apo_target_path = DEFAULT_APO_PATH

    def save_settings(self) -> None:
        try:
            self.settings.update({
                "apo_path": self.apo_target_path,
                "active_preset": self.active_preset_name,
                "audio_device_index": self.audio_device_index,
                "active_device_name": self.active_device_name,
                "favorite_databases": self.favorite_databases,
            })
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"[PresetManager] Error saving settings: {e}")

    def _load_presets(self) -> None:
        if not os.path.exists(PRESETS_FILE):
            candidate_sources = [
                os.path.join(getattr(sys, '_MEIPASS', ''), "presets.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.json"),
                os.path.abspath("presets.json"),
            ]
            for c in candidate_sources:
                if c and os.path.exists(c) and os.path.abspath(c) != os.path.abspath(PRESETS_FILE):
                    try:
                        shutil.copyfile(c, PRESETS_FILE)
                        break
                    except Exception:
                        pass

        if os.path.exists(PRESETS_FILE):
            try:
                with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and loaded:
                        self.presets = loaded
                    else:
                        self.presets = copy.deepcopy(FACTORY_PRESETS)
            except Exception as e:
                print(f"[PresetManager] Error reading presets JSON: {e}")
                self.presets = copy.deepcopy(FACTORY_PRESETS)
        else:
            self.presets = copy.deepcopy(FACTORY_PRESETS)
            self.save_presets()

        if self.active_preset_name not in self.presets:
            self.active_preset_name = next(iter(self.presets.keys()))

    def save_presets(self) -> None:
        try:
            with open(PRESETS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.presets, f, indent=2)
        except Exception as e:
            print(f"[PresetManager] Error writing presets JSON: {e}")

    # -------------------------------------------------------------------------
    # Preset Operations
    # -------------------------------------------------------------------------
    def get_preset_names(self) -> List[str]:
        return list(self.presets.keys())

    def get_active_preset_data(self) -> Tuple[float, List[EQBand]]:
        data = self.presets.get(self.active_preset_name, FACTORY_PRESETS["Flat (Default)"])
        preamp = float(data.get("preamp_db", 0.0))
        bands_data = data.get("bands", [])
        bands = [EQBand.from_dict(b) for b in bands_data]
        return preamp, bands

    def set_active_preset(self, name: str) -> bool:
        if name in self.presets:
            self.active_preset_name = name
            self.save_settings()
            self.sigPresetChanged.emit(name)
            return True
        return False

    def update_active_preset_content(self, preamp_db: float, bands: List[EQBand]) -> None:
        curr = dict(self.presets.get(self.active_preset_name, {}))
        curr["preamp_db"] = float(preamp_db)
        curr["bands"] = [b.to_dict() for b in bands]
        self.presets[self.active_preset_name] = curr
        self.save_presets()

    def add_preset(self, name: str, preamp_db: float, bands: List[EQBand]) -> bool:
        name = name.strip()
        if not name or name in self.presets:
            return False
        self.presets[name] = {
            "preamp_db": float(preamp_db),
            "bands": [b.to_dict() for b in bands],
        }
        self.active_preset_name = name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        self.sigPresetChanged.emit(name)
        return True

    def get_favorite_databases(self) -> List[str]:
        return list(self.favorite_databases)

    def is_favorite_database(self, db_id: str) -> bool:
        return db_id in self.favorite_databases

    def toggle_favorite_database(self, db_id: str) -> bool:
        if db_id in self.favorite_databases:
            self.favorite_databases.remove(db_id)
            is_fav = False
        else:
            self.favorite_databases.append(db_id)
            is_fav = True
        self.save_settings()
        return is_fav

    def set_active_device_name(self, name: Optional[str]) -> None:
        self.active_device_name = name
        self.save_settings()

    def get_preset_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self.presets.get(name, {}).get("autoeq")

    def has_preset_metadata(self, name: str) -> bool:
        meta = self.get_preset_metadata(name)
        return bool(meta and (meta.get("is_autoeq") or meta.get("model_file") or meta.get("target_file")))

    def set_preset_metadata(self, name: str, metadata: Optional[Dict[str, Any]]) -> bool:
        name = name.strip()
        if name not in self.presets:
            return False
        if metadata:
            self.presets[name]["autoeq"] = metadata
        else:
            self.presets[name].pop("autoeq", None)
        self.save_presets()
        self.sigPresetsListChanged.emit()
        return True

    def save_or_update_preset(
        self,
        name: str,
        preamp_db: float,
        bands: List[EQBand],
        autoeq_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        name = name.strip()
        if not name:
            name = "AutoEQ Profile"
        data: Dict[str, Any] = {
            "preamp_db": float(preamp_db),
            "bands": [b.to_dict() for b in bands],
        }
        if autoeq_metadata:
            data["autoeq"] = autoeq_metadata
        elif name in self.presets and "autoeq" in self.presets[name]:
            data["autoeq"] = self.presets[name]["autoeq"]
        self.presets[name] = data
        self.active_preset_name = name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        self.sigPresetChanged.emit(name)
        return name

    def duplicate_preset(self, source_name: str) -> Optional[str]:
        if source_name not in self.presets:
            return None
        base = f"{source_name} (Copy)"
        new_name = base
        count = 1
        while new_name in self.presets:
            count += 1
            new_name = f"{base} {count}"

        self.presets[new_name] = json.loads(json.dumps(self.presets[source_name]))
        self.active_preset_name = new_name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        self.sigPresetChanged.emit(new_name)
        return new_name

    def rename_preset(self, old_name: str, new_name: str) -> bool:
        new_name = new_name.strip()
        if not new_name or old_name not in self.presets or (new_name != old_name and new_name in self.presets):
            return False
        self.presets[new_name] = self.presets.pop(old_name)
        if self.active_preset_name == old_name:
            self.active_preset_name = new_name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        self.sigPresetChanged.emit(new_name)
        return True

    def delete_preset(self, name: str) -> bool:
        if name not in self.presets or len(self.presets) <= 1:
            return False
        del self.presets[name]
        if self.active_preset_name == name:
            self.active_preset_name = next(iter(self.presets.keys()))
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        self.sigPresetChanged.emit(self.active_preset_name)
        return True

    # -------------------------------------------------------------------------
    # Equalizer APO File Writer with Debounce
    # -------------------------------------------------------------------------
    def schedule_apo_export(
        self,
        preamp_db: float,
        bands: List[EQBand],
        preset_name: Optional[str] = None,
        bypassed: bool = False,
    ) -> None:
        """
        Schedules a debounced write to Equalizer APO file (~50 ms).
        """
        self._pending_bands = [EQBand.from_dict(b.to_dict()) for b in bands]
        self._pending_preamp = float(preamp_db)
        self._pending_preset_name = preset_name or self.active_preset_name
        self._pending_bypassed = bool(bypassed)
        self._debounce_timer.start(50)

    def _format_apo_filter_line(self, index: int, band: EQBand, force_off: bool = False) -> str:
        state = "OFF" if (force_off or not band.enabled) else "ON"
        ftype = FilterType.from_str(band.filter_type)
        freq_str = f"{band.freq:.1f}" if band.freq < 100 else f"{band.freq:.0f}"

        if ftype == FilterType.PEAK:
            return f"Filter {index}: {state} PK Fc {freq_str} Hz Gain {band.gain:.1f} dB Q {band.q:.3f}"
        elif ftype == FilterType.LOW_SHELF:
            return f"Filter {index}: {state} LSC Fc {freq_str} Hz Gain {band.gain:.1f} dB Q {band.q:.3f}"
        elif ftype == FilterType.HIGH_SHELF:
            return f"Filter {index}: {state} HSC Fc {freq_str} Hz Gain {band.gain:.1f} dB Q {band.q:.3f}"
        elif ftype == FilterType.HIGH_PASS:
            return f"Filter {index}: {state} HPQ Fc {freq_str} Hz Q {band.q:.3f}"
        elif ftype == FilterType.LOW_PASS:
            return f"Filter {index}: {state} LPQ Fc {freq_str} Hz Q {band.q:.3f}"
        return f"Filter {index}: {state} PK Fc {freq_str} Hz Gain {band.gain:.1f} dB Q {band.q:.3f}"

    def _do_write_apo_file(self) -> None:
        """Executes the file write to APO target path with fallback and auto-include in config.txt."""
        preamp_val = 0.0 if self._pending_bypassed else self._pending_preamp
        lines = [
            f"# Generated by Clean APO for Equalizer APO",
            f"# Preset: {self._pending_preset_name}",
            f"# Status: {'BYPASSED' if self._pending_bypassed else 'ACTIVE'}",
            f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        apo_dev = clean_apo_device_name(self.active_device_name)
        if apo_dev == "all":
            lines.append("Device: all")
        else:
            lines.append(f"Device: {apo_dev}")

        lines.append(f"Preamp: {preamp_val:.1f} dB")

        for i, band in enumerate(self._pending_bands, start=1):
            lines.append(self._format_apo_filter_line(i, band, force_off=self._pending_bypassed))

        content = "\n".join(lines) + "\n"

        target_path = self.apo_target_path
        appdata_path = os.path.join(get_app_data_dir(), "clean_apo_generated.txt")
        used_path = target_path
        success = False
        msg = ""

        # Check write permission for target_path
        if not can_write_to_path(target_path):
            # Target is not directly writable without elevation; redirect to AppData path
            used_path = appdata_path
            try:
                os.makedirs(os.path.dirname(appdata_path), exist_ok=True)
                with open(appdata_path, "w", encoding="utf-8") as f:
                    f.write(content)
                # Ensure Equalizer APO's main config.txt includes this absolute AppData path
                inc_ok = ensure_apo_root_config_include(appdata_path)
                success = True
                if inc_ok:
                    msg = f"Synced with Equalizer APO via AppData ({os.path.basename(appdata_path)})"
                else:
                    msg = f"Saved to AppData. Equalizer APO config.txt needs UAC elevation to include it."
            except Exception as e_appdata:
                used_path = LOCAL_FALLBACK_PATH
                try:
                    with open(LOCAL_FALLBACK_PATH, "w", encoding="utf-8") as f:
                        f.write(content)
                    success = False
                    msg = f"Permission denied for APO folder. Saved to local: {LOCAL_FALLBACK_PATH}"
                except Exception as e_local:
                    success = False
                    msg = f"Failed writing EQ file: {e_local}"
        else:
            try:
                target_dir = os.path.dirname(target_path)
                if target_dir and not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)

                # 1. Ensure Equalizer APO's main config.txt includes our file(s)
                ensure_apo_root_config_include(target_path)

                # 2. Write primary target file
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)

                # 3. Clean legacy peqdb_generated.txt so it does not contain duplicate filters
                if target_dir and os.path.exists(target_dir):
                    legacy_path = os.path.join(target_dir, "peqdb_generated.txt")
                    if legacy_path != target_path and os.path.exists(legacy_path):
                        try:
                            with open(legacy_path, "w", encoding="utf-8") as fm:
                                fm.write("# Clean APO now outputs to clean_apo_generated.txt\n")
                        except Exception:
                            pass

                success = True
                used_path = target_path
                msg = f"Synced with Equalizer APO ({os.path.basename(target_path)})"
            except (PermissionError, OSError) as e:
                # If unexpected permission error during primary write, fall back to AppData path
                used_path = appdata_path
                try:
                    os.makedirs(os.path.dirname(appdata_path), exist_ok=True)
                    with open(appdata_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    ensure_apo_root_config_include(appdata_path)
                    success = True
                    msg = f"Permission denied for Program Files. Synced via AppData ({os.path.basename(appdata_path)})"
                except Exception as e_fallback:
                    used_path = LOCAL_FALLBACK_PATH
                    try:
                        with open(LOCAL_FALLBACK_PATH, "w", encoding="utf-8") as f:
                            f.write(content)
                        success = False
                        msg = f"Saved to local fallback: {LOCAL_FALLBACK_PATH}"
                    except Exception as e_final:
                        success = False
                        msg = f"Failed writing EQ file: {e_final}"

        self.active_apo_path = used_path
        self.sigApoSyncStatus.emit(success, msg)

    # -------------------------------------------------------------------------
    # EQ Import Engine
    # -------------------------------------------------------------------------
    def import_eq_from_file(self, file_path: str) -> Optional[Tuple[str, float, List[EQBand]]]:
        """
        Imports an EQ file (.txt, .peace, .json) and returns (preset_name, preamp_db, bands).
        Supports:
        - Equalizer APO .txt configs
        - Room EQ Wizard (REW) exports
        - Peace GUI .peace configs
        - JSON presets
        """
        if not os.path.exists(file_path):
            return None

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1].lower()

        # 1. JSON Format
        if ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if "bands" in data:
                            preamp = float(data.get("preamp_db", 0.0))
                            bands = [EQBand.from_dict(b) for b in data["bands"]]
                            return base_name, preamp, bands
                        elif "presets" in data:
                            first_name = next(iter(data["presets"].keys()))
                            pdata = data["presets"][first_name]
                            preamp = float(pdata.get("preamp_db", 0.0))
                            bands = [EQBand.from_dict(b) for b in pdata.get("bands", [])]
                            return first_name, preamp, bands
            except Exception as e:
                print(f"[Import] JSON parse failed: {e}")

        # 2. Peace .peace Format
        if ext == ".peace":
            try:
                cp = configparser.ConfigParser(strict=False)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    cp.read_file(f)

                preamp = float(cp.get("General", "Preamp", fallback="0.0"))
                freqs = [float(v) for k, v in cp.items("Frequencies")] if cp.has_section("Frequencies") else []
                gains = [float(v) for k, v in cp.items("Gains")] if cp.has_section("Gains") else []
                qs = [float(v) for k, v in cp.items("Qualities")] if cp.has_section("Qualities") else []

                bands: List[EQBand] = []
                for i in range(len(freqs)):
                    f_val = freqs[i]
                    g_val = gains[i] if i < len(gains) else 0.0
                    q_val = qs[i] if i < len(qs) else 1.414
                    bands.append(EQBand(band_id=i+1, enabled=True, filter_type="Peak", freq=f_val, gain=g_val, q=q_val))

                if bands:
                    return base_name, preamp, bands
            except Exception as e:
                print(f"[Import] Peace parse failed: {e}")

        # 3. Standard Equalizer APO / REW Text Format
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            preamp = 0.0
            bands: List[EQBand] = []
            band_idx = 1

            for line in lines:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                # Preamp line
                if line_str.lower().startswith("preamp:"):
                    m = re.search(r"[-+]?\d*\.?\d+", line_str)
                    if m:
                        preamp = float(m.group())
                    continue

                # Filter line
                if "filter" in line_str.lower() and "fc" in line_str.lower():
                    enabled = "off" not in line_str.lower()
                    lower = line_str.lower()
                    
                    ftype = "Peak"
                    if "lsc" in lower or "low shelf" in lower or "ls " in lower:
                        ftype = "Low Shelf"
                    elif "hsc" in lower or "high shelf" in lower or "hs " in lower:
                        ftype = "High Shelf"
                    elif "hp" in lower or "high pass" in lower:
                        ftype = "High Pass"
                    elif "lp" in lower or "low pass" in lower:
                        ftype = "Low Pass"

                    fc_m = re.search(r"fc\s+([0-9.,]+)", line_str, re.IGNORECASE)
                    gain_m = re.search(r"gain\s+([-+0-9.,]+)", line_str, re.IGNORECASE)
                    q_m = re.search(r"q\s+([0-9.,]+)", line_str, re.IGNORECASE)

                    freq = float(fc_m.group(1).replace(",", ".")) if fc_m else 1000.0
                    gain = float(gain_m.group(1).replace(",", ".")) if gain_m else 0.0
                    q = float(q_m.group(1).replace(",", ".")) if q_m else 1.414

                    bands.append(EQBand(
                        band_id=band_idx,
                        enabled=enabled,
                        filter_type=ftype,
                        freq=freq,
                        gain=gain,
                        q=q,
                    ))
                    band_idx += 1

            if bands:
                return base_name, preamp, bands
        except Exception as e:
            print(f"[Import] Text parse failed: {e}")

        return None
