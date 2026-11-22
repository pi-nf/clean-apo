"""
Preset management, JSON persistence, Equalizer APO file writer, and universal EQ importer.
"""

from __future__ import annotations
import os
import sys
import re
import json
import math
import time
import copy
import shutil
import threading
import configparser
from typing import Dict, List, Optional, Tuple, Any, Sequence
from PyQt6 import QtCore
from biquad import EQBand, FilterType


def _atomic_write_text(
    path: str,
    content: str,
    retries: int = 4,
    retry_delay: float = 0.025,
    fsync: bool = False,
) -> None:
    """
    Atomically writes UTF-8 text to `path` via a sibling temporary file and `os.replace`,
    retrying briefly on Windows sharing violations (e.g. Equalizer APO reading the file).
    """
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp_path = f"{abs_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            if fsync:
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

        last_err: Optional[Exception] = None
        for attempt in range(max(1, retries)):
            try:
                os.replace(tmp_path, abs_path)
                return
            except PermissionError as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(retry_delay)
            except OSError as e:
                last_err = e
                if getattr(e, "winerror", None) in (5, 32) and attempt + 1 < retries:
                    time.sleep(retry_delay)
                else:
                    raise
        if last_err is not None:
            raise last_err
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


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


def get_default_apo_path() -> str:
    """Returns the preferred Equalizer APO generated config path (respecting custom install paths)."""
    cfg_dir = get_apo_config_dir()
    if cfg_dir:
        return os.path.join(cfg_dir, "clean_apo_generated.txt")
    return r"C:\Program Files\EqualizerAPO\config\clean_apo_generated.txt"


DEFAULT_APO_PATH = get_default_apo_path()
LEGACY_APO_PATH = r"C:\Program Files\EqualizerAPO\config\peqdb_generated.txt"
LOCAL_FALLBACK_PATH = get_fallback_output_path()
PRESETS_FILE = get_presets_file()
SETTINGS_FILE = get_settings_file()

_ALL_DEVICE_SENTINELS = {
    "",
    "all",
    "all endpoints",
    "all devices",
    "default",
    "default output",
    "default device",
    "default audio device",
    "system default",
    "auto-follow",
}


def clean_apo_device_name(dev_name: Optional[str]) -> str:
    """
    Cleans audio device name for Equalizer APO directive matching.
    Removes PyAudio '(Loopback)' artifact, quotes, control characters, and WASAPI prefixes.
    Returns 'all' for auto-follow/all-endpoints/default sentinels, while preserving
    real device names (including USB port prefixes like '(2- Topping DX3 Pro+)' or
    names that merely contain the word 'default').
    """
    if not dev_name:
        return "all"
    # Strip control characters / newlines to prevent APO directive injection
    name = re.sub(r"[\x00-\x1f\x7f]+", " ", str(dev_name)).strip()
    low = name.lower()
    if low in _ALL_DEVICE_SENTINELS or low.startswith("auto-follow") or low.startswith("system default"):
        return "all"

    # 1. Remove (Loopback), [Loopback], (WASAPI), and outer quotes
    name = re.sub(r"\s*[\(\[]\s*(?:Loopback|WASAPI)\s*[\)\]]", "", name, flags=re.IGNORECASE).strip()
    name = name.strip('"\'').strip()
    low = name.lower()
    if low in _ALL_DEVICE_SENTINELS or low.startswith("auto-follow") or low.startswith("system default"):
        return "all"

    # 2. Strip only a leading top-level numeric prefix (e.g. "2- Topping") which remains a valid substring;
    # NEVER strip "(2- ...)" inside parentheses, as doing so breaks Equalizer APO substring matching!
    name = re.sub(r"^[0-9]+-\s*", "", name)

    # 3. Preserve the full friendly device name without truncating inner parentheses
    return name.strip() or "all"


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
    Waits synchronously via ShellExecuteExW + WaitForSingleObject + GetExitCodeProcess.
    """
    if not cfg_dir:
        cfg_dir = get_apo_config_dir() or r"C:\Program Files\EqualizerAPO\config"
    cfg_dir = os.path.normpath(cfg_dir).rstrip("\\/")
    if not os.path.isdir(cfg_dir):
        return False, "Equalizer APO config directory does not exist."
    try:
        import ctypes
        from ctypes import wintypes

        SEE_MASK_NOCLOSEPROCESS = 0x00000040
        WAIT_OBJECT_0 = 0x00000000
        WAIT_TIMEOUT = 0x00000102
        ERROR_CANCELLED = 1223

        class SHELLEXECUTEINFOW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("fMask", wintypes.ULONG),
                ("hwnd", wintypes.HWND),
                ("lpVerb", wintypes.LPCWSTR),
                ("lpFile", wintypes.LPCWSTR),
                ("lpParameters", wintypes.LPCWSTR),
                ("lpDirectory", wintypes.LPCWSTR),
                ("nShow", ctypes.c_int),
                ("hInstApp", wintypes.HINSTANCE),
                ("lpIDList", ctypes.c_void_p),
                ("lpClass", wintypes.LPCWSTR),
                ("hkeyClass", wintypes.HKEY),
                ("dwHotKey", wintypes.DWORD),
                ("hIconOrMonitor", wintypes.HANDLE),
                ("hProcess", wintypes.HANDLE),
            ]

        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
        shell32.ShellExecuteExW.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        # *S-1-5-32-545 is the Builtin\Users SID (locale-independent across all Windows languages)
        params = f'"{cfg_dir}" /grant *S-1-5-32-545:(OI)(CI)M /T'
        sei = SHELLEXECUTEINFOW()
        sei.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
        sei.fMask = SEE_MASK_NOCLOSEPROCESS
        sei.hwnd = None
        sei.lpVerb = "runas"
        sei.lpFile = "icacls.exe"
        sei.lpParameters = params
        sei.lpDirectory = None
        sei.nShow = 0  # SW_HIDE

        ok = shell32.ShellExecuteExW(ctypes.byref(sei))
        if not ok:
            err = ctypes.get_last_error()
            if err == ERROR_CANCELLED:
                return False, "UAC elevation was cancelled by user."
            return False, f"Elevation failed (Error code {err})."

        if not sei.hProcess:
            return False, "Elevation launched without process handle."

        try:
            wait_res = kernel32.WaitForSingleObject(sei.hProcess, 30000)
            if wait_res == WAIT_TIMEOUT:
                return False, "Timed out waiting for permission update."
            if wait_res != WAIT_OBJECT_0:
                return False, f"Wait failed (status {wait_res})."
            exit_code = wintypes.DWORD(1)
            if not kernel32.GetExitCodeProcess(sei.hProcess, ctypes.byref(exit_code)):
                return False, "Could not verify icacls exit code."
            if exit_code.value == 0:
                return True, "Permissions granted successfully via UAC."
            return False, f"icacls failed (exit code {exit_code.value})."
        finally:
            kernel32.CloseHandle(sei.hProcess)
    except Exception as e:
        return False, f"Failed granting permissions: {e}"


def ensure_apo_root_config_include(apo_file_path: str, allow_elevation: bool = True) -> bool:
    """
    Ensures Equalizer APO's main config.txt includes ONLY the target Clean APO file.
    Creates config.txt if it does not exist yet.
    Deduplicates stale Clean APO / legacy peqdb Include directives so filters are never applied twice.
    Preserves user comments, formatting, and unrelated directives.
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

        def _write_config(text: str) -> bool:
            try:
                _atomic_write_text(config_txt, text, retries=3, retry_delay=0.03)
                return True
            except (PermissionError, OSError):
                if allow_elevation:
                    ok, _ = grant_apo_directory_permissions(os.path.dirname(config_txt))
                    if ok:
                        try:
                            _atomic_write_text(config_txt, text, retries=3, retry_delay=0.03)
                            return True
                        except Exception:
                            pass
                return False

        # If config.txt does not exist, create it with include directive
        if not os.path.exists(config_txt):
            return _write_config(f"# Equalizer APO Main Configuration\n{directive}\n")

        with open(config_txt, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.read().splitlines()

        changed = False
        has_exact_directive = False
        cleaned_lines: List[str] = []
        directive_lower = directive.lower()

        for raw_line in raw_lines:
            stripped = raw_line.strip()
            low = stripped.lower()
            # Deduplicate any prior Include: targeting clean_apo_generated.txt or legacy peqdb_generated.txt
            if low.startswith("include:"):
                inc_target = stripped.split(":", 1)[1].strip().strip('"\'').lower()
                inc_base = os.path.basename(inc_target.replace("\\", "/"))
                if inc_base == "peqdb_generated.txt":
                    changed = True
                    continue
                if inc_base == "clean_apo_generated.txt":
                    if low == directive_lower and not has_exact_directive:
                        has_exact_directive = True
                        cleaned_lines.append(raw_line)
                    else:
                        changed = True
                    continue
                if low == directive_lower:
                    if not has_exact_directive:
                        has_exact_directive = True
                        cleaned_lines.append(raw_line)
                    else:
                        changed = True
                    continue
            cleaned_lines.append(raw_line)

        if not has_exact_directive:
            cleaned_lines.append(directive)
            changed = True

        if changed:
            return _write_config("\n".join(cleaned_lines) + "\n")
        return True
    except Exception as e:
        print(f"[PresetManager] Notice: Could not update config.txt: {e}")
        return False


def is_equalizer_apo_installed() -> bool:
    """Returns True if Equalizer APO is installed on this machine."""
    return get_apo_install_dir() is not None


def _resolve_config_txt_path(apo_file_path: str) -> str:
    """Resolves the active Equalizer APO config.txt path for diagnostics and repair."""
    target_dir = os.path.dirname(os.path.abspath(apo_file_path)) if apo_file_path else ""
    if target_dir:
        sibling_cfg = os.path.join(target_dir, "config.txt")
        if os.path.isfile(sibling_cfg):
            return sibling_cfg
    cfg_dir = get_apo_config_dir()
    if cfg_dir and os.path.isdir(cfg_dir):
        return os.path.join(cfg_dir, "config.txt")
    if os.path.isdir(r"C:\Program Files\EqualizerAPO\config"):
        return r"C:\Program Files\EqualizerAPO\config\config.txt"
    if target_dir:
        return os.path.join(target_dir, "config.txt")
    return ""


def diagnose_apo_setup(apo_file_path: str, active_device_name: str = "all") -> Dict[str, Any]:
    """
    Inspects Equalizer APO installation, config.txt Include wiring, duplicates,
    and write permissions. Returns a structured diagnostics dictionary with plain
    English status text.
    """
    apo_installed = is_equalizer_apo_installed()
    apo_config_dir = get_apo_config_dir() or ""
    target_path = os.path.abspath(apo_file_path) if apo_file_path else ""
    config_txt_path = _resolve_config_txt_path(apo_file_path)
    config_txt_exists = bool(config_txt_path and os.path.isfile(config_txt_path))
    active_device = clean_apo_device_name(active_device_name)

    include_present = False
    duplicate_includes = 0

    if config_txt_exists and target_path:
        try:
            with open(config_txt_path, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.read().splitlines()
            cfg_dir_abs = os.path.dirname(os.path.abspath(config_txt_path))
            target_norm = os.path.normcase(target_path)
            matching_count = 0
            stale_count = 0

            for raw_line in raw_lines:
                code_part = raw_line.split("#", 1)[0].strip()
                if not code_part or not code_part.lower().startswith("include:"):
                    continue
                inc_raw = code_part.split(":", 1)[1].strip().strip('"\'')
                if not inc_raw:
                    continue
                if os.path.isabs(inc_raw):
                    inc_abs = os.path.normcase(os.path.abspath(inc_raw))
                else:
                    inc_abs = os.path.normcase(os.path.abspath(os.path.join(cfg_dir_abs, inc_raw)))
                inc_base = os.path.basename(inc_raw.replace("\\", "/")).lower()

                if inc_abs == target_norm:
                    matching_count += 1
                elif inc_base in ("clean_apo_generated.txt", "peqdb_generated.txt"):
                    stale_count += 1

            include_present = matching_count >= 1
            duplicate_includes = max(0, matching_count - 1) + stale_count
        except Exception:
            include_present = False
            duplicate_includes = 0

    can_write_apo = False
    if target_path:
        parent_dir = os.path.dirname(target_path)
        if os.path.exists(target_path) or (parent_dir and os.path.isdir(parent_dir)):
            can_write_apo = can_write_to_path(target_path)

    norm_target = os.path.normcase(target_path) if target_path else ""
    norm_appdata = os.path.normcase(os.path.abspath(get_app_data_dir()))
    norm_fallback = os.path.normcase(os.path.abspath(LOCAL_FALLBACK_PATH))
    using_fallback = bool(
        norm_target
        and (
            norm_target == norm_fallback
            or norm_target.startswith(norm_appdata + os.sep)
        )
    )

    issues: List[str] = []
    if not apo_installed:
        issues.append("Equalizer APO is not installed on this system.")
        status_level = "error"
        status_summary = "Equalizer APO is not installed"
    else:
        if not config_txt_exists:
            issues.append("Equalizer APO config.txt was not found.")
        if not include_present:
            issues.append("Equalizer APO config.txt does not include the Clean APO EQ file.")
        if duplicate_includes > 0:
            issues.append(
                f"Found {duplicate_includes} duplicate or legacy Include directive(s) in config.txt."
            )
        if not can_write_apo:
            issues.append("Write permission is missing for the Equalizer APO target path.")
        if using_fallback:
            issues.append("EQ output is using the AppData fallback path instead of the Equalizer APO config folder.")

        if not config_txt_exists or not include_present or (not can_write_apo and not using_fallback):
            status_level = "error"
            if not config_txt_exists:
                status_summary = "Equalizer APO config.txt is missing"
            elif not include_present:
                status_summary = "Clean APO is not linked in Equalizer APO config.txt"
            else:
                status_summary = "Equalizer APO config directory is not writable"
        elif duplicate_includes > 0 or using_fallback or not can_write_apo:
            status_level = "warning"
            if duplicate_includes > 0:
                status_summary = "Duplicate Include directives found in config.txt"
            elif using_fallback:
                status_summary = "Using AppData fallback output path"
            else:
                status_summary = "Equalizer APO config folder permissions need repair"
        else:
            status_level = "ok"
            if active_device == "all":
                status_summary = "Equalizer APO is configured and active"
            else:
                status_summary = f"Equalizer APO is active for {active_device}"

    return {
        "apo_installed": apo_installed,
        "apo_config_dir": apo_config_dir,
        "config_txt_path": config_txt_path,
        "config_txt_exists": config_txt_exists,
        "include_present": include_present,
        "duplicate_includes": duplicate_includes,
        "can_write_apo": can_write_apo,
        "using_fallback": using_fallback,
        "target_path": target_path,
        "active_device": active_device,
        "status_level": status_level,
        "status_summary": status_summary,
        "issues": issues,
    }


def repair_apo_setup(
    apo_file_path: str,
    active_device_name: str = "all",
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Repairs Equalizer APO folder permissions (if needed) and ensures config.txt
    contains a single deduplicated Include directive for `apo_file_path`.
    Returns `(success, message, diagnostics)`.
    """
    diag = diagnose_apo_setup(apo_file_path, active_device_name=active_device_name)
    if not diag["apo_installed"]:
        return False, "Equalizer APO is not installed on this system.", diag

    cfg_dir = diag["apo_config_dir"] or (os.path.dirname(os.path.abspath(apo_file_path)) if apo_file_path else "")
    cfg_txt = diag["config_txt_path"]

    needs_perm = not diag["can_write_apo"]
    if not needs_perm and cfg_txt:
        cfg_parent = os.path.dirname(cfg_txt)
        if os.path.exists(cfg_txt) or (cfg_parent and os.path.isdir(cfg_parent)):
            needs_perm = not can_write_to_path(cfg_txt)

    if needs_perm and cfg_dir:
        perm_ok, perm_msg = grant_apo_directory_permissions(cfg_dir)
        if not perm_ok:
            new_diag = diagnose_apo_setup(apo_file_path, active_device_name=active_device_name)
            return False, perm_msg, new_diag

    inc_ok = ensure_apo_root_config_include(apo_file_path, allow_elevation=True)
    new_diag = diagnose_apo_setup(apo_file_path, active_device_name=active_device_name)

    if inc_ok and new_diag["include_present"] and new_diag["duplicate_includes"] == 0:
        return True, "Equalizer APO setup verified and repaired.", new_diag
    if not inc_ok:
        return False, "Could not update Equalizer APO config.txt.", new_diag
    return False, new_diag["status_summary"], new_diag


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
    subdirs = ("assets", "installers")
    for b in base_dirs:
        if not b:
            continue
        for n in names:
            for sub in subdirs:
                p = os.path.join(b, sub, n)
                if os.path.isfile(p):
                    return p
            p_root = os.path.join(b, n)
            if os.path.isfile(p_root):
                return p_root
        for sub in subdirs:
            sub_dir = os.path.join(b, sub)
            if os.path.isdir(sub_dir):
                try:
                    for f in os.listdir(sub_dir):
                        if f.lower().startswith("equalizerapo") and f.lower().endswith(".exe"):
                            return os.path.join(sub_dir, f)
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
    1. Administrator elevation (via Windows ShellExecute 'runas'), respecting UAC cancellation.
    2. Thread-safe sanitized child environment (never mutates process-global os.environ).
    3. Working directory set to the Equalizer APO install directory.

    Returns (success: bool, message: str).
    """
    import ctypes

    if not config_path:
        config_path = get_apo_configurator_path()

    if not config_path or not os.path.exists(config_path):
        return False, "Equalizer APO Device Selector / Configurator was not found."

    apo_dir = os.path.dirname(os.path.abspath(config_path))

    try:
        # Attempt ShellExecuteW with verb 'runas' for UAC elevation
        try:
            res = ctypes.windll.shell32.ShellExecuteW(
                None, "runas", config_path, None, apo_dir, 1
            )
            if res > 32:
                return True, "Launched successfully with elevation."
            if res == 5:
                return False, "Administrator elevation was cancelled by user."
        except Exception:
            res = 0

        # Fallback to verb 'open'
        try:
            res2 = ctypes.windll.shell32.ShellExecuteW(
                None, "open", config_path, None, apo_dir, 1
            )
            if res2 > 32:
                return True, "Launched successfully."
            if res2 == 5:
                return False, "Administrator elevation was cancelled by user."
        except Exception:
            res2 = 0

        # Final fallback: subprocess.Popen with sanitized local env copy (never mutates os.environ)
        import subprocess
        child_env = dict(os.environ)
        raw_paths = child_env.get("PATH", "").split(";")
        clean_paths = [
            p for p in raw_paths
            if p.strip() and not any(tok in p.lower() for tok in ("pyqt6", "site-packages", "clean_apo"))
        ]
        clean_paths.insert(0, apo_dir)
        child_env["PATH"] = ";".join(clean_paths)

        qt_dir = os.path.join(apo_dir, "qt")
        if os.path.isdir(qt_dir):
            child_env["QT_PLUGIN_PATH"] = qt_dir
            platforms_dir = os.path.join(qt_dir, "platforms")
            if os.path.isdir(platforms_dir):
                child_env["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms_dir
        else:
            child_env.pop("QT_PLUGIN_PATH", None)
            child_env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

        child_env.pop("PYTHONPATH", None)
        child_env.pop("PYTHONHOME", None)

        subprocess.Popen([config_path], cwd=apo_dir, env=child_env)
        return True, "Launched via subprocess."
    except Exception as e:
        return False, str(e)


def _ps_single_quote(value: str) -> str:
    """Safely quotes a literal string for PowerShell single-quoted context (immune to $ / backtick expansion)."""
    return "'" + value.replace("'", "''") + "'"


def create_windows_shortcuts() -> Tuple[bool, str]:
    """Creates or updates Desktop and Start Menu shortcuts for Clean APO."""
    try:
        import subprocess

        target_path = sys.executable if getattr(sys, "frozen", False) else os.path.abspath(sys.argv[0])
        target_args = ""

        # If running in development/source mode, prioritize installed or dist CleanAPO.exe
        if not getattr(sys, "frozen", False):
            candidate_installed = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Clean APO\CleanAPO.exe")
            candidate_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "CleanAPO.exe")
            if os.path.isfile(candidate_installed):
                target_path = candidate_installed
            elif os.path.isfile(candidate_dist):
                target_path = candidate_dist
            elif target_path.lower().endswith(".py"):
                # Do not point shortcut TargetPath directly at .py (would open in editor); launch via pythonw.exe
                script_path = target_path
                py_dir = os.path.dirname(sys.executable)
                pythonw = os.path.join(py_dir, "pythonw.exe")
                target_path = pythonw if os.path.isfile(pythonw) else sys.executable
                target_args = f'"{script_path}"'

        work_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if target_args else os.path.dirname(target_path)
        icon_path = os.path.join(work_dir, "clean_apo.ico")
        if not os.path.isfile(icon_path):
            alt_icon = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "clean_apo.ico")
            if os.path.isfile(alt_icon):
                icon_path = alt_icon
            else:
                icon_path = target_path

        ps_target = _ps_single_quote(target_path)
        ps_args = _ps_single_quote(target_args)
        ps_workdir = _ps_single_quote(work_dir)
        ps_icon = _ps_single_quote(f"{icon_path},0")

        ps_script = f'''
        $ErrorActionPreference = 'Stop'
        $WshShell = New-Object -ComObject WScript.Shell
        $desktopPath = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Clean APO.lnk'
        $s1 = $WshShell.CreateShortcut($desktopPath)
        $s1.TargetPath = {ps_target}
        $s1.Arguments = {ps_args}
        $s1.WorkingDirectory = {ps_workdir}
        $s1.IconLocation = {ps_icon}
        $s1.Description = 'Clean APO - High-Performance Audio Equalizer & AutoEQ Studio'
        $s1.Save()

        $startMenuPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'Clean APO.lnk'
        $s2 = $WshShell.CreateShortcut($startMenuPath)
        $s2.TargetPath = {ps_target}
        $s2.Arguments = {ps_args}
        $s2.WorkingDirectory = {ps_workdir}
        $s2.IconLocation = {ps_icon}
        $s2.Description = 'Clean APO - High-Performance Audio Equalizer & AutoEQ Studio'
        $s2.Save()
        '''
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=creation_flags,
        )
        if res.returncode != 0:
            err_msg = (res.stderr or res.stdout or f"PowerShell exited with code {res.returncode}").strip()
            return False, err_msg
        return True, "Shortcuts created on Desktop and Start Menu!"
    except Exception as e:
        return False, str(e)


def _parse_locale_float(raw: str) -> float:
    """
    Parses float strings handling US thousands separators ('1,250', '10,500.5'),
    European decimal commas ('-4,5', '0,707'), and European thousands + comma ('1.250,5').
    """
    s = re.sub(r"\s+", "", (raw or "").strip())
    if not s:
        raise ValueError("Empty float string")
    if "," in s and "." in s:
        # Whichever separator appears last is the decimal separator; strip the other
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Exact thousands grouping like "1,250" or "12,500" -> strip commas; otherwise comma is decimal point
        if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    val = float(s)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite float: {raw!r}")
    return val


def _default_q_for_filter(filter_type: str) -> float:
    ftype = FilterType.from_str(filter_type)
    if ftype in (FilterType.LOW_SHELF, FilterType.HIGH_SHELF, FilterType.HIGH_PASS, FilterType.LOW_PASS):
        return 0.707
    return 1.414


def _normalize_channel(channel: Any) -> str:
    """Normalizes a channel specification to 'ALL', 'L', or 'R'."""
    val = str(channel or "ALL").strip().upper()
    if val in ("L", "LEFT"):
        return "L"
    if val in ("R", "RIGHT"):
        return "R"
    return "ALL"


def _band_channel(band: EQBand) -> str:
    """Safely reads normalized channel from an EQBand (compatible with older and newer EQBand)."""
    return _normalize_channel(getattr(band, "channel", "ALL"))


def _make_eq_band(
    band_id: int,
    enabled: bool,
    filter_type: str,
    freq: float,
    gain: float,
    q: float,
    channel: str = "ALL",
) -> EQBand:
    """Constructs an EQBand while remaining compatible whether EQBand defines `channel` or not."""
    norm_ch = _normalize_channel(channel)
    fields = getattr(EQBand, "__dataclass_fields__", {})
    if "channel" in fields:
        return EQBand(
            band_id=int(band_id),
            enabled=bool(enabled),
            filter_type=filter_type,
            freq=freq,
            gain=gain,
            q=q,
            channel=norm_ch,
        )
    band = EQBand(
        band_id=int(band_id),
        enabled=bool(enabled),
        filter_type=filter_type,
        freq=freq,
        gain=gain,
        q=q,
    )
    if norm_ch != "ALL":
        setattr(band, "channel", norm_ch)
    return band


def _band_to_dict(band: Any) -> Dict[str, Any]:
    """Serializes an EQBand or band dict cleanly, omitting 'channel' when 'ALL'."""
    if isinstance(band, EQBand):
        d = dict(band.to_dict())
        ch = _band_channel(band)
    elif isinstance(band, dict):
        d = dict(band)
        ch = _normalize_channel(d.get("channel", "ALL"))
    else:
        raise TypeError(f"Unsupported band object: {type(band)!r}")
    if ch in ("L", "R"):
        d["channel"] = ch
    else:
        d.pop("channel", None)
    return d


def _sanitize_band(
    band_id: int,
    enabled: bool,
    filter_type: str,
    freq: float,
    gain: float,
    q: float,
    channel: str = "ALL",
) -> Optional[EQBand]:
    """Validates and clamps an EQ band so malformed/NaN/Inf/out-of-range imports never corrupt DSP."""
    try:
        f_val = float(freq)
        g_val = float(gain)
        q_val = float(q)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(f_val) and math.isfinite(g_val) and math.isfinite(q_val)):
        return None
    if f_val <= 0.0 or q_val <= 0.0:
        return None
    f_val = max(10.0, min(24000.0, f_val))
    g_val = max(-40.0, min(40.0, g_val))
    q_val = max(0.05, min(50.0, q_val))
    canonical_type = FilterType.from_str(str(filter_type or "Peak")).value
    return _make_eq_band(
        band_id=int(band_id),
        enabled=bool(enabled),
        filter_type=canonical_type,
        freq=f_val,
        gain=g_val,
        q=q_val,
        channel=_normalize_channel(channel),
    )


def _coerce_band(raw: Any, default_band_id: int = 1, default_channel: str = "ALL") -> Optional[EQBand]:
    """Converts an EQBand or dictionary into a sanitized EQBand."""
    if isinstance(raw, EQBand):
        ch = _band_channel(raw)
        if ch == "ALL" and default_channel != "ALL":
            ch = _normalize_channel(default_channel)
        return _sanitize_band(
            raw.band_id or default_band_id,
            raw.enabled,
            raw.filter_type,
            raw.freq,
            raw.gain,
            raw.q,
            channel=ch,
        )
    if isinstance(raw, dict):
        try:
            eb = EQBand.from_dict(raw)
        except Exception:
            return None
        ch = _normalize_channel(raw.get("channel", getattr(eb, "channel", default_channel)))
        if ch == "ALL" and default_channel != "ALL":
            ch = _normalize_channel(default_channel)
        return _sanitize_band(
            eb.band_id or default_band_id,
            eb.enabled,
            eb.filter_type,
            eb.freq,
            eb.gain,
            eb.q,
            channel=ch,
        )
    return None


def _normalize_channel_bands_dict(
    channel_bands: Optional[Dict[str, Sequence[Any]]],
) -> Dict[str, List[EQBand]]:
    """Normalizes a channel_bands mapping into sanitized {'L': [...], 'R': [...]} EQBand lists."""
    out: Dict[str, List[EQBand]] = {}
    if not isinstance(channel_bands, dict):
        return out
    for raw_key, raw_list in channel_bands.items():
        ch = _normalize_channel(raw_key)
        if ch not in ("L", "R") or not isinstance(raw_list, (list, tuple)):
            continue
        parsed: List[EQBand] = []
        for idx, item in enumerate(raw_list, start=1):
            sb = _coerce_band(item, default_band_id=idx, default_channel=ch)
            if sb is not None:
                parsed.append(sb)
        if parsed:
            out.setdefault(ch, []).extend(parsed)
    return out


def _serialize_channel_bands(
    channel_bands: Optional[Dict[str, Sequence[Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Serializes channel_bands into JSON-ready {'L': [...], 'R': [...]} band dicts."""
    normalized = _normalize_channel_bands_dict(channel_bands)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for ch in ("L", "R"):
        if ch in normalized and normalized[ch]:
            result[ch] = [_band_to_dict(b) for b in normalized[ch]]
    return result


def _map_peace_filter_type(raw_val: str) -> str:
    """Maps Peace Equalizer [Filters] string or numeric code to Clean APO FilterType value."""
    s = (raw_val or "").strip()
    if not s:
        return "Peak"
    # Peace numeric filter type IDs (fallback if stored numerically)
    peace_num_map = {
        "0": "Peak",
        "1": "Low Shelf",
        "2": "High Shelf",
        "3": "Low Pass",
        "4": "High Pass",
        "5": "Peak",
        "6": "Low Shelf",
        "7": "High Shelf",
    }
    if s in peace_num_map:
        return peace_num_map[s]
    return FilterType.from_str(s).value


def _format_apo_filter_line(index: int, band: EQBand, force_off: bool = False) -> str:
    """Formats a single Equalizer APO Filter directive line."""
    state = "OFF" if (force_off or not band.enabled) else "ON"
    ftype = FilterType.from_str(band.filter_type)
    freq_str = f"{band.freq:.1f}"

    if ftype == FilterType.PEAK:
        return f"Filter {index}: {state} PK Fc {freq_str} Hz Gain {band.gain:.2f} dB Q {band.q:.3f}"
    if ftype == FilterType.LOW_SHELF:
        return f"Filter {index}: {state} LSC Fc {freq_str} Hz Gain {band.gain:.2f} dB Q {band.q:.3f}"
    if ftype == FilterType.HIGH_SHELF:
        return f"Filter {index}: {state} HSC Fc {freq_str} Hz Gain {band.gain:.2f} dB Q {band.q:.3f}"
    if ftype == FilterType.HIGH_PASS:
        return f"Filter {index}: {state} HPQ Fc {freq_str} Hz Q {band.q:.3f}"
    if ftype == FilterType.LOW_PASS:
        return f"Filter {index}: {state} LPQ Fc {freq_str} Hz Q {band.q:.3f}"
    return f"Filter {index}: {state} PK Fc {freq_str} Hz Gain {band.gain:.2f} dB Q {band.q:.3f}"


# -----------------------------------------------------------------------------
# Universal EQ Export, Clipboard, GraphicEQ Conversion & Import Preview Engine
# -----------------------------------------------------------------------------
def export_eq_to_apo_text(
    preamp_db: float,
    bands: Sequence[EQBand],
    device_name: str = "",
    channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    preset_name: str = "",
    bypassed: bool = False,
    include_timestamp: bool = False,
) -> str:
    """
    Formats clean Equalizer APO configuration text.
    Supports optional per-channel `Channel: L` / `Channel: R` blocks when `channel_bands`
    is provided or when individual bands carry `channel` in ('L', 'R').
    """
    try:
        preamp_val = 0.0 if bypassed else float(preamp_db)
        if not math.isfinite(preamp_val):
            preamp_val = 0.0
    except (TypeError, ValueError):
        preamp_val = 0.0
    if abs(preamp_val) < 1e-9:
        preamp_val = 0.0

    lines: List[str] = ["# Generated by Clean APO for Equalizer APO"]
    if preset_name:
        safe_label = re.sub(r"[\r\n]+", " ", str(preset_name)).strip()
        if safe_label:
            lines.append(f"# Preset: {safe_label}")
    if bypassed or include_timestamp:
        lines.append(f"# Status: {'BYPASSED' if bypassed else 'ACTIVE'}")
    if include_timestamp:
        lines.append(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if device_name is not None and str(device_name).strip():
        apo_dev = clean_apo_device_name(device_name)
        lines.append(f"Device: {apo_dev}")

    lines.append(f"Preamp: {preamp_val:.2f} dB")

    shared_bands: List[EQBand] = []
    left_bands: List[EQBand] = []
    right_bands: List[EQBand] = []

    for idx, raw_band in enumerate(bands or (), start=1):
        sb = _coerce_band(raw_band, default_band_id=idx)
        if sb is None:
            continue
        ch = _band_channel(sb)
        if ch == "L":
            left_bands.append(sb)
        elif ch == "R":
            right_bands.append(sb)
        else:
            shared_bands.append(sb)

    if isinstance(channel_bands, dict):
        for raw_key, raw_seq in channel_bands.items():
            if not isinstance(raw_seq, (list, tuple)):
                continue
            ch = _normalize_channel(raw_key)
            for idx, item in enumerate(raw_seq, start=1):
                sb = _coerce_band(item, default_band_id=idx, default_channel=ch)
                if sb is None:
                    continue
                if ch == "L":
                    left_bands.append(sb)
                elif ch == "R":
                    right_bands.append(sb)
                else:
                    shared_bands.append(sb)

    has_split_channels = bool(left_bands or right_bands)
    filter_idx = 1

    if shared_bands:
        if has_split_channels:
            lines.append("Channel: all")
        for band in shared_bands:
            lines.append(_format_apo_filter_line(filter_idx, band, force_off=bypassed))
            filter_idx += 1

    if left_bands:
        lines.append("Channel: L")
        for band in left_bands:
            lines.append(_format_apo_filter_line(filter_idx, band, force_off=bypassed))
            filter_idx += 1

    if right_bands:
        lines.append("Channel: R")
        for band in right_bands:
            lines.append(_format_apo_filter_line(filter_idx, band, force_off=bypassed))
            filter_idx += 1

    if has_split_channels:
        lines.append("Channel: all")

    return "\n".join(lines) + "\n"


def export_eq_to_peace_text(
    name: str,
    preamp_db: float,
    bands: Sequence[EQBand],
) -> str:
    """
    Formats standard Peace Equalizer `.peace` INI text with
    [General], [Frequencies], [Gains], [Qualities], [Filters], and [Switches] sections.
    """
    try:
        preamp_val = float(preamp_db)
        if not math.isfinite(preamp_val):
            preamp_val = 0.0
    except (TypeError, ValueError):
        preamp_val = 0.0
    if abs(preamp_val) < 1e-9:
        preamp_val = 0.0

    safe_name = re.sub(r"[\r\n]+", " ", str(name or "Clean APO Preset")).strip() or "Clean APO Preset"
    sanitized: List[EQBand] = []
    for idx, b in enumerate(bands or (), start=1):
        sb = _coerce_band(b, default_band_id=idx)
        if sb is not None:
            sanitized.append(sb)

    lines: List[str] = [
        "[General]",
        f"Preamp={preamp_val:.2f}",
        f"Name={safe_name}",
        "",
        "[Frequencies]",
    ]
    for i, b in enumerate(sanitized, start=1):
        lines.append(f"Frequency{i}={b.freq:.1f}")

    lines.extend(["", "[Gains]"])
    for i, b in enumerate(sanitized, start=1):
        ftype = FilterType.from_str(b.filter_type)
        g_val = 0.0 if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS) else b.gain
        lines.append(f"Gain{i}={g_val:.2f}")

    lines.extend(["", "[Qualities]"])
    for i, b in enumerate(sanitized, start=1):
        lines.append(f"Quality{i}={b.q:.3f}")

    lines.extend(["", "[Filters]"])
    for i, b in enumerate(sanitized, start=1):
        lines.append(f"Filter{i}={FilterType.from_str(b.filter_type).value}")

    lines.extend(["", "[Switches]"])
    for i, b in enumerate(sanitized, start=1):
        lines.append(f"Switch{i}={1 if b.enabled else 0}")

    return "\n".join(lines) + "\n"


def _normalize_preset_dict_for_json(preset_data: Dict[str, Any]) -> Dict[str, Any]:
    """Produces a clean JSON-serializable copy of a preset dict preserving autoeq & channel_bands."""
    out: Dict[str, Any] = {}
    try:
        pre = float(preset_data.get("preamp_db", 0.0))
        if not math.isfinite(pre):
            pre = 0.0
    except (TypeError, ValueError):
        pre = 0.0
    if abs(pre) < 1e-9:
        pre = 0.0
    out["preamp_db"] = pre

    raw_bands = preset_data.get("bands", [])
    bands_list: List[Dict[str, Any]] = []
    if isinstance(raw_bands, (list, tuple)):
        for idx, b in enumerate(raw_bands, start=1):
            sb = _coerce_band(b, default_band_id=idx)
            if sb is not None:
                bands_list.append(_band_to_dict(sb))
    out["bands"] = bands_list

    if isinstance(preset_data.get("autoeq"), dict) and preset_data["autoeq"]:
        out["autoeq"] = copy.deepcopy(preset_data["autoeq"])

    ch_bands = _serialize_channel_bands(preset_data.get("channel_bands"))
    if ch_bands:
        out["channel_bands"] = ch_bands

    for k, v in preset_data.items():
        if k not in ("preamp_db", "bands", "autoeq", "channel_bands", "name"):
            out[k] = copy.deepcopy(v)
    return out


def export_eq_to_json_text(name: str, preset_data: Dict[str, Any]) -> str:
    """Exports a single preset dictionary to formatted JSON text, preserving metadata & channel_bands."""
    safe_name = str(name or "Preset").strip() or "Preset"
    payload = {"name": safe_name}
    payload.update(_normalize_preset_dict_for_json(preset_data if isinstance(preset_data, dict) else {}))
    return json.dumps(payload, indent=2) + "\n"


def export_all_presets_to_json_text(presets_dict: Dict[str, Dict[str, Any]]) -> str:
    """Exports a full preset library dictionary to formatted JSON text, preserving metadata & channel_bands."""
    normalized: Dict[str, Dict[str, Any]] = {}
    if isinstance(presets_dict, dict):
        for pname, pdata in presets_dict.items():
            if isinstance(pdata, dict):
                normalized[str(pname)] = _normalize_preset_dict_for_json(pdata)
    return json.dumps(normalized, indent=2) + "\n"


def export_eq_to_file(
    filepath: str,
    name: str,
    preamp_db: float,
    bands: Sequence[EQBand],
    metadata: Optional[Dict[str, Any]] = None,
    channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    device_name: str = "",
) -> Tuple[bool, str]:
    """
    Exports an EQ preset to disk, auto-detecting `.txt` (Equalizer APO), `.peace` (Peace INI),
    or `.json` (Clean APO JSON) based on the file extension. Uses atomic writes.
    """
    if not filepath or not str(filepath).strip():
        return False, "Export file path is empty."
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".peace":
            content = export_eq_to_peace_text(name, preamp_db, bands)
        elif ext == ".json":
            preset_data: Dict[str, Any] = {
                "preamp_db": float(preamp_db),
                "bands": [_band_to_dict(b) for b in (bands or ())],
            }
            if metadata:
                preset_data["autoeq"] = copy.deepcopy(metadata)
            if channel_bands:
                preset_data["channel_bands"] = _serialize_channel_bands(channel_bands)
            content = export_eq_to_json_text(name, preset_data)
        else:
            content = export_eq_to_apo_text(
                preamp_db,
                bands,
                device_name=device_name,
                channel_bands=channel_bands,
                preset_name=name,
            )
        _atomic_write_text(filepath, content, fsync=True)
        return True, f"Exported '{name}' to {os.path.basename(filepath)}"
    except Exception as e:
        return False, f"Failed to export EQ file: {e}"


def _convert_graphiceq_points_to_bands(
    points: Sequence[Tuple[float, float]],
    target_bands: int = 12,
    channel: str = "ALL",
) -> Tuple[List[EQBand], float, List[str]]:
    """
    Converts GraphicEQ `(freq_hz, gain_db)` control points into 10-15 parametric EQ bands
    using `autoeq_optimizer.optimize_parametric_eq`, with a safe fallback to sampled
    peaking filters if SciPy/optimizer fails.
    """
    warnings: List[str] = []
    norm_ch = _normalize_channel(channel)

    # Deduplicate & sort points in ascending frequency
    cleaned: Dict[float, float] = {}
    for f_raw, g_raw in points:
        try:
            f_val = float(f_raw)
            g_val = float(g_raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f_val) and math.isfinite(g_val) and 10.0 <= f_val <= 24000.0:
            cleaned[f_val] = max(-40.0, min(40.0, g_val))

    if not cleaned:
        return [], 0.0, ["GraphicEQ directive contained no valid frequency/gain points."]

    sorted_pts = sorted(cleaned.items(), key=lambda kv: kv[0])
    n_pts = len(sorted_pts)
    n_bands = max(1, min(15, target_bands, n_pts))
    if n_pts >= 10:
        n_bands = max(10, min(15, target_bands))

    # Try high-precision parametric optimization when >= 5 control points are available
    if n_pts >= 5:
        try:
            import numpy as np
            from autoeq_optimizer import optimize_parametric_eq

            pts_f = np.array([p[0] for p in sorted_pts], dtype=np.float64)
            pts_g = np.array([p[1] for p in sorted_pts], dtype=np.float64)
            spread = float(np.max(np.abs(pts_g - float(np.median(pts_g)))))
            gain_limit = max(12.0, min(30.0, spread + 3.0))

            opt_res = optimize_parametric_eq(
                meas_freqs=pts_f,
                meas_spls=np.zeros_like(pts_f),
                target_freqs=pts_f,
                target_spls=pts_g,
                num_bands=n_bands,
                use_bass_shelf=True,
                max_boost_db=gain_limit,
                max_cut_db=gain_limit,
                min_freq=max(20.0, float(pts_f[0])),
                max_freq=min(16000.0, float(pts_f[-1])),
                min_q=0.4,
                max_q=8.0,
            )
            if opt_res.bands:
                out_bands: List[EQBand] = []
                for idx, b in enumerate(opt_res.bands, start=1):
                    sb = _sanitize_band(idx, b.enabled, b.filter_type, b.freq, b.gain, b.q, channel=norm_ch)
                    if sb is not None:
                        out_bands.append(sb)
                if out_bands:
                    warnings.append(
                        f"Converted {n_pts}-point GraphicEQ into {len(out_bands)} parametric bands "
                        f"(RMS fit {opt_res.final_rms_db:.2f} dB)."
                    )
                    return out_bands, float(opt_res.preamp_db), warnings
        except Exception as e:
            warnings.append(f"Parametric GraphicEQ fit unavailable ({e}); used sampled peaking bands.")

    # Safe fallback: sample up to 12 log-spaced peaking bands directly from the GraphicEQ points
    fallback_count = min(12, n_pts)
    if fallback_count == n_pts:
        sampled = sorted_pts
    else:
        log_min = math.log10(sorted_pts[0][0])
        log_max = math.log10(sorted_pts[-1][0])
        sampled = []
        used_indices: set[int] = set()
        for i in range(fallback_count):
            frac = i / max(1, fallback_count - 1)
            target_log_f = log_min + frac * (log_max - log_min)
            best_idx = min(
                range(n_pts),
                key=lambda j: abs(math.log10(sorted_pts[j][0]) - target_log_f),
            )
            if best_idx not in used_indices:
                used_indices.add(best_idx)
                sampled.append(sorted_pts[best_idx])

    gains_only = sorted(g for _, g in sampled)
    median_g = gains_only[len(gains_only) // 2] if gains_only else 0.0
    fallback_bands: List[EQBand] = []
    for idx, (f_val, g_val) in enumerate(sampled, start=1):
        rel_g = round(g_val - median_g, 2)
        if abs(rel_g) < 1e-9:
            rel_g = 0.0
        # Estimate Q from adjacent band octave spacing
        q_est = 1.414
        if len(sampled) > 1:
            if idx == 1:
                bw = math.log2(sampled[1][0] / max(10.0, f_val))
            elif idx == len(sampled):
                bw = math.log2(f_val / max(10.0, sampled[-2][0]))
            else:
                bw = 0.5 * math.log2(sampled[idx][0] / max(10.0, sampled[idx - 2][0]))
            if bw > 0.05:
                two_bw = 2.0 ** bw
                q_est = max(0.7, min(4.0, math.sqrt(two_bw) / max(1e-6, two_bw - 1.0)))
        sb = _sanitize_band(
            idx,
            True,
            "Peak",
            round(f_val, 1) if f_val < 100.0 else round(f_val),
            rel_g,
            round(q_est, 2),
            channel=norm_ch,
        )
        if sb is not None:
            fallback_bands.append(sb)

    max_boost = max((b.gain for b in fallback_bands), default=0.0)
    fallback_preamp = -round(max_boost + 0.1, 1) if max_boost > 0.0 else 0.0
    if abs(fallback_preamp) < 1e-9:
        fallback_preamp = 0.0
    if not warnings:
        warnings.append(f"Converted {n_pts}-point GraphicEQ into {len(fallback_bands)} sampled peaking bands.")
    return fallback_bands, fallback_preamp, warnings


def parse_eq_from_text(
    text: str,
    source_name: str = "Clipboard",
) -> Tuple[str, float, List[EQBand], Dict[str, Any]]:
    """
    Universal EQ text parser supporting:
    - Equalizer APO `Filter:` configs (including REW exports and `Channel: L` / `Channel: R` blocks)
    - Equalizer APO `GraphicEQ:` lines (converted into 10-15 parametric bands)
    - Peace Equalizer `.peace` INI text
    - Clean APO single-preset and multi-preset JSON text

    Returns `(preset_name, preamp_db, bands, info_dict)` where `info_dict` contains
    `format`, `warnings`, `multi_presets`, `metadata`, and `channel_bands`.
    """
    clean_text = (text or "").lstrip("\ufeff").strip()
    default_name = (source_name or "Clipboard").strip() or "Clipboard"
    warnings: List[str] = []

    if not clean_text:
        return (
            default_name,
            0.0,
            [],
            {
                "format": "unknown",
                "warnings": ["Input text is empty."],
                "multi_presets": None,
                "metadata": None,
                "channel_bands": None,
            },
        )

    # 1. JSON Format
    if clean_text.startswith("{"):
        try:
            data = json.loads(clean_text)
        except Exception as e:
            data = None
            warnings.append(f"JSON parse failed: {e}")

        if isinstance(data, dict):
            def _extract_json_entry(
                p_name: str,
                pdata: Dict[str, Any],
            ) -> Optional[Tuple[str, float, List[EQBand], Optional[Dict[str, Any]], Dict[str, List[EQBand]]]]:
                raw_bands = pdata.get("bands")
                if not isinstance(raw_bands, list):
                    return None
                try:
                    preamp_val = float(pdata.get("preamp_db", 0.0))
                    if not math.isfinite(preamp_val):
                        preamp_val = 0.0
                except (TypeError, ValueError):
                    preamp_val = 0.0
                preamp_val = max(-40.0, min(40.0, preamp_val))

                parsed_bands: List[EQBand] = []
                for idx, b in enumerate(raw_bands, start=1):
                    sb = _coerce_band(b, default_band_id=idx)
                    if sb is not None:
                        parsed_bands.append(sb)

                ch_bands = _normalize_channel_bands_dict(pdata.get("channel_bands"))
                meta = copy.deepcopy(pdata["autoeq"]) if isinstance(pdata.get("autoeq"), dict) else None
                if parsed_bands or ch_bands:
                    return p_name, preamp_val, parsed_bands, meta, ch_bands
                return None

            # 1a. Single preset object: {"name"?: ..., "preamp_db": ..., "bands": [...]}
            if "bands" in data and isinstance(data["bands"], list):
                resolved_name = str(data.get("name") or "").strip() or default_name
                extracted = _extract_json_entry(resolved_name, data)
                if extracted is not None:
                    p_name, preamp_val, parsed_bands, meta, ch_bands = extracted
                    return (
                        p_name,
                        preamp_val,
                        parsed_bands,
                        {
                            "format": "json",
                            "warnings": warnings,
                            "multi_presets": None,
                            "metadata": meta,
                            "channel_bands": ch_bands or None,
                        },
                    )

            # 1b. Wrapped multi-preset: {"presets": {"Name": {"bands": [...]}}}
            # 1c. Clean APO flat library format: {"Preset Name": {"preamp_db": ..., "bands": [...]}}
            candidates_map: Optional[Dict[str, Any]] = None
            if "presets" in data and isinstance(data["presets"], dict) and data["presets"]:
                candidates_map = data["presets"]
            else:
                candidates_map = data

            multi_presets: Dict[str, Dict[str, Any]] = {}
            first_entry: Optional[Tuple[str, float, List[EQBand], Optional[Dict[str, Any]], Dict[str, List[EQBand]]]] = None
            for pname, pdata in candidates_map.items():
                if not isinstance(pdata, dict) or not isinstance(pdata.get("bands"), list):
                    continue
                extracted = _extract_json_entry(str(pname), pdata)
                if extracted is not None:
                    ep_name, e_pre, e_bands, e_meta, e_ch = extracted
                    entry_dict: Dict[str, Any] = {
                        "preamp_db": e_pre,
                        "bands": [_band_to_dict(b) for b in e_bands],
                    }
                    if e_meta:
                        entry_dict["autoeq"] = e_meta
                    if e_ch:
                        entry_dict["channel_bands"] = _serialize_channel_bands(e_ch)
                    multi_presets[ep_name] = entry_dict
                    if first_entry is None:
                        first_entry = extracted

            if first_entry is not None:
                p_name, preamp_val, parsed_bands, meta, ch_bands = first_entry
                return (
                    p_name,
                    preamp_val,
                    parsed_bands,
                    {
                        "format": "json",
                        "warnings": warnings,
                        "multi_presets": multi_presets,
                        "metadata": meta,
                        "channel_bands": ch_bands or None,
                    },
                )

    # 2. Peace .peace INI Format
    looks_like_peace = bool(
        re.search(r"^\s*\[Frequencies\]\s*$", clean_text, re.IGNORECASE | re.MULTILINE)
        or (
            re.search(r"^\s*\[General\]\s*$", clean_text, re.IGNORECASE | re.MULTILINE)
            and re.search(r"^\s*\[(Gains|Filters|Qualities)\]\s*$", clean_text, re.IGNORECASE | re.MULTILINE)
        )
    )
    if looks_like_peace:
        try:
            cp = configparser.ConfigParser(strict=False, interpolation=None)
            cp.read_string(clean_text)

            preamp = 0.0
            peace_name = default_name
            if cp.has_section("General"):
                if cp.has_option("General", "Preamp"):
                    raw_pre = cp.get("General", "Preamp", fallback="").strip()
                    if raw_pre:
                        try:
                            preamp = max(-40.0, min(40.0, _parse_locale_float(raw_pre)))
                        except ValueError:
                            warnings.append(f"Ignored invalid Peace Preamp value: {raw_pre!r}")
                if default_name == "Clipboard" and cp.has_option("General", "Name"):
                    cand_name = cp.get("General", "Name", fallback="").strip()
                    if cand_name:
                        peace_name = cand_name

            def _indexed_section(section_names: Tuple[str, ...]) -> Dict[int, str]:
                out: Dict[int, str] = {}
                for sec in section_names:
                    if cp.has_section(sec):
                        for pos, (k, v) in enumerate(cp.items(sec), start=1):
                            m_idx = re.search(r"(\d+)$", k.strip())
                            idx_num = int(m_idx.group(1)) if m_idx else pos
                            out[idx_num] = (v or "").strip()
                        break
                return out

            freq_map = _indexed_section(("Frequencies",))
            gain_map = _indexed_section(("Gains",))
            q_map = _indexed_section(("Qualities",))
            filter_map = _indexed_section(("Filters", "Filter Types", "FilterTypes"))
            switch_map = _indexed_section(("Switches", "Enabled", "States"))

            peace_bands: List[EQBand] = []
            band_idx = 1
            for key_num in sorted(freq_map.keys()):
                raw_f = freq_map.get(key_num, "")
                if not raw_f:
                    continue
                try:
                    f_val = _parse_locale_float(raw_f)
                except ValueError:
                    continue

                ftype = _map_peace_filter_type(filter_map.get(key_num, ""))
                raw_g = gain_map.get(key_num, "")
                try:
                    g_val = _parse_locale_float(raw_g) if raw_g else 0.0
                except ValueError:
                    g_val = 0.0

                raw_q = q_map.get(key_num, "")
                try:
                    q_val = _parse_locale_float(raw_q) if raw_q else _default_q_for_filter(ftype)
                except ValueError:
                    q_val = _default_q_for_filter(ftype)

                raw_sw = switch_map.get(key_num, "").lower()
                enabled = raw_sw not in ("0", "false", "off", "no")

                sb = _sanitize_band(band_idx, enabled, ftype, f_val, g_val, q_val)
                if sb is not None:
                    peace_bands.append(sb)
                    band_idx += 1

            if peace_bands:
                return (
                    peace_name,
                    preamp,
                    peace_bands,
                    {
                        "format": "peace",
                        "warnings": warnings,
                        "multi_presets": None,
                        "metadata": None,
                        "channel_bands": None,
                    },
                )
        except Exception as e:
            warnings.append(f"Peace INI parse failed: {e}")

    # 3. Equalizer APO / REW / GraphicEQ Text Format
    preamp = 0.0
    has_explicit_preamp = False
    parsed_name = default_name
    current_channel = "ALL"
    apo_bands: List[EQBand] = []
    channel_bands_map: Dict[str, List[EQBand]] = {}
    graphiceq_blocks: List[Tuple[str, List[Tuple[float, float]]]] = []
    band_idx = 1

    for raw_line in clean_text.splitlines():
        stripped_raw = raw_line.strip()
        # Extract preset name from Clean APO header comment when parsing from clipboard
        if default_name == "Clipboard" and stripped_raw.startswith("#"):
            preset_m = re.match(r"^#\s*preset\s*:\s*(.+)$", stripped_raw, re.IGNORECASE)
            if preset_m and preset_m.group(1).strip():
                parsed_name = preset_m.group(1).strip()

        # Strip inline comments BEFORE matching ON/OFF, filter types, or Q tokens
        code_part = raw_line.split("#", 1)[0].strip()
        if not code_part:
            continue

        # Channel directive: Channel: L | R | all | L R
        ch_m = re.match(r"^channel\s*:\s*(.+)$", code_part, re.IGNORECASE)
        if ch_m:
            ch_tokens = [tok.upper() for tok in ch_m.group(1).strip().split()]
            if ch_tokens == ["L"] or ch_tokens == ["LEFT"]:
                current_channel = "L"
            elif ch_tokens == ["R"] or ch_tokens == ["RIGHT"]:
                current_channel = "R"
            else:
                current_channel = "ALL"
            continue

        # Preamp directive
        pre_m = re.match(r"^preamp\s*:\s*([-+]?\s*\d+(?:[.,]\d+)?)", code_part, re.IGNORECASE)
        if pre_m:
            try:
                preamp = max(-40.0, min(40.0, _parse_locale_float(pre_m.group(1))))
                has_explicit_preamp = True
            except ValueError:
                pass
            continue

        # GraphicEQ directive: GraphicEQ: freq1 gain1; freq2 gain2; ...
        geq_m = re.match(r"^graphiceq\s*:\s*(.+)$", code_part, re.IGNORECASE)
        if geq_m:
            pts: List[Tuple[float, float]] = []
            for segment in geq_m.group(1).split(";"):
                seg = segment.strip()
                if not seg:
                    continue
                parts = seg.split()
                if len(parts) >= 2:
                    try:
                        f_v = _parse_locale_float(parts[0])
                        g_v = _parse_locale_float(parts[1])
                        pts.append((f_v, g_v))
                    except ValueError:
                        continue
            if pts:
                graphiceq_blocks.append((current_channel, pts))
            continue

        # Filter directive: anchored 'Filter [N]: ON|OFF ...'
        flt_m = re.match(r"^filter\s*\d*\s*:\s*(on|off)\b(.*)$", code_part, re.IGNORECASE)
        if not flt_m:
            continue

        enabled = flt_m.group(1).lower() == "on"
        rest = flt_m.group(2).strip()
        if not re.search(r"\bfc\b", rest, re.IGNORECASE):
            continue

        # Determine filter type strictly from tokens preceding 'Fc'
        type_part = re.split(r"\bfc\b", rest, maxsplit=1, flags=re.IGNORECASE)[0].strip().lower()
        ftype = "Peak"
        if re.search(r"\b(?:lsc|lsq|ls)\b|low[\s_-]*shelf", type_part):
            ftype = "Low Shelf"
        elif re.search(r"\b(?:hsc|hsq|hs)\b|high[\s_-]*shelf", type_part):
            ftype = "High Shelf"
        elif re.search(r"\bhpq?\b|high[\s_-]*pass", type_part):
            ftype = "High Pass"
        elif re.search(r"\blpq?\b|low[\s_-]*pass", type_part):
            ftype = "Low Pass"

        fc_m = re.search(r"\bfc\s+([0-9][0-9.,]*)\s*(khz|hz)?\b", rest, re.IGNORECASE)
        if not fc_m:
            continue
        try:
            freq = _parse_locale_float(fc_m.group(1))
            if fc_m.group(2) and fc_m.group(2).lower() == "khz":
                freq *= 1000.0
        except ValueError:
            continue

        gain = 0.0
        gain_m = re.search(r"\bgain\s+([-+]?\s*[0-9]+(?:[.,][0-9]+)?)", rest, re.IGNORECASE)
        if gain_m:
            try:
                gain = _parse_locale_float(gain_m.group(1))
            except ValueError:
                gain = 0.0

        q = _default_q_for_filter(ftype)
        q_m = re.search(r"\bq\s+([0-9]+(?:[.,][0-9]+)?)", rest, re.IGNORECASE)
        if q_m:
            try:
                q = _parse_locale_float(q_m.group(1))
            except ValueError:
                q = _default_q_for_filter(ftype)

        sb = _sanitize_band(band_idx, enabled, ftype, freq, gain, q, channel=current_channel)
        if sb is not None:
            apo_bands.append(sb)
            if current_channel in ("L", "R"):
                channel_bands_map.setdefault(current_channel, []).append(sb)
            band_idx += 1

    had_parametric = bool(apo_bands)
    had_graphiceq = False

    for geq_channel, pts in graphiceq_blocks:
        conv_bands, conv_preamp, conv_warnings = _convert_graphiceq_points_to_bands(
            pts,
            target_bands=12,
            channel=geq_channel,
        )
        warnings.extend(conv_warnings)
        if conv_bands:
            had_graphiceq = True
            for cb in conv_bands:
                re_band = _sanitize_band(
                    band_idx,
                    cb.enabled,
                    cb.filter_type,
                    cb.freq,
                    cb.gain,
                    cb.q,
                    channel=geq_channel,
                )
                if re_band is not None:
                    apo_bands.append(re_band)
                    if geq_channel in ("L", "R"):
                        channel_bands_map.setdefault(geq_channel, []).append(re_band)
                    band_idx += 1
            if not has_explicit_preamp:
                preamp = min(preamp, conv_preamp) if preamp < 0.0 else conv_preamp
            else:
                preamp = min(preamp, conv_preamp)

    if apo_bands:
        fmt = "graphiceq" if (had_graphiceq and not had_parametric) else "apo"
        return (
            parsed_name,
            preamp,
            apo_bands,
            {
                "format": fmt,
                "warnings": warnings,
                "multi_presets": None,
                "metadata": None,
                "channel_bands": channel_bands_map or None,
            },
        )

    if not warnings:
        warnings.append("No valid EQ bands were found in the input.")
    return (
        default_name,
        preamp,
        [],
        {
            "format": "unknown",
            "warnings": warnings,
            "multi_presets": None,
            "metadata": None,
            "channel_bands": None,
        },
    )


def preview_eq_import(filepath: str) -> Tuple[str, float, List[EQBand], Dict[str, Any]]:
    """
    Reads an EQ file (`.txt`, `.peace`, `.json`) and returns a parsed preview
    `(preset_name, preamp_db, bands, info_dict)` without mutating `PresetManager`.
    """
    base_name = os.path.splitext(os.path.basename(filepath))[0] if filepath else "Import"
    if not filepath or not os.path.isfile(filepath):
        return (
            base_name,
            0.0,
            [],
            {
                "format": "unknown",
                "warnings": [f"File not found: {filepath}"],
                "multi_presets": None,
                "metadata": None,
                "channel_bands": None,
            },
        )
    try:
        with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
            content = f.read()
        return parse_eq_from_text(content, source_name=base_name)
    except Exception as e:
        return (
            base_name,
            0.0,
            [],
            {
                "format": "unknown",
                "warnings": [f"Could not read file: {e}"],
                "multi_presets": None,
                "metadata": None,
                "channel_bands": None,
            },
        )


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
        self.apo_target_path: str = get_default_apo_path()
        self.active_apo_path: str = self.apo_target_path
        self.audio_device_index: Optional[int] = None
        self.audio_device_name: Optional[str] = None
        self.active_device_name: Optional[str] = None
        self.favorite_databases: List[str] = ["peqdb", "superreview", "superreview_hp", "jaraba", "aden"]

        # Debounce timer for Equalizer APO file writes (~50 ms)
        self._debounce_timer = QtCore.QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(50)
        self._debounce_timer.timeout.connect(self._do_write_apo_file)

        # Pending data for debounced write
        self._pending_bands: List[EQBand] = []
        self._pending_channel_bands: Optional[Dict[str, List[EQBand]]] = None
        self._pending_preamp: float = 0.0
        self._pending_preset_name: str = ""
        self._pending_bypassed: bool = False
        self._has_pending_export: bool = False
        self._verified_include_path: Optional[str] = None
        self._legacy_cleaned_dirs: set[str] = set()

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
        default_apo = get_default_apo_path()
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        self.settings = loaded
                    self.apo_target_path = self.settings.get("apo_path") or default_apo
                    self.active_preset_name = self.settings.get("active_preset", "Flat (Default)")
                    self.audio_device_index = self.settings.get("audio_device_index", None)
                    self.audio_device_name = self.settings.get("audio_device_name", None)
                    self.active_device_name = self.settings.get("active_device_name", None)
                    favs = self.settings.get(
                        "favorite_databases",
                        ["peqdb", "superreview", "superreview_hp", "jaraba", "aden"],
                    )
                    self.favorite_databases = list(favs) if isinstance(favs, list) else ["peqdb", "superreview", "superreview_hp", "jaraba", "aden"]
            except Exception as e:
                print(f"[PresetManager] Error loading settings: {e}")
                self.settings = {}
                self.apo_target_path = default_apo
        else:
            self.apo_target_path = default_apo

        # If Equalizer APO is detected and target was set to fallback, restore detected APO path
        if is_equalizer_apo_installed() and (not self.apo_target_path or self.apo_target_path == LOCAL_FALLBACK_PATH):
            self.apo_target_path = default_apo

    def save_settings(self) -> None:
        try:
            self.settings.update({
                "apo_path": self.apo_target_path,
                "active_preset": self.active_preset_name,
                "audio_device_index": self.audio_device_index,
                "audio_device_name": self.audio_device_name,
                "active_device_name": self.active_device_name,
                "favorite_databases": self.favorite_databases,
            })
            _atomic_write_text(SETTINGS_FILE, json.dumps(self.settings, indent=2), fsync=True)
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
                # Preserve corrupt presets.json before falling back so user data isn't silently overwritten
                try:
                    bak_path = PRESETS_FILE + ".corrupt.bak"
                    if not os.path.exists(bak_path):
                        shutil.copy2(PRESETS_FILE, bak_path)
                except Exception:
                    pass
                self.presets = copy.deepcopy(FACTORY_PRESETS)
        else:
            self.presets = copy.deepcopy(FACTORY_PRESETS)
            self.save_presets()

        if self.active_preset_name not in self.presets:
            self.active_preset_name = next(iter(self.presets.keys()))

    def save_presets(self) -> None:
        try:
            _atomic_write_text(PRESETS_FILE, json.dumps(self.presets, indent=2), fsync=True)
        except Exception as e:
            print(f"[PresetManager] Error writing presets JSON: {e}")

    # -------------------------------------------------------------------------
    # Preset Operations
    # -------------------------------------------------------------------------
    def get_preset_names(self) -> List[str]:
        return list(self.presets.keys())

    def get_active_preset_data(self) -> Tuple[float, List[EQBand]]:
        raw = self.presets.get(self.active_preset_name)
        if not isinstance(raw, dict):
            raw = FACTORY_PRESETS["Flat (Default)"]
        try:
            preamp = float(raw.get("preamp_db", 0.0))
            if not math.isfinite(preamp):
                preamp = 0.0
        except (TypeError, ValueError):
            preamp = 0.0

        bands_data = raw.get("bands", [])
        if not isinstance(bands_data, list):
            bands_data = []

        bands: List[EQBand] = []
        for idx, b in enumerate(bands_data, start=1):
            sanitized = _coerce_band(b, default_band_id=idx)
            if sanitized is not None:
                bands.append(sanitized)
        return preamp, bands

    def get_active_channel_bands(self) -> Dict[str, List[EQBand]]:
        """Returns optional per-channel EQ bands ({'L': [...], 'R': [...]}) for the active preset."""
        raw = self.presets.get(self.active_preset_name)
        if not isinstance(raw, dict):
            return {}
        return _normalize_channel_bands_dict(raw.get("channel_bands"))

    def set_active_channel_bands(self, channel_bands: Optional[Dict[str, Sequence[Any]]]) -> None:
        """Sets or clears optional per-channel bands on the active preset and persists to disk."""
        curr = dict(self.presets.get(self.active_preset_name, {}))
        serialized = _serialize_channel_bands(channel_bands)
        if serialized:
            curr["channel_bands"] = serialized
        else:
            curr.pop("channel_bands", None)
        self.presets[self.active_preset_name] = curr
        self.save_presets()

    def set_active_preset(self, name: str) -> bool:
        if name in self.presets:
            self.active_preset_name = name
            self.save_settings()
            self.sigPresetChanged.emit(name)
            return True
        return False

    def update_active_preset_content(
        self,
        preamp_db: float,
        bands: List[EQBand],
        channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        curr = dict(self.presets.get(self.active_preset_name, {}))
        curr["preamp_db"] = float(preamp_db)
        curr["bands"] = [_band_to_dict(b) for b in bands]
        if channel_bands is not None:
            serialized = _serialize_channel_bands(channel_bands)
            if serialized:
                curr["channel_bands"] = serialized
            else:
                curr.pop("channel_bands", None)
        self.presets[self.active_preset_name] = curr
        self.save_presets()

    def add_preset(
        self,
        name: str,
        preamp_db: float,
        bands: List[EQBand],
        channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> bool:
        name = name.strip()
        if not name or name in self.presets:
            return False
        data: Dict[str, Any] = {
            "preamp_db": float(preamp_db),
            "bands": [_band_to_dict(b) for b in bands],
        }
        serialized_ch = _serialize_channel_bands(channel_bands)
        if serialized_ch:
            data["channel_bands"] = serialized_ch
        self.presets[name] = data
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
        emit_changed: bool = True,
        channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> str:
        name = name.strip()
        if not name:
            name = "AutoEQ Profile"
        data: Dict[str, Any] = {
            "preamp_db": float(preamp_db),
            "bands": [_band_to_dict(b) for b in bands],
        }
        if autoeq_metadata:
            data["autoeq"] = autoeq_metadata
        elif name in self.presets and "autoeq" in self.presets[name]:
            data["autoeq"] = self.presets[name]["autoeq"]

        if channel_bands is not None:
            serialized_ch = _serialize_channel_bands(channel_bands)
            if serialized_ch:
                data["channel_bands"] = serialized_ch
        elif name in self.presets and "channel_bands" in self.presets[name]:
            existing_ch = _serialize_channel_bands(self.presets[name].get("channel_bands"))
            if existing_ch:
                data["channel_bands"] = existing_ch

        self.presets[name] = data
        self.active_preset_name = name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        if emit_changed:
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
        was_active = (self.active_preset_name == old_name)
        self.presets[new_name] = self.presets.pop(old_name)
        if was_active:
            self.active_preset_name = new_name
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        if was_active:
            self.sigPresetChanged.emit(new_name)
        return True

    def delete_preset(self, name: str) -> bool:
        if name not in self.presets or len(self.presets) <= 1:
            return False
        was_active = (self.active_preset_name == name)
        del self.presets[name]
        if was_active:
            self.active_preset_name = next(iter(self.presets.keys()))
        self.save_presets()
        self.save_settings()
        self.sigPresetsListChanged.emit()
        if was_active:
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
        channel_bands: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        """
        Schedules a debounced write to Equalizer APO file (~50 ms).
        Supports optional per-channel bands (`channel_bands={'L': [...], 'R': [...]}`)
        as well as per-band `channel` attributes ('L'/'R'/'ALL').
        """
        cloned: List[EQBand] = []
        for idx, b in enumerate(bands or (), start=1):
            sb = _coerce_band(b, default_band_id=idx)
            if sb is not None:
                cloned.append(sb)
        self._pending_bands = cloned
        self._pending_channel_bands = (
            _normalize_channel_bands_dict(channel_bands) if channel_bands is not None else None
        )
        self._pending_preamp = float(preamp_db)
        self._pending_preset_name = preset_name or self.active_preset_name
        self._pending_bypassed = bool(bypassed)
        self._has_pending_export = True
        self._debounce_timer.start(50)

    def flush_apo_export(self) -> None:
        """Synchronously flushes any pending debounced APO export (e.g. on application exit)."""
        if self._debounce_timer.isActive() or self._has_pending_export:
            self._debounce_timer.stop()
            self._do_write_apo_file()

    def diagnose_apo(self) -> Dict[str, Any]:
        """Runs structured Equalizer APO diagnostics for this manager's active APO path and device."""
        return diagnose_apo_setup(
            self.active_apo_path or self.apo_target_path,
            active_device_name=self.active_device_name or "all",
        )

    def repair_apo(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Attempts one-click permission & config.txt include repair for this manager's APO target."""
        self._verified_include_path = None
        ok, msg, diag = repair_apo_setup(
            self.apo_target_path,
            active_device_name=self.active_device_name or "all",
        )
        if ok:
            self._verified_include_path = os.path.normcase(os.path.abspath(self.apo_target_path))
        return ok, msg, diag

    def _format_apo_filter_line(self, index: int, band: EQBand, force_off: bool = False) -> str:
        return _format_apo_filter_line(index, band, force_off=force_off)

    def _do_write_apo_file(self) -> None:
        """Executes an atomic write to APO target path with AppData fallback and non-elevating config.txt include check."""
        self._has_pending_export = False

        # Resolve optional per-channel bands without duplicating inline L/R bands
        effective_channel_bands = self._pending_channel_bands
        if effective_channel_bands is None:
            has_inline_lr = any(_band_channel(b) in ("L", "R") for b in self._pending_bands)
            if not has_inline_lr:
                raw_preset = self.presets.get(self._pending_preset_name or self.active_preset_name)
                if isinstance(raw_preset, dict) and raw_preset.get("channel_bands"):
                    effective_channel_bands = _normalize_channel_bands_dict(raw_preset.get("channel_bands"))

        content = export_eq_to_apo_text(
            preamp_db=self._pending_preamp,
            bands=self._pending_bands,
            device_name=self.active_device_name or "all",
            channel_bands=effective_channel_bands,
            preset_name=self._pending_preset_name,
            bypassed=self._pending_bypassed,
            include_timestamp=True,
        )

        target_path = self.apo_target_path
        appdata_path = os.path.join(get_app_data_dir(), "clean_apo_generated.txt")
        used_path = target_path
        success = False
        msg = ""

        def _ensure_include_cached(path_to_include: str) -> bool:
            norm_p = os.path.normcase(os.path.abspath(path_to_include))
            if self._verified_include_path == norm_p:
                return True
            # Never auto-trigger UAC prompts inside the 50 ms debounce write timer
            inc_ok = ensure_apo_root_config_include(path_to_include, allow_elevation=False)
            if inc_ok:
                self._verified_include_path = norm_p
            return inc_ok

        try:
            # 1. Atomically write primary target file directly (retries brief Windows sharing locks)
            _atomic_write_text(target_path, content, retries=4, retry_delay=0.025)

            # 2. Verify Equalizer APO's config.txt includes our target file (cached per target path)
            inc_ok = _ensure_include_cached(target_path)

            # 3. Clean legacy peqdb_generated.txt once per target dir so it does not hold duplicate filters
            target_dir = os.path.dirname(os.path.abspath(target_path))
            norm_dir = os.path.normcase(target_dir)
            if norm_dir not in self._legacy_cleaned_dirs and os.path.isdir(target_dir):
                self._legacy_cleaned_dirs.add(norm_dir)
                legacy_path = os.path.join(target_dir, "peqdb_generated.txt")
                if os.path.normcase(legacy_path) != os.path.normcase(os.path.abspath(target_path)) and os.path.exists(legacy_path):
                    try:
                        _atomic_write_text(legacy_path, "# Clean APO now outputs to clean_apo_generated.txt\n")
                    except Exception:
                        pass

            success = True
            used_path = target_path
            if inc_ok:
                msg = f"Synced with Equalizer APO ({os.path.basename(target_path)})"
            else:
                msg = f"Wrote {os.path.basename(target_path)} (Grant APO permissions to update config.txt)"
        except (PermissionError, OSError):
            # Target is not directly writable without elevation; fall back atomically to AppData path
            used_path = appdata_path
            try:
                _atomic_write_text(appdata_path, content, retries=4, retry_delay=0.025)
                inc_ok = _ensure_include_cached(appdata_path)
                success = True
                if inc_ok:
                    msg = f"Synced with Equalizer APO via AppData ({os.path.basename(appdata_path)})"
                else:
                    msg = "Saved to AppData. Equalizer APO config.txt needs permissions to include it."
            except Exception:
                used_path = LOCAL_FALLBACK_PATH
                try:
                    _atomic_write_text(LOCAL_FALLBACK_PATH, content, retries=2, retry_delay=0.02)
                    success = False
                    msg = f"Permission denied for APO folder. Saved to local: {LOCAL_FALLBACK_PATH}"
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
        - Equalizer APO .txt configs (parametric Filter: and GraphicEQ: conversion)
        - Room EQ Wizard (REW) exports (incl. thousands commas, tabs, kHz, inline comments)
        - Peace GUI .peace configs (indexed keys, filter types, enable switches)
        - JSON presets (single preset, wrapped 'presets', or Clean APO's flat presets.json)
        """
        if not file_path or not os.path.exists(file_path):
            return None
        preset_name, preamp_db, bands, _info = preview_eq_import(file_path)
        if bands:
            return preset_name, preamp_db, bands
        return None
