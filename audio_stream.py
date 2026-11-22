"""
Background audio capture worker for Windows WASAPI Loopback and real-time FFT spectrum analysis.
Includes full-frequency pink-noise tilt compensation and dynamic auto-gain normalization.
"""

from __future__ import annotations
import time
import queue
import threading
from typing import Optional, Tuple, List, Any
import numpy as np
import scipy.sparse as sp
from PyQt6 import QtCore
from presets import clean_apo_device_name

_PYAUDIO_LOCK = threading.RLock()

HAVE_PYAUDIOWPATCH = False
try:
    import pyaudiowpatch as pyaudio
    HAVE_PYAUDIOWPATCH = True
except ImportError:
    pass

HAVE_SOUNDDEVICE = False
try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except ImportError:
    pass


def get_available_audio_devices() -> List[Tuple[Optional[int], str]]:
    """
    Returns a list of available audio capture / loopback devices as (index, display_name).
    Includes auto-tracking Windows Default Audio endpoint.
    Dynamically integrates with QMediaDevices so newly plugged-in devices are always visible.
    """
    devices: List[Tuple[Optional[int], str]] = []
    # 0. Always offer automatic Default Windows Audio tracker as the top choice
    devices.append((None, "Default Windows Audio (Auto-Follow)"))

    seen_indices = set()
    seen_names = set()

    if HAVE_PYAUDIOWPATCH:
        with _PYAUDIO_LOCK:
            p = None
            try:
                p = pyaudio.PyAudio()

                # 1. Query all loopback generator devices
                try:
                    for dev in p.get_loopback_device_info_generator():
                        idx = dev["index"]
                        if idx not in seen_indices:
                            seen_indices.add(idx)
                            name = dev["name"].replace(" [Loopback]", "").strip()
                            seen_names.add(clean_apo_device_name(name).lower())
                            devices.append((idx, f"{name} (Loopback)"))
                except Exception as e:
                    print(f"[AudioDevices] Loopback generator error: {e}")

                # 2. Also inspect all WASAPI output devices to ensure plugged-in headphones/DACs are found
                try:
                    for i in range(p.get_device_count()):
                        dev = p.get_device_info_by_index(i)
                        if dev.get("maxOutputChannels", 0) > 0 and dev.get("hostApi") is not None:
                            try:
                                host_api = p.get_host_api_info_by_index(dev["hostApi"])
                                if "wasapi" in host_api.get("name", "").lower():
                                    analog = p.get_wasapi_loopback_analogue_by_index(dev["index"])
                                    a_idx = analog["index"]
                                    if a_idx not in seen_indices:
                                        seen_indices.add(a_idx)
                                        a_name = analog["name"].replace(" [Loopback]", "").strip()
                                        seen_names.add(clean_apo_device_name(a_name).lower())
                                        devices.append((a_idx, f"{a_name} (Loopback)"))
                            except Exception:
                                pass
                except Exception as e:
                    print(f"[AudioDevices] WASAPI output inspection error: {e}")

            except Exception as e:
                print(f"[AudioDevices] Error querying PyAudioWPatch: {e}")
            finally:
                if p:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    # 3. Seamlessly merge in any active Windows playback devices from QMediaDevices
    # This guarantees that newly plugged-in USB DACs or headphones are immediately visible in the menu!
    try:
        from PyQt6.QtMultimedia import QMediaDevices
        for d in QMediaDevices.audioOutputs():
            desc = d.description().strip()
            clean_d = clean_apo_device_name(desc).lower()
            if desc and clean_d not in seen_names:
                seen_names.add(clean_d)
                devices.append((None, f"{desc} (Loopback)"))
    except Exception as e:
        print(f"[AudioDevices] QMediaDevices loopback discovery notice: {e}")

    if len(devices) <= 1 and HAVE_SOUNDDEVICE:
        try:
            sd_devs = sd.query_devices()
            for i, d in enumerate(sd_devs):
                if d.get("max_input_channels", 0) > 0 or "stereo mix" in d.get("name", "").lower():
                    devices.append((i, d["name"]))
        except Exception as e:
            print(f"[AudioDevices] Error querying sounddevice: {e}")

    return devices


def get_available_playback_devices() -> List[str]:
    """
    Returns a list of available audio playback/output device names for Equalizer APO targeting.
    Queries QMediaDevices (Windows MMDevice API) first for instant real-time hotplug awareness,
    falling back to PyAudio and sounddevice.
    """
    playback_devs: List[str] = []
    seen = set()

    # 1. Primary: Direct Windows MMDevice query via QMediaDevices (instant hotplug awareness)
    try:
        from PyQt6.QtMultimedia import QMediaDevices
        outputs = QMediaDevices.audioOutputs()
        if outputs:
            for d in outputs:
                desc = d.description().strip()
                if desc and desc not in seen:
                    seen.add(desc)
                    playback_devs.append(desc)
            if playback_devs:
                return playback_devs
    except Exception as e:
        print(f"[PlaybackDevices] QMediaDevices query notice: {e}")

    # 2. Secondary fallback: PyAudio
    if HAVE_PYAUDIOWPATCH:
        with _PYAUDIO_LOCK:
            p = None
            try:
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    d = p.get_device_info_by_index(i)
                    if d.get("maxOutputChannels", 0) > 0 and d.get("hostApi") is not None:
                        try:
                            api = p.get_host_api_info_by_index(d["hostApi"])
                            if "wasapi" in api.get("name", "").lower():
                                name = d["name"].strip()
                                if name not in seen:
                                    seen.add(name)
                                    playback_devs.append(name)
                        except Exception:
                            pass
            except Exception as e:
                print(f"[PlaybackDevices] PyAudio error: {e}")
            finally:
                if p:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    # 3. Tertiary fallback: sounddevice
    if not playback_devs and HAVE_SOUNDDEVICE:
        try:
            for d in sd.query_devices():
                if d.get("max_output_channels", 0) > 0:
                    name = d["name"].strip()
                    if name not in seen and not name.startswith("Microsoft Sound Mapper") and not name.startswith("Primary Sound Driver"):
                        seen.add(name)
                        playback_devs.append(name)
        except Exception as e:
            print(f"[PlaybackDevices] Sounddevice error: {e}")

    return playback_devs


class AudioSpectrumWorker(QtCore.QThread):
    """
    Worker thread capturing desktop audio loopback from the chosen device,
    computing 2048-sample Hanning FFTs with selectable tilt compensation
    ("pink" +3.0 dB/oct, "slope" +4.5 dB/oct, "flat" 0 dB/oct),
    selectable attack/release speed presets, freeze mode, decaying peak-hold,
    dynamic auto-gain normalization, and log-frequency smoothing.
    """
    sigSpectrumReady = QtCore.pyqtSignal(np.ndarray, np.ndarray)  # (freqs, mags_db)
    sigPeakHoldReady = QtCore.pyqtSignal(object, object)          # (freqs, peak_hold_mags_db)
    sigStatusChanged = QtCore.pyqtSignal(str)                     # Status message

    _TILT_MODES = ("pink", "slope", "flat")
    _SPEED_PRESETS = {
        "fast": (0.65, 0.28),
        "medium": (0.42, 0.15),
        "slow": (0.22, 0.06),
    }
    _FLOOR_DB = -20.0
    _PEAK_HOLD_SEC = 0.8
    _PEAK_DECAY_DB_PER_SEC = 6.0

    # Safe class-level defaults for partial/mock instances constructed via __new__
    _tilt_mode: str = "pink"
    _speed_mode: str = "medium"
    _attack_alpha: float = 0.42
    _release_alpha: float = 0.15
    _frozen: bool = False
    _peak_hold_enabled: bool = False
    _rta_sample_rate: int = 48000
    _rta_tilt_mode: str = "pink"
    _peak_at_floor: bool = True

    def __init__(
        self,
        window_size: int = 2048,
        target_points: int = 1000,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.window_size = window_size
        self.target_points = target_points
        self.target_freqs = np.geomspace(20.0, 20000.0, target_points)
        self.hanning_window = np.hanning(window_size)

        self._tilt_mode: str = "pink"
        self._speed_mode: str = "medium"
        self._attack_alpha: float = self._SPEED_PRESETS["medium"][0]
        self._release_alpha: float = self._SPEED_PRESETS["medium"][1]
        self._frozen: bool = False
        self._peak_hold_enabled: bool = False

        self.sample_rate = 48000
        self._rta_sample_rate = self.sample_rate
        self._rta_tilt_mode = self._tilt_mode
        self._rta_matrix = self._build_rta_matrix(self.sample_rate, tilt_mode=self._tilt_mode)

        self._state_lock = threading.Lock()
        self._smoothed_mags = np.full(target_points, self._FLOOR_DB, dtype=np.float32)
        self._peak_hold_mags = np.full(target_points, self._FLOOR_DB, dtype=np.float32)
        self._peak_hold_age = np.zeros(target_points, dtype=np.float32)
        self._at_floor = True
        self._peak_at_floor = True
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._ever_started = False
        self._enabled = True
        self._device_index: Optional[int] = None
        self._device_name: Optional[str] = None

        # Dynamic AGC normalization gain
        self._agc_gain_db = 38.0

        self._audio_queue: queue.Queue = queue.Queue(maxsize=32)

    def _build_rta_matrix(self, sample_rate: int, tilt_mode: Optional[str] = None) -> sp.csr_matrix:
        """
        Builds a sparse 1/6th-octave logarithmic filterbank matrix mapping FFT power
        bins to log-spaced frequencies with selectable tilt compensation:
        - "pink":  +3.0 dB/oct (power ~ f)
        - "slope": +4.5 dB/oct (power ~ f**1.5)
        - "flat":   0.0 dB/oct (power ~ 1.0)
        """
        sr = max(8000, int(sample_rate))
        nyquist = 0.5 * sr
        mode = (tilt_mode or self._tilt_mode or "pink").strip().lower()
        if mode not in self._TILT_MODES:
            mode = "pink"

        fft_freqs = np.fft.rfftfreq(self.window_size, 1.0 / sr)
        sigma = 1.0 / 12.0  # 1/6th octave bandwidth
        log_fft = np.log2(np.maximum(10.0, fft_freqs))
        log_targets = np.log2(self.target_freqs)

        rows, cols, data = [], [], []
        for i, log_fc in enumerate(log_targets):
            fc = float(self.target_freqs[i])
            if fc > nyquist:
                # Never extrapolate or ramp above Nyquist for low sample-rate devices
                continue
            diff = log_fft - log_fc
            mask = (np.abs(diff) < 3.0 * sigma) & (fft_freqs >= 15.0)
            if mode == "flat":
                tilt_scale = 1.0
            elif mode == "slope":
                tilt_scale = (fc / 1000.0) ** 1.5  # +4.52 dB/oct analyzer slope, normalized at 1 kHz
            else:
                tilt_scale = fc / 1000.0           # +3.01 dB/oct pink tilt, normalized at 1 kHz
            if np.any(mask):
                w = np.exp(-0.5 * (diff[mask] / sigma) ** 2)
                w = (w / np.sum(w)) * tilt_scale
                col_indices = np.where(mask)[0]
                for c, weight in zip(col_indices, w):
                    rows.append(i)
                    cols.append(int(c))
                    data.append(float(weight))
            elif len(fft_freqs) > 1:
                # Exclude bin 0 (DC offset) on high sample rates (e.g., 96/192 kHz)
                closest = 1 + int(np.argmin(np.abs(fft_freqs[1:] - fc)))
                rows.append(i)
                cols.append(closest)
                data.append(tilt_scale)

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.target_freqs), len(fft_freqs)),
            dtype=np.float32,
        )

    def _sync_rta_matrix_if_needed(self) -> None:
        """Rebuilds the sparse RTA matrix outside _state_lock if tilt mode or sample rate changed."""
        with self._state_lock:
            target_sr = self.sample_rate
            target_tilt = self._tilt_mode
            if self._rta_matrix is not None and self._rta_sample_rate == target_sr and self._rta_tilt_mode == target_tilt:
                return
        new_matrix = self._build_rta_matrix(target_sr, tilt_mode=target_tilt)
        with self._state_lock:
            if self.sample_rate == target_sr and self._tilt_mode == target_tilt:
                self._rta_matrix = new_matrix
                self._rta_sample_rate = target_sr
                self._rta_tilt_mode = target_tilt

    def set_tilt_mode(self, mode: str) -> None:
        """Thread-safe setter for spectrum tilt mode: 'pink', 'slope', or 'flat'."""
        norm = str(mode or "").strip().lower()
        if norm not in self._TILT_MODES:
            return
        with self._state_lock:
            if self._tilt_mode == norm:
                return
            self._tilt_mode = norm
            sr = self.sample_rate
            frozen = self._frozen
            if not frozen:
                self._peak_hold_mags.fill(self._FLOOR_DB)
                self._peak_hold_age.fill(0.0)
                self._peak_at_floor = True
        # Rebuild outside the lock so the audio thread is never stalled
        new_matrix = self._build_rta_matrix(sr, tilt_mode=norm)
        peak_copy: Optional[np.ndarray] = None
        with self._state_lock:
            if self._tilt_mode == norm and self.sample_rate == sr:
                self._rta_matrix = new_matrix
                self._rta_sample_rate = sr
                self._rta_tilt_mode = norm
            if self._peak_hold_enabled and not self._frozen:
                peak_copy = self._peak_hold_mags.copy()
        if peak_copy is not None:
            self.sigPeakHoldReady.emit(self.target_freqs, peak_copy)

    def get_tilt_mode(self) -> str:
        """Returns the active tilt mode ('pink', 'slope', or 'flat')."""
        with self._state_lock:
            return self._tilt_mode

    def set_speed_mode(self, mode: str) -> None:
        """Thread-safe setter for spectrum attack/release smoothing speed: 'fast', 'medium', or 'slow'."""
        norm = str(mode or "").strip().lower()
        preset = self._SPEED_PRESETS.get(norm)
        if preset is None:
            return
        with self._state_lock:
            self._speed_mode = norm
            self._attack_alpha, self._release_alpha = preset

    def get_speed_mode(self) -> str:
        """Returns the active smoothing speed preset ('fast', 'medium', or 'slow')."""
        with self._state_lock:
            return self._speed_mode

    def set_frozen(self, frozen: bool) -> None:
        """Pauses or resumes live spectrum and peak-hold display updates without dropping audio capture."""
        with self._state_lock:
            self._frozen = bool(frozen)

    def is_frozen(self) -> bool:
        """Returns True if spectrum display updates are currently frozen."""
        with self._state_lock:
            return self._frozen

    def set_peak_hold_enabled(self, enabled: bool) -> None:
        """Enables or disables the decaying peak-hold envelope emitted via sigPeakHoldReady."""
        flag = bool(enabled)
        emit_peak: Optional[np.ndarray] = None
        with self._state_lock:
            was_enabled = self._peak_hold_enabled
            self._peak_hold_enabled = flag
            if flag:
                if not was_enabled:
                    if self._enabled and not self._at_floor:
                        self._peak_hold_mags[:] = self._smoothed_mags
                    else:
                        self._peak_hold_mags.fill(self._FLOOR_DB)
                    self._peak_hold_age.fill(0.0)
                    self._peak_at_floor = bool(np.all(self._peak_hold_mags <= self._FLOOR_DB))
                emit_peak = self._peak_hold_mags.copy()
            else:
                self._peak_hold_mags.fill(self._FLOOR_DB)
                self._peak_hold_age.fill(0.0)
                self._peak_at_floor = True
                emit_peak = self._peak_hold_mags.copy()
        if emit_peak is not None:
            self.sigPeakHoldReady.emit(self.target_freqs, emit_peak)

    def is_peak_hold_enabled(self) -> bool:
        """Returns True if the decaying peak-hold envelope is enabled."""
        with self._state_lock:
            return self._peak_hold_enabled

    def reset_peak_hold(self) -> None:
        """Resets the peak-hold envelope back to the current live spectrum (or floor when idle)."""
        emit_peak: Optional[np.ndarray] = None
        with self._state_lock:
            if self._enabled and not self._at_floor:
                self._peak_hold_mags[:] = self._smoothed_mags
            else:
                self._peak_hold_mags.fill(self._FLOOR_DB)
            self._peak_hold_age.fill(0.0)
            self._peak_at_floor = bool(np.all(self._peak_hold_mags <= self._FLOOR_DB))
            if self._peak_hold_enabled:
                emit_peak = self._peak_hold_mags.copy()
        if emit_peak is not None:
            self.sigPeakHoldReady.emit(self.target_freqs, emit_peak)

    def _update_peak_hold_locked(self, dt: float) -> Optional[np.ndarray]:
        """Updates the decaying peak-hold envelope (caller must hold self._state_lock)."""
        if not self._peak_hold_enabled:
            return None
        dt_safe = float(min(max(dt, 0.0), 0.1))
        current = self._smoothed_mags
        rising = current >= self._peak_hold_mags
        if np.any(rising):
            self._peak_hold_mags[rising] = current[rising]
            self._peak_hold_age[rising] = 0.0

        holding = ~rising
        if np.any(holding):
            self._peak_hold_age[holding] += dt_safe
            decaying = holding & (self._peak_hold_age > self._PEAK_HOLD_SEC)
            if np.any(decaying):
                decayed = self._peak_hold_mags[decaying] - (self._PEAK_DECAY_DB_PER_SEC * dt_safe)
                floor_bound = np.maximum(current[decaying], self._FLOOR_DB)
                self._peak_hold_mags[decaying] = np.maximum(floor_bound, decayed)

        np.clip(self._peak_hold_mags, self._FLOOR_DB, 18.0, out=self._peak_hold_mags)
        if bool(np.all(self._peak_hold_mags <= self._FLOOR_DB)):
            if self._peak_at_floor:
                return None
            self._peak_at_floor = True
            return self._peak_hold_mags.copy()
        self._peak_at_floor = False
        return self._peak_hold_mags.copy()

    def start(self, priority: QtCore.QThread.Priority = QtCore.QThread.Priority.InheritPriority) -> None:
        self._stop_event.clear()
        self._ever_started = True
        super().start(priority)

    def _ensure_running_if_started(self) -> None:
        if self._ever_started and not self.isRunning() and not self._stop_event.is_set():
            super().start()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or bypass the spectrum analyzer stream."""
        flat_copy: Optional[np.ndarray] = None
        flat_peak_copy: Optional[np.ndarray] = None
        with self._state_lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._smoothed_mags.fill(self._FLOOR_DB)
                self._at_floor = True
                flat_copy = self._smoothed_mags.copy()
                self._peak_hold_mags.fill(self._FLOOR_DB)
                self._peak_hold_age.fill(0.0)
                self._peak_at_floor = True
                if self._peak_hold_enabled:
                    flat_peak_copy = self._peak_hold_mags.copy()
            else:
                self._at_floor = False
        if flat_copy is not None:
            self.sigSpectrumReady.emit(self.target_freqs, flat_copy)
        if flat_peak_copy is not None:
            self.sigPeakHoldReady.emit(self.target_freqs, flat_peak_copy)

    def set_device_index(self, device_index: Optional[int]) -> None:
        """Switches the active audio loopback capture device by index."""
        changed = False
        with self._state_lock:
            if self._device_index != device_index or self._device_name is not None:
                self._device_index = device_index
                self._device_name = None
                changed = True
        if changed:
            self._restart_event.set()
            self._ensure_running_if_started()

    def set_device_target(self, device_index: Optional[int], device_name: Optional[str] = None) -> None:
        """Switches the active audio loopback capture device by index and/or friendly name."""
        changed = False
        with self._state_lock:
            if self._device_index != device_index or self._device_name != device_name:
                self._device_index = device_index
                self._device_name = device_name
                changed = True
        if changed:
            self._restart_event.set()
            self._ensure_running_if_started()

    def restart_stream(self) -> None:
        """Triggers a clean restart of the audio loopback stream and device re-enumeration."""
        self._restart_event.set()
        self._ensure_running_if_started()

    def stop(self) -> None:
        """Stops the worker thread cleanly."""
        self._stop_event.set()
        self._restart_event.set()
        if self.isRunning() and not self.wait(2000):
            print("[AudioStream] Warning: worker thread did not stop within 2000ms.")

    def run(self) -> None:
        while not self._stop_event.is_set():
            self._restart_event.clear()
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

            if HAVE_PYAUDIOWPATCH:
                self._run_pyaudiowpatch_stream()
            elif HAVE_SOUNDDEVICE:
                self._run_sounddevice_stream()
            else:
                self._run_fallback_decay_loop(timeout_sec=0.0)
                break

    # -------------------------------------------------------------------------
    # PyAudioWPatch Loopback Stream
    # -------------------------------------------------------------------------
    def _run_pyaudiowpatch_stream(self) -> None:
        pa = None
        stream = None
        unexpected_disconnect = False
        try:
            with self._state_lock:
                target_index = self._device_index
                target_name = self._device_name

            with _PYAUDIO_LOCK:
                if self._stop_event.is_set():
                    return
                pa = pyaudio.PyAudio()
                loopback_device = None

                # 1. Target matching by device friendly name (exact match first, then substring)
                if target_name and "auto-follow" not in target_name.lower():
                    clean_target = clean_apo_device_name(target_name).lower()
                    if clean_target and clean_target != "all":
                        try:
                            candidates = list(pa.get_loopback_device_info_generator())
                            partial_match = None
                            for dev in candidates:
                                d_name = dev["name"].replace(" [Loopback]", "").strip()
                                clean_d = clean_apo_device_name(d_name).lower()
                                if not clean_d or clean_d == "all":
                                    continue
                                if clean_target == clean_d:
                                    loopback_device = dev
                                    break
                                if partial_match is None and (clean_target in clean_d or clean_d in clean_target):
                                    partial_match = dev
                            if loopback_device is None:
                                loopback_device = partial_match
                        except Exception as e:
                            print(f"[AudioStream] Name lookup notice: {e}")

                # 2. Target lookup by device index
                if not loopback_device and target_index is not None:
                    try:
                        dev_info = pa.get_device_info_by_index(target_index)
                        if dev_info.get("isLoopbackDevice", False) or dev_info.get("maxInputChannels", 0) > 0:
                            loopback_device = dev_info
                        else:
                            loopback_device = pa.get_wasapi_loopback_analogue_by_index(target_index)
                    except Exception:
                        loopback_device = None

                # 3. Default Windows Audio (Auto-Follow) loopback endpoint
                if not loopback_device:
                    try:
                        loopback_device = pa.get_default_wasapi_loopback()
                    except Exception:
                        pass

                # 4. Fallback search via WASAPI default output device
                if not loopback_device:
                    try:
                        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
                        default_speakers = pa.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                        for dev in pa.get_loopback_device_info_generator():
                            if default_speakers["name"] in dev["name"]:
                                loopback_device = dev
                                break
                    except Exception:
                        pass

                if not loopback_device:
                    try:
                        loopback_device = next(pa.get_loopback_device_info_generator())
                    except StopIteration:
                        pass

                if not loopback_device:
                    self.sigStatusChanged.emit("No loopback audio device found.")
                    # Release _PYAUDIO_LOCK before waiting in fallback decay
                    pa_to_close = pa
                    pa = None
                    try:
                        pa_to_close.terminate()
                    except Exception:
                        pass
                else:
                    new_rate = max(8000, int(loopback_device.get("defaultSampleRate", 48000)))
                    with self._state_lock:
                        need_rta = (
                            self._rta_matrix is None
                            or new_rate != self.sample_rate
                            or self._rta_sample_rate != new_rate
                            or self._rta_tilt_mode != self._tilt_mode
                        )
                        cur_tilt = self._tilt_mode
                    if need_rta:
                        new_mat = self._build_rta_matrix(new_rate, tilt_mode=cur_tilt)
                        with self._state_lock:
                            self.sample_rate = new_rate
                            if self._tilt_mode == cur_tilt:
                                self._rta_matrix = new_mat
                                self._rta_sample_rate = new_rate
                                self._rta_tilt_mode = cur_tilt
                    channels = max(1, int(loopback_device.get("maxInputChannels", 2)))
                    dev_name = loopback_device["name"].replace(" [Loopback]", "")
                    self.sigStatusChanged.emit(f"Capturing: {dev_name}")

                    def callback(in_data, frame_count, time_info, status):
                        if self._stop_event.is_set() or self._restart_event.is_set():
                            return (None, pyaudio.paComplete)
                        if not self._enabled:
                            return (None, pyaudio.paContinue)
                        try:
                            arr = np.frombuffer(in_data, dtype=np.float32)
                            if channels > 1:
                                usable = (len(arr) // channels) * channels
                                if usable == 0:
                                    return (None, pyaudio.paContinue)
                                arr = arr[:usable].reshape(-1, channels).mean(axis=1)
                            if not self._audio_queue.full():
                                self._audio_queue.put_nowait(arr.copy())
                        except Exception:
                            pass
                        return (None, pyaudio.paContinue)

                    stream = pa.open(
                        format=pyaudio.paFloat32,
                        channels=channels,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=loopback_device["index"],
                        frames_per_buffer=1024,
                        stream_callback=callback,
                    )
                    stream.start_stream()

            if stream is None:
                self._run_fallback_decay_loop(timeout_sec=2.0)
                return

            self._process_audio_stream(stream=stream)
            if not self._stop_event.is_set() and not self._restart_event.is_set():
                unexpected_disconnect = True

        except Exception as e:
            self.sigStatusChanged.emit(f"Audio error: {e}")
            self._run_fallback_decay_loop(timeout_sec=2.0)
        finally:
            with _PYAUDIO_LOCK:
                if stream:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
                if pa:
                    try:
                        pa.terminate()
                    except Exception:
                        pass

        if unexpected_disconnect:
            self._run_fallback_decay_loop(timeout_sec=1.0)

    # -------------------------------------------------------------------------
    # sounddevice Fallback Stream
    # -------------------------------------------------------------------------
    def _run_sounddevice_stream(self) -> None:
        unexpected_disconnect = False
        try:
            with self._state_lock:
                target_index = self._device_index

            dev_idx = target_index if target_index is not None else sd.default.device[0]
            if dev_idx is None or int(dev_idx) < 0:
                self.sigStatusChanged.emit("No input audio device found.")
                self._run_fallback_decay_loop(timeout_sec=2.0)
                return

            dev_info = sd.query_devices(dev_idx)
            new_rate = max(8000, int(dev_info.get("default_samplerate", 48000)))
            with self._state_lock:
                need_rta = (
                    self._rta_matrix is None
                    or new_rate != self.sample_rate
                    or self._rta_sample_rate != new_rate
                    or self._rta_tilt_mode != self._tilt_mode
                )
                cur_tilt = self._tilt_mode
            if need_rta:
                new_mat = self._build_rta_matrix(new_rate, tilt_mode=cur_tilt)
                with self._state_lock:
                    self.sample_rate = new_rate
                    if self._tilt_mode == cur_tilt:
                        self._rta_matrix = new_mat
                        self._rta_sample_rate = new_rate
                        self._rta_tilt_mode = cur_tilt
            channels = min(2, max(1, int(dev_info.get("max_input_channels", 1))))
            self.sigStatusChanged.emit(f"Capturing: {dev_info['name']}")

            def sd_callback(indata, frames, time_info, status):
                if self._stop_event.is_set() or self._restart_event.is_set() or not self._enabled:
                    return
                try:
                    arr = indata.mean(axis=1) if indata.ndim > 1 else indata
                    if not self._audio_queue.full():
                        self._audio_queue.put_nowait(arr.copy())
                except Exception:
                    pass

            with sd.InputStream(
                device=dev_idx,
                channels=channels,
                samplerate=self.sample_rate,
                blocksize=1024,
                callback=sd_callback,
            ) as sd_stream:
                self._process_audio_stream(stream=sd_stream)
                if not self._stop_event.is_set() and not self._restart_event.is_set():
                    unexpected_disconnect = True

        except Exception as e:
            self.sigStatusChanged.emit(f"sounddevice error: {e}")
            self._run_fallback_decay_loop(timeout_sec=2.0)

        if unexpected_disconnect:
            self._run_fallback_decay_loop(timeout_sec=1.0)

    # -------------------------------------------------------------------------
    # FFT DSP Loop with Tilt & Dynamic Normalization
    # -------------------------------------------------------------------------
    def _ingest_chunk(self, buffer: np.ndarray, chunk: np.ndarray) -> bool:
        if len(chunk) == 0:
            return False
        if not np.all(np.isfinite(chunk)):
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        if len(chunk) >= self.window_size:
            buffer[:] = chunk[-self.window_size:]
        else:
            n = len(chunk)
            buffer[:-n] = buffer[n:]
            buffer[-n:] = chunk
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        return np.isfinite(rms) and rms > 1e-5

    def _process_audio_stream(self, stream: Optional[Any] = None) -> None:
        last_emit_time = time.perf_counter()
        last_audio_activity_time = last_emit_time - 1.0
        buffer = np.zeros(self.window_size, dtype=np.float32)

        while not self._stop_event.is_set() and not self._restart_event.is_set():
            # Check if underlying hardware stream stopped or was unplugged
            if stream is not None:
                try:
                    if hasattr(stream, "is_active"):
                        if not stream.is_active():
                            print("[AudioStream] Hardware stream stopped or endpoint disconnected.")
                            break
                    elif hasattr(stream, "active"):
                        if not stream.active:
                            print("[AudioStream] Sounddevice stream stopped or endpoint disconnected.")
                            break
                except Exception:
                    break

            # 1. Fetch available chunk with small timeout
            try:
                chunk = self._audio_queue.get(timeout=0.015)
                if self._ingest_chunk(buffer, chunk):
                    last_audio_activity_time = time.perf_counter()
            except queue.Empty:
                pass

            # Drain any accumulated chunks so audio visualization stays strictly in real-time
            while not self._audio_queue.empty():
                try:
                    chunk = self._audio_queue.get_nowait()
                    if self._ingest_chunk(buffer, chunk):
                        last_audio_activity_time = time.perf_counter()
                except queue.Empty:
                    break

            now = time.perf_counter()
            if now - last_emit_time >= 0.025:
                dt = now - last_emit_time
                self._sync_rta_matrix_if_needed()
                emit_copy: Optional[np.ndarray] = None
                peak_copy: Optional[np.ndarray] = None
                with self._state_lock:
                    if self._enabled and not self._frozen:
                        is_active = (now - last_audio_activity_time) < 0.25
                        if is_active:
                            win_buf = buffer * self.hanning_window
                            X = np.fft.rfft(win_buf)
                            power = ((2.0 / self.window_size) * np.abs(X)) ** 2
                            binned_power = self._rta_matrix.dot(power)
                            if not np.all(np.isfinite(binned_power)):
                                binned_power = np.nan_to_num(binned_power, nan=0.0, posinf=1.0, neginf=0.0)
                            mags_db = 10.0 * np.log10(np.maximum(1e-12, binned_power))

                            # Dynamic Auto-Gain Normalization (AGC)
                            current_peak = float(np.max(mags_db)) if mags_db.size > 0 else -120.0
                            if not np.isfinite(current_peak):
                                current_peak = -120.0
                            target_peak_graph = -3.0  # Center peak around -3 dB on [-20, +20] dB canvas
                            required_gain = float(np.clip(target_peak_graph - current_peak, -30.0, 65.0))

                            if not np.isfinite(self._agc_gain_db):
                                self._agc_gain_db = 38.0
                            self._agc_gain_db = 0.90 * self._agc_gain_db + 0.10 * required_gain

                            normalized_db = np.clip(mags_db + self._agc_gain_db, self._FLOOR_DB, 18.0).astype(np.float32)
                            if not np.all(np.isfinite(normalized_db)):
                                normalized_db = np.nan_to_num(
                                    normalized_db, nan=self._FLOOR_DB, posinf=18.0, neginf=self._FLOOR_DB
                                )

                            # Attack / Release smoothing governed by active speed preset
                            attack_a = self._attack_alpha
                            release_a = self._release_alpha
                            rising = normalized_db > self._smoothed_mags
                            self._smoothed_mags[rising] = (
                                attack_a * normalized_db[rising] + (1.0 - attack_a) * self._smoothed_mags[rising]
                            )
                            self._smoothed_mags[~rising] = (
                                release_a * normalized_db[~rising] + (1.0 - release_a) * self._smoothed_mags[~rising]
                            )
                            self._at_floor = False
                            emit_copy = self._smoothed_mags.copy()
                            peak_copy = self._update_peak_hold_locked(dt)
                        else:
                            if not self._at_floor:
                                # Smooth graceful decay during silence towards bottom floor (-20 dB)
                                np.maximum(self._FLOOR_DB, self._smoothed_mags - 0.75, out=self._smoothed_mags)
                                emit_copy = self._smoothed_mags.copy()
                                if bool(np.all(self._smoothed_mags <= self._FLOOR_DB)):
                                    self._at_floor = True
                            peak_copy = self._update_peak_hold_locked(dt)

                if emit_copy is not None:
                    self.sigSpectrumReady.emit(self.target_freqs, emit_copy)
                if peak_copy is not None:
                    self.sigPeakHoldReady.emit(self.target_freqs, peak_copy)
                last_emit_time = now

    def _run_fallback_decay_loop(self, timeout_sec: float = 2.0) -> None:
        start_t = time.perf_counter()
        last_t = start_t
        while not self._stop_event.is_set() and not self._restart_event.is_set():
            self.msleep(40)
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            emit_copy: Optional[np.ndarray] = None
            peak_copy: Optional[np.ndarray] = None
            with self._state_lock:
                if self._enabled and not self._frozen:
                    if not self._at_floor:
                        np.maximum(self._FLOOR_DB, self._smoothed_mags - 0.75, out=self._smoothed_mags)
                        emit_copy = self._smoothed_mags.copy()
                        if bool(np.all(self._smoothed_mags <= self._FLOOR_DB)):
                            self._at_floor = True
                    peak_copy = self._update_peak_hold_locked(dt)
            if emit_copy is not None:
                self.sigSpectrumReady.emit(self.target_freqs, emit_copy)
            if peak_copy is not None:
                self.sigPeakHoldReady.emit(self.target_freqs, peak_copy)
            if timeout_sec > 0 and (now - start_t >= timeout_sec):
                break
