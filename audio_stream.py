"""
Background audio capture worker for Windows WASAPI Loopback and real-time FFT spectrum analysis.
Includes full-frequency pink-noise tilt compensation and dynamic auto-gain normalization.
"""

from __future__ import annotations
import sys
import time
import queue
import threading
from typing import Optional, Tuple, List, Any
import numpy as np
import scipy.sparse as sp
from PyQt6 import QtCore
from presets import clean_apo_device_name

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
    computing 2048-sample Hanning FFTs with +4.0 dB/octave tilt compensation,
    dynamic auto-gain normalization, and log-frequency smoothing.
    """
    sigSpectrumReady = QtCore.pyqtSignal(np.ndarray, np.ndarray)  # (freqs, mags_db)
    sigStatusChanged = QtCore.pyqtSignal(str)                     # Status message

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
        
        self.sample_rate = 48000
        self._rta_matrix = self._build_rta_matrix(self.sample_rate)

        self._smoothed_mags = np.full(target_points, -20.0, dtype=np.float32)
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()
        self._enabled = True
        self._device_index: Optional[int] = None
        self._device_name: Optional[str] = None
        
        # Dynamic AGC normalization gain
        self._agc_gain_db = 38.0
        
        self._audio_queue: queue.Queue = queue.Queue(maxsize=32)

    def _build_rta_matrix(self, sample_rate: int) -> sp.csr_matrix:
        """
        Builds a sparse 1/6th-octave logarithmic filterbank matrix mapping FFT power
        bins to log-spaced frequencies. Produces a flat horizontal line for pink noise
        and eliminates raw treble bin grass while executing in <0.05 ms.
        """
        fft_freqs = np.fft.rfftfreq(self.window_size, 1.0 / sample_rate)
        sigma = 1.0 / 12.0  # 1/6th octave bandwidth
        log_fft = np.log2(np.maximum(10.0, fft_freqs))
        log_targets = np.log2(self.target_freqs)

        rows, cols, data = [], [], []
        for i, log_fc in enumerate(log_targets):
            diff = log_fft - log_fc
            mask = (np.abs(diff) < 3.0 * sigma) & (fft_freqs >= 15.0)
            if np.any(mask):
                w = np.exp(-0.5 * (diff[mask] / sigma) ** 2)
                # Power in fractional octave is proportional to center frequency
                w = (w / np.sum(w)) * self.target_freqs[i]
                col_indices = np.where(mask)[0]
                for c, weight in zip(col_indices, w):
                    rows.append(i)
                    cols.append(c)
                    data.append(weight)
            else:
                closest = int(np.argmin(np.abs(fft_freqs - self.target_freqs[i])))
                rows.append(i)
                cols.append(closest)
                data.append(self.target_freqs[i])

        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.target_freqs), len(fft_freqs)),
            dtype=np.float32,
        )

    def set_enabled(self, enabled: bool) -> None:
        """Enable or bypass the spectrum analyzer stream."""
        self._enabled = enabled
        if not enabled:
            self._smoothed_mags.fill(-20.0)
            self.sigSpectrumReady.emit(self.target_freqs, self._smoothed_mags.copy())

    def set_device_index(self, device_index: Optional[int]) -> None:
        """Switches the active audio loopback capture device by index."""
        if self._device_index != device_index:
            self._device_index = device_index
            self._restart_event.set()

    def set_device_target(self, device_index: Optional[int], device_name: Optional[str] = None) -> None:
        """Switches the active audio loopback capture device by index and/or friendly name."""
        if self._device_index != device_index or self._device_name != device_name:
            self._device_index = device_index
            self._device_name = device_name
            self._restart_event.set()

    def restart_stream(self) -> None:
        """Triggers a clean restart of the audio loopback stream and device re-enumeration."""
        self._restart_event.set()

    def stop(self) -> None:
        """Stops the worker thread cleanly."""
        self._stop_event.set()
        if not self.wait(1000):
            print("[AudioStream] Warning: worker thread did not stop within 1000ms.")

    def run(self) -> None:
        self._stop_event.clear()

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
                self._run_fallback_decay_loop()

            if not self._restart_event.is_set():
                break

    # -------------------------------------------------------------------------
    # PyAudioWPatch Loopback Stream
    # -------------------------------------------------------------------------
    def _run_pyaudiowpatch_stream(self) -> None:
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            loopback_device = None

            # 1. Target matching by device friendly name (highest resilience to index shifts)
            if self._device_name and "auto-follow" not in self._device_name.lower() and "default" not in self._device_name.lower():
                clean_target = clean_apo_device_name(self._device_name).lower()
                try:
                    for dev in pa.get_loopback_device_info_generator():
                        d_name = dev["name"].replace(" [Loopback]", "").strip()
                        clean_d = clean_apo_device_name(d_name).lower()
                        if clean_target == clean_d or clean_target in clean_d or clean_d in clean_target:
                            loopback_device = dev
                            break
                except Exception as e:
                    print(f"[AudioStream] Name lookup notice: {e}")

            # 2. Target lookup by device index
            if not loopback_device and self._device_index is not None:
                try:
                    dev_info = pa.get_device_info_by_index(self._device_index)
                    if dev_info.get("isLoopbackDevice", False) or dev_info.get("maxInputChannels", 0) > 0:
                        loopback_device = dev_info
                    else:
                        loopback_device = pa.get_wasapi_loopback_analogue_by_index(self._device_index)
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
                self._run_fallback_decay_loop(timeout_sec=2.0)
                return

            new_rate = int(loopback_device["defaultSampleRate"])
            if new_rate != self.sample_rate:
                self.sample_rate = new_rate
                self._rta_matrix = self._build_rta_matrix(self.sample_rate)
            channels = int(loopback_device["maxInputChannels"])
            dev_name = loopback_device["name"].replace(" [Loopback]", "")
            self.sigStatusChanged.emit(f"Capturing: {dev_name}")

            def callback(in_data, frame_count, time_info, status):
                if not self._enabled:
                    return (None, pyaudio.paContinue)
                try:
                    arr = np.frombuffer(in_data, dtype=np.float32)
                    if channels > 1:
                        arr = arr.reshape(-1, channels).mean(axis=1)
                    if not self._audio_queue.full():
                        self._audio_queue.put_nowait(arr)
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

            self._process_audio_stream(stream=stream)

        except Exception as e:
            self.sigStatusChanged.emit(f"Audio error: {e}")
            self._run_fallback_decay_loop(timeout_sec=2.0)
        finally:
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

    # -------------------------------------------------------------------------
    # sounddevice Fallback Stream
    # -------------------------------------------------------------------------
    def _run_sounddevice_stream(self) -> None:
        try:
            dev_idx = self._device_index if self._device_index is not None else sd.default.device[0]
            dev_info = sd.query_devices(dev_idx)
            new_rate = int(dev_info["default_samplerate"])
            if self._rta_matrix is None or new_rate != self.sample_rate:
                self.sample_rate = new_rate
                self._rta_matrix = self._build_rta_matrix(self.sample_rate)
            channels = min(2, max(1, dev_info["max_input_channels"]))
            self.sigStatusChanged.emit(f"Capturing: {dev_info['name']}")

            def sd_callback(indata, frames, time_info, status):
                if not self._enabled:
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
            ):
                self._process_audio_stream()

        except Exception as e:
            self.sigStatusChanged.emit(f"sounddevice error: {e}")
            self._run_fallback_decay_loop()

    # -------------------------------------------------------------------------
    # FFT DSP Loop with Tilt & Dynamic Normalization
    # -------------------------------------------------------------------------
    def _process_audio_stream(self, stream: Optional[Any] = None) -> None:
        fft_freqs = np.fft.rfftfreq(self.window_size, 1.0 / self.sample_rate)
        fft_freqs[0] = 1.0
        
        last_emit_time = time.perf_counter()
        last_audio_activity_time = 0.0
        buffer = np.zeros(self.window_size, dtype=np.float32)

        while not self._stop_event.is_set() and not self._restart_event.is_set():
            # Check if underlying hardware stream stopped or was unplugged
            if stream is not None:
                try:
                    if hasattr(stream, "is_active") and not stream.is_active():
                        print("[AudioStream] Hardware stream stopped or endpoint disconnected.")
                        break
                except Exception:
                    break

            # 1. Fetch available chunk with small timeout
            try:
                chunk = self._audio_queue.get(timeout=0.015)
                if len(chunk) > 0:
                    if len(chunk) >= self.window_size:
                        buffer[:] = chunk[-self.window_size:]
                    else:
                        n = len(chunk)
                        buffer[:-n] = buffer[n:]
                        buffer[-n:] = chunk
                    if np.sqrt(np.mean(chunk**2)) > 1e-5:
                        last_audio_activity_time = time.perf_counter()
            except queue.Empty:
                pass

            # Drain any accumulated chunks so audio visualization stays strictly in real-time
            while not self._audio_queue.empty():
                try:
                    chunk = self._audio_queue.get_nowait()
                    if len(chunk) > 0:
                        if len(chunk) >= self.window_size:
                            buffer[:] = chunk[-self.window_size:]
                        else:
                            n = len(chunk)
                            buffer[:-n] = buffer[n:]
                            buffer[-n:] = chunk
                        if np.sqrt(np.mean(chunk**2)) > 1e-5:
                            last_audio_activity_time = time.perf_counter()
                except queue.Empty:
                    break

            now = time.perf_counter()
            if now - last_emit_time >= 0.025:
                if self._enabled:
                    # Active playback state: an audible chunk was received in the last 250ms
                    is_active = (now - last_audio_activity_time) < 0.25

                    if is_active:
                        win_buf = buffer * self.hanning_window
                        X = np.fft.rfft(win_buf)
                        power = ((2.0 / self.window_size) * np.abs(X)) ** 2
                        binned_power = self._rta_matrix.dot(power)
                        mags_db = 10.0 * np.log10(np.maximum(1e-9, binned_power))

                        # Dynamic Auto-Gain Normalization (AGC):
                        current_peak = float(np.max(mags_db))
                        target_peak_graph = -3.0  # Center peak comfortably around -3 dB on [-20, +20] dB canvas
                        required_gain = target_peak_graph - current_peak
                        required_gain = float(np.clip(required_gain, 10.0, 60.0))

                        # Smooth AGC gain adjustment
                        self._agc_gain_db = 0.90 * self._agc_gain_db + 0.10 * required_gain

                        # Final calibrated response
                        normalized_db = np.clip(mags_db + self._agc_gain_db, -20.0, 18.0)

                        # Attack / Decay smoothing
                        rising = normalized_db > self._smoothed_mags
                        self._smoothed_mags[rising] = (
                            0.60 * normalized_db[rising] + 0.40 * self._smoothed_mags[rising]
                        )
                        self._smoothed_mags[~rising] = (
                            0.15 * normalized_db[~rising] + 0.85 * self._smoothed_mags[~rising]
                        )
                    else:
                        # Smooth graceful decay during silence towards bottom floor (-20 dB)
                        self._smoothed_mags = np.maximum(-20.0, self._smoothed_mags - 0.75)

                    self.sigSpectrumReady.emit(self.target_freqs, self._smoothed_mags.copy())
                last_emit_time = now

    def _run_fallback_decay_loop(self, timeout_sec: float = 2.0) -> None:
        start_t = time.perf_counter()
        while not self._stop_event.is_set() and not self._restart_event.is_set():
            self.msleep(40)
            if self._enabled:
                self._smoothed_mags = np.maximum(-20.0, self._smoothed_mags - 0.75)
                self.sigSpectrumReady.emit(self.target_freqs, self._smoothed_mags.copy())
            if timeout_sec > 0 and (time.perf_counter() - start_t >= timeout_sec):
                break
