"""
Hearing & Headphone Test Suite for Clean APO.
Interactive audiometric threshold-in-noise measurement and EQ curve generator.
Inspired by Peace Equalizer APO Headphones and Hearing Test.

Features:
- Real-time pure sine wave synthesis with phase continuity and zero-click exponential gain ramping.
- Background reference noise generator (Gaussian white noise & 1/f pink noise).
- Pulsed tone mode (raised-cosine windowed beeps) to avoid auditory adaptation.
- Channel routing: Both Ears (Stereo), Left Ear, Right Ear.
- ISO 226:2003 Equal-Loudness contour normalization.
- Automatic anti-clipping preamp computation.
- Seamless integration with Clean APO filter table, pyqtgraph plot, and Equalizer APO presets.
"""

from __future__ import annotations
import math
import time
import threading
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import lfilter
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from biquad import EQBand, FilterType, compute_composite_response, compute_peak_headroom
from presets import clean_apo_device_name

HAVE_SOUNDDEVICE = False
try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except ImportError:
    pass


# -----------------------------------------------------------------------------
# ISO 226:2003 Equal-Loudness Contours Implementation
# -----------------------------------------------------------------------------
class ISO226:
    """
    Computes ISO 226:2003 Normal Equal-Loudness Contours across arbitrary frequencies.
    """
    FREQUENCIES = np.array([
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
        200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
        1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
        10000.0, 12500.0
    ], dtype=np.float64)

    ALPHA = np.array([
        0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330,
        0.315, 0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246,
        0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301
    ], dtype=np.float64)

    LU = np.array([
        -31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5,
        -3.1, -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7,
        -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1
    ], dtype=np.float64)

    TF = np.array([
        78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9,
        14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5,
        1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3
    ], dtype=np.float64)

    _interp_alpha = PchipInterpolator(np.log10(FREQUENCIES), ALPHA)
    _interp_lu = PchipInterpolator(np.log10(FREQUENCIES), LU)
    _interp_tf = PchipInterpolator(np.log10(FREQUENCIES), TF)

    @classmethod
    def evaluate(cls, frequencies: np.ndarray | List[float], phon: float = 60.0) -> np.ndarray:
        """
        Calculates Sound Pressure Level (dB SPL) required for target phon level.
        Clamped to valid ISO 226:2003 bounds (20 Hz - 12.5 kHz) to prevent unstable extrapolation.
        """
        f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        log_f = np.log10(np.clip(f, 20.0, 12500.0))

        af = cls._interp_alpha(log_f)
        lu = cls._interp_lu(log_f)
        tf = cls._interp_tf(log_f)

        phon = float(np.clip(phon, 0.0, 90.0))
        A = 0.00447 * (10.0 ** (0.025 * phon) - 1.15) + (0.4 * 10.0 ** (((tf + lu) / 10.0) - 9.0)) ** af
        A = np.maximum(A, 1e-12)

        lp = (10.0 / af) * np.log10(A) - lu + 94.0
        return lp

    @classmethod
    def normalized_delta(cls, frequencies: np.ndarray | List[float], phon: float = 60.0) -> np.ndarray:
        """
        Returns the relative ISO-226 deviation relative to 1,000 Hz.
        Positive value means human ear is less sensitive than at 1 kHz (e.g. low bass).
        """
        lp = cls.evaluate(frequencies, phon=phon)
        ref_lp = float(cls.evaluate([1000.0], phon=phon)[0])
        return lp - ref_lp


# -----------------------------------------------------------------------------
# High-Performance Real-Time Audio Synthesis Engine
# -----------------------------------------------------------------------------
class HearingAudioEngine:
    """
    Low-latency audio synthesis worker using sounddevice.OutputStream.
    Generates pure sine tones and reference white/pink noise with per-sample
    clickless exponential parameter smoothing and pulsed beep modulation.
    """
    _PINK_POLES = (
        (0.0555179, 0.99886),
        (0.0750759, 0.99332),
        (0.1538520, 0.96900),
        (0.3104856, 0.86650),
        (0.5329522, 0.55000),
        (-0.0168980, -0.7616),
    )
    _PINK_RMS_SCALE = 0.326  # RMS-matched to uniform(-1, 1) white noise

    def __init__(
        self,
        sample_rate: int = 48000,
        device_index: Optional[int] = None,
        device_name: Optional[str] = None,
    ):
        self.sample_rate = max(8000, int(sample_rate))
        self.device_index = device_index
        self.device_name = device_name
        self._output_channels: int = 2
        self._stream: Optional[Any] = None
        self.audio_error: Optional[str] = None

        # Thread synchronization lock
        self._lock = threading.Lock()

        # Engine parameters
        self._freq: float = 1000.0
        self._target_sine_gain: float = 10.0 ** (-30.0 / 20.0)
        self._current_sine_gain: float = 0.0
        self._target_noise_gain: float = 10.0 ** (-35.0 / 20.0)
        self._current_noise_gain: float = 0.0

        self._sine_enabled: bool = True
        self._noise_enabled: bool = True
        self._noise_type: str = "white"  # "white" or "pink"
        self._tone_mode: str = "pulsed"  # "pulsed" or "continuous"
        self._channel: str = "both"      # "both", "left", "right"

        # Synthesis state
        self._phase: float = 0.0
        self._pulse_time: float = 0.0
        self._pulse_period: float = 0.50  # 250ms ON, 250ms OFF
        self._pulse_ramp: float = 0.025   # 25ms raised-cosine ramp

        # Vectorized Paul Kellet 7-pole pink noise IIR filter states
        self._pink_zi: List[np.ndarray] = [np.zeros(1, dtype=np.float64) for _ in range(7)]

        self._running: bool = False
        self._start_stream()

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _hostapi_rank(hostapis: Any, hostapi_idx: Any) -> int:
        try:
            api_name = str(hostapis[int(hostapi_idx)].get("name", "")).lower()
            if "wasapi" in api_name:
                return 3
            if "directsound" in api_name:
                return 2
            if "mme" in api_name:
                return 1
        except Exception:
            pass
        return 0

    @classmethod
    def list_output_devices(cls) -> List[Tuple[Optional[int], str]]:
        """
        Lists available sounddevice playback output devices as (device_index, display_name).
        Always begins with (None, "System default") and deduplicates physical devices
        by preferring WASAPI > DirectSound > MME.
        """
        devices: List[Tuple[Optional[int], str]] = [(None, "System default")]
        if not HAVE_SOUNDDEVICE:
            return devices

        try:
            all_devs = sd.query_devices()
            if isinstance(all_devs, dict):
                all_devs = [all_devs]
            try:
                hostapis = sd.query_hostapis()
            except Exception:
                hostapis = []

            best_by_name: Dict[str, Tuple[int, int, str]] = {}
            order: List[str] = []
            for idx, dev in enumerate(all_devs):
                if not isinstance(dev, dict):
                    continue
                if int(dev.get("max_output_channels", 0)) < 1:
                    continue
                raw_name = str(dev.get("name", "")).strip()
                if not raw_name:
                    continue
                lower_raw = raw_name.lower()
                if lower_raw.startswith("microsoft sound mapper") or lower_raw.startswith("primary sound driver"):
                    continue
                key = clean_apo_device_name(raw_name).lower() or lower_raw
                rank = cls._hostapi_rank(hostapis, dev.get("hostapi", -1))
                prev = best_by_name.get(key)
                if prev is None:
                    best_by_name[key] = (rank, idx, raw_name)
                    order.append(key)
                elif rank > prev[0]:
                    best_by_name[key] = (rank, idx, raw_name)

            for key in order:
                _, idx, display_name = best_by_name[key]
                devices.append((idx, display_name))
        except Exception as e:
            print(f"[HearingAudioEngine] Output device listing failed: {e}")

        return devices

    def _resolve_output_device(self) -> Optional[int]:
        """
        Resolves a valid sounddevice output device index.
        Prefers matching device_name (favoring WASAPI), then validates device_index
        (requiring max_output_channels >= 1), and falls back to None (system default output).
        """
        if not HAVE_SOUNDDEVICE:
            return None

        # 1. Match by friendly device name if provided
        if self.device_name and "auto-follow" not in self.device_name.lower():
            clean_target = clean_apo_device_name(self.device_name).lower()
            if clean_target and clean_target != "all":
                try:
                    all_devs = sd.query_devices()
                    if isinstance(all_devs, dict):
                        all_devs = [all_devs]
                    try:
                        hostapis = sd.query_hostapis()
                    except Exception:
                        hostapis = []

                    best_exact: Optional[Tuple[int, int]] = None
                    best_partial: Optional[Tuple[int, int]] = None
                    for idx, dev in enumerate(all_devs):
                        if int(dev.get("max_output_channels", 0)) < 1:
                            continue
                        clean_d = clean_apo_device_name(str(dev.get("name", ""))).lower()
                        if not clean_d or clean_d == "all":
                            continue
                        rank = self._hostapi_rank(hostapis, dev.get("hostapi", -1))
                        if clean_d == clean_target:
                            if best_exact is None or rank > best_exact[0]:
                                best_exact = (rank, idx)
                        elif clean_target in clean_d or clean_d in clean_target:
                            if best_partial is None or rank > best_partial[0]:
                                best_partial = (rank, idx)

                    if best_exact is not None:
                        return best_exact[1]
                    if best_partial is not None:
                        return best_partial[1]
                except Exception as e:
                    print(f"[HearingAudioEngine] Output device name lookup failed: {e}")

        # 2. Validate explicit device_index as an actual output device
        if self.device_index is not None:
            try:
                info = sd.query_devices(self.device_index)
                if int(info.get("max_output_channels", 0)) >= 1:
                    return int(self.device_index)
            except Exception:
                pass

        # 3. Default system output
        return None

    def _start_stream(self) -> None:
        self.audio_error = None
        if not HAVE_SOUNDDEVICE:
            self.audio_error = "Audio playback unavailable: sounddevice is not installed."
            self._running = False
            return

        resolved_dev = self._resolve_output_device()
        candidates: List[Optional[int]] = [resolved_dev]
        if resolved_dev is not None:
            candidates.append(None)  # Always fall back to default system output if specific device fails

        last_err: Optional[Exception] = None
        for dev in candidates:
            # Probe device default sample rate & output channels
            dev_rate = 0
            out_channels = 2
            try:
                info = sd.query_devices(dev, kind="output") if dev is None else sd.query_devices(dev)
                if isinstance(info, dict):
                    max_out = int(info.get("max_output_channels", 2))
                    if max_out < 1:
                        continue
                    out_channels = min(2, max_out)
                    dev_rate = int(info.get("default_samplerate", 0) or 0)
            except Exception:
                out_channels = 2
                dev_rate = 0

            rate_candidates: List[int] = []
            for r in (dev_rate, self.sample_rate, 48000, 44100):
                if r and r >= 8000 and r not in rate_candidates:
                    rate_candidates.append(int(r))

            for sr in rate_candidates:
                stream = None
                try:
                    kwargs: Dict[str, Any] = {
                        "samplerate": sr,
                        "channels": out_channels,
                        "dtype": "float32",
                        "callback": self._audio_callback,
                        "blocksize": 1024,
                    }
                    if dev is not None:
                        kwargs["device"] = dev
                    stream = sd.OutputStream(**kwargs)
                    stream.start()
                    self._stream = stream
                    self.sample_rate = sr
                    self._output_channels = out_channels
                    self._running = True
                    self.audio_error = None
                    with self._lock:
                        max_freq = min(20000.0, 0.48 * float(self.sample_rate))
                        self._freq = float(np.clip(self._freq, 20.0, max(20.0, max_freq)))
                    return
                except Exception as e:
                    last_err = e
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception:
                            pass

        self.audio_error = f"Audio output could not be started ({last_err}). Check Windows playback device."
        print(f"[HearingAudioEngine] Failed to start OutputStream: {last_err}")
        self._stream = None
        self._running = False

    def _generate_pink_noise(self, white_f64: np.ndarray) -> np.ndarray:
        zi_list = getattr(self, "_pink_zi", None)
        if zi_list is None or len(zi_list) != 7:
            zi_list = [np.zeros(1, dtype=np.float64) for _ in range(7)]
            self._pink_zi = zi_list

        pink = white_f64 * 0.5362
        for idx, (b_coeff, a_coeff) in enumerate(self._PINK_POLES):
            y, zi_next = lfilter([b_coeff], [1.0, -a_coeff], white_f64, zi=zi_list[idx])
            zi_list[idx] = zi_next
            pink += y
        y6, zi6_next = lfilter([0.0, 0.115926], [1.0], white_f64, zi=zi_list[6])
        zi_list[6] = zi6_next
        pink += y6
        return (pink * self._PINK_RMS_SCALE).astype(np.float32)

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if frames <= 0:
            return
        with self._lock:
            freq = self._freq
            target_sg = self._target_sine_gain if self._sine_enabled else 0.0
            target_ng = self._target_noise_gain if self._noise_enabled else 0.0
            noise_type = self._noise_type
            tone_mode = self._tone_mode
            channel = self._channel
            phase = self._phase
            pulse_t = self._pulse_time

        # 1. Per-sample exponential gain ramping (15ms time constant, zero block clicks)
        tau_samples = max(1.0, float(self.sample_rate) * 0.015)
        decay = np.exp(-np.arange(1, frames + 1, dtype=np.float64) / tau_samples)
        prev_sg = float(self._current_sine_gain)
        prev_ng = float(self._current_noise_gain)
        sg_env = (target_sg + (prev_sg - target_sg) * decay).astype(np.float32)
        ng_env = (target_ng + (prev_ng - target_ng) * decay).astype(np.float32)
        self._current_sine_gain = float(sg_env[-1])
        self._current_noise_gain = float(ng_env[-1])

        # 2. Pure Sine Tone Generation (Phase-Continuous, float64 phase precision)
        t = np.arange(frames, dtype=np.float64) / float(self.sample_rate)
        omega = 2.0 * math.pi * freq
        sine_wave = np.sin(phase + omega * t).astype(np.float32)
        new_phase = (phase + omega * frames / float(self.sample_rate)) % (2.0 * math.pi)

        # 3. Tone Modulation (Pulsed Beep vs Continuous)
        if tone_mode == "pulsed":
            sample_times = (pulse_t + t) % self._pulse_period
            on_duration = self._pulse_period * 0.5
            ramp_duration = self._pulse_ramp

            envelope = np.zeros(frames, dtype=np.float32)
            on_mask = sample_times < on_duration

            fade_in_mask = on_mask & (sample_times < ramp_duration)
            envelope[fade_in_mask] = 0.5 * (1.0 - np.cos(math.pi * sample_times[fade_in_mask] / ramp_duration))

            sustain_mask = on_mask & (sample_times >= ramp_duration) & (sample_times <= (on_duration - ramp_duration))
            envelope[sustain_mask] = 1.0

            fade_out_mask = on_mask & (sample_times > (on_duration - ramp_duration))
            ramp_out_t = (on_duration - sample_times[fade_out_mask]) / ramp_duration
            envelope[fade_out_mask] = 0.5 * (1.0 - np.cos(math.pi * ramp_out_t))

            sine_wave *= envelope
            new_pulse_t = (pulse_t + frames / float(self.sample_rate)) % self._pulse_period
        else:
            new_pulse_t = 0.0

        with self._lock:
            self._phase = new_phase
            self._pulse_time = new_pulse_t

        sine_signal = sine_wave * sg_env

        # 4. Reference Noise Generation (White or Vectorized 7-Pole Pink)
        if max(prev_ng, target_ng) > 1e-6:
            raw_noise_f64 = np.random.uniform(-1.0, 1.0, size=frames)
            if noise_type == "pink":
                noise_signal = self._generate_pink_noise(raw_noise_f64) * ng_env
            else:
                noise_signal = raw_noise_f64.astype(np.float32) * ng_env
        else:
            noise_signal = np.zeros(frames, dtype=np.float32)

        # 5. Summation and Channel Routing (handles mono & stereo outputs safely)
        mixed = sine_signal + noise_signal

        outdata.fill(0.0)
        if outdata.ndim == 1:
            outdata[:] = mixed
        elif outdata.shape[1] == 1:
            outdata[:, 0] = mixed
        elif channel == "left":
            outdata[:, 0] = mixed
        elif channel == "right":
            outdata[:, 1] = mixed
        else:  # "both" (stereo)
            outdata[:, 0] = mixed
            outdata[:, 1] = mixed

        # 6. Safety Digital Limiter
        np.clip(outdata, -0.98, 0.98, out=outdata)

    def set_frequency(self, freq: float) -> None:
        max_freq = min(20000.0, 0.48 * float(self.sample_rate))
        with self._lock:
            self._freq = float(np.clip(freq, 20.0, max(20.0, max_freq)))

    def set_sine_volume_db(self, vol_db: float) -> None:
        vol_db = float(np.clip(vol_db, -90.0, 0.0))
        linear_gain = 10.0 ** (vol_db / 20.0)
        with self._lock:
            self._target_sine_gain = linear_gain

    def set_noise_volume_db(self, vol_db: float) -> None:
        vol_db = float(np.clip(vol_db, -70.0, 0.0))
        linear_gain = 10.0 ** (vol_db / 20.0)
        with self._lock:
            self._target_noise_gain = linear_gain

    def set_sine_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._sine_enabled = bool(enabled)

    def set_noise_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._noise_enabled = bool(enabled)

    def set_tone_mode(self, mode: str) -> None:
        with self._lock:
            self._tone_mode = "pulsed" if mode == "pulsed" else "continuous"

    def set_noise_type(self, ntype: str) -> None:
        with self._lock:
            self._noise_type = "pink" if ntype == "pink" else "white"

    def set_channel(self, channel: str) -> None:
        with self._lock:
            self._channel = channel.lower()

    def set_device(self, device_index: Optional[int], device_name: Optional[str] = None) -> None:
        self.stop()
        with self._lock:
            self._current_sine_gain = 0.0
            self._current_noise_gain = 0.0
            self._phase = 0.0
            self._pulse_time = 0.0
            self._pink_zi = [np.zeros(1, dtype=np.float64) for _ in range(7)]
        self.device_index = device_index
        self.device_name = device_name
        self._start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._running = False


# -----------------------------------------------------------------------------
# Hearing Test Session & Calibration Data Model
# -----------------------------------------------------------------------------
STANDARD_10_BANDS = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
DETAILED_15_BANDS = [25.0, 40.0, 63.0, 100.0, 160.0, 250.0, 500.0, 1000.0, 2000.0, 3150.0, 4000.0, 6300.0, 8000.0, 12500.0, 16000.0]
FULL_31_BANDS = [
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
    1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0,
    10000.0, 12500.0, 16000.0, 20000.0
]


def _default_q_for_suite(freqs: List[float]) -> float:
    """Returns appropriate peaking Q for the active band density to avoid overlapping summation."""
    n = len(freqs)
    if n >= 25:
        return 4.318  # Standard 1/3-octave graphic EQ Q
    if n >= 13:
        return 2.100  # 15-band (~2/3-octave) Q
    return 1.414      # Standard 1-octave Q


def _anchor_index_1khz(freqs: List[float]) -> int:
    """Returns the index nearest to 1,000 Hz (in log-frequency space) for safe initial calibration."""
    if not freqs:
        return 0
    return min(
        range(len(freqs)),
        key=lambda i: abs(math.log2(max(float(freqs[i]), 1.0) / 1000.0)),
    )


@dataclass
class HearingTestSession:
    frequencies: List[float] = field(default_factory=lambda: list(STANDARD_10_BANDS))
    current_index: int = 0
    channel: str = "both"  # "both", "left", "right"

    # Measured threshold levels in dB per channel
    thresholds: Dict[str, Dict[float, float]] = field(default_factory=lambda: {
        "both": {},
        "left": {},
        "right": {},
    })
    tested: Dict[str, Dict[float, bool]] = field(default_factory=lambda: {
        "both": {},
        "left": {},
        "right": {},
    })

    # EQ Generation Parameters
    compensation_mode: str = "direct"    # "direct" or "iso226"
    export_channel_mode: str = "stereo"  # "stereo" (Avg) or "per_ear" (L/R)
    strength: float = 1.0                # 0.0 to 1.0 (0% to 100%)
    max_boost_db: float = 10.0           # Maximum boost clamp (+3 to +18 dB)
    q_factor: float = 1.414              # Peaking Q (auto-scaled per suite)

    def __post_init__(self):
        self.init_frequency_suite(self.frequencies)

    def init_frequency_suite(self, freqs: List[float]) -> None:
        self.frequencies = sorted(list(set(float(f) for f in freqs)))
        self.current_index = _anchor_index_1khz(self.frequencies)
        self.q_factor = _default_q_for_suite(self.frequencies)
        for ch in ["both", "left", "right"]:
            for f in self.frequencies:
                if f not in self.thresholds[ch]:
                    self.thresholds[ch][f] = -30.0  # Safe initial default
                    self.tested[ch][f] = False

    @property
    def current_freq(self) -> float:
        if 0 <= self.current_index < len(self.frequencies):
            return self.frequencies[self.current_index]
        return 1000.0

    def _resolve_channel(self, ch: Optional[str]) -> str:
        resolved = self.channel if ch is None else ch
        return resolved if resolved in self.thresholds else "both"

    def get_threshold(self, ch: Optional[str] = None, freq: Optional[float] = None) -> float:
        resolved_ch = self._resolve_channel(ch)
        resolved_f = self.current_freq if freq is None else float(freq)
        return self.thresholds[resolved_ch].get(resolved_f, -30.0)

    def set_threshold(self, value_db: float, ch: Optional[str] = None, freq: Optional[float] = None) -> None:
        resolved_ch = self._resolve_channel(ch)
        resolved_f = self.current_freq if freq is None else float(freq)
        val = float(np.clip(value_db, -90.0, 0.0))
        self.thresholds[resolved_ch][resolved_f] = val
        self.tested[resolved_ch][resolved_f] = True

    def is_tested(self, ch: Optional[str] = None, freq: Optional[float] = None) -> bool:
        resolved_ch = self._resolve_channel(ch)
        resolved_f = self.current_freq if freq is None else float(freq)
        return self.tested[resolved_ch].get(resolved_f, False)

    def has_independent_lr_data(self) -> bool:
        """Returns True when both Left and Right ears have at least one explicitly recorded threshold."""
        has_l = any(self.tested["left"].get(f, False) for f in self.frequencies)
        has_r = any(self.tested["right"].get(f, False) for f in self.frequencies)
        return has_l and has_r

    def next_frequency(self) -> bool:
        if self.current_index < len(self.frequencies) - 1:
            self.current_index += 1
            return True
        return False

    def prev_frequency(self) -> bool:
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def reset_all(self) -> None:
        for ch in ["both", "left", "right"]:
            self.thresholds[ch].clear()
            self.tested[ch].clear()
            for f in self.frequencies:
                self.thresholds[ch][f] = -30.0
                self.tested[ch][f] = False
        self.current_index = _anchor_index_1khz(self.frequencies)
        self.export_channel_mode = "stereo"

    def get_effective_thresholds_and_baseline(
        self, target_channel: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Resolves effective thresholds, tested mask, and reference baseline for target_channel.
        In 'both' mode, falls back to tested L/R measurements only when explicitly tested.
        """
        ch = self._resolve_channel(target_channel)
        freqs = np.array(self.frequencies, dtype=np.float64)
        if freqs.size == 0:
            return freqs, np.array([], dtype=np.float64), np.array([], dtype=bool), -30.0

        raw_thresholds: List[float] = []
        tested_flags: List[bool] = []
        for f in freqs:
            f_key = float(f)
            if ch == "both":
                if self.tested["both"].get(f_key, False):
                    raw_thresholds.append(self.thresholds["both"].get(f_key, -30.0))
                    tested_flags.append(True)
                else:
                    l_tested = self.tested["left"].get(f_key, False)
                    r_tested = self.tested["right"].get(f_key, False)
                    if l_tested and r_tested:
                        raw_thresholds.append(
                            0.5 * (self.thresholds["left"][f_key] + self.thresholds["right"][f_key])
                        )
                        tested_flags.append(True)
                    elif l_tested:
                        raw_thresholds.append(self.thresholds["left"][f_key])
                        tested_flags.append(True)
                    elif r_tested:
                        raw_thresholds.append(self.thresholds["right"][f_key])
                        tested_flags.append(True)
                    else:
                        raw_thresholds.append(self.thresholds["both"].get(f_key, -30.0))
                        tested_flags.append(False)
            else:
                raw_thresholds.append(self.thresholds[ch].get(f_key, -30.0))
                tested_flags.append(bool(self.tested[ch].get(f_key, False)))

        thresh_arr = np.array(raw_thresholds, dtype=np.float64)
        tested_mask = np.array(tested_flags, dtype=bool)

        if not np.any(tested_mask):
            baseline = -30.0
        else:
            idx_1k = None
            for i, f_val in enumerate(freqs):
                if abs(f_val - 1000.0) < 1e-6:
                    idx_1k = i
                    break
            if idx_1k is not None and tested_mask[idx_1k]:
                baseline = float(thresh_arr[idx_1k])
            else:
                baseline = float(np.median(thresh_arr[tested_mask]))

        return freqs, thresh_arr, tested_mask, baseline

    def _build_channel_bands(
        self,
        target_channel: Optional[str] = None,
        band_channel: str = "ALL",
        start_band_id: int = 1,
    ) -> List[EQBand]:
        """Builds parametric EQ bands for a single channel profile ('both', 'left', or 'right')."""
        freqs, thresh_arr, tested_mask, baseline = self.get_effective_thresholds_and_baseline(target_channel)
        if freqs.size == 0:
            return []

        delta = np.zeros_like(thresh_arr)
        if np.any(tested_mask):
            delta[tested_mask] = thresh_arr[tested_mask] - baseline
            if self.compensation_mode == "iso226":
                iso_deltas = ISO226.normalized_delta(freqs, phon=60.0)
                delta[tested_mask] = delta[tested_mask] - iso_deltas[tested_mask]

        scaled_gains = delta * float(np.clip(self.strength, 0.0, 1.0))
        max_b = float(max(1.0, self.max_boost_db))
        clamped_gains = np.clip(scaled_gains, -max_b, max_b)

        bands: List[EQBand] = []
        for i, (f, g) in enumerate(zip(freqs, clamped_gains)):
            rounded_gain = float(round(g, 1))
            if abs(rounded_gain) < 1e-9:
                rounded_gain = 0.0
            bands.append(
                EQBand(
                    band_id=start_band_id + i,
                    enabled=True,
                    filter_type=FilterType.PEAK.value,
                    freq=float(round(f, 1)),
                    gain=rounded_gain,
                    q=float(self.q_factor),
                    channel=band_channel,
                )
            )
        return bands

    def compute_per_ear_compensation(self) -> Tuple[float, List[EQBand]]:
        """
        Computes independent Left (channel='L') and Right (channel='R') parametric EQ bands
        and overall worst-case anti-clipping preamp via biquad.compute_peak_headroom.
        """
        if not self.frequencies:
            return 0.0, []
        left_bands = self._build_channel_bands(target_channel="left", band_channel="L", start_band_id=1)
        right_bands = self._build_channel_bands(
            target_channel="right", band_channel="R", start_band_id=len(left_bands) + 1
        )
        bands = left_bands + right_bands
        _, preamp_db = compute_peak_headroom(bands, preamp_db=0.0, fs=48000.0)
        return preamp_db, bands

    def compute_compensation(
        self,
        target_channel: Optional[str] = None,
        per_ear: bool = False,
    ) -> Tuple[float, List[EQBand]]:
        """
        Calculates optimal parametric EQ bands and composite anti-clipping preamp
        from measured hearing thresholds.
        When per_ear=True, returns independent Left ('L') and Right ('R') channel bands.
        """
        if per_ear:
            return self.compute_per_ear_compensation()

        bands = self._build_channel_bands(target_channel=target_channel, band_channel="ALL", start_band_id=1)
        if not bands:
            return 0.0, []

        _, preamp_db = compute_peak_headroom(bands, preamp_db=0.0, fs=48000.0)
        return preamp_db, bands


# -----------------------------------------------------------------------------
# Custom UI Components & Dialog Title Bar
# -----------------------------------------------------------------------------
_UI_FONT_FAMILY = '"Segoe UI", "Inter", sans-serif'
_MONO_FONT_FAMILY = '"Cascadia Mono", "Consolas", monospace'


class _DialogCloseButton(QtWidgets.QPushButton):
    """Crisp 28x28 vector-painted close button (1.5px antialiased X stroke)."""
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Close hearing test")
        self._hovered = False
        self._pressed = False
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hovered = False
        self._pressed = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._pressed = True
            self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._pressed = False
        self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._pressed:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#1c1c1f"))
            painter.drawRoundedRect(rect, 4.0, 4.0)
        elif self._hovered:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("#27272a"))
            painter.drawRoundedRect(rect, 4.0, 4.0)

        stroke_color = QtGui.QColor("#ededed" if self._hovered else "#a1a1aa")
        pen = QtGui.QPen(stroke_color, 1.5, QtCore.Qt.PenStyle.SolidLine, QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        half = 4.5
        painter.drawLine(QtCore.QPointF(cx - half, cy - half), QtCore.QPointF(cx + half, cy + half))
        painter.drawLine(QtCore.QPointF(cx + half, cy - half), QtCore.QPointF(cx - half, cy + half))
        painter.end()


class _LegendSwatch(QtWidgets.QWidget):
    """Crisp vector legend swatch (dot, solid line, or dashed line) with zero unicode symbols."""
    def __init__(self, kind: str, color: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._kind = kind
        self._color = QtGui.QColor(color)
        if kind == "dot":
            self.setFixedSize(8, 12)
        else:
            self.setFixedSize(14, 12)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2.0
        if self._kind == "dot":
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(self._color)
            painter.drawEllipse(QtCore.QPointF(4.0, cy), 3.2, 3.2)
        else:
            style = QtCore.Qt.PenStyle.DashLine if self._kind == "dashed" else QtCore.Qt.PenStyle.SolidLine
            pen = QtGui.QPen(self._color, 1.5, style, QtCore.Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(1.0, cy), QtCore.QPointF(self.width() - 1.0, cy))
        painter.end()


class _StudioCheckBox(QtWidgets.QCheckBox):
    """QCheckBox that paints a crisp antialiased vector checkmark inside its checked indicator."""
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self.isChecked():
            return
        opt = QtWidgets.QStyleOptionButton()
        self.initStyleOption(opt)
        rect = self.style().subElementRect(QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, opt, self)
        if rect.isEmpty():
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(
            QtGui.QColor("#0b0b0c"),
            1.75,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        x = rect.left()
        y = rect.top()
        w = rect.width()
        h = rect.height()
        p1 = QtCore.QPointF(x + w * 0.24, y + h * 0.52)
        p2 = QtCore.QPointF(x + w * 0.43, y + h * 0.72)
        p3 = QtCore.QPointF(x + w * 0.77, y + h * 0.30)
        painter.drawPolyline(QtGui.QPolygonF([p1, p2, p3]))
        painter.end()


class _StudioComboBox(QtWidgets.QComboBox):
    """QComboBox with a quiet vector downward chevron so dropdowns read clearly at a glance."""
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor("#a1a1aa" if self.underMouse() else "#71717a")
        pen = QtGui.QPen(
            color,
            1.4,
            QtCore.Qt.PenStyle.SolidLine,
            QtCore.Qt.PenCapStyle.RoundCap,
            QtCore.Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(pen)
        cx = self.width() - 12.0
        cy = self.height() / 2.0 + 0.5
        painter.drawPolyline(
            QtGui.QPolygonF([
                QtCore.QPointF(cx - 3.5, cy - 2.0),
                QtCore.QPointF(cx, cy + 1.8),
                QtCore.QPointF(cx + 3.5, cy - 2.0),
            ])
        )
        painter.end()


class HearingDialogTitleBar(QtWidgets.QFrame):
    """Minimalist studio dark title bar with drag-to-move."""
    def __init__(self, dialog: QtWidgets.QDialog, title_text: str, parent=None):
        super().__init__(parent)
        self.dialog = dialog
        self.setObjectName("hearingTitleBar")
        self.setFixedHeight(40)
        self.setStyleSheet(f"""
            QFrame#hearingTitleBar {{
                background-color: #0b0b0c;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom: 1px solid #27272a;
            }}
            QFrame#hearingTitleBar QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(8)

        title_lbl = QtWidgets.QLabel(title_text)
        title_lbl.setStyleSheet("font-size: 13px; font-weight: 500; color: #ededed;")
        layout.addWidget(title_lbl)

        layout.addStretch()

        close_btn = _DialogCloseButton(self)
        close_btn.clicked.connect(self.dialog.reject)
        layout.addWidget(close_btn)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            handle = self.dialog.windowHandle()
            if handle and hasattr(handle, "startSystemMove"):
                handle.startSystemMove()
                event.accept()
                return
            else:
                self._drag_pos = event.globalPosition().toPoint() - self.dialog.frameGeometry().topLeft()
                event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if getattr(self, "_drag_pos", None) is not None and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.dialog.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)


# -----------------------------------------------------------------------------
# Interactive Hearing Test Dialog
# -----------------------------------------------------------------------------
class HearingTestDialog(QtWidgets.QDialog):
    """
    Studio-grade Hearing & Headphone Calibration Dialog.
    Maps out auditory thresholds in white/pink noise and translates results
    into Equalizer APO correction curves.
    """
    sigApplyEQ = QtCore.pyqtSignal(list, float)           # (bands: List[EQBand], preamp_db: float)
    sigSavePreset = QtCore.pyqtSignal(str, list, float)   # (name: str, bands: List[EQBand], preamp_db: float)
    sigPreviewEQ = QtCore.pyqtSignal(list, float, bool)   # (bands: List[EQBand], preamp_db: float, is_bypassed: bool)
    sigPreviewEnded = QtCore.pyqtSignal()
    sigMeasurementBypassRequested = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        device_index: Optional[int] = None,
        parent=None,
        device_name: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Hearing & headphone test - Clean APO")
        self.resize(980, 760)
        self.setMinimumSize(940, 720)

        self._initial_device_index = device_index
        self._initial_device_name = device_name

        self.session = HearingTestSession()
        self.audio_engine = HearingAudioEngine(
            sample_rate=48000,
            device_index=device_index,
            device_name=device_name,
        )

        self._ab_bypassed: bool = False
        self._has_previewed: bool = False
        self._committed_eq: bool = False
        self._centered_once: bool = False
        self._bypass_emitted_on_show: bool = False
        self._measurement_bypass_active: bool = False
        self._user_override_export_mode: bool = False
        self._shortcuts: List[QtGui.QShortcut] = []

        self._init_ui()
        self._init_shortcuts()
        self._sync_audio_engine()
        self._update_ui_state()
        self._refresh_plot()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._centered_once:
            self._centered_once = True
            dialog_geo = self.frameGeometry()
            if self.parent() is not None and hasattr(self.parent(), "frameGeometry"):
                dialog_geo.moveCenter(self.parent().frameGeometry().center())
                screen = self.parent().screen() if hasattr(self.parent(), "screen") else None
            else:
                screen = self.screen()
            if screen is None:
                screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                x = max(avail.left(), min(dialog_geo.left(), avail.right() - dialog_geo.width() + 1))
                y = max(avail.top(), min(dialog_geo.top(), avail.bottom() - dialog_geo.height() + 1))
                self.move(x, y)
            else:
                self.move(dialog_geo.topLeft())

        if not self._bypass_emitted_on_show:
            self._bypass_emitted_on_show = True
            if self.bypass_eq_checkbox.isChecked() and not self._has_previewed:
                self._measurement_bypass_active = True
                self.sigMeasurementBypassRequested.emit(True)

    def _release_measurement_bypass(self) -> None:
        if self._measurement_bypass_active:
            self._measurement_bypass_active = False
            self.sigMeasurementBypassRequested.emit(False)

    def _finalize_session(self) -> None:
        self.audio_engine.stop()
        self._release_measurement_bypass()
        if self._has_previewed and not self._committed_eq:
            self._has_previewed = False
            self.sigPreviewEnded.emit()

    def done(self, r: int) -> None:
        self._finalize_session()
        super().done(r)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._finalize_session()
        super().closeEvent(event)

    def reject(self) -> None:
        self._finalize_session()
        super().reject()

    # -------------------------------------------------------------------------
    # UI Layout & Construction
    # -------------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.setObjectName("hearingTestDialog")
        self.setStyleSheet(f"""
            QDialog#hearingTestDialog {{
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 8px;
                font-family: {_UI_FONT_FAMILY};
            }}
            QLabel {{
                color: #ededed;
                font-family: {_UI_FONT_FAMILY};
                border: none;
                background: transparent;
            }}
            QPushButton {{
                background-color: #1c1c1f;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 500;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                background-color: #27272a;
                border-color: #3f3f46;
            }}
            QPushButton:pressed {{
                background-color: #121214;
            }}
            QPushButton:checked {{
                background-color: #27272a;
                border-color: #3f3f46;
                color: #ededed;
            }}
            QCheckBox {{
                color: #a1a1aa;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 400;
                spacing: 8px;
                background: transparent;
            }}
            QCheckBox:hover {{
                color: #ededed;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #3f3f46;
                background-color: #1c1c1f;
            }}
            QCheckBox::indicator:checked {{
                background-color: #ededed;
                border-color: #ededed;
            }}
            QSlider {{
                background: transparent;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: #27272a;
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: #a1a1aa;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #ededed;
                border: 1px solid #3f3f46;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #ffffff;
                border-color: #a1a1aa;
            }}
            QComboBox {{
                background-color: #1c1c1f;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 400;
                padding: 4px 8px;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: #3f3f46;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                selection-background-color: #27272a;
                selection-color: #ededed;
                outline: 0;
                padding: 4px;
            }}
            QScrollBar:vertical {{
                background: #121214;
                width: 8px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: #27272a;
                border-radius: 4px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #3f3f46;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = HearingDialogTitleBar(self, "Hearing & headphone test")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(16, 12, 16, 16)
        content_layout.setSpacing(12)

        # Top Utility Strip: Output Device Picker + Bypass EQ during test toggle
        top_bar = self._create_top_utility_bar()
        content_layout.addWidget(top_bar)

        # Optional audio playback error/warning banner
        self.lbl_audio_warning = QtWidgets.QLabel()
        self.lbl_audio_warning.setObjectName("audioWarningBanner")
        self.lbl_audio_warning.setWordWrap(True)
        self.lbl_audio_warning.setStyleSheet(f"""
            QLabel#audioWarningBanner {{
                background-color: #221214;
                border: 1px solid #e5484d;
                border-radius: 4px;
                color: #f5a8ab;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 500;
                padding: 8px 12px;
            }}
        """)
        self._update_audio_warning_banner()
        content_layout.addWidget(self.lbl_audio_warning)

        # 1. Top pyqtgraph Plot Canvas (Measured Thresholds + Generated EQ)
        self._init_plot(content_layout)

        # 2. Middle Row: Frequency Stepper + Signal Generator Cards
        mid_layout = QtWidgets.QHBoxLayout()
        mid_layout.setSpacing(12)

        left_card = self._create_stepper_card()
        mid_layout.addWidget(left_card, stretch=4)

        right_card = self._create_signal_generator_card()
        mid_layout.addWidget(right_card, stretch=6)

        content_layout.addLayout(mid_layout)

        # 3. Bottom Card: Compensation Tuning & Action Bar
        bottom_card = self._create_compensation_and_action_card()
        content_layout.addWidget(bottom_card)

        main_layout.addWidget(content)

    def _create_top_utility_bar(self) -> QtWidgets.QFrame:
        bar = QtWidgets.QFrame()
        bar.setObjectName("hearingTopBar")
        bar.setStyleSheet(f"""
            QFrame#hearingTopBar {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 6px;
            }}
            QFrame#hearingTopBar QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)

        lbl_out = QtWidgets.QLabel("Output")
        lbl_out.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        layout.addWidget(lbl_out)

        self.output_device_combo = _StudioComboBox()
        self.output_device_combo.setFixedHeight(28)
        self.output_device_combo.setMinimumWidth(260)
        self.output_device_combo.setMaximumWidth(420)
        self._populate_output_devices()
        self.output_device_combo.currentIndexChanged.connect(self._on_output_device_changed)
        layout.addWidget(self.output_device_combo)

        layout.addStretch()

        self.bypass_eq_checkbox = _StudioCheckBox("Bypass EQ during test")
        self.bypass_eq_checkbox.setChecked(True)
        self.bypass_eq_checkbox.setToolTip("Temporarily bypass active Equalizer APO filters while measuring hearing thresholds")
        self.bypass_eq_checkbox.toggled.connect(self._on_measurement_bypass_toggled)
        layout.addWidget(self.bypass_eq_checkbox)

        return bar

    def _populate_output_devices(self) -> None:
        devices = HearingAudioEngine.list_output_devices()
        self.output_device_combo.blockSignals(True)
        self.output_device_combo.clear()

        target_clean = (
            clean_apo_device_name(self._initial_device_name).lower()
            if self._initial_device_name and "auto-follow" not in self._initial_device_name.lower()
            else ""
        )
        exact_idx: Optional[int] = None
        partial_idx: Optional[int] = None
        id_match_idx: Optional[int] = None

        for combo_i, (dev_idx, display_name) in enumerate(devices):
            self.output_device_combo.addItem(display_name, (dev_idx, display_name))
            if dev_idx is None:
                continue
            if self._initial_device_index is not None and dev_idx == self._initial_device_index:
                id_match_idx = combo_i
            if target_clean and target_clean != "all":
                cand_clean = clean_apo_device_name(display_name).lower()
                if cand_clean == target_clean and exact_idx is None:
                    exact_idx = combo_i
                elif (target_clean in cand_clean or cand_clean in target_clean) and partial_idx is None:
                    partial_idx = combo_i

        selected = exact_idx if exact_idx is not None else (
            partial_idx if partial_idx is not None else (id_match_idx if id_match_idx is not None else 0)
        )
        self.output_device_combo.setCurrentIndex(selected)
        self.output_device_combo.blockSignals(False)

    def _update_audio_warning_banner(self) -> None:
        if self.audio_engine.audio_error:
            msg = self.audio_engine.audio_error.rstrip(".")
            self.lbl_audio_warning.setText(msg)
            self.lbl_audio_warning.show()
        else:
            self.lbl_audio_warning.hide()

    def _init_plot(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        plot_frame = QtWidgets.QFrame()
        plot_frame.setObjectName("hearingPlotFrame")
        plot_frame.setStyleSheet(f"""
            QFrame#hearingPlotFrame {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 6px;
            }}
            QFrame#hearingPlotFrame QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)
        plot_layout = QtWidgets.QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(12, 8, 12, 12)
        plot_layout.setSpacing(8)

        # Plot Header Legend (crisp vector swatches, zero unicode)
        hdr_layout = QtWidgets.QHBoxLayout()
        hdr_layout.setSpacing(16)

        def _add_legend_item(kind: str, color: str, text: str) -> None:
            item_row = QtWidgets.QHBoxLayout()
            item_row.setSpacing(6)
            item_row.addWidget(_LegendSwatch(kind, color, plot_frame))
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("font-size: 11px; font-weight: 400; color: #a1a1aa;")
            item_row.addWidget(lbl)
            hdr_layout.addLayout(item_row)

        _add_legend_item("dot", "#4cc38a", "Recorded threshold")
        _add_legend_item("solid", "#ededed", "Compensation EQ")
        _add_legend_item("dashed", "#71717a", "ISO reference")

        hdr_layout.addStretch()

        self.lbl_plot_status = QtWidgets.QLabel("0/10 recorded")
        self.lbl_plot_status.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #a1a1aa;"
        )
        hdr_layout.addWidget(self.lbl_plot_status)
        plot_layout.addLayout(hdr_layout)

        # pyqtgraph PlotWidget (all items share log10(Hz) X coordinates; avoid double-logging)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#000000")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.14)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.hideButtons()
        self.plot_widget.setXRange(math.log10(20.0), math.log10(20000.0), padding=0.02)
        self.plot_widget.setYRange(-92.0, 22.0, padding=0.02)
        self.plot_widget.setMinimumHeight(180)
        self.plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        x_ticks = [
            (math.log10(31.5), "31"),
            (math.log10(63), "63"),
            (math.log10(125), "125"),
            (math.log10(250), "250"),
            (math.log10(500), "500"),
            (math.log10(1000), "1k"),
            (math.log10(2000), "2k"),
            (math.log10(4000), "4k"),
            (math.log10(8000), "8k"),
            (math.log10(16000), "16k"),
        ]
        axis_font = QtGui.QFont("Consolas", 8)
        bottom_axis = self.plot_widget.getAxis("bottom")
        left_axis = self.plot_widget.getAxis("left")
        bottom_axis.setTicks([x_ticks])
        bottom_axis.setPen(pg.mkPen("#27272a", width=1))
        bottom_axis.setTextPen(pg.mkPen("#71717a"))
        bottom_axis.setTickFont(axis_font)
        left_axis.setPen(pg.mkPen("#27272a", width=1))
        left_axis.setTextPen(pg.mkPen("#71717a"))
        left_axis.setTickFont(axis_font)

        # Baseline zero line
        self.zero_line = pg.InfiniteLine(
            pos=0.0, angle=0, pen=pg.mkPen("#27272a", width=1, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.zero_line)

        # Reference / ISO curve
        self.ref_curve = self.plot_widget.plot(
            pen=pg.mkPen("#52525b", width=1.2, style=QtCore.Qt.PenStyle.DotLine)
        )

        # Secondary EQ curve (used for opposite channel in Per-ear L/R mode)
        self.eq_curve_secondary = self.plot_widget.plot(
            pen=pg.mkPen("#a1a1aa", width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.eq_curve_secondary.hide()

        # Primary Composite Resultant EQ curve
        self.eq_curve = self.plot_widget.plot(pen=pg.mkPen("#ffffff", width=2))

        # Threshold points (interactive scatter handles)
        self.scatter_plot = pg.ScatterPlotItem(
            size=9,
            pen=pg.mkPen("#4cc38a", width=1.5),
            brush=pg.mkBrush("#13261d"),
            hoverable=True,
            hoverPen=pg.mkPen("#ffffff", width=1.5),
            hoverBrush=pg.mkBrush("#ededed"),
        )
        self.scatter_plot.sigClicked.connect(self._on_scatter_point_clicked)
        self.plot_widget.addItem(self.scatter_plot)

        # Active Frequency Cursor Line (quiet hairline)
        self.freq_cursor = pg.InfiniteLine(
            pos=math.log10(1000.0), angle=90, pen=pg.mkPen("#71717a", width=1.2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.freq_cursor)

        plot_layout.addWidget(self.plot_widget, stretch=1)
        parent_layout.addWidget(plot_frame, stretch=1)

    def _create_stepper_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("hearingStepperCard")
        card.setStyleSheet(f"""
            QFrame#hearingStepperCard {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 6px;
            }}
            QFrame#hearingStepperCard QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Top Title & Suite Selector
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)
        lbl = QtWidgets.QLabel("Frequency stepper")
        lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        top_row.addWidget(lbl)
        top_row.addStretch()

        self.suite_combo = _StudioComboBox()
        self.suite_combo.setFixedHeight(28)
        self.suite_combo.addItems(["10-band standard", "15-band detailed", "31-band 1/3 octave"])
        self.suite_combo.currentIndexChanged.connect(self._on_suite_changed)
        top_row.addWidget(self.suite_combo)
        layout.addLayout(top_row)

        # Channel Selector Buttons [Both ears] [Left ear] [Right ear]
        ch_row = QtWidgets.QHBoxLayout()
        ch_row.setSpacing(4)
        self.ch_both_btn = QtWidgets.QPushButton("Both ears")
        self.ch_left_btn = QtWidgets.QPushButton("Left ear")
        self.ch_right_btn = QtWidgets.QPushButton("Right ear")

        ch_btn_qss = f"""
            QPushButton {{
                background-color: #1c1c1f;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #a1a1aa;
                font-family: {_UI_FONT_FAMILY};
                font-size: 11px;
                font-weight: 500;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #27272a;
                border-color: #3f3f46;
                color: #ededed;
            }}
            QPushButton:checked {{
                background-color: #27272a;
                border-color: #52525b;
                color: #ededed;
            }}
        """
        for b in [self.ch_both_btn, self.ch_left_btn, self.ch_right_btn]:
            b.setCheckable(True)
            b.setFixedHeight(28)
            b.setStyleSheet(ch_btn_qss)

        self.ch_both_btn.setChecked(True)
        self.ch_both_btn.clicked.connect(lambda: self._set_channel("both"))
        self.ch_left_btn.clicked.connect(lambda: self._set_channel("left"))
        self.ch_right_btn.clicked.connect(lambda: self._set_channel("right"))

        ch_row.addWidget(self.ch_both_btn)
        ch_row.addWidget(self.ch_left_btn)
        ch_row.addWidget(self.ch_right_btn)
        layout.addLayout(ch_row)

        # Center: Prominent Frequency Readout
        freq_box = QtWidgets.QVBoxLayout()
        freq_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        freq_box.setSpacing(4)

        self.lbl_freq_step = QtWidgets.QLabel("Band 6 of 10")
        self.lbl_freq_step.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 400; color: #71717a;"
        )
        freq_box.addWidget(self.lbl_freq_step, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.lbl_freq_big = QtWidgets.QLabel("1 kHz")
        self.lbl_freq_big.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 28px; font-weight: 600; color: #ededed;"
        )
        freq_box.addWidget(self.lbl_freq_big, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.lbl_threshold_status = QtWidgets.QLabel("Pending  -30.0 dB")
        self.lbl_threshold_status.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #d99b38;"
        )
        freq_box.addWidget(self.lbl_threshold_status, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addLayout(freq_box)

        # Stepper Buttons [Previous] [Next]
        step_row = QtWidgets.QHBoxLayout()
        step_row.setSpacing(8)

        self.prev_btn = QtWidgets.QPushButton("Previous")
        self.prev_btn.setFixedHeight(32)
        self.prev_btn.setToolTip("Go to previous frequency (Left Arrow)")
        self.prev_btn.clicked.connect(self._on_prev_clicked)
        step_row.addWidget(self.prev_btn)

        self.next_btn = QtWidgets.QPushButton("Next")
        self.next_btn.setFixedHeight(32)
        self.next_btn.setObjectName("primaryStepBtn")
        self.next_btn.setStyleSheet(f"""
            QPushButton#primaryStepBtn {{
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                color: #ededed;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 500;
                padding: 4px 12px;
            }}
            QPushButton#primaryStepBtn:hover {{
                background-color: #3f3f46;
                border-color: #52525b;
            }}
            QPushButton#primaryStepBtn:pressed {{
                background-color: #1c1c1f;
            }}
        """)
        self.next_btn.setToolTip("Record threshold and advance to next frequency (Right Arrow or Enter)")
        self.next_btn.clicked.connect(self._on_next_clicked)
        step_row.addWidget(self.next_btn)

        layout.addLayout(step_row)
        return card

    def _create_signal_generator_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("hearingSignalCard")
        card.setStyleSheet(f"""
            QFrame#hearingSignalCard {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 6px;
            }}
            QFrame#hearingSignalCard QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Card Header
        hdr_row = QtWidgets.QHBoxLayout()
        hdr_row.setSpacing(8)
        lbl = QtWidgets.QLabel("Signal generator & masking")
        lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        hdr_row.addWidget(lbl)
        hdr_row.addStretch()

        # Tone Mode Switch: [Pulsed beep] vs [Continuous]
        self.tone_mode_btn = QtWidgets.QPushButton("Pulsed beep")
        self.tone_mode_btn.setCheckable(True)
        self.tone_mode_btn.setChecked(True)
        self.tone_mode_btn.setFixedHeight(28)
        self.tone_mode_btn.setToolTip("Pulsed beeps prevent ear adaptation and make thresholds easier to pinpoint")
        self.tone_mode_btn.clicked.connect(self._toggle_tone_mode)
        hdr_row.addWidget(self.tone_mode_btn)
        layout.addLayout(hdr_row)

        # 1. Sine Wave Tone Section
        sine_box = QtWidgets.QVBoxLayout()
        sine_box.setSpacing(8)

        sine_top = QtWidgets.QHBoxLayout()
        sine_top.setSpacing(8)
        lbl_sine = QtWidgets.QLabel("Sine wave level")
        lbl_sine.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed;")
        sine_top.addWidget(lbl_sine)

        lbl_sine_hint = QtWidgets.QLabel("Adjust until barely audible")
        lbl_sine_hint.setStyleSheet("font-size: 11px; font-weight: 400; color: #71717a;")
        sine_top.addWidget(lbl_sine_hint)
        sine_top.addStretch()

        self.sine_mute_btn = QtWidgets.QPushButton("Tone on")
        self.sine_mute_btn.setCheckable(True)
        self.sine_mute_btn.setChecked(True)
        self.sine_mute_btn.setFixedSize(76, 28)
        self.sine_mute_btn.clicked.connect(self._toggle_sine_mute)
        sine_top.addWidget(self.sine_mute_btn)

        self.lbl_sine_val = QtWidgets.QLabel("-30.0 dB")
        self.lbl_sine_val.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 13px; font-weight: 500; color: #ededed; min-width: 68px;"
        )
        self.lbl_sine_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        sine_top.addWidget(self.lbl_sine_val)
        sine_box.addLayout(sine_top)

        # Sine Volume Slider (-90.0 dB to 0.0 dB in 0.5 dB steps = 180 ticks)
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.setSpacing(8)

        self.sine_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sine_slider.setRange(-180, 0)
        self.sine_slider.setValue(int(-30.0 * 2))
        self.sine_slider.valueChanged.connect(self._on_sine_slider_changed)
        slider_row.addWidget(self.sine_slider, stretch=1)

        step_btn_qss = f"""
            QPushButton {{
                background-color: #1c1c1f;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                font-family: {_MONO_FONT_FAMILY};
                font-size: 11px;
                font-weight: 500;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: #27272a;
                border-color: #3f3f46;
            }}
        """
        for label, delta in [("-5", -5.0), ("-1", -1.0), ("+1", 1.0), ("+5", 5.0)]:
            b = QtWidgets.QPushButton(label)
            b.setFixedSize(32, 28)
            b.setStyleSheet(step_btn_qss)
            b.clicked.connect(lambda _, d=delta: self._step_sine_volume(d))
            slider_row.addWidget(b)

        sine_box.addLayout(slider_row)
        layout.addLayout(sine_box)

        # 2. Reference Noise Floor Section (White Noise / Pink Noise)
        noise_box = QtWidgets.QVBoxLayout()
        noise_box.setSpacing(8)

        noise_top = QtWidgets.QHBoxLayout()
        noise_top.setSpacing(8)
        lbl_noise = QtWidgets.QLabel("Reference noise floor")
        lbl_noise.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed;")
        noise_top.addWidget(lbl_noise)

        self.noise_type_btn = QtWidgets.QPushButton("White noise")
        self.noise_type_btn.setFixedHeight(28)
        self.noise_type_btn.setToolTip("Toggle between white noise (flat per Hz) and pink noise (flat per octave)")
        self.noise_type_btn.clicked.connect(self._toggle_noise_type)
        noise_top.addWidget(self.noise_type_btn)

        noise_top.addStretch()

        self.noise_mute_btn = QtWidgets.QPushButton("Noise on")
        self.noise_mute_btn.setCheckable(True)
        self.noise_mute_btn.setChecked(True)
        self.noise_mute_btn.setFixedSize(76, 28)
        self.noise_mute_btn.clicked.connect(self._toggle_noise_mute)
        noise_top.addWidget(self.noise_mute_btn)

        self.lbl_noise_val = QtWidgets.QLabel("-35.0 dB")
        self.lbl_noise_val.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 13px; font-weight: 500; color: #a1a1aa; min-width: 68px;"
        )
        self.lbl_noise_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        noise_top.addWidget(self.lbl_noise_val)
        noise_box.addLayout(noise_top)

        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.noise_slider.setRange(-70, 0)
        self.noise_slider.setValue(-35)
        self.noise_slider.valueChanged.connect(self._on_noise_slider_changed)
        noise_box.addWidget(self.noise_slider)

        layout.addLayout(noise_box)
        return card

    def _create_compensation_and_action_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("hearingActionCard")
        card.setStyleSheet(f"""
            QFrame#hearingActionCard {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 6px;
            }}
            QFrame#hearingActionCard QLabel {{
                border: none;
                background: transparent;
                font-family: {_UI_FONT_FAMILY};
            }}
        """)
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Compensation Tuning Controls Row
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(12)

        # Mode Selection
        mode_box = QtWidgets.QHBoxLayout()
        mode_box.setSpacing(8)
        lbl_mode = QtWidgets.QLabel("Model")
        lbl_mode.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        mode_box.addWidget(lbl_mode)

        self.mode_combo = _StudioComboBox()
        self.mode_combo.setFixedHeight(28)
        self.mode_combo.addItems(["Direct masking", "ISO 226 equal-loudness"])
        self.mode_combo.currentIndexChanged.connect(self._on_compensation_mode_changed)
        mode_box.addWidget(self.mode_combo)
        ctrl_row.addLayout(mode_box)

        # Export Channel Mode (Stereo Avg vs Per-ear L/R)
        export_box = QtWidgets.QHBoxLayout()
        export_box.setSpacing(8)
        lbl_export = QtWidgets.QLabel("Channels")
        lbl_export.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        export_box.addWidget(lbl_export)

        self.export_mode_combo = _StudioComboBox()
        self.export_mode_combo.setFixedHeight(28)
        self.export_mode_combo.addItems(["Stereo (Avg)", "Per-ear (L/R)"])
        self.export_mode_combo.setToolTip("Export one shared stereo curve or independent Left and Right channel bands")
        self.export_mode_combo.currentIndexChanged.connect(self._on_export_mode_changed)
        export_box.addWidget(self.export_mode_combo)
        ctrl_row.addLayout(export_box)

        # Strength Slider (0% - 100%)
        strength_box = QtWidgets.QHBoxLayout()
        strength_box.setSpacing(8)
        lbl_str = QtWidgets.QLabel("Strength")
        lbl_str.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        strength_box.addWidget(lbl_str)

        self.strength_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.setFixedWidth(72)
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        strength_box.addWidget(self.strength_slider)

        self.lbl_strength_val = QtWidgets.QLabel("100%")
        self.lbl_strength_val.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #ededed; min-width: 36px;"
        )
        strength_box.addWidget(self.lbl_strength_val)
        ctrl_row.addLayout(strength_box)

        # Max Boost Limit (+3 dB to +18 dB)
        boost_box = QtWidgets.QHBoxLayout()
        boost_box.setSpacing(8)
        lbl_boost = QtWidgets.QLabel("Max boost")
        lbl_boost.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa;")
        boost_box.addWidget(lbl_boost)

        self.boost_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.boost_slider.setRange(3, 18)
        self.boost_slider.setValue(10)
        self.boost_slider.setFixedWidth(72)
        self.boost_slider.valueChanged.connect(self._on_boost_changed)
        boost_box.addWidget(self.boost_slider)

        self.lbl_boost_val = QtWidgets.QLabel("+10 dB")
        self.lbl_boost_val.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #ededed; min-width: 44px;"
        )
        boost_box.addWidget(self.lbl_boost_val)
        ctrl_row.addLayout(boost_box)

        # Calculated Anti-Clipping Preamp Readout
        ctrl_row.addStretch()
        self.lbl_preamp_status = QtWidgets.QLabel("Preamp 0.0 dB")
        self.lbl_preamp_status.setObjectName("preampStatusLbl")
        self.lbl_preamp_status.setStyleSheet(
            f"QLabel#preampStatusLbl {{ font-family: {_MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #a1a1aa; }}"
        )
        ctrl_row.addWidget(self.lbl_preamp_status)

        layout.addLayout(ctrl_row)

        # Divider line
        div = QtWidgets.QFrame()
        div.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        div.setStyleSheet("background-color: #27272a; border: none; max-height: 1px;")
        layout.addWidget(div)

        # Bottom Action Bar
        act_row = QtWidgets.QHBoxLayout()
        act_row.setSpacing(8)

        self.reset_btn = QtWidgets.QPushButton("Reset test")
        self.reset_btn.setFixedHeight(32)
        self.reset_btn.setToolTip("Clear all measurements back to default")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        act_row.addWidget(self.reset_btn)

        self.ab_compare_btn = QtWidgets.QPushButton("A/B compare: off")
        self.ab_compare_btn.setFixedHeight(32)
        self.ab_compare_btn.setCheckable(True)
        self.ab_compare_btn.setToolTip("Toggle between flat bypass and calculated EQ profile")
        self.ab_compare_btn.clicked.connect(self._toggle_ab_compare)
        act_row.addWidget(self.ab_compare_btn)

        act_row.addStretch()

        self.save_preset_btn = QtWidgets.QPushButton("Save as preset")
        self.save_preset_btn.setFixedHeight(32)
        self.save_preset_btn.setToolTip("Save hearing correction curve as a new profile in Clean APO")
        self.save_preset_btn.clicked.connect(self._on_save_preset_clicked)
        act_row.addWidget(self.save_preset_btn)

        self.apply_btn = QtWidgets.QPushButton("Apply to current EQ")
        self.apply_btn.setFixedHeight(32)
        self.apply_btn.setObjectName("applyBtn")
        self.apply_btn.setStyleSheet(f"""
            QPushButton#applyBtn {{
                background-color: #ededed;
                border: 1px solid #ededed;
                border-radius: 4px;
                color: #0b0b0c;
                font-family: {_UI_FONT_FAMILY};
                font-size: 12px;
                font-weight: 600;
                padding: 4px 16px;
            }}
            QPushButton#applyBtn:hover {{
                background-color: #ffffff;
                border-color: #ffffff;
            }}
            QPushButton#applyBtn:pressed {{
                background-color: #d4d4d8;
            }}
        """)
        self.apply_btn.setToolTip("Write filter configuration to Equalizer APO")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        act_row.addWidget(self.apply_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setFixedHeight(32)
        self.close_btn.clicked.connect(self.reject)
        act_row.addWidget(self.close_btn)

        layout.addLayout(act_row)
        return card

    def _init_shortcuts(self) -> None:
        """Bind dialog-wide shortcuts so child widgets cannot steal navigation keys."""
        for widget in self.findChildren((QtWidgets.QPushButton, QtWidgets.QSlider, QtWidgets.QCheckBox)):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        bindings = [
            (QtCore.Qt.Key.Key_Up, lambda: self._step_sine_volume(+1.0)),
            (QtCore.Qt.Key.Key_Down, lambda: self._step_sine_volume(-1.0)),
            (QtCore.Qt.Key.Key_PageUp, lambda: self._step_sine_volume(+5.0)),
            (QtCore.Qt.Key.Key_PageDown, lambda: self._step_sine_volume(-5.0)),
            (QtCore.Qt.Key.Key_Right, self._on_next_clicked),
            (QtCore.Qt.Key.Key_Return, self._on_next_clicked),
            (QtCore.Qt.Key.Key_Enter, self._on_next_clicked),
            (QtCore.Qt.Key.Key_Left, self._on_prev_clicked),
            (QtCore.Qt.Key.Key_Space, self._toggle_sine_mute_shortcut),
        ]
        for key, slot in bindings:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    def _toggle_sine_mute_shortcut(self) -> None:
        self.sine_mute_btn.toggle()
        self._toggle_sine_mute()

    # -------------------------------------------------------------------------
    # Event Handlers & State Management
    # -------------------------------------------------------------------------
    def _on_output_device_changed(self, idx: int) -> None:
        data = self.output_device_combo.itemData(idx)
        if isinstance(data, tuple) and len(data) == 2:
            dev_idx, dev_name = data
        else:
            dev_idx, dev_name = None, None
        target_name = None if dev_idx is None else dev_name
        self.audio_engine.set_device(dev_idx, target_name)
        self._update_audio_warning_banner()
        self._sync_audio_engine()

    def _on_measurement_bypass_toggled(self, checked: bool) -> None:
        if checked:
            if not self._measurement_bypass_active:
                self._measurement_bypass_active = True
                self.sigMeasurementBypassRequested.emit(True)
        else:
            self._release_measurement_bypass()

    def _maybe_auto_select_per_ear_mode(self) -> None:
        if (
            not self._user_override_export_mode
            and self.session.export_channel_mode != "per_ear"
            and self.session.has_independent_lr_data()
        ):
            self.session.export_channel_mode = "per_ear"
            self.export_mode_combo.blockSignals(True)
            self.export_mode_combo.setCurrentIndex(1)
            self.export_mode_combo.blockSignals(False)

    def _on_export_mode_changed(self, idx: int) -> None:
        self._user_override_export_mode = True
        self.session.export_channel_mode = "per_ear" if idx == 1 else "stereo"
        self._update_ui_state()
        self._refresh_plot()

    def _compute_active_compensation(self) -> Tuple[float, List[EQBand]]:
        return self.session.compute_compensation(per_ear=(self.session.export_channel_mode == "per_ear"))

    def _sync_audio_engine(self) -> None:
        freq = self.session.current_freq
        sine_vol = self.session.get_threshold()
        self.audio_engine.set_frequency(freq)
        self.audio_engine.set_sine_volume_db(sine_vol)
        self.audio_engine.set_channel(self.session.channel)

    def _update_ui_state(self) -> None:
        idx = self.session.current_index
        total = len(self.session.frequencies)
        freq = self.session.current_freq
        vol_db = self.session.get_threshold()
        is_rec = self.session.is_tested()

        # Update frequency labels
        self.lbl_freq_step.setText(f"Band {idx + 1} of {total}")
        if freq >= 1000.0:
            self.lbl_freq_big.setText(
                f"{freq / 1000.0:g} kHz" if freq % 1000 == 0 or (freq * 10) % 1000 == 0 else f"{freq:,.0f} Hz"
            )
        else:
            self.lbl_freq_big.setText(f"{freq:,.1f} Hz" if freq % 1 != 0 else f"{freq:,.0f} Hz")

        # Threshold status readout
        if is_rec:
            self.lbl_threshold_status.setText(f"Recorded  {vol_db:.1f} dB")
            self.lbl_threshold_status.setStyleSheet(
                f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #4cc38a;"
            )
        else:
            self.lbl_threshold_status.setText(f"Pending  {vol_db:.1f} dB")
            self.lbl_threshold_status.setStyleSheet(
                f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #d99b38;"
            )

        # Update sine volume slider and readout (without re-triggering signal loop)
        self.sine_slider.blockSignals(True)
        self.sine_slider.setValue(int(round(vol_db * 2.0)))
        self.sine_slider.blockSignals(False)
        self.lbl_sine_val.setText(f"{vol_db:.1f} dB")

        # Update cursor line on plot
        self.freq_cursor.setValue(math.log10(max(1.0, freq)))

        # Count recorded bands for active channel
        num_tested = sum(1 for f in self.session.frequencies if self.session.is_tested(freq=f))
        mode_tag = "  L/R" if self.session.export_channel_mode == "per_ear" else ""
        self.lbl_plot_status.setText(f"{num_tested}/{total} recorded{mode_tag}")
        self.lbl_plot_status.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #a1a1aa;"
        )

        # Preamp indicator (functional amber warning when attenuation required)
        preamp, _ = self._compute_active_compensation()
        self.lbl_preamp_status.setText(f"Preamp {preamp:.1f} dB")
        if preamp < 0.0:
            self.lbl_preamp_status.setStyleSheet(
                f"QLabel#preampStatusLbl {{ font-family: {_MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #d99b38; }}"
            )
        else:
            self.lbl_preamp_status.setStyleSheet(
                f"QLabel#preampStatusLbl {{ font-family: {_MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #a1a1aa; }}"
            )

    def _refresh_plot(self) -> None:
        freqs = self.session.frequencies
        log_f = [math.log10(f) for f in freqs]

        # 1. Update Threshold Scatter Points
        spots = []
        curr_freq = self.session.current_freq
        for f, lf in zip(freqs, log_f):
            t_val = self.session.get_threshold(freq=f)
            is_tested = self.session.is_tested(freq=f)
            is_active = (f == curr_freq)

            if is_active:
                spots.append({
                    "pos": (lf, t_val),
                    "size": 13,
                    "pen": pg.mkPen("#ffffff", width=2.0),
                    "brush": pg.mkBrush("#4cc38a" if is_tested else "#ededed"),
                })
            elif is_tested:
                spots.append({
                    "pos": (lf, t_val),
                    "size": 9,
                    "pen": pg.mkPen("#4cc38a", width=1.5),
                    "brush": pg.mkBrush("#13261d"),
                })
            else:
                spots.append({
                    "pos": (lf, t_val),
                    "size": 7,
                    "pen": pg.mkPen("#52525b", width=1.0),
                    "brush": pg.mkBrush("#1c1c1f"),
                })
        self.scatter_plot.setData(spots)

        # 2. Update Reference / Target curve (aligned with exact effective baseline)
        if self.session.compensation_mode == "iso226":
            dense_f = np.logspace(np.log10(20.0), np.log10(20000.0), 400)
            iso_delta = ISO226.normalized_delta(dense_f, phon=60.0)
            _, _, _, baseline = self.session.get_effective_thresholds_and_baseline()
            ref_y = baseline + iso_delta
            self.ref_curve.setData(np.log10(dense_f), ref_y)
            self.ref_curve.show()
        else:
            self.ref_curve.hide()

        # 3. Calculate and display Resultant Parametric EQ Curve(s)
        preamp, bands = self._compute_active_compensation()
        eval_f = np.logspace(np.log10(20.0), np.log10(20000.0), 500)
        log_eval = np.log10(eval_f)
        if self.session.export_channel_mode == "per_ear":
            left_bands = [b for b in bands if b.channel in ("ALL", "L")]
            right_bands = [b for b in bands if b.channel in ("ALL", "R")]
            primary_bands = right_bands if self.session.channel == "right" else left_bands
            secondary_bands = left_bands if self.session.channel == "right" else right_bands
            primary_y = compute_composite_response(primary_bands, eval_f, preamp_db=preamp, fs=48000.0)
            secondary_y = compute_composite_response(secondary_bands, eval_f, preamp_db=preamp, fs=48000.0)
            self.eq_curve.setData(log_eval, primary_y)
            self.eq_curve_secondary.setData(log_eval, secondary_y)
            self.eq_curve_secondary.show()
        else:
            composite_y = compute_composite_response(bands, eval_f, preamp_db=preamp, fs=48000.0)
            self.eq_curve.setData(log_eval, composite_y)
            self.eq_curve_secondary.hide()

    def _on_scatter_point_clicked(self, item: Any, points: Any) -> None:
        if points and self.session.frequencies:
            x_val = points[0].pos().x()
            best_idx = 0
            best_dist = 1e9
            for i, f in enumerate(self.session.frequencies):
                d = abs(math.log10(f) - x_val)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            self.session.current_index = best_idx
            self._sync_audio_engine()
            self._update_ui_state()
            self._refresh_plot()

    def _on_suite_changed(self, idx: int) -> None:
        if idx == 0:
            self.session.init_frequency_suite(STANDARD_10_BANDS)
        elif idx == 1:
            self.session.init_frequency_suite(DETAILED_15_BANDS)
        else:
            self.session.init_frequency_suite(FULL_31_BANDS)
        self._sync_audio_engine()
        self._update_ui_state()
        self._refresh_plot()

    def _set_channel(self, ch: str) -> None:
        self.session.channel = ch
        self.ch_both_btn.setChecked(ch == "both")
        self.ch_left_btn.setChecked(ch == "left")
        self.ch_right_btn.setChecked(ch == "right")
        self._sync_audio_engine()
        self._update_ui_state()
        self._refresh_plot()

    def _on_sine_slider_changed(self, val: int) -> None:
        vol_db = val / 2.0
        self.session.set_threshold(vol_db)
        self.audio_engine.set_sine_volume_db(vol_db)
        self._maybe_auto_select_per_ear_mode()
        self._update_ui_state()
        self._refresh_plot()

    def _step_sine_volume(self, delta_db: float) -> None:
        curr = self.session.get_threshold()
        new_val = float(np.clip(curr + delta_db, -90.0, 0.0))
        self.session.set_threshold(new_val)
        self.audio_engine.set_sine_volume_db(new_val)
        self._maybe_auto_select_per_ear_mode()
        self._update_ui_state()
        self._refresh_plot()

    def _on_noise_slider_changed(self, val: int) -> None:
        vol_db = float(val)
        self.audio_engine.set_noise_volume_db(vol_db)
        self.lbl_noise_val.setText(f"{vol_db:.1f} dB")

    def _toggle_sine_mute(self) -> None:
        is_on = self.sine_mute_btn.isChecked()
        self.audio_engine.set_sine_enabled(is_on)
        self.sine_mute_btn.setText("Tone on" if is_on else "Tone off")
        color = "#ededed" if is_on else "#71717a"
        self.sine_mute_btn.setStyleSheet(f"font-size: 11px; padding: 4px 8px; color: {color};")

    def _toggle_noise_mute(self) -> None:
        is_on = self.noise_mute_btn.isChecked()
        self.audio_engine.set_noise_enabled(is_on)
        self.noise_mute_btn.setText("Noise on" if is_on else "Noise off")
        color = "#ededed" if is_on else "#71717a"
        self.noise_mute_btn.setStyleSheet(f"font-size: 11px; padding: 4px 8px; color: {color};")

    def _toggle_tone_mode(self) -> None:
        is_pulsed = self.tone_mode_btn.isChecked()
        mode_str = "pulsed" if is_pulsed else "continuous"
        self.audio_engine.set_tone_mode(mode_str)
        self.tone_mode_btn.setText("Pulsed beep" if is_pulsed else "Continuous")

    def _toggle_noise_type(self) -> None:
        current = self.audio_engine._noise_type
        new_type = "pink" if current == "white" else "white"
        self.audio_engine.set_noise_type(new_type)
        self.noise_type_btn.setText(f"{new_type.capitalize()} noise")

    def _on_next_clicked(self) -> None:
        # Mark current frequency as tested
        self.session.set_threshold(self.session.get_threshold())
        self._maybe_auto_select_per_ear_mode()
        if not self.session.next_frequency():
            self.session.current_index = 0
        self._sync_audio_engine()
        self._update_ui_state()
        self._refresh_plot()

    def _on_prev_clicked(self) -> None:
        self.session.prev_frequency()
        self._sync_audio_engine()
        self._update_ui_state()
        self._refresh_plot()

    def _on_compensation_mode_changed(self, idx: int) -> None:
        self.session.compensation_mode = "iso226" if idx == 1 else "direct"
        self._update_ui_state()
        self._refresh_plot()

    def _on_strength_changed(self, val: int) -> None:
        self.session.strength = val / 100.0
        self.lbl_strength_val.setText(f"{val}%")
        self._update_ui_state()
        self._refresh_plot()

    def _on_boost_changed(self, val: int) -> None:
        self.session.max_boost_db = float(val)
        self.lbl_boost_val.setText(f"+{val} dB")
        self._update_ui_state()
        self._refresh_plot()

    def _on_reset_clicked(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset hearing test",
            "Reset all measured frequency thresholds back to defaults?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.session.reset_all()
            self._user_override_export_mode = False
            self.export_mode_combo.blockSignals(True)
            self.export_mode_combo.setCurrentIndex(0)
            self.export_mode_combo.blockSignals(False)
            self._sync_audio_engine()
            self._update_ui_state()
            self._refresh_plot()

    def _reset_ab_compare_ui(self) -> None:
        self._ab_bypassed = False
        self.ab_compare_btn.blockSignals(True)
        self.ab_compare_btn.setChecked(False)
        self.ab_compare_btn.setText("A/B compare: off")
        self.ab_compare_btn.blockSignals(False)

    def _toggle_ab_compare(self) -> None:
        self._ab_bypassed = self.ab_compare_btn.isChecked()
        self._has_previewed = True
        self._committed_eq = False
        self.ab_compare_btn.setText("A/B compare: bypass" if self._ab_bypassed else "A/B compare: EQ on")
        if not self._ab_bypassed:
            # Release measurement bypass so generated EQ preview is audible
            self._release_measurement_bypass()
        preamp, bands = self._compute_active_compensation()
        self.sigPreviewEQ.emit(bands, preamp, self._ab_bypassed)

    def _on_apply_clicked(self) -> None:
        preamp, bands = self._compute_active_compensation()
        self._reset_ab_compare_ui()
        self._release_measurement_bypass()
        self._committed_eq = True
        self._has_previewed = False
        self.sigApplyEQ.emit(bands, preamp)
        self.lbl_plot_status.setText("Applied to Equalizer APO")
        self.lbl_plot_status.setStyleSheet(
            f"font-family: {_MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #4cc38a;"
        )

    def _on_save_preset_clicked(self) -> None:
        default_name = f"Hearing Calibration ({time.strftime('%Y-%m-%d')})"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save calibration preset",
            "Preset name",
            QtWidgets.QLineEdit.EchoMode.Normal,
            default_name,
        )
        if ok and name.strip():
            preamp, bands = self._compute_active_compensation()
            self._reset_ab_compare_ui()
            self._release_measurement_bypass()
            self._committed_eq = True
            self._has_previewed = False
            self.sigSavePreset.emit(name.strip(), bands, preamp)
            QtWidgets.QMessageBox.information(
                self,
                "Preset saved",
                f"Saved hearing calibration preset '{name.strip()}'",
            )

    # -------------------------------------------------------------------------
    # Keyboard Navigation (fallback for direct synthetic key events)
    # -------------------------------------------------------------------------
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Up:
            self._step_sine_volume(+1.0)
            event.accept()
        elif key == QtCore.Qt.Key.Key_Down:
            self._step_sine_volume(-1.0)
            event.accept()
        elif key == QtCore.Qt.Key.Key_PageUp:
            self._step_sine_volume(+5.0)
            event.accept()
        elif key == QtCore.Qt.Key.Key_PageDown:
            self._step_sine_volume(-5.0)
            event.accept()
        elif key in (QtCore.Qt.Key.Key_Right, QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._on_next_clicked()
            event.accept()
        elif key == QtCore.Qt.Key.Key_Left:
            self._on_prev_clicked()
            event.accept()
        elif key == QtCore.Qt.Key.Key_Space:
            self._toggle_sine_mute_shortcut()
            event.accept()
        elif key == QtCore.Qt.Key.Key_Escape:
            self.reject()
            event.accept()
        else:
            super().keyPressEvent(event)
