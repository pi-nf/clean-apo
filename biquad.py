"""
Vectorized Robert Bristow-Johnson (RBJ) Audio EQ Cookbook Transfer Functions
Computes exact complex frequency response H(f) across log-spaced frequency points.
"""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
import math
import threading
from typing import Sequence, Tuple, Dict, Any
import numpy as np


class FilterType(str, Enum):
    PEAK = "Peak"
    LOW_SHELF = "Low Shelf"
    HIGH_SHELF = "High Shelf"
    HIGH_PASS = "High Pass"
    LOW_PASS = "Low Pass"

    @classmethod
    def from_str(cls, val: Any) -> FilterType:
        if isinstance(val, cls):
            return val
        val_lower = str(val or "").lower().replace("-", " ").replace("_", " ").strip()
        compact = val_lower.replace(" ", "")
        tokens = val_lower.split()
        first = tokens[0] if tokens else ""

        if "peak" in compact or first in ("pk", "peq"):
            return cls.PEAK
        elif (
            "lowshelf" in compact
            or first in ("ls", "lsc", "lsq", "ls6", "ls12")
            or "lsc" in tokens
            or "lsq" in tokens
        ):
            return cls.LOW_SHELF
        elif (
            "highshelf" in compact
            or first in ("hs", "hsc", "hsq", "hs6", "hs12")
            or "hsc" in tokens
            or "hsq" in tokens
        ):
            return cls.HIGH_SHELF
        elif (
            "highpass" in compact
            or first in ("hp", "hpq", "hp6", "hp12")
            or "hpq" in tokens
        ):
            return cls.HIGH_PASS
        elif (
            "lowpass" in compact
            or first in ("lp", "lpq", "lp6", "lp12")
            or "lpq" in tokens
        ):
            return cls.LOW_PASS
        return cls.PEAK

    @property
    def apo_code(self) -> str:
        """Equalizer APO filter abbreviation."""
        mapping = {
            FilterType.PEAK: "PK",
            FilterType.LOW_SHELF: "LSC",
            FilterType.HIGH_SHELF: "HSC",
            FilterType.HIGH_PASS: "HPQ",
            FilterType.LOW_PASS: "LPQ",
        }
        return mapping[self]


def _normalize_channel(val: Any) -> str:
    """Normalizes a band channel identifier to 'ALL', 'L', or 'R'."""
    token = str(val or "ALL").strip().upper()
    if token in ("L", "LEFT"):
        return "L"
    if token in ("R", "RIGHT"):
        return "R"
    return "ALL"


@dataclass
class EQBand:
    band_id: int
    enabled: bool = True
    filter_type: str = "Peak"  # "Peak", "Low Shelf", "High Shelf", "High Pass", "Low Pass"
    freq: float = 1000.0       # Hz (20 - 20000)
    gain: float = 0.0          # dB (-30 to +20)
    q: float = 1.414           # Q factor (0.1 to 20.0)
    channel: str = "ALL"       # "ALL", "L", "R"

    def __post_init__(self) -> None:
        self.channel = _normalize_channel(self.channel)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        ch = _normalize_channel(data.get("channel", "ALL"))
        if ch == "ALL":
            data.pop("channel", None)
        else:
            data["channel"] = ch
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EQBand:
        return cls(
            band_id=int(data.get("band_id", 1)),
            enabled=bool(data.get("enabled", True)),
            filter_type=str(data.get("filter_type", "Peak")),
            freq=float(data.get("freq", 1000.0)),
            gain=float(data.get("gain", 0.0)),
            q=float(data.get("q", 1.414)),
            channel=_normalize_channel(data.get("channel", "ALL")),
        )


def biquad_coeffs(
    filter_type: FilterType | str,
    f0: float,
    gain_db: float,
    q: float,
    fs: float = 48000.0,
) -> Tuple[float, float, float, float, float]:
    """
    Computes normalized biquad coefficients (b0, b1, b2, a1, a2) where a0 = 1.0.
    Follows Robert Bristow-Johnson's Audio EQ Cookbook.
    """
    if not isinstance(filter_type, FilterType):
        filter_type = FilterType.from_str(filter_type)

    # Validate and clamp parameters to avoid numerical instability / overflow / NaN
    fs_val = float(fs)
    if not math.isfinite(fs_val) or fs_val <= 0.0:
        fs_val = 48000.0
    fs_val = max(100.0, fs_val)

    f0_val = float(f0)
    if not math.isfinite(f0_val):
        f0_val = 1000.0
    nyq_max = max(1.0, min(fs_val * 0.499, fs_val * 0.5 - 10.0)) if fs_val > 22.0 else max(1.0, fs_val * 0.499)
    f0_min = min(10.0, nyq_max * 0.5)
    f0 = float(np.clip(f0_val, f0_min, nyq_max))

    q_val = float(q)
    if not math.isfinite(q_val):
        q_val = 1.414
    q = min(max(0.01, q_val), 100.0)

    gain_val = float(gain_db)
    if not math.isfinite(gain_val):
        gain_val = 0.0
    gain_db = min(max(-120.0, gain_val), 120.0)

    w0 = 2.0 * np.pi * f0 / fs_val
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    A = 10.0 ** (gain_db / 40.0)

    if filter_type == FilterType.PEAK:
        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / A

    elif filter_type == FilterType.LOW_SHELF:
        beta = np.sqrt(A) * sin_w0 / q
        b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + beta)
        b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - beta)
        a0 = (A + 1.0) + (A - 1.0) * cos_w0 + beta
        a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
        a2 = (A + 1.0) + (A - 1.0) * cos_w0 - beta

    elif filter_type == FilterType.HIGH_SHELF:
        beta = np.sqrt(A) * sin_w0 / q
        b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + beta)
        b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
        b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - beta)
        a0 = (A + 1.0) - (A - 1.0) * cos_w0 + beta
        a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
        a2 = (A + 1.0) - (A - 1.0) * cos_w0 - beta

    elif filter_type == FilterType.HIGH_PASS:
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

    elif filter_type == FilterType.LOW_PASS:
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

    else:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    if not math.isfinite(a0) or abs(a0) < 1e-30:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    return float(b0 / a0), float(b1 / a0), float(b2 / a0), float(a1 / a0), float(a2 / a0)


_Z_CACHE_LOCK = threading.Lock()
_Z_CACHE_MAX = 32
_Z_CACHE: OrderedDict[Tuple[float, bytes], Tuple[np.ndarray, np.ndarray]] = OrderedDict()


def _get_cached_z(freqs: Sequence[float] | np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    freqs_arr = np.ascontiguousarray(freqs, dtype=np.float64)
    if freqs_arr.size == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128)
    fs_val = float(fs)
    if not math.isfinite(fs_val) or fs_val <= 0.0:
        fs_val = 48000.0
    fs_val = max(100.0, fs_val)

    key = (fs_val, freqs_arr.tobytes())
    with _Z_CACHE_LOCK:
        cached = _Z_CACHE.get(key)
        if cached is not None:
            _Z_CACHE.move_to_end(key)
            return cached

    w = 2.0 * np.pi * freqs_arr / fs_val
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    result = (z1, z2)

    with _Z_CACHE_LOCK:
        _Z_CACHE[key] = result
        _Z_CACHE.move_to_end(key)
        while len(_Z_CACHE) > _Z_CACHE_MAX:
            _Z_CACHE.popitem(last=False)

    return result


def compute_filter_response(
    filter_type: FilterType | str,
    f0: float,
    gain_db: float,
    q: float,
    freqs: Sequence[float] | np.ndarray,
    fs: float = 48000.0,
) -> np.ndarray:
    """
    Computes magnitude response in dB across the given frequency array.
    """
    b0, b1, b2, a1, a2 = biquad_coeffs(filter_type, f0, gain_db, q, fs)
    z1, z2 = _get_cached_z(freqs, fs)

    num = b0 + b1 * z1 + b2 * z2
    den = 1.0 + a1 * z1 + a2 * z2
    mag = np.abs(num) / np.maximum(np.abs(den), 1e-30)
    return 20.0 * np.log10(np.maximum(mag, 1e-12))


def compute_composite_response(
    bands: Sequence[EQBand],
    freqs: Sequence[float] | np.ndarray,
    preamp_db: float = 0.0,
    fs: float = 48000.0,
) -> np.ndarray:
    """
    Computes total composite frequency response in dB across all enabled bands.
    Vectorized and optimized for high-speed 60+ FPS real-time rendering.
    """
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    preamp_val = float(preamp_db)
    if not math.isfinite(preamp_val):
        preamp_val = 0.0
    total = np.full(freqs_arr.shape, preamp_val, dtype=np.float64)
    if freqs_arr.size == 0:
        return total

    z1, z2 = _get_cached_z(freqs_arr, fs)

    for band in bands:
        if not band.enabled:
            continue
        b0, b1, b2, a1, a2 = biquad_coeffs(
            band.filter_type, band.freq, band.gain, band.q, fs
        )
        num = b0 + b1 * z1 + b2 * z2
        den = 1.0 + a1 * z1 + a2 * z2
        mag = np.abs(num) / np.maximum(np.abs(den), 1e-30)
        total += 20.0 * np.log10(np.maximum(mag, 1e-12))

    return total


_HEADROOM_BASE_FREQS: np.ndarray = np.geomspace(20.0, 20000.0, 1000, dtype=np.float64)


def compute_peak_headroom(
    bands: Sequence[EQBand],
    preamp_db: float = 0.0,
    fs: float = 48000.0,
) -> Tuple[float, float]:
    """
    Evaluates worst-case peak headroom across 1,000 log-spaced points in [20, 20000] Hz
    unioned with all enabled band center frequencies.

    When bands contain separate 'L' or 'R' channels, evaluates worst-case peak across
    ('ALL' + 'L') and ('ALL' + 'R').

    Returns:
        (peak_with_preamp_db, recommended_preamp_db)
        where recommended_preamp_db is 0.0 if raw peak <= 0.0 else -round(raw_peak + 0.1, 1).
    """
    fs_val = float(fs)
    if not math.isfinite(fs_val) or fs_val <= 0.0:
        fs_val = 48000.0
    fs_val = max(100.0, fs_val)

    preamp_val = float(preamp_db)
    if not math.isfinite(preamp_val):
        preamp_val = 0.0

    enabled_bands = [b for b in bands if getattr(b, "enabled", True)]
    if not enabled_bands:
        peak_with_preamp = 0.0 if abs(preamp_val) < 1e-12 else preamp_val
        return float(peak_with_preamp), 0.0

    nyq_max = max(20.0, min(20000.0, fs_val * 0.499))
    center_freqs = []
    all_bands = []
    left_bands = []
    right_bands = []

    for b in enabled_bands:
        f_val = float(getattr(b, "freq", 1000.0))
        if math.isfinite(f_val) and 10.0 <= f_val <= nyq_max:
            center_freqs.append(f_val)
        ch = _normalize_channel(getattr(b, "channel", "ALL"))
        if ch == "L":
            left_bands.append(b)
        elif ch == "R":
            right_bands.append(b)
        else:
            all_bands.append(b)

    base_freqs = _HEADROOM_BASE_FREQS
    if nyq_max < 20000.0:
        base_freqs = base_freqs[base_freqs <= nyq_max]
        if base_freqs.size == 0:
            base_freqs = np.array([min(1000.0, nyq_max)], dtype=np.float64)

    if center_freqs:
        eval_freqs = np.unique(
            np.concatenate([base_freqs, np.asarray(center_freqs, dtype=np.float64)])
        )
    else:
        eval_freqs = base_freqs

    if left_bands or right_bands:
        base_resp = (
            compute_composite_response(all_bands, eval_freqs, preamp_db=0.0, fs=fs_val)
            if all_bands
            else np.zeros(eval_freqs.shape, dtype=np.float64)
        )
        left_resp = (
            base_resp + compute_composite_response(left_bands, eval_freqs, preamp_db=0.0, fs=fs_val)
            if left_bands
            else base_resp
        )
        right_resp = (
            base_resp + compute_composite_response(right_bands, eval_freqs, preamp_db=0.0, fs=fs_val)
            if right_bands
            else base_resp
        )
        raw_peak = max(float(np.max(left_resp)), float(np.max(right_resp)))
    else:
        resp = compute_composite_response(all_bands, eval_freqs, preamp_db=0.0, fs=fs_val)
        raw_peak = float(np.max(resp)) if resp.size > 0 else 0.0

    if not math.isfinite(raw_peak):
        raw_peak = 0.0

    peak_with_preamp = raw_peak + preamp_val
    if abs(peak_with_preamp) < 1e-12:
        peak_with_preamp = 0.0

    if raw_peak <= 0.0:
        recommended_preamp = 0.0
    else:
        recommended_preamp = float(-round(raw_peak + 0.1, 1))
        if abs(recommended_preamp) < 1e-9:
            recommended_preamp = 0.0

    return float(peak_with_preamp), float(recommended_preamp)
