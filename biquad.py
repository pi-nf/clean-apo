"""
Vectorized Robert Bristow-Johnson (RBJ) Audio EQ Cookbook Transfer Functions
Computes exact complex frequency response H(f) across log-spaced frequency points.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Sequence, Tuple, Dict, Any
import numpy as np


class FilterType(str, Enum):
    PEAK = "Peak"
    LOW_SHELF = "Low Shelf"
    HIGH_SHELF = "High Shelf"
    HIGH_PASS = "High Pass"
    LOW_PASS = "Low Pass"

    @classmethod
    def from_str(cls, val: str) -> FilterType:
        val_lower = val.lower().replace("-", " ").replace("_", " ").strip()
        if "peak" in val_lower or val_lower == "pk":
            return cls.PEAK
        elif "low shelf" in val_lower or "lsc" in val_lower or val_lower == "ls":
            return cls.LOW_SHELF
        elif "high shelf" in val_lower or "hsc" in val_lower or val_lower == "hs":
            return cls.HIGH_SHELF
        elif "high pass" in val_lower or "hp" in val_lower or "hpq" in val_lower:
            return cls.HIGH_PASS
        elif "low pass" in val_lower or "lp" in val_lower or "lpq" in val_lower:
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


@dataclass
class EQBand:
    band_id: int
    enabled: bool = True
    filter_type: str = "Peak"  # "Peak", "Low Shelf", "High Shelf", "High Pass", "Low Pass"
    freq: float = 1000.0       # Hz (20 - 20000)
    gain: float = 0.0          # dB (-30 to +20)
    q: float = 1.414           # Q factor (0.1 to 20.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EQBand:
        return cls(
            band_id=int(data.get("band_id", 1)),
            enabled=bool(data.get("enabled", True)),
            filter_type=str(data.get("filter_type", "Peak")),
            freq=float(data.get("freq", 1000.0)),
            gain=float(data.get("gain", 0.0)),
            q=float(data.get("q", 1.414)),
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
    if isinstance(filter_type, str):
        filter_type = FilterType.from_str(filter_type)

    # Clamp parameters to avoid numerical instability
    f0 = np.clip(float(f0), 10.0, fs / 2.0 - 10.0)
    q = max(0.01, float(q))
    gain_db = float(gain_db)

    w0 = 2.0 * np.pi * f0 / fs
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

    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


_Z_CACHE: Dict[Tuple[int, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}


def _get_cached_z(freqs: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(freqs)
    if n == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128)
    key = (n, float(fs), float(freqs[0]), float(freqs[-1]))
    cached = _Z_CACHE.get(key)
    if cached is None:
        w = 2.0 * np.pi * freqs / fs
        z1 = np.exp(-1j * w)
        z2 = np.exp(-2j * w)
        cached = (z1, z2)
        if len(_Z_CACHE) > 32:
            _Z_CACHE.clear()
        _Z_CACHE[key] = cached
    return cached


def compute_filter_response(
    filter_type: FilterType | str,
    f0: float,
    gain_db: float,
    q: float,
    freqs: np.ndarray,
    fs: float = 48000.0,
) -> np.ndarray:
    """
    Computes magnitude response in dB across the given frequency array.
    """
    b0, b1, b2, a1, a2 = biquad_coeffs(filter_type, f0, gain_db, q, fs)
    z1, z2 = _get_cached_z(freqs, fs)

    num = b0 + b1 * z1 + b2 * z2
    den = 1.0 + a1 * z1 + a2 * z2
    H = num / den
    return 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))


def compute_composite_response(
    bands: Sequence[EQBand],
    freqs: np.ndarray,
    preamp_db: float = 0.0,
    fs: float = 48000.0,
) -> np.ndarray:
    """
    Computes total composite frequency response in dB across all enabled bands.
    Vectorized and optimized for high-speed 60+ FPS real-time rendering.
    """
    total = np.full(len(freqs), float(preamp_db), dtype=np.float64)
    z1, z2 = _get_cached_z(freqs, fs)

    for band in bands:
        if not band.enabled:
            continue
        b0, b1, b2, a1, a2 = biquad_coeffs(
            band.filter_type, band.freq, band.gain, band.q, fs
        )
        num = b0 + b1 * z1 + b2 * z2
        den = 1.0 + a1 * z1 + a2 * z2
        H = num / den
        total += 20.0 * np.log10(np.maximum(np.abs(H), 1e-12))

    return total
