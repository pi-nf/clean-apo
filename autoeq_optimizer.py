"""
Native AutoEQ Optimization Engine using SciPy.
Calculates high-precision parametric biquad curve fitting (Peak and Low Shelf)
to match raw IEM frequency response measurements to target compensation curves.
Strictly local and offline; non-blocking QThread worker for seamless UI responsiveness.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from PyQt6 import QtCore

from biquad import biquad_coeffs, FilterType, EQBand


@dataclass
class OptimizationResult:
    bands: List[EQBand]
    preamp_db: float
    initial_rms_db: float
    final_rms_db: float
    elapsed_ms: float
    f_grid: np.ndarray
    meas_aligned_spl: np.ndarray
    target_aligned_spl: np.ndarray
    error_curve: np.ndarray
    eq_response_curve: np.ndarray


def optimize_parametric_eq(
    meas_freqs: Sequence[float],
    meas_spls: Sequence[float],
    target_freqs: Sequence[float],
    target_spls: Sequence[float],
    num_bands: int = 7,
    use_bass_shelf: bool = True,
    max_boost_db: float = 9.0,
    max_cut_db: float = 12.0,
    min_freq: float = 20.0,
    max_freq: float = 8500.0,
    min_q: float = 0.5,
    max_q: float = 6.0,
    norm_mode: str = "midband",  # "midband" (200-2000Hz MSE) or "1khz"
    fs: float = 48000.0,
    num_grid_points: int = 300,
) -> OptimizationResult:
    """
    Performs fast, vectorized least-squares optimization to fit parametric EQ bands
    to the target error curve: Error(f) = Target(f) - Measurement(f).
    """
    t_start = time.perf_counter()

    meas_f = np.asarray(meas_freqs, dtype=np.float64)
    meas_s = np.asarray(meas_spls, dtype=np.float64)
    targ_f = np.asarray(target_freqs, dtype=np.float64)
    targ_s = np.asarray(target_spls, dtype=np.float64)

    if len(meas_f) < 5 or len(targ_f) < 5:
        raise ValueError("Insufficient measurement or target curve data points.")

    # 1. Standard Logarithmic Grid (20 Hz - 20,000 Hz)
    f_grid = np.geomspace(20.0, 20000.0, num=num_grid_points)

    # 2. Interpolate measurement and target onto common log grid
    spl_meas = np.interp(f_grid, meas_f, meas_s)
    spl_target = np.interp(f_grid, targ_f, targ_s)

    # 3. Normalization Alignment
    if norm_mode == "1khz":
        targ_1k = float(np.interp(1000.0, f_grid, spl_target))
        meas_1k = float(np.interp(1000.0, f_grid, spl_meas))
        offset = targ_1k - meas_1k
    else:
        # Default: minimize MSE across midband (200 Hz - 2000 Hz)
        mid_mask = (f_grid >= 200.0) & (f_grid <= 2000.0)
        if np.any(mid_mask):
            offset = float(np.mean(spl_target[mid_mask] - spl_meas[mid_mask]))
        else:
            offset = float(np.mean(spl_target - spl_meas))

    spl_meas_aligned = spl_meas + offset
    error_curve = spl_target - spl_meas_aligned

    # 4. Band Allocation & Types
    num_bands = max(1, min(32, int(num_bands)))
    band_types: List[FilterType] = []
    if use_bass_shelf and num_bands > 1:
        band_types.append(FilterType.LOW_SHELF)
        band_types.extend([FilterType.PEAK] * (num_bands - 1))
    elif use_bass_shelf and num_bands == 1:
        band_types.append(FilterType.LOW_SHELF)
    else:
        band_types.extend([FilterType.PEAK] * num_bands)

    # 5. Frequency Weighting
    # Weight mid-frequencies (100 Hz to 6 kHz) more heavily than high treble (>8 kHz)
    # Restrict aggressive corrections above 10 kHz due to 711 coupler insertion resonances
    weights = np.ones_like(f_grid)
    for idx, f in enumerate(f_grid):
        if f < min_freq:
            weights[idx] = 0.05
        elif min_freq <= f < 100.0:
            weights[idx] = 0.85
        elif 100.0 <= f <= 6000.0:
            weights[idx] = 1.6
        elif 6000.0 < f <= max_freq:
            weights[idx] = 1.0
        elif max_freq < f <= 10000.0:
            weights[idx] = 0.35
        else:
            weights[idx] = 0.02

    # 6. Precompute Biquad Complex Constants for Fast Vectorized Evaluation
    w = 2.0 * np.pi * f_grid / fs
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)

    def _eval_bands(params: np.ndarray) -> np.ndarray:
        total = np.zeros(len(f_grid), dtype=np.float64)
        for i in range(num_bands):
            f0 = float(params[i * 3])
            g = float(params[i * 3 + 1])
            q = float(params[i * 3 + 2])
            ftype = band_types[i]
            
            b0, b1, b2, a1, a2 = biquad_coeffs(ftype, f0, g, q, fs)
            num = b0 + b1 * z1 + b2 * z2
            den = 1.0 + a1 * z1 + a2 * z2
            total += 20.0 * np.log10(np.abs(num / den) + 1e-12)
        return total

    # 7. Initial Parameter Estimates & Bounds
    # Log-spaced initial frequencies
    if use_bass_shelf and num_bands > 1:
        shelf_f = 60.0
        peak_freqs = np.geomspace(120.0, max_freq * 0.95, num_bands - 1)
        init_freqs = [shelf_f] + list(peak_freqs)
    elif use_bass_shelf and num_bands == 1:
        init_freqs = [60.0]
    else:
        init_freqs = list(np.geomspace(min_freq * 1.5, max_freq * 0.95, num_bands))

    x0: List[float] = []
    bounds_lower: List[float] = []
    bounds_upper: List[float] = []

    for i in range(num_bands):
        f_init = init_freqs[i]
        # Initial gain from error curve at f_init
        g_raw = float(np.interp(f_init, f_grid, error_curve))
        g_init = np.clip(g_raw, -max_cut_db, max_boost_db)
        q_init = 0.707 if (i == 0 and use_bass_shelf) else 1.414

        x0.extend([f_init, g_init, q_init])

        # Bounds: frequency, gain, Q
        if i == 0 and use_bass_shelf:
            bounds_lower.extend([20.0, -max_cut_db, 0.4])
            bounds_upper.extend([250.0, max_boost_db, 1.8])
        else:
            bounds_lower.extend([min_freq, -max_cut_db, min_q])
            bounds_upper.extend([max_freq, max_boost_db, max_q])

    # 8. Optimization Residuals Function
    def _residuals(params: np.ndarray) -> np.ndarray:
        eq_curve = _eval_bands(params)
        return weights * (eq_curve - error_curve)

    # 9. Solve using Least Squares (Trust Region Reflective with bounds)
    res = least_squares(
        _residuals,
        np.array(x0, dtype=np.float64),
        bounds=(bounds_lower, bounds_upper),
        method="trf",
        ftol=1e-3,
        xtol=1e-3,
        max_nfev=150,
    )

    opt_params = res.x
    final_eq_curve = _eval_bands(opt_params)

    # Calculate RMS Errors across the active optimization range
    active_mask = (f_grid >= min_freq) & (f_grid <= max_freq)
    init_rms = float(np.sqrt(np.mean(error_curve[active_mask] ** 2)))
    final_rms = float(np.sqrt(np.mean((error_curve - final_eq_curve)[active_mask] ** 2)))

    # 10. Extract and Sort Bands
    extracted_bands: List[Tuple[float, float, float, FilterType]] = []
    for i in range(num_bands):
        f0 = float(opt_params[i * 3])
        g = float(opt_params[i * 3 + 1])
        q = float(opt_params[i * 3 + 2])
        ftype = band_types[i]
        extracted_bands.append((f0, g, q, ftype))

    # Sort primarily by frequency (keep bass shelf first if present)
    if use_bass_shelf:
        shelf_band = extracted_bands[0]
        peak_bands = sorted(extracted_bands[1:], key=lambda b: b[0])
        sorted_bands_data = [shelf_band] + peak_bands
    else:
        sorted_bands_data = sorted(extracted_bands, key=lambda b: b[0])

    final_bands: List[EQBand] = []
    for idx, (f0, g, q, ftype) in enumerate(sorted_bands_data):
        # Clean rounding
        rounded_freq = round(f0, 1) if f0 < 100 else round(f0)
        rounded_gain = round(g, 1)
        rounded_q = round(q, 2)
        final_bands.append(
            EQBand(
                band_id=idx + 1,
                enabled=True,
                filter_type=ftype.value,
                freq=float(rounded_freq),
                gain=float(rounded_gain),
                q=float(rounded_q),
            )
        )

    # 11. Preamp Calculation
    # Set digital preamp gain to -max(EQ_Gain) across entire audible spectrum to avoid digital clipping
    peak_gain = float(np.max(final_eq_curve))
    preamp_db = -round(max(0.0, peak_gain), 1)

    t_end = time.perf_counter()
    elapsed_ms = (t_end - t_start) * 1000.0

    return OptimizationResult(
        bands=final_bands,
        preamp_db=preamp_db,
        initial_rms_db=init_rms,
        final_rms_db=final_rms,
        elapsed_ms=elapsed_ms,
        f_grid=f_grid,
        meas_aligned_spl=spl_meas_aligned,
        target_aligned_spl=spl_target,
        error_curve=error_curve,
        eq_response_curve=final_eq_curve,
    )


class AutoEQWorker(QtCore.QThread):
    """
    Non-blocking worker thread for parametric EQ optimization.
    Keeps the PyQt6 user interface fluid and responsive during curve fitting.
    """
    sigOptimized = QtCore.pyqtSignal(object)  # Emits OptimizationResult
    sigFailed = QtCore.pyqtSignal(str)        # Emits error message

    def __init__(
        self,
        meas_freqs: Sequence[float],
        meas_spls: Sequence[float],
        target_freqs: Sequence[float],
        target_spls: Sequence[float],
        num_bands: int = 7,
        use_bass_shelf: bool = True,
        max_boost_db: float = 9.0,
        max_cut_db: float = 12.0,
        max_freq: float = 8500.0,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.meas_freqs = meas_freqs
        self.meas_spls = meas_spls
        self.target_freqs = target_freqs
        self.target_spls = target_spls
        self.num_bands = num_bands
        self.use_bass_shelf = use_bass_shelf
        self.max_boost_db = max_boost_db
        self.max_cut_db = max_cut_db
        self.max_freq = max_freq

    def cancel(self) -> None:
        self.requestInterruption()
        self.terminate()
        self.wait(300)

    def run(self) -> None:
        if self.isInterruptionRequested():
            return
        try:
            result = optimize_parametric_eq(
                meas_freqs=self.meas_freqs,
                meas_spls=self.meas_spls,
                target_freqs=self.target_freqs,
                target_spls=self.target_spls,
                num_bands=self.num_bands,
                use_bass_shelf=self.use_bass_shelf,
                max_boost_db=self.max_boost_db,
                max_cut_db=self.max_cut_db,
                max_freq=self.max_freq,
            )
            if not self.isInterruptionRequested():
                self.sigOptimized.emit(result)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.sigFailed.emit(str(e))
