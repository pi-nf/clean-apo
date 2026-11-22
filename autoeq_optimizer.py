"""
Native AutoEQ Optimization Engine using SciPy.
Calculates high-precision parametric biquad curve fitting (Peak and Low Shelf)
to match raw IEM frequency response measurements to target compensation curves.
Strictly local and offline; non-blocking QThread worker for seamless UI responsiveness.
"""

from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from PyQt6 import QtCore

from biquad import (
    biquad_coeffs,
    compute_composite_response,
    compute_filter_response,
    compute_peak_headroom,
    FilterType,
    EQBand,
)


class _OptimizationCancelled(Exception):
    """Internal sentinel raised to interrupt SciPy optimization promptly."""


def _clean_curve(freqs: np.ndarray, spls: np.ndarray, label: str) -> Tuple[np.ndarray, np.ndarray]:
    if freqs.ndim != 1 or spls.ndim != 1 or len(freqs) != len(spls):
        raise ValueError(f"{label} frequency and SPL arrays must be 1D and equal length.")
    valid = np.isfinite(freqs) & np.isfinite(spls) & (freqs > 0.0)
    f = freqs[valid]
    s = spls[valid]
    if len(f) < 5:
        raise ValueError(f"Insufficient valid {label} curve data points.")
    order = np.argsort(f)
    f = f[order]
    s = s[order]
    uniq_f, inv = np.unique(f, return_inverse=True)
    if len(uniq_f) < len(f):
        uniq_s = np.zeros(len(uniq_f), dtype=np.float64)
        counts = np.bincount(inv)
        np.add.at(uniq_s, inv, s)
        s = uniq_s / counts
        f = uniq_f
    if len(f) < 5:
        raise ValueError(f"Insufficient unique {label} curve data points.")
    return f, s


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
    region_rms_before: Dict[str, float] = field(default_factory=dict)
    region_rms_after: Dict[str, float] = field(default_factory=dict)
    pruned_count: int = 0
    adjusted_target_freqs: Optional[List[float]] = None
    adjusted_target_spl: Optional[List[float]] = None


# Backward/forward-compatible alias
AutoEQResult = OptimizationResult


def apply_target_preferences(
    freqs: Sequence[float] | np.ndarray,
    spl: Sequence[float] | np.ndarray,
    bass_boost_db: float = 0.0,
    treble_tilt_db: float = 0.0,
) -> np.ndarray:
    """
    Applies smooth listener preference curves to a target SPL array:
    - bass_boost_db (-12.0..+12.0): RBJ Low Shelf at 105 Hz, Q=0.707, anchored at 1 kHz.
    - treble_tilt_db (-6.0..+6.0 dB/decade): smooth log10(f / 1000 Hz) tilt pivoted at 1 kHz.
    """
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    spl_arr = np.asarray(spl, dtype=np.float64)
    if freqs_arr.shape != spl_arr.shape:
        raise ValueError("freqs and spl must have identical shapes.")

    adjusted = spl_arr.copy()
    if adjusted.size == 0:
        return adjusted

    bass_val = float(bass_boost_db) if math.isfinite(float(bass_boost_db)) else 0.0
    bass_val = float(np.clip(bass_val, -12.0, 12.0))

    tilt_val = float(treble_tilt_db) if math.isfinite(float(treble_tilt_db)) else 0.0
    tilt_val = float(np.clip(tilt_val, -6.0, 6.0))

    if abs(bass_val) < 1e-9 and abs(tilt_val) < 1e-9:
        return adjusted

    valid_f = np.isfinite(freqs_arr) & (freqs_arr > 0.0)
    if not np.any(valid_f):
        return adjusted

    safe_freqs = np.where(valid_f, freqs_arr, 1000.0)

    if abs(bass_val) >= 1e-9:
        shelf_resp = compute_filter_response(
            FilterType.LOW_SHELF, 105.0, bass_val, 0.707, safe_freqs, fs=48000.0
        )
        shelf_1k = float(
            compute_filter_response(
                FilterType.LOW_SHELF, 105.0, bass_val, 0.707, np.array([1000.0]), fs=48000.0
            )[0]
        )
        adjusted = np.where(valid_f, adjusted + (shelf_resp - shelf_1k), adjusted)

    if abs(tilt_val) >= 1e-9:
        tilt_resp = tilt_val * np.log10(np.maximum(safe_freqs, 1e-6) / 1000.0)
        adjusted = np.where(valid_f, adjusted + tilt_resp, adjusted)

    return adjusted


def _compute_region_rms(
    f_grid: np.ndarray,
    residual: np.ndarray,
    overlap_mask: np.ndarray,
) -> Dict[str, float]:
    """Computes RMS error in dB for bass (20-250 Hz), mid (250-4000 Hz), and treble (4000-20000 Hz)."""
    masks = {
        "bass": overlap_mask & (f_grid >= 20.0) & (f_grid <= 250.0),
        "mid": overlap_mask & (f_grid > 250.0) & (f_grid <= 4000.0),
        "treble": overlap_mask & (f_grid > 4000.0) & (f_grid <= 20000.0),
    }
    result: Dict[str, float] = {}
    for key, mask in masks.items():
        if np.any(mask):
            val = float(np.sqrt(np.mean(residual[mask] ** 2)))
            result[key] = val if math.isfinite(val) else 0.0
        else:
            result[key] = 0.0
    return result


def _classify_rig_guardrail(rig_type: str) -> str:
    """
    Classifies a measurement rig string into an AutoEQ upper-treble guardrail profile:
    - 'wideband': B&K 5128 / GRAS 45CA / GRAS 43AG (trusted up to 10 kHz)
    - 'coupler_711': IEC 711 / clone / KB006x / EARS (conservative above 7.5 kHz)
    - 'default': standard legacy weighting
    """
    r = (rig_type or "").strip().lower()
    if not r:
        return "default"
    if "5128" in r or "45ca" in r or "43ag" in r or "gras" in r:
        return "wideband"
    if (
        "711" in r
        or "60318" in r
        or "clone" in r
        or "kb006" in r
        or "kb50" in r
        or "ears" in r
        or "coupler" in r
    ):
        return "coupler_711"
    return "default"


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
    cancel_check: Optional[Callable[[], bool]] = None,
    bass_boost_db: float = 0.0,
    treble_tilt_db: float = 0.0,
    rig_type: str = "",
    auto_prune: bool = True,
    prune_threshold_db: float = 0.2,
) -> OptimizationResult:
    """
    Performs fast, vectorized least-squares optimization to fit parametric EQ bands
    to the target error curve: Error(f) = Target(f) - Measurement(f).
    """
    t_start = time.perf_counter()

    if cancel_check is not None and cancel_check():
        raise _OptimizationCancelled()

    meas_f_raw = np.asarray(meas_freqs, dtype=np.float64)
    meas_s_raw = np.asarray(meas_spls, dtype=np.float64)
    targ_f_raw = np.asarray(target_freqs, dtype=np.float64)
    targ_s_raw = np.asarray(target_spls, dtype=np.float64)

    meas_f, meas_s = _clean_curve(meas_f_raw, meas_s_raw, "measurement")
    targ_f, targ_s = _clean_curve(targ_f_raw, targ_s_raw, "target")

    # Validate sampling rate and parameter bounds
    fs_val = float(fs)
    if not math.isfinite(fs_val) or fs_val <= 0.0:
        fs_val = 48000.0
    fs = max(100.0, fs_val)

    nyq_safe = min(20000.0, 0.48 * fs)
    min_freq = float(min_freq) if math.isfinite(float(min_freq)) else 20.0
    max_freq = float(max_freq) if math.isfinite(float(max_freq)) else 8500.0
    min_freq = float(np.clip(min_freq, 20.0, max(20.0, nyq_safe - 10.0)))
    max_freq = float(np.clip(max_freq, min(min_freq + 1.0, nyq_safe), nyq_safe))
    if min_freq >= max_freq:
        min_freq = max(20.0, max_freq * 0.5)

    min_q = float(min_q) if math.isfinite(float(min_q)) else 0.5
    max_q = float(max_q) if math.isfinite(float(max_q)) else 6.0
    min_q = float(np.clip(min_q, 0.1, 15.0))
    max_q = float(np.clip(max_q, min_q + 0.05, 20.0))

    max_boost_db = max(0.0, float(max_boost_db) if math.isfinite(float(max_boost_db)) else 9.0)
    max_cut_db = max(0.0, float(max_cut_db) if math.isfinite(float(max_cut_db)) else 12.0)
    gain_lb = -max_cut_db
    gain_ub = max_boost_db
    if gain_ub <= gain_lb:
        gain_ub = gain_lb + 1e-3

    # 1. Standard Logarithmic Grid (20 Hz - 20,000 Hz)
    num_grid_points = max(20, int(num_grid_points))
    f_grid = np.geomspace(20.0, 20000.0, num=num_grid_points)

    # 2. Interpolate measurement and target onto common log grid in log-frequency space
    log_grid = np.log10(f_grid)
    spl_meas = np.interp(log_grid, np.log10(meas_f), meas_s)
    spl_target = np.interp(log_grid, np.log10(targ_f), targ_s)

    # Restrict weighting to overlapping frequency coverage of measurement and target
    overlap_min = max(float(meas_f[0]), float(targ_f[0]))
    overlap_max = min(float(meas_f[-1]), float(targ_f[-1]))
    if overlap_min >= overlap_max:
        raise ValueError("Measurement and target frequency ranges do not overlap.")
    overlap_mask = (f_grid >= overlap_min) & (f_grid <= overlap_max)

    # 3. Normalization Alignment
    if norm_mode == "1khz" and overlap_min <= 1000.0 <= overlap_max:
        targ_1k = float(np.interp(np.log10(1000.0), log_grid, spl_target))
        meas_1k = float(np.interp(np.log10(1000.0), log_grid, spl_meas))
        offset = targ_1k - meas_1k
    else:
        # Default: minimize MSE across midband (200 Hz - 2000 Hz) within overlap
        mid_mask = overlap_mask & (f_grid >= 200.0) & (f_grid <= 2000.0)
        if np.any(mid_mask):
            offset = float(np.mean(spl_target[mid_mask] - spl_meas[mid_mask]))
        elif np.any(overlap_mask):
            offset = float(np.mean(spl_target[overlap_mask] - spl_meas[overlap_mask]))
        else:
            offset = float(np.mean(spl_target - spl_meas))

    spl_meas_aligned = spl_meas + offset

    # 3b. Apply Preference Sliders (Bass Boost Shelf & Treble Tilt) to Target Curve
    bass_boost_val = float(bass_boost_db) if math.isfinite(float(bass_boost_db)) else 0.0
    treble_tilt_val = float(treble_tilt_db) if math.isfinite(float(treble_tilt_db)) else 0.0
    if abs(bass_boost_val) >= 1e-9 or abs(treble_tilt_val) >= 1e-9:
        spl_target = apply_target_preferences(
            f_grid,
            spl_target,
            bass_boost_db=bass_boost_val,
            treble_tilt_db=treble_tilt_val,
        )

    error_curve = spl_target - spl_meas_aligned
    region_rms_before = _compute_region_rms(f_grid, error_curve, overlap_mask)

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

    # 5. Rig-Aware Frequency Weighting & Upper-Treble Guardrails
    rig_profile = _classify_rig_guardrail(rig_type)
    if rig_profile == "wideband":
        # B&K 5128 / GRAS 45CA / 43AG: extend full weighting up to 10 kHz
        conds = [
            (f_grid < overlap_min) | (f_grid > overlap_max),
            f_grid < min_freq,
            (f_grid > max_freq) & (f_grid <= 10000.0),
            f_grid > max_freq,
            f_grid < 100.0,
            f_grid <= 10000.0,
        ]
        choices = [0.0, 0.05, 0.25, 0.02, 0.85, 1.6]
        weights = np.select(conds, choices, default=1.0)
    else:
        # Zero weight outside shared measured/target range; honor max_freq BEFORE 100-6000 Hz weight
        conds = [
            (f_grid < overlap_min) | (f_grid > overlap_max),
            f_grid < min_freq,
            (f_grid > max_freq) & (f_grid <= 10000.0),
            f_grid > max_freq,
            f_grid < 100.0,
            f_grid <= 6000.0,
        ]
        choices = [0.0, 0.05, 0.15, 0.02, 0.85, 1.6]
        weights = np.select(conds, choices, default=1.0)

        if rig_profile == "coupler_711":
            # IEC 711 / clone / KB006x / EARS: taper weight conservatively above 7500 Hz down to 0.25
            t_taper = np.clip(
                (np.log2(f_grid) - np.log2(7500.0)) / (np.log2(10000.0) - np.log2(7500.0)),
                0.0,
                1.0,
            )
            taper_w = 1.0 - 0.75 * t_taper
            in_treble_guard = overlap_mask & (f_grid > 7500.0) & (f_grid <= max_freq)
            weights = np.where(in_treble_guard, np.minimum(weights, taper_w), weights)

    # Upper-treble gain/Q caps for 711-class couplers to avoid chasing 8 kHz pinna/coupler notches
    coupler_treble_q_cap = max(min_q, min(max_q, 3.5))
    coupler_treble_gain_ub = min(gain_ub, 6.0)
    coupler_treble_gain_lb = max(gain_lb, -8.0)
    if coupler_treble_gain_lb > coupler_treble_gain_ub:
        coupler_treble_gain_lb = gain_lb
        coupler_treble_gain_ub = gain_ub

    # 6. Precompute Biquad Complex Constants for Fast Vectorized Evaluation
    w = 2.0 * np.pi * f_grid / fs
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)

    def _eval_bands(params: np.ndarray) -> np.ndarray:
        total = np.zeros(len(f_grid), dtype=np.float64)
        for i in range(num_bands):
            f0 = float(2.0 ** params[i * 3])
            g = float(params[i * 3 + 1])
            q = float(params[i * 3 + 2])
            ftype = band_types[i]

            b0, b1, b2, a1, a2 = biquad_coeffs(ftype, f0, g, q, fs)
            num = b0 + b1 * z1 + b2 * z2
            den = 1.0 + a1 * z1 + a2 * z2
            mag = np.abs(num) / np.maximum(np.abs(den), 1e-30)
            total += 20.0 * np.log10(np.maximum(mag, 1e-12))
        return total

    # 7. Initial Parameter Estimates & Bounds (log2 frequency parameterization for balanced TRF scaling)
    num_peak_bands = max(0, num_bands - 1) if use_bass_shelf else num_bands
    selected_peak_freqs: List[float] = []

    if num_peak_bands > 0:
        abs_err = np.abs(error_curve)
        search_f_min = max(min_freq, overlap_min, 100.0 if use_bass_shelf else min_freq)
        if rig_profile == "wideband":
            search_cap = 10000.0
        elif rig_profile == "coupler_711":
            search_cap = 7500.0
        else:
            search_cap = 8500.0
        search_f_max = min(max_freq, overlap_max, search_cap if max_freq >= 100.0 else max_freq)
        if search_f_min >= search_f_max:
            search_f_min = min_freq
            search_f_max = max_freq

        # 1. Identify dominant resonance peaks and dips within valid range
        detected_indices, _ = find_peaks(abs_err, distance=4)
        valid_peaks = [idx for idx in detected_indices if search_f_min <= f_grid[idx] <= search_f_max]
        valid_peaks.sort(key=lambda idx: abs_err[idx], reverse=True)

        # 2. Seed peak filter initial frequencies on detected peaks first
        for idx in valid_peaks:
            cand_f = float(f_grid[idx])
            if all(abs(np.log2(cand_f / sf)) >= 0.2 for sf in selected_peak_freqs):
                selected_peak_freqs.append(cand_f)
                if len(selected_peak_freqs) >= num_peak_bands:
                    break

        # 3. Fill any remaining unallocated bands with logarithmic spacing across uncovered intervals
        if len(selected_peak_freqs) < num_peak_bands:
            f_low = float(np.clip(120.0 if use_bass_shelf else min_freq * 1.1, min_freq, max_freq))
            f_high = float(np.clip(max_freq * 0.95, f_low, max_freq))
            if f_high <= f_low * 1.001:
                f_low = min_freq
                f_high = max_freq
            boundaries = sorted(set([f_low, f_high] + selected_peak_freqs))
            if len(boundaries) < 2:
                needed = num_peak_bands - len(selected_peak_freqs)
                selected_peak_freqs.extend(
                    float(f) for f in np.geomspace(min_freq, max_freq, needed + 2)[1:-1]
                )
            else:
                while len(selected_peak_freqs) < num_peak_bands:
                    best_idx = 0
                    max_ratio = 0.0
                    for i in range(len(boundaries) - 1):
                        ratio = boundaries[i + 1] / boundaries[i]
                        if ratio > max_ratio:
                            max_ratio = ratio
                            best_idx = i
                    mid = float(np.sqrt(boundaries[best_idx] * boundaries[best_idx + 1]))
                    selected_peak_freqs.append(mid)
                    boundaries.append(mid)
                    boundaries.sort()

    shelf_lb_f = 20.0
    shelf_ub_f = max(shelf_lb_f + 1.0, min(250.0, max_freq))
    shelf_f = float(np.clip(60.0, shelf_lb_f, shelf_ub_f))

    if use_bass_shelf and num_bands > 1:
        init_freqs = [shelf_f] + sorted(selected_peak_freqs)
    elif use_bass_shelf and num_bands == 1:
        init_freqs = [shelf_f]
    else:
        init_freqs = sorted(selected_peak_freqs)

    x0: List[float] = []
    bounds_lower: List[float] = []
    bounds_upper: List[float] = []

    for i in range(num_bands):
        is_shelf = (i == 0 and use_bass_shelf)
        f_lb = shelf_lb_f if is_shelf else min_freq
        f_ub = shelf_ub_f if is_shelf else max_freq
        q_lb = 0.4 if is_shelf else min_q
        q_ub = 1.8 if is_shelf else max_q

        f_init = float(np.clip(init_freqs[i], f_lb, f_ub))
        g_raw = float(np.interp(np.log10(f_init), log_grid, error_curve))
        g_init = float(np.clip(g_raw, gain_lb, gain_ub))
        q_default = 0.707 if is_shelf else 1.414
        q_init = float(np.clip(q_default, q_lb, q_ub))

        x0.extend([math.log2(f_init), g_init, q_init])
        bounds_lower.extend([math.log2(f_lb), gain_lb, q_lb])
        bounds_upper.extend([math.log2(f_ub), gain_ub, q_ub])

    # 8. Optimization Residuals Function
    log2_7500 = math.log2(7500.0)

    def _residuals(params: np.ndarray) -> np.ndarray:
        if cancel_check is not None and cancel_check():
            raise _OptimizationCancelled()
        eq_curve = _eval_bands(params)
        base_res = weights * (eq_curve - error_curve)
        if rig_profile != "coupler_711":
            return base_res

        # Soft guardrail penalties above 7500 Hz for 711-class couplers
        penalties = np.zeros(num_bands * 2, dtype=np.float64)
        for i in range(num_bands):
            if band_types[i] != FilterType.PEAK:
                continue
            log2_f = float(params[i * 3])
            if log2_f <= log2_7500:
                continue
            # Smooth activation over ~0.1 octave above 7500 Hz
            act = min(1.0, (log2_f - log2_7500) / 0.1)
            g = float(params[i * 3 + 1])
            q = float(params[i * 3 + 2])
            if g > coupler_treble_gain_ub:
                penalties[i * 2] = 6.0 * act * (g - coupler_treble_gain_ub)
            elif g < coupler_treble_gain_lb:
                penalties[i * 2] = 6.0 * act * (coupler_treble_gain_lb - g)
            if q > coupler_treble_q_cap:
                penalties[i * 2 + 1] = 8.0 * act * (q - coupler_treble_q_cap)
        return np.concatenate([base_res, penalties])

    # 9. Solve using Least Squares (Trust Region Reflective with Jacobian scaling)
    res = least_squares(
        _residuals,
        np.array(x0, dtype=np.float64),
        bounds=(bounds_lower, bounds_upper),
        method="trf",
        x_scale="jac",
        ftol=1e-3,
        xtol=1e-3,
        max_nfev=150,
    )

    opt_params = res.x

    # Calculate Initial RMS Error across active optimization range within overlap
    active_mask = overlap_mask & (f_grid >= min_freq) & (f_grid <= max_freq)
    if not np.any(active_mask):
        active_mask = overlap_mask if np.any(overlap_mask) else np.ones(len(f_grid), dtype=bool)
    init_rms = float(np.sqrt(np.mean(error_curve[active_mask] ** 2)))

    # 10. Extract and Sort Bands
    extracted_bands: List[Tuple[float, float, float, FilterType]] = []
    for i in range(num_bands):
        f0 = float(2.0 ** opt_params[i * 3])
        g = float(opt_params[i * 3 + 1])
        q = float(opt_params[i * 3 + 2])
        ftype = band_types[i]
        if rig_profile == "coupler_711" and ftype == FilterType.PEAK and f0 > 7500.0:
            g = float(np.clip(g, coupler_treble_gain_lb, coupler_treble_gain_ub))
            q = float(min(q, coupler_treble_q_cap))
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
        rounded_freq = float(round(f0, 1) if f0 < 100 else round(f0))
        rounded_gain = float(round(g, 1))
        if abs(rounded_gain) < 1e-9:
            rounded_gain = 0.0
        rounded_q = float(round(q, 2))
        final_bands.append(
            EQBand(
                band_id=idx + 1,
                enabled=True,
                filter_type=ftype.value,
                freq=rounded_freq,
                gain=rounded_gain,
                q=rounded_q,
            )
        )

    # 10b. Auto-Prune Near-Zero Bands
    prune_thresh = (
        max(0.0, float(prune_threshold_db))
        if math.isfinite(float(prune_threshold_db))
        else 0.2
    )
    pruned_count = 0
    if auto_prune and prune_thresh > 0.0 and len(final_bands) > 0:
        total_before_prune = len(final_bands)
        kept_bands: List[EQBand] = [
            b
            for b in final_bands
            if FilterType.from_str(b.filter_type) in (FilterType.HIGH_PASS, FilterType.LOW_PASS)
            or abs(b.gain) >= prune_thresh
        ]
        if not kept_bands:
            kept_bands = [
                EQBand(
                    band_id=1,
                    enabled=True,
                    filter_type=FilterType.PEAK.value,
                    freq=1000.0,
                    gain=0.0,
                    q=1.414,
                )
            ]
        for new_idx, band in enumerate(kept_bands, start=1):
            band.band_id = new_idx
        pruned_count = max(0, total_before_prune - len(kept_bands))
        final_bands = kept_bands

    # 11. Preamp Calculation & Post-Prune Metrics
    # Evaluate composite response of final_bands (including band center freqs) to prevent clipping
    final_eval = compute_composite_response(final_bands, f_grid, preamp_db=0.0, fs=fs)
    _, preamp_db = compute_peak_headroom(final_bands, preamp_db=0.0, fs=fs)

    # Calculate final RMS & per-region RMS with the exact final bands
    final_residual = error_curve - final_eval
    final_rms = float(np.sqrt(np.mean(final_residual[active_mask] ** 2)))
    region_rms_after = _compute_region_rms(f_grid, final_residual, overlap_mask)

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
        eq_response_curve=final_eval,
        region_rms_before=region_rms_before,
        region_rms_after=region_rms_after,
        pruned_count=pruned_count,
        adjusted_target_freqs=f_grid.tolist(),
        adjusted_target_spl=spl_target.tolist(),
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
        bass_boost_db: float = 0.0,
        treble_tilt_db: float = 0.0,
        rig_type: str = "",
        auto_prune: bool = True,
        prune_threshold_db: float = 0.2,
    ):
        super().__init__(parent)
        self.meas_freqs = np.asarray(meas_freqs, dtype=np.float64).copy()
        self.meas_spls = np.asarray(meas_spls, dtype=np.float64).copy()
        self.target_freqs = np.asarray(target_freqs, dtype=np.float64).copy()
        self.target_spls = np.asarray(target_spls, dtype=np.float64).copy()
        self.num_bands = num_bands
        self.use_bass_shelf = use_bass_shelf
        self.max_boost_db = max_boost_db
        self.max_cut_db = max_cut_db
        self.max_freq = max_freq
        self.bass_boost_db = bass_boost_db
        self.treble_tilt_db = treble_tilt_db
        self.rig_type = rig_type
        self.auto_prune = auto_prune
        self.prune_threshold_db = prune_threshold_db
        self._cancel_event = threading.Event()

    def _is_cancelled(self) -> bool:
        return self._cancel_event.is_set() or self.isInterruptionRequested()

    def cancel(self) -> None:
        self._cancel_event.set()
        self.requestInterruption()
        if self.isRunning():
            self.wait(1500)

    def run(self) -> None:
        if self._is_cancelled():
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
                cancel_check=self._is_cancelled,
                bass_boost_db=self.bass_boost_db,
                treble_tilt_db=self.treble_tilt_db,
                rig_type=self.rig_type,
                auto_prune=self.auto_prune,
                prune_threshold_db=self.prune_threshold_db,
            )
            if not self._is_cancelled():
                self.sigOptimized.emit(result)
        except _OptimizationCancelled:
            return
        except Exception as e:
            if not self._is_cancelled():
                self.sigFailed.emit(str(e))
