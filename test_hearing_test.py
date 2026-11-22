"""
Unit & regression tests for Clean APO DSP, audio stream, AutoEQ, and hearing test modules.
Verifies ISO 226:2003 equal-loudness math, real-time audio synthesis (sine, white/pink noise,
per-sample ramping, output device fallback), session threshold & composite anti-clipping logic,
HearingTestDialog UI/plot/signals (with sounddevice mocked), biquad edge cases & caching,
AutoEQ optimizer bounds/weighting/cancellation, and AudioSpectrumWorker RTA/AGC/NaN guards.
"""

import math
import queue
import threading
import unittest
from unittest import mock

import numpy as np

from biquad import (
    EQBand,
    FilterType,
    biquad_coeffs,
    compute_composite_response,
    compute_filter_response,
    compute_peak_headroom,
)
from autoeq_optimizer import (
    AutoEQResult,
    AutoEQWorker,
    OptimizationResult,
    _OptimizationCancelled,
    apply_target_preferences,
    optimize_parametric_eq,
)
from audio_stream import AudioSpectrumWorker
import hearing_test
from hearing_test import (
    DETAILED_15_BANDS,
    FULL_31_BANDS,
    ISO226,
    STANDARD_10_BANDS,
    HearingAudioEngine,
    HearingTestDialog,
    HearingTestSession,
)


class _DummyOutputStream:
    """Mock sounddevice.OutputStream so tests never open hardware audio output."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class TestISO226(unittest.TestCase):
    def test_evaluate_reference_points(self):
        # By definition of phon, 1000 Hz at 60 phon must evaluate to ~60.0 dB SPL
        freqs = np.array([50.0, 100.0, 1000.0, 4000.0])
        spl = ISO226.evaluate(freqs, phon=60.0)
        self.assertEqual(len(spl), 4)
        self.assertTrue(np.all(np.isfinite(spl)))
        self.assertAlmostEqual(float(spl[2]), 60.0, places=1)
        # ISO 226:2003 at 50 Hz, 60 phon is ~90.0 dB SPL
        self.assertAlmostEqual(float(spl[0]), 90.0, delta=0.6)
        # Low frequencies require higher SPL than 1000 Hz for equal loudness
        self.assertGreater(spl[1], spl[2])

    def test_normalized_delta(self):
        delta = ISO226.normalized_delta([1000.0], phon=60.0)
        self.assertAlmostEqual(float(delta[0]), 0.0, places=4)

        delta_50 = ISO226.normalized_delta([50.0], phon=60.0)
        self.assertAlmostEqual(float(delta_50[0]), 30.0, delta=0.6)

    def test_high_frequency_clamping_above_12500hz(self):
        spl = ISO226.evaluate([12500.0, 16000.0, 20000.0], phon=60.0)
        self.assertTrue(np.all(np.isfinite(spl)))
        self.assertAlmostEqual(float(spl[1]), float(spl[0]), places=6)
        self.assertAlmostEqual(float(spl[2]), float(spl[0]), places=6)


class TestHearingAudioEngine(unittest.TestCase):
    def _make_bare_engine(self, sample_rate: int = 48000) -> HearingAudioEngine:
        engine = HearingAudioEngine.__new__(HearingAudioEngine)
        engine.sample_rate = sample_rate
        engine.device_index = None
        engine.device_name = None
        engine._output_channels = 2
        engine._stream = None
        engine.audio_error = None
        engine._freq = 1000.0
        engine._target_sine_gain = 0.1
        engine._current_sine_gain = 0.1
        engine._target_noise_gain = 0.05
        engine._current_noise_gain = 0.05
        engine._sine_enabled = True
        engine._noise_enabled = True
        engine._noise_type = "white"
        engine._tone_mode = "continuous"
        engine._channel = "both"
        engine._phase = 0.0
        engine._pulse_time = 0.0
        engine._pulse_period = 0.5
        engine._pulse_ramp = 0.025
        engine._pink_zi = [np.zeros(1, dtype=np.float64) for _ in range(7)]
        engine._running = False
        engine._lock = threading.Lock()
        return engine

    def test_audio_callback_dsp(self):
        engine = self._make_bare_engine()
        outdata = np.zeros((1024, 2), dtype=np.float32)
        engine._audio_callback(outdata, 1024, None, None)

        self.assertTrue(np.all(np.isfinite(outdata)))
        self.assertTrue(np.all(outdata <= 0.98))
        self.assertTrue(np.all(outdata >= -0.98))
        self.assertGreater(np.max(np.abs(outdata)), 0.0)

    def test_pulsed_tone_and_channel_routing(self):
        engine = self._make_bare_engine()
        engine._freq = 500.0
        engine._target_sine_gain = 0.2
        engine._current_sine_gain = 0.2
        engine._target_noise_gain = 0.0
        engine._current_noise_gain = 0.0
        engine._noise_enabled = False
        engine._tone_mode = "pulsed"
        engine._channel = "left"

        outdata = np.zeros((1024, 2), dtype=np.float32)
        engine._audio_callback(outdata, 1024, None, None)
        self.assertGreater(np.max(np.abs(outdata[:, 0])), 0.0)
        self.assertEqual(np.max(np.abs(outdata[:, 1])), 0.0)

        # Mono output buffer must also succeed without IndexError
        mono = np.zeros((256, 1), dtype=np.float32)
        engine._channel = "right"
        engine._audio_callback(mono, 256, None, None)
        self.assertTrue(np.all(np.isfinite(mono)))
        self.assertGreater(np.max(np.abs(mono[:, 0])), 0.0)

    def test_per_sample_gain_smoothing_prevents_block_step(self):
        engine = self._make_bare_engine()
        engine._freq = 1000.0
        engine._phase = math.pi / 2.0  # cos peak at sample 0
        engine._current_sine_gain = 0.0
        engine._target_sine_gain = 0.5
        engine._noise_enabled = False
        engine._current_noise_gain = 0.0
        engine._target_noise_gain = 0.0
        engine._tone_mode = "continuous"

        outdata = np.zeros((1024, 2), dtype=np.float32)
        engine._audio_callback(outdata, 1024, None, None)

        # Sample 0 should start near 0 (continuous ramp), far smaller than block-end amplitude
        self.assertLess(abs(float(outdata[0, 0])), 0.01)
        self.assertGreater(float(np.max(np.abs(outdata[-48:, 0]))), 0.30)

    def test_pink_noise_spectrum_rms_and_nyquist_clamp(self):
        engine = self._make_bare_engine(sample_rate=48000)
        engine._sine_enabled = False
        engine._current_sine_gain = 0.0
        engine._target_sine_gain = 0.0
        engine._noise_enabled = True
        engine._current_noise_gain = 0.5
        engine._target_noise_gain = 0.5

        np.random.seed(7)
        engine._noise_type = "white"
        white_blocks = []
        for _ in range(80):
            buf = np.zeros((1024, 2), dtype=np.float32)
            engine._audio_callback(buf, 1024, None, None)
            white_blocks.append(buf[:, 0].copy())
        white_sig = np.concatenate(white_blocks)

        np.random.seed(7)
        engine._noise_type = "pink"
        pink_blocks = []
        for _ in range(80):
            buf = np.zeros((1024, 2), dtype=np.float32)
            engine._audio_callback(buf, 1024, None, None)
            pink_blocks.append(buf[:, 0].copy())
        pink_sig = np.concatenate(pink_blocks)[16384:]  # skip startup transient

        self.assertTrue(np.all(np.isfinite(pink_sig)))
        rms_w = float(np.sqrt(np.mean(white_sig ** 2)))
        rms_p = float(np.sqrt(np.mean(pink_sig ** 2)))
        rms_diff_db = abs(20.0 * math.log10(max(rms_p, 1e-9) / max(rms_w, 1e-9)))
        self.assertLess(rms_diff_db, 1.5)

        # Verify ~-10 dB/decade (1/f) spectral slope from 1 kHz to 10 kHz
        fp = np.fft.rfftfreq(len(pink_sig), 1.0 / 48000.0)
        Sp = np.abs(np.fft.rfft(pink_sig)) ** 2
        p_1k = float(np.mean(Sp[(fp >= 850) & (fp <= 1150)]))
        p_10k = float(np.mean(Sp[(fp >= 8500) & (fp <= 11500)]))
        slope_db = 10.0 * math.log10(p_10k / p_1k)
        self.assertAlmostEqual(slope_db, -10.0, delta=1.5)

        # Nyquist clamping on low sample-rate devices (e.g., 16 kHz Bluetooth HFP)
        engine.sample_rate = 16000
        engine.set_frequency(16000.0)
        self.assertLessEqual(engine._freq, 0.48 * 16000.0)

    def test_output_device_validation_and_fallback(self):
        fake_devs = [
            {"name": "Loopback Input Only", "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Speakers (USB DAC)", "max_output_channels": 2, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "Speakers (USB DAC)", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 1},
        ]
        fake_hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]

        def _query_devices(dev=None, kind=None):
            if dev is None and kind is None:
                return fake_devs
            if dev is None and kind == "output":
                return fake_devs[2]
            return fake_devs[int(dev)]

        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(hearing_test.sd, "query_devices", side_effect=_query_devices),
            mock.patch.object(hearing_test.sd, "query_hostapis", return_value=fake_hostapis),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=lambda **kw: _DummyOutputStream(**kw)),
        ):
            # 1. Input-only device_index=0 must be ignored -> falls back to default output (no device kwarg)
            eng1 = HearingAudioEngine(device_index=0)
            self.assertTrue(eng1.is_running)
            self.assertIsNone(eng1.audio_error)
            self.assertNotIn("device", eng1._stream.kwargs)
            eng1.stop()

            # 2. device_name matching prefers WASAPI output (index 2) over MME (index 1)
            eng2 = HearingAudioEngine(device_index=0, device_name="Speakers (USB DAC)")
            self.assertTrue(eng2.is_running)
            self.assertEqual(eng2._stream.kwargs.get("device"), 2)
            self.assertEqual(eng2.sample_rate, 48000)
            eng2.stop()

        # 3. Total stream open failure sets audio_error cleanly
        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(hearing_test.sd, "query_devices", side_effect=_query_devices),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=RuntimeError("No device")),
        ):
            eng3 = HearingAudioEngine(device_index=2)
            self.assertFalse(eng3.is_running)
            self.assertIsNotNone(eng3.audio_error)


class TestHearingTestSession(unittest.TestCase):
    def test_session_initialization_and_stepper(self):
        session = HearingTestSession(frequencies=list(STANDARD_10_BANDS))
        self.assertEqual(len(session.frequencies), 10)
        # Starts safely anchored at 1,000 Hz (index 5 in STANDARD_10_BANDS)
        self.assertEqual(session.current_index, 5)
        self.assertEqual(session.current_freq, 1000.0)
        self.assertAlmostEqual(session.q_factor, 1.414, places=3)

        self.assertTrue(session.next_frequency())
        self.assertEqual(session.current_index, 6)
        self.assertEqual(session.current_freq, 2000.0)

        self.assertTrue(session.prev_frequency())
        self.assertEqual(session.current_index, 5)
        self.assertEqual(session.current_freq, 1000.0)

        # Suite Q scaling & 1 kHz anchor across suites
        session.init_frequency_suite(DETAILED_15_BANDS)
        self.assertAlmostEqual(session.q_factor, 2.1, places=2)
        self.assertEqual(session.current_freq, 1000.0)
        self.assertEqual(session.current_index, DETAILED_15_BANDS.index(1000.0))

        session.init_frequency_suite(FULL_31_BANDS)
        self.assertAlmostEqual(session.q_factor, 4.318, places=3)
        self.assertEqual(session.current_freq, 1000.0)
        self.assertEqual(session.current_index, FULL_31_BANDS.index(1000.0))

        # Reset also re-anchors at 1,000 Hz
        session.current_index = 0
        session.reset_all()
        self.assertEqual(session.current_freq, 1000.0)

        # Empty frequency list must not crash
        empty = HearingTestSession(frequencies=[])
        preamp, bands = empty.compute_compensation()
        self.assertEqual(preamp, 0.0)
        self.assertEqual(bands, [])

    def test_set_threshold_and_reset_clears_all_suites(self):
        session = HearingTestSession(frequencies=list(FULL_31_BANDS))
        self.assertFalse(session.is_tested(freq=20.0))
        session.set_threshold(-42.5, ch="both", freq=20.0)
        self.assertTrue(session.is_tested(freq=20.0))
        self.assertEqual(session.get_threshold(ch="both", freq=20.0), -42.5)

        # Switch to 10-band, reset, and switch back to 31-band: 20 Hz must also be cleared
        session.init_frequency_suite(STANDARD_10_BANDS)
        session.reset_all()
        session.init_frequency_suite(FULL_31_BANDS)
        self.assertFalse(session.is_tested(freq=20.0))
        self.assertEqual(session.get_threshold(freq=20.0), -30.0)

    def test_direct_compensation_calculation(self):
        session = HearingTestSession(frequencies=[500.0, 1000.0, 2000.0])
        session.set_threshold(-30.0, freq=1000.0)
        session.set_threshold(-20.0, freq=2000.0)
        session.set_threshold(-35.0, freq=500.0)

        session.compensation_mode = "direct"
        session.strength = 1.0
        session.max_boost_db = 12.0

        preamp, bands = session.compute_compensation()

        gains = {b.freq: b.gain for b in bands}
        self.assertAlmostEqual(gains[1000.0], 0.0, places=1)
        self.assertAlmostEqual(gains[2000.0], 10.0, places=1)
        self.assertAlmostEqual(gains[500.0], -5.0, places=1)

        # Composite response with preamp must never exceed 0 dBFS
        eval_f = np.geomspace(20.0, 20000.0, 1000)
        comp = compute_composite_response(bands, eval_f, preamp_db=preamp)
        self.assertLessEqual(float(np.max(comp)), 0.01)
        self.assertGreaterEqual(float(np.max(comp)), -0.15)

    def test_strength_and_max_boost_clamp(self):
        session = HearingTestSession(frequencies=[1000.0, 4000.0])
        session.set_threshold(-40.0, freq=1000.0)
        session.set_threshold(-10.0, freq=4000.0)

        session.max_boost_db = 8.0
        session.strength = 1.0
        preamp, bands = session.compute_compensation()

        gains = {b.freq: b.gain for b in bands}
        self.assertEqual(gains[4000.0], 8.0)
        self.assertEqual(preamp, -8.1)

        session.strength = 0.5
        preamp_50, bands_50 = session.compute_compensation()
        gains_50 = {b.freq: b.gain for b in bands_50}
        self.assertEqual(gains_50[4000.0], 8.0)
        self.assertEqual(preamp_50, -8.1)

        session.set_threshold(-30.0, freq=4000.0)
        preamp_half, bands_half = session.compute_compensation()
        gains_half = {b.freq: b.gain for b in bands_half}
        self.assertAlmostEqual(gains_half[4000.0], 5.0, places=1)
        self.assertAlmostEqual(preamp_half, -5.1, places=1)

    def test_iso226_compensation_mode(self):
        session = HearingTestSession(frequencies=list(STANDARD_10_BANDS))
        session.compensation_mode = "iso226"

        # Untested session stays flat (0 dB)
        preamp_empty, bands_empty = session.compute_compensation()
        self.assertEqual(len(bands_empty), len(STANDARD_10_BANDS))
        self.assertEqual(preamp_empty, 0.0)
        self.assertTrue(all(b.gain == 0.0 for b in bands_empty))

        # Tested equal-loudness thresholds matching ISO 226 should cancel out to ~0 dB EQ
        iso_deltas = ISO226.normalized_delta(STANDARD_10_BANDS, phon=60.0)
        for f, d in zip(STANDARD_10_BANDS, iso_deltas):
            session.set_threshold(-60.0 + float(d), freq=f)
        preamp, bands = session.compute_compensation()
        for b in bands:
            self.assertIsInstance(b, EQBand)
            self.assertEqual(b.filter_type, FilterType.PEAK.value)
            self.assertAlmostEqual(b.gain, 0.0, delta=0.2)
        self.assertAlmostEqual(preamp, 0.0, delta=0.2)

    def test_untested_bands_and_partial_both_fallback(self):
        session = HearingTestSession(frequencies=[31.5, 63.0, 1000.0, 4000.0])
        # User starts at 31.5 Hz and 63 Hz before ever testing 1000 Hz
        session.set_threshold(-55.0, ch="both", freq=31.5)
        session.set_threshold(-51.0, ch="both", freq=63.0)
        _, bands = session.compute_compensation()
        gains = {b.freq: b.gain for b in bands}
        # Baseline is median([-55, -51]) = -53 -> -2 dB / +2 dB, NOT -10 dB against untested -30 dB!
        self.assertAlmostEqual(gains[31.5], -2.0, places=1)
        self.assertAlmostEqual(gains[63.0], 2.0, places=1)
        # Untested bands stay flat at 0.0 dB
        self.assertEqual(gains[1000.0], 0.0)
        self.assertEqual(gains[4000.0], 0.0)

        # In 'both' mode with ONLY Left ear tested, must not average with untested Right (-30 dB)
        session.reset_all()
        session.set_threshold(-50.0, ch="left", freq=1000.0)
        session.set_threshold(-46.0, ch="left", freq=4000.0)
        _, both_bands = session.compute_compensation(target_channel="both")
        both_gains = {b.freq: b.gain for b in both_bands}
        self.assertAlmostEqual(both_gains[1000.0], 0.0, places=1)
        self.assertAlmostEqual(both_gains[4000.0], 4.0, places=1)
        self.assertEqual(both_gains[31.5], 0.0)

    def test_overlapping_multiband_composite_anti_clipping(self):
        session = HearingTestSession(frequencies=list(FULL_31_BANDS))
        for f in FULL_31_BANDS:
            session.set_threshold(-40.0, freq=f)
        for f in (4000.0, 5000.0, 6300.0):
            session.set_threshold(-32.0, freq=f)  # +8 dB adjacent peaks

        preamp, bands = session.compute_compensation()
        eval_f = np.geomspace(20.0, 20000.0, 1500)
        comp = compute_composite_response(bands, eval_f, preamp_db=preamp)
        self.assertLessEqual(float(np.max(comp)), 0.01)
        self.assertLess(preamp, -8.0)  # Must account for adjacent-band summation

    def test_per_ear_lr_channel_generation_and_composite_headroom(self):
        session = HearingTestSession(frequencies=[500.0, 1000.0, 4000.0])
        self.assertFalse(session.has_independent_lr_data())

        # Record asymmetric L/R thresholds
        session.set_threshold(-40.0, ch="left", freq=1000.0)
        session.set_threshold(-33.0, ch="left", freq=4000.0)   # Left needs +7.0 dB at 4 kHz
        self.assertFalse(session.has_independent_lr_data())

        session.set_threshold(-40.0, ch="right", freq=1000.0)
        session.set_threshold(-36.0, ch="right", freq=500.0)   # Right needs +4.0 dB at 500 Hz
        self.assertTrue(session.has_independent_lr_data())

        preamp, bands = session.compute_compensation(per_ear=True)
        left_bands = [b for b in bands if b.channel == "L"]
        right_bands = [b for b in bands if b.channel == "R"]
        self.assertEqual(len(left_bands), 3)
        self.assertEqual(len(right_bands), 3)
        self.assertEqual([b.band_id for b in bands], list(range(1, 7)))

        left_gains = {b.freq: b.gain for b in left_bands}
        right_gains = {b.freq: b.gain for b in right_bands}
        self.assertAlmostEqual(left_gains[4000.0], 7.0, places=1)
        self.assertAlmostEqual(left_gains[500.0], 0.0, places=1)
        self.assertAlmostEqual(right_gains[500.0], 4.0, places=1)
        self.assertAlmostEqual(right_gains[4000.0], 0.0, places=1)

        # Preamp must match worst-case ear (+7.0 dB Left -> -7.1 dB) so neither ear clips
        eval_f = np.geomspace(20.0, 20000.0, 1000)
        left_resp = compute_composite_response(left_bands, eval_f, preamp_db=preamp)
        right_resp = compute_composite_response(right_bands, eval_f, preamp_db=preamp)
        self.assertLessEqual(float(np.max(left_resp)), 0.0)
        self.assertLessEqual(float(np.max(right_resp)), 0.0)
        self.assertAlmostEqual(preamp, -7.1, places=1)


class TestHearingTestDialog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from PyQt6 import QtWidgets
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    def test_dialog_ui_workflow_plot_and_non_destructive_ab(self):
        mock_dev = {"name": "Mock Speakers", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0}
        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(
                hearing_test.sd,
                "query_devices",
                side_effect=lambda dev=None, kind=None: [mock_dev] if (dev is None and kind is None) else mock_dev,
            ),
            mock.patch.object(hearing_test.sd, "query_hostapis", return_value=[{"name": "Windows WASAPI"}]),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=lambda **kw: _DummyOutputStream(**kw)),
        ):
            dlg = HearingTestDialog(device_index=None, device_name="Mock Speakers")
            try:
                # 1. Check initial state (1 kHz anchor!) & hidden audio error banner
                self.assertEqual(dlg.session.current_index, 5)
                self.assertEqual(dlg.session.current_freq, 1000.0)
                self.assertTrue(dlg.tone_mode_btn.isChecked())
                self.assertTrue(dlg.lbl_audio_warning.isHidden())

                # 2. Verify single-log X-axis: eq_curve & ref_curve fall inside plot viewRange
                dlg.mode_combo.setCurrentIndex(1)  # Enable ISO 226 reference curve
                x_eq, _ = dlg.eq_curve.getData()
                x_ref, _ = dlg.ref_curve.getData()
                x_view_min, x_view_max = dlg.plot_widget.viewRange()[0]
                y_view_min, y_view_max = dlg.plot_widget.viewRange()[1]
                self.assertGreaterEqual(float(np.min(x_eq)), x_view_min)
                self.assertLessEqual(float(np.max(x_eq)), x_view_max)
                self.assertGreaterEqual(float(np.min(x_ref)), x_view_min)
                self.assertLessEqual(float(np.max(x_ref)), x_view_max)
                self.assertLessEqual(y_view_min, -90.0)
                self.assertGreaterEqual(y_view_max, 20.0)

                # 3. Stepper advance / reverse from 1 kHz anchor
                dlg._on_next_clicked()
                self.assertEqual(dlg.session.current_index, 6)
                self.assertEqual(dlg.session.current_freq, 2000.0)
                dlg._on_prev_clicked()
                self.assertEqual(dlg.session.current_index, 5)
                self.assertEqual(dlg.session.current_freq, 1000.0)

                # 4. Volume fine step adjustments
                curr_vol = dlg.session.get_threshold()
                dlg._step_sine_volume(-5.0)
                self.assertAlmostEqual(dlg.session.get_threshold(), curr_vol - 5.0, places=1)

                # 5. Channel switching must sync BOTH channel and sine gain to audio_engine
                dlg.session.set_threshold(-50.0, ch="left", freq=1000.0)
                dlg.session.set_threshold(-20.0, ch="right", freq=1000.0)
                dlg._set_channel("left")
                self.assertEqual(dlg.audio_engine._channel, "left")
                self.assertAlmostEqual(dlg.audio_engine._target_sine_gain, 10.0 ** (-50.0 / 20.0), places=5)
                dlg._set_channel("right")
                self.assertEqual(dlg.audio_engine._channel, "right")
                self.assertAlmostEqual(dlg.audio_engine._target_sine_gain, 10.0 ** (-20.0 / 20.0), places=5)

                # 6. Mute button toggle preserves compact font/padding stylesheet
                dlg.sine_mute_btn.setChecked(False)
                dlg._toggle_sine_mute()
                self.assertIn("font-size: 11px", dlg.sine_mute_btn.styleSheet())
                self.assertIn("padding: 4px 8px", dlg.sine_mute_btn.styleSheet())

                # 7. Non-destructive A/B Compare emits sigPreviewEQ (NOT sigApplyEQ) & sigPreviewEnded on close
                emitted_apply = []
                emitted_preview = []
                preview_ended = []
                dlg.sigApplyEQ.connect(lambda bands, p: emitted_apply.append((bands, p)))
                dlg.sigPreviewEQ.connect(lambda bands, p, byp: emitted_preview.append((bands, p, byp)))
                dlg.sigPreviewEnded.connect(lambda: preview_ended.append(True))

                dlg.ab_compare_btn.setChecked(True)
                dlg._toggle_ab_compare()
                self.assertTrue(dlg._ab_bypassed)
                self.assertEqual(len(emitted_apply), 0)
                self.assertEqual(len(emitted_preview), 1)
                self.assertTrue(emitted_preview[0][2])

                # Apply resets A/B Compare state and emits sigApplyEQ
                dlg._on_apply_clicked()
                self.assertFalse(dlg._ab_bypassed)
                self.assertFalse(dlg.ab_compare_btn.isChecked())
                self.assertEqual(len(emitted_apply), 1)

                # Preview again without applying, then close via done() -> emits sigPreviewEnded & stops audio
                dlg.ab_compare_btn.setChecked(True)
                dlg._toggle_ab_compare()
                dlg.done(0)
                self.assertEqual(len(preview_ended), 1)
                self.assertFalse(dlg.audio_engine.is_running)
            finally:
                dlg.close()

    def test_dialog_shows_audio_error_banner_when_output_fails(self):
        mock_dev = {"name": "Mock Speakers", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0}
        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(hearing_test.sd, "query_devices", return_value=mock_dev),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=RuntimeError("Device busy")),
        ):
            dlg = HearingTestDialog()
            try:
                self.assertFalse(dlg.audio_engine.is_running)
                self.assertFalse(dlg.lbl_audio_warning.isHidden())
                self.assertIn("Device busy", dlg.lbl_audio_warning.text())
            finally:
                dlg.close()

    def test_output_device_listing_live_switching_and_bypass_lifecycle(self):
        fake_devs = [
            {"name": "Input Mic Only", "max_output_channels": 0, "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Speakers (USB DAC)", "max_output_channels": 2, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "Speakers (USB DAC)", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 1},
            {"name": "Headphones (Studio)", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 1},
        ]
        fake_hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]

        def _query_devices(dev=None, kind=None):
            if dev is None and kind is None:
                return fake_devs
            if dev is None and kind == "output":
                return fake_devs[2]
            return fake_devs[int(dev)]

        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(hearing_test.sd, "query_devices", side_effect=_query_devices),
            mock.patch.object(hearing_test.sd, "query_hostapis", return_value=fake_hostapis),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=lambda **kw: _DummyOutputStream(**kw)),
        ):
            # 1. list_output_devices deduplicates by name preferring WASAPI and filters out input-only devices
            listed = HearingAudioEngine.list_output_devices()
            self.assertEqual(listed[0], (None, "System default"))
            self.assertEqual(len(listed), 3)
            self.assertEqual(listed[1], (2, "Speakers (USB DAC)"))
            self.assertEqual(listed[2], (3, "Headphones (Studio)"))

            # 2. Dialog pre-selects matching device_name and emits sigMeasurementBypassRequested on show/close
            dlg = HearingTestDialog(device_index=None, device_name="Headphones (Studio)")
            bypass_events = []
            dlg.sigMeasurementBypassRequested.connect(lambda flag: bypass_events.append(bool(flag)))
            try:
                self.assertEqual(dlg.output_device_combo.currentText(), "Headphones (Studio)")
                self.assertTrue(dlg.bypass_eq_checkbox.isChecked())

                # Show triggers bypass=True
                dlg.show()
                self.app.processEvents()
                self.assertEqual(bypass_events, [True])

                # Switching output device live re-opens stream on the selected device
                dlg.output_device_combo.setCurrentIndex(1)  # Speakers (USB DAC) -> idx 2
                self.assertTrue(dlg.audio_engine.is_running)
                self.assertEqual(dlg.audio_engine._stream.kwargs.get("device"), 2)
                self.assertTrue(dlg.lbl_audio_warning.isHidden())

                # Unchecking bypass checkbox releases bypass immediately
                dlg.bypass_eq_checkbox.setChecked(False)
                self.assertEqual(bypass_events, [True, False])

                # Re-checking re-activates bypass, and A/B Compare EQ ON releases it so EQ is audible
                dlg.bypass_eq_checkbox.setChecked(True)
                self.assertEqual(bypass_events, [True, False, True])
                dlg.ab_compare_btn.setChecked(False)
                dlg._toggle_ab_compare()
                self.assertEqual(bypass_events, [True, False, True, False])

                # Re-enable bypass then close via done() -> emits False exactly once
                dlg.bypass_eq_checkbox.setChecked(False)
                dlg.bypass_eq_checkbox.setChecked(True)
                dlg.done(0)
                dlg.close()
                self.assertEqual(bypass_events[-2:], [True, False])
            finally:
                dlg.close()

    def test_dialog_per_ear_auto_select_and_apply_channels(self):
        mock_dev = {"name": "Mock Speakers", "max_output_channels": 2, "default_samplerate": 48000.0, "hostapi": 0}
        with (
            mock.patch.object(hearing_test, "HAVE_SOUNDDEVICE", True),
            mock.patch.object(
                hearing_test.sd,
                "query_devices",
                side_effect=lambda dev=None, kind=None: [mock_dev] if (dev is None and kind is None) else mock_dev,
            ),
            mock.patch.object(hearing_test.sd, "query_hostapis", return_value=[{"name": "Windows WASAPI"}]),
            mock.patch.object(hearing_test.sd, "OutputStream", side_effect=lambda **kw: _DummyOutputStream(**kw)),
        ):
            dlg = HearingTestDialog(device_index=None, device_name="Mock Speakers")
            applied = []
            dlg.sigApplyEQ.connect(lambda bands, preamp: applied.append((bands, preamp)))
            try:
                self.assertEqual(dlg.export_mode_combo.currentIndex(), 0)
                self.assertEqual(dlg.session.export_channel_mode, "stereo")

                # Record Left ear at 1 kHz & 2 kHz
                dlg._set_channel("left")
                dlg._step_sine_volume(-2.0)  # marks 1 kHz left tested
                dlg._on_next_clicked()       # moves to 2 kHz
                dlg._step_sine_volume(+6.0)  # +6 dB dip on Left 2 kHz
                # Still only Left tested -> remains stereo
                self.assertEqual(dlg.session.export_channel_mode, "stereo")

                # Now record Right ear -> auto-selects Per-ear (L/R)
                dlg._set_channel("right")
                dlg._step_sine_volume(-1.0)
                self.assertEqual(dlg.session.export_channel_mode, "per_ear")
                self.assertEqual(dlg.export_mode_combo.currentIndex(), 1)

                # Apply emits L & R channel bands with safe headroom preamp
                dlg._on_apply_clicked()
                self.assertEqual(len(applied), 1)
                bands, preamp = applied[0]
                self.assertTrue(any(b.channel == "L" for b in bands))
                self.assertTrue(any(b.channel == "R" for b in bands))
                self.assertLessEqual(preamp, 0.0)

                # User can manually toggle back to Stereo (Avg)
                dlg.export_mode_combo.setCurrentIndex(0)
                self.assertEqual(dlg.session.export_channel_mode, "stereo")
                dlg._on_apply_clicked()
                stereo_bands, _ = applied[1]
                self.assertTrue(all(b.channel == "ALL" for b in stereo_bands))
            finally:
                dlg.close()


class TestBiquadDSP(unittest.TestCase):
    def test_z_cache_no_collision_and_list_input(self):
        f_log = np.geomspace(20.0, 20000.0, 100)
        f_lin = np.linspace(20.0, 20000.0, 100)
        r_log = compute_filter_response("Peak", 1000.0, 6.0, 2.0, f_log)
        r_lin = compute_filter_response("Peak", 1000.0, 6.0, 2.0, f_lin)
        self.assertFalse(np.allclose(r_log, r_lin))

        # Python list input must work cleanly
        r_list = compute_composite_response(
            [EQBand(1, True, "Peak", 1000.0, 6.0, 2.0)],
            [100.0, 1000.0, 10000.0],
        )
        self.assertEqual(len(r_list), 3)
        self.assertAlmostEqual(float(r_list[1]), 6.0, places=1)

    def test_numerical_edge_cases(self):
        freqs = np.geomspace(20.0, 20000.0, 64)
        for ftype, f0, g, q, fs in [
            ("Peak", 1000.0, 15000.0, 1.0, 48000.0),
            ("Peak", 1000.0, -15000.0, 1.0, 48000.0),
            ("Low Shelf", float("nan"), float("inf"), float("nan"), 48000.0),
            ("High Pass", 500.0, 0.0, 0.0, 10.0),
            ("Low Pass", 24000.0, 0.0, 1000.0, float("nan")),
        ]:
            coeffs = biquad_coeffs(ftype, f0, g, q, fs)
            self.assertTrue(all(math.isfinite(c) for c in coeffs))
            resp = compute_filter_response(ftype, f0, g, q, freqs, fs)
            self.assertTrue(np.all(np.isfinite(resp)))

    def test_filter_type_from_str_aliases(self):
        self.assertEqual(FilterType.from_str("lowshelf"), FilterType.LOW_SHELF)
        self.assertEqual(FilterType.from_str("LSQ"), FilterType.LOW_SHELF)
        self.assertEqual(FilterType.from_str("LS 6dB"), FilterType.LOW_SHELF)
        self.assertEqual(FilterType.from_str("highshelf"), FilterType.HIGH_SHELF)
        self.assertEqual(FilterType.from_str("HSQ"), FilterType.HIGH_SHELF)
        self.assertEqual(FilterType.from_str("highpass"), FilterType.HIGH_PASS)
        self.assertEqual(FilterType.from_str("lowpass"), FilterType.LOW_PASS)
        self.assertEqual(FilterType.from_str(None), FilterType.PEAK)

    def test_band_channel_serialization_and_peak_headroom(self):
        b_all = EQBand(1, True, "Peak", 1000.0, 2.0, 1.414)
        self.assertEqual(b_all.channel, "ALL")
        self.assertNotIn("channel", b_all.to_dict())

        b_l = EQBand(2, True, "Peak", 1000.0, 4.0, 1.414, channel="left")
        self.assertEqual(b_l.channel, "L")
        self.assertEqual(b_l.to_dict()["channel"], "L")

        b_r = EQBand.from_dict({"band_id": 3, "freq": 1000.0, "gain": -3.0, "channel": "r"})
        self.assertEqual(b_r.channel, "R")
        self.assertEqual(b_r.to_dict()["channel"], "R")

        b_default = EQBand.from_dict({"band_id": 4, "freq": 500.0})
        self.assertEqual(b_default.channel, "ALL")
        self.assertNotIn("channel", b_default.to_dict())

        # Empty & cut-only bands normalize recommended preamp to positive 0.0
        p0, rec0 = compute_peak_headroom([])
        self.assertEqual(p0, 0.0)
        self.assertEqual(rec0, 0.0)
        self.assertGreater(math.copysign(1.0, rec0), 0.0)

        p_cut, rec_cut = compute_peak_headroom([EQBand(1, True, "Peak", 1000.0, -6.0, 1.414)])
        self.assertEqual(rec_cut, 0.0)
        self.assertGreater(math.copysign(1.0, rec_cut), 0.0)

        # Separate L/R channel headroom evaluates worst-case of (ALL+L) and (ALL+R), NOT ALL+L+R
        p_lr, rec_lr = compute_peak_headroom([b_all, b_l, b_r], preamp_db=0.0)
        self.assertAlmostEqual(p_lr, 6.0, delta=0.05)
        self.assertEqual(rec_lr, -6.1)

        # High-Q narrow peak at off-grid frequency is captured via center-freq union
        narrow = EQBand(1, True, "Peak", 1234.56, 6.0, 18.0)
        disabled = EQBand(2, False, "Peak", 2000.0, 15.0, 1.414)
        p_nar, rec_nar = compute_peak_headroom([narrow, disabled], preamp_db=-6.1)
        self.assertAlmostEqual(p_nar, -0.1, delta=0.05)
        self.assertEqual(rec_nar, -6.1)


class TestAutoEQOptimizer(unittest.TestCase):
    def test_bounds_edge_cases_and_flat_preamp(self):
        mf = np.geomspace(20.0, 20000.0, 200)
        ms = np.zeros_like(mf)
        tf = mf.copy()
        ts = np.zeros_like(mf)

        res_flat = optimize_parametric_eq(mf, ms, tf, ts)
        self.assertEqual(res_flat.preamp_db, 0.0)

        # Edge bounds that previously raised ValueError / IndexError
        for kwargs in [
            {"min_q": 2.0, "max_q": 6.0},
            {"max_freq": 100.0, "use_bass_shelf": True},
            {"min_freq": 200.0, "max_freq": 8500.0, "num_bands": 10},
            {"max_boost_db": 0.0, "max_cut_db": 0.0},
        ]:
            res = optimize_parametric_eq(mf, ms, tf, ts, **kwargs)
            self.assertTrue(math.isfinite(res.final_rms_db))
            self.assertTrue(math.isfinite(res.preamp_db))

    def test_unsorted_nan_partial_range_and_max_freq_weight_precedence(self):
        mf = np.geomspace(40.0, 12000.0, 150)
        ms = 4.0 * np.exp(-0.5 * ((np.log2(mf / 1000.0)) / 0.35) ** 2)
        # Shuffle order, add duplicate and NaN
        perm = np.random.default_rng(42).permutation(len(mf))
        mf_bad = np.concatenate([mf[perm], [1000.0, 500.0]])
        ms_bad = np.concatenate([ms[perm], [4.0, float("nan")]])

        tf = np.geomspace(20.0, 20000.0, 200)
        ts = np.zeros_like(tf)
        res = optimize_parametric_eq(mf_bad, ms_bad, tf, ts, num_bands=5, max_freq=4000.0)
        self.assertLess(res.final_rms_db, res.initial_rms_db)
        self.assertTrue(all(b.freq <= 4000.5 for b in res.bands))

        # Verify max_freq=3500 doesn't chase a 5 kHz spike with max weight
        mf2 = np.geomspace(20.0, 20000.0, 250)
        ms2 = 10.0 * np.exp(-0.5 * ((np.log2(mf2 / 5000.0)) / 0.15) ** 2)
        res2 = optimize_parametric_eq(mf2, ms2, mf2, np.zeros_like(mf2), num_bands=4, max_freq=3500.0)
        self.assertTrue(all(b.freq <= 3500.5 for b in res2.bands))

    def test_cancel_check_interrupts_promptly(self):
        mf = np.geomspace(20.0, 20000.0, 150)
        ms = np.sin(np.log2(mf)) * 5.0
        calls = {"n": 0}

        def _cancel_after_two():
            calls["n"] += 1
            return calls["n"] >= 2

        with self.assertRaises(_OptimizationCancelled):
            optimize_parametric_eq(mf, ms, mf, np.zeros_like(mf), cancel_check=_cancel_after_two)

        # AutoEQWorker.cancel() stops cleanly without emitting signals (both pre-run and live thread)
        worker = AutoEQWorker(mf, ms, mf, np.zeros_like(mf), num_bands=8)
        emitted = []
        worker.sigOptimized.connect(lambda r: emitted.append(("ok", r)))
        worker.sigFailed.connect(lambda e: emitted.append(("err", e)))
        worker.cancel()
        worker.run()
        self.assertEqual(emitted, [])

        worker2 = AutoEQWorker(mf, ms, mf, np.zeros_like(mf), num_bands=12)
        emitted2 = []
        worker2.sigOptimized.connect(lambda r: emitted2.append(("ok", r)))
        worker2.sigFailed.connect(lambda e: emitted2.append(("err", e)))
        worker2.start()
        worker2.cancel()
        self.assertFalse(worker2.isRunning())

    def test_target_preferences_region_rms_auto_prune_and_rig_guardrails(self):
        self.assertIs(AutoEQResult, OptimizationResult)
        freqs = np.geomspace(20.0, 20000.0, 300)
        flat = np.full_like(freqs, 80.0)

        # Preference sliders: bass low-shelf @ 105 Hz and 1 kHz-pivoted treble tilt
        adj_bass = apply_target_preferences(freqs, flat, bass_boost_db=6.0, treble_tilt_db=0.0)
        self.assertAlmostEqual(
            float(np.interp(np.log10(1000.0), np.log10(freqs), adj_bass - flat)), 0.0, delta=0.05
        )
        self.assertGreater(float(np.interp(np.log10(25.0), np.log10(freqs), adj_bass - flat)), 5.5)

        adj_tilt = apply_target_preferences(freqs, flat, bass_boost_db=0.0, treble_tilt_db=3.0)
        self.assertAlmostEqual(
            float(np.interp(np.log10(10000.0), np.log10(freqs), adj_tilt - flat)), 3.0, delta=0.05
        )
        self.assertAlmostEqual(
            float(np.interp(np.log10(100.0), np.log10(freqs), adj_tilt - flat)), -3.0, delta=0.05
        )

        # Auto-prune on flat curve keeps 1 fallback band and sets pruned_count
        res_flat = optimize_parametric_eq(freqs, flat, freqs, flat, num_bands=7, auto_prune=True, prune_threshold_db=0.2)
        self.assertEqual(len(res_flat.bands), 1)
        self.assertEqual(res_flat.pruned_count, 6)
        self.assertEqual(set(res_flat.region_rms_before.keys()), {"bass", "mid", "treble"})
        self.assertEqual(set(res_flat.region_rms_after.keys()), {"bass", "mid", "treble"})
        self.assertEqual(len(res_flat.adjusted_target_freqs or []), len(res_flat.f_grid))
        self.assertEqual(len(res_flat.adjusted_target_spl or []), len(res_flat.f_grid))

        res_noprune = optimize_parametric_eq(freqs, flat, freqs, flat, num_bands=7, auto_prune=False)
        self.assertEqual(len(res_noprune.bands), 7)
        self.assertEqual(res_noprune.pruned_count, 0)

        # Rig-aware guardrails: 711 caps >7.5 kHz gain/Q vs 5128 trusts up to 10 kHz
        notch_8k = flat - 11.0 * np.exp(-0.5 * ((np.log2(freqs) - np.log2(8200.0)) / 0.12) ** 2)
        res_711 = optimize_parametric_eq(
            freqs, notch_8k, freqs, flat,
            num_bands=5, use_bass_shelf=False, max_freq=10000.0, max_boost_db=10.0, max_q=6.0,
            rig_type="clone IEC 711", auto_prune=True,
        )
        for b in res_711.bands:
            if b.freq > 7500.0:
                self.assertLessEqual(b.q, 3.5 + 1e-6)
                self.assertLessEqual(b.gain, 6.0 + 1e-6)
                self.assertGreaterEqual(b.gain, -8.0 - 1e-6)

        res_5128 = optimize_parametric_eq(
            freqs, notch_8k, freqs, flat,
            num_bands=5, use_bass_shelf=False, max_freq=10000.0, max_boost_db=10.0, max_q=6.0,
            rig_type="B&K 5128", auto_prune=True,
        )
        self.assertLess(res_5128.region_rms_after["treble"], res_711.region_rms_after["treble"])


class TestAudioStreamDSP(unittest.TestCase):
    def _make_worker(self) -> AudioSpectrumWorker:
        w = AudioSpectrumWorker.__new__(AudioSpectrumWorker)
        w.window_size = 2048
        w.target_points = 1000
        w.target_freqs = np.geomspace(20.0, 20000.0, 1000)
        w.hanning_window = np.hanning(2048)
        w.sample_rate = 48000
        w._rta_matrix = w._build_rta_matrix(48000)
        w._state_lock = threading.Lock()
        w._smoothed_mags = np.full(1000, -20.0, dtype=np.float32)
        w._at_floor = True
        w._stop_event = threading.Event()
        w._restart_event = threading.Event()
        w._ever_started = False
        w._enabled = True
        w._device_index = None
        w._device_name = None
        w._agc_gain_db = 38.0
        w._audio_queue = queue.Queue(maxsize=32)
        return w

    def test_rta_matrix_excludes_dc_bin_and_zeros_above_nyquist(self):
        w = self._make_worker()
        m96 = w._build_rta_matrix(96000)
        m192 = w._build_rta_matrix(192000)
        self.assertEqual(m96[:, 0].nnz, 0)
        self.assertEqual(m192[:, 0].nnz, 0)

        m16 = w._build_rta_matrix(16000)
        above_nyq = w.target_freqs > 8000.0
        self.assertEqual(m16[above_nyq, :].nnz, 0)

    def test_agc_range_nan_guard_and_device_index_clears_name(self):
        w = self._make_worker()
        # Verify -10 dBFS 1 kHz sine produces sensible mags_db within AGC attenuation range
        t = np.arange(2048) / 48000.0
        sine = (0.316 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)
        X = np.fft.rfft(sine * w.hanning_window)
        power = ((2.0 / 2048.0) * np.abs(X)) ** 2
        mags_db = 10.0 * np.log10(np.maximum(1e-12, w._rta_matrix.dot(power)))
        peak = float(np.max(mags_db))
        req_gain = float(np.clip(-3.0 - peak, -30.0, 65.0))
        self.assertAlmostEqual(peak + req_gain, -3.0, places=1)

        # Verify NaN/Inf chunk sanitization
        buf = np.zeros(2048, dtype=np.float32)
        bad_chunk = np.array([float("nan"), float("inf"), float("-inf"), 0.1], dtype=np.float32)
        w._ingest_chunk(buf, bad_chunk)
        self.assertTrue(np.all(np.isfinite(buf)))

        # set_device_index must clear stale _device_name
        w.set_device_target(3, "Headphones (USB)")
        self.assertEqual(w._device_name, "Headphones (USB)")
        w.set_device_index(5)
        self.assertEqual(w._device_index, 5)
        self.assertIsNone(w._device_name)

    def test_tilt_speed_freeze_and_peak_hold_envelope(self):
        import sys
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.assertIsNotNone(app)
        w = AudioSpectrumWorker()
        self.assertEqual(w.get_tilt_mode(), "pink")
        self.assertEqual(w.get_speed_mode(), "medium")
        self.assertFalse(w.is_frozen())
        self.assertFalse(w.is_peak_hold_enabled())

        idx_1k = int(np.argmin(np.abs(w.target_freqs - 1000.0)))
        idx_4k = int(np.argmin(np.abs(w.target_freqs - 4000.0)))
        # Pink (+3 dB/oct -> 4x power from 1k to 4k)
        self.assertAlmostEqual(
            float(w._rta_matrix[idx_4k].sum()) / float(w._rta_matrix[idx_1k].sum()), 4.0, delta=0.15
        )
        # Slope (+4.5 dB/oct -> 8x power from 1k to 4k)
        w.set_tilt_mode("slope")
        self.assertEqual(w.get_tilt_mode(), "slope")
        self.assertAlmostEqual(
            float(w._rta_matrix[idx_4k].sum()) / float(w._rta_matrix[idx_1k].sum()), 8.0, delta=0.25
        )
        # Flat (0 dB/oct -> 1x power)
        w.set_tilt_mode("flat")
        self.assertEqual(w.get_tilt_mode(), "flat")
        self.assertAlmostEqual(
            float(w._rta_matrix[idx_4k].sum()) / float(w._rta_matrix[idx_1k].sum()), 1.0, delta=0.05
        )

        # Speed presets
        w.set_speed_mode("fast")
        self.assertEqual(w.get_speed_mode(), "fast")
        self.assertAlmostEqual(w._attack_alpha, 0.65)
        self.assertAlmostEqual(w._release_alpha, 0.28)
        w.set_speed_mode("slow")
        self.assertEqual(w.get_speed_mode(), "slow")
        self.assertAlmostEqual(w._attack_alpha, 0.22)
        self.assertAlmostEqual(w._release_alpha, 0.06)

        # Freeze toggle
        w.set_frozen(True)
        self.assertTrue(w.is_frozen())
        w.set_frozen(False)
        self.assertFalse(w.is_frozen())

        # Peak-hold signal, 0.8s hold, ~6 dB/s decay, reset, disable
        events = []
        w.sigPeakHoldReady.connect(lambda f, m: events.append((f, m.copy())))
        w.set_peak_hold_enabled(True)
        self.assertTrue(w.is_peak_hold_enabled())
        self.assertEqual(len(events), 1)

        with w._state_lock:
            w._smoothed_mags.fill(0.0)
            w._at_floor = False
            p1 = w._update_peak_hold_locked(0.025)
            self.assertTrue(np.allclose(p1, 0.0))
            w._smoothed_mags.fill(-15.0)
            for _ in range(28):  # 0.70s < 0.80s hold
                p_hold = w._update_peak_hold_locked(0.025)
            self.assertTrue(np.allclose(p_hold, 0.0))
            for _ in range(24):  # +0.60s -> ~0.50s decay at 6 dB/s -> ~-3.0 dB
                p_dec = w._update_peak_hold_locked(0.025)
            self.assertGreater(float(p_dec[0]), -3.5)
            self.assertLess(float(p_dec[0]), -2.5)

        w.reset_peak_hold()
        self.assertTrue(np.allclose(events[-1][1], -15.0))
        w.set_peak_hold_enabled(False)
        self.assertFalse(w.is_peak_hold_enabled())
        self.assertTrue(np.allclose(events[-1][1], -20.0))


if __name__ == "__main__":
    unittest.main()
