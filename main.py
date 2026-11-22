"""
Clean APO - Production-Ready Native Desktop Application for Windows.
Minimalist Black & White aesthetic, real-time Equalizer APO integration,
sweeping profile animations, audio device loopback routing, and universal EQ controls.
"""

from __future__ import annotations
import os
import sys
import urllib.parse
from collections import deque
from typing import Optional, List, Dict, Tuple, Any, Sequence
import ctypes
from ctypes import wintypes
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets, sip
from PyQt6.QtMultimedia import QMediaDevices

_SHUTDOWN_THREADS: List[QtCore.QThread] = []


def _qt_message_filter(mode, context, message):
    try:
        msg = str(message)
        if "setPointSize" in msg or "Point size" in msg or "pointSize" in msg:
            return
        if mode == QtCore.QtMsgType.QtFatalMsg:
            if sys.stderr is not None:
                sys.stderr.write(f"Fatal: {message}\n")
        elif mode == QtCore.QtMsgType.QtCriticalMsg:
            if sys.stderr is not None:
                sys.stderr.write(f"Critical: {message}\n")
        elif mode == QtCore.QtMsgType.QtWarningMsg:
            if sys.stderr is not None:
                sys.stderr.write(f"Warning: {message}\n")
    except Exception:
        pass


QtCore.qInstallMessageHandler(_qt_message_filter)

from biquad import EQBand, FilterType, compute_composite_response, compute_peak_headroom
from presets import (
    PresetManager,
    DEFAULT_APO_PATH,
    can_write_to_path,
    get_apo_configurator_path,
    launch_apo_configurator,
    clean_apo_device_name,
    is_equalizer_apo_installed,
    launch_apo_installer,
    ensure_apo_root_config_include,
    grant_apo_directory_permissions,
    create_windows_shortcuts,
    export_eq_to_apo_text,
    export_all_presets_to_json_text,
    export_eq_to_file,
    parse_eq_from_text,
    preview_eq_import,
    diagnose_apo_setup,
    repair_apo_setup,
)
from audio_stream import AudioSpectrumWorker, get_available_audio_devices, get_available_playback_devices
from eq_plot import (
    EQPlotWidget,
    CHECK_WHITE_PATH,
    CHECK_BLACK_PATH,
    SCROLL_UP_PATH,
    SCROLL_DOWN_PATH,
    SCROLL_LEFT_PATH,
    SCROLL_RIGHT_PATH,
    CHEVRON_DOWN_PATH,
    CHEVRON_DOWN_HOVER_PATH,
)
from squig_client import SquigClient, SquigModel, SquigTarget, GlobalSearchResult, DEFAULT_DATABASES, get_rig_badge
from autoeq_optimizer import AutoEQWorker, OptimizationResult, apply_target_preferences
from offline_db import get_offline_db, OfflineSyncWorker
from hearing_test import HearingTestDialog

UI_FONT_FAMILY = '"Segoe UI", "Inter", sans-serif'
MONO_FONT_FAMILY = '"Cascadia Mono", "Consolas", monospace'


# -----------------------------------------------------------------------------
# Background Worker Threads for Squiglink Database Operations
# -----------------------------------------------------------------------------
class DatabaseIndexWorker(QtCore.QThread):
    sigIndexLoaded = QtCore.pyqtSignal(list, list, str)  # models, targets, rig
    sigIndexFailed = QtCore.pyqtSignal(str)

    def __init__(self, client: SquigClient, base_url: str, force_refresh: bool = False, parent=None):
        super().__init__(parent)
        self.client = client
        self.base_url = base_url
        self.force_refresh = force_refresh

    def run(self) -> None:
        try:
            models, targets, rig = self.client.fetch_database_index(self.base_url, self.force_refresh)
            if not self.isInterruptionRequested():
                self.sigIndexLoaded.emit(models, targets, rig)
        except Exception as e:
            if not self.isInterruptionRequested():
                self.sigIndexFailed.emit(str(e))


class CurveFetchWorker(QtCore.QThread):
    sigCurvesLoaded = QtCore.pyqtSignal(object, object)  # (meas_f, meas_s), (targ_f, targ_s)
    sigTarget2Loaded = QtCore.pyqtSignal(object)  # (t2_f, t2_s)
    sigCurveFailed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        client: SquigClient,
        base_url: str,
        model: Optional[SquigModel],
        target_mode: int = 0,  # 0 = Target Curve, 1 = IEM Tuning, 2 = Resultant Profile
        target: Optional[SquigTarget] = None,
        target_iem_model: Optional[SquigModel] = None,
        target_iem_chan: str = "avg",
        target_profile_name: str = "",
        preset_manager: Optional[PresetManager] = None,
        sample_rate: float = 48000.0,
        channel_mode: str = "avg",
        target2: Optional[SquigTarget] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.client = client
        self.base_url = base_url
        self.model = model
        self.target_mode = target_mode
        self.target = target
        self.target_iem_model = target_iem_model
        self.target_iem_chan = target_iem_chan
        self.target_profile_name = target_profile_name
        self.sample_rate = sample_rate
        self.channel_mode = channel_mode
        self.target2 = target2
        # Snapshot target profile data on the GUI thread to avoid cross-thread dict mutation
        self._target_profile_snapshot: Optional[Dict[str, Any]] = None
        if target_mode == 2 and target_profile_name and preset_manager:
            p_raw = preset_manager.presets.get(target_profile_name)
            if isinstance(p_raw, dict):
                self._target_profile_snapshot = {
                    "bands": [dict(b) for b in p_raw.get("bands", []) if isinstance(b, dict)],
                    "autoeq": dict(p_raw.get("autoeq") or {}),
                }

    def run(self) -> None:
        try:
            mf, ms = ([], [])
            tf, ts = ([], [])
            t2f, t2s = ([], [])

            if self.isInterruptionRequested():
                return

            if self.model:
                m_url = getattr(self.model, "base_url", "") or self.base_url
                mf, ms = self.client.fetch_iem_measurement(m_url, self.model, self.channel_mode)

            if self.isInterruptionRequested():
                return

            if self.target_mode == 0:
                # Mode 0: Standard Target Curve (e.g. Harman, IEF)
                if self.target:
                    tf, ts = self.client.fetch_target_curve(self.base_url, self.target)
            elif self.target_mode == 1:
                # Mode 1: IEM / Headphone Tuning (Raw measured acoustic response)
                if self.target_iem_model:
                    t_url = getattr(self.target_iem_model, "base_url", "") or self.base_url
                    tf, ts = self.client.fetch_iem_measurement(t_url, self.target_iem_model, self.target_iem_chan)
            elif self.target_mode == 2:
                # Mode 2: Resultant Profile Response (Measurement + EQ filters)
                p_data = self._target_profile_snapshot
                if p_data:
                    bands: List[EQBand] = []
                    for b in p_data.get("bands", []):
                        try:
                            bands.append(EQBand.from_dict(b))
                        except Exception:
                            continue
                    meta = p_data.get("autoeq") or {}
                    db_id = meta.get("database_id", "")
                    m_file = meta.get("model_file") or meta.get("model_name", "")
                    p_chan = meta.get("channel_mode", "avg")
                    p_base_url = meta.get("base_url", "") or self.base_url

                    meas_res = None
                    if db_id and m_file:
                        try:
                            meas_res = get_offline_db().get_measurement(db_id, m_file, p_chan)
                        except Exception:
                            meas_res = None

                    if not meas_res and m_file and p_base_url:
                        try:
                            temp_m = SquigModel(
                                brand="",
                                name=m_file,
                                full_name=m_file,
                                file_name=m_file,
                                display_name=m_file,
                            )
                            meas_res = self.client.fetch_iem_measurement(p_base_url, temp_m, p_chan)
                        except Exception:
                            meas_res = None

                    if meas_res and len(meas_res[0]) > 0 and len(meas_res[0]) == len(meas_res[1]):
                        pmf, pms = meas_res
                        eq_resp = compute_composite_response(
                            bands, np.asarray(pmf, dtype=np.float64), preamp_db=0.0, fs=self.sample_rate
                        )
                        res_spls = np.asarray(pms, dtype=np.float64) + eq_resp
                        tf, ts = (list(pmf), res_spls.tolist())
                    else:
                        # Fallback: pure EQ response on standard logarithmic grid
                        f_grid = np.geomspace(20.0, 20000.0, num=300)
                        eq_resp = compute_composite_response(
                            bands, f_grid, preamp_db=0.0, fs=self.sample_rate
                        )
                        tf, ts = (f_grid.tolist(), eq_resp.tolist())

            if self.isInterruptionRequested():
                return

            if self.target2 and self.base_url:
                try:
                    t2f, t2s = self.client.fetch_target_curve(self.base_url, self.target2)
                except Exception:
                    t2f, t2s = ([], [])

            if not self.isInterruptionRequested():
                self.sigCurvesLoaded.emit((mf, ms), (tf, ts))
                if self.target2:
                    self.sigTarget2Loaded.emit((t2f, t2s))
        except Exception as e:
            if not self.isInterruptionRequested():
                self.sigCurveFailed.emit(str(e))


# -----------------------------------------------------------------------------
# CleanSpinBox: Highlights strictly the numeric value on click / focus
# Guarantees that the suffix (" Hz", " dB") is never selected or edited
# -----------------------------------------------------------------------------
class CleanSpinBox(QtWidgets.QDoubleSpinBox):
    """
    Custom QDoubleSpinBox that highlights only the numeric portion on click or focus.
    Prevents selection of the unit suffix (" Hz", " dB").
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lineEdit().setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._select_timer = QtCore.QTimer(self)
        self._select_timer.setSingleShot(True)
        self._select_timer.setInterval(0)
        self._select_timer.timeout.connect(self._select_numeric_only)

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._select_timer.start()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        self._select_timer.start()

    def _select_numeric_only(self) -> None:
        if sip.isdeleted(self):
            return
        le = self.lineEdit()
        if not le or sip.isdeleted(le):
            return
        text = self.text()
        suffix = self.suffix()
        num_len = len(text) - len(suffix) if suffix and text.endswith(suffix) else len(text)
        le.setSelection(0, max(0, num_len))


# -----------------------------------------------------------------------------
# AutoRefreshComboBox: Re-scans audio devices on dropdown click / popup
# -----------------------------------------------------------------------------
class AutoRefreshComboBox(QtWidgets.QComboBox):
    """
    QComboBox that emits sigAboutToShowPopup before displaying the popup dropdown,
    allowing dynamic refreshing of audio devices upon user interaction.
    """
    sigAboutToShowPopup = QtCore.pyqtSignal()

    def showPopup(self) -> None:
        self.sigAboutToShowPopup.emit()
        super().showPopup()


class NoScrollComboBox(QtWidgets.QComboBox):
    """
    QComboBox that ignores mouse wheel events to prevent accidental filter type switching.
    Allows wheel events to scroll the underlying table or container naturally.
    """
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        event.ignore()


class BandBadgeButton(QtWidgets.QPushButton):
    """
    Interactive circular band badge button for the EQ bands table.
    - Active (ON): Crisp off-white circle with dark band number.
    - Bypassed (OFF): Dark graphite circle with dashed border and muted band number.
    - Hover: Subtle border contrast step (no halo glow).
    - Clicking toggles bypass state of the band.
    """
    def __init__(self, band_id: int, enabled: bool = True, parent=None):
        super().__init__(parent)
        self.band_id = band_id
        self.setProperty("band_id", band_id)
        self.setCheckable(True)
        self.setChecked(enabled)
        self.setFixedSize(24, 24)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; padding: 0px; min-height: 0px; min-width: 0px; }"
        )
        self._update_tooltip()
        self.toggled.connect(lambda _: self._update_tooltip())

    def _update_tooltip(self) -> None:
        state = "bypass" if self.isChecked() else "enable"
        self.setToolTip(f"Band {self.band_id} - Click to {state}")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect_f = QtCore.QRectF(self.rect())
        center = rect_f.center()
        radius = 9.5
        hovered = self.underMouse()

        if self.isChecked():
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff" if hovered else "#ededed")))
            painter.setPen(QtGui.QPen(QtGui.QColor("#3f3f46" if hovered else "#27272a"), 1.0))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0b0b0c")))
        else:
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#1c1c1f" if hovered else "#121214")))
            painter.setPen(
                QtGui.QPen(
                    QtGui.QColor("#a1a1aa" if hovered else "#71717a"),
                    1.2,
                    QtCore.Qt.PenStyle.DashLine,
                )
            )
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#ededed" if hovered else "#a1a1aa")))

        font = _make_mono_font(8, QtGui.QFont.Weight.DemiBold)
        painter.setFont(font)
        text_rect = QtCore.QRectF(rect_f)
        text_rect.translate(0, -0.5)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(self.band_id))
        painter.end()


def _make_mono_font(point_size: int = 9, weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal) -> QtGui.QFont:
    """Returns a clean monospace QFont with reliable Windows fallbacks."""
    font = QtGui.QFont()
    font.setFamilies(["Cascadia Mono", "Consolas", "Courier New"])
    font.setPointSize(point_size)
    font.setWeight(weight)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return font


class ApoHealthButton(QtWidgets.QPushButton):
    """QPushButton whose sizeHint tracks its internal QHBoxLayout so label text never clips."""
    def sizeHint(self) -> QtCore.QSize:
        lay = self.layout()
        if lay is not None:
            s = lay.sizeHint()
            return QtCore.QSize(max(s.width() + 4, 80), max(s.height(), 28))
        return super().sizeHint()

    def minimumSizeHint(self) -> QtCore.QSize:
        return self.sizeHint()


class StatusDotWidget(QtWidgets.QWidget):
    """Crisp 6px vector status dot for APO health and diagnostic indicators."""
    def __init__(self, color: str = "#4cc38a", filled: bool = True, parent=None):
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self._filled = filled
        self.setFixedSize(10, 10)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_status(self, color: str, filled: bool = True) -> None:
        self._color = QtGui.QColor(color)
        self._filled = filled
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        center = QtCore.QRectF(self.rect()).center()
        r = 3.0
        if self._filled:
            painter.setBrush(QtGui.QBrush(self._color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(center, r, r)
        else:
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(self._color, 1.3))
            painter.drawEllipse(center, r, r)
        painter.end()


class StatusToggleButton(QtWidgets.QPushButton):
    """
    Quiet raised graphite toggle button without decorative status dots.
    Avoids solid-white toggle bloat so primary actions remain distinct.
    """
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._dot_color = QtGui.QColor("#4cc38a")
        self._dot_filled = True
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def set_status_dot(self, color: str, filled: bool = True) -> None:
        self._dot_color = QtGui.QColor(color)
        self._dot_filled = filled
        self.update()


class FavoriteStarButton(QtWidgets.QPushButton):
    """Crisp vector star button replacing unicode star glyphs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._favorited = False
        self.setFixedSize(28, 28)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton { background-color: #121214; border: 1px solid #27272a; border-radius: 4px; padding: 0px; }"
            "QPushButton:hover { background-color: #1c1c1f; border-color: #3f3f46; }"
            "QPushButton:disabled { background-color: #0b0b0c; border-color: #27272a; }"
        )

    def set_favorited(self, fav: bool) -> None:
        self._favorited = bool(fav)
        self.update()

    def is_favorited(self) -> bool:
        return self._favorited

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        import math
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        r_outer = 6.2
        r_inner = 2.8
        poly = QtGui.QPolygonF()
        for i in range(10):
            ang = math.radians(-90 + i * 36)
            r = r_outer if i % 2 == 0 else r_inner
            poly.append(QtCore.QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        if not self.isEnabled():
            pen = QtGui.QPen(QtGui.QColor("#3f3f46"), 1.3)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        elif self._favorited:
            pen = QtGui.QPen(QtGui.QColor("#d99b38"), 1.2)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#d99b38")))
        else:
            col = QtGui.QColor("#ededed") if self.underMouse() else QtGui.QColor("#71717a")
            pen = QtGui.QPen(col, 1.3)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)
        painter.end()


class TableDeleteButton(QtWidgets.QPushButton):
    """Crisp vector X button for deleting a filter table row."""
    def __init__(self, band_id: int, parent=None):
        super().__init__(parent)
        self.setProperty("band_id", band_id)
        self.setFixedSize(24, 24)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Delete filter band")
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; }")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        hovered = self.underMouse()
        if hovered:
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#1c1c1f")))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QtCore.QRectF(2, 2, self.width() - 4, self.height() - 4), 4.0, 4.0)
        color = QtGui.QColor("#e5484d") if hovered else QtGui.QColor("#71717a")
        pen = QtGui.QPen(color, 1.5)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        half = 3.8
        painter.drawLine(QtCore.QPointF(cx - half, cy - half), QtCore.QPointF(cx + half, cy + half))
        painter.drawLine(QtCore.QPointF(cx - half, cy + half), QtCore.QPointF(cx + half, cy - half))
        painter.end()


class ChevronMenuButton(QtWidgets.QPushButton):
    """Compact 28x28 vector chevron button for secondary toolbar menus (e.g., Spectrum options)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            "QPushButton { background-color: #121214; border: 1px solid #27272a; border-radius: 4px; padding: 0px; }"
            "QPushButton:hover { background-color: #1c1c1f; border-color: #3f3f46; }"
            "QPushButton:pressed { background-color: #27272a; border-color: #3f3f46; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        col = QtGui.QColor("#ededed") if self.underMouse() else QtGui.QColor("#a1a1aa")
        pen = QtGui.QPen(col, 1.5)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        cx = self.width() / 2.0
        cy = self.height() / 2.0
        path = QtGui.QPainterPath()
        path.moveTo(cx - 4.0, cy - 1.8)
        path.lineTo(cx, cy + 2.2)
        path.lineTo(cx + 4.0, cy - 1.8)
        painter.drawPath(path)
        painter.end()


def apply_frameless_window_shadow(widget: QtWidgets.QWidget) -> None:
    """Applies native Windows OS DWM drop shadow, rounded corners, and window frame to frameless windows."""
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = int(widget.winId())
        class MARGINS(ctypes.Structure):
            _fields_ = [
                ("cxLeftWidth", wintypes.INT),
                ("cxRightWidth", wintypes.INT),
                ("cyTopHeight", wintypes.INT),
                ("cyBottomHeight", wintypes.INT),
            ]
        m = MARGINS(1, 1, 1, 1)
        ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(m))

        # Windows 11 DWM rounded corners attribute
        DWMWA_WINDOW_CORNER_PREFERENCE = 33
        DWMWCP_ROUND = 2
        pref = ctypes.c_int(DWMWCP_ROUND)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(pref),
            ctypes.sizeof(pref)
        )
    except Exception:
        pass


class WindowControlButton(QtWidgets.QPushButton):
    """
    Vector-rendered minimalist window control button.
    Supports 'min', 'max', 'restore', and 'close' with crisp 1.4px anti-aliased geometry.
    """
    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind  # 'min', 'max', 'restore', 'close'
        self.setFixedSize(32, 28)
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def set_kind(self, kind: str) -> None:
        if self.kind != kind:
            self.kind = kind
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        is_hovered = self.underMouse()
        is_pressed = self.isDown()

        if self.kind == "close":
            if is_hovered:
                bg_color = QtGui.QColor("#e5484d") if not is_pressed else QtGui.QColor("#b91c1c")
                painter.setBrush(QtGui.QBrush(bg_color))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2), 4.0, 4.0)
                icon_color = QtGui.QColor("#ededed")
            else:
                icon_color = QtGui.QColor("#a1a1aa")
        else:
            if is_hovered:
                bg_color = QtGui.QColor("#1c1c1f") if not is_pressed else QtGui.QColor("#27272a")
                painter.setBrush(QtGui.QBrush(bg_color))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2), 4.0, 4.0)
                icon_color = QtGui.QColor("#ededed")
            else:
                icon_color = QtGui.QColor("#a1a1aa")

        pen = QtGui.QPen(icon_color)
        pen.setWidthF(1.3)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        if self.kind == "min":
            w = 10.0
            painter.drawLine(QtCore.QPointF(cx - w / 2.0, cy), QtCore.QPointF(cx + w / 2.0, cy))

        elif self.kind == "max":
            size = 9.6
            r = QtCore.QRectF(cx - size / 2.0, cy - size / 2.0, size, size)
            painter.drawRoundedRect(r, 2.0, 2.0)

        elif self.kind == "restore":
            size = 8.0
            shift = 2.4

            bx = cx - size / 2.0 + shift / 2.0
            by = cy - size / 2.0 - shift / 2.0
            back_rect = QtCore.QRectF(bx, by, size, size)

            fx = cx - size / 2.0 - shift / 2.0
            fy = cy - size / 2.0 + shift / 2.0
            front_rect = QtCore.QRectF(fx, fy, size, size)

            painter.drawRoundedRect(back_rect, 2.0, 2.0)

            front_bg = QtGui.QColor("#1c1c1f") if is_hovered else QtGui.QColor("#0b0b0c")
            painter.setBrush(QtGui.QBrush(front_bg))
            painter.drawRoundedRect(front_rect, 2.0, 2.0)

        elif self.kind == "close":
            half = 4.4
            painter.drawLine(QtCore.QPointF(cx - half, cy - half), QtCore.QPointF(cx + half, cy + half))
            painter.drawLine(QtCore.QPointF(cx - half, cy + half), QtCore.QPointF(cx + half, cy - half))

        painter.end()


class DialogTitleBar(QtWidgets.QFrame):
    """
    Quiet studio title bar for modal dialogs.
    Features drag-to-move, sentence-case typography, and a vector close button.
    """
    def __init__(self, dialog: QtWidgets.QDialog, title_text: str, parent=None):
        super().__init__(parent)
        self.dialog = dialog
        self.setObjectName("dialogTitleBar")
        self.setFixedHeight(36)
        self.setStyleSheet("""
            QFrame#dialogTitleBar {
                background-color: #0b0b0c;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border-bottom: 1px solid #27272a;
            }
            QFrame#dialogTitleBar QLabel {
                border: none;
                background: transparent;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 8, 0)
        layout.setSpacing(8)

        title_lbl = QtWidgets.QLabel(title_text)
        title_lbl.setStyleSheet(
            f"font-family: {UI_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #ededed; background: transparent; border: none;"
        )
        layout.addWidget(title_lbl)

        layout.addStretch()

        close_btn = WindowControlButton("close", self)
        close_btn.setToolTip("Close")
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


class HeaderFrame(QtWidgets.QFrame):
    """
    Unified DAW-style top title bar for Clean APO.
    Supports native Windows window dragging (startSystemMove) and double-click to maximize/restore.
    """
    def __init__(self, main_win: QtWidgets.QMainWindow, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setObjectName("headerFrame")
        self._drag_pos = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            handle = self.main_win.windowHandle()
            if handle and hasattr(handle, "startSystemMove"):
                handle.startSystemMove()
                event.accept()
                return
            else:
                self._drag_pos = event.globalPosition().toPoint() - self.main_win.frameGeometry().topLeft()
                event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if getattr(self, "_drag_pos", None) is not None and event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.main_win.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if hasattr(self.main_win, "_toggle_maximize_restore"):
                self.main_win._toggle_maximize_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class WindowResizeFilter(QtCore.QObject):
    """Handles native window edge resizing for frameless windows."""
    BORDER_WIDTH = 6

    def __init__(self, window: QtWidgets.QMainWindow):
        super().__init__(window)
        self.window = window
        self.window.setMouseTracking(True)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj != self.window or self.window.isMaximized():
            return False

        if event.type() == QtCore.QEvent.Type.MouseMove:
            pos = event.pos()
            w = self.window.width()
            h = self.window.height()
            bw = self.BORDER_WIDTH

            left = pos.x() <= bw
            right = pos.x() >= w - bw
            top = pos.y() <= bw
            bottom = pos.y() >= h - bw

            if (left and top) or (right and bottom):
                self.window.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            elif (right and top) or (left and bottom):
                self.window.setCursor(QtCore.Qt.CursorShape.SizeBDiagCursor)
            elif left or right:
                self.window.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            elif top or bottom:
                self.window.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
            else:
                self.window.unsetCursor()

        elif event.type() == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.pos()
            w = self.window.width()
            h = self.window.height()
            bw = self.BORDER_WIDTH

            edges = QtCore.Qt.Edge(0)
            if pos.x() <= bw:
                edges |= QtCore.Qt.Edge.LeftEdge
            elif pos.x() >= w - bw:
                edges |= QtCore.Qt.Edge.RightEdge
            if pos.y() <= bw:
                edges |= QtCore.Qt.Edge.TopEdge
            elif pos.y() >= h - bw:
                edges |= QtCore.Qt.Edge.BottomEdge

            if edges != QtCore.Qt.Edge(0):
                handle = self.window.windowHandle()
                if handle and hasattr(handle, "startSystemResize"):
                    handle.startSystemResize(edges)
                    return True

        return False


class SmoothScrollFilter(QtCore.QObject):
    """
    Installs continuous, non-quantised pixel-smooth cubic scrolling on any QAbstractScrollArea.
    Interpolates mouse wheel and trackpad events using QEasingCurve.Type.OutCubic.
    """
    def __init__(self, scroll_area: QtWidgets.QAbstractScrollArea):
        super().__init__(scroll_area)
        self.scroll_area = scroll_area
        self.anim_v = QtCore.QVariantAnimation(self)
        self.anim_v.setDuration(160)
        self.anim_v.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.anim_v.valueChanged.connect(self._on_anim_v_step)
        self._target_v = 0

        self.anim_h = QtCore.QVariantAnimation(self)
        self.anim_h.setDuration(160)
        self.anim_h.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.anim_h.valueChanged.connect(self._on_anim_h_step)
        self._target_h = 0

    def _on_anim_v_step(self, val: Any) -> None:
        self.scroll_area.verticalScrollBar().setValue(int(val))

    def _on_anim_h_step(self, val: Any) -> None:
        self.scroll_area.horizontalScrollBar().setValue(int(val))

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Wheel and obj == self.scroll_area.viewport():
            wheel_event: QtGui.QWheelEvent = event
            delta_y = wheel_event.angleDelta().y()
            delta_x = wheel_event.angleDelta().x()
            handled = False

            if delta_y != 0:
                bar = self.scroll_area.verticalScrollBar()
                if bar.maximum() > bar.minimum():
                    start_val = bar.value()
                    base = self._target_v if self.anim_v.state() == QtCore.QAbstractAnimation.State.Running else start_val
                    target = max(bar.minimum(), min(bar.maximum(), base - delta_y))
                    self._target_v = target
                    self.anim_v.stop()
                    self.anim_v.setStartValue(start_val)
                    self.anim_v.setEndValue(target)
                    self.anim_v.start()
                    handled = True

            if delta_x != 0:
                bar_h = self.scroll_area.horizontalScrollBar()
                if bar_h.maximum() > bar_h.minimum():
                    start_val_h = bar_h.value()
                    base_h = self._target_h if self.anim_h.state() == QtCore.QAbstractAnimation.State.Running else start_val_h
                    target_h = max(bar_h.minimum(), min(bar_h.maximum(), base_h - delta_x))
                    self._target_h = target_h
                    self.anim_h.stop()
                    self.anim_h.setStartValue(start_val_h)
                    self.anim_h.setEndValue(target_h)
                    self.anim_h.start()
                    handled = True

            if handled:
                return True
        return super().eventFilter(obj, event)


def _resolve_db_id(client: Optional[SquigClient], base_url: str, fallback: str = "") -> str:
    """Resolves the canonical database ID for a Squiglink URL via SquigClient.get_db_id."""
    if base_url and client is not None:
        fn = getattr(client, "get_db_id", None) or getattr(client, "_get_db_id", None)
        if callable(fn):
            try:
                resolved = fn(base_url)
                if resolved:
                    return resolved
            except Exception:
                pass
    return fallback


def make_smooth_scroll(scroll_area: QtWidgets.QAbstractScrollArea) -> None:
    """Configures scroll area for continuous pixel-smooth scrolling."""
    if not scroll_area or isinstance(scroll_area, QtWidgets.QGraphicsView):
        return
    if getattr(scroll_area, "_smooth_scroll_installed", False):
        return
    scroll_area._smooth_scroll_installed = True

    if isinstance(scroll_area, QtWidgets.QAbstractItemView):
        scroll_area.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        scroll_area.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
    if scroll_area.viewport():
        scroll_area.viewport().installEventFilter(SmoothScrollFilter(scroll_area))


class AppGlobalScrollFilter(QtCore.QObject):
    """Monitors for newly shown scroll areas (like combo popups or tables) and ensures smooth scrolling."""
    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Show:
            if isinstance(obj, QtWidgets.QAbstractScrollArea) and not isinstance(obj, QtWidgets.QGraphicsView):
                make_smooth_scroll(obj)
            elif isinstance(obj, QtWidgets.QComboBox):
                if obj.view():
                    make_smooth_scroll(obj.view())
        return False


class SearchableModelInput(QtWidgets.QLineEdit):
    """
    Searchable model input with live suggestion popup powered by QCompleter.
    - Preserves continuous native typing focus in the QLineEdit.
    - MatchContains filter mode allows partial/substring searches.
    - Up/Down arrow keys, Enter selection, mouse click, and down chevron button support.
    - Fully operational within both top-level windows and modal QDialogs without focus blocking.
    """
    sigModelSelected = QtCore.pyqtSignal(object)  # Emits SquigModel

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setPlaceholderText("Search model")
        self._all_models: List[SquigModel] = []
        self._models_by_name: Dict[str, SquigModel] = {}
        self._selected_model: Optional[SquigModel] = None

        self.item_model = QtGui.QStandardItemModel(self)
        self.completer = QtWidgets.QCompleter(self.item_model, self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(10)

        completer_popup = self.completer.popup()
        completer_popup.setMinimumWidth(320)
        completer_popup.setUniformItemSizes(True)
        completer_popup.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        completer_popup.setStyleSheet(f"""
            QListView {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                font-family: {UI_FONT_FAMILY};
                font-size: 12px;
                padding: 4px;
                outline: none;
            }}
            QListView::item {{
                padding: 4px 8px;
                border-radius: 4px;
                min-height: 24px;
                color: #a1a1aa;
            }}
            QListView::item:hover, QListView::item:selected {{
                background-color: #1c1c1f;
                color: #ededed;
                font-weight: 500;
            }}
        """)
        make_smooth_scroll(completer_popup)
        self.setCompleter(self.completer)

        self.completer.activated[QtCore.QModelIndex].connect(self._on_completion_activated)
        self.textEdited.connect(self._on_text_edited)
        self.returnPressed.connect(self._on_return_pressed)

        self._down_arrow_action = self.addAction(
            QtGui.QIcon(CHEVRON_DOWN_PATH),
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition
        )
        self._down_arrow_action.triggered.connect(self._toggle_popup)

        self._select_all_timer = QtCore.QTimer(self)
        self._select_all_timer.setSingleShot(True)
        self._select_all_timer.setInterval(0)
        self._select_all_timer.timeout.connect(self._safe_select_all)

    def _safe_select_all(self) -> None:
        if not sip.isdeleted(self):
            self.selectAll()

    @property
    def popup(self) -> QtWidgets.QAbstractItemView:
        return self.completer.popup()

    def set_models(self, models: List[SquigModel]) -> None:
        self._all_models = models or []
        self._models_by_name = {m.display_name.lower(): m for m in self._all_models}
        self.item_model.clear()
        for m in self._all_models:
            item = QtGui.QStandardItem(m.display_name)
            item.setData(m, QtCore.Qt.ItemDataRole.UserRole)
            self.item_model.appendRow(item)
        # Clear stale selection if it does not belong to the new model list
        if self._selected_model is not None:
            key = self._selected_model.display_name.lower()
            matched = self._models_by_name.get(key)
            if matched is None or matched.file_name != self._selected_model.file_name:
                self._selected_model = None
                self.blockSignals(True)
                self.clear()
                self.blockSignals(False)

    def set_selected_model(self, model: Optional[SquigModel]) -> None:
        self._selected_model = model
        self.blockSignals(True)
        if model:
            self.setText(model.display_name)
        else:
            self.clear()
        self.blockSignals(False)
        if self.completer.popup().isVisible():
            self.completer.popup().hide()

    def currentData(self) -> Optional[SquigModel]:
        if self._selected_model:
            return self._selected_model
        txt = self.text().strip().lower()
        if txt and hasattr(self, "_models_by_name"):
            exact = self._models_by_name.get(txt)
            if exact:
                return exact
            for m in getattr(self, "_all_models", []):
                if m.display_name.lower() == txt or m.name.lower() == txt or m.file_name.lower() == txt:
                    return m
            for m in getattr(self, "_all_models", []):
                if txt in m.display_name.lower():
                    return m
        return None

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._select_all_timer.start()

    def currentText(self) -> str:
        return self.text().strip()

    def lineEdit(self) -> "SearchableModelInput":
        return self

    def _show_suggestions(self, force_all: bool = False) -> None:
        if force_all or not self.text().strip():
            self.completer.setCompletionPrefix("")
        else:
            self.completer.setCompletionPrefix(self.text().strip())
        self.completer.popup().setMinimumWidth(max(self.width(), 320))
        self.completer.complete()

    def _toggle_popup(self) -> None:
        if self.completer.popup().isVisible():
            self.completer.popup().hide()
        else:
            if self._selected_model and self.text().strip() == self._selected_model.display_name:
                self._show_suggestions(force_all=True)
            else:
                self._show_suggestions(force_all=False)

    def _on_text_edited(self, text: str) -> None:
        if self._selected_model and self._selected_model.display_name != text:
            self._selected_model = None
        txt_clean = text.strip()
        if txt_clean:
            exact = self._models_by_name.get(txt_clean.lower())
            if exact and exact != self._selected_model:
                self._selected_model = exact
                self.sigModelSelected.emit(exact)

    def _on_return_pressed(self) -> None:
        if self._selected_model:
            return
        if self.completer.popup().isVisible():
            idx = self.completer.popup().currentIndex()
            if idx.isValid():
                self._on_completion_activated(idx)
                self.completer.popup().hide()
                return
        if self.completer.completionCount() == 1:
            idx0 = self.completer.completionModel().index(0, 0)
            self._on_completion_activated(idx0)
            self.completer.popup().hide()
            return
        exact = self._models_by_name.get(self.text().strip().lower())
        if exact:
            self.set_selected_model(exact)
            self.sigModelSelected.emit(exact)

    def _on_completion_activated(self, index: QtCore.QModelIndex) -> None:
        src_idx = self.completer.completionModel().mapToSource(index)
        item = self.item_model.itemFromIndex(src_idx)
        if not item:
            return
        model = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if model:
            self._selected_model = model
            self.blockSignals(True)
            self.setText(model.display_name)
            self.blockSignals(False)
            self.sigModelSelected.emit(model)


class SearchableDatabaseInput(QtWidgets.QLineEdit):
    """
    Searchable provider/database input with live suggestion popup powered by QCompleter.
    Matches the UX, styling, and feel of SearchableModelInput, allowing users to type
    multiple characters to search and filter across all reviewer & aggregator databases.
    Compatible with QComboBox API (addItem, insertItem, clear, count, itemData, itemText,
    currentData, currentText, currentIndex, setCurrentIndex, currentIndexChanged, findData, findText).
    """
    currentIndexChanged = QtCore.pyqtSignal(int)
    sigDatabaseSelected = QtCore.pyqtSignal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setPlaceholderText("Search database")
        self._items: List[Tuple[str, Any]] = []
        self._current_index: int = -1

        self.item_model = QtGui.QStandardItemModel(self)
        self.completer = QtWidgets.QCompleter(self.item_model, self)
        self.completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        self.completer.setMaxVisibleItems(12)

        completer_popup = self.completer.popup()
        completer_popup.setMinimumWidth(320)
        completer_popup.setUniformItemSizes(True)
        completer_popup.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        completer_popup.setStyleSheet(f"""
            QListView {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                font-family: {UI_FONT_FAMILY};
                font-size: 12px;
                padding: 4px;
                outline: none;
            }}
            QListView::item {{
                padding: 4px 8px;
                border-radius: 4px;
                min-height: 24px;
                color: #a1a1aa;
            }}
            QListView::item:hover, QListView::item:selected {{
                background-color: #1c1c1f;
                color: #ededed;
                font-weight: 500;
            }}
        """)
        make_smooth_scroll(completer_popup)
        self.setCompleter(self.completer)

        self.completer.activated[QtCore.QModelIndex].connect(self._on_completion_activated)
        self.textEdited.connect(self._on_text_edited)
        self.returnPressed.connect(self._on_return_pressed)

        self._down_arrow_action = self.addAction(
            QtGui.QIcon(CHEVRON_DOWN_PATH),
            QtWidgets.QLineEdit.ActionPosition.TrailingPosition
        )
        self._down_arrow_action.triggered.connect(self._toggle_popup)

        self._select_all_timer = QtCore.QTimer(self)
        self._select_all_timer.setSingleShot(True)
        self._select_all_timer.setInterval(0)
        self._select_all_timer.timeout.connect(self._safe_select_all)

    def _safe_select_all(self) -> None:
        if not sip.isdeleted(self):
            self.selectAll()

    @property
    def popup(self) -> QtWidgets.QAbstractItemView:
        return self.completer.popup()

    def view(self) -> QtWidgets.QAbstractItemView:
        return self.completer.popup()

    def lineEdit(self) -> "SearchableDatabaseInput":
        return self

    def clear(self) -> None:
        self._items.clear()
        self.item_model.clear()
        self._current_index = -1
        self.blockSignals(True)
        super().clear()
        self.blockSignals(False)

    def count(self) -> int:
        return len(self._items)

    def insertItem(self, index: int, text: str, data: Any = None) -> None:
        index = max(0, min(index, len(self._items)))
        self._items.insert(index, (text, data))
        item = QtGui.QStandardItem(text)
        item.setData(data, QtCore.Qt.ItemDataRole.UserRole)
        self.item_model.insertRow(index, item)
        if self._current_index >= index:
            self._current_index += 1

    def addItem(self, text: str, data: Any = None) -> None:
        self.insertItem(len(self._items), text, data)

    def itemData(self, index: int) -> Any:
        if 0 <= index < len(self._items):
            return self._items[index][1]
        return None

    def itemText(self, index: int) -> str:
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return ""

    def currentIndex(self) -> int:
        return self._current_index

    def currentData(self) -> Any:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][1]
        txt = self.text().strip().lower()
        if txt:
            for it_text, it_data in self._items:
                if it_text.lower() == txt:
                    return it_data
            for it_text, it_data in self._items:
                if txt in it_text.lower():
                    return it_data
        return None

    def currentText(self) -> str:
        if 0 <= self._current_index < len(self._items):
            return self._items[self._current_index][0]
        return self.text().strip()

    def findData(self, data: Any) -> int:
        for i, (_, d) in enumerate(self._items):
            if d == data:
                return i
            if isinstance(d, dict) and isinstance(data, dict):
                if d.get("id") and d.get("id") == data.get("id"):
                    return i
        return -1

    def findText(self, text: str) -> int:
        clean = text.strip().lower()
        for i, (t, _) in enumerate(self._items):
            if t.strip().lower() == clean:
                return i
        return -1

    def setCurrentIndex(self, index: int) -> None:
        if 0 <= index < len(self._items):
            changed = (self._current_index != index)
            self._current_index = index
            orig_blocked = self.signalsBlocked()
            self.blockSignals(True)
            self.setText(self._items[index][0])
            self.blockSignals(orig_blocked)
            if changed and not orig_blocked:
                self.currentIndexChanged.emit(index)
                self.sigDatabaseSelected.emit(self._items[index][1])
        elif index == -1:
            changed = (self._current_index != -1)
            self._current_index = -1
            orig_blocked = self.signalsBlocked()
            self.blockSignals(True)
            super().clear()
            self.blockSignals(orig_blocked)
            if changed and not orig_blocked:
                self.currentIndexChanged.emit(-1)
                self.sigDatabaseSelected.emit(None)

    def _show_suggestions(self, force_all: bool = False) -> None:
        if force_all or not self.text().strip():
            self.completer.setCompletionPrefix("")
        else:
            self.completer.setCompletionPrefix(self.text().strip())
        self.completer.popup().setMinimumWidth(max(self.width(), 320))
        self.completer.complete()

    def _toggle_popup(self) -> None:
        if self.completer.popup().isVisible():
            self.completer.popup().hide()
        else:
            self.setFocus()
            self._show_suggestions(force_all=True)

    def _on_completion_activated(self, index: QtCore.QModelIndex) -> None:
        src_idx = self.completer.completionModel().mapToSource(index)
        row = src_idx.row()
        if 0 <= row < len(self._items):
            self.setCurrentIndex(row)

    def _on_text_edited(self, text: str) -> None:
        txt_clean = text.strip().lower()
        for i, (it_text, it_data) in enumerate(self._items):
            if it_text.lower() == txt_clean:
                if self._current_index != i:
                    self._current_index = i
                    self.currentIndexChanged.emit(i)
                    self.sigDatabaseSelected.emit(it_data)
                return

    def _on_return_pressed(self) -> None:
        if self.completer.popup().isVisible():
            idx = self.completer.popup().currentIndex()
            if idx.isValid():
                self._on_completion_activated(idx)
                self.completer.popup().hide()
                return
        if self.completer.completionCount() == 1:
            idx0 = self.completer.completionModel().index(0, 0)
            self._on_completion_activated(idx0)
            self.completer.popup().hide()
            return
        txt = self.text().strip().lower()
        if not txt:
            return
        for i, (it_text, _) in enumerate(self._items):
            if it_text.lower() == txt:
                self.setCurrentIndex(i)
                return
        for i, (it_text, _) in enumerate(self._items):
            if it_text.lower().startswith(txt):
                self.setCurrentIndex(i)
                return
        for i, (it_text, _) in enumerate(self._items):
            if txt in it_text.lower():
                self.setCurrentIndex(i)
                return

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        self._select_all_timer.start()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusOutEvent(event)
        if 0 <= self._current_index < len(self._items):
            if self.text().strip() != self._items[self._current_index][0]:
                txt = self.text().strip().lower()
                matched = -1
                for i, (it_text, _) in enumerate(self._items):
                    if it_text.lower() == txt:
                        matched = i
                        break
                if matched != -1:
                    self.setCurrentIndex(matched)
                else:
                    self.blockSignals(True)
                    self.setText(self._items[self._current_index][0])
                    self.blockSignals(False)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_F4):
            if not self.completer.popup().isVisible():
                self._show_suggestions(force_all=True)
                return
        super().keyPressEvent(event)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        try:
            if hasattr(self, "completer") and self.completer.popup().isVisible():
                self.completer.popup().hide()
        except Exception:
            pass


class SearchableTargetIEMInput(SearchableModelInput):
    """
    Searchable model input specifically for target IEM/headphone selection.
    Supports same-provider search or global catalog search with current provider prioritized.
    """
    def __init__(self, squig_client: Optional[SquigClient] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.squig_client = squig_client
        self.setPlaceholderText("Search target model")
        self.same_provider_only: bool = True
        self.current_provider_base_url: str = ""
        self.current_provider_name: str = ""
        self.current_provider_rig: str = ""

        self._global_search_timer = QtCore.QTimer(self)
        self._global_search_timer.setSingleShot(True)
        self._global_search_timer.setInterval(120)
        self._global_search_timer.timeout.connect(self._do_global_search)

    def set_provider_info(self, base_url: str, provider_name: str, rig: str = "") -> None:
        self.current_provider_base_url = (base_url or "").rstrip("/")
        self.current_provider_name = provider_name
        self.current_provider_rig = rig

    def set_same_provider_only(self, enabled: bool) -> None:
        self.same_provider_only = enabled
        if enabled:
            # Restore model suggestions to active database models
            self.set_models(self._all_models)
        else:
            txt = self.text().strip()
            if txt:
                self._global_search_timer.start(50)

    def _on_text_edited(self, text: str) -> None:
        if self._selected_model and self._selected_model.display_name != text:
            self._selected_model = None

        if self.same_provider_only:
            super()._on_text_edited(text)
        else:
            txt_clean = text.strip()
            if txt_clean:
                self._global_search_timer.start(120)
            else:
                self.set_models(self._all_models)

    def set_models(self, models: List[SquigModel]) -> None:
        prev_external = None
        if not self.same_provider_only and self._selected_model is not None:
            prev_url = getattr(self._selected_model, "base_url", "").rstrip("/").lower()
            if prev_url and prev_url != self.current_provider_base_url.lower():
                prev_external = self._selected_model
        super().set_models(models)
        if prev_external is not None:
            self.set_selected_model(prev_external)

    def _do_global_search(self) -> None:
        if self.same_provider_only or not self.squig_client:
            return
        query = self.text().strip()
        if not query:
            return
        results = self.squig_client.search_global_catalog(query, limit=120)
        if not results:
            self.item_model.clear()
            self._models_by_name.clear()
            if self.completer.popup().isVisible():
                self.completer.popup().hide()
            return

        # Prioritize measurements from the same provider!
        curr_url = self.current_provider_base_url.lower()
        same_prov: List[GlobalSearchResult] = []
        other_prov: List[GlobalSearchResult] = []
        for r in results:
            if curr_url and r.base_url.rstrip("/").lower() == curr_url:
                same_prov.append(r)
            else:
                other_prov.append(r)

        sorted_results = same_prov + other_prov

        self.item_model.clear()
        self._models_by_name.clear()
        for r in sorted_results:
            is_same = bool(curr_url and r.base_url.rstrip("/").lower() == curr_url)
            tag = "Same provider" if is_same else r.provider_name
            disp_item = f"{r.full_name}  [{tag}]"
            m = SquigModel(
                brand=r.brand,
                name=r.name,
                full_name=r.full_name,
                file_name=r.file_name,
                display_name=r.full_name,
            )
            # Attach dynamic provider metadata
            m.base_url = r.base_url
            m.provider_name = r.provider_name
            m.rig = r.rig
            m.database_id = _resolve_db_id(self.squig_client, r.base_url)

            self._models_by_name[r.full_name.lower()] = m
            self._models_by_name[disp_item.lower()] = m

            item = QtGui.QStandardItem(disp_item)
            item.setData(m, QtCore.Qt.ItemDataRole.UserRole)
            self.item_model.appendRow(item)

        if self.hasFocus():
            self.completer.setCompletionPrefix("")
            self.completer.popup().setMinimumWidth(max(self.width(), 360))
            self.completer.complete()

    def _on_completion_activated(self, index: QtCore.QModelIndex) -> None:
        src_idx = self.completer.completionModel().mapToSource(index)
        item = self.item_model.itemFromIndex(src_idx)
        if not item:
            return
        model = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if model:
            self._selected_model = model
            self.blockSignals(True)
            self.setText(model.display_name)
            self.blockSignals(False)
            self.sigModelSelected.emit(model)


# -----------------------------------------------------------------------------
# SegmentedPillTabBar: iOS/macOS-style segmented pill tab switcher
# -----------------------------------------------------------------------------
class SegmentedPillTabBar(QtWidgets.QWidget):
    """
    Quiet graphite segmented switcher with smooth cubic horizontal sliding indicator
    using QEasingCurve.Type.InOutCubic.
    """
    sigTabChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        tabs: List[str],
        parent: Optional[QtWidgets.QWidget] = None,
        height: int = 32,
        font_size: int = 9,
    ):
        super().__init__(parent)
        self.tabs = tabs
        self._current_index = 0
        self._pill_pos = 0.0
        self._font_size = font_size
        self.setFixedHeight(height)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim_value)

    def _on_anim_value(self, val: Any) -> None:
        self._pill_pos = float(val)
        self.update()

    def set_current_index(self, index: int, animate: bool = True) -> None:
        if index < 0 or index >= len(self.tabs):
            return
        self._current_index = index
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._pill_pos)
            self._anim.setEndValue(float(index))
            self._anim.start()
        else:
            self._pill_pos = float(index)
            self.update()

    def current_index(self) -> int:
        return self._current_index

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        h = rect.height()
        w = rect.width()
        num_tabs = len(self.tabs)
        if num_tabs == 0 or w <= 0:
            return
        tab_w = w / num_tabs

        # Outer graphite container
        track_radius = 4.0
        painter.setPen(QtGui.QPen(QtGui.QColor("#27272a"), 1.0))
        painter.setBrush(QtGui.QColor("#121214"))
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, w - 1.0, h - 1.0), track_radius, track_radius)

        # Sliding quiet raised graphite indicator
        padding = 2.0
        pill_w = tab_w - (padding * 2.0)
        pill_h = h - (padding * 2.0)
        pill_x = padding + (self._pill_pos * tab_w)
        pill_y = padding
        pill_radius = 3.0

        painter.setPen(QtGui.QPen(QtGui.QColor("#3f3f46"), 1.0))
        painter.setBrush(QtGui.QColor("#1c1c1f"))
        painter.drawRoundedRect(
            QtCore.QRectF(pill_x + 0.5, pill_y + 0.5, pill_w - 1.0, pill_h - 1.0),
            pill_radius,
            pill_radius,
        )

        # Tab text labels (sentence case, 400/500 weight, zero extra tracking)
        for i, text in enumerate(self.tabs):
            tab_rect = QtCore.QRectF(i * tab_w, 0, tab_w, h)
            dist = abs(self._pill_pos - i)
            if dist < 0.5:
                font = QtGui.QFont("Segoe UI", self._font_size, QtGui.QFont.Weight.Medium)
                painter.setPen(QtGui.QColor("#ededed"))
            else:
                font = QtGui.QFont("Segoe UI", self._font_size, QtGui.QFont.Weight.Normal)
                painter.setPen(QtGui.QColor("#71717a"))
            painter.setFont(font)
            painter.drawText(tab_rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            w = self.width()
            num_tabs = len(self.tabs)
            tab_w = w / num_tabs
            clicked_idx = int(event.position().x() // tab_w)
            clicked_idx = max(0, min(num_tabs - 1, clicked_idx))
            if clicked_idx != self._current_index:
                self.set_current_index(clicked_idx, animate=True)
                self.sigTabChanged.emit(clicked_idx)


class ApoDeviceConfigDialog(QtWidgets.QDialog):
    """Studio modal prompt when Equalizer APO is not yet registered on an audio device."""
    def __init__(self, device_name: str, configurator_path: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Equalizer APO device setup")
        self.setFixedSize(460, 232)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                font-family: {UI_FONT_FAMILY};
            }}
            QLabel {{
                color: #ededed;
                font-family: {UI_FONT_FAMILY};
                background: transparent;
                border: none;
            }}
            QPushButton {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                padding: 4px 12px;
                min-height: 28px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #1c1c1f;
                border-color: #3f3f46;
            }}
            QPushButton#primaryLaunchBtn {{
                background-color: #ededed;
                color: #0b0b0c;
                border: 1px solid #ededed;
                font-weight: 600;
            }}
            QPushButton#primaryLaunchBtn:hover {{
                background-color: #ffffff;
                border-color: #ffffff;
            }}
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Equalizer APO device setup")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        title = QtWidgets.QLabel("Equalizer APO is not attached to this device")
        title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ededed;")
        layout.addWidget(title)

        dev_lbl = QtWidgets.QLabel(f"Device: {device_name}")
        dev_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; font-weight: 500; color: #ededed;")
        dev_lbl.setWordWrap(True)
        layout.addWidget(dev_lbl)

        desc = QtWidgets.QLabel(
            "Windows requires Equalizer APO to be attached to each playback device. "
            "Open the Configurator to enable Equalizer APO on this endpoint."
        )
        desc.setStyleSheet("font-size: 12px; color: #a1a1aa;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        dismiss_btn = QtWidgets.QPushButton("Close")
        dismiss_btn.clicked.connect(self.accept)
        btn_row.addWidget(dismiss_btn)

        launch_btn = QtWidgets.QPushButton("Run APO Configurator")
        launch_btn.setObjectName("primaryLaunchBtn")
        launch_btn.clicked.connect(self._launch_configurator)
        btn_row.addWidget(launch_btn)

        layout.addLayout(btn_row)
        self.configurator_path = configurator_path

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _launch_configurator(self) -> None:
        if not self.configurator_path or not os.path.exists(self.configurator_path):
            if not is_equalizer_apo_installed():
                parent_win = self.parent()
                dlg = EqualizerApoInstallDialog(parent_win)
                if parent_win is not None and hasattr(parent_win, "_on_apo_installed_success"):
                    dlg.sigInstalled.connect(parent_win._on_apo_installed_success)
                dlg.exec()
                dlg.deleteLater()
                return
        ok, msg = launch_apo_configurator(self.configurator_path)
        if ok:
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Launch error", f"Could not launch Configurator: {msg}")


class EqualizerApoInstallDialog(QtWidgets.QDialog):
    """
    Studio setup modal shown when Equalizer APO is not installed on the system.
    Provides one-click installation using the bundled Equalizer APO installer,
    real-time installation polling, automatic config linking, and audio device setup.
    """
    sigInstalled = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Equalizer APO setup")
        self.setFixedSize(520, 360)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                font-family: {UI_FONT_FAMILY};
            }}
            QLabel {{
                color: #ededed;
                font-family: {UI_FONT_FAMILY};
                background: transparent;
                border: none;
            }}
            QFrame#installFeatureCard {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }}
            QFrame#installFeatureCard QLabel {{
                border: none;
                background: transparent;
            }}
            QPushButton {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
                padding: 4px 12px;
                min-height: 28px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #1c1c1f;
                border-color: #3f3f46;
            }}
            QPushButton#primaryInstallBtn {{
                background-color: #ededed;
                color: #0b0b0c;
                border: 1px solid #ededed;
                font-weight: 600;
            }}
            QPushButton#primaryInstallBtn:hover {{
                background-color: #ffffff;
                border-color: #ffffff;
            }}
            QCheckBox {{
                color: #a1a1aa;
                font-size: 12px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                background: #121214;
            }}
            QCheckBox::indicator:checked {{
                background: #1c1c1f;
                border-color: #ededed;
                image: url('{CHECK_WHITE_PATH}');
            }}
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Equalizer APO setup")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        self.header_title = QtWidgets.QLabel("Equalizer APO is required")
        self.header_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #ededed;")
        layout.addWidget(self.header_title)

        self.desc_lbl = QtWidgets.QLabel(
            "Clean APO applies system-wide parametric EQ across Windows audio "
            "through Equalizer APO. The official installer is bundled for local setup."
        )
        self.desc_lbl.setStyleSheet("font-size: 12px; color: #a1a1aa;")
        self.desc_lbl.setWordWrap(True)
        layout.addWidget(self.desc_lbl)

        features_box = QtWidgets.QFrame()
        features_box.setObjectName("installFeatureCard")
        f_layout = QtWidgets.QVBoxLayout(features_box)
        f_layout.setContentsMargins(12, 10, 12, 10)
        f_layout.setSpacing(6)

        for text in (
            "Official Equalizer APO x64 v1.4.2 installer bundled",
            "Automatically adds clean_apo_generated.txt to config.txt",
            "Configurator opens so you can select playback endpoints",
        ):
            lbl = QtWidgets.QLabel(f"- {text}")
            lbl.setStyleSheet("color: #a1a1aa; font-size: 11px;")
            f_layout.addWidget(lbl)
        layout.addWidget(features_box)

        self.status_msg = QtWidgets.QLabel("")
        self.status_msg.setStyleSheet("color: #ededed; font-size: 11px; font-weight: 500;")
        self.status_msg.setWordWrap(True)
        self.status_msg.setVisible(False)
        layout.addWidget(self.status_msg)

        layout.addStretch()

        self.chk_dont_ask = QtWidgets.QCheckBox("Do not prompt again on startup")
        layout.addWidget(self.chk_dont_ask)

        self.btn_row = QtWidgets.QHBoxLayout()
        self.btn_row.setSpacing(8)
        self.btn_row.addStretch()

        self.dismiss_btn = QtWidgets.QPushButton("Continue offline")
        self.dismiss_btn.clicked.connect(self._on_dismiss)
        self.btn_row.addWidget(self.dismiss_btn)

        self.launch_cfg_btn = QtWidgets.QPushButton("Launch Configurator")
        self.launch_cfg_btn.setVisible(False)
        self.launch_cfg_btn.clicked.connect(self._launch_configurator)
        self.btn_row.addWidget(self.launch_cfg_btn)

        self.install_btn = QtWidgets.QPushButton("Install Equalizer APO")
        self.install_btn.setObjectName("primaryInstallBtn")
        self.install_btn.clicked.connect(self._start_installation)
        self.btn_row.addWidget(self.install_btn)

        layout.addLayout(self.btn_row)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(1200)
        self._poll_timer.timeout.connect(self._check_installation_polling)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _start_installation(self) -> None:
        ok, msg = launch_apo_installer()
        if not ok:
            self.status_msg.setText(f"Install notice: {msg}")
            self.status_msg.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500;")
            self.status_msg.setVisible(True)
            return

        self.install_btn.setEnabled(False)
        self.install_btn.setText("Installer running...")
        self.status_msg.setText("Setup launched. Follow the installer prompts, select your audio devices, and click Finish.")
        self.status_msg.setStyleSheet("color: #a1a1aa; font-size: 11px; font-weight: 500;")
        self.status_msg.setVisible(True)
        self._poll_timer.start()

    def _check_installation_polling(self) -> None:
        if is_equalizer_apo_installed():
            self._poll_timer.stop()
            self._on_installation_complete()

    def _on_installation_complete(self) -> None:
        ensure_apo_root_config_include(DEFAULT_APO_PATH)
        self.header_title.setText("Equalizer APO ready")
        self.desc_lbl.setText(
            "Equalizer APO was installed and linked to Clean APO. "
            "Parametric EQ updates are now routed to your selected Windows audio devices."
        )
        self.status_msg.setText("Connected to clean_apo_generated.txt")
        self.status_msg.setStyleSheet("color: #4cc38a; font-size: 11px; font-weight: 500;")
        self.status_msg.setVisible(True)

        self.install_btn.setVisible(False)
        self.launch_cfg_btn.setVisible(True)
        self.dismiss_btn.setText("Done")
        self.dismiss_btn.setObjectName("primaryInstallBtn")
        self.dismiss_btn.setStyle(self.dismiss_btn.style())
        self.sigInstalled.emit()

    def _launch_configurator(self) -> None:
        cfg = get_apo_configurator_path()
        if cfg:
            launch_apo_configurator(cfg)
        self.accept()

    def _on_dismiss(self) -> None:
        self._poll_timer.stop()
        self.accept()

    def done(self, r: int) -> None:
        if hasattr(self, "_poll_timer") and self._poll_timer.isActive():
            self._poll_timer.stop()
        super().done(r)


# -----------------------------------------------------------------------------
# SlidingStackedWidget: Smooth cubic horizontal slide transition
# -----------------------------------------------------------------------------
class SlidingStackedWidget(QtWidgets.QStackedWidget):
    """
    QStackedWidget featuring smooth cubic horizontal slide transitions
    with QEasingCurve.Type.InOutCubic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)
        self._anim.finished.connect(self._on_anim_finished)
        self._animating = False
        self._from_idx = 0
        self._to_idx = 0
        self._direction = 1  # 1: slide left (new enters from right), -1: slide right

    def slideToIndex(self, next_idx: int) -> None:
        if self._animating:
            self._anim.stop()
            self._on_anim_finished()

        current_idx = self.currentIndex()
        if next_idx == current_idx or next_idx < 0 or next_idx >= self.count():
            return

        width = self.frameRect().width()
        height = self.frameRect().height()
        if width <= 0 or height <= 0:
            self.setCurrentIndex(next_idx)
            return

        self._from_idx = current_idx
        self._to_idx = next_idx
        self._direction = 1 if next_idx > current_idx else -1

        next_widget = self.widget(next_idx)
        current_widget = self.widget(current_idx)

        start_x = width if self._direction == 1 else -width
        next_widget.setGeometry(start_x, 0, width, height)
        next_widget.show()
        next_widget.raise_()

        self._animating = True
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()

    def _on_anim_step(self, val: float) -> None:
        p = float(val)
        width = self.frameRect().width()
        offset = int(p * width * self._direction)

        current_widget = self.widget(self._from_idx)
        next_widget = self.widget(self._to_idx)

        if current_widget:
            current_widget.move(-offset, 0)
        if next_widget:
            start_x = width if self._direction == 1 else -width
            next_widget.move(start_x - offset, 0)

    def _on_anim_finished(self) -> None:
        self._animating = False
        current_widget = self.widget(self._from_idx)
        next_widget = self.widget(self._to_idx)
        if current_widget:
            current_widget.hide()
            current_widget.move(0, 0)
        if next_widget:
            next_widget.move(0, 0)
        self.setCurrentIndex(self._to_idx)


# -----------------------------------------------------------------------------
# Studio Black & Graphite QSS Stylesheet
# Divided strictly by 1px graphite hairlines (#27272a); 4px control radius.
# -----------------------------------------------------------------------------
BW_MINIMALIST_STYLESHEET = """
QMainWindow {
    background-color: transparent;
    border: none;
}

QWidget#centralWidget {
    background-color: #000000;
    border: 1px solid #27272a;
    border-radius: 8px;
}

QDialog {
    background-color: #0b0b0c;
    border: 1px solid #27272a;
    border-radius: 4px;
    font-family: "Segoe UI", "Inter", sans-serif;
    color: #ededed;
}

QWidget {
    background-color: #000000;
    color: #ededed;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}

/* Dividing lines: 1px graphite hairlines dividing chrome, sidebar, plot, and table */
QFrame#titleBarFrame {
    background-color: #0b0b0c;
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 1px solid #27272a;
}

QFrame#controlBarFrame {
    background-color: #0b0b0c;
    border: none;
    border-bottom: 1px solid #27272a;
}

QPushButton#topMenuBtn {
    background-color: transparent;
    color: #a1a1aa;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton#topMenuBtn:hover {
    background-color: #1c1c1f;
    color: #ededed;
}
QPushButton#topMenuBtn:pressed {
    background-color: #27272a;
    color: #ededed;
}

QPushButton#apoHealthBtn {
    background-color: transparent;
    color: #a1a1aa;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
    font-size: 11px;
    font-weight: 500;
}
QPushButton#apoHealthBtn:hover {
    background-color: #1c1c1f;
    border-color: #27272a;
    color: #ededed;
}

QFrame#headerFrame {
    background-color: #0b0b0c;
    border: none;
    border-bottom: 1px solid #27272a;
    border-radius: 0px;
}

QFrame#sidebarFrame {
    background-color: #000000;
    border: none;
    border-radius: 0px;
}

QFrame#tableFrame {
    background-color: #000000;
    border: none;
    border-radius: 0px;
}

QSplitter::handle {
    background-color: #27272a;
}
QSplitter::handle:hover {
    background-color: #3f3f46;
}

/* Buttons: quiet raised graphite default, single off-white primary action per pane */
QPushButton {
    background-color: #121214;
    color: #ededed;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 12px;
    min-height: 24px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1c1c1f;
    border-color: #3f3f46;
}
QPushButton:pressed {
    background-color: #27272a;
    border-color: #3f3f46;
}
QPushButton:disabled {
    background-color: #0b0b0c;
    color: #71717a;
    border-color: #27272a;
}
QPushButton#primaryBtn {
    background-color: #ededed;
    color: #0b0b0c;
    font-weight: 600;
    border: 1px solid #ededed;
    border-radius: 4px;
}
QPushButton#primaryBtn:hover {
    background-color: #ffffff;
    border-color: #ffffff;
}
QPushButton#primaryBtn:disabled {
    background-color: #121214;
    color: #71717a;
    border-color: #27272a;
}

/* StatusToggleButton states (exact 28px toolbar height, centered text) */
QPushButton#activeEqBtn {
    background-color: #1c1c1f;
    color: #ededed;
    font-weight: 500;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 0px 10px;
    min-height: 0px;
    text-align: center;
}
QPushButton#activeEqBtn:hover {
    background-color: #27272a;
    border-color: #3f3f46;
}
QPushButton#bypassedEqBtn {
    background-color: #121214;
    color: #71717a;
    font-weight: 500;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 0px 10px;
    min-height: 0px;
    text-align: center;
}
QPushButton#bypassedEqBtn:hover {
    background-color: #1c1c1f;
    color: #a1a1aa;
    border-color: #3f3f46;
}

QPushButton#dangerBtn {
    background-color: #121214;
    color: #e5484d;
    border: 1px solid #27272a;
    border-radius: 4px;
}
QPushButton#dangerBtn:hover {
    background-color: #1c1c1f;
    color: #e5484d;
    border-color: #e5484d;
}

QPushButton#globalSearchBtn {
    background-color: #1c1c1f;
    color: #ededed;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    padding: 4px 12px;
    min-height: 28px;
    text-align: center;
}
QPushButton#globalSearchBtn:hover {
    background-color: #27272a;
    border-color: #3f3f46;
    color: #ededed;
}

/* Scroll Area for AutoEQ Settings */
QScrollArea {
    background-color: #000000;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: #000000;
}

/* Input Fields & SpinBoxes */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #121214;
    color: #ededed;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 24px;
}
QSpinBox, QDoubleSpinBox {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
}
QComboBox {
    background-color: #121214;
    color: #ededed;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 24px 4px 8px;
    min-height: 24px;
}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #3f3f46;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #ededed;
}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background: transparent;
    border: none;
    width: 0px;
    height: 0px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: url('{CHEVRON_DOWN_PATH}');
    width: 10px;
    height: 10px;
}
QComboBox::down-arrow:hover {
    image: url('{CHEVRON_DOWN_HOVER_PATH}');
}
QComboBox:editable {
    background-color: #121214;
}
QComboBox:editable QLineEdit {
    background: transparent;
    border: none;
    color: #ededed;
    padding: 0px;
}
QComboBox QAbstractItemView {
    background-color: #121214;
    color: #ededed;
    selection-background-color: #1c1c1f;
    selection-color: #ededed;
    border: 1px solid #27272a;
    border-radius: 4px;
    outline: none;
    padding: 4px;
}

/* Table Widget (Architectural 0px outer frame, clean 1px hairlines) */
QTableWidget {
    background-color: #000000;
    alternate-background-color: #0b0b0c;
    border: none;
    border-radius: 0px;
    gridline-color: #27272a;
    selection-background-color: #1c1c1f;
    selection-color: #ededed;
    outline: none;
}
QTableWidget QLineEdit,
QTableWidget QSpinBox,
QTableWidget QDoubleSpinBox,
QTableWidget QComboBox {
    border-radius: 0px;
    background-color: transparent;
    border: 1px solid transparent;
}
QTableWidget QSpinBox:focus,
QTableWidget QDoubleSpinBox:focus {
    background-color: #121214;
    border: 1px solid #3f3f46;
}
QTableWidget QComboBox:hover,
QTableWidget QComboBox:focus {
    border: 1px solid transparent;
}
QHeaderView::section {
    background-color: #0b0b0c;
    color: #a1a1aa;
    font-weight: 500;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #27272a;
    padding: 6px 8px;
}

/* Checkbox */
QCheckBox {
    spacing: 8px;
    margin: 0px;
    padding: 0px;
    color: #a1a1aa;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3f3f46;
    background-color: #121214;
    border-radius: 4px;
}
QCheckBox::indicator:hover {
    border-color: #ededed;
}
QCheckBox::indicator:checked {
    background-color: #1c1c1f;
    border: 1px solid #ededed;
    image: url('{CHECK_WHITE_PATH}');
}

/* Context & Dropdown Menus */
QMenu {
    background-color: #121214;
    color: #ededed;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}
QMenu::item {
    padding: 6px 24px 6px 28px;
    border-radius: 4px;
    background-color: transparent;
    color: #ededed;
}
QMenu::item:selected {
    background-color: #1c1c1f;
    color: #ededed;
}
QMenu::item:disabled {
    color: #71717a;
}
QMenu::separator {
    height: 1px;
    background: #27272a;
    margin: 4px 8px;
}
QMenu::indicator {
    width: 14px;
    height: 14px;
    left: 8px;
}
QMenu::indicator:unchecked {
    image: none;
}
QMenu::indicator:checked, QMenu::indicator:checked:selected {
    image: url('{CHECK_WHITE_PATH}');
}

/* Scrollbars: Dark Graphite Track & Handle */
QScrollBar:vertical {
    border: none;
    background: #0b0b0c;
    width: 12px;
    margin: 16px 0 16px 0;
    border-radius: 0px;
}
QScrollBar::handle:vertical {
    background: #3f3f46;
    min-height: 28px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:vertical:hover {
    background: #71717a;
}
QScrollBar::handle:vertical:pressed {
    background: #a1a1aa;
}
QScrollBar::sub-line:vertical {
    border: none;
    background: #0b0b0c;
    height: 16px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}
QScrollBar::add-line:vertical {
    border: none;
    background: #0b0b0c;
    height: 16px;
    subcontrol-position: bottom;
    subcontrol-origin: margin;
}
QScrollBar::up-arrow:vertical {
    image: url('{SCROLL_UP_PATH}');
    width: 8px;
    height: 8px;
}
QScrollBar::down-arrow:vertical {
    image: url('{SCROLL_DOWN_PATH}');
    width: 8px;
    height: 8px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

QScrollBar:horizontal {
    border: none;
    background: #0b0b0c;
    height: 12px;
    margin: 0 16px 0 16px;
    border-radius: 0px;
}
QScrollBar::handle:horizontal {
    background: #3f3f46;
    min-width: 28px;
    border-radius: 4px;
    margin: 2px;
}
QScrollBar::handle:horizontal:hover {
    background: #71717a;
}
QScrollBar::handle:horizontal:pressed {
    background: #a1a1aa;
}
QScrollBar::sub-line:horizontal {
    border: none;
    background: #0b0b0c;
    width: 16px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal {
    border: none;
    background: #0b0b0c;
    width: 16px;
    subcontrol-position: right;
    subcontrol-origin: margin;
}
QScrollBar::left-arrow:horizontal {
    image: url('{SCROLL_LEFT_PATH}');
    width: 8px;
    height: 8px;
}
QScrollBar::right-arrow:horizontal {
    image: url('{SCROLL_RIGHT_PATH}');
    width: 8px;
    height: 8px;
}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}

/* Horizontal Sliders */
QSlider:horizontal {
    min-height: 24px;
    margin-left: 4px;
    margin-right: 4px;
}
QSlider::groove:horizontal {
    height: 2px;
    background: #27272a;
    margin: 0px 4px;
}
QSlider::sub-page:horizontal {
    background: #ededed;
    margin: 0px 4px;
}
QSlider::handle:horizontal {
    background: #ededed;
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
}
""".replace("{CHECK_WHITE_PATH}", CHECK_WHITE_PATH).replace("{CHECK_BLACK_PATH}", CHECK_BLACK_PATH).replace("{SCROLL_UP_PATH}", SCROLL_UP_PATH).replace("{SCROLL_DOWN_PATH}", SCROLL_DOWN_PATH).replace("{SCROLL_LEFT_PATH}", SCROLL_LEFT_PATH).replace("{SCROLL_RIGHT_PATH}", SCROLL_RIGHT_PATH).replace("{CHEVRON_DOWN_PATH}", CHEVRON_DOWN_PATH).replace("{CHEVRON_DOWN_HOVER_PATH}", CHEVRON_DOWN_HOVER_PATH)


class AnimatedPresetListWidget(QtWidgets.QListWidget):
    """
    Custom QListWidget featuring:
    - Smooth cubic-interpolated highlight indicator
    - Double-click inline renaming (no modal window)
    - Delete key shortcut to delete profile
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QListWidget { background-color: #000000; border: none; border-radius: 0px; outline: none; padding: 0px; }"
            "QListWidget::item { padding: 8px 12px; margin: 2px 0px; color: #a1a1aa; border: none; }"
            "QListWidget::item:hover { color: #ededed; background-color: #121214; }"
            "QListWidget::item:selected { color: #ededed; font-weight: 500; background-color: transparent; }"
        )
        self._current_indicator_y = 0.0
        self._target_indicator_y = 0.0
        self._indicator_height = 24.0
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(320)
        self._anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)
        self.currentItemChanged.connect(lambda *_: self.animate_to_current())
        self.verticalScrollBar().valueChanged.connect(lambda: self.animate_to_current(immediate=True))
        self._show_sync_timer = QtCore.QTimer(self)
        self._show_sync_timer.setSingleShot(True)
        self._show_sync_timer.setInterval(0)
        self._show_sync_timer.timeout.connect(lambda: self.animate_to_current(immediate=True))
        make_smooth_scroll(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._show_sync_timer.start()

    def _on_anim_step(self, val: float) -> None:
        self._current_indicator_y = float(val)
        self.viewport().update()

    def animate_to_current(self, immediate: bool = False) -> None:
        item = self.currentItem()
        if not item or item.isHidden():
            self._anim.stop()
            self.viewport().update()
            return
        rect = self.visualItemRect(item)
        if rect.isValid() and rect.height() > 0:
            target_y = float(rect.y())
            target_h = float(rect.height())

            # If already smoothly animating to this exact target, do not restart!
            if not immediate and self._anim.state() == QtCore.QAbstractAnimation.State.Running:
                if abs(self._target_indicator_y - target_y) < 1.0:
                    return

            self._target_indicator_y = target_y
            self._indicator_height = target_h
            if immediate or abs(self._current_indicator_y - self._target_indicator_y) < 1.0:
                self._anim.stop()
                self._current_indicator_y = self._target_indicator_y
                self.viewport().update()
            else:
                self._anim.stop()
                self._anim.setStartValue(self._current_indicator_y)
                self._anim.setEndValue(self._target_indicator_y)
                self._anim.start()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Delete:
            if self.state() != QtWidgets.QAbstractItemView.State.EditingState:
                main_win = self.window()
                if hasattr(main_win, "_delete_preset"):
                    main_win._delete_preset()
                    return
        super().keyPressEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        item = self.currentItem()
        if self.count() > 0 and item and not item.isHidden():
            rect = self.visualItemRect(item)
            if rect.isValid() and rect.height() > 0:
                # Sleek white indicator bar on left edge with rounded corners
                bar_rect = QtCore.QRectF(2, self._current_indicator_y + 4, 3, max(4.0, self._indicator_height - 8))
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(bar_rect, 1.5, 1.5)
                
                # Subtle wash background with rounded corners (5px)
                bg_rect = QtCore.QRectF(6, self._current_indicator_y + 1, self.viewport().width() - 10, self._indicator_height - 2)
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 16)))
                painter.drawRoundedRect(bg_rect, 5.0, 5.0)
        painter.end()
        super().paintEvent(event)


class SettingsDialog(QtWidgets.QDialog):
    """
    Settings dialog for Equalizer APO configuration, structured diagnostics,
    and one-click APO repair.
    """
    sigPathUpdated = QtCore.pyqtSignal(str)
    sigApoRepaired = QtCore.pyqtSignal()

    def __init__(self, current_path: str, parent=None, preset_manager: Optional[PresetManager] = None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Settings")
        self.resize(600, 380)
        self.setMinimumWidth(580)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + """
            QDialog {
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QWidget#settingsContent {
                background-color: #0b0b0c;
            }
            QFrame#apoDiagCard {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QFrame#apoDiagCard QLabel {
                background: transparent;
                border: none;
            }
            """
        )
        self.current_path = current_path
        self.preset_manager = preset_manager or getattr(parent, "preset_manager", None)
        self._init_ui()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Equalizer APO settings")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("settingsContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Output file path section
        path_lbl = QtWidgets.QLabel("Equalizer APO output file")
        path_lbl.setStyleSheet("color: #a1a1aa; font-size: 12px; font-weight: 500; background: transparent;")
        layout.addWidget(path_lbl)

        path_box = QtWidgets.QHBoxLayout()
        path_box.setSpacing(8)
        self.path_edit = QtWidgets.QLineEdit(self.current_path)
        self.path_edit.setFixedHeight(32)
        self.path_edit.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px;")
        path_box.addWidget(self.path_edit, stretch=1)

        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.setFixedHeight(32)
        browse_btn.clicked.connect(self._on_browse)
        path_box.addWidget(browse_btn)
        layout.addLayout(path_box)

        # Structured APO Diagnostics Card
        diag_card = QtWidgets.QFrame()
        diag_card.setObjectName("apoDiagCard")
        diag_layout = QtWidgets.QVBoxLayout(diag_card)
        diag_layout.setContentsMargins(12, 12, 12, 12)
        diag_layout.setSpacing(8)

        diag_head = QtWidgets.QHBoxLayout()
        diag_head.setSpacing(8)
        self.apo_stat_dot = StatusDotWidget("#4cc38a", filled=True, parent=diag_card)
        self.apo_stat_lbl = QtWidgets.QLabel("")
        self.apo_stat_lbl.setStyleSheet("color: #ededed; font-size: 12px; font-weight: 500;")
        diag_head.addWidget(self.apo_stat_dot)
        diag_head.addWidget(self.apo_stat_lbl, stretch=1)

        self.repair_btn = QtWidgets.QPushButton("Repair APO")
        self.repair_btn.setFixedHeight(28)
        self.repair_btn.setToolTip("Ensure Equalizer APO include directive and write access are healthy")
        self.repair_btn.clicked.connect(self._on_repair_apo)
        diag_head.addWidget(self.repair_btn)

        self.apo_action_btn = QtWidgets.QPushButton("Launch Configurator")
        self.apo_action_btn.setFixedHeight(28)
        self.apo_action_btn.clicked.connect(self._on_apo_action_clicked)
        diag_head.addWidget(self.apo_action_btn)
        diag_layout.addLayout(diag_head)

        # Structured key/value diagnostics grid
        diag_grid = QtWidgets.QGridLayout()
        diag_grid.setHorizontalSpacing(16)
        diag_grid.setVerticalSpacing(4)

        self._diag_vals: Dict[str, QtWidgets.QLabel] = {}
        diag_rows = [
            ("installed", "Installation"),
            ("include", "Root config.txt"),
            ("write", "Write access"),
            ("device", "Active output"),
        ]
        for idx, (key, title) in enumerate(diag_rows):
            k_lbl = QtWidgets.QLabel(title)
            k_lbl.setStyleSheet("color: #71717a; font-size: 11px;")
            v_lbl = QtWidgets.QLabel("")
            v_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; color: #a1a1aa; font-size: 11px;")
            row_i = idx // 2
            col_base = (idx % 2) * 2
            diag_grid.addWidget(k_lbl, row_i, col_base)
            diag_grid.addWidget(v_lbl, row_i, col_base + 1)
            self._diag_vals[key] = v_lbl
        diag_grid.setColumnStretch(1, 1)
        diag_grid.setColumnStretch(3, 1)
        diag_layout.addLayout(diag_grid)

        self.issues_lbl = QtWidgets.QLabel("")
        self.issues_lbl.setWordWrap(True)
        self.issues_lbl.setStyleSheet("color: #d99b38; font-size: 11px;")
        self.issues_lbl.setVisible(False)
        diag_layout.addWidget(self.issues_lbl)

        layout.addWidget(diag_card)

        # Transient status feedback (hidden when empty to prevent dead vertical gap)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 11px; font-weight: 500; background: transparent;")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # Utility button row (Row 1 — fits cleanly inside 600px without overflow)
        util_row = QtWidgets.QHBoxLayout()
        util_row.setSpacing(8)

        test_btn = QtWidgets.QPushButton("Test permissions")
        test_btn.setFixedHeight(28)
        test_btn.clicked.connect(self._test_permissions)
        util_row.addWidget(test_btn)

        fix_btn = QtWidgets.QPushButton("Grant permissions (UAC)")
        fix_btn.setFixedHeight(28)
        fix_btn.setToolTip("Request administrator privileges via UAC to grant write access to Equalizer APO folder")
        fix_btn.clicked.connect(self._fix_permissions)
        util_row.addWidget(fix_btn)

        default_btn = QtWidgets.QPushButton("Default path")
        default_btn.setFixedHeight(28)
        default_btn.clicked.connect(self._reset_default_path)
        util_row.addWidget(default_btn)

        shortcut_btn = QtWidgets.QPushButton("Create shortcuts")
        shortcut_btn.setFixedHeight(28)
        shortcut_btn.setToolTip("Create or update Clean APO shortcuts on Desktop and Start Menu")
        shortcut_btn.clicked.connect(self._create_shortcuts)
        util_row.addWidget(shortcut_btn)

        util_row.addStretch()
        layout.addLayout(util_row)

        layout.addStretch()

        # Dialog footer row (Row 2)
        footer_row = QtWidgets.QHBoxLayout()
        footer_row.setSpacing(8)
        footer_row.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedHeight(32)
        cancel_btn.clicked.connect(self.reject)
        footer_row.addWidget(cancel_btn)

        save_btn = QtWidgets.QPushButton("Apply")
        save_btn.setObjectName("primaryBtn")
        save_btn.setFixedHeight(32)
        save_btn.clicked.connect(self._on_save)
        footer_row.addWidget(save_btn)

        layout.addLayout(footer_row)
        main_layout.addWidget(content)

        self._refresh_apo_status()

    def _set_status(self, text: str, color: str = "#a1a1aa") -> None:
        clean = (text or "").strip().rstrip(".")
        self.status_label.setText(clean)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 500; background: transparent;")
        self.status_label.setVisible(bool(clean))

    def _reset_default_path(self) -> None:
        self.path_edit.setText(DEFAULT_APO_PATH)
        self._refresh_apo_status()

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Equalizer APO config file",
            self.path_edit.text(),
            "Equalizer APO Config (*.txt);;All Files (*.*)",
        )
        if path:
            self.path_edit.setText(path)
            self._refresh_apo_status()

    def _get_diagnostics(self) -> Dict[str, Any]:
        path = self.path_edit.text().strip() if hasattr(self, "path_edit") else self.current_path
        if self.preset_manager is not None and path == self.preset_manager.apo_target_path:
            return self.preset_manager.diagnose_apo()
        active_dev = getattr(self.preset_manager, "active_device_name", "all") if self.preset_manager else "all"
        return diagnose_apo_setup(path or DEFAULT_APO_PATH, active_dev or "all")

    def _refresh_apo_status(self) -> None:
        diag = self._get_diagnostics()
        self._apo_installed = bool(diag.get("apo_installed", False))
        level = diag.get("status_level", "warning")
        color_map = {"ok": "#4cc38a", "warning": "#d99b38", "error": "#e5484d"}
        dot_color = color_map.get(level, "#d99b38")

        self.apo_stat_dot.set_status(dot_color, filled=self._apo_installed)
        summary = str(diag.get("status_summary") or ("Equalizer APO ready" if self._apo_installed else "Equalizer APO not installed")).rstrip(".")
        self.apo_stat_lbl.setText(summary)
        self.apo_stat_lbl.setStyleSheet(f"color: {'#ededed' if level == 'ok' else dot_color}; font-size: 12px; font-weight: 500;")

        if self._apo_installed:
            self.apo_action_btn.setText("Launch Configurator")
            self.apo_action_btn.setToolTip("Open Equalizer APO Device Selector / Configurator")
        else:
            self.apo_action_btn.setText("Install Equalizer APO")
            self.apo_action_btn.setToolTip("Install bundled Equalizer APO x64 installer")

        self._diag_vals["installed"].setText("Detected" if self._apo_installed else "Not found")
        self._diag_vals["installed"].setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: {'#4cc38a' if self._apo_installed else '#e5484d'};"
        )

        dup_inc = int(diag.get("duplicate_includes", 0))
        inc_ok = bool(diag.get("include_present", False))
        if inc_ok and dup_inc == 0:
            inc_text, inc_col = "Linked", "#4cc38a"
        elif inc_ok and dup_inc > 0:
            inc_text, inc_col = f"Linked ({dup_inc} dup)", "#d99b38"
        else:
            inc_text, inc_col = "Missing include", "#d99b38"
        self._diag_vals["include"].setText(inc_text)
        self._diag_vals["include"].setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: {inc_col};")

        can_write = bool(diag.get("can_write_apo", False))
        using_fb = bool(diag.get("using_fallback", False))
        if can_write and not using_fb:
            w_text, w_col = "Writable", "#4cc38a"
        elif using_fb:
            w_text, w_col = "Fallback file", "#d99b38"
        else:
            w_text, w_col = "Read-only", "#e5484d"
        self._diag_vals["write"].setText(w_text)
        self._diag_vals["write"].setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: {w_col};")

        dev_raw = str(diag.get("active_device") or "all")
        dev_clean = clean_apo_device_name(dev_raw)
        dev_disp = "All endpoints" if dev_clean.lower() == "all" else dev_clean
        if len(dev_disp) > 26:
            dev_disp = dev_disp[:25] + "..."
        self._diag_vals["device"].setText(dev_disp)
        self._diag_vals["device"].setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #ededed;")

        issues = [str(i).rstrip(".") for i in (diag.get("issues") or []) if i]
        if issues:
            self.issues_lbl.setText("Issues: " + " · ".join(issues))
            self.issues_lbl.setVisible(True)
        else:
            self.issues_lbl.setText("")
            self.issues_lbl.setVisible(False)

    def _on_repair_apo(self) -> None:
        path = self.path_edit.text().strip() or DEFAULT_APO_PATH
        if self.preset_manager is not None:
            self.preset_manager.apo_target_path = path
            ok, msg, _diag = self.preset_manager.repair_apo()
        else:
            ok, msg, _diag = repair_apo_setup(path, "all")
        self._set_status(msg, "#4cc38a" if ok else "#e5484d")
        self._refresh_apo_status()
        self.sigApoRepaired.emit()
        parent_win = self.parent()
        if parent_win is not None and hasattr(parent_win, "_refresh_apo_health_indicator"):
            parent_win._refresh_apo_health_indicator()

    def _on_apo_action_clicked(self) -> None:
        if getattr(self, "_apo_installed", False):
            ok, msg = launch_apo_configurator()
            if not ok:
                self._set_status(f"Configurator error: {msg}", "#e5484d")
        else:
            self._open_installer_dialog()

    def _open_installer_dialog(self) -> None:
        dlg = EqualizerApoInstallDialog(self)
        parent_win = self.parent()
        if parent_win is not None and hasattr(parent_win, "_on_apo_installed_success"):
            dlg.sigInstalled.connect(parent_win._on_apo_installed_success)
        dlg.exec()
        dlg.deleteLater()
        if is_equalizer_apo_installed():
            self.path_edit.setText(DEFAULT_APO_PATH)
        self._refresh_apo_status()
        self.sigApoRepaired.emit()

    def _test_permissions(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            self._set_status("Output path cannot be empty", "#e5484d")
            return
        if can_write_to_path(path):
            self._set_status("Write access confirmed", "#4cc38a")
        else:
            self._set_status("No write access to target path or folder", "#e5484d")
        self._refresh_apo_status()

    def _fix_permissions(self) -> None:
        folder = os.path.dirname(self.path_edit.text().strip())
        ok, msg = grant_apo_directory_permissions(folder)
        if ok:
            self._set_status("Permissions granted via UAC", "#4cc38a")
            self._refresh_apo_status()
            self.sigApoRepaired.emit()
        else:
            self._set_status(f"Could not elevate: {msg}", "#e5484d")

    def _create_shortcuts(self) -> None:
        ok, msg = create_windows_shortcuts()
        if ok:
            self._set_status("Shortcuts created on Desktop and Start Menu", "#4cc38a")
        else:
            self._set_status(f"Shortcut error: {msg}", "#e5484d")

    def _on_save(self) -> None:
        new_path = self.path_edit.text().strip()
        if new_path:
            self.sigPathUpdated.emit(new_path)
        self.accept()


# -----------------------------------------------------------------------------
# Global Squiglink Search Dialog
# -----------------------------------------------------------------------------
class GlobalSearchDialog(QtWidgets.QDialog):
    """
    Global model search across indexed Squiglink measurement databases.
    Displays IEM & headphone models, measuring reviewer, coupler rig badge,
    offline cache status, and available target curves.
    """
    def __init__(self, squig_client: SquigClient, initial_query: str = "", preset_manager: Optional[PresetManager] = None, parent=None):
        super().__init__(parent)
        self.client = squig_client
        self.preset_manager = preset_manager
        self.offline_db = get_offline_db()
        self.selected_result: Optional[GlobalSearchResult] = None
        self.load_role: str = "source"
        self._current_results: List[GlobalSearchResult] = []

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Global model search")
        self.resize(1040, 640)
        self.setMinimumSize(880, 520)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + """
            QDialog {
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
            }
            QWidget#searchContent {
                background-color: #0b0b0c;
            }
            QTableWidget {
                background-color: #000000;
                alternate-background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                gridline-color: #27272a;
                color: #ededed;
                font-size: 12px;
                selection-background-color: #1c1c1f;
                selection-color: #ededed;
            }
            QHeaderView::section {
                background-color: #121214;
                color: #a1a1aa;
                border: none;
                border-bottom: 1px solid #27272a;
                border-right: 1px solid #27272a;
                padding: 6px 8px;
                font-weight: 500;
                font-size: 11px;
            }
            QFrame#searchInfoFrame {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QFrame#searchInfoFrame QLabel {
                background: transparent;
                border: none;
            }
            """
        )

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Global model search")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("searchContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        # Search Controls (4px scale, aligned labels)
        search_grid = QtWidgets.QGridLayout()
        search_grid.setHorizontalSpacing(12)
        search_grid.setVerticalSpacing(8)

        model_lbl = QtWidgets.QLabel("Model")
        model_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        search_grid.addWidget(model_lbl, 0, 0)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setFixedHeight(32)
        self.search_input.setPlaceholderText("Model or brand")
        self.search_input.setClearButtonEnabled(True)
        if initial_query:
            self.search_input.setText(initial_query)
        search_grid.addWidget(self.search_input, 0, 1, 1, 2)

        target_lbl = QtWidgets.QLabel("Target")
        target_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        search_grid.addWidget(target_lbl, 1, 0)

        self.target_input = QtWidgets.QLineEdit()
        self.target_input.setFixedHeight(32)
        self.target_input.setPlaceholderText("Target curves (comma-separated)")
        self.target_input.setClearButtonEnabled(True)
        search_grid.addWidget(self.target_input, 1, 1)

        self.type_filter_combo = QtWidgets.QComboBox()
        self.type_filter_combo.setFixedHeight(32)
        self.type_filter_combo.setFixedWidth(132)
        self.type_filter_combo.addItem("All types", "all")
        self.type_filter_combo.addItem("Headphones", "headphones")
        self.type_filter_combo.addItem("IEMs", "iems")
        self.type_filter_combo.currentIndexChanged.connect(self._on_search_query_changed)
        search_grid.addWidget(self.type_filter_combo, 1, 2)

        search_grid.setColumnStretch(1, 1)
        layout.addLayout(search_grid)

        # Status & Stats Bar
        self.stats_label = QtWidgets.QLabel("Search indexed reviewer catalogs")
        self.stats_label.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; color: #71717a; font-size: 11px; background: transparent;")
        layout.addWidget(self.stats_label)

        # Results Table (6 columns; crisp rig badge + cache status + unclipped targets)
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Model",
            "Type",
            "Reviewer",
            "Rig",
            "Cache",
            "Available targets",
        ])
        hdr = self.table.horizontalHeader()
        hdr.setMinimumSectionSize(56)
        hdr.setStretchLastSection(True)
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 240)
        self.table.setColumnWidth(2, 200)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        make_smooth_scroll(self.table)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, stretch=1)

        # Detail Box for Selected Result (properly scoped #searchInfoFrame)
        info_frame = QtWidgets.QFrame()
        info_frame.setObjectName("searchInfoFrame")
        info_layout = QtWidgets.QVBoxLayout(info_frame)
        info_layout.setContentsMargins(12, 10, 12, 10)
        info_layout.setSpacing(4)

        self.selected_model_lbl = QtWidgets.QLabel("Select a measurement to preview coupler rig and target curves")
        self.selected_model_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed;")
        info_layout.addWidget(self.selected_model_lbl)

        self.targets_list_lbl = QtWidgets.QLabel("Targets: none")
        self.targets_list_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        self.targets_list_lbl.setWordWrap(True)
        info_layout.addWidget(self.targets_list_lbl)

        layout.addWidget(info_frame)

        # Bottom Actions (one primary off-white button: Load selected)
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(32)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.sync_db_btn = QtWidgets.QPushButton("Offline library")
        self.sync_db_btn.setFixedHeight(32)
        self.sync_db_btn.setToolTip("Download and manage local offline measurement and target curves")
        self.sync_db_btn.clicked.connect(self._open_offline_sync)
        btn_layout.addWidget(self.sync_db_btn)

        btn_layout.addStretch()

        self.load_target_btn = QtWidgets.QPushButton("Load as target")
        self.load_target_btn.setFixedHeight(32)
        self.load_target_btn.setEnabled(False)
        self.load_target_btn.setToolTip("Load this measurement into AutoEQ as the target response to match")
        self.load_target_btn.clicked.connect(self._on_load_target_clicked)
        btn_layout.addWidget(self.load_target_btn)

        self.load_btn = QtWidgets.QPushButton("Load selected")
        self.load_btn.setObjectName("primaryBtn")
        self.load_btn.setFixedHeight(32)
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        btn_layout.addWidget(self.load_btn)

        layout.addLayout(btn_layout)

        # Search debounce timer (120 ms)
        self._search_timer = QtCore.QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(120)
        self._search_timer.timeout.connect(self._execute_search)

        self.search_input.textChanged.connect(self._on_search_query_changed)
        self.target_input.textChanged.connect(self._on_search_query_changed)

        # Populate initial search
        self._execute_search()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _on_search_query_changed(self) -> None:
        self._search_timer.start(120)

    def _execute_search(self) -> None:
        query = self.search_input.text().strip()
        target_query = self.target_input.text().strip()
        type_filter = self.type_filter_combo.currentData() or "all"
        favs = self.preset_manager.get_favorite_databases() if self.preset_manager else []
        category_arg = None
        if type_filter == "headphones":
            category_arg = "Headphones"
        elif type_filter == "iems":
            category_arg = "IEMs"

        cached_keys = None
        try:
            cached_keys = self.offline_db.get_cached_measurement_keys()
        except Exception:
            cached_keys = None

        try:
            results = self.client.search_global_catalog(
                query,
                target_query=target_query,
                limit=150,
                category=category_arg,
                favorite_providers=favs,
                cached_keys=cached_keys,
            )
        except TypeError:
            results = self.client.search_global_catalog(query, target_query=target_query, limit=150)

        if type_filter != "all":
            results = [r for r in results if r.db_type.lower() == type_filter.lower()]
        results = results[:150]

        def is_fav(r: GlobalSearchResult) -> bool:
            r_id = _resolve_db_id(self.client, r.base_url).lower()
            return any(
                f.lower() == r_id or f.lower() in r.base_url.lower() or f.lower() in r.provider_name.lower()
                for f in favs
            )

        self._current_results = results
        self.table.setRowCount(len(results))
        self.table.blockSignals(True)
        mono_font = _make_mono_font(9)

        for row, res in enumerate(results):
            # 0. Model Name
            m_text = res.full_name or res.file_name
            item_model = QtWidgets.QTableWidgetItem(m_text)
            item_model.setData(QtCore.Qt.ItemDataRole.UserRole, res)
            self.table.setItem(row, 0, item_model)

            # 1. Type
            item_type = QtWidgets.QTableWidgetItem(res.db_type)
            item_type.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_type.setForeground(QtGui.QBrush(QtGui.QColor("#a1a1aa")))
            self.table.setItem(row, 1, item_type)

            # 2. Reviewer / Provider (quiet amber highlight if pinned favorite, no star unicode)
            res_fav = is_fav(res)
            prov_disp = f"{res.provider_name} (Pinned)" if res_fav else res.provider_name
            item_prov = QtWidgets.QTableWidgetItem(prov_disp)
            if res_fav:
                item_prov.setForeground(QtGui.QBrush(QtGui.QColor("#d99b38")))
            else:
                item_prov.setForeground(QtGui.QBrush(QtGui.QColor("#ededed")))
            self.table.setItem(row, 2, item_prov)

            # 3. Crisp Rig Badge
            badge_txt = getattr(res, "rig_badge", "") or get_rig_badge(res.rig, res.base_url, res.db_type) or res.rig
            item_rig = QtWidgets.QTableWidgetItem(badge_txt)
            item_rig.setFont(mono_font)
            item_rig.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_rig.setToolTip(res.rig or badge_txt)
            item_rig.setForeground(QtGui.QBrush(QtGui.QColor("#ededed" if badge_txt in ("5128", "45CA", "43AG") else "#a1a1aa")))
            self.table.setItem(row, 3, item_rig)

            # 4. Offline Cache Indicator
            is_cached = bool(getattr(res, "is_offline_cached", False))
            item_cache = QtWidgets.QTableWidgetItem("Offline" if is_cached else "Online")
            item_cache.setFont(mono_font)
            item_cache.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_cache.setForeground(QtGui.QBrush(QtGui.QColor("#4cc38a" if is_cached else "#71717a")))
            self.table.setItem(row, 4, item_cache)

            # 5. Targets
            if res.targets:
                t_str = ", ".join(res.targets[:5])
                if len(res.targets) > 5:
                    t_str += f" (+{len(res.targets) - 5})"
            else:
                t_str = "Provider default targets"
            item_targets = QtWidgets.QTableWidgetItem(t_str)
            item_targets.setForeground(QtGui.QBrush(QtGui.QColor("#a1a1aa")))
            item_targets.setToolTip(", ".join(res.targets) if res.targets else t_str)
            self.table.setItem(row, 5, item_targets)

        self.table.blockSignals(False)

        count = len(results)
        filters_applied = []
        if query:
            filters_applied.append(f"model '{query}'")
        if target_query:
            filters_applied.append(f"target '{target_query}'")
        if type_filter != "all":
            filters_applied.append(f"type '{type_filter}'")

        if filters_applied:
            self.stats_label.setText(f"{count} matches for {', '.join(filters_applied)}")
        else:
            self.stats_label.setText(f"Showing {count} entries - type above to filter catalog")

        if results:
            self.table.selectRow(0)
            self._update_selected_info(results[0])
        else:
            self._update_selected_info(None)

    def _on_table_selection_changed(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self._current_results):
            res = self._current_results[row]
            self._update_selected_info(res)
        else:
            self._update_selected_info(None)

    def _update_selected_info(self, res: Optional[GlobalSearchResult]) -> None:
        if res:
            self.selected_result = res
            self.load_btn.setEnabled(True)
            self.load_target_btn.setEnabled(True)
            rig_badge = getattr(res, "rig_badge", "") or get_rig_badge(res.rig, res.base_url, res.db_type)
            rig_str = f"{res.rig} [{rig_badge}]" if rig_badge and rig_badge not in (res.rig or "") else (res.rig or rig_badge)
            cache_str = " · Offline cached" if getattr(res, "is_offline_cached", False) else ""
            self.selected_model_lbl.setText(
                f"{res.full_name or res.file_name}  ·  {res.provider_name} ({rig_str}){cache_str}"
            )
            if res.targets:
                t_preview = ", ".join(res.targets)
                self.targets_list_lbl.setText(f"Targets ({len(res.targets)}): {t_preview}")
            else:
                self.targets_list_lbl.setText("Targets: provider default target curves")
        else:
            self.selected_result = None
            self.load_btn.setEnabled(False)
            self.load_target_btn.setEnabled(False)
            self.selected_model_lbl.setText("No measurement selected")
            self.targets_list_lbl.setText("Targets: none")

    def _on_row_double_clicked(self) -> None:
        if self.selected_result:
            self.load_role = "source"
            self.accept()

    def _on_load_clicked(self) -> None:
        if self.selected_result:
            self.load_role = "source"
            self.accept()

    def _on_load_target_clicked(self) -> None:
        if self.selected_result:
            self.load_role = "target"
            self.accept()

    def _open_offline_sync(self) -> None:
        dlg = OfflineSyncDialog(self.client, self)
        dlg.exec()
        dlg.deleteLater()
        self._execute_search()


class OfflineSyncDialog(QtWidgets.QDialog):
    """
    Studio modal dialog for selectively syncing the offline measurement library
    and managing local SQLite cache storage.
    """
    def __init__(self, squig_client: SquigClient, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.client = squig_client
        self.offline_db = get_offline_db()
        target = parent
        while target is not None and not hasattr(target, "offline_sync_worker"):
            target = target.parent()
        self.main_win = target if target is not None else (parent.window() if parent else None)
        self.preset_manager = getattr(self.main_win, "preset_manager", None) or getattr(parent, "preset_manager", None)

        self._catalog_model_total = 0
        self._catalog_target_total = 0
        try:
            cat = self.client.get_global_catalog()
            self._catalog_model_total = sum(len(c.get("phones", [])) for c in cat)
            self._catalog_target_total = sum(len(c.get("targets", [])) for c in cat)
        except Exception:
            pass

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Offline database sync")
        self.setFixedSize(620, 600)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + """
            QDialog {
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
            }
            QWidget#syncContent {
                background-color: #0b0b0c;
            }
            QFrame#statCard {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QFrame#statCard QLabel {
                background: transparent;
                border: none;
            }
            QProgressBar {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                height: 12px;
                text-align: center;
                font-size: 10px;
                color: #a1a1aa;
            }
            QProgressBar::chunk {
                background-color: #ededed;
                border-radius: 3px;
            }
            QPushButton#stopActionBtn {
                background-color: #1c1c1f;
                color: #e5484d;
                border: 1px solid #e5484d;
                font-weight: 500;
            }
            QPushButton#stopActionBtn:hover {
                background-color: #27272a;
                color: #e5484d;
            }
            """
        )

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Offline database sync")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("syncContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        desc = QtWidgets.QLabel(
            "Download measurement curves and targets into local SQLite storage for offline AutoEQ loading."
        )
        desc.setStyleSheet("font-size: 12px; color: #a1a1aa; background: transparent;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Stats Row
        stats_frame = QtWidgets.QFrame()
        stats_frame.setObjectName("statCard")
        stats_layout = QtWidgets.QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(16, 12, 16, 12)
        stats_layout.setSpacing(0)

        def _make_stat_col(label_text: str, init_val: str) -> Tuple[QtWidgets.QVBoxLayout, QtWidgets.QLabel]:
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl = QtWidgets.QLabel(label_text)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #71717a;")
            val = QtWidgets.QLabel(init_val)
            val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            val.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 14px; font-weight: 600; color: #ededed;")
            col.addWidget(lbl)
            col.addWidget(val)
            return col, val

        col1, self.meas_stat_val = _make_stat_col("Cached models", "0")
        stats_layout.addLayout(col1, 1)

        div1 = QtWidgets.QFrame()
        div1.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        div1.setStyleSheet("background-color: #27272a; max-width: 1px; border: none; margin: 2px 12px;")
        stats_layout.addWidget(div1)

        col2, self.target_stat_val = _make_stat_col("Cached targets", "0")
        stats_layout.addLayout(col2, 1)

        div2 = QtWidgets.QFrame()
        div2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        div2.setStyleSheet("background-color: #27272a; max-width: 1px; border: none; margin: 2px 12px;")
        stats_layout.addWidget(div2)

        col3, self.size_stat_val = _make_stat_col("Disk usage", "0.0 MB")
        stats_layout.addLayout(col3, 1)

        layout.addWidget(stats_frame)

        # Selective Sync Scope Selection
        scope_lbl = QtWidgets.QLabel("Sync scope")
        scope_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        layout.addWidget(scope_lbl)

        self.scope_list = AnimatedPresetListWidget(self)
        self.scope_list.setFixedHeight(228)
        self.scope_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scope_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scope_list.setStyleSheet(
            "QListWidget { background-color: #000000; border: 1px solid #27272a; border-radius: 4px; outline: none; padding: 4px; }"
            "QListWidget::item { padding: 6px 12px; margin: 1px 0px; color: #a1a1aa; border: none; font-size: 12px; }"
            "QListWidget::item:hover { color: #ededed; background-color: #121214; }"
            "QListWidget::item:selected { color: #ededed; font-weight: 500; background-color: transparent; }"
        )

        scopes = [
            ("Favorites only  -  Pinned reviewer databases", {"id": "favorites", "scope": "full", "favorites_only": True}),
            ("Essential  -  Core reviewer catalogs and reference targets", {"id": "essential", "scope": "essential"}),
            ("IEMs  -  In-ear monitor measurement catalogs", {"id": "iems", "scope": "full", "categories": ["IEMs"]}),
            ("Headphones  -  Over-ear headphone measurement catalogs", {"id": "headphones", "scope": "full", "categories": ["Headphones"]}),
            ("B&K 5128  -  High-frequency B&K 5128 coupler databases", {"id": "bk5128", "scope": "full", "rig_filter": "5128"}),
            ("Full catalog  -  All indexed reviewer databases and targets", {"id": "full", "scope": "full"}),
        ]
        default_item = None
        for label, payload in scopes:
            item = QtWidgets.QListWidgetItem(label)
            item.setSizeHint(QtCore.QSize(0, 36))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, payload)
            self.scope_list.addItem(item)
            if payload["id"] == "essential":
                default_item = item

        if default_item is not None:
            self.scope_list.setCurrentItem(default_item)
        layout.addWidget(self.scope_list)

        # Progress Section
        prog_box = QtWidgets.QVBoxLayout()
        prog_box.setSpacing(4)

        prog_head = QtWidgets.QHBoxLayout()
        self.phase_lbl = QtWidgets.QLabel("Ready to sync")
        self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed; background: transparent;")
        prog_head.addWidget(self.phase_lbl, stretch=1)

        self.eta_lbl = QtWidgets.QLabel("")
        self.eta_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; background: transparent;")
        prog_head.addWidget(self.eta_lbl)
        prog_box.addLayout(prog_head)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_box.addWidget(self.progress_bar)

        self.detail_lbl = QtWidgets.QLabel("Select a scope and start sync to cache curves locally")
        self.detail_lbl.setStyleSheet("font-size: 11px; color: #71717a; background: transparent;")
        self.detail_lbl.setWordWrap(True)
        prog_box.addWidget(self.detail_lbl)

        layout.addLayout(prog_box)

        # Background Download Note (quiet, no italic)
        self.disclaimer_lbl = QtWidgets.QLabel(
            "Closing this dialog leaves an active sync running in the background."
        )
        self.disclaimer_lbl.setStyleSheet("font-size: 11px; color: #71717a; background: transparent;")
        self.disclaimer_lbl.setWordWrap(True)
        layout.addWidget(self.disclaimer_lbl)

        layout.addStretch()

        # Action Buttons (single primary off-white button: Start sync)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.compact_btn = QtWidgets.QPushButton("Compact DB")
        self.compact_btn.setFixedHeight(32)
        self.compact_btn.setToolTip("Reclaim unused SQLite space and checkpoint WAL journal")
        self.compact_btn.clicked.connect(self._compact_database)
        btn_row.addWidget(self.compact_btn)

        self.delete_btn = QtWidgets.QPushButton("Clear cache")
        self.delete_btn.setObjectName("dangerBtn")
        self.delete_btn.setFixedHeight(32)
        self.delete_btn.setToolTip("Delete local offline measurement cache")
        self.delete_btn.clicked.connect(self._delete_database)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch()

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setFixedHeight(32)
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)

        self.start_btn = QtWidgets.QPushButton("Start sync")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setFixedHeight(32)
        self.start_btn.clicked.connect(self._toggle_sync)
        btn_row.addWidget(self.start_btn)

        layout.addLayout(btn_row)

        self._attached_worker: Optional[OfflineSyncWorker] = None
        self._refresh_stats()

        # Check if main window already has an active background worker
        if self.main_win and getattr(self.main_win, "offline_sync_worker", None):
            w = self.main_win.offline_sync_worker
            if w.isRunning():
                self._attach_worker(w)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def done(self, r: int) -> None:
        self._detach_worker()
        super().done(r)

    def _refresh_stats(self) -> None:
        stats = self.offline_db.get_stats()
        m_cnt = stats.get("measurements", 0)
        t_cnt = stats.get("targets", 0)
        size_mb = stats.get("size_mb", 0.0)
        if self._catalog_model_total > 0:
            self.meas_stat_val.setText(f"{m_cnt:,} / {self._catalog_model_total:,}")
        else:
            self.meas_stat_val.setText(f"{m_cnt:,}")
        if self._catalog_target_total > 0:
            self.target_stat_val.setText(f"{t_cnt:,} / {self._catalog_target_total:,}")
        else:
            self.target_stat_val.setText(f"{t_cnt:,}")
        self.size_stat_val.setText(f"{float(size_mb):.1f} MB")

    def _compact_database(self) -> None:
        worker = getattr(self.main_win, "offline_sync_worker", None) if self.main_win else None
        if worker and worker.isRunning():
            self.detail_lbl.setText("Stop the active sync before compacting the database")
            return
        ok, msg = self.offline_db.compact_database()
        self._refresh_stats()
        self.phase_lbl.setText("Database compacted" if ok else "Compact failed")
        self.phase_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 500; color: {'#4cc38a' if ok else '#e5484d'}; background: transparent;"
        )
        self.detail_lbl.setText((msg or "").rstrip("."))
        if self.main_win and hasattr(self.main_win, "_update_offline_db_badge"):
            self.main_win._update_offline_db_badge()

    def _delete_database(self) -> None:
        worker = getattr(self.main_win, "offline_sync_worker", None) if self.main_win else None
        if worker and worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Sync active",
                "Stop the active sync before clearing the local cache.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear offline cache",
            "Delete all cached measurements and target curves from local storage?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.offline_db.clear_all()
            self._refresh_stats()
            self.progress_bar.setValue(0)
            self.eta_lbl.setText("")
            self.phase_lbl.setText("Cache cleared")
            self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed; background: transparent;")
            self.detail_lbl.setText("All local cached curves and targets were removed")
            if self.main_win and hasattr(self.main_win, "_update_offline_db_badge"):
                self.main_win._update_offline_db_badge()

    def _detach_worker(self) -> None:
        worker = self._attached_worker
        self._attached_worker = None
        if worker is None or sip.isdeleted(worker):
            return
        sigs = [
            (worker.sigPhaseChanged, self._on_phase_changed),
            (worker.sigProgress, self._on_progress),
            (worker.sigStatsUpdated, self._on_stats_updated),
            (worker.sigFinished, self._on_finished),
            (worker.sigError, self._on_error),
        ]
        if hasattr(worker, "sigProgressDetail"):
            sigs.append((worker.sigProgressDetail, self._on_progress_detail))
        for sig, slot in sigs:
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def _attach_worker(self, worker: OfflineSyncWorker) -> None:
        self._detach_worker()
        self._attached_worker = worker
        worker.sigPhaseChanged.connect(self._on_phase_changed)
        if hasattr(worker, "sigProgressDetail"):
            worker.sigProgressDetail.connect(self._on_progress_detail)
        else:
            worker.sigProgress.connect(self._on_progress)
        worker.sigStatsUpdated.connect(self._on_stats_updated)
        worker.sigFinished.connect(self._on_finished)
        worker.sigError.connect(self._on_error)

        self.start_btn.setEnabled(not getattr(worker, "_abort", False))
        self.start_btn.setText("Stopping..." if getattr(worker, "_abort", False) else "Stop sync")
        self.start_btn.setObjectName("stopActionBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.phase_lbl.setText("Cancelling sync..." if getattr(worker, "_abort", False) else "Syncing curves...")
        self.scope_list.setEnabled(False)
        self.compact_btn.setEnabled(False)

    def _resolve_rig_db_ids(self, rig_token: str) -> List[str]:
        ids: List[str] = []
        tok = rig_token.lower()
        try:
            for c in self.client.get_global_catalog():
                rig = str(c.get("rig", "") or "")
                b_url = str(c.get("base_url", "") or "")
                cat_type = str(c.get("db_type", "") or "")
                badge = get_rig_badge(rig, b_url, cat_type).lower()
                if tok in rig.lower() or tok in b_url.lower() or badge == tok:
                    db_id = _resolve_db_id(self.client, b_url)
                    if db_id:
                        ids.append(db_id)
        except Exception:
            pass
        return ids

    def _toggle_sync(self) -> None:
        worker = getattr(self.main_win, "offline_sync_worker", None) if self.main_win else None
        if worker and worker.isRunning():
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Stopping...")
            self.phase_lbl.setText("Cancelling sync...")
            worker.cancel()
            return

        cur_item = self.scope_list.currentItem()
        payload = cur_item.data(QtCore.Qt.ItemDataRole.UserRole) if cur_item else {"id": "essential", "scope": "essential"}
        if isinstance(payload, str):
            payload = {"id": payload, "scope": payload}

        scope = payload.get("scope", "essential")
        categories = payload.get("categories")
        favorites_only = bool(payload.get("favorites_only", False))
        fav_providers = self.preset_manager.get_favorite_databases() if self.preset_manager else []
        selected_db_ids = None

        if favorites_only and not fav_providers:
            self.phase_lbl.setText("No favorite databases pinned")
            self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #d99b38; background: transparent;")
            self.detail_lbl.setText("Pin at least one reviewer database in AutoEQ before syncing favorites")
            return

        if payload.get("rig_filter"):
            selected_db_ids = self._resolve_rig_db_ids(payload["rig_filter"])
            if not selected_db_ids:
                self.phase_lbl.setText("No matching coupler databases found")
                self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #d99b38; background: transparent;")
                return

        self.start_btn.setText("Stop sync")
        self.start_btn.setObjectName("stopActionBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.phase_lbl.setText("Initializing sync...")
        self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed; background: transparent;")
        self.eta_lbl.setText("")
        self.detail_lbl.setText("Connecting to reviewer catalogs...")
        self.scope_list.setEnabled(False)
        self.compact_btn.setEnabled(False)

        new_worker = OfflineSyncWorker(
            self.client,
            scope=scope,
            max_workers=12,
            parent=self.main_win,
            selected_db_ids=selected_db_ids,
            categories=categories,
            favorites_only=favorites_only,
            favorite_providers=fav_providers if favorites_only else None,
        )
        if self.main_win:
            self.main_win.offline_sync_worker = new_worker
            new_worker.sigProgress.connect(self.main_win._on_bg_sync_progress)
            new_worker.sigFinished.connect(self.main_win._on_bg_sync_finished)

        self._attach_worker(new_worker)
        new_worker.start()

    def _on_phase_changed(self, phase: str) -> None:
        self.phase_lbl.setText((phase or "").rstrip("."))

    def _on_progress(self, current: int, total: int, status: str, item_name: str) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.detail_lbl.setText(f"{status} · {item_name}")

    def _on_progress_detail(self, current: int, total: int, status: str, item_name: str, eta_seconds: float) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        eta_txt = OfflineSyncWorker.format_eta(eta_seconds) if eta_seconds and eta_seconds > 0 else ""
        self.eta_lbl.setText(f"ETA {eta_txt}" if eta_txt else "")
        self.detail_lbl.setText(f"{current:,}/{total:,} · {item_name}")

    def _on_stats_updated(self, meas: int, targets: int, size_mb: float) -> None:
        if self._catalog_model_total > 0:
            self.meas_stat_val.setText(f"{meas:,} / {self._catalog_model_total:,}")
        else:
            self.meas_stat_val.setText(f"{meas:,}")
        if self._catalog_target_total > 0:
            self.target_stat_val.setText(f"{targets:,} / {self._catalog_target_total:,}")
        else:
            self.target_stat_val.setText(f"{targets:,}")
        self.size_stat_val.setText(f"{float(size_mb):.1f} MB")

    def _on_finished(self, new_saved: int, total_meas: int, total_targets: int) -> None:
        was_cancelled = bool(getattr(self._attached_worker, "_abort", False))
        self._detach_worker()
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start sync")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.scope_list.setEnabled(True)
        self.compact_btn.setEnabled(True)
        self.eta_lbl.setText("")
        if not was_cancelled:
            self.progress_bar.setValue(100)
            self.phase_lbl.setText("Sync complete")
            self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #4cc38a; background: transparent;")
            if new_saved > 0:
                self.detail_lbl.setText(f"Cached {new_saved:,} new curves in local SQLite storage")
            else:
                self.detail_lbl.setText("All curves in this scope are already cached offline")
        else:
            self.phase_lbl.setText("Sync stopped")
            self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
            self.detail_lbl.setText("Sync cancelled - existing cached curves were preserved")
        self._refresh_stats()
        if self.main_win and hasattr(self.main_win, "_update_offline_db_badge"):
            self.main_win._update_offline_db_badge()

    def _on_error(self, err: str) -> None:
        self._detach_worker()
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start sync")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setStyle(self.start_btn.style())
        self.scope_list.setEnabled(True)
        self.compact_btn.setEnabled(True)
        self.eta_lbl.setText("")
        self.phase_lbl.setText(f"Sync error: {err}")
        self.phase_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #e5484d; background: transparent;")


class ProfileCurveAssignmentDialog(QtWidgets.QDialog):
    """
    Dialog to assign or edit consistent measurement and target reference curves for any EQ profile.
    Enforces strict reviewer database / coupler rig consistency so that both the IEM/headphone
    measurement and the target curve originate from the exact same provider.
    """
    def __init__(
        self,
        client: SquigClient,
        preset_manager: PresetManager,
        preset_name: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.client = client
        self.preset_manager = preset_manager
        self.preset_name = preset_name
        self.existing_meta = self.preset_manager.get_preset_metadata(preset_name) or {}

        self._db_worker: Optional[DatabaseIndexWorker] = None
        self._retired_workers: List[QtCore.QThread] = []
        self._db_req_id: int = 0
        self._current_models: List[SquigModel] = []
        self._current_targets: List[SquigTarget] = []
        self._pending_model_file: Optional[str] = (
            self.existing_meta.get("model_file") or self.existing_meta.get("model_name")
        )
        self._pending_target_file: Optional[str] = (
            self.existing_meta.get("target_file") or self.existing_meta.get("target_name")
        )
        self._pending_target2_file: Optional[str] = (
            self.existing_meta.get("target2_file") or self.existing_meta.get("target2_name")
        )
        self._selected_chan: str = self.existing_meta.get("channel_mode", "avg")

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle(f"Reference curves - {preset_name}")
        self.resize(620, 540)
        self.setMinimumSize(560, 480)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + """
            QDialog {
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
            }
            QWidget#curveAssignContent {
                background-color: #0b0b0c;
            }
            QFrame#curveBannerCard {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QFrame#curveBannerCard QLabel {
                background: transparent;
                border: none;
            }
            """
        )

        self._init_ui()
        self._populate_providers()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _retire_db_worker(self) -> None:
        worker = self._db_worker
        self._db_worker = None
        if worker is None or sip.isdeleted(worker):
            return
        for sig_name in ("sigIndexLoaded", "sigIndexFailed"):
            sig = getattr(worker, sig_name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
        worker.requestInterruption()
        if worker.isRunning():
            worker.setParent(None)
            self._retired_workers.append(worker)
            def _cleanup(w=worker):
                try:
                    self._retired_workers.remove(w)
                except ValueError:
                    pass
                try:
                    _SHUTDOWN_THREADS.remove(w)
                except ValueError:
                    pass
                if not sip.isdeleted(w):
                    w.deleteLater()
            worker.finished.connect(_cleanup)
            if not worker.wait(150):
                _SHUTDOWN_THREADS.append(worker)
        else:
            worker.deleteLater()

    def done(self, r: int) -> None:
        self._db_req_id += 1
        self._retire_db_worker()
        super().done(r)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._db_req_id += 1
        self._retire_db_worker()
        super().closeEvent(event)

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, f"Reference curves - {self.preset_name}")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("curveAssignContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Top banner (scoped #curveBannerCard)
        banner_box = QtWidgets.QFrame()
        banner_box.setObjectName("curveBannerCard")
        banner_layout = QtWidgets.QVBoxLayout(banner_box)
        banner_layout.setContentsMargins(12, 10, 12, 10)
        banner_layout.setSpacing(4)

        p_lbl = QtWidgets.QLabel(f"Profile: {self.preset_name}")
        p_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #ededed;")
        banner_layout.addWidget(p_lbl)

        p_desc = QtWidgets.QLabel(
            "Attach a headphone measurement and reference target to display the raw trace, target curve, and live resultant response."
        )
        p_desc.setStyleSheet("font-size: 11px; color: #71717a;")
        p_desc.setWordWrap(True)
        banner_layout.addWidget(p_desc)
        layout.addWidget(banner_box)

        # Provider / Reviewer Database selection
        sec_prov = QtWidgets.QVBoxLayout()
        sec_prov.setSpacing(4)
        prov_head = QtWidgets.QHBoxLayout()
        prov_lbl = QtWidgets.QLabel("Reviewer database")
        prov_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        prov_head.addWidget(prov_lbl)
        prov_head.addStretch()

        self.btn_global_search = QtWidgets.QPushButton("Global search")
        self.btn_global_search.setFixedHeight(28)
        self.btn_global_search.setToolTip("Search indexed reviewer measurement catalogs")
        self.btn_global_search.clicked.connect(self._open_global_search)
        prov_head.addWidget(self.btn_global_search)
        sec_prov.addLayout(prov_head)

        self.provider_combo = SearchableDatabaseInput(self)
        self.provider_combo.setFixedHeight(32)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        sec_prov.addWidget(self.provider_combo)

        rig_box = QtWidgets.QHBoxLayout()
        rig_box.setSpacing(8)
        self.rig_badge = QtWidgets.QLabel("Rig: IEC 60318-4 (711) · Coupler isolated")
        self.rig_badge.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; background: transparent;")
        rig_box.addWidget(self.rig_badge)
        rig_box.addStretch()
        sec_prov.addLayout(rig_box)

        consistency_note = QtWidgets.QLabel(
            "Measurement and target are locked to this reviewer coupler to keep alignment consistent."
        )
        consistency_note.setStyleSheet("font-size: 11px; color: #71717a; background: transparent;")
        sec_prov.addWidget(consistency_note)
        layout.addLayout(sec_prov)

        # Measurement / Model selection
        sec_meas = QtWidgets.QVBoxLayout()
        sec_meas.setSpacing(4)
        meas_lbl = QtWidgets.QLabel("Headphone or IEM measurement")
        meas_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        sec_meas.addWidget(meas_lbl)

        meas_row = QtWidgets.QHBoxLayout()
        meas_row.setSpacing(8)

        self.model_input = SearchableModelInput(self)
        self.model_input.setFixedHeight(32)
        self.model_input.sigModelSelected.connect(self._on_model_selected)
        meas_row.addWidget(self.model_input, stretch=3)

        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.setFixedHeight(32)
        self.chan_combo.addItem("Avg", "avg")
        self.chan_combo.addItem("L", "L")
        self.chan_combo.addItem("R", "R")
        self.chan_combo.setToolTip("Channel: Average (L+R), Left (L), or Right (R)")
        self.chan_combo.setFixedWidth(80)
        meas_row.addWidget(self.chan_combo)
        sec_meas.addLayout(meas_row)
        layout.addLayout(sec_meas)

        # Primary Target Curve selection
        sec_targ = QtWidgets.QVBoxLayout()
        sec_targ.setSpacing(4)
        targ_lbl = QtWidgets.QLabel("Primary target curve")
        targ_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        sec_targ.addWidget(targ_lbl)

        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.setFixedHeight(32)
        sec_targ.addWidget(self.target_combo)
        layout.addLayout(sec_targ)

        # Secondary Target Curve selection (optional)
        sec_targ2 = QtWidgets.QVBoxLayout()
        sec_targ2.setSpacing(4)
        targ2_lbl = QtWidgets.QLabel("Secondary target curve (optional)")
        targ2_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        sec_targ2.addWidget(targ2_lbl)

        self.target2_combo = QtWidgets.QComboBox()
        self.target2_combo.setFixedHeight(32)
        self.target2_combo.addItem("None", None)
        sec_targ2.addWidget(self.target2_combo)
        layout.addLayout(sec_targ2)

        # Status Label
        self.status_label = QtWidgets.QLabel("Select a provider, model, and target to attach")
        self.status_label.setStyleSheet("font-size: 11px; color: #71717a; background: transparent;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Action Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setFixedHeight(32)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        has_curves = bool(
            self.existing_meta and (
                self.existing_meta.get("model_file") or
                self.existing_meta.get("target_file") or
                self.existing_meta.get("is_autoeq")
            )
        )
        self.remove_btn = QtWidgets.QPushButton("Remove curves")
        self.remove_btn.setObjectName("dangerBtn")
        self.remove_btn.setFixedHeight(32)
        self.remove_btn.setEnabled(has_curves)
        self.remove_btn.setToolTip("Remove reference curves from this profile")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_btn)

        btn_row.addStretch()

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.setObjectName("primaryBtn")
        self.apply_btn.setFixedHeight(32)
        self.apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self.apply_btn)

        layout.addLayout(btn_row)

        main_layout.addWidget(content)

    def _populate_providers(self) -> None:
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()

        favs = self.preset_manager.get_favorite_databases() if self.preset_manager else []
        db_list = list(DEFAULT_DATABASES)
        seen_urls = {d.get("base_url", "").rstrip("/") for d in db_list}

        cat = self.client.get_global_catalog()
        for c in cat:
            b_url = c.get("base_url", "").rstrip("/")
            if b_url and b_url not in seen_urls:
                seen_urls.add(b_url)
                cat_type = c.get("db_type", "IEMs")
                p_name = c.get("provider", "Reviewer")
                disp = f"{p_name} ({cat_type})" if f"({cat_type.lower()}" not in p_name.lower() else p_name
                db_list.append({
                    "id": _resolve_db_id(self.client, c.get("base_url", ""), f"{c.get('username') or p_name.lower().replace(' ', '_')}_{cat_type.lower()}"),
                    "name": disp,
                    "base_url": c.get("base_url", ""),
                    "rig": c.get("rig", "IEC 711"),
                    "category": cat_type,
                })

        sorted_dbs = sorted(
            db_list,
            key=lambda d: (0 if d.get("id") in favs else 1, d.get("name", "").lower()),
        )

        existing_db_id = self.existing_meta.get("database_id", "")
        existing_url = self.existing_meta.get("base_url", "").rstrip("/")
        selected_idx = 0

        for i, db in enumerate(sorted_dbs):
            name = db.get("name", "")
            self.provider_combo.addItem(name, db)
            if existing_db_id and db.get("id") == existing_db_id:
                selected_idx = i
            elif existing_url and db.get("base_url", "").rstrip("/") == existing_url:
                selected_idx = i

        self.provider_combo.setCurrentIndex(selected_idx)
        self.provider_combo.blockSignals(False)

        # Restore channel
        for i in range(self.chan_combo.count()):
            if self.chan_combo.itemData(i) == self._selected_chan:
                self.chan_combo.setCurrentIndex(i)
                break

        # Trigger initial provider load
        self._on_provider_changed(selected_idx)

    def _on_provider_changed(self, idx: int = -1) -> None:
        db_data = self.provider_combo.currentData()
        if not db_data:
            return
        base_url = db_data.get("base_url", "")
        rig = db_data.get("rig", "IEC 60318-4 (711)")
        self.rig_badge.setText(f"Rig: {rig} [Coupler Isolated]")
        self.rig_badge.setToolTip(
            f"Strict Rig Consistency: Measurement and target are strictly locked to {db_data.get('name', 'this provider')} ({rig})."
        )

        self.model_input.setEnabled(False)
        self.target_combo.setEnabled(False)
        self.target2_combo.setEnabled(False)
        if not self._pending_model_file:
            self.model_input.set_selected_model(None)
        self.status_label.setText(f"Loading {db_data.get('name', '')} catalog...")
        self.status_label.setStyleSheet("color: #71717a; font-size: 11px;")

        self._db_req_id += 1
        req_id = self._db_req_id
        self._retire_db_worker()

        worker = DatabaseIndexWorker(self.client, base_url, parent=self)
        worker.sigIndexLoaded.connect(
            lambda m, t, r, rid=req_id: self._on_db_index_loaded(m, t, r, rid)
        )
        worker.sigIndexFailed.connect(
            lambda err, rid=req_id: self._on_db_index_failed(err, rid)
        )
        self._db_worker = worker
        worker.start()

    def _on_db_index_loaded(
        self,
        models: List[SquigModel],
        targets: List[SquigTarget],
        rig: str,
        req_id: Optional[int] = None,
    ) -> None:
        if req_id is not None and req_id != self._db_req_id:
            return
        self.model_input.setEnabled(True)
        self.target_combo.setEnabled(True)
        self.target2_combo.setEnabled(True)
        self._current_models = models
        self._current_targets = targets
        self.model_input.set_models(models)

        if rig:
            self.rig_badge.setText(f"Rig: {rig} [Coupler Isolated]")

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        target_sel_idx = 0
        for idx, t in enumerate(targets):
            label = f"[{t.category}] {t.name}" if t.category else t.name
            self.target_combo.addItem(label, t)
            if self._pending_target_file:
                tf_clean = self._pending_target_file.lower().replace(" target", "").strip()
                t_name_clean = t.name.lower().replace(" target", "").strip()
                t_file_clean = t.file_name.lower().replace(" target", "").strip()
                if tf_clean == t_name_clean or tf_clean == t_file_clean:
                    target_sel_idx = idx

        if targets:
            self.target_combo.setCurrentIndex(target_sel_idx)
        self._pending_target_file = None
        self.target_combo.blockSignals(False)

        self.target2_combo.blockSignals(True)
        self.target2_combo.clear()
        self.target2_combo.addItem("(None)", None)
        target2_sel_idx = 0
        for idx, t in enumerate(targets):
            label = f"[{t.category}] {t.name}" if t.category else t.name
            self.target2_combo.addItem(label, t)
            if self._pending_target2_file:
                tf_clean = self._pending_target2_file.lower().replace(" target", "").strip()
                t_name_clean = t.name.lower().replace(" target", "").strip()
                t_file_clean = t.file_name.lower().replace(" target", "").strip()
                if tf_clean == t_name_clean or tf_clean == t_file_clean:
                    target2_sel_idx = idx + 1

        self.target2_combo.setCurrentIndex(target2_sel_idx)
        self._pending_target2_file = None
        self.target2_combo.blockSignals(False)

        if self._pending_model_file:
            mf_clean = self._pending_model_file.lower().strip()
            matched = None
            for m in models:
                if (m.file_name.lower().strip() == mf_clean or
                    m.name.lower().strip() == mf_clean or
                    m.display_name.lower().strip() == mf_clean or
                    mf_clean in m.display_name.lower()):
                    matched = m
                    break
            if matched:
                self.model_input.set_selected_model(matched)
            self._pending_model_file = None

        db_name = self.provider_combo.currentData().get("name", "provider") if self.provider_combo.currentData() else "provider"
        self.status_label.setText(f"Loaded {len(models)} models and {len(targets)} targets from {db_name}")
        self.status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")

    def _on_db_index_failed(self, err: str, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._db_req_id:
            return
        db_data = self.provider_combo.currentData()
        base_url = db_data.get("base_url", "").rstrip("/") if db_data else ""
        cat = self.client.get_global_catalog()
        fallback_models = []
        fallback_targets = []
        for c in cat:
            if c.get("base_url", "").rstrip("/") == base_url:
                for p in c.get("phones", []):
                    fn = p.get("file", "")
                    if isinstance(fn, list):
                        fn = fn[0] if fn else ""
                    fallback_models.append(SquigModel(
                        brand=p.get("brand", "Unknown"),
                        name=p.get("name", "Model"),
                        full_name=p.get("full_name", p.get("name", "Model")),
                        file_name=fn,
                        display_name=p.get("full_name", p.get("name", "Model")),
                    ))
                for t in c.get("targets", []):
                    fallback_targets.append(SquigTarget(
                        name=t,
                        file_name=f"{t} Target" if not t.lower().endswith("target") else t,
                        category="Reference",
                    ))
                break

        if fallback_models or fallback_targets:
            self._on_db_index_loaded(fallback_models, fallback_targets, db_data.get("rig", "IEC 711"), req_id)
        else:
            self.model_input.setEnabled(True)
            self.target_combo.setEnabled(True)
            self.target2_combo.setEnabled(True)
            self.status_label.setText(f"Notice: Offline or error loading index: {err}")
            self.status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _on_model_selected(self, model: SquigModel) -> None:
        if model:
            self.status_label.setText(f"Selected: {model.display_name}")
            self.status_label.setStyleSheet("color: #ffffff; font-size: 11px;")

    def _open_global_search(self) -> None:
        initial_q = self.model_input.text().strip()
        dlg = GlobalSearchDialog(self.client, initial_query=initial_q, preset_manager=self.preset_manager, parent=self)
        accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        res = dlg.selected_result if accepted else None
        dlg.deleteLater()
        if res:
            res_url = res.base_url.rstrip("/")
            found_idx = -1
            for i in range(self.provider_combo.count()):
                d = self.provider_combo.itemData(i)
                if isinstance(d, dict) and d.get("base_url", "").rstrip("/") == res_url:
                    found_idx = i
                    break
            if found_idx == -1:
                new_info = {
                    "id": _resolve_db_id(self.client, res.base_url, res.provider_name.lower().replace(" ", "_")),
                    "name": res.provider_name,
                    "base_url": res.base_url,
                    "rig": res.rig,
                    "category": res.db_type,
                }
                self.provider_combo.addItem(res.provider_name, new_info)
                found_idx = self.provider_combo.count() - 1

            self._pending_model_file = res.file_name
            if res.targets:
                self._pending_target_file = res.targets[0]
            if self.provider_combo.currentIndex() == found_idx:
                self._on_provider_changed(found_idx)
            else:
                self.provider_combo.setCurrentIndex(found_idx)

    def _on_apply(self) -> None:
        model = self.model_input.currentData()
        if not model and self.model_input.text().strip():
            q = self.model_input.text().strip().lower()
            for m in self._current_models:
                if m.display_name.lower() == q or m.name.lower() == q or m.file_name.lower() == q:
                    model = m
                    break
            if not model:
                for m in self._current_models:
                    if q in m.display_name.lower():
                        model = m
                        break

        target = self.target_combo.currentData()
        target2 = self.target2_combo.currentData()
        provider_data = self.provider_combo.currentData()
        if not provider_data:
            QtWidgets.QMessageBox.warning(self, "Missing Provider", "Please select a provider database.")
            return

        if not model and not target and not target2:
            QtWidgets.QMessageBox.warning(self, "No Selection", "Please select a measurement model or target curve to attach.")
            return

        base_url = provider_data.get("base_url", "")
        db_id = provider_data.get("id", "") or _resolve_db_id(self.client, base_url)
        db_name = provider_data.get("name", "")
        rig = provider_data.get("rig", "")
        chan = self.chan_combo.currentData() or "avg"

        meta = dict(self.existing_meta) if isinstance(self.existing_meta, dict) else {}
        meta.update({
            "is_autoeq": bool(meta.get("is_autoeq", False)),
            "database_id": db_id,
            "database_name": db_name,
            "base_url": base_url,
            "model_name": model.name if model else "",
            "model_file": model.file_name if model else "",
            "target_name": target.name if target else "",
            "target_file": (target.file_name or target.name) if target else "",
            "target2_name": target2.name if target2 else "",
            "target2_file": (target2.file_name or target2.name) if target2 else "",
            "channel_mode": chan,
            "rig": rig,
        })

        self.preset_manager.set_preset_metadata(self.preset_name, meta)
        self.accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Down:
            if not self.model_input.popup.isVisible() and self.focusWidget() not in (
                self.provider_combo, self.chan_combo, self.target_combo, getattr(self, "target2_combo", None)
            ):
                self.model_input.setFocus()
                self.model_input._show_suggestions(force_all=False)
                return
        super().keyPressEvent(event)

    def _on_remove(self) -> None:
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Curves",
            f"Are you sure you want to remove target and measurement reference curves from profile '{self.preset_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self.preset_manager.set_preset_metadata(self.preset_name, None)
            self.accept()


class ImportPreviewDialog(QtWidgets.QDialog):
    """
    Studio import preview dialog for Equalizer APO, GraphicEQ, Peace, single JSON,
    and multi-preset JSON library imports (from file or clipboard).
    """
    _FORMAT_LABELS = {
        "apo": "Equalizer APO",
        "graphiceq": "GraphicEQ (fitted)",
        "peace": "Peace config",
        "json": "Clean APO JSON",
        "unknown": "Unknown format",
    }

    def __init__(
        self,
        name: str,
        preamp_db: float,
        bands: List[EQBand],
        info_dict: Dict[str, Any],
        existing_names: Sequence[str],
        source_label: str = "File",
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Import preview")
        self.resize(640, 520)
        self.setMinimumSize(560, 440)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + """
            QDialog {
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
            }
            QWidget#importPreviewContent {
                background-color: #0b0b0c;
            }
            QFrame#importMetaCard {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }
            QFrame#importMetaCard QLabel {
                background: transparent;
                border: none;
            }
            QFrame#importWarnCard {
                background-color: #121214;
                border: 1px solid #d99b38;
                border-radius: 4px;
            }
            QFrame#importWarnCard QLabel {
                background: transparent;
                border: none;
            }
            QTableWidget {
                background-color: #000000;
                alternate-background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                gridline-color: #27272a;
                color: #ededed;
                font-size: 11px;
            }
            """
        )

        self._existing_names = set(existing_names)
        self._info_dict = info_dict or {}
        self._multi_presets: Optional[Dict[str, Dict[str, Any]]] = self._info_dict.get("multi_presets")
        self._is_multi = bool(self._multi_presets and len(self._multi_presets) > 1)
        self.result_mode: str = "multi" if self._is_multi else "single"
        self.chosen_name: str = name or "Imported Profile"
        self.conflict_action: str = "rename"
        self.selected_multi_presets: Dict[str, Dict[str, Any]] = {}

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, f"Import preview - {source_label}")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("importPreviewContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        # Metadata summary card
        meta_card = QtWidgets.QFrame()
        meta_card.setObjectName("importMetaCard")
        meta_layout = QtWidgets.QHBoxLayout(meta_card)
        meta_layout.setContentsMargins(12, 10, 12, 10)
        meta_layout.setSpacing(16)

        fmt_key = str(self._info_dict.get("format", "unknown")).lower()
        fmt_label = "JSON library" if self._is_multi else self._FORMAT_LABELS.get(fmt_key, fmt_key)

        def _meta_item(caption: str, value: str) -> QtWidgets.QVBoxLayout:
            col = QtWidgets.QVBoxLayout()
            col.setSpacing(2)
            c_lbl = QtWidgets.QLabel(caption)
            c_lbl.setStyleSheet("font-size: 11px; color: #71717a;")
            v_lbl = QtWidgets.QLabel(value)
            v_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; color: #ededed;")
            col.addWidget(c_lbl)
            col.addWidget(v_lbl)
            return col

        meta_layout.addLayout(_meta_item("Format", fmt_label))
        if self._is_multi and self._multi_presets:
            meta_layout.addLayout(_meta_item("Profiles", f"{len(self._multi_presets)}"))
        else:
            meta_layout.addLayout(_meta_item("Preamp", f"{preamp_db:+.1f} dB"))
            meta_layout.addLayout(_meta_item("Bands", f"{len(bands)}"))
            ch_set = {getattr(b, "channel", "ALL") for b in bands if getattr(b, "channel", "ALL") in ("L", "R")}
            if ch_set:
                meta_layout.addLayout(_meta_item("Channels", "/".join(sorted(ch_set))))
        meta_layout.addStretch()
        layout.addWidget(meta_card)

        # Warnings card (hidden when empty)
        warnings = [str(w).rstrip(".") for w in (self._info_dict.get("warnings") or []) if w]
        if warnings:
            warn_card = QtWidgets.QFrame()
            warn_card.setObjectName("importWarnCard")
            w_layout = QtWidgets.QVBoxLayout(warn_card)
            w_layout.setContentsMargins(12, 8, 12, 8)
            w_layout.setSpacing(4)
            w_lbl = QtWidgets.QLabel(" · ".join(warnings))
            w_lbl.setStyleSheet("font-size: 11px; color: #d99b38;")
            w_lbl.setWordWrap(True)
            w_layout.addWidget(w_lbl)
            layout.addWidget(warn_card)

        if self._is_multi and self._multi_presets:
            # Multi-preset library selection table
            sub_lbl = QtWidgets.QLabel("Select profiles to import")
            sub_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
            layout.addWidget(sub_lbl)

            self.multi_table = QtWidgets.QTableWidget()
            self.multi_table.setColumnCount(4)
            self.multi_table.setHorizontalHeaderLabels(["Profile", "Preamp", "Bands", "Status"])
            hdr = self.multi_table.horizontalHeader()
            hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            self.multi_table.verticalHeader().setVisible(False)
            self.multi_table.verticalHeader().setDefaultSectionSize(28)
            self.multi_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.multi_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            self.multi_table.setAlternatingRowColors(True)

            mono_font = _make_mono_font(9)

            items_list = list(self._multi_presets.items())
            self.multi_table.setRowCount(len(items_list))
            for r, (p_name, p_data) in enumerate(items_list):
                it_name = QtWidgets.QTableWidgetItem(p_name)
                it_name.setFlags(it_name.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                it_name.setCheckState(QtCore.Qt.CheckState.Checked)
                it_name.setData(QtCore.Qt.ItemDataRole.UserRole, p_name)
                self.multi_table.setItem(r, 0, it_name)

                p_val = float(p_data.get("preamp", p_data.get("preamp_db", 0.0)))
                it_pre = QtWidgets.QTableWidgetItem(f"{p_val:+.1f} dB")
                it_pre.setFont(mono_font)
                it_pre.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.multi_table.setItem(r, 1, it_pre)

                b_cnt = len(p_data.get("bands", []))
                it_cnt = QtWidgets.QTableWidgetItem(str(b_cnt))
                it_cnt.setFont(mono_font)
                it_cnt.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.multi_table.setItem(r, 2, it_cnt)

                exists = p_name in self._existing_names
                it_st = QtWidgets.QTableWidgetItem("Exists" if exists else "New")
                it_st.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                it_st.setForeground(QtGui.QBrush(QtGui.QColor("#d99b38" if exists else "#4cc38a")))
                self.multi_table.setItem(r, 3, it_st)

            make_smooth_scroll(self.multi_table)
            layout.addWidget(self.multi_table, stretch=1)

            conflict_row = QtWidgets.QHBoxLayout()
            conflict_row.setSpacing(8)
            c_lbl = QtWidgets.QLabel("Name conflict")
            c_lbl.setStyleSheet("font-size: 12px; color: #a1a1aa; background: transparent;")
            conflict_row.addWidget(c_lbl)

            self.conflict_combo = QtWidgets.QComboBox()
            self.conflict_combo.setFixedHeight(28)
            self.conflict_combo.addItem("Rename duplicates", "rename")
            self.conflict_combo.addItem("Overwrite existing", "overwrite")
            self.conflict_combo.addItem("Skip existing", "skip")
            conflict_row.addWidget(self.conflict_combo)
            conflict_row.addStretch()
            layout.addLayout(conflict_row)
        else:
            # Single preset name + conflict resolution
            name_row = QtWidgets.QHBoxLayout()
            name_row.setSpacing(8)
            n_lbl = QtWidgets.QLabel("Profile name")
            n_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
            name_row.addWidget(n_lbl)

            self.name_edit = QtWidgets.QLineEdit(self.chosen_name)
            self.name_edit.setFixedHeight(32)
            self.name_edit.textChanged.connect(self._update_conflict_hint)
            name_row.addWidget(self.name_edit, stretch=1)

            self.conflict_combo = QtWidgets.QComboBox()
            self.conflict_combo.setFixedHeight(32)
            self.conflict_combo.addItem("Rename if exists", "rename")
            self.conflict_combo.addItem("Overwrite existing", "overwrite")
            name_row.addWidget(self.conflict_combo)
            layout.addLayout(name_row)

            self.conflict_hint = QtWidgets.QLabel("")
            self.conflict_hint.setStyleSheet("font-size: 11px; color: #d99b38; background: transparent;")
            self.conflict_hint.setVisible(False)
            layout.addWidget(self.conflict_hint)

            # Read-only band preview table
            self.band_table = QtWidgets.QTableWidget()
            self.band_table.setColumnCount(6)
            self.band_table.setHorizontalHeaderLabels(["Band", "Ch", "Type", "Frequency", "Gain", "Q"])
            hdr = self.band_table.horizontalHeader()
            hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
            hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            self.band_table.verticalHeader().setVisible(False)
            self.band_table.verticalHeader().setDefaultSectionSize(26)
            self.band_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.band_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
            self.band_table.setAlternatingRowColors(True)

            mono_font = _make_mono_font(9)

            self.band_table.setRowCount(len(bands))
            for r, b in enumerate(bands):
                it_id = QtWidgets.QTableWidgetItem(str(b.band_id))
                it_id.setFont(mono_font)
                it_id.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.band_table.setItem(r, 0, it_id)

                ch = getattr(b, "channel", "ALL")
                it_ch = QtWidgets.QTableWidgetItem(ch if ch in ("L", "R") else "All")
                it_ch.setFont(mono_font)
                it_ch.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                it_ch.setForeground(QtGui.QBrush(QtGui.QColor("#ededed" if ch in ("L", "R") else "#71717a")))
                self.band_table.setItem(r, 1, it_ch)

                it_type = QtWidgets.QTableWidgetItem(str(b.filter_type))
                if not b.enabled:
                    it_type.setForeground(QtGui.QBrush(QtGui.QColor("#71717a")))
                self.band_table.setItem(r, 2, it_type)

                it_f = QtWidgets.QTableWidgetItem(f"{b.freq:.1f} Hz")
                it_f.setFont(mono_font)
                it_f.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.band_table.setItem(r, 3, it_f)

                it_g = QtWidgets.QTableWidgetItem(f"{b.gain:+.1f} dB")
                it_g.setFont(mono_font)
                it_g.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.band_table.setItem(r, 4, it_g)

                it_q = QtWidgets.QTableWidgetItem(f"{b.q:.2f}")
                it_q.setFont(mono_font)
                it_q.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.band_table.setItem(r, 5, it_q)

            make_smooth_scroll(self.band_table)
            layout.addWidget(self.band_table, stretch=1)
            self._update_conflict_hint()

        # Footer buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedHeight(32)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        import_btn = QtWidgets.QPushButton("Import")
        import_btn.setObjectName("primaryBtn")
        import_btn.setFixedHeight(32)
        import_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(import_btn)
        layout.addLayout(btn_row)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def _update_conflict_hint(self) -> None:
        if self._is_multi or not hasattr(self, "name_edit"):
            return
        candidate = self.name_edit.text().strip()
        exists = candidate in self._existing_names
        self.conflict_combo.setVisible(exists)
        if exists:
            self.conflict_hint.setText(f"A profile named '{candidate}' already exists")
            self.conflict_hint.setVisible(True)
        else:
            self.conflict_hint.setVisible(False)

    def _on_confirm(self) -> None:
        self.conflict_action = str(self.conflict_combo.currentData() or "rename")
        if self._is_multi and self._multi_presets:
            selected: Dict[str, Dict[str, Any]] = {}
            for r in range(self.multi_table.rowCount()):
                it = self.multi_table.item(r, 0)
                if it and it.checkState() == QtCore.Qt.CheckState.Checked:
                    key = it.data(QtCore.Qt.ItemDataRole.UserRole)
                    if key in self._multi_presets:
                        selected[key] = self._multi_presets[key]
            self.selected_multi_presets = selected
            if not selected:
                return
        else:
            self.chosen_name = self.name_edit.text().strip() or self.chosen_name
        self.accept()


class ShortcutsDialog(QtWidgets.QDialog):
    """
    Quiet studio keyboard shortcuts and EQ graph gesture cheatsheet dialog
    with crisp monospace keycaps.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Keyboard shortcuts")
        self.resize(720, 620)
        self.setMinimumSize(660, 580)
        self.setStyleSheet(
            BW_MINIMALIST_STYLESHEET
            + f"""
            QDialog {{
                background-color: #0b0b0c;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ededed;
            }}
            QWidget#shortcutsContent {{
                background-color: #0b0b0c;
            }}
            QFrame#shortcutCard {{
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
            }}
            QFrame#shortcutCard QLabel {{
                background: transparent;
                border: none;
            }}
            QFrame#shortcutCard QLabel#keycap {{
                font-family: {MONO_FONT_FAMILY};
                font-size: 11px;
                font-weight: 500;
                color: #ededed;
                background-color: #1c1c1f;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                padding: 2px 8px;
            }}
            """
        )

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Keyboard shortcuts & gestures")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        content.setObjectName("shortcutsContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        sections = [
            (
                "Playback & global",
                [
                    ("Ctrl+B", "Toggle universal EQ bypass"),
                    ("Ctrl+H", "Open hearing calibration test"),
                    ("Ctrl+Z", "Undo EQ parameter change"),
                    ("Ctrl+Y", "Redo EQ parameter change"),
                    ("?", "Show keyboard shortcuts"),
                ],
            ),
            (
                "Profiles & navigation",
                [
                    ("Ctrl+1", "Switch to Profiles tab"),
                    ("Ctrl+2", "Switch to AutoEQ tab"),
                    ("Ctrl+N", "Create new flat profile"),
                    ("Ctrl+E", "Export active profile"),
                    ("F2", "Rename selected profile"),
                    ("Delete", "Delete selected profile"),
                ],
            ),
            (
                "EQ graph gestures",
                [
                    ("Drag node", "Adjust frequency and gain"),
                    ("Shift+Drag", "Fine adjustment (5x precision)"),
                    ("Ctrl+Drag", "Lock frequency (gain only)"),
                    ("Alt+Drag", "Lock gain (frequency only)"),
                    ("Scroll", "Adjust Q factor over node"),
                    ("Middle-click", "Toggle band bypass"),
                    ("Double-click node", "Reset band gain to 0 dB"),
                    ("Double-click canvas", "Add new peak band"),
                    ("Double-click Y-axis", "Toggle ±12 / ±20 dB scale"),
                ],
            ),
            (
                "Hearing test",
                [
                    ("Space", "Play or stop test tone"),
                    ("Up / Down", "Adjust tone level"),
                    ("Left / Right", "Previous / next frequency"),
                    ("Enter", "Record current threshold"),
                ],
            ),
        ]

        for idx, (sec_title, rows) in enumerate(sections):
            card = QtWidgets.QFrame()
            card.setObjectName("shortcutCard")
            c_layout = QtWidgets.QVBoxLayout(card)
            c_layout.setContentsMargins(12, 12, 12, 12)
            c_layout.setSpacing(6)

            h_lbl = QtWidgets.QLabel(sec_title)
            h_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #ededed;")
            c_layout.addWidget(h_lbl)

            for key_str, desc_str in rows:
                r_box = QtWidgets.QHBoxLayout()
                r_box.setSpacing(8)
                d_lbl = QtWidgets.QLabel(desc_str)
                d_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
                r_box.addWidget(d_lbl, stretch=1)

                k_lbl = QtWidgets.QLabel(key_str)
                k_lbl.setObjectName("keycap")
                k_lbl.setMinimumHeight(22)
                k_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                r_box.addWidget(k_lbl)
                c_layout.addLayout(r_box)

            c_layout.addStretch()
            grid.addWidget(card, idx // 2, idx % 2)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.setObjectName("primaryBtn")
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()


class AutoResizingScrollArea(QtWidgets.QScrollArea):
    """ScrollArea that synchronizes its widget's width to the viewport width to prevent horizontal clipping."""
    def __init__(self, parent=None):
        super().__init__(parent)
        make_smooth_scroll(self)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.widget():
            self.widget().setFixedWidth(self.viewport().width())


class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window implementing Clean APO:
    - Left Sidebar: Profile / Preset Manager with inline renaming & animated highlight
    - Main Center: pyqtgraph interactive EQ plot and WASAPI spectrum overlay
    - Bottom Panel: Filter parameter table with CleanSpinBox and Expand/Compress tools
    - Top Header: Preamp slider, Audio device dropdown, Universal EQ toggle
    - Minimalist 1px grey divider lines dividing all sections cleanly
    """
    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Window)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("Clean APO")
        self.resize(1280, 820)
        self.setMinimumSize(960, 640)
        self.setStyleSheet(BW_MINIMALIST_STYLESHEET)

        self.resize_filter = WindowResizeFilter(self)
        self.installEventFilter(self.resize_filter)

        self._global_scroll_filter = AppGlobalScrollFilter(self)
        app_inst = QtWidgets.QApplication.instance()
        if app_inst:
            app_inst.installEventFilter(self._global_scroll_filter)

        # Core Engines
        self.preset_manager = PresetManager(self)
        self.spectrum_worker = AudioSpectrumWorker(parent=self)
        self.squig_client = SquigClient()

        self._updating_table = False
        self.is_eq_bypassed = False
        self._hearing_test_measurement_bypass = False
        self._spectrum_peak_hold = False
        self._spectrum_frozen = False
        self._spectrum_tilt = "pink"
        self._spectrum_speed = "medium"
        self._autoeq_bass_boost_db = 0.0
        self._autoeq_treble_tilt_db = 0.0

        # AutoEQ State Tracking
        self._autoeq_target_mode: int = 0  # 0 = Target Curve, 1 = IEM Tuning, 2 = Resultant Profile
        self._autoeq_meas_raw: Tuple[List[float], List[float]] = ([], [])
        self._autoeq_target_raw: Tuple[List[float], List[float]] = ([], [])
        self._disp_meas_freqs: Optional[np.ndarray] = None
        self._disp_meas_spls: Optional[np.ndarray] = None
        self.sample_rate = 48000.0
        self._autoeq_models: List[SquigModel] = []
        self._autoeq_targets: List[SquigTarget] = []
        self._active_db_info: Dict[str, str] = dict(DEFAULT_DATABASES[0])
        self._db_worker: Optional[DatabaseIndexWorker] = None
        self._curve_worker: Optional[CurveFetchWorker] = None
        self._target2_worker: Optional[CurveFetchWorker] = None
        self._autoeq_worker: Optional[AutoEQWorker] = None
        self._retired_workers: List[QtCore.QThread] = []
        self._db_req_id: int = 0
        self._curve_req_id: int = 0
        self._target2_req_id: int = 0
        self._autoeq_req_id: int = 0
        self._suppress_curve_fetch: bool = False
        self._suppress_preset_reload: bool = False
        self._is_dragging_preamp_slider: bool = False
        self._pending_edit_preset_name: Optional[str] = None
        self._pending_global_model: Optional[GlobalSearchResult] = None
        self._pending_restore_model_file: Optional[str] = None
        self._pending_restore_target_file: Optional[str] = None
        self._pending_restore_skip_fetch: bool = False

        self._db_switch_timer = QtCore.QTimer(self)
        self._db_switch_timer.setSingleShot(True)
        self._db_switch_timer.setInterval(100)
        self._db_switch_timer.timeout.connect(self._do_autoeq_db_changed)

        self._hw_change_debounce_timer = QtCore.QTimer(self)
        self._hw_change_debounce_timer.setSingleShot(True)
        self._hw_change_debounce_timer.setInterval(250)
        self._hw_change_debounce_timer.timeout.connect(self._on_audio_hardware_changed)

        self._preset_save_timer = QtCore.QTimer(self)
        self._preset_save_timer.setSingleShot(True)
        self._preset_save_timer.setInterval(150)
        self._preset_save_timer.timeout.connect(self._flush_preset_save)

        self._pending_edit_timer = QtCore.QTimer(self)
        self._pending_edit_timer.setSingleShot(True)
        self._pending_edit_timer.setInterval(50)
        self._pending_edit_timer.timeout.connect(self._trigger_pending_preset_edit)

        # Preamp cubic smoothing animation
        self._preamp_anim = QtCore.QVariantAnimation(self)
        self._preamp_anim.setDuration(320)
        self._preamp_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._preamp_anim.valueChanged.connect(self._on_preamp_anim_step)

        # QMediaDevices listener for instant native Windows hotplug / default change events
        self._media_devices: Optional[QMediaDevices] = None
        self._last_default_output_id: Optional[bytes] = None
        self._last_output_devices_snapshot: List[str] = []
        try:
            self._media_devices = QMediaDevices(self)
            self._media_devices.audioOutputsChanged.connect(self._on_audio_hardware_changed)
            def_out = self._media_devices.defaultAudioOutput()
            if def_out:
                self._last_default_output_id = bytes(def_out.id())
            self._last_output_devices_snapshot = [d.description() for d in self._media_devices.audioOutputs()]
        except Exception as e:
            print(f"[MainWindow] QMediaDevices init error: {e}")

        # Device hotplug monitor timer (polling backup checking every 1.5s)
        self._device_hotplug_timer = QtCore.QTimer(self)
        self._device_hotplug_timer.setInterval(1500)
        self._device_hotplug_timer.timeout.connect(self._check_device_hotplug)
        self._device_hotplug_timer.start()

        # Undo / Redo Parameter History Stack (DAW-grade parameter undo)
        self._undo_stack: deque[Tuple[float, List[Dict[str, Any]]]] = deque(maxlen=50)
        self._redo_stack: List[Tuple[float, List[Dict[str, Any]]]] = []
        self._last_state: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._is_undoing_redoing = False

        self._undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self._undo_shortcut.activated.connect(self._undo)

        self._redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self)
        self._redo_shortcut.activated.connect(self._redo)

        self._redo_shortcut_y = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self)
        self._redo_shortcut_y.activated.connect(self._redo)

        # Wheel scroll undo grouping (groups continuous wheel turns into a single undo step)
        self._is_wheel_scrolling = False
        self._has_active_scroll_snapshot = False
        self._wheel_scroll_timer = QtCore.QTimer(self)
        self._wheel_scroll_timer.setSingleShot(True)
        self._wheel_scroll_timer.timeout.connect(self._on_wheel_scroll_timeout)
        if app_inst:
            app_inst.installEventFilter(self)

        # Background Offline Sync Worker handle
        self.offline_sync_worker: Optional[OfflineSyncWorker] = None

        self._init_ui()
        self._populate_autoeq_databases()
        self._update_offline_db_badge()
        if self.autoeq_db_combo.currentData():
            self._active_db_info = dict(self.autoeq_db_combo.currentData())
        self._connect_signals()
        self._init_audio_device()
        self._load_initial_preset()
        self._refresh_apo_health_indicator()

        # Check Equalizer APO presence & active device registration after window appears
        self._launch_apo_timer = QtCore.QTimer(self)
        self._launch_apo_timer.setSingleShot(True)
        self._launch_apo_timer.setInterval(750)
        self._launch_apo_timer.timeout.connect(self._check_equalizer_apo_on_launch)
        self._launch_apo_timer.start()

        # Start background loopback audio worker
        self.spectrum_worker.start()

        # Trigger initial database loading in background (reads disk cache instantly, network fallback)
        self._load_database_index(self._active_db_info, force_refresh=False)

    def _retire_worker(
        self,
        worker: Optional[QtCore.QThread],
        use_cancel: bool = False,
        wait_ms: int = 0,
    ) -> None:
        if worker is None or sip.isdeleted(worker):
            return
        for sig_name in (
            "sigIndexLoaded",
            "sigIndexFailed",
            "sigCurvesLoaded",
            "sigTarget2Loaded",
            "sigCurveFailed",
            "sigOptimized",
            "sigFailed",
            "sigPhaseChanged",
            "sigProgress",
            "sigStatsUpdated",
            "sigFinished",
            "sigError",
        ):
            sig = getattr(worker, sig_name, None)
            if sig is not None:
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
        if use_cancel and hasattr(worker, "cancel"):
            try:
                worker.cancel()
            except Exception:
                worker.requestInterruption()
        else:
            worker.requestInterruption()
        if worker.isRunning():
            worker.setParent(None)
            self._retired_workers.append(worker)
            def _cleanup(w=worker):
                try:
                    self._retired_workers.remove(w)
                except ValueError:
                    pass
                try:
                    _SHUTDOWN_THREADS.remove(w)
                except ValueError:
                    pass
                if not sip.isdeleted(w):
                    w.deleteLater()
            worker.finished.connect(_cleanup)
            if wait_ms > 0:
                worker.wait(wait_ms)
        else:
            worker.deleteLater()

    def _flush_preset_save(self) -> None:
        if self._preset_save_timer.isActive():
            self._preset_save_timer.stop()
        if hasattr(self, "preamp_spin") and hasattr(self, "eq_plot"):
            self.preset_manager.update_active_preset_content(
                self.preamp_spin.value(), self.eq_plot.bands
            )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for timer_name in (
            "_device_hotplug_timer",
            "_db_switch_timer",
            "_wheel_scroll_timer",
            "_hw_change_debounce_timer",
            "_pending_edit_timer",
            "_launch_apo_timer",
        ):
            t = getattr(self, timer_name, None)
            if t is not None and t.isActive():
                t.stop()

        if self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()

        if self._preset_save_timer.isActive():
            self._flush_preset_save()

        if hasattr(self.preset_manager, "flush_apo_export"):
            self.preset_manager.flush_apo_export()
        elif getattr(self.preset_manager, "_debounce_timer", None) and self.preset_manager._debounce_timer.isActive():
            self.preset_manager._debounce_timer.stop()
            self.preset_manager._do_write_apo_file()

        app_inst = QtWidgets.QApplication.instance()
        if app_inst:
            app_inst.removeEventFilter(self)
            app_inst.removeEventFilter(self._global_scroll_filter)

        self._db_req_id += 1
        self._curve_req_id += 1
        self._target2_req_id += 1
        self._autoeq_req_id += 1

        self._retire_worker(self._db_worker, use_cancel=False, wait_ms=200)
        self._db_worker = None
        self._retire_worker(self._curve_worker, use_cancel=False, wait_ms=200)
        self._curve_worker = None
        self._retire_worker(self._target2_worker, use_cancel=False, wait_ms=200)
        self._target2_worker = None
        self._retire_worker(self._autoeq_worker, use_cancel=True, wait_ms=800)
        self._autoeq_worker = None
        self._retire_worker(self.offline_sync_worker, use_cancel=True, wait_ms=1000)
        self.offline_sync_worker = None

        self.spectrum_worker.stop()
        if self.spectrum_worker.isRunning():
            self.spectrum_worker.wait(500)
            if self.spectrum_worker.isRunning():
                self.spectrum_worker.setParent(None)
                _SHUTDOWN_THREADS.append(self.spectrum_worker)

        for w in list(self._retired_workers):
            if not sip.isdeleted(w) and w.isRunning():
                w.wait(200)
                if w.isRunning():
                    w.setParent(None)
                    if w not in _SHUTDOWN_THREADS:
                        _SHUTDOWN_THREADS.append(w)

        super().closeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        if hasattr(self, "filter_table"):
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            is_max = self.isMaximized()
            if hasattr(self, "btn_max"):
                self.btn_max.set_kind("restore" if is_max else "max")
            if hasattr(self, "central_widget"):
                if is_max:
                    self.central_widget.setStyleSheet("QWidget#centralWidget { background-color: #000000; border: none; border-radius: 0px; }")
                else:
                    self.central_widget.setStyleSheet("QWidget#centralWidget { background-color: #000000; border: 1px solid #27272a; border-radius: 8px; }")
        super().changeEvent(event)

    def nativeEvent(self, eventType: QtCore.QByteArray | bytes, message: int | Any) -> Tuple[bool, int]:
        # Handle WM_DEVICECHANGE (0x0219) for USB, DAC, and 3.5mm plug/unplug events (debounced)
        try:
            if eventType == b"windows_generic_MSG":
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == 0x0219 and msg.wParam in (0x0007, 0x8000, 0x8004):
                    self._hw_change_debounce_timer.start(250)
        except Exception:
            pass
        return False, 0

    def _toggle_maximize_restore(self) -> None:
        if self.isMaximized():
            self.showNormal()
            if hasattr(self, "btn_max"):
                self.btn_max.set_kind("max")
        else:
            self.showMaximized()
            if hasattr(self, "btn_max"):
                self.btn_max.set_kind("restore")

    # -------------------------------------------------------------------------
    # UI Assembly
    # -------------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("centralWidget")
        self.setCentralWidget(self.central_widget)
        
        # 1px border margin to ensure smooth rounded anti-aliased corners
        main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(0)

        # Row 1: Top Title & DAW Menu Bar
        main_layout.addWidget(self._create_top_title_bar(), stretch=0)

        # Row 2: Secondary Control Bar (EQ On, Spectrum, Preamp)
        main_layout.addWidget(self._create_control_bar(), stretch=0)

        # Main Splitter (Left Sidebar / Center EQ + Bottom Table)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setChildrenCollapsible(False)

        # 1. Left Sidebar: Presets (1px right divider line)
        sidebar_frame = self._create_sidebar()
        sidebar_frame.setMinimumWidth(310)
        sidebar_frame.setMaximumWidth(420)
        main_splitter.addWidget(sidebar_frame)

        # 2. Right Vertical Splitter (Plot + Bottom Table)
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(1)
        right_splitter.setChildrenCollapsible(False)

        # Plot Widget
        self.eq_plot = EQPlotWidget()
        right_splitter.addWidget(self.eq_plot)

        # Bottom Filter Table Panel (1px top divider line)
        table_frame = self._create_filter_table_panel()
        table_frame.setMinimumHeight(180)
        right_splitter.addWidget(table_frame)

        right_splitter.setSizes([480, 240])
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([340, 940])
        self.main_splitter = main_splitter

        main_layout.addWidget(main_splitter, stretch=1)

        # Keyboard shortcuts (Delete is handled strictly by AnimatedPresetListWidget.keyPressEvent when focused)
        self.rename_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F2), self)
        self.rename_shortcut.activated.connect(self._rename_preset_inline_shortcut)

        self.hearing_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+H"), self)
        self.hearing_shortcut.activated.connect(self._open_hearing_test)

        self.bypass_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        self.bypass_shortcut.activated.connect(self._toggle_universal_eq)

        self.new_profile_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+N"), self)
        self.new_profile_shortcut.activated.connect(self._add_preset_auto)

        self.export_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+E"), self)
        self.export_shortcut.activated.connect(lambda: self._export_eq_dialog())

        self.tab_profiles_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+1"), self)
        self.tab_profiles_shortcut.activated.connect(lambda: self._switch_sidebar_tab(0))

        self.tab_autoeq_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+2"), self)
        self.tab_autoeq_shortcut.activated.connect(lambda: self._switch_sidebar_tab(1))

        self.shortcuts_help_shortcut = QtGui.QShortcut(QtGui.QKeySequence("?"), self)
        self.shortcuts_help_shortcut.activated.connect(self._open_shortcuts_dialog_shortcut)

    # -------------------------------------------------------------------------
    # Top Title & Menu Bar (Row 1)
    # -------------------------------------------------------------------------
    def _create_top_title_bar(self) -> QtWidgets.QFrame:
        frame = HeaderFrame(self)
        frame.setObjectName("titleBarFrame")
        frame.setFixedHeight(36)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(4)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Brand Title
        self.brand_title_lbl = QtWidgets.QLabel("clean APO")
        self.brand_title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #ededed; padding-right: 12px; background: transparent;"
        )
        layout.addWidget(self.brand_title_lbl)

        # Menu 1: Settings
        self.btn_menu_settings = QtWidgets.QPushButton("Settings")
        self.btn_menu_settings.setObjectName("topMenuBtn")
        self.btn_menu_settings.setToolTip("Open Equalizer APO settings")
        self.btn_menu_settings.clicked.connect(self._open_settings)
        layout.addWidget(self.btn_menu_settings)

        # Menu 2: Output (Target Playback Device for Equalizer APO)
        self.btn_menu_output = QtWidgets.QPushButton("Output")
        self.btn_menu_output.setObjectName("topMenuBtn")
        self.btn_menu_output.setToolTip("Select Equalizer APO playback output device")
        self.btn_menu_output.clicked.connect(self._show_output_menu)
        self.btn_menu_input = self.btn_menu_output  # Backward-compatibility alias
        layout.addWidget(self.btn_menu_output)

        # Menu 3: Capture device (Loopback Capture for Spectrum Analyzer)
        self.btn_menu_capture = QtWidgets.QPushButton("Capture device")
        self.btn_menu_capture.setObjectName("topMenuBtn")
        self.btn_menu_capture.setToolTip("Select loopback capture device for spectrum analyzer")
        self.btn_menu_capture.clicked.connect(self._show_capture_menu)
        layout.addWidget(self.btn_menu_capture)

        # Menu 4: Hearing test
        self.btn_menu_hearing_test = QtWidgets.QPushButton("Hearing test")
        self.btn_menu_hearing_test.setObjectName("topMenuBtn")
        self.btn_menu_hearing_test.setToolTip("Open hearing calibration test (Ctrl+H)")
        self.btn_menu_hearing_test.clicked.connect(self._open_hearing_test)
        layout.addWidget(self.btn_menu_hearing_test)

        # Middle stretch allows window dragging across the top bar
        layout.addStretch(1)

        # Active output & APO health indicator (clicking opens Settings)
        self.btn_apo_health = ApoHealthButton()
        self.btn_apo_health.setObjectName("apoHealthBtn")
        self.btn_apo_health.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_apo_health.setFixedHeight(28)
        self.btn_apo_health.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        health_layout = QtWidgets.QHBoxLayout(self.btn_apo_health)
        health_layout.setContentsMargins(8, 0, 8, 0)
        health_layout.setSpacing(6)
        self.apo_health_dot = StatusDotWidget("#4cc38a", filled=True, parent=self.btn_apo_health)
        self.apo_health_dot.setVisible(False)
        self.apo_health_lbl = QtWidgets.QLabel("Output: All")
        self.apo_health_lbl.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; background: transparent; border: none;"
        )
        self.apo_health_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        health_layout.addWidget(self.apo_health_lbl)
        self.btn_apo_health.clicked.connect(self._open_settings)
        layout.addWidget(self.btn_apo_health)

        # Quiet '?' Shortcuts button
        self.btn_menu_shortcuts = QtWidgets.QPushButton("?")
        self.btn_menu_shortcuts.setObjectName("topMenuBtn")
        self.btn_menu_shortcuts.setFixedSize(28, 28)
        self.btn_menu_shortcuts.setStyleSheet(
            f"QPushButton#topMenuBtn {{ font-family: {MONO_FONT_FAMILY}; font-size: 12px; font-weight: 500; padding: 0px; }}"
        )
        self.btn_menu_shortcuts.setToolTip("Keyboard shortcuts & gestures (?)")
        self.btn_menu_shortcuts.clicked.connect(self._open_shortcuts_dialog)
        layout.addWidget(self.btn_menu_shortcuts)

        layout.addSpacing(4)

        # Far Right: Window Controls
        win_controls_box = QtWidgets.QHBoxLayout()
        win_controls_box.setSpacing(2)
        win_controls_box.setContentsMargins(0, 0, 0, 0)

        self.btn_min = WindowControlButton("min", self)
        self.btn_min.setToolTip("Minimize")
        self.btn_min.clicked.connect(self.showMinimized)
        win_controls_box.addWidget(self.btn_min)

        self.btn_max = WindowControlButton("max", self)
        self.btn_max.setToolTip("Maximize / Restore")
        self.btn_max.clicked.connect(self._toggle_maximize_restore)
        win_controls_box.addWidget(self.btn_max)

        self.btn_close = WindowControlButton("close", self)
        self.btn_close.setToolTip("Close")
        self.btn_close.clicked.connect(self.close)
        win_controls_box.addWidget(self.btn_close)

        layout.addLayout(win_controls_box)
        return frame

    # -------------------------------------------------------------------------
    # Operational Control Bar (Row 2)
    # -------------------------------------------------------------------------
    def _create_control_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("controlBarFrame")
        frame.setFixedHeight(40)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(8)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Left: Quiet raised graphite EQ On toggle with crisp 6px painted status dot
        self.eq_universal_toggle = StatusToggleButton("EQ On")
        self.eq_universal_toggle.setObjectName("activeEqBtn")
        self.eq_universal_toggle.set_status_dot("#4cc38a", filled=True)
        self.eq_universal_toggle.setFixedHeight(28)
        self.eq_universal_toggle.setFixedWidth(80)
        self.eq_universal_toggle.setToolTip("Toggle universal Equalizer APO bypass (Ctrl+B)")
        self.eq_universal_toggle.clicked.connect(self._toggle_universal_eq)
        layout.addWidget(self.eq_universal_toggle)

        # Left: Quiet raised graphite Spectrum toggle + compact options chevron
        spec_group = QtWidgets.QHBoxLayout()
        spec_group.setSpacing(4)
        spec_group.setContentsMargins(0, 0, 0, 0)

        self.spectrum_toggle_btn = StatusToggleButton("Spectrum")
        self.spectrum_toggle_btn.setCheckable(True)
        self.spectrum_toggle_btn.setChecked(True)
        self.spectrum_toggle_btn.setObjectName("activeEqBtn")
        self.spectrum_toggle_btn.set_status_dot("#4cc38a", filled=True)
        self.spectrum_toggle_btn.setFixedHeight(28)
        self.spectrum_toggle_btn.setFixedWidth(96)
        self.spectrum_toggle_btn.setToolTip("Toggle live audio spectrum analyzer")
        self.spectrum_toggle_btn.clicked.connect(self._toggle_spectrum)
        spec_group.addWidget(self.spectrum_toggle_btn)

        self.spectrum_options_btn = ChevronMenuButton(self)
        self.spectrum_options_btn.setToolTip("Spectrum options (peak hold, freeze, tilt, speed)")
        self.spectrum_options_btn.clicked.connect(self._show_spectrum_options_menu)
        spec_group.addWidget(self.spectrum_options_btn)

        self.spectrum_frozen_badge = QtWidgets.QPushButton("Frozen")
        self.spectrum_frozen_badge.setFixedHeight(28)
        self.spectrum_frozen_badge.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.spectrum_frozen_badge.setToolTip("Spectrum display frozen - click to resume")
        self.spectrum_frozen_badge.setStyleSheet(
            f"QPushButton {{ font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #d99b38; "
            f"background-color: #121214; border: 1px solid #3f3f46; border-radius: 4px; padding: 0px 8px; }}"
            f"QPushButton:hover {{ background-color: #1c1c1f; border-color: #d99b38; }}"
        )
        self.spectrum_frozen_badge.clicked.connect(lambda: self._set_spectrum_frozen(False))
        self.spectrum_frozen_badge.setVisible(False)
        spec_group.addWidget(self.spectrum_frozen_badge)
        layout.addLayout(spec_group)

        layout.addStretch(1)

        # Right: Preamp Control + Live Peak Headroom Readout + Auto Preamp
        preamp_box = QtWidgets.QHBoxLayout()
        preamp_box.setSpacing(8)
        preamp_box.setContentsMargins(0, 0, 0, 0)
        preamp_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        preamp_lbl = QtWidgets.QLabel("Preamp")
        preamp_lbl.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        preamp_box.addWidget(preamp_lbl)

        self.preamp_spin = CleanSpinBox()
        self.preamp_spin.setRange(-30.0, 15.0)
        self.preamp_spin.setSingleStep(0.5)
        self.preamp_spin.setSuffix(" dB")
        self.preamp_spin.setValue(0.0)
        self.preamp_spin.setFixedHeight(28)
        self.preamp_spin.setFixedWidth(84)
        self.preamp_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preamp_spin.setStyleSheet(
            f"QDoubleSpinBox {{ font-family: {MONO_FONT_FAMILY}; font-size: 11px; "
            f"background-color: #121214; color: #ededed; border: 1px solid #27272a; "
            f"border-radius: 4px; padding: 0px 8px; min-height: 0px; }}"
            f"QDoubleSpinBox:hover {{ border-color: #3f3f46; }}"
            f"QDoubleSpinBox:focus {{ border: 1px solid #ededed; }}"
        )
        preamp_box.addWidget(self.preamp_spin)

        self.preamp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.preamp_slider.setRange(-300, 150)
        self.preamp_slider.setValue(0)
        self.preamp_slider.setFixedHeight(28)
        self.preamp_slider.setFixedWidth(116)
        preamp_box.addWidget(self.preamp_slider)

        self.preamp_headroom_lbl = QtWidgets.QLabel("0.0 dB peak")
        self.preamp_headroom_lbl.setFixedWidth(92)
        self.preamp_headroom_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.preamp_headroom_lbl.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #71717a; background: transparent;"
        )
        preamp_box.addWidget(self.preamp_headroom_lbl)

        self.btn_auto_preamp = QtWidgets.QPushButton("Auto")
        self.btn_auto_preamp.setFixedHeight(28)
        self.btn_auto_preamp.setFixedWidth(52)
        self.btn_auto_preamp.setStyleSheet(
            "QPushButton { background-color: #121214; color: #ededed; border: 1px solid #27272a; border-radius: 4px; padding: 0px 8px; font-size: 11px; font-weight: 500; }"
            "QPushButton:hover { background-color: #1c1c1f; border-color: #3f3f46; }"
            "QPushButton:pressed { background-color: #27272a; }"
        )
        self.btn_auto_preamp.setToolTip("Set preamp automatically to prevent digital clipping")
        self.btn_auto_preamp.clicked.connect(self._apply_auto_preamp)
        preamp_box.addWidget(self.btn_auto_preamp)

        layout.addLayout(preamp_box)

        return frame

    def _refresh_apo_health_indicator(self) -> None:
        """Refreshes the top-bar active output label and 6px APO health dot."""
        if not hasattr(self, "apo_health_dot") or not hasattr(self, "apo_health_lbl"):
            return
        try:
            diag = self.preset_manager.diagnose_apo()
        except Exception:
            diag = {"status_level": "warning", "status_summary": "APO status unknown", "issues": []}
        level = diag.get("status_level", "warning")
        color_map = {"ok": "#4cc38a", "warning": "#d99b38", "error": "#e5484d"}
        dot_color = color_map.get(level, "#d99b38")
        self.apo_health_dot.set_status(dot_color, filled=bool(diag.get("apo_installed", True)))

        dev_raw = self.preset_manager.active_device_name or "all"
        dev_clean = clean_apo_device_name(dev_raw)
        short_dev = "All" if dev_clean.lower() == "all" else dev_clean
        if len(short_dev) > 22:
            short_dev = short_dev[:21] + "..."
        self.apo_health_lbl.setText(f"Output: {short_dev}")
        self.apo_health_lbl.adjustSize()
        self.btn_apo_health.updateGeometry()
        self.btn_apo_health.adjustSize()

        summary = str(diag.get("status_summary") or "").rstrip(".")
        issues = [str(i).rstrip(".") for i in (diag.get("issues") or []) if i]
        tip_lines = [summary] if summary else []
        if issues:
            tip_lines.extend(f"- {i}" for i in issues)
        tip_lines.append("Click to open Equalizer APO settings")
        self.btn_apo_health.setToolTip("\n".join(tip_lines))

    def _open_shortcuts_dialog(self) -> None:
        dlg = ShortcutsDialog(self)
        dlg.exec()
        dlg.deleteLater()

    def _open_shortcuts_dialog_shortcut(self) -> None:
        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox)):
            return
        self._open_shortcuts_dialog()

    def _update_preamp_headroom_badge(self, preamp_override: Optional[float] = None) -> None:
        """Updates the tabular monospace peak headroom badge beside Preamp."""
        if not hasattr(self, "preamp_headroom_lbl") or not hasattr(self, "eq_plot"):
            return
        preamp_val = self.preamp_spin.value() if preamp_override is None else float(preamp_override)
        peak_db, rec_preamp = compute_peak_headroom(self.eq_plot.bands, preamp_val, fs=self.sample_rate)
        if peak_db > 0.0:
            col = "#e5484d"
            state_txt = "Clipping"
        elif peak_db >= -0.5:
            col = "#d99b38"
            state_txt = "Near 0 dB"
        else:
            col = "#71717a"
            state_txt = "Headroom ok"
        self.preamp_headroom_lbl.setText(f"{peak_db:+.1f} dB peak")
        self.preamp_headroom_lbl.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; font-weight: {'600' if peak_db > 0.0 else '400'}; "
            f"color: {col}; background: transparent;"
        )
        self.preamp_headroom_lbl.setToolTip(
            f"{state_txt} · Composite peak {peak_db:+.2f} dB · Recommended preamp {rec_preamp:+.1f} dB"
        )

    def _apply_auto_preamp(self) -> None:
        """One-click Auto Preamp: sets preamp to recommended anti-clipping headroom with undo snapshot."""
        _, rec_preamp = compute_peak_headroom(self.eq_plot.bands, self.preamp_spin.value(), fs=self.sample_rate)
        rec_preamp = round(max(-30.0, min(15.0, float(rec_preamp))), 1)
        if abs(rec_preamp - self.preamp_spin.value()) < 1e-4:
            self._update_preamp_headroom_badge(rec_preamp)
            return
        self._record_undo_snapshot()
        self._apply_preamp_value(rec_preamp)
        self.preset_manager.update_active_preset_content(rec_preamp, self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(rec_preamp, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._commit_state()

    def _show_spectrum_options_menu(self) -> None:
        menu = QtWidgets.QMenu(self)

        act_peak = menu.addAction("Peak hold")
        act_peak.setCheckable(True)
        act_peak.setChecked(self._spectrum_peak_hold)
        act_peak.triggered.connect(self._set_spectrum_peak_hold)

        act_freeze = menu.addAction("Freeze")
        act_freeze.setCheckable(True)
        act_freeze.setChecked(self._spectrum_frozen)
        act_freeze.triggered.connect(self._set_spectrum_frozen)

        menu.addSeparator()

        tilt_head = menu.addAction("Tilt")
        tilt_head.setEnabled(False)
        for label, mode in (
            ("Pink (+3 dB/oct)", "pink"),
            ("Slope (+4.5 dB/oct)", "slope"),
            ("Flat (0 dB/oct)", "flat"),
        ):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._spectrum_tilt == mode)
            act.triggered.connect(lambda _=False, m=mode: self._set_spectrum_tilt(m))

        menu.addSeparator()

        speed_head = menu.addAction("Speed")
        speed_head.setEnabled(False)
        for label, mode in (("Fast", "fast"), ("Medium", "medium"), ("Slow", "slow")):
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._spectrum_speed == mode)
            act.triggered.connect(lambda _=False, m=mode: self._set_spectrum_speed(m))

        menu.addSeparator()

        act_reset = menu.addAction("Reset peak")
        act_reset.triggered.connect(self._reset_spectrum_peak)

        menu.exec(self.spectrum_options_btn.mapToGlobal(QtCore.QPoint(0, self.spectrum_options_btn.height() + 4)))

    def _set_spectrum_peak_hold(self, enabled: bool) -> None:
        self._spectrum_peak_hold = bool(enabled)
        if hasattr(self.spectrum_worker, "set_peak_hold_enabled"):
            self.spectrum_worker.set_peak_hold_enabled(self._spectrum_peak_hold)
        if hasattr(self.eq_plot, "set_peak_hold_visible"):
            self.eq_plot.set_peak_hold_visible(self._spectrum_peak_hold and self.spectrum_toggle_btn.isChecked())

    def _set_spectrum_frozen(self, frozen: bool) -> None:
        self._spectrum_frozen = bool(frozen)
        if hasattr(self.spectrum_worker, "set_frozen"):
            self.spectrum_worker.set_frozen(self._spectrum_frozen)
        self.spectrum_frozen_badge.setVisible(self._spectrum_frozen)
        if self.spectrum_toggle_btn.isChecked():
            self.spectrum_toggle_btn.set_status_dot("#d99b38" if self._spectrum_frozen else "#4cc38a", filled=True)

    def _set_spectrum_tilt(self, mode: str) -> None:
        self._spectrum_tilt = mode
        if hasattr(self.spectrum_worker, "set_tilt_mode"):
            self.spectrum_worker.set_tilt_mode(mode)

    def _set_spectrum_speed(self, mode: str) -> None:
        self._spectrum_speed = mode
        if hasattr(self.spectrum_worker, "set_speed_mode"):
            self.spectrum_worker.set_speed_mode(mode)

    def _reset_spectrum_peak(self) -> None:
        if hasattr(self.spectrum_worker, "reset_peak_hold"):
            self.spectrum_worker.reset_peak_hold()
        if hasattr(self.eq_plot, "update_peak_hold_curve"):
            self.eq_plot.update_peak_hold_curve([], [])

    def _on_peak_hold_ready(self, freqs: Any, mags: Any) -> None:
        if hasattr(self.eq_plot, "update_peak_hold_curve"):
            self.eq_plot.update_peak_hold_curve(freqs, mags)

    def _show_output_menu(self) -> None:
        self.btn_menu_output.setStyleSheet("background-color: #27272a; color: #ffffff;")
        menu = self._create_output_menu()
        menu.exec(self.btn_menu_output.mapToGlobal(QtCore.QPoint(0, self.btn_menu_output.height() + 3)))
        self.btn_menu_output.setStyleSheet("")

    def _show_input_menu(self) -> None:
        """Backward-compatibility alias for _show_output_menu."""
        self._show_output_menu()

    def _create_output_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        active_name = self.preset_manager.active_device_name
        is_all = clean_apo_device_name(active_name) == "all"

        act_all = menu.addAction("All Endpoints (Default / Global)")
        act_all.setCheckable(True)
        act_all.setChecked(is_all)
        act_all.triggered.connect(lambda: self._on_output_device_selected(None, "All Endpoints"))

        devices = self._get_playback_devices()
        if devices:
            menu.addSeparator()

            # Identify Windows Default output device for helpful badging
            default_desc = ""
            if self._media_devices:
                try:
                    def_out = self._media_devices.defaultAudioOutput()
                    if def_out:
                        default_desc = def_out.description().strip()
                except Exception:
                    pass

            for dev_name in devices:
                label = dev_name
                if default_desc and (clean_apo_device_name(default_desc).lower() == clean_apo_device_name(dev_name).lower()):
                    label = f"{dev_name}  (Windows Default)"
                act = menu.addAction(label)
                act.setCheckable(True)
                checked = not is_all and (clean_apo_device_name(active_name).lower() == clean_apo_device_name(dev_name).lower())
                act.setChecked(checked)
                act.triggered.connect(lambda checked, d=dev_name: self._on_output_device_selected(d, d))

        return menu

    def _create_input_menu(self) -> QtWidgets.QMenu:
        """Backward-compatibility alias for _create_output_menu."""
        return self._create_output_menu()

    def _get_playback_devices(self) -> List[str]:
        return get_available_playback_devices()

    def _on_output_device_selected(self, dev_id: Optional[str], dev_name: str) -> None:
        self.preset_manager.set_active_device_name(dev_id)
        preamp_val, bands = self._collect_current_eq_state()
        self.preset_manager.schedule_apo_export(
            preamp_val, bands, bypassed=self.is_eq_bypassed
        )
        self._refresh_apo_health_indicator()
        self._check_device_apo_status(dev_name, user_initiated=True)

    def _on_input_device_selected(self, dev_id: Optional[str], dev_name: str) -> None:
        """Backward-compatibility alias for _on_output_device_selected."""
        self._on_output_device_selected(dev_id, dev_name)

    def _show_capture_menu(self) -> None:
        self.btn_menu_capture.setStyleSheet("background-color: #27272a; color: #ffffff;")
        menu = self._create_capture_menu()
        menu.exec(self.btn_menu_capture.mapToGlobal(QtCore.QPoint(0, self.btn_menu_capture.height() + 3)))
        self.btn_menu_capture.setStyleSheet("")

    def _create_capture_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        current_idx = self.preset_manager.audio_device_index
        current_name = getattr(self.preset_manager, "audio_device_name", None)
        devices = get_available_audio_devices()

        is_auto_follow = (current_idx is None and not current_name) or (current_name and ("auto-follow" in current_name.lower() or "default" in current_name.lower()))

        act_default = menu.addAction("Default Windows Audio (Auto-Follow)")
        act_default.setCheckable(True)
        act_default.setChecked(is_auto_follow)
        act_default.triggered.connect(lambda: self._on_capture_device_selected(None, None))

        non_default = [(idx, name) for idx, name in devices if name and "auto-follow" not in name.lower()]
        if non_default:
            menu.addSeparator()
            for idx, name in non_default:
                act = menu.addAction(name)
                act.setCheckable(True)
                checked = False
                if not is_auto_follow:
                    if current_name and clean_apo_device_name(current_name).lower() == clean_apo_device_name(name).lower():
                        checked = True
                    elif current_idx is not None and idx is not None and current_idx == idx:
                        checked = True
                act.setChecked(checked)
                act.triggered.connect(lambda checked, i=idx, n=name: self._on_capture_device_selected(i, n))

        return menu

    def _on_capture_device_selected(self, dev_idx: Optional[int], dev_name: Optional[str]) -> None:
        self.spectrum_worker.set_device_target(dev_idx, dev_name)
        self.preset_manager.audio_device_index = dev_idx
        self.preset_manager.audio_device_name = dev_name
        self.preset_manager.save_settings()

    # -------------------------------------------------------------------------
    # Left Sidebar: Profile Management & AutoEQ
    # -------------------------------------------------------------------------
    def _create_sidebar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("sidebarFrame")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top Navigation Tab Switcher (sentence case, quiet raised graphite active pill)
        self.sidebar_tab_bar = SegmentedPillTabBar(["Profiles", "AutoEQ"], self, height=32, font_size=9)
        self.sidebar_tab_bar.sigTabChanged.connect(self._switch_sidebar_tab)
        layout.addWidget(self.sidebar_tab_bar)

        # Stacked Container with smooth cubic slide animation
        self.sidebar_stack = SlidingStackedWidget()
        self.sidebar_stack.addWidget(self._create_profiles_panel())
        self.sidebar_stack.addWidget(self._create_autoeq_panel())
        layout.addWidget(self.sidebar_stack)

        return frame

    def _create_profiles_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # Header with Import and Export controls
        head_box = QtWidgets.QHBoxLayout()
        head_box.setSpacing(4)
        header = QtWidgets.QLabel("Profiles")
        header.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        head_box.addWidget(header)
        head_box.addStretch()

        self.import_btn = QtWidgets.QPushButton("Import")
        self.import_btn.setFixedHeight(28)
        self.import_btn.setStyleSheet("padding: 0px 10px; font-size: 11px;")
        self.import_btn.setToolTip("Import EQ from file or paste APO text from clipboard")
        self.import_btn.clicked.connect(self._show_import_menu)
        head_box.addWidget(self.import_btn)

        self.export_btn = QtWidgets.QPushButton("Export")
        self.export_btn.setFixedHeight(28)
        self.export_btn.setStyleSheet("padding: 0px 10px; font-size: 11px;")
        self.export_btn.setToolTip("Export active profile or full library (Ctrl+E)")
        self.export_btn.clicked.connect(self._show_export_menu)
        head_box.addWidget(self.export_btn)
        layout.addLayout(head_box)

        # Search box
        self.preset_search = QtWidgets.QLineEdit()
        self.preset_search.setFixedHeight(32)
        self.preset_search.setPlaceholderText("Search profiles")
        self.preset_search.textChanged.connect(self._filter_presets_list)
        layout.addWidget(self.preset_search)

        # Preset list with inline editing & animated highlight
        self.preset_list_widget = AnimatedPresetListWidget(self)
        self.preset_list_widget.itemClicked.connect(self._on_preset_item_clicked)
        self.preset_list_widget.itemDoubleClicked.connect(self._on_preset_item_double_clicked)
        self.preset_list_widget.itemChanged.connect(self._on_preset_item_renamed)
        self.preset_list_widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.preset_list_widget.customContextMenuRequested.connect(self._show_preset_context_menu)
        layout.addWidget(self.preset_list_widget)

        # Action Buttons (all quiet graphite; no competing solid-white button in Profiles pane)
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)

        ren_btn = QtWidgets.QPushButton("Rename")
        ren_btn.setFixedHeight(32)
        ren_btn.setToolTip("Rename selected profile (F2)")
        ren_btn.clicked.connect(self._rename_preset_inline)
        btn_grid.addWidget(ren_btn, 0, 0)

        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.setFixedHeight(32)
        dup_btn.clicked.connect(self._duplicate_preset)
        btn_grid.addWidget(dup_btn, 0, 1)

        add_btn = QtWidgets.QPushButton("+ Add")
        add_btn.setFixedHeight(32)
        add_btn.setStyleSheet(
            "QPushButton { background-color: #1c1c1f; color: #ededed; border: 1px solid #3f3f46; border-radius: 4px; font-weight: 500; }"
            "QPushButton:hover { background-color: #27272a; border-color: #3f3f46; }"
        )
        add_btn.setToolTip("Create a new flat EQ profile (Ctrl+N)")
        add_btn.clicked.connect(self._add_preset_auto)
        btn_grid.addWidget(add_btn, 1, 0)

        del_btn = QtWidgets.QPushButton("Delete")
        del_btn.setObjectName("dangerBtn")
        del_btn.setFixedHeight(32)
        del_btn.clicked.connect(self._delete_preset)
        btn_grid.addWidget(del_btn, 1, 1)

        layout.addLayout(btn_grid)
        return panel

    def _create_autoeq_panel(self) -> QtWidgets.QWidget:
        scroll = AutoResizingScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("QScrollArea { border: none; background: #000000; }")

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # 0. Global Database Search & Offline Library
        search_sync_row = QtWidgets.QHBoxLayout()
        search_sync_row.setSpacing(4)

        self.autoeq_global_search_btn = QtWidgets.QPushButton("Global search")
        self.autoeq_global_search_btn.setObjectName("globalSearchBtn")
        self.autoeq_global_search_btn.setFixedHeight(32)
        self.autoeq_global_search_btn.setToolTip("Search indexed IEM and headphone measurement catalogs")
        self.autoeq_global_search_btn.clicked.connect(self._open_global_search)
        search_sync_row.addWidget(self.autoeq_global_search_btn, stretch=3)

        self.autoeq_offline_sync_btn = QtWidgets.QPushButton("Offline DB")
        self.autoeq_offline_sync_btn.setFixedHeight(32)
        self.autoeq_offline_sync_btn.setToolTip("Download and manage offline measurement and target curves")
        self.autoeq_offline_sync_btn.setStyleSheet(
            "QPushButton { background-color: #121214; border: 1px solid #27272a; border-radius: 4px; color: #ededed; font-size: 11px; font-weight: 500; padding: 0px 8px; }"
            "QPushButton:hover { background-color: #1c1c1f; border-color: #3f3f46; }"
        )
        self.autoeq_offline_sync_btn.clicked.connect(self._open_offline_sync)
        search_sync_row.addWidget(self.autoeq_offline_sync_btn, stretch=2)

        layout.addLayout(search_sync_row)

        # Database Selection
        db_sec = QtWidgets.QVBoxLayout()
        db_sec.setSpacing(4)
        db_lbl = QtWidgets.QLabel("Database")
        db_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        db_sec.addWidget(db_lbl)

        db_row = QtWidgets.QHBoxLayout()
        db_row.setSpacing(4)

        self.autoeq_db_combo = SearchableDatabaseInput(self)
        self.autoeq_db_combo.setFixedHeight(28)
        self.autoeq_db_combo.currentIndexChanged.connect(self._on_autoeq_db_changed)
        db_row.addWidget(self.autoeq_db_combo, stretch=1)

        self.autoeq_db_star_btn = FavoriteStarButton(self)
        self.autoeq_db_star_btn.setToolTip("Pin database to top of list")
        self.autoeq_db_star_btn.clicked.connect(self._toggle_current_db_favorite)
        db_row.addWidget(self.autoeq_db_star_btn)

        db_sec.addLayout(db_row)

        self.autoeq_rig_badge = QtWidgets.QLabel("Rig: IEC 60318-4 (711) · Isolated")
        self.autoeq_rig_badge.setWordWrap(True)
        self.autoeq_rig_badge.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 10px; color: #71717a; background: transparent;")
        self.autoeq_rig_badge.setToolTip("Targets and measurements stay isolated to this reviewer's coupler rig")
        db_sec.addWidget(self.autoeq_rig_badge)
        layout.addLayout(db_sec)

        # Headphone / IEM Model Selection
        model_sec = QtWidgets.QVBoxLayout()
        model_sec.setSpacing(4)
        model_lbl = QtWidgets.QLabel("Headphone / IEM model")
        model_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        model_sec.addWidget(model_lbl)

        model_row = QtWidgets.QHBoxLayout()
        model_row.setSpacing(4)

        self.autoeq_model_combo = SearchableModelInput()
        self.autoeq_model_combo.setFixedHeight(28)
        self.autoeq_model_combo.sigModelSelected.connect(self._on_searchable_model_selected)
        model_row.addWidget(self.autoeq_model_combo, stretch=3)

        self.autoeq_chan_combo = QtWidgets.QComboBox()
        self.autoeq_chan_combo.setFixedHeight(28)
        self.autoeq_chan_combo.addItem("Avg", "avg")
        self.autoeq_chan_combo.addItem("L", "L")
        self.autoeq_chan_combo.addItem("R", "R")
        self.autoeq_chan_combo.setToolTip("Channel: Average (L+R), Left (L), or Right (R)")
        self.autoeq_chan_combo.setFixedWidth(72)
        self.autoeq_chan_combo.setStyleSheet("QComboBox { padding: 2px 20px 2px 8px; }")
        self.autoeq_chan_combo.currentIndexChanged.connect(self._on_autoeq_model_changed)
        model_row.addWidget(self.autoeq_chan_combo)

        model_sec.addLayout(model_row)
        layout.addLayout(model_sec)

        # Target Specification (Target curve / IEM tuning / Resultant)
        target_sec = QtWidgets.QVBoxLayout()
        target_sec.setSpacing(4)
        target_lbl = QtWidgets.QLabel("Target response")
        target_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        target_sec.addWidget(target_lbl)

        self.autoeq_target_mode_tab_bar = SegmentedPillTabBar(
            ["Target curve", "IEM tuning", "Resultant"],
            self,
            height=28,
            font_size=8,
        )
        self.autoeq_target_mode_tab_bar.sigTabChanged.connect(self._on_target_mode_changed)
        target_sec.addWidget(self.autoeq_target_mode_tab_bar)

        self.autoeq_target_stack = QtWidgets.QStackedWidget()

        # Page 0: Target curve
        p0_widget = QtWidgets.QWidget()
        p0_layout = QtWidgets.QVBoxLayout(p0_widget)
        p0_layout.setContentsMargins(0, 4, 0, 0)
        p0_layout.setSpacing(4)
        self.autoeq_target_combo = QtWidgets.QComboBox()
        self.autoeq_target_combo.setFixedHeight(28)
        self.autoeq_target_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p0_layout.addWidget(self.autoeq_target_combo)
        self.autoeq_target_info_label = QtWidgets.QLabel("Aligned: midband MSE (200 - 2,000 Hz)")
        self.autoeq_target_info_label.setStyleSheet("font-size: 10px; color: #71717a; background: transparent;")
        p0_layout.addWidget(self.autoeq_target_info_label)
        self.autoeq_target_stack.addWidget(p0_widget)

        # Page 1: IEM tuning
        p1_widget = QtWidgets.QWidget()
        p1_layout = QtWidgets.QVBoxLayout(p1_widget)
        p1_layout.setContentsMargins(0, 4, 0, 0)
        p1_layout.setSpacing(4)
        p1_row = QtWidgets.QHBoxLayout()
        p1_row.setSpacing(4)
        self.autoeq_target_iem_combo = SearchableTargetIEMInput(squig_client=self.squig_client)
        self.autoeq_target_iem_combo.setFixedHeight(28)
        self.autoeq_target_iem_combo.sigModelSelected.connect(lambda _: self._on_autoeq_target_changed())
        p1_row.addWidget(self.autoeq_target_iem_combo, stretch=3)
        self.autoeq_target_iem_chan_combo = QtWidgets.QComboBox()
        self.autoeq_target_iem_chan_combo.setFixedHeight(28)
        self.autoeq_target_iem_chan_combo.addItem("Avg", "avg")
        self.autoeq_target_iem_chan_combo.addItem("L", "L")
        self.autoeq_target_iem_chan_combo.addItem("R", "R")
        self.autoeq_target_iem_chan_combo.setToolTip("Target channel: Average (L+R), Left (L), or Right (R)")
        self.autoeq_target_iem_chan_combo.setFixedWidth(72)
        self.autoeq_target_iem_chan_combo.setStyleSheet("QComboBox { padding: 2px 20px 2px 8px; }")
        self.autoeq_target_iem_chan_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p1_row.addWidget(self.autoeq_target_iem_chan_combo)
        p1_layout.addLayout(p1_row)

        p1_sub_row = QtWidgets.QHBoxLayout()
        p1_sub_row.setSpacing(8)
        self.autoeq_target_same_provider_cb = QtWidgets.QCheckBox("Same provider only")
        self.autoeq_target_same_provider_cb.setChecked(True)
        self.autoeq_target_same_provider_cb.setToolTip(
            "Keep target search within the current reviewer coupler for acoustic consistency"
        )
        self.autoeq_target_same_provider_cb.setStyleSheet("QCheckBox { color: #a1a1aa; font-size: 11px; }")
        self.autoeq_target_same_provider_cb.toggled.connect(self._on_target_same_provider_toggled)
        p1_sub_row.addWidget(self.autoeq_target_same_provider_cb)

        self.autoeq_target_iem_badge = QtWidgets.QLabel("Rig: matched")
        self.autoeq_target_iem_badge.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 10px; color: #71717a; background: transparent;")
        self.autoeq_target_iem_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        p1_sub_row.addWidget(self.autoeq_target_iem_badge, stretch=1)
        p1_layout.addLayout(p1_sub_row)
        self.autoeq_target_stack.addWidget(p1_widget)

        # Page 2: Resultant profile
        p2_widget = QtWidgets.QWidget()
        p2_layout = QtWidgets.QVBoxLayout(p2_widget)
        p2_layout.setContentsMargins(0, 4, 0, 0)
        p2_layout.setSpacing(4)
        self.autoeq_target_profile_combo = QtWidgets.QComboBox()
        self.autoeq_target_profile_combo.setFixedHeight(28)
        self.autoeq_target_profile_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p2_layout.addWidget(self.autoeq_target_profile_combo)
        self.autoeq_target_profile_info_label = QtWidgets.QLabel("Resultant acoustic response of selected profile")
        self.autoeq_target_profile_info_label.setStyleSheet("font-size: 10px; color: #71717a; background: transparent;")
        p2_layout.addWidget(self.autoeq_target_profile_info_label)
        self.autoeq_target_stack.addWidget(p2_widget)

        target_sec.addWidget(self.autoeq_target_stack)
        self._sync_target_stack_height()
        layout.addLayout(target_sec)

        # Target Preference Sliders (Bass shelf -6..+8 dB, Tilt -3..+3 dB/dec)
        pref_sec = QtWidgets.QVBoxLayout()
        pref_sec.setSpacing(4)
        pref_head = QtWidgets.QHBoxLayout()
        pref_lbl = QtWidgets.QLabel("Target preferences")
        pref_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        pref_head.addWidget(pref_lbl)
        pref_head.addStretch()

        self.autoeq_pref_reset_btn = QtWidgets.QPushButton("Reset")
        self.autoeq_pref_reset_btn.setFixedHeight(20)
        self.autoeq_pref_reset_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #71717a; font-size: 11px; padding: 0px 4px; min-height: 16px; }"
            "QPushButton:hover { color: #ededed; }"
        )
        self.autoeq_pref_reset_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.autoeq_pref_reset_btn.setVisible(False)
        self.autoeq_pref_reset_btn.clicked.connect(self._reset_autoeq_preferences)
        pref_head.addWidget(self.autoeq_pref_reset_btn)
        pref_sec.addLayout(pref_head)

        pref_grid = QtWidgets.QGridLayout()
        pref_grid.setHorizontalSpacing(8)
        pref_grid.setVerticalSpacing(4)

        bass_lbl = QtWidgets.QLabel("Bass")
        bass_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa; background: transparent;")
        pref_grid.addWidget(bass_lbl, 0, 0)

        self.autoeq_bass_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.autoeq_bass_slider.setRange(-60, 80)  # -6.0 to +8.0 dB (0.1 dB units)
        self.autoeq_bass_slider.setSingleStep(5)
        self.autoeq_bass_slider.setPageStep(10)
        self.autoeq_bass_slider.setValue(0)
        self.autoeq_bass_slider.setFixedHeight(24)
        self.autoeq_bass_slider.setToolTip("Low-shelf bass boost/cut applied to target (-6 to +8 dB)")
        self.autoeq_bass_slider.valueChanged.connect(self._on_autoeq_preferences_changed)
        pref_grid.addWidget(self.autoeq_bass_slider, 0, 1)

        self.autoeq_bass_val_lbl = QtWidgets.QLabel("+0.0 dB")
        self.autoeq_bass_val_lbl.setFixedWidth(78)
        self.autoeq_bass_val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.autoeq_bass_val_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; background: transparent;")
        pref_grid.addWidget(self.autoeq_bass_val_lbl, 0, 2)

        tilt_lbl = QtWidgets.QLabel("Tilt")
        tilt_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa; background: transparent;")
        pref_grid.addWidget(tilt_lbl, 1, 0)

        self.autoeq_tilt_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.autoeq_tilt_slider.setRange(-30, 30)  # -3.0 to +3.0 dB/dec (0.1 units)
        self.autoeq_tilt_slider.setSingleStep(2)
        self.autoeq_tilt_slider.setPageStep(5)
        self.autoeq_tilt_slider.setValue(0)
        self.autoeq_tilt_slider.setFixedHeight(24)
        self.autoeq_tilt_slider.setToolTip("Spectral tilt anchored at 1 kHz (-3 to +3 dB/dec)")
        self.autoeq_tilt_slider.valueChanged.connect(self._on_autoeq_preferences_changed)
        pref_grid.addWidget(self.autoeq_tilt_slider, 1, 1)

        self.autoeq_tilt_val_lbl = QtWidgets.QLabel("+0.0 dB/dec")
        self.autoeq_tilt_val_lbl.setFixedWidth(78)
        self.autoeq_tilt_val_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.autoeq_tilt_val_lbl.setStyleSheet(f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; background: transparent;")
        pref_grid.addWidget(self.autoeq_tilt_val_lbl, 1, 2)

        pref_grid.setColumnStretch(1, 1)
        pref_sec.addLayout(pref_grid)
        layout.addLayout(pref_sec)

        # Optimization Settings (2x2 Grid, aligned 28px heights)
        opt_sec = QtWidgets.QVBoxLayout()
        opt_sec.setSpacing(4)
        opt_lbl = QtWidgets.QLabel("Optimizer")
        opt_lbl.setStyleSheet("font-size: 11px; font-weight: 500; color: #a1a1aa; background: transparent;")
        opt_sec.addWidget(opt_lbl)

        opt_grid = QtWidgets.QGridLayout()
        opt_grid.setHorizontalSpacing(8)
        opt_grid.setVerticalSpacing(4)

        bands_lbl = QtWidgets.QLabel("Bands")
        bands_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa; background: transparent;")
        opt_grid.addWidget(bands_lbl, 0, 0)

        self.autoeq_bands_spin = QtWidgets.QSpinBox()
        self.autoeq_bands_spin.setRange(1, 32)
        self.autoeq_bands_spin.setValue(7)
        self.autoeq_bands_spin.setFixedHeight(28)
        self.autoeq_bands_spin.setFixedWidth(84)
        self.autoeq_bands_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        opt_grid.addWidget(self.autoeq_bands_spin, 0, 1)

        self.autoeq_bass_shelf_cb = QtWidgets.QPushButton("Bass shelf")
        self.autoeq_bass_shelf_cb.setCheckable(True)
        self.autoeq_bass_shelf_cb.setChecked(True)
        self.autoeq_bass_shelf_cb.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.autoeq_bass_shelf_cb.setToolTip("Allocate Band 1 as a Low Shelf filter for sub-bass correction")
        self.autoeq_bass_shelf_cb.setFixedHeight(28)
        self.autoeq_bass_shelf_cb.setStyleSheet("""
            QPushButton {
                background-color: #121214;
                color: #71717a;
                border: 1px solid #27272a;
                border-radius: 4px;
                padding: 0px 8px;
                min-height: 26px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1c1c1f;
                border-color: #3f3f46;
                color: #a1a1aa;
            }
            QPushButton:checked {
                background-color: #1c1c1f;
                color: #ededed;
                border: 1px solid #3f3f46;
                font-weight: 500;
            }
            QPushButton:checked:hover {
                background-color: #27272a;
                color: #ededed;
                border-color: #3f3f46;
            }
        """)
        opt_grid.addWidget(self.autoeq_bass_shelf_cb, 0, 2, 1, 2)

        boost_lbl = QtWidgets.QLabel("Boost")
        boost_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa; background: transparent;")
        opt_grid.addWidget(boost_lbl, 1, 0)

        self.autoeq_max_boost_spin = CleanSpinBox()
        self.autoeq_max_boost_spin.setRange(0.0, 15.0)
        self.autoeq_max_boost_spin.setValue(9.0)
        self.autoeq_max_boost_spin.setSingleStep(0.5)
        self.autoeq_max_boost_spin.setSuffix(" dB")
        self.autoeq_max_boost_spin.setFixedHeight(28)
        self.autoeq_max_boost_spin.setFixedWidth(84)
        self.autoeq_max_boost_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        opt_grid.addWidget(self.autoeq_max_boost_spin, 1, 1)

        treble_lbl = QtWidgets.QLabel("Limit")
        treble_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa; background: transparent;")
        opt_grid.addWidget(treble_lbl, 1, 2)

        self.autoeq_max_freq_spin = CleanSpinBox()
        self.autoeq_max_freq_spin.setRange(4000.0, 10000.0)
        self.autoeq_max_freq_spin.setDecimals(0)
        self.autoeq_max_freq_spin.setValue(8500.0)
        self.autoeq_max_freq_spin.setSingleStep(500.0)
        self.autoeq_max_freq_spin.setSuffix(" Hz")
        self.autoeq_max_freq_spin.setFixedHeight(28)
        self.autoeq_max_freq_spin.setFixedWidth(92)
        self.autoeq_max_freq_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.autoeq_max_freq_spin.setToolTip("Limit aggressive high-frequency peaks above coupler resonance range")
        opt_grid.addWidget(self.autoeq_max_freq_spin, 1, 3)

        opt_sec.addLayout(opt_grid)
        layout.addLayout(opt_sec)

        # Single Primary Action Button in AutoEQ pane
        self.autoeq_compute_btn = QtWidgets.QPushButton("Compute AutoEQ")
        self.autoeq_compute_btn.setObjectName("primaryBtn")
        self.autoeq_compute_btn.setFixedHeight(32)
        self.autoeq_compute_btn.clicked.connect(self._compute_autoeq)
        layout.addWidget(self.autoeq_compute_btn)

        self.autoeq_status_label = QtWidgets.QLabel("Ready")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")
        self.autoeq_status_label.setWordWrap(True)
        layout.addWidget(self.autoeq_status_label)

        # Region RMS tabular-mono summary (hidden until optimization finishes)
        self.autoeq_rms_label = QtWidgets.QLabel("")
        self.autoeq_rms_label.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; color: #a1a1aa; font-size: 10px; background: transparent;"
        )
        self.autoeq_rms_label.setWordWrap(True)
        self.autoeq_rms_label.setVisible(False)
        layout.addWidget(self.autoeq_rms_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        self.autoeq_save_preset_btn = QtWidgets.QPushButton("Save profile")
        self.autoeq_save_preset_btn.setFixedHeight(28)
        self.autoeq_save_preset_btn.clicked.connect(self._save_autoeq_profile)
        btn_row.addWidget(self.autoeq_save_preset_btn)

        self.autoeq_clear_overlays_btn = QtWidgets.QPushButton("Clear traces")
        self.autoeq_clear_overlays_btn.setFixedHeight(28)
        self.autoeq_clear_overlays_btn.clicked.connect(self._clear_autoeq_overlays)
        btn_row.addWidget(self.autoeq_clear_overlays_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _sync_target_stack_height(self) -> None:
        """Prevents QStackedWidget from reserving inactive taller-page height."""
        if not hasattr(self, "autoeq_target_stack"):
            return
        cur = self.autoeq_target_stack.currentIndex()
        for i in range(self.autoeq_target_stack.count()):
            w = self.autoeq_target_stack.widget(i)
            if w is None:
                continue
            if i == cur:
                w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
            else:
                w.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        self.autoeq_target_stack.adjustSize()

    def _on_autoeq_preferences_changed(self) -> None:
        self._autoeq_bass_boost_db = round(self.autoeq_bass_slider.value() / 10.0, 1)
        self._autoeq_treble_tilt_db = round(self.autoeq_tilt_slider.value() / 10.0, 1)
        self.autoeq_bass_val_lbl.setText(f"{self._autoeq_bass_boost_db:+.1f} dB")
        self.autoeq_tilt_val_lbl.setText(f"{self._autoeq_treble_tilt_db:+.1f} dB/dec")
        is_mod = (abs(self._autoeq_bass_boost_db) > 1e-4) or (abs(self._autoeq_treble_tilt_db) > 1e-4)
        self.autoeq_pref_reset_btn.setVisible(is_mod)
        self.autoeq_bass_val_lbl.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: {'#ededed' if abs(self._autoeq_bass_boost_db) > 1e-4 else '#71717a'}; background: transparent;"
        )
        self.autoeq_tilt_val_lbl.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: {'#ededed' if abs(self._autoeq_treble_tilt_db) > 1e-4 else '#71717a'}; background: transparent;"
        )
        self._refresh_target_preference_preview()

    def _reset_autoeq_preferences(self) -> None:
        self.autoeq_bass_slider.blockSignals(True)
        self.autoeq_tilt_slider.blockSignals(True)
        self.autoeq_bass_slider.setValue(0)
        self.autoeq_tilt_slider.setValue(0)
        self.autoeq_bass_slider.blockSignals(False)
        self.autoeq_tilt_slider.blockSignals(False)
        self._on_autoeq_preferences_changed()

    def _get_effective_target_raw(self) -> Tuple[List[float], List[float]]:
        """Returns target curve adjusted by live Bass/Tilt preference sliders."""
        tf, ts = self._autoeq_target_raw
        if not tf or not ts or len(tf) != len(ts):
            return tf, ts
        if abs(self._autoeq_bass_boost_db) < 1e-4 and abs(self._autoeq_treble_tilt_db) < 1e-4:
            return tf, ts
        adj = apply_target_preferences(
            tf,
            ts,
            bass_boost_db=self._autoeq_bass_boost_db,
            treble_tilt_db=self._autoeq_treble_tilt_db,
        )
        return list(tf), adj.tolist()

    def _refresh_target_preference_preview(self) -> None:
        """Updates the displayed Target curve overlay live when Bass/Tilt sliders move."""
        mf, ms = self._autoeq_meas_raw
        tf, ts = self._get_effective_target_raw()
        if not tf or not ts:
            return
        dm, dt, dr = self._compute_normalized_display_curves(mf, ms, tf, ts, self.eq_plot.bands)
        if dt[0]:
            mode = getattr(self, "_autoeq_target_mode", 0)
            if mode == 0:
                target = self.autoeq_target_combo.currentData() if hasattr(self, "autoeq_target_combo") else None
                t_label = target.name if (target and getattr(target, "name", "")) else "Target"
            elif mode == 1:
                t_iem = self.autoeq_target_iem_combo.currentData() if hasattr(self, "autoeq_target_iem_combo") else None
                t_label = f"{t_iem.display_name} (Target)" if t_iem else "Target IEM"
            else:
                prof = self.autoeq_target_profile_combo.currentData() if hasattr(self, "autoeq_target_profile_combo") else ""
                t_label = f"{prof} (Resultant)" if prof else "Target Resultant"
            if abs(self._autoeq_bass_boost_db) > 1e-4 or abs(self._autoeq_treble_tilt_db) > 1e-4:
                t_label = f"{t_label} (Pref)"
            self.eq_plot.set_target_overlay(dt[0], dt[1], label=t_label)
            if dm[0]:
                self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
                self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)
                self.eq_plot.set_measurement_overlay(dm[0], dm[1])
                if dr[0]:
                    self.eq_plot.set_resultant_overlay(dr[0], dr[1])
            self.eq_plot.set_overlays_visible(
                self.undergraph_meas_btn.isChecked(),
                self.undergraph_target_btn.isChecked(),
                self.undergraph_resultant_btn.isChecked(),
                show_target2=self.undergraph_target2_btn.isChecked(),
            )

    # -------------------------------------------------------------------------
    # Bottom Filter Table Panel
    # -------------------------------------------------------------------------
    def _create_filter_table_panel(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("tableFrame")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(8)
        self.band_count_label = QtWidgets.QLabel("Bands (0)")
        self.band_count_label.setStyleSheet("font-size: 12px; font-weight: 500; color: #a1a1aa; background: transparent;")
        header_row.addWidget(self.band_count_label)

        header_row.addSpacing(8)

        # Graph curve overlay toggles
        self.undergraph_target_btn = QtWidgets.QPushButton("Target")
        self.undergraph_target_btn.setCheckable(True)
        self.undergraph_target_btn.setChecked(False)
        self.undergraph_target_btn.setFixedHeight(28)
        self.undergraph_target_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_target_btn.setToolTip("Toggle target reference curve overlay")
        self.undergraph_target_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_target_btn)

        self.undergraph_meas_btn = QtWidgets.QPushButton("Measurement")
        self.undergraph_meas_btn.setCheckable(True)
        self.undergraph_meas_btn.setChecked(False)
        self.undergraph_meas_btn.setFixedHeight(28)
        self.undergraph_meas_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_meas_btn.setToolTip("Toggle measurement curve overlay")
        self.undergraph_meas_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_meas_btn)

        self.undergraph_resultant_btn = QtWidgets.QPushButton("Resultant")
        self.undergraph_resultant_btn.setCheckable(True)
        self.undergraph_resultant_btn.setChecked(False)
        self.undergraph_resultant_btn.setFixedHeight(28)
        self.undergraph_resultant_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_resultant_btn.setToolTip("Toggle predicted resultant response overlay")
        self.undergraph_resultant_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_resultant_btn)

        self.undergraph_target2_btn = QtWidgets.QPushButton("Target 2")
        self.undergraph_target2_btn.setCheckable(True)
        self.undergraph_target2_btn.setChecked(False)
        self.undergraph_target2_btn.setFixedHeight(28)
        self.undergraph_target2_btn.setVisible(False)
        self.undergraph_target2_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_target2_btn.setToolTip("Toggle secondary target reference curve overlay")
        self.undergraph_target2_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_target2_btn)

        self.undergraph_add_target_btn = QtWidgets.QPushButton("+")
        self.undergraph_add_target_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_add_target_btn.setToolTip("Add or change secondary target curve from same provider")
        self.undergraph_add_target_btn.setFixedSize(28, 28)
        self.undergraph_add_target_btn.setStyleSheet("""
            QPushButton {
                background-color: #121214;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #a1a1aa;
                font-size: 13px;
                font-weight: 500;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #1c1c1f;
                border-color: #3f3f46;
                color: #ededed;
            }
            QPushButton:pressed {
                background-color: #27272a;
            }
        """)
        self.undergraph_add_target_btn.clicked.connect(self._on_add_target_clicked)
        header_row.addWidget(self.undergraph_add_target_btn)

        # Quiet Y-axis scale toggle synced with EQPlotWidget (±20 dB / ±12 dB)
        self.graph_scale_btn = QtWidgets.QPushButton("±20 dB")
        self.graph_scale_btn.setFixedHeight(28)
        self.graph_scale_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.graph_scale_btn.setToolTip("Toggle graph vertical scale (±20 dB / ±12 dB)")
        self.graph_scale_btn.setStyleSheet(
            f"QPushButton {{ font-family: {MONO_FONT_FAMILY}; font-size: 11px; color: #a1a1aa; "
            f"background-color: #121214; border: 1px solid #27272a; border-radius: 4px; padding: 0px 8px; }}"
            f"QPushButton:hover {{ background-color: #1c1c1f; border-color: #3f3f46; color: #ededed; }}"
        )
        self.graph_scale_btn.clicked.connect(self._toggle_graph_y_scale)
        header_row.addWidget(self.graph_scale_btn)

        self._style_undergraph_toggle_buttons()

        header_row.addStretch()

        _band_action_style = (
            "QPushButton { background-color: #121214; color: #ededed; border: 1px solid #27272a; "
            "border-radius: 4px; padding: 0px 10px; min-height: 0px; font-size: 11px; font-weight: 500; }"
            "QPushButton:hover { background-color: #1c1c1f; border-color: #3f3f46; }"
            "QPushButton:pressed { background-color: #27272a; }"
        )
        _band_primary_style = (
            "QPushButton { background-color: #ededed; color: #0b0b0c; border: 1px solid #ededed; "
            "border-radius: 4px; padding: 0px 10px; min-height: 0px; font-size: 11px; font-weight: 600; }"
            "QPushButton:hover { background-color: #ffffff; border-color: #ffffff; }"
            "QPushButton:pressed { background-color: #d4d4d8; }"
        )

        flatten_btn = QtWidgets.QPushButton("Flatten")
        flatten_btn.setFixedHeight(28)
        flatten_btn.setStyleSheet(_band_action_style)
        flatten_btn.setToolTip("Reset all band gains to 0 dB")
        flatten_btn.clicked.connect(self._flatten_all_bands)
        header_row.addWidget(flatten_btn)

        compress_btn = QtWidgets.QPushButton("Compress")
        compress_btn.setFixedHeight(28)
        compress_btn.setStyleSheet(_band_action_style)
        compress_btn.setToolTip("Scale all band gains down")
        compress_btn.clicked.connect(self._compress_all_bands)
        header_row.addWidget(compress_btn)

        expand_btn = QtWidgets.QPushButton("Expand")
        expand_btn.setFixedHeight(28)
        expand_btn.setStyleSheet(_band_action_style)
        expand_btn.setToolTip("Scale all band gains up")
        expand_btn.clicked.connect(self._expand_all_bands)
        header_row.addWidget(expand_btn)

        add_band_btn = QtWidgets.QPushButton("+ Add Band")
        add_band_btn.setObjectName("primaryBtn")
        add_band_btn.setFixedHeight(28)
        add_band_btn.setStyleSheet(_band_primary_style)
        add_band_btn.clicked.connect(self._add_band_default)
        header_row.addWidget(add_band_btn)

        reset_bands_btn = QtWidgets.QPushButton("Reset")
        reset_bands_btn.setFixedHeight(28)
        reset_bands_btn.setStyleSheet(_band_action_style)
        reset_bands_btn.clicked.connect(self._reset_all_bands)
        header_row.addWidget(reset_bands_btn)

        layout.addLayout(header_row)

        # Table Widget (Column 5 is Delete)
        self.filter_table = QtWidgets.QTableWidget()
        self.filter_table.setColumnCount(6)
        self.filter_table.setHorizontalHeaderLabels([
            "Band", "Filter type", "Frequency (Hz)", "Gain (dB)", "Q factor", "Delete"
        ])
        for col_idx in range(6):
            item = self.filter_table.horizontalHeaderItem(col_idx)
            if item:
                if col_idx in (0, 5):
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                else:
                    item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.filter_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.filter_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.filter_table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.filter_table.verticalHeader().setVisible(False)
        self.filter_table.verticalHeader().setDefaultSectionSize(32)
        self.filter_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        make_smooth_scroll(self.filter_table)

        layout.addWidget(self.filter_table)
        return frame

    def _toggle_graph_y_scale(self) -> None:
        curr = self.eq_plot.get_y_range_mode() if hasattr(self.eq_plot, "get_y_range_mode") else (20 if "20" in self.graph_scale_btn.text() else 12)
        new_mode = 12 if int(curr) == 20 else 20
        if hasattr(self.eq_plot, "set_y_range_mode"):
            self.eq_plot.set_y_range_mode(new_mode)
        self._on_plot_y_range_changed(new_mode)

    def _on_plot_y_range_changed(self, db_range: int) -> None:
        if hasattr(self, "graph_scale_btn"):
            self.graph_scale_btn.setText(f"±{int(db_range)} dB")

    # -------------------------------------------------------------------------
    # Audio Device Initialization & Dynamic Hotplug Tracking
    # -------------------------------------------------------------------------
    def _init_audio_device(self) -> None:
        active_idx = self.preset_manager.audio_device_index
        active_capture_name = getattr(self.preset_manager, "audio_device_name", None)
        self.spectrum_worker.set_device_target(active_idx, active_capture_name)

    def _on_audio_hardware_changed(self) -> None:
        """Triggered when Windows audio hardware is connected, disconnected, or default changes."""
        self._check_device_hotplug(force=False)

    def _check_device_hotplug(self, force: bool = False) -> None:
        """
        Detects audio hardware changes and default audio output switches in real time.
        Seamlessly triggers spectrum loopback reconnection so audio visualization never drops.
        """
        default_changed = False
        endpoints_changed = False
        new_def_id = None
        new_devs: List[str] = []

        if self._media_devices:
            try:
                def_out = self._media_devices.defaultAudioOutput()
                if def_out:
                    new_def_id = bytes(def_out.id())
                new_devs = [d.description() for d in self._media_devices.audioOutputs()]

                if new_def_id != self._last_default_output_id:
                    default_changed = True
                    def_name = def_out.description() if def_out else "Unknown"
                    print(f"[AudioDevices] Windows Default Output switched to: {def_name}")
                    self._last_default_output_id = new_def_id

                if set(new_devs) != set(self._last_output_devices_snapshot):
                    endpoints_changed = True
                    print(f"[AudioDevices] Audio endpoints updated: {new_devs}")
                    self._last_output_devices_snapshot = new_devs
            except Exception as e:
                print(f"[AudioDevices] Hotplug check error: {e}")

        if not (force or default_changed or endpoints_changed):
            return

        if hasattr(self, "spectrum_toggle_btn") and not self.spectrum_toggle_btn.isChecked():
            return

        current_idx = self.preset_manager.audio_device_index
        current_name = getattr(self.preset_manager, "audio_device_name", None)
        is_auto_follow = (current_idx is None and not current_name) or (
            bool(current_name) and ("auto-follow" in current_name.lower() or "default" in current_name.lower())
        )

        if is_auto_follow:
            if force or default_changed or endpoints_changed:
                print("[AudioDevices] Auto-Follow active: seamlessly updating spectrum loopback to new default device.")
                self.spectrum_worker.restart_stream()
        elif endpoints_changed:
            self.spectrum_worker.restart_stream()

    def _collect_current_eq_state(self) -> Tuple[float, List[EQBand]]:
        return self.preamp_spin.value(), list(self.eq_plot.bands)

    def _launch_apo_configurator(self) -> None:
        ok, msg = launch_apo_configurator()
        if not ok:
            QtWidgets.QMessageBox.warning(
                self,
                "Equalizer APO",
                f"Could not launch Equalizer APO Device Selector / Configurator:\n{msg}\n\nPlease verify Equalizer APO is installed.",
            )

    def _check_equalizer_apo_on_launch(self) -> None:
        if not is_equalizer_apo_installed():
            if not self.preset_manager.get_setting("ignore_missing_apo", False):
                self._prompt_install_equalizer_apo()
            return
        active_name = self.preset_manager.active_device_name
        if active_name and clean_apo_device_name(active_name).lower() not in ("all", "all endpoints"):
            self._check_device_apo_status(active_name, user_initiated=False)

    def _prompt_install_equalizer_apo(self) -> None:
        dlg = EqualizerApoInstallDialog(self)
        dlg.sigInstalled.connect(self._on_apo_installed_success)
        dlg.exec()
        dont_ask = dlg.chk_dont_ask.isChecked()
        dlg.deleteLater()
        if dont_ask:
            self.preset_manager.set_setting("ignore_missing_apo", True)

    def _on_apo_installed_success(self) -> None:
        self.preset_manager.apo_target_path = DEFAULT_APO_PATH
        self.preset_manager.active_apo_path = DEFAULT_APO_PATH
        ensure_apo_root_config_include(DEFAULT_APO_PATH)
        self.preset_manager.save_settings()
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
        )
        self._refresh_apo_health_indicator()

    def _check_device_apo_status(self, dev_name: str, user_initiated: bool = False) -> None:
        if not dev_name:
            return
        low = dev_name.strip().lower()
        if "auto-follow" in low or low == "all endpoints" or clean_apo_device_name(dev_name).lower() == "all":
            return
        if not is_equalizer_apo_installed():
            if user_initiated or not self.preset_manager.get_setting("ignore_missing_apo", False):
                self._prompt_install_equalizer_apo()
            return
        cfg = get_apo_configurator_path()
        if not cfg:
            return
        if not self._is_device_installed_in_apo(dev_name):
            dlg = ApoDeviceConfigDialog(dev_name, cfg, self)
            dlg.exec()
            dlg.deleteLater()

    def _is_device_installed_in_apo(self, dev_name: str) -> bool:
        if not dev_name:
            return True
        low = dev_name.strip().lower()
        if "auto-follow" in low or low == "all endpoints":
            return True
        clean_name = clean_apo_device_name(dev_name).lower()
        if clean_name in ("all", "all endpoints", "") or len(clean_name) < 2:
            return True
        try:
            import winreg
            child_apos = set()
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\EqualizerAPO\Child APOs") as k:
                    for i in range(winreg.QueryInfoKey(k)[0]):
                        child_apos.add(winreg.EnumKey(k, i).lower())
            except Exception:
                return True

            # Only compare against Windows MMDevice endpoint friendly-name / description properties
            name_prop_keys = {
                "{a45c254e-df1c-4efd-8020-67d146a850e0},2",   # PKEY_Device_DeviceDesc
                "{a45c254e-df1c-4efd-8020-67d146a850e0},14",  # PKEY_Device_FriendlyName
                "{b3f8fa53-0004-438e-9003-51a46e139bfc},6",   # PKEY_DeviceInterface_FriendlyName
            }
            render_base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
            matched_any = False
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, render_base) as rk:
                for i in range(winreg.QueryInfoKey(rk)[0]):
                    sub = winreg.EnumKey(rk, i)
                    dev_key = f"{render_base}\\{sub}"
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, dev_key) as dk:
                            state, _ = winreg.QueryValueEx(dk, "DeviceState")
                        if state != 1:  # Only evaluate active endpoints
                            continue
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{dev_key}\\Properties") as pk:
                            for pi in range(winreg.QueryInfoKey(pk)[1]):
                                vn, vd, _ = winreg.EnumValue(pk, pi)
                                if vn.lower() not in name_prop_keys or not isinstance(vd, str):
                                    continue
                                vd_clean = clean_apo_device_name(vd).lower()
                                if len(vd_clean) < 3 or vd_clean == "all":
                                    continue
                                if clean_name == vd_clean or clean_name in vd_clean or vd_clean in clean_name:
                                    matched_any = True
                                    if sub.lower() in child_apos:
                                        return True
                    except Exception:
                        pass
            if matched_any:
                return False
        except Exception:
            pass
        return True

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.preamp_spin.valueChanged.connect(self._on_preamp_spin_changed)
        self.preamp_slider.valueChanged.connect(self._on_preamp_slider_changed)
        self.preamp_slider.sliderPressed.connect(self._on_preamp_slider_pressed)
        self.preamp_slider.sliderReleased.connect(self._on_preamp_slider_released)

        self.preset_manager.sigPresetChanged.connect(self._on_preset_loaded)
        self.preset_manager.sigPresetsListChanged.connect(self._refresh_presets_list)

        self.eq_plot.sigBandModified.connect(self._on_plot_band_modified)
        self.eq_plot.sigBandAdded.connect(self._on_plot_band_added)
        self.eq_plot.sigBandDeleted.connect(self._delete_band)
        self.eq_plot.sigBandDragStarted.connect(self._on_plot_drag_started)

        self.spectrum_worker.sigSpectrumReady.connect(self.eq_plot.update_spectrum_curve)
        if hasattr(self.spectrum_worker, "sigPeakHoldReady"):
            self.spectrum_worker.sigPeakHoldReady.connect(self._on_peak_hold_ready)
        self.eq_plot.sigSpectrumToggled.connect(self._on_plot_spectrum_toggled)
        self.eq_plot.sigLegendCurveClicked.connect(self._on_graph_legend_clicked)
        self.eq_plot.sigWheelScrollStarted.connect(self._notify_wheel_scroll)
        if hasattr(self.eq_plot, "sigYRangeChanged"):
            self.eq_plot.sigYRangeChanged.connect(self._on_plot_y_range_changed)

    def _is_eq_wheel_control(self, widget: QtCore.QObject) -> bool:
        if widget is self.preamp_spin or widget is self.preamp_slider:
            return True
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            parent = widget.parent()
            while parent is not None:
                if parent is self.filter_table:
                    return True
                parent = parent.parent()
        return False

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Wheel:
            if isinstance(watched, (QtWidgets.QAbstractSpinBox, QtWidgets.QSlider)):
                if self._is_eq_wheel_control(watched):
                    self._notify_wheel_scroll()
        return super().eventFilter(watched, event)

    def _notify_wheel_scroll(self) -> None:
        if not self._is_wheel_scrolling:
            self._is_wheel_scrolling = True
            self._record_undo_snapshot()
            self._has_active_scroll_snapshot = True
        self._wheel_scroll_timer.start(450)

    def _on_wheel_scroll_timeout(self) -> None:
        self._is_wheel_scrolling = False
        self._has_active_scroll_snapshot = False
        self._commit_state()

    # -------------------------------------------------------------------------
    # Universal EQ Toggle
    # -------------------------------------------------------------------------
    def _set_universal_eq_bypassed(self, bypassed: bool, export: bool = True) -> None:
        self.is_eq_bypassed = bool(bypassed)
        if self.is_eq_bypassed:
            self.eq_universal_toggle.setText("EQ Off")
            self.eq_universal_toggle.setObjectName("bypassedEqBtn")
            self.eq_universal_toggle.set_status_dot("#71717a", filled=False)
        else:
            self.eq_universal_toggle.setText("EQ On")
            self.eq_universal_toggle.setObjectName("activeEqBtn")
            self.eq_universal_toggle.set_status_dot("#4cc38a", filled=True)

        self.eq_universal_toggle.setStyle(self.eq_universal_toggle.style())
        self.eq_plot.set_eq_bypassed(self.is_eq_bypassed)
        if export:
            self.preset_manager.schedule_apo_export(
                self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
            )

    def _toggle_universal_eq(self) -> None:
        self._set_universal_eq_bypassed(not self.is_eq_bypassed, export=True)

    # -------------------------------------------------------------------------
    # State Sanitization (Consistent bounds across Table, Plot, and APO)
    # -------------------------------------------------------------------------
    @staticmethod
    def _sanitize_eq_state(preamp_db: float, bands: List[EQBand]) -> Tuple[float, List[EQBand]]:
        try:
            p = float(preamp_db)
            if not np.isfinite(p):
                p = 0.0
        except Exception:
            p = 0.0
        p = max(-30.0, min(15.0, p))

        sanitized: List[EQBand] = []
        used_ids = set()
        next_id = 1
        for raw_b in bands:
            try:
                bid = int(raw_b.band_id)
            except Exception:
                bid = 0
            if bid <= 0 or bid in used_ids:
                while next_id in used_ids:
                    next_id += 1
                bid = next_id
            used_ids.add(bid)

            ftype = FilterType.from_str(getattr(raw_b, "filter_type", "Peak"))
            try:
                freq = float(raw_b.freq)
                if not np.isfinite(freq):
                    freq = 1000.0
            except Exception:
                freq = 1000.0
            freq = max(20.0, min(20000.0, freq))

            if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                gain = 0.0
            else:
                try:
                    gain = float(raw_b.gain)
                    if not np.isfinite(gain):
                        gain = 0.0
                except Exception:
                    gain = 0.0
                gain = max(-20.0, min(20.0, gain))

            try:
                q = float(raw_b.q)
                if not np.isfinite(q):
                    q = 1.414
            except Exception:
                q = 1.414
            q = max(0.1, min(20.0, q))

            sanitized.append(
                EQBand(
                    band_id=bid,
                    enabled=bool(getattr(raw_b, "enabled", True)),
                    filter_type=ftype.value,
                    freq=freq,
                    gain=gain,
                    q=q,
                    channel=str(getattr(raw_b, "channel", "ALL") or "ALL"),
                )
            )
        return p, sanitized

    # -------------------------------------------------------------------------
    # Parameter Undo / Redo Engine (Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z)
    # -------------------------------------------------------------------------
    def _get_current_state(self) -> Tuple[float, List[Dict[str, Any]]]:
        return (
            round(self.preamp_spin.value(), 2),
            [b.to_dict() for b in self.eq_plot.bands]
        )

    def _record_undo_snapshot(self) -> None:
        if self._is_undoing_redoing:
            return
        if self._is_wheel_scrolling and self._has_active_scroll_snapshot:
            return
        state = self._last_state if self._last_state is not None else self._get_current_state()
        if not self._undo_stack or self._undo_stack[-1] != state:
            self._undo_stack.append(state)
            self._redo_stack.clear()

    def _commit_state(self) -> None:
        if not self._is_undoing_redoing:
            self._last_state = self._get_current_state()

    def _on_plot_drag_started(self, band_id: int) -> None:
        self._record_undo_snapshot()

    def _undo(self) -> None:
        if self._wheel_scroll_timer.isActive():
            self._wheel_scroll_timer.stop()
        self._is_wheel_scrolling = False
        self._has_active_scroll_snapshot = False
        if self._preset_save_timer.isActive():
            self._flush_preset_save()

        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return
        if isinstance(focused, QtWidgets.QLineEdit) and not isinstance(focused.parent(), QtWidgets.QAbstractSpinBox):
            return

        if not self._undo_stack:
            return

        current_state = self._get_current_state()
        prev_state = self._undo_stack.pop()

        self._redo_stack.append(current_state)
        self._apply_state_snapshot(prev_state)
        self._last_state = prev_state

    def _redo(self) -> None:
        if self._wheel_scroll_timer.isActive():
            self._wheel_scroll_timer.stop()
        self._is_wheel_scrolling = False
        self._has_active_scroll_snapshot = False
        if self._preset_save_timer.isActive():
            self._flush_preset_save()

        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
            return
        if isinstance(focused, QtWidgets.QLineEdit) and not isinstance(focused.parent(), QtWidgets.QAbstractSpinBox):
            return

        if not self._redo_stack:
            return

        current_state = self._get_current_state()
        next_state = self._redo_stack.pop()

        self._undo_stack.append(current_state)
        self._apply_state_snapshot(next_state)
        self._last_state = next_state

    def _apply_state_snapshot(self, snapshot: Tuple[float, List[Dict[str, Any]]]) -> None:
        preamp_val, bands_dict = snapshot
        if self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()
        self._is_undoing_redoing = True
        try:
            raw_bands = [EQBand.from_dict(d) for d in bands_dict]
            preamp_val, restored_bands = self._sanitize_eq_state(preamp_val, raw_bands)
            self.preamp_spin.blockSignals(True)
            self.preamp_slider.blockSignals(True)
            self.preamp_spin.setValue(preamp_val)
            self.preamp_slider.setValue(int(round(preamp_val * 10)))
            self.preamp_spin.blockSignals(False)
            self.preamp_slider.blockSignals(False)

            self.eq_plot.set_bands(restored_bands, preamp_db=preamp_val)
            self._populate_table(restored_bands)
            self.preset_manager.update_active_preset_content(preamp_val, restored_bands)
            self.preset_manager.schedule_apo_export(preamp_val, restored_bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge(preamp_val)
        finally:
            self._is_undoing_redoing = False

    # -------------------------------------------------------------------------
    # Preset Synchronization & Smooth Animations
    # -------------------------------------------------------------------------
    def _load_initial_preset(self) -> None:
        self._refresh_presets_list()
        active = self.preset_manager.active_preset_name
        self._on_preset_loaded(active, animate=False)

    def _refresh_presets_list(self) -> None:
        self.preset_list_widget.blockSignals(True)
        self.preset_list_widget.clear()
        names = self.preset_manager.get_preset_names()
        active = self.preset_manager.active_preset_name
        
        for name in names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            meta = self.preset_manager.get_preset_metadata(name)
            if meta and (meta.get("model_name") or meta.get("target_name") or meta.get("is_autoeq")):
                m_info = meta.get("model_name") or "Measurement"
                t_info = meta.get("target_name") or "Target"
                db_info = meta.get("database_name") or meta.get("database_id") or "Reviewer"
                item.setToolTip(f"{name}\nReference: {m_info}\nTarget: {t_info} ({db_info})")
            else:
                item.setToolTip(f"{name}\nNo reference curves attached")
            self.preset_list_widget.addItem(item)
            if name == active:
                self.preset_list_widget.setCurrentItem(item)

        self.preset_list_widget.blockSignals(False)
        self.preset_list_widget.animate_to_current()
        self._update_curves_button_state()

    def _filter_presets_list(self, text: str) -> None:
        text = text.lower().strip()
        for i in range(self.preset_list_widget.count()):
            item = self.preset_list_widget.item(i)
            item.setHidden(bool(text) and (text not in item.text().lower()))
        self.preset_list_widget.doItemsLayout()
        self.preset_list_widget.animate_to_current(immediate=True)

    def _on_preset_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self.preset_list_widget.animate_to_current()
        if item.text() == self.preset_manager.active_preset_name:
            return
        if self._preset_save_timer.isActive():
            self._flush_preset_save()
        self.preset_manager.set_active_preset(item.text())

    def _on_preset_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Double clicking profile prompts for inline renaming (no modal window)."""
        self.preset_list_widget.editItem(item)

    def _rename_preset_inline_shortcut(self) -> None:
        """F2 shortcut: only triggers when focus is not inside a text editor or spinbox."""
        focused = QtWidgets.QApplication.focusWidget()
        if isinstance(focused, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit, QtWidgets.QAbstractSpinBox)):
            return
        if hasattr(self, "sidebar_tab_bar") and self.sidebar_tab_bar.current_index() != 0:
            return
        self._rename_preset_inline()

    def _rename_preset_inline(self, target_item: Optional[QtWidgets.QListWidgetItem] = None) -> None:
        """F2, Rename button, or context menu triggers inline renaming without opening any new window."""
        item = target_item if isinstance(target_item, QtWidgets.QListWidgetItem) else self.preset_list_widget.currentItem()
        if item:
            self.preset_list_widget.editItem(item)

    def _on_preset_item_renamed(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handles completion of inline renaming safely without stale C++ pointer crashes."""
        if getattr(self, "_is_renaming_preset", False):
            return
        self._is_renaming_preset = True
        try:
            try:
                old_name = item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                new_name = item.text().strip()
            except RuntimeError:
                return

            if not new_name or new_name == old_name:
                self.preset_list_widget.blockSignals(True)
                try:
                    item.setText(old_name)
                except RuntimeError:
                    pass
                finally:
                    self.preset_list_widget.blockSignals(False)
                return

            if new_name in self.preset_manager.presets and new_name != old_name:
                self.preset_list_widget.blockSignals(True)
                try:
                    item.setText(old_name)
                except RuntimeError:
                    pass
                finally:
                    self.preset_list_widget.blockSignals(False)
                QtWidgets.QMessageBox.warning(self, "Profile Exists", f"A profile named '{new_name}' already exists.")
                return

            # Suppress redundant full preset reload & undo wipe on a pure rename
            self._suppress_preset_reload = True
            try:
                self.preset_manager.rename_preset(old_name, new_name)
            finally:
                self._suppress_preset_reload = False
        except RuntimeError:
            pass
        finally:
            self._is_renaming_preset = False

    def _show_preset_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.preset_list_widget.itemAt(pos)
        if not item:
            return
        name = item.text()
        has_curves = self.preset_manager.has_preset_metadata(name)
        menu = QtWidgets.QMenu(self)

        act_curves = menu.addAction("Edit target & measurement..." if has_curves else "Add target & measurement...")
        act_curves.triggered.connect(lambda _=False, n=name: self._open_assign_curves_dialog(n))

        if has_curves:
            act_remove = menu.addAction("Remove target & measurement")
            act_remove.triggered.connect(lambda _=False, n=name: self._remove_preset_curves(n))

        menu.addSeparator()

        act_export = menu.addAction("Export profile...")
        act_export.triggered.connect(lambda _=False, n=name: self._export_eq_dialog(n))

        act_copy = menu.addAction("Copy APO")
        act_copy.triggered.connect(lambda _=False, n=name: self._copy_preset_apo(n))

        act_paste = menu.addAction("Paste APO")
        act_paste.triggered.connect(self._paste_apo_from_clipboard)

        menu.addSeparator()

        act_ren = menu.addAction("Rename (F2)")
        act_ren.triggered.connect(lambda _=False, it=item: self._rename_preset_inline(it))

        act_dup = menu.addAction("Duplicate")
        act_dup.triggered.connect(lambda _=False, n=name: self._duplicate_preset(n))

        act_del = menu.addAction("Delete")
        act_del.triggered.connect(lambda _=False, n=name: self._delete_preset(n))

        menu.exec(self.preset_list_widget.mapToGlobal(pos))

    def _open_assign_curves_dialog(self, preset_name: Optional[str] = None) -> None:
        target_preset = preset_name or self.preset_manager.active_preset_name
        dlg = ProfileCurveAssignmentDialog(
            client=self.squig_client,
            preset_manager=self.preset_manager,
            preset_name=target_preset,
            parent=self,
        )
        accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        dlg.deleteLater()
        if accepted:
            if target_preset == self.preset_manager.active_preset_name:
                self._on_preset_loaded(target_preset, animate=False)
            self._refresh_presets_list()
            self._update_curves_button_state()

    def _remove_preset_curves(self, preset_name: str) -> None:
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Curves",
            f"Are you sure you want to remove target and measurement reference curves from '{preset_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self.preset_manager.set_preset_metadata(preset_name, None)
            if preset_name == self.preset_manager.active_preset_name:
                self._clear_autoeq_overlays()
            self._refresh_presets_list()
            self._update_curves_button_state()

    def _update_curves_button_state(self) -> None:
        pass

    @staticmethod
    def _clean_curve_arrays(
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if freqs is None or spls is None or len(freqs) == 0 or len(spls) == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        if len(freqs) != len(spls):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        f_arr = np.asarray(freqs, dtype=np.float64)
        s_arr = np.asarray(spls, dtype=np.float64)
        valid = np.isfinite(f_arr) & np.isfinite(s_arr) & (f_arr > 0.0)
        if not np.any(valid):
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        f_arr = f_arr[valid]
        s_arr = s_arr[valid]
        if len(f_arr) > 1 and np.any(np.diff(f_arr) < 0):
            order = np.argsort(f_arr)
            f_arr = f_arr[order]
            s_arr = s_arr[order]
        return f_arr, s_arr

    def _compute_normalized_display_curves(
        self,
        mf: Optional[Sequence[float]],
        ms: Optional[Sequence[float]],
        tf: Optional[Sequence[float]],
        ts: Optional[Sequence[float]],
        bands: List[EQBand],
    ) -> Tuple[
        Tuple[List[float], List[float]],
        Tuple[List[float], List[float]],
        Tuple[List[float], List[float]],
    ]:
        mf_arr, ms_arr = self._clean_curve_arrays(mf, ms)
        tf_arr, ts_arr = self._clean_curve_arrays(tf, ts)

        if len(mf_arr) > 0 and len(tf_arr) > 0:
            mid_mask = (tf_arr >= 200.0) & (tf_arr <= 2000.0)
            target_norm = float(np.mean(ts_arr[mid_mask])) if np.any(mid_mask) else float(np.mean(ts_arr))

            meas_interp = np.interp(tf_arr, mf_arr, ms_arr)
            offset = float(np.mean(ts_arr[mid_mask] - meas_interp[mid_mask])) if np.any(mid_mask) else float(np.mean(ts_arr - meas_interp))
            meas_aligned = ms_arr + offset

            disp_target = ts_arr - target_norm
            disp_meas = meas_aligned - target_norm

            eq_response = compute_composite_response(
                bands, mf_arr, preamp_db=0.0, fs=self.sample_rate
            )
            disp_res = disp_meas + eq_response

            return (
                (mf_arr.tolist(), disp_meas.tolist()),
                (tf_arr.tolist(), disp_target.tolist()),
                (mf_arr.tolist(), disp_res.tolist()),
            )
        elif len(mf_arr) > 0:
            m_norm = float(np.mean(ms_arr))
            disp_meas = ms_arr - m_norm
            eq_response = compute_composite_response(
                bands, mf_arr, preamp_db=0.0, fs=self.sample_rate
            )
            disp_res = disp_meas + eq_response
            return (
                (mf_arr.tolist(), disp_meas.tolist()),
                ([], []),
                (mf_arr.tolist(), disp_res.tolist()),
            )
        elif len(tf_arr) > 0:
            t_norm = float(np.mean(ts_arr))
            disp_target = ts_arr - t_norm
            return (
                ([], []),
                (tf_arr.tolist(), disp_target.tolist()),
                ([], []),
            )
        return (([], []), ([], []), ([], []))

    def _apply_target2_curve_data(
        self,
        t2_f: Optional[Sequence[float]],
        t2_s: Optional[Sequence[float]],
        label: str = "Target 2",
    ) -> None:
        t2_f_arr, t2_s_arr = self._clean_curve_arrays(t2_f, t2_s)
        if len(t2_f_arr) > 0:
            mid_mask2 = (t2_f_arr >= 200.0) & (t2_f_arr <= 2000.0)
            target2_norm = float(np.mean(t2_s_arr[mid_mask2])) if np.any(mid_mask2) else float(np.mean(t2_s_arr))
            disp_target2 = t2_s_arr - target2_norm
            self.eq_plot.set_target2_overlay(t2_f_arr.tolist(), disp_target2.tolist(), label=label or "Target 2")
            self.undergraph_target2_btn.setVisible(True)
            self.undergraph_target2_btn.blockSignals(True)
            self.undergraph_target2_btn.setChecked(True)
            self.undergraph_target2_btn.blockSignals(False)
            self._style_undergraph_toggle_buttons()
            self.eq_plot.set_overlays_visible(
                self.undergraph_meas_btn.isChecked(),
                self.undergraph_target_btn.isChecked(),
                self.undergraph_resultant_btn.isChecked(),
                show_target2=True,
            )
        else:
            self.eq_plot.set_target2_overlay(None, None)
            self.undergraph_target2_btn.blockSignals(True)
            self.undergraph_target2_btn.setChecked(False)
            self.undergraph_target2_btn.setVisible(False)
            self.undergraph_target2_btn.blockSignals(False)
            self._style_undergraph_toggle_buttons()
            self.eq_plot.set_overlays_visible(
                self.undergraph_meas_btn.isChecked(),
                self.undergraph_target_btn.isChecked(),
                self.undergraph_resultant_btn.isChecked(),
                show_target2=False,
            )

    def _async_fetch_preset_reference_curves(self, preset_name: str, autoeq_meta: Dict[str, Any], bands: List[EQBand]) -> None:
        """Fetches missing preset reference curves asynchronously to avoid blocking the UI thread."""
        m_file = autoeq_meta.get("model_file", "")
        t_file = autoeq_meta.get("target_file", "")
        t_name = autoeq_meta.get("target_name", "")
        t2_file = autoeq_meta.get("target2_file", "")
        t2_name = autoeq_meta.get("target2_name", "")
        chan = autoeq_meta.get("channel_mode", "avg")
        m_name = autoeq_meta.get("model_name", "")
        t_mode = autoeq_meta.get("target_mode", "target_curve")
        base_u = autoeq_meta.get("base_url", "")

        model = (
            SquigModel(
                brand="",
                name=m_name or m_file,
                full_name=m_name or m_file,
                file_name=m_file,
                display_name=m_name or m_file,
            )
            if m_file
            else None
        )
        target_mode_int = 0
        target = None
        target_iem = None
        target_iem_chan = "avg"
        target_profile_name = ""

        if t_mode == "iem_tuning" or autoeq_meta.get("target_iem_file"):
            target_mode_int = 1
            t_iem_file = autoeq_meta.get("target_iem_file", "")
            t_iem_name = autoeq_meta.get("target_iem_name", "")
            target_iem_chan = autoeq_meta.get("target_iem_channel", "avg")
            t_iem_base = autoeq_meta.get("target_iem_base_url") or base_u
            target_iem = (
                SquigModel(
                    brand="",
                    name=t_iem_name or t_iem_file,
                    full_name=t_iem_name or t_iem_file,
                    file_name=t_iem_file,
                    display_name=t_iem_name or t_iem_file,
                )
                if t_iem_file
                else None
            )
            if target_iem:
                target_iem.base_url = t_iem_base
            t_name = f"{t_iem_name or t_iem_file} (Target)"
        elif t_mode == "resultant_profile" or autoeq_meta.get("target_profile_name"):
            target_mode_int = 2
            target_profile_name = autoeq_meta.get("target_profile_name", "")
            t_name = f"{target_profile_name} (Resultant)"
        else:
            target_mode_int = 0
            if t_file:
                target = SquigTarget(name=t_name or t_file, file_name=t_file)

        target2 = SquigTarget(name=t2_name or t2_file, file_name=t2_file) if t2_file else None

        self._curve_req_id += 1
        req_id = self._curve_req_id
        self._retire_worker(self._curve_worker)
        self._curve_worker = None

        worker = CurveFetchWorker(
            client=self.squig_client,
            base_url=base_u,
            model=model,
            target_mode=target_mode_int,
            target=target,
            target_iem_model=target_iem,
            target_iem_chan=target_iem_chan,
            target_profile_name=target_profile_name,
            preset_manager=self.preset_manager,
            sample_rate=self.sample_rate,
            channel_mode=chan,
            target2=target2,
            parent=self,
        )

        def _on_loaded(meas_data: Tuple, target_data: Tuple, rid: int = req_id) -> None:
            if rid != self._curve_req_id or self.preset_manager.active_preset_name != preset_name:
                return
            mf, ms = meas_data
            tf, ts = target_data
            if mf or tf:
                self._autoeq_meas_raw = (mf, ms)
                self._autoeq_target_raw = (tf, ts)
                dm, dt, dr = self._compute_normalized_display_curves(mf, ms, tf, ts, self.eq_plot.bands)
                has_dm = bool(dm[0])
                has_dt = bool(dt[0])
                has_dr = bool(dr[0])
                if has_dm:
                    self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
                    self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)
                    self.eq_plot.set_measurement_overlay(dm[0], dm[1], label=m_name or "Measurement")
                else:
                    self._disp_meas_freqs = None
                    self._disp_meas_spls = None
                    self.eq_plot.set_measurement_overlay(None, None)
                if has_dt:
                    self.eq_plot.set_target_overlay(dt[0], dt[1], label=t_name or "Target")
                else:
                    self.eq_plot.set_target_overlay(None, None)
                if has_dr:
                    self.eq_plot.set_resultant_overlay(dr[0], dr[1], label=f"{m_name or 'Resultant'} + EQ")
                else:
                    self.eq_plot.set_resultant_overlay(None, None)

                self.undergraph_meas_btn.blockSignals(True)
                self.undergraph_target_btn.blockSignals(True)
                self.undergraph_resultant_btn.blockSignals(True)
                self.undergraph_meas_btn.setChecked(has_dm)
                self.undergraph_target_btn.setChecked(has_dt)
                self.undergraph_resultant_btn.setChecked(has_dr)
                self.undergraph_meas_btn.blockSignals(False)
                self.undergraph_target_btn.blockSignals(False)
                self.undergraph_resultant_btn.blockSignals(False)
                self._style_undergraph_toggle_buttons()
                self.eq_plot.set_overlays_visible(
                    has_dm, has_dt, has_dr,
                    show_target2=self.undergraph_target2_btn.isChecked(),
                )

        def _on_t2_loaded(t2_data: Tuple, rid: int = req_id) -> None:
            if rid != self._curve_req_id or self.preset_manager.active_preset_name != preset_name:
                return
            t2f, t2s = t2_data
            if t2f and t2s:
                self._apply_target2_curve_data(t2f, t2s, label=t2_name or "Target 2")

        worker.sigCurvesLoaded.connect(_on_loaded)
        if target2 is not None:
            worker.sigTarget2Loaded.connect(_on_t2_loaded)
        self._curve_worker = worker
        worker.start()

    def _on_preset_loaded(self, preset_name: str, animate: bool = True) -> None:
        if getattr(self, "_suppress_preset_reload", False):
            return

        if self._preset_save_timer.isActive():
            self._preset_save_timer.stop()

        # Cancel pending AutoEQ restore / DB switch / in-flight curve fetch from previous preset
        if self._db_switch_timer.isActive():
            self._db_switch_timer.stop()
        self._pending_restore_model_file = None
        self._pending_restore_target_file = None
        self._pending_restore_skip_fetch = False
        self._pending_global_model = None
        self._suppress_curve_fetch = False

        self._curve_req_id += 1
        self._retire_worker(self._curve_worker)
        self._curve_worker = None
        self._target2_req_id += 1
        self._retire_worker(self._target2_worker)
        self._target2_worker = None

        self._undo_stack.clear()
        self._redo_stack.clear()
        for i in range(self.preset_list_widget.count()):
            item = self.preset_list_widget.item(i)
            if item.text() == preset_name:
                self.preset_list_widget.setCurrentItem(item)
                self.preset_list_widget.animate_to_current()
                break

        raw_preamp, raw_bands = self.preset_manager.get_active_preset_data()
        preamp, bands = self._sanitize_eq_state(raw_preamp, raw_bands)

        # Check if preset was generated with AutoEQ metadata or has reference curves attached
        autoeq_meta = self.preset_manager.get_preset_metadata(preset_name)
        new_meas_data: Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str] = (None, None, "")
        new_target_data: Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str] = (None, None, "")
        new_res_data: Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str] = (None, None, "")

        has_curves = bool(
            autoeq_meta and (
                autoeq_meta.get("is_autoeq") or
                autoeq_meta.get("model_file") or
                autoeq_meta.get("target_file") or
                autoeq_meta.get("target_iem_file") or
                autoeq_meta.get("target_profile_name")
            )
        )

        if has_curves:
            base_url_meta = autoeq_meta.get("base_url", "")
            db_id = autoeq_meta.get("database_id", "") or _resolve_db_id(self.squig_client, base_url_meta)
            m_file = autoeq_meta.get("model_file", "")
            t_file = autoeq_meta.get("target_file", "")
            t_name = autoeq_meta.get("target_name", "")
            t2_file = autoeq_meta.get("target2_file", "")
            t2_name = autoeq_meta.get("target2_name", "")
            chan = autoeq_meta.get("channel_mode", "avg")
            m_name = autoeq_meta.get("model_name", "")
            t_mode = autoeq_meta.get("target_mode", "target_curve")

            # Query offline database synchronously (ultra-fast SQLite lookup)
            off_db = get_offline_db()
            m_res = off_db.get_measurement(db_id, m_file, chan) if m_file else None
            needs_network_fetch = False
            if not m_res and m_file and base_url_meta:
                needs_network_fetch = True

            t_res = None
            if t_mode == "iem_tuning" or autoeq_meta.get("target_iem_file"):
                t_iem_file = autoeq_meta.get("target_iem_file", "")
                t_iem_name = autoeq_meta.get("target_iem_name", "")
                t_iem_chan = autoeq_meta.get("target_iem_channel", "avg")
                t_iem_base = autoeq_meta.get("target_iem_base_url") or base_url_meta
                t_iem_db_id = autoeq_meta.get("target_iem_database_id") or (
                    _resolve_db_id(self.squig_client, t_iem_base, db_id) if t_iem_base else db_id
                )
                t_res = off_db.get_measurement(t_iem_db_id, t_iem_file, t_iem_chan) if t_iem_file else None
                if not t_res and t_iem_file and t_iem_base:
                    needs_network_fetch = True
                t_name = f"{t_iem_name or t_iem_file} (Target)"
            elif t_mode == "resultant_profile" or autoeq_meta.get("target_profile_name"):
                target_p_name = autoeq_meta.get("target_profile_name", "")
                t_res, ref_needs_net = self._compute_preset_resultant_curve(target_p_name, allow_network=False)
                if ref_needs_net:
                    needs_network_fetch = True
                t_name = f"{target_p_name} (Resultant)"
            else:
                t_res = off_db.get_target(db_id, t_file, t_name) if t_file else None
                if not t_res and t_file and base_url_meta:
                    needs_network_fetch = True

            t2_res = off_db.get_target(db_id, t2_file, t2_name) if t2_file else None
            if not t2_res and t2_file and base_url_meta:
                needs_network_fetch = True

            if needs_network_fetch:
                self._async_fetch_preset_reference_curves(preset_name, autoeq_meta, bands)

            if m_res or t_res:
                mf, ms = m_res if m_res else ([], [])
                tf, ts = t_res if t_res else ([], [])
                self._autoeq_meas_raw = (mf, ms)
                self._autoeq_target_raw = (tf, ts)

                dm, dt, dr = self._compute_normalized_display_curves(mf, ms, tf, ts, bands)

                if dm[0]:
                    self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
                    self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)
                    new_meas_data = (dm[0], dm[1], m_name or "Measurement")
                else:
                    self._disp_meas_freqs = None
                    self._disp_meas_spls = None

                if dt[0]:
                    new_target_data = (dt[0], dt[1], t_name or "Target")

                if dr[0]:
                    new_res_data = (dr[0], dr[1], f"{m_name or 'Resultant'} + EQ")

                # Handle secondary target curve (target 2)
                if t2_res and len(t2_res[0]) > 0 and len(t2_res[0]) == len(t2_res[1]):
                    t2_f_arr, t2_s_arr = self._clean_curve_arrays(t2_res[0], t2_res[1])
                    if len(t2_f_arr) > 0:
                        mid_mask2 = (t2_f_arr >= 200.0) & (t2_f_arr <= 2000.0)
                        target2_norm = float(np.mean(t2_s_arr[mid_mask2])) if np.any(mid_mask2) else float(np.mean(t2_s_arr))
                        disp_target2 = t2_s_arr - target2_norm
                        self.eq_plot.set_target2_overlay(t2_f_arr.tolist(), disp_target2.tolist(), label=t2_name or "Target 2")
                        self.undergraph_target2_btn.setVisible(True)
                        self.undergraph_target2_btn.blockSignals(True)
                        self.undergraph_target2_btn.setChecked(True)
                        self.undergraph_target2_btn.blockSignals(False)
                    else:
                        self.eq_plot.set_target2_overlay(None, None)
                        self.undergraph_target2_btn.blockSignals(True)
                        self.undergraph_target2_btn.setChecked(False)
                        self.undergraph_target2_btn.setVisible(False)
                        self.undergraph_target2_btn.blockSignals(False)
                else:
                    self.eq_plot.set_target2_overlay(None, None)
                    self.undergraph_target2_btn.blockSignals(True)
                    self.undergraph_target2_btn.setChecked(False)
                    self.undergraph_target2_btn.setVisible(False)
                    self.undergraph_target2_btn.blockSignals(False)

                self.undergraph_meas_btn.blockSignals(True)
                self.undergraph_target_btn.blockSignals(True)
                self.undergraph_resultant_btn.blockSignals(True)
                self.undergraph_meas_btn.setChecked(bool(dm[0]))
                self.undergraph_target_btn.setChecked(bool(dt[0]))
                self.undergraph_resultant_btn.setChecked(bool(dr[0]))
                self.undergraph_meas_btn.blockSignals(False)
                self.undergraph_target_btn.blockSignals(False)
                self.undergraph_resultant_btn.blockSignals(False)
                self.eq_plot._is_meas_checked = bool(dm[0])
                self.eq_plot._is_target_checked = bool(dt[0])
                self.eq_plot._is_resultant_checked = bool(dr[0])
                if not animate:
                    self.eq_plot.set_overlays_visible(
                        bool(dm[0]), bool(dt[0]), bool(dr[0]),
                        show_target2=self.undergraph_target2_btn.isChecked(),
                    )
                self._style_undergraph_toggle_buttons()

                self._restore_autoeq_preset_ui(autoeq_meta, skip_curve_fetch=True)
            else:
                self._autoeq_meas_raw = ([], [])
                self._autoeq_target_raw = ([], [])
                self._disp_meas_freqs = None
                self._disp_meas_spls = None
                self.eq_plot.set_target2_overlay(None, None)
                self.undergraph_target2_btn.blockSignals(True)
                self.undergraph_target2_btn.setChecked(False)
                self.undergraph_target2_btn.setVisible(False)
                self.undergraph_target2_btn.blockSignals(False)
                self.undergraph_meas_btn.blockSignals(True)
                self.undergraph_target_btn.blockSignals(True)
                self.undergraph_resultant_btn.blockSignals(True)
                self.undergraph_meas_btn.setChecked(True)
                self.undergraph_target_btn.setChecked(True)
                self.undergraph_resultant_btn.setChecked(True)
                self.undergraph_meas_btn.blockSignals(False)
                self.undergraph_target_btn.blockSignals(False)
                self.undergraph_resultant_btn.blockSignals(False)
                if not animate:
                    self.eq_plot.set_overlays_visible(True, True, True, show_target2=False)
                self._style_undergraph_toggle_buttons()
                # Avoid launching a duplicate CurveFetchWorker when async fetch was already dispatched
                self._restore_autoeq_preset_ui(autoeq_meta, skip_curve_fetch=needs_network_fetch)
        else:
            self._autoeq_meas_raw = ([], [])
            self._autoeq_target_raw = ([], [])
            self._disp_meas_freqs = None
            self._disp_meas_spls = None
            self.eq_plot.set_target2_overlay(None, None)
            self.undergraph_target2_btn.blockSignals(True)
            self.undergraph_target2_btn.setChecked(False)
            self.undergraph_target2_btn.setVisible(False)
            self.undergraph_target2_btn.blockSignals(False)
            self.undergraph_meas_btn.blockSignals(True)
            self.undergraph_target_btn.blockSignals(True)
            self.undergraph_resultant_btn.blockSignals(True)
            self.undergraph_target_btn.setChecked(False)
            self.undergraph_meas_btn.setChecked(False)
            self.undergraph_resultant_btn.setChecked(False)
            self.undergraph_meas_btn.blockSignals(False)
            self.undergraph_target_btn.blockSignals(False)
            self.undergraph_resultant_btn.blockSignals(False)
            if not animate:
                self.eq_plot.set_overlays_visible(False, False, False, show_target2=False)
            self._style_undergraph_toggle_buttons()

        # Preamp smooth cubic animation
        if animate:
            self._preamp_anim.stop()
            self._preamp_anim.setStartValue(self.preamp_spin.value())
            self._preamp_anim.setEndValue(preamp)
            self._preamp_anim.start()
        else:
            self._apply_preamp_value(preamp)

        # Plot sweep animation (smoothly sweeps EQ curve + target + meas + resultant simultaneously)
        if animate:
            self.eq_plot.animate_to_preset(
                bands,
                preamp,
                target_data=new_target_data,
                meas_data=new_meas_data,
                resultant_data=new_res_data,
            )
        else:
            self.eq_plot.set_bands(bands, preamp_db=preamp)
            self.eq_plot.set_measurement_overlay(new_meas_data[0], new_meas_data[1], label=new_meas_data[2])
            self.eq_plot.set_target_overlay(new_target_data[0], new_target_data[1], label=new_target_data[2])
            self.eq_plot.set_resultant_overlay(new_res_data[0], new_res_data[1], label=new_res_data[2])
            self.eq_plot.set_overlays_visible(
                self.undergraph_meas_btn.isChecked(),
                self.undergraph_target_btn.isChecked(),
                self.undergraph_resultant_btn.isChecked(),
                show_target2=self.undergraph_target2_btn.isChecked(),
            )

        self._populate_table(bands)
        self.preset_manager.schedule_apo_export(preamp, bands, bypassed=self.is_eq_bypassed)
        # Capture target state directly (not mid-animation preamp_spin.value())
        self._last_state = (round(preamp, 2), [b.to_dict() for b in bands])
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_curves_button_state()
        self._update_preamp_headroom_badge(preamp)

    def _restore_autoeq_preset_ui(self, meta: Dict[str, Any], skip_curve_fetch: bool = False) -> None:
        db_id = meta.get("database_id")
        base_url = meta.get("base_url")
        model_file = meta.get("model_file")
        target_file = meta.get("target_file")
        chan = meta.get("channel_mode", "avg")

        self._pending_restore_model_file = model_file
        self._pending_restore_target_file = target_file
        self._pending_restore_skip_fetch = skip_curve_fetch
        if skip_curve_fetch:
            self._suppress_curve_fetch = True

        for i in range(self.autoeq_chan_combo.count()):
            if self.autoeq_chan_combo.itemData(i) == chan:
                self.autoeq_chan_combo.blockSignals(True)
                self.autoeq_chan_combo.setCurrentIndex(i)
                self.autoeq_chan_combo.blockSignals(False)
                break

        found_db = False
        for i in range(self.autoeq_db_combo.count()):
            d = self.autoeq_db_combo.itemData(i)
            if isinstance(d, dict) and (d.get("id") == db_id or (base_url and d.get("base_url", "").rstrip("/") == (base_url or "").rstrip("/"))):
                found_db = True
                if self.autoeq_db_combo.currentIndex() == i:
                    self._apply_pending_autoeq_restore()
                else:
                    self.autoeq_db_combo.setCurrentIndex(i)
                break

        if not found_db and base_url:
            db_name = meta.get("database_name") or db_id or "Reviewer DB"
            new_db_info = {
                "id": db_id or _resolve_db_id(self.squig_client, base_url, "custom"),
                "name": db_name,
                "base_url": base_url,
                "rig": meta.get("rig", "IEC 711"),
            }
            ins_idx = max(0, self.autoeq_db_combo.count() - 1)
            self.autoeq_db_combo.blockSignals(True)
            self.autoeq_db_combo.insertItem(ins_idx, db_name, new_db_info)
            self.autoeq_db_combo.blockSignals(False)
            self.autoeq_db_combo.setCurrentIndex(ins_idx)
        elif not found_db:
            # Nothing to switch to: ensure suppression cannot stay stuck
            self._pending_restore_model_file = None
            self._pending_restore_target_file = None
            self._pending_restore_skip_fetch = False
            self._suppress_curve_fetch = False

    def _apply_pending_autoeq_restore(self) -> None:
        skip = getattr(self, "_pending_restore_skip_fetch", False)
        should_fetch = False
        try:
            if self._pending_restore_model_file and self._autoeq_models:
                target_mf = self._pending_restore_model_file.lower()
                matched_model: Optional[SquigModel] = None
                for m in self._autoeq_models:
                    if (m.file_name.lower() == target_mf or
                        m.name.lower() == target_mf or
                        target_mf in m.file_name.lower() or
                        target_mf in m.name.lower() or
                        target_mf in m.display_name.lower()):
                        matched_model = m
                        break
                if matched_model:
                    self.autoeq_model_combo.set_selected_model(matched_model)
                    if not skip:
                        should_fetch = True

            if self._pending_restore_target_file and self._autoeq_targets:
                target_tf = self._pending_restore_target_file.lower()
                for t_idx, t in enumerate(self._autoeq_targets):
                    if (t.file_name.lower() == target_tf or
                        t.name.lower() == target_tf or
                        target_tf in t.name.lower() or
                        t.name.lower() in target_tf or
                        target_tf in t.file_name.lower()):
                        self.autoeq_target_combo.blockSignals(True)
                        self.autoeq_target_combo.setCurrentIndex(t_idx)
                        self.autoeq_target_combo.blockSignals(False)
                        break
        finally:
            self._pending_restore_model_file = None
            self._pending_restore_target_file = None
            self._pending_restore_skip_fetch = False
            self._suppress_curve_fetch = False

        if should_fetch:
            self._fetch_selected_curves()

    def _on_preamp_anim_step(self, val: float) -> None:
        self._apply_preamp_value(float(val), stop_anim=False)

    def _apply_preamp_value(self, val: float, stop_anim: bool = True) -> None:
        if stop_anim and self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()
        val = max(-30.0, min(15.0, float(val)))
        self.preamp_spin.blockSignals(True)
        self.preamp_slider.blockSignals(True)
        self.preamp_spin.setValue(val)
        self.preamp_slider.setValue(int(round(val * 10)))
        self.preamp_spin.blockSignals(False)
        self.preamp_slider.blockSignals(False)
        self.eq_plot.set_preamp(val)
        self._update_preamp_headroom_badge(val)

    def _schedule_inline_edit_by_name(self, preset_name: str) -> None:
        self._pending_edit_preset_name = preset_name
        self._pending_edit_timer.start(50)

    def _trigger_pending_preset_edit(self) -> None:
        name = self._pending_edit_preset_name
        self._pending_edit_preset_name = None
        if not name or sip.isdeleted(self.preset_list_widget):
            return
        for i in range(self.preset_list_widget.count()):
            item = self.preset_list_widget.item(i)
            if item and item.text() == name:
                self.preset_list_widget.setCurrentItem(item)
                self.preset_list_widget.editItem(item)
                break

    def _add_preset_auto(self) -> None:
        """
        Creates a new, flat EQ profile automatically named 'New Profile [N]'.
        Immediately activates inline editing so user can optionally rename it.
        """
        if self._preset_save_timer.isActive():
            self._flush_preset_save()
        existing = set(self.preset_manager.get_preset_names())
        count = 1
        while f"New Profile {count}" in existing:
            count += 1
        name = f"New Profile {count}"

        # Standard flat default bands
        flat_bands = [
            EQBand(1, True, "Low Shelf", 100.0, 0.0, 0.707),
            EQBand(2, True, "Peak", 250.0, 0.0, 1.414),
            EQBand(3, True, "Peak", 1000.0, 0.0, 1.414),
            EQBand(4, True, "Peak", 4000.0, 0.0, 1.414),
            EQBand(5, True, "High Shelf", 8000.0, 0.0, 0.707),
        ]

        if self.preset_manager.add_preset(name, 0.0, flat_bands):
            self._schedule_inline_edit_by_name(name)

    def _duplicate_preset(self, preset_name: Optional[str] = None) -> None:
        if self._preset_save_timer.isActive():
            self._flush_preset_save()
        source = preset_name if isinstance(preset_name, str) and preset_name else self.preset_manager.active_preset_name
        new_name = self.preset_manager.duplicate_preset(source)
        if new_name:
            self._schedule_inline_edit_by_name(new_name)

    def _delete_preset(self, preset_name: Optional[str] = None) -> None:
        target = preset_name if isinstance(preset_name, str) and preset_name else self.preset_manager.active_preset_name
        if len(self.preset_manager.get_preset_names()) <= 1:
            QtWidgets.QMessageBox.information(self, "Delete Profile", "Cannot delete the only remaining profile.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete", f"Delete profile '{target}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if self._preset_save_timer.isActive():
                self._preset_save_timer.stop()
            self.preset_manager.delete_preset(target)

    def _show_import_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        act_file = menu.addAction("Import from file...")
        act_file.triggered.connect(self._import_eq_dialog)

        act_paste = menu.addAction("Paste APO from clipboard")
        act_paste.triggered.connect(self._paste_apo_from_clipboard)

        menu.exec(self.import_btn.mapToGlobal(QtCore.QPoint(0, self.import_btn.height() + 4)))

    def _show_export_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        act_active = menu.addAction("Export active profile...")
        act_active.triggered.connect(lambda: self._export_eq_dialog(None, full_library=False))

        act_all = menu.addAction("Export full library (.json)...")
        act_all.triggered.connect(lambda: self._export_eq_dialog(None, full_library=True))

        menu.addSeparator()

        act_copy = menu.addAction("Copy APO to clipboard")
        act_copy.triggered.connect(lambda: self._copy_preset_apo(None))

        menu.exec(self.export_btn.mapToGlobal(QtCore.QPoint(0, self.export_btn.height() + 4)))

    def _get_preset_payload(self, preset_name: Optional[str] = None) -> Tuple[str, float, List[EQBand], Optional[Dict[str, Any]], Optional[Dict[str, List[EQBand]]]]:
        if self._preset_save_timer.isActive():
            self._flush_preset_save()
        target = preset_name if isinstance(preset_name, str) and preset_name else self.preset_manager.active_preset_name
        meta = self.preset_manager.get_preset_metadata(target)
        if target == self.preset_manager.active_preset_name:
            ch_bands = self.preset_manager.get_active_channel_bands() or None
            return target, self.preamp_spin.value(), list(self.eq_plot.bands), meta, ch_bands
        p_raw = self.preset_manager.presets.get(target, {})
        preamp_val = float(p_raw.get("preamp", p_raw.get("preamp_db", 0.0)))
        bands = [EQBand.from_dict(b) for b in p_raw.get("bands", []) if isinstance(b, dict)]
        ch_raw = p_raw.get("channel_bands")
        ch_bands = None
        if isinstance(ch_raw, dict):
            parsed_ch: Dict[str, List[EQBand]] = {}
            for ch_k, b_list in ch_raw.items():
                if isinstance(b_list, list):
                    parsed_ch[ch_k] = [EQBand.from_dict(b) for b in b_list if isinstance(b, dict)]
            if parsed_ch:
                ch_bands = parsed_ch
        return target, preamp_val, bands, meta, ch_bands

    def _apply_parsed_import_with_preview(
        self,
        name: str,
        raw_preamp: float,
        raw_bands: List[EQBand],
        info_dict: Dict[str, Any],
        source_label: str = "File",
    ) -> None:
        multi_presets = info_dict.get("multi_presets")
        if not raw_bands and not (multi_presets and len(multi_presets) > 1):
            QtWidgets.QMessageBox.warning(
                self,
                "Import failed",
                "No valid EQ filters were found in the selected input.",
            )
            return

        preamp, bands = self._sanitize_eq_state(raw_preamp, raw_bands)
        existing_names = self.preset_manager.get_preset_names()
        dlg = ImportPreviewDialog(
            name=name,
            preamp_db=preamp,
            bands=bands,
            info_dict=info_dict,
            existing_names=existing_names,
            source_label=source_label,
            parent=self,
        )
        accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        mode = dlg.result_mode
        chosen_name = dlg.chosen_name
        conflict_action = dlg.conflict_action
        selected_multi = dict(dlg.selected_multi_presets)
        dlg.deleteLater()

        if not accepted:
            return

        existing_set = set(self.preset_manager.get_preset_names())
        if mode == "multi" and selected_multi:
            last_imported = None
            for p_name, p_data in selected_multi.items():
                if not isinstance(p_data, dict):
                    continue
                target_name = p_name
                if target_name in existing_set:
                    if conflict_action == "skip":
                        continue
                    if conflict_action == "rename":
                        idx = 2
                        while f"{p_name} ({idx})" in existing_set:
                            idx += 1
                        target_name = f"{p_name} ({idx})"
                p_preamp = float(p_data.get("preamp", p_data.get("preamp_db", 0.0)))
                p_bands_raw = [EQBand.from_dict(b) for b in p_data.get("bands", []) if isinstance(b, dict)]
                p_preamp, p_bands = self._sanitize_eq_state(p_preamp, p_bands_raw)
                p_meta = p_data.get("autoeq") if isinstance(p_data.get("autoeq"), dict) else None
                p_ch = p_data.get("channel_bands") if isinstance(p_data.get("channel_bands"), dict) else None
                self.preset_manager.save_or_update_preset(
                    target_name,
                    p_preamp,
                    p_bands,
                    autoeq_metadata=p_meta,
                    emit_changed=False,
                    channel_bands=p_ch,
                )
                existing_set.add(target_name)
                last_imported = target_name
            if last_imported:
                self.preset_manager.set_active_preset(last_imported)
            return

        # Single preset import
        final_name = chosen_name.strip() or name or "Imported Profile"
        if final_name in existing_set and conflict_action == "rename":
            idx = 2
            while f"{final_name} ({idx})" in existing_set:
                idx += 1
            final_name = f"{final_name} ({idx})"

        meta = info_dict.get("metadata") if isinstance(info_dict.get("metadata"), dict) else None
        ch_bands = info_dict.get("channel_bands") if isinstance(info_dict.get("channel_bands"), dict) else None

        if final_name in existing_set and conflict_action == "overwrite":
            self.preset_manager.save_or_update_preset(
                final_name,
                preamp,
                bands,
                autoeq_metadata=meta,
                emit_changed=True,
                channel_bands=ch_bands,
            )
        else:
            self.preset_manager.add_preset(final_name, preamp, bands, channel_bands=ch_bands)
            if meta:
                self.preset_manager.set_preset_metadata(final_name, meta)

    def _import_eq_dialog(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import equalizer configuration",
            "",
            "EQ Files (*.txt *.peace *.json);;Equalizer APO (*.txt);;Peace (*.peace);;JSON (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            name, raw_preamp, raw_bands, info_dict = preview_eq_import(file_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", f"Could not parse file: {exc}")
            return
        self._apply_parsed_import_with_preview(
            name, raw_preamp, raw_bands, info_dict, source_label=os.path.basename(file_path)
        )

    def _paste_apo_from_clipboard(self) -> None:
        cb = QtWidgets.QApplication.clipboard()
        text = cb.text() if cb else ""
        if not text or not text.strip():
            QtWidgets.QMessageBox.information(self, "Paste APO", "Clipboard is empty.")
            return
        try:
            name, raw_preamp, raw_bands, info_dict = parse_eq_from_text(text, source_name="Clipboard EQ")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Paste failed", f"Could not parse clipboard text: {exc}")
            return
        self._apply_parsed_import_with_preview(
            name, raw_preamp, raw_bands, info_dict, source_label="Clipboard"
        )

    def _copy_preset_apo(self, preset_name: Optional[str] = None) -> None:
        target, preamp_val, bands, _meta, ch_bands = self._get_preset_payload(preset_name)
        dev_name = self.preset_manager.active_device_name or ""
        apo_text = export_eq_to_apo_text(
            preamp_val,
            bands,
            device_name=dev_name,
            channel_bands=ch_bands,
            preset_name=target,
            bypassed=False,
        )
        cb = QtWidgets.QApplication.clipboard()
        if cb:
            cb.setText(apo_text)

    def _export_eq_dialog(self, preset_name: Optional[str] = None, full_library: bool = False) -> None:
        target, preamp_val, bands, meta, ch_bands = self._get_preset_payload(preset_name)
        safe_base = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in target).strip() or "clean_apo_preset"

        if full_library:
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export full profile library",
                "clean_apo_library.json",
                "Full Library JSON (*.json);;All Files (*.*)",
            )
            if not file_path:
                return
            try:
                json_text = export_all_presets_to_json_text(self.preset_manager.presets)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_text)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not export library: {exc}")
            return

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export profile - {target}",
            f"{safe_base}.txt",
            "Equalizer APO (*.txt);;Peace Config (*.peace);;Clean APO Preset (*.json);;Full Library Backup (*.json)",
        )
        if not file_path:
            return

        if "Full Library" in selected_filter:
            try:
                json_text = export_all_presets_to_json_text(self.preset_manager.presets)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(json_text)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not export library: {exc}")
            return

        ok, msg = export_eq_to_file(
            file_path,
            target,
            preamp_val,
            bands,
            metadata=meta,
            channel_bands=ch_bands,
            device_name=self.preset_manager.active_device_name or "",
        )
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Export failed", f"Could not export profile: {msg}")

    # -------------------------------------------------------------------------
    # Preamp Synchronization
    # -------------------------------------------------------------------------
    def _on_preamp_slider_pressed(self) -> None:
        if self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()
        self._is_dragging_preamp_slider = True
        self._record_undo_snapshot()

    def _on_preamp_slider_released(self) -> None:
        if not self._is_dragging_preamp_slider:
            return
        self._is_dragging_preamp_slider = False
        val = self.preamp_spin.value()
        self.preset_manager.update_active_preset_content(val, self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(val, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._commit_state()

    def _on_preamp_spin_changed(self, val: float) -> None:
        if self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()
        self.preamp_slider.blockSignals(True)
        self.preamp_slider.setValue(int(round(val * 10)))
        self.preamp_slider.blockSignals(False)
        self._apply_preamp_change(val)

    def _on_preamp_slider_changed(self, int_val: int) -> None:
        if self._preamp_anim.state() == QtCore.QAbstractAnimation.State.Running:
            self._preamp_anim.stop()
        val = int_val / 10.0
        self.preamp_spin.blockSignals(True)
        self.preamp_spin.setValue(val)
        self.preamp_spin.blockSignals(False)
        if self._is_dragging_preamp_slider:
            self.eq_plot.set_preamp(val)
            self._update_preamp_headroom_badge(val)
            self.preset_manager.schedule_apo_export(val, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        else:
            self._apply_preamp_change(val)

    def _apply_preamp_change(self, preamp_db: float) -> None:
        self._record_undo_snapshot()
        self.eq_plot.set_preamp(preamp_db)
        self._update_preamp_headroom_badge(preamp_db)
        self.preset_manager.update_active_preset_content(preamp_db, self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(preamp_db, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._commit_state()

    # -------------------------------------------------------------------------
    # Flatten, Compress, and Expand All Bands
    # -------------------------------------------------------------------------
    def _flatten_all_bands(self) -> None:
        self._record_undo_snapshot()
        for band in self.eq_plot.bands:
            band.gain = 0.0
        self._apply_preamp_value(0.0)
        self.eq_plot.set_bands(self.eq_plot.bands, preamp_db=0.0)
        self._populate_table(self.eq_plot.bands)
        self.preset_manager.update_active_preset_content(0.0, self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(0.0, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge(0.0)
        self._commit_state()

    def _compress_all_bands(self) -> None:
        self._record_undo_snapshot()
        for band in self.eq_plot.bands:
            band.gain = round(max(-20.0, min(20.0, band.gain / 1.25)), 1)
        self.eq_plot.set_bands(self.eq_plot.bands, preamp_db=self.preamp_spin.value())
        self._populate_table(self.eq_plot.bands)
        self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge()
        self._commit_state()

    def _expand_all_bands(self) -> None:
        self._record_undo_snapshot()
        for band in self.eq_plot.bands:
            band.gain = round(max(-20.0, min(20.0, band.gain * 1.25)), 1)
        self.eq_plot.set_bands(self.eq_plot.bands, preamp_db=self.preamp_spin.value())
        self._populate_table(self.eq_plot.bands)
        self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge()
        self._commit_state()

    # -------------------------------------------------------------------------
    # Table & Plot Bi-directional Binding
    # -------------------------------------------------------------------------
    def _populate_table(self, bands: List[EQBand]) -> None:
        self._updating_table = True
        self.filter_table.setUpdatesEnabled(False)
        try:
            # Explicitly detach & hide old cell widgets so DeferredDelete widgets never linger at (0, 0)
            for r in range(self.filter_table.rowCount()):
                for c in range(self.filter_table.columnCount()):
                    old_w = self.filter_table.cellWidget(r, c)
                    if old_w is not None:
                        self.filter_table.removeCellWidget(r, c)
                        old_w.hide()
                        old_w.setParent(None)
                        old_w.deleteLater()
            self.filter_table.setRowCount(0)
            self.filter_table.setRowCount(len(bands))
            self.band_count_label.setText(f"Bands ({len(bands)})")

            for row, band in enumerate(bands):
                # 0. Band Badge Button (+ subtle L/R channel tag when channel-scoped)
                badge_btn = BandBadgeButton(band.band_id, band.enabled)
                badge_btn.toggled.connect(self._on_table_active_toggled)
                badge_container = QtWidgets.QWidget()
                badge_container.setStyleSheet("background: transparent;")
                b_layout = QtWidgets.QHBoxLayout(badge_container)
                b_layout.setContentsMargins(4, 0, 4, 0)
                b_layout.setSpacing(4)
                b_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                b_layout.addWidget(badge_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

                ch_val = str(getattr(band, "channel", "ALL") or "ALL").upper()
                if ch_val in ("L", "R"):
                    ch_tag = QtWidgets.QLabel(ch_val)
                    ch_tag.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    ch_tag.setFixedSize(18, 18)
                    ch_tag.setToolTip("Left channel only" if ch_val == "L" else "Right channel only")
                    ch_tag.setStyleSheet(
                        f"font-family: {MONO_FONT_FAMILY}; font-size: 10px; font-weight: 500; "
                        f"color: #a1a1aa; background-color: #1c1c1f; border: 1px solid #3f3f46; border-radius: 4px;"
                    )
                    b_layout.addWidget(ch_tag)

                self.filter_table.setCellWidget(row, 0, badge_container)

                # 1. Type ComboBox (NoScrollComboBox ignores wheel to prevent accidental filter type switching)
                type_combo = NoScrollComboBox()
                for ft in [FilterType.PEAK, FilterType.LOW_SHELF, FilterType.HIGH_SHELF, FilterType.HIGH_PASS, FilterType.LOW_PASS]:
                    type_combo.addItem(ft.value)
                type_combo.setCurrentText(band.filter_type)
                type_combo.setProperty("band_id", band.band_id)
                type_combo.currentTextChanged.connect(self._on_table_type_changed)
                self.filter_table.setCellWidget(row, 1, type_combo)

                # 2. Frequency SpinBox (CleanSpinBox: highlights numeric value only)
                freq_spin = CleanSpinBox()
                freq_spin.setRange(20.0, 20000.0)
                freq_spin.setDecimals(1)
                freq_spin.setSingleStep(10.0)
                freq_spin.setSuffix(" Hz")
                freq_spin.setValue(band.freq)
                freq_spin.setProperty("band_id", band.band_id)
                freq_spin.valueChanged.connect(self._on_table_freq_changed)
                self.filter_table.setCellWidget(row, 2, freq_spin)

                # 3. Gain SpinBox (CleanSpinBox: highlights numeric value only)
                gain_spin = CleanSpinBox()
                gain_spin.setRange(-20.0, 20.0)
                gain_spin.setDecimals(1)
                gain_spin.setSingleStep(0.5)
                gain_spin.setSuffix(" dB")
                gain_spin.setValue(band.gain)
                ftype = FilterType.from_str(band.filter_type)
                gain_spin.setEnabled(ftype not in (FilterType.HIGH_PASS, FilterType.LOW_PASS))
                gain_spin.setProperty("band_id", band.band_id)
                gain_spin.valueChanged.connect(self._on_table_gain_changed)
                self.filter_table.setCellWidget(row, 3, gain_spin)

                # 4. Q Factor SpinBox
                q_spin = CleanSpinBox()
                q_spin.setRange(0.1, 20.0)
                q_spin.setDecimals(2)
                q_spin.setSingleStep(0.1)
                q_spin.setValue(band.q)
                q_spin.setProperty("band_id", band.band_id)
                q_spin.valueChanged.connect(self._on_table_q_changed)
                self.filter_table.setCellWidget(row, 4, q_spin)

                # 5. Vector Delete Button (no unicode X)
                del_btn = TableDeleteButton(band.band_id)
                del_btn.clicked.connect(self._on_table_delete_clicked)

                del_container = QtWidgets.QWidget()
                del_container.setStyleSheet("background: transparent;")
                del_layout = QtWidgets.QHBoxLayout(del_container)
                del_layout.setContentsMargins(0, 0, 0, 0)
                del_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                del_layout.addWidget(del_btn)
                self.filter_table.setCellWidget(row, 5, del_container)
        finally:
            self.filter_table.setUpdatesEnabled(True)
            self.filter_table.updateGeometries()
            self.filter_table.doItemsLayout()
            self.filter_table.viewport().update()
            self._updating_table = False

    def _sync_table_row_for_band(self, band_id: int, freq: float, gain: float, q: float, filter_type: str, enabled: bool) -> None:
        self._updating_table = True
        for row in range(self.filter_table.rowCount()):
            badge_widget = self.filter_table.cellWidget(row, 0)
            if not badge_widget:
                continue
            badge_btn = badge_widget.findChild(BandBadgeButton)
            if badge_btn and badge_btn.band_id == band_id:
                badge_btn.blockSignals(True)
                badge_btn.setChecked(enabled)
                badge_btn.blockSignals(False)

                type_combo = self.filter_table.cellWidget(row, 1)
                if isinstance(type_combo, QtWidgets.QComboBox):
                    type_combo.blockSignals(True)
                    type_combo.setCurrentText(filter_type)
                    type_combo.blockSignals(False)

                freq_spin = self.filter_table.cellWidget(row, 2)
                if isinstance(freq_spin, QtWidgets.QDoubleSpinBox):
                    freq_spin.blockSignals(True)
                    freq_spin.setValue(freq)
                    freq_spin.blockSignals(False)

                gain_spin = self.filter_table.cellWidget(row, 3)
                if isinstance(gain_spin, QtWidgets.QDoubleSpinBox):
                    gain_spin.blockSignals(True)
                    gain_spin.setValue(gain)
                    ftype = FilterType.from_str(filter_type)
                    gain_spin.setEnabled(ftype not in (FilterType.HIGH_PASS, FilterType.LOW_PASS))
                    gain_spin.blockSignals(False)

                q_spin = self.filter_table.cellWidget(row, 4)
                if isinstance(q_spin, QtWidgets.QDoubleSpinBox):
                    q_spin.blockSignals(True)
                    q_spin.setValue(q)
                    q_spin.blockSignals(False)
                break
        self._updating_table = False

    # -------------------------------------------------------------------------
    # Table Callbacks
    # -------------------------------------------------------------------------
    def _on_table_active_toggled(self, checked: bool) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            self.eq_plot.update_band_params(band_id, enabled=checked)
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge()
            self._commit_state()

    def _on_table_type_changed(self, filter_type: str) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            ftype = FilterType.from_str(filter_type)
            is_pass = ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS)
            gain = 0.0 if is_pass else None
            self.eq_plot.update_band_params(band_id, filter_type=filter_type, gain=gain)
            self._updating_table = True
            try:
                for row in range(self.filter_table.rowCount()):
                    badge_widget = self.filter_table.cellWidget(row, 0)
                    badge_btn = badge_widget.findChild(BandBadgeButton) if badge_widget else None
                    if badge_btn and badge_btn.band_id == band_id:
                        gain_spin = self.filter_table.cellWidget(row, 3)
                        if isinstance(gain_spin, QtWidgets.QDoubleSpinBox):
                            gain_spin.blockSignals(True)
                            if is_pass:
                                gain_spin.setValue(0.0)
                            gain_spin.setEnabled(not is_pass)
                            gain_spin.blockSignals(False)
                        break
            finally:
                self._updating_table = False
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge()
            self._commit_state()

    def _on_table_freq_changed(self, freq: float) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            self.eq_plot.update_band_params(band_id, freq=freq)
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge()
            self._commit_state()

    def _on_table_gain_changed(self, gain: float) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            self.eq_plot.update_band_params(band_id, gain=gain)
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge()
            self._commit_state()

    def _on_table_q_changed(self, q: float) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            self.eq_plot.update_band_params(band_id, q=q)
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge()
            self._commit_state()

    def _on_table_delete_clicked(self) -> None:
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._delete_band(band_id)

    # -------------------------------------------------------------------------
    # Plot Callbacks
    # -------------------------------------------------------------------------
    def _on_plot_band_modified(self, band_id: int, freq: float, gain: float, q: float, filter_type: str, enabled: bool) -> None:
        self._sync_table_row_for_band(band_id, freq, gain, q, filter_type, enabled)
        self._preset_save_timer.start(150)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge()
        self._commit_state()

    def _on_plot_band_added(self, freq: float, gain: float) -> None:
        self._record_undo_snapshot()
        existing_ids = [b.band_id for b in self.eq_plot.bands]
        new_id = 1
        while new_id in existing_ids:
            new_id += 1

        new_band = EQBand(
            band_id=new_id,
            enabled=True,
            filter_type="Peak",
            freq=round(max(20.0, min(20000.0, freq)), 1),
            gain=round(max(-20.0, min(20.0, gain)), 1),
            q=1.414,
        )
        bands = list(self.eq_plot.bands) + [new_band]
        bands.sort(key=lambda b: b.freq)
        for idx, b in enumerate(bands, start=1):
            b.band_id = idx

        self.eq_plot.set_bands(bands, preamp_db=self.preamp_spin.value())
        self._populate_table(bands)
        self.preset_manager.update_active_preset_content(self.preamp_spin.value(), bands)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge()
        self._commit_state()

    def _add_band_default(self) -> None:
        self._on_plot_band_added(1000.0, 0.0)

    def _delete_band(self, band_id: int) -> None:
        self._record_undo_snapshot()
        bands = [b for b in self.eq_plot.bands if b.band_id != band_id]
        if not bands:
            bands = [EQBand(band_id=1, enabled=True, filter_type="Peak", freq=1000.0, gain=0.0, q=1.414)]
        else:
            bands.sort(key=lambda b: b.freq)
            for idx, b in enumerate(bands, start=1):
                b.band_id = idx

        self.eq_plot.set_bands(bands, preamp_db=self.preamp_spin.value())
        self._populate_table(bands)
        self.preset_manager.update_active_preset_content(self.preamp_spin.value(), bands)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge()
        self._commit_state()

    def _reset_all_bands(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Reset bands", "Reset all filter bands to default flat settings?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._record_undo_snapshot()
            bands = [
                EQBand(1, True, "Low Shelf", 100.0, 0.0, 0.707),
                EQBand(2, True, "Peak", 1000.0, 0.0, 1.414),
                EQBand(3, True, "High Shelf", 8000.0, 0.0, 0.707),
            ]
            self._apply_preamp_value(0.0)
            self.eq_plot.set_bands(bands, preamp_db=0.0)
            self._populate_table(bands)
            self.preset_manager.update_active_preset_content(0.0, bands)
            self.preset_manager.schedule_apo_export(0.0, bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._update_preamp_headroom_badge(0.0)
            self._commit_state()

    # -------------------------------------------------------------------------
    # Header & Settings Handlers
    # -------------------------------------------------------------------------
    def _sync_spectrum_button_style(self, is_on: bool) -> None:
        self.spectrum_toggle_btn.setText("Spectrum")
        if is_on:
            self.spectrum_toggle_btn.setObjectName("activeEqBtn")
            self.spectrum_toggle_btn.set_status_dot("#d99b38" if self._spectrum_frozen else "#4cc38a", filled=True)
        else:
            self.spectrum_toggle_btn.setObjectName("bypassedEqBtn")
            self.spectrum_toggle_btn.set_status_dot("#71717a", filled=False)
        self.spectrum_toggle_btn.setStyle(self.spectrum_toggle_btn.style())
        if hasattr(self.eq_plot, "set_peak_hold_visible"):
            self.eq_plot.set_peak_hold_visible(bool(is_on and self._spectrum_peak_hold))

    def _toggle_spectrum(self) -> None:
        is_on = self.spectrum_toggle_btn.isChecked()
        self.spectrum_worker.set_enabled(is_on)
        self.eq_plot.spectrum_curve.setVisible(is_on)
        self._sync_spectrum_button_style(is_on)

    def _on_plot_spectrum_toggled(self, visible: bool) -> None:
        self.spectrum_toggle_btn.blockSignals(True)
        self.spectrum_toggle_btn.setChecked(visible)
        self.spectrum_worker.set_enabled(visible)
        self._sync_spectrum_button_style(visible)
        self.spectrum_toggle_btn.blockSignals(False)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.preset_manager.apo_target_path,
            self,
            preset_manager=self.preset_manager,
        )
        dialog.sigPathUpdated.connect(self._on_apo_path_changed)
        dialog.sigApoRepaired.connect(self._refresh_apo_health_indicator)
        dialog.exec()
        dialog.deleteLater()
        self._refresh_apo_health_indicator()

    def _open_hearing_test(self) -> None:
        """Opens the interactive Hearing & Headphone Calibration Test dialog."""
        active_out = self.preset_manager.active_device_name
        if not active_out or clean_apo_device_name(active_out).lower() == "all":
            active_out = None
        try:
            dialog = HearingTestDialog(device_index=None, parent=self, device_name=active_out)
        except TypeError:
            dialog = HearingTestDialog(device_index=None, parent=self)
        dialog.sigApplyEQ.connect(self._on_hearing_test_applied)
        dialog.sigSavePreset.connect(self._on_hearing_test_save_preset)
        if hasattr(dialog, "sigPreviewEQ"):
            dialog.sigPreviewEQ.connect(self._on_hearing_test_preview)
        if hasattr(dialog, "sigPreviewEnded"):
            dialog.sigPreviewEnded.connect(self._on_hearing_test_preview_ended)
        if hasattr(dialog, "sigMeasurementBypassRequested"):
            dialog.sigMeasurementBypassRequested.connect(self._on_hearing_test_measurement_bypass)
        dialog.exec()
        dialog.deleteLater()
        # Guarantee temporary measurement bypass is cleanly restored even if closed unexpectedly
        if self._hearing_test_measurement_bypass:
            self._on_hearing_test_measurement_bypass(False)

    def _on_hearing_test_measurement_bypass(self, bypass: bool) -> None:
        """
        Temporarily bypasses Equalizer APO output during hearing measurement
        without mutating the saved preset or user's universal bypass toggle.
        """
        self._hearing_test_measurement_bypass = bool(bypass)
        effective_bypass = True if self._hearing_test_measurement_bypass else self.is_eq_bypassed
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(),
            self.eq_plot.bands,
            bypassed=effective_bypass,
        )

    def _on_hearing_test_preview(self, bands: List[EQBand], preamp_db: float, is_bypassed: bool) -> None:
        """Temporary A/B preview for Equalizer APO without mutating the active preset or UI state."""
        preamp_db, clean_bands = self._sanitize_eq_state(preamp_db, bands)
        self.preset_manager.schedule_apo_export(preamp_db, clean_bands, bypassed=bool(is_bypassed))

    def _on_hearing_test_preview_ended(self) -> None:
        """Restores the real active preset to Equalizer APO when Hearing Test closes without Apply/Save."""
        self._hearing_test_measurement_bypass = False
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
        )

    def _on_hearing_test_applied(self, bands: List[EQBand], preamp_db: float) -> None:
        """Applies calibrated hearing correction bands (preserving L/R channels) and anti-clipping preamp."""
        self._hearing_test_measurement_bypass = False
        self._record_undo_snapshot()
        preamp_db, clean_bands = self._sanitize_eq_state(preamp_db, bands)
        if self.is_eq_bypassed:
            self._set_universal_eq_bypassed(False, export=False)
        self.eq_plot.animate_to_preset(clean_bands, preamp_db)
        self._apply_preamp_value(preamp_db)
        self._populate_table(clean_bands)
        self.preset_manager.update_active_preset_content(preamp_db, clean_bands)
        self.preset_manager.schedule_apo_export(preamp_db, clean_bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
        self._update_preamp_headroom_badge(preamp_db)
        self._last_state = (round(preamp_db, 2), [b.to_dict() for b in clean_bands])

    def _on_hearing_test_save_preset(self, name: str, bands: List[EQBand], preamp_db: float) -> None:
        """Saves calibrated hearing correction bands (preserving L/R channels) as a new preset."""
        self._hearing_test_measurement_bypass = False
        preamp_db, clean_bands = self._sanitize_eq_state(preamp_db, bands)
        if self.is_eq_bypassed:
            self._set_universal_eq_bypassed(False, export=False)
        self.preset_manager.save_or_update_preset(name, preamp_db, clean_bands)

    def _on_apo_path_changed(self, new_path: str) -> None:
        self.preset_manager.apo_target_path = new_path
        self.preset_manager.save_settings()
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
        )
        self._refresh_apo_health_indicator()

    # -------------------------------------------------------------------------
    # AutoEQ & Squiglink Database Handlers
    # -------------------------------------------------------------------------
    def _switch_sidebar_tab(self, index: int) -> None:
        self.sidebar_stack.slideToIndex(index)
        self.sidebar_tab_bar.set_current_index(index, animate=True)

    def _open_global_search(self) -> None:
        initial_q = self.autoeq_model_combo.currentText().strip()
        dlg = GlobalSearchDialog(self.squig_client, initial_query=initial_q, preset_manager=self.preset_manager, parent=self)
        accepted = dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        res = dlg.selected_result if accepted else None
        role = getattr(dlg, "load_role", "source")
        dlg.deleteLater()
        if res:
            if role == "target":
                self._on_global_search_applied_as_target(res)
            else:
                self._on_global_search_applied(res)

    def _on_global_search_applied_as_target(self, res: GlobalSearchResult) -> None:
        self._switch_sidebar_tab(1)
        self.autoeq_target_mode_tab_bar.set_current_index(1)
        self._on_target_mode_changed(1)

        is_same = bool(
            self._active_db_info.get("base_url") and
            res.base_url.rstrip("/").lower() == self._active_db_info.get("base_url", "").rstrip("/").lower()
        )
        self.autoeq_target_same_provider_cb.blockSignals(True)
        self.autoeq_target_same_provider_cb.setChecked(is_same)
        self.autoeq_target_same_provider_cb.blockSignals(False)
        self.autoeq_target_iem_combo.same_provider_only = is_same

        disp = res.full_name or f"{res.brand} {res.name}".strip() or res.name
        target_m = SquigModel(
            brand=res.brand,
            name=res.name,
            full_name=res.full_name,
            file_name=res.file_name,
            display_name=disp,
        )
        target_m.base_url = res.base_url
        target_m.provider_name = res.provider_name
        target_m.rig = res.rig
        target_m.database_id = _resolve_db_id(self.squig_client, res.base_url)

        self.autoeq_target_iem_combo.set_selected_model(target_m)
        self.autoeq_target_iem_badge.setText(f"Rig: {res.rig} [{'Matched' if is_same else 'External'}]")
        self._fetch_selected_curves()

    def _on_target_mode_changed(self, idx: int) -> None:
        self._autoeq_target_mode = idx
        self.autoeq_target_stack.setCurrentIndex(idx)
        self._sync_target_stack_height()
        if idx == 2:
            self._populate_target_profiles()
        self._fetch_selected_curves()

    def _on_target_same_provider_toggled(self, checked: bool) -> None:
        if hasattr(self, "autoeq_target_iem_combo"):
            self.autoeq_target_iem_combo.set_same_provider_only(checked)
            self._fetch_selected_curves()

    def _populate_target_profiles(self) -> None:
        if not hasattr(self, "autoeq_target_profile_combo"):
            return
        curr = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
        self.autoeq_target_profile_combo.blockSignals(True)
        self.autoeq_target_profile_combo.clear()
        presets = self.preset_manager.get_preset_names()
        sel_idx = 0
        for i, p in enumerate(presets):
            meta = self.preset_manager.get_preset_metadata(p)
            has_autoeq = bool(meta and meta.get("is_autoeq"))
            tag = " [AutoEQ]" if has_autoeq else ""
            self.autoeq_target_profile_combo.addItem(f"{p}{tag}", p)
            if curr and p == curr:
                sel_idx = i
        if presets:
            self.autoeq_target_profile_combo.setCurrentIndex(sel_idx)
        self.autoeq_target_profile_combo.blockSignals(False)

    def _compute_preset_resultant_curve(
        self,
        preset_name: str,
        allow_network: bool = True,
    ) -> Tuple[Optional[Tuple[List[float], List[float]]], bool]:
        p_data = self.preset_manager.presets.get(preset_name)
        if not isinstance(p_data, dict):
            return None, False
        bands: List[EQBand] = []
        for b in p_data.get("bands", []):
            if not isinstance(b, dict):
                continue
            try:
                bands.append(EQBand.from_dict(b))
            except Exception:
                continue
        meta = p_data.get("autoeq") or {}
        base_u = meta.get("base_url", "")
        db_id = meta.get("database_id", "") or _resolve_db_id(self.squig_client, base_u)
        m_file = meta.get("model_file") or meta.get("model_name", "")
        chan = meta.get("channel_mode", "avg")

        off_db = get_offline_db()
        m_res = off_db.get_measurement(db_id, m_file, chan) if (db_id and m_file) else None
        needs_net = False
        if not m_res and m_file and base_u:
            if allow_network:
                try:
                    temp_m = SquigModel(name=m_file, file_name=m_file, brand="", full_name=m_file, display_name=m_file)
                    m_res = self.squig_client.fetch_iem_measurement(base_u, temp_m, chan)
                except Exception:
                    m_res = None
            else:
                needs_net = True

        if m_res and len(m_res[0]) > 0 and len(m_res[0]) == len(m_res[1]):
            pmf, pms = m_res
            eq_resp = compute_composite_response(bands, np.asarray(pmf, dtype=np.float64), preamp_db=0.0, fs=self.sample_rate)
            res_spls = np.asarray(pms, dtype=np.float64) + eq_resp
            return (list(pmf), res_spls.tolist()), False
        if needs_net:
            return None, True
        f_grid = np.geomspace(20.0, 20000.0, num=300)
        eq_resp = compute_composite_response(bands, f_grid, preamp_db=0.0, fs=self.sample_rate)
        return (f_grid.tolist(), eq_resp.tolist()), False

    def _build_current_autoeq_preset_meta(self) -> Tuple[str, Dict[str, Any]]:
        model = self.autoeq_model_combo.currentData()
        chan = self.autoeq_chan_combo.currentData() or "avg"
        m_name = model.name if model else "IEM"
        mode = getattr(self, "_autoeq_target_mode", 0)
        active_base_url = self._active_db_info.get("base_url", "")
        active_db_id = self._active_db_info.get("id", "") or _resolve_db_id(self.squig_client, active_base_url)

        if mode == 0:
            target = self.autoeq_target_combo.currentData()
            t_name = target.name if target else "Target"
            preset_name = f"AutoEQ: {m_name} ({t_name})"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "target_curve",
                "database_id": active_db_id,
                "database_name": self._active_db_info.get("name", ""),
                "base_url": active_base_url,
                "model_name": m_name,
                "model_file": model.file_name if model else "",
                "target_name": t_name,
                "target_file": (target.file_name or target.name) if target else "",
                "channel_mode": chan,
            }
        elif mode == 1:
            target_iem = self.autoeq_target_iem_combo.currentData()
            t_name = target_iem.name if target_iem else "Target IEM"
            t_url = getattr(target_iem, "base_url", "") or active_base_url
            t_prov = getattr(target_iem, "provider_name", "") or self._active_db_info.get("name", "")
            t_rig = getattr(target_iem, "rig", "") or self._active_db_info.get("rig", "")
            t_db_id = getattr(target_iem, "database_id", "") or _resolve_db_id(self.squig_client, t_url, active_db_id)
            t_chan = self.autoeq_target_iem_chan_combo.currentData() or "avg"
            preset_name = f"AutoEQ: {m_name} -> {t_name}"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "iem_tuning",
                "database_id": active_db_id,
                "database_name": self._active_db_info.get("name", ""),
                "base_url": active_base_url,
                "model_name": m_name,
                "model_file": model.file_name if model else "",
                "channel_mode": chan,
                "target_name": f"{t_name} (Tuning)",
                "target_iem_name": t_name,
                "target_iem_file": target_iem.file_name if target_iem else "",
                "target_iem_channel": t_chan,
                "target_iem_base_url": t_url,
                "target_iem_database_id": t_db_id,
                "target_iem_provider": t_prov,
                "target_iem_rig": t_rig,
            }
        else:  # mode == 2
            target_prof = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
            preset_name = f"AutoEQ: {m_name} -> {target_prof}"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "resultant_profile",
                "database_id": active_db_id,
                "database_name": self._active_db_info.get("name", ""),
                "base_url": active_base_url,
                "model_name": m_name,
                "model_file": model.file_name if model else "",
                "channel_mode": chan,
                "target_name": f"{target_prof} (Resultant)",
                "target_profile_name": target_prof,
            }
        return preset_name, autoeq_meta

    def _open_offline_sync(self) -> None:
        dlg = OfflineSyncDialog(self.squig_client, parent=self)
        dlg.exec()
        dlg.deleteLater()
        self._update_offline_db_badge()

    def _update_offline_db_badge(self) -> None:
        try:
            stats = get_offline_db().get_stats()
            total_items = stats["measurements"] + stats["targets"]
            if total_items > 0:
                self.autoeq_offline_sync_btn.setText(f"Offline ({total_items:,})")
                self.autoeq_offline_sync_btn.setToolTip(
                    f"Offline Database Library: {stats['measurements']:,} models, {stats['targets']:,} targets ({stats['size_mb']} MB cached)"
                )
            else:
                self.autoeq_offline_sync_btn.setText("Offline DB")
                self.autoeq_offline_sync_btn.setToolTip("Download and manage offline measurement and target curves (~60 MB)")
        except Exception:
            pass

    def _on_bg_sync_progress(self, current: int, total: int, status: str, item_name: str) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self.autoeq_offline_sync_btn.setText(f"Sync: {pct}%")
        self.autoeq_offline_sync_btn.setToolTip(f"Offline Syncing in Background: {current}/{total} ({status})")

    def _on_bg_sync_finished(self, new_saved: int, total_meas: int, total_targets: int) -> None:
        self._update_offline_db_badge()
        self.autoeq_status_label.setText(f"Offline sync complete - {total_meas + total_targets:,} curves cached")
        self.autoeq_status_label.setStyleSheet("color: #4cc38a; font-size: 11px; background: transparent;")

    def _on_global_search_applied(self, res: GlobalSearchResult) -> None:
        # Switch to AutoEQ sidebar tab
        self._switch_sidebar_tab(1)

        # Check if database is already in the combo
        res_url = res.base_url.rstrip("/")
        target_idx = -1
        for i in range(self.autoeq_db_combo.count()):
            d = self.autoeq_db_combo.itemData(i)
            if isinstance(d, dict) and d.get("base_url", "").rstrip("/") == res_url:
                target_idx = i
                break

        # If not present in DEFAULT_DATABASES, dynamically register it
        if target_idx == -1:
            new_db_info = {
                "id": _resolve_db_id(self.squig_client, res.base_url, res.provider_name.lower().replace(" ", "_")),
                "name": res.provider_name,
                "base_url": res.base_url,
                "rig": res.rig,
                "category": res.db_type,
            }
            custom_idx = max(0, self.autoeq_db_combo.count() - 1)
            self.autoeq_db_combo.blockSignals(True)
            self.autoeq_db_combo.insertItem(custom_idx, res.provider_name, new_db_info)
            self.autoeq_db_combo.blockSignals(False)
            target_idx = custom_idx

        self._pending_global_model = res
        if self.autoeq_db_combo.currentIndex() == target_idx:
            # Load immediately if already at this database index
            d = self.autoeq_db_combo.itemData(target_idx)
            self._load_database_index(d)
        else:
            self.autoeq_db_combo.setCurrentIndex(target_idx)

    def _on_autoeq_db_changed(self, idx: int = -1) -> None:
        # Debounce database index changes slightly to ensure butter-smooth transitions
        self._db_switch_timer.start(100)

    def _do_autoeq_db_changed(self) -> None:
        db_data = self.autoeq_db_combo.currentData()
        if not db_data:
            return
        if db_data.get("id") == "custom":
            url, ok = QtWidgets.QInputDialog.getText(
                self,
                "Custom Squiglink Database",
                "Enter Squiglink Base URL:\n(e.g. https://squig.link/ or https://crinacle.squig.link/)",
                text="https://",
            )
            if ok and url.strip() and url.strip() != "https://":
                url_clean = url.strip()
                if not url_clean.startswith(("http://", "https://")):
                    url_clean = "https://" + url_clean
                if not url_clean.endswith("/"):
                    url_clean += "/"
                import time
                custom_db = {
                    "id": _resolve_db_id(self.squig_client, url_clean, f"custom_{int(time.time())}"),
                    "name": f"Custom ({urllib.parse.urlparse(url_clean).netloc})",
                    "base_url": url_clean,
                    "rig": "Custom Coupler",
                }
                ins_idx = max(0, self.autoeq_db_combo.count() - 1)
                self.autoeq_db_combo.blockSignals(True)
                self.autoeq_db_combo.insertItem(ins_idx, custom_db["name"], custom_db)
                self.autoeq_db_combo.setCurrentIndex(ins_idx)
                self.autoeq_db_combo.blockSignals(False)
                self._active_db_info = custom_db
                self._load_database_index(custom_db)
            else:
                self._revert_db_combo_to_active()
            return

        self._active_db_info = dict(db_data)
        self._update_db_star_btn_style()
        self._load_database_index(self._active_db_info)

    def _populate_autoeq_databases(self, preserve_id: Optional[str] = None) -> None:
        curr_id = preserve_id or (self.autoeq_db_combo.currentData().get("id") if self.autoeq_db_combo.currentData() else None)
        self.autoeq_db_combo.blockSignals(True)
        self.autoeq_db_combo.clear()

        favs = self.preset_manager.get_favorite_databases() if self.preset_manager else []
        db_list = list(DEFAULT_DATABASES)
        seen_urls = {d.get("base_url", "").rstrip("/") for d in db_list}

        if hasattr(self, "squig_client") and self.squig_client:
            cat = self.squig_client.get_global_catalog()
            for c in cat:
                b_url = c.get("base_url", "").rstrip("/")
                if b_url and b_url not in seen_urls:
                    seen_urls.add(b_url)
                    cat_type = c.get("db_type", "IEMs")
                    p_name = c.get("provider", "Reviewer")
                    disp = f"{p_name} ({cat_type})" if f"({cat_type.lower()}" not in p_name.lower() else p_name
                    db_list.append({
                        "id": _resolve_db_id(self.squig_client, c.get("base_url", ""), f"{c.get('username') or p_name.lower().replace(' ', '_')}_{cat_type.lower()}"),
                        "name": disp,
                        "base_url": c.get("base_url", ""),
                        "rig": c.get("rig", "IEC 711"),
                        "category": cat_type,
                    })

        sorted_dbs = sorted(
            db_list,
            key=lambda d: (0 if d.get("id") in favs else 1, d.get("name", "").lower()),
        )

        sel_idx = 0
        for i, db in enumerate(sorted_dbs):
            disp_name = db["name"]
            self.autoeq_db_combo.addItem(disp_name, db)
            if curr_id and db.get("id") == curr_id:
                sel_idx = i

        self.autoeq_db_combo.addItem("+ Custom Squiglink URL...", {"id": "custom", "name": "Custom", "base_url": "", "rig": "Custom"})
        self.autoeq_db_combo.setCurrentIndex(sel_idx)
        self.autoeq_db_combo.blockSignals(False)
        self._update_db_star_btn_style()

    def _update_db_star_btn_style(self) -> None:
        db_data = self.autoeq_db_combo.currentData()
        if not db_data or db_data.get("id") == "custom":
            self.autoeq_db_star_btn.setEnabled(False)
            self.autoeq_db_star_btn.set_favorited(False)
            self.autoeq_db_star_btn.setToolTip("Pin database to top of list")
            return
        self.autoeq_db_star_btn.setEnabled(True)
        db_id = db_data.get("id", "")
        is_fav = self.preset_manager.is_favorite_database(db_id)
        self.autoeq_db_star_btn.set_favorited(is_fav)
        self.autoeq_db_star_btn.setToolTip("Unpin database from top of list" if is_fav else "Pin database to top of list")

    def _toggle_current_db_favorite(self) -> None:
        db_data = self.autoeq_db_combo.currentData()
        if not db_data or db_data.get("id") == "custom":
            return
        db_id = db_data.get("id", "")
        self.preset_manager.toggle_favorite_database(db_id)
        self._populate_autoeq_databases(preserve_id=db_id)

    def _style_undergraph_toggle_buttons(self) -> None:
        t_active = self.undergraph_target_btn.isChecked()
        self.undergraph_target_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1c1c1f" if t_active else "#121214"};
                color: {"#d99b38" if t_active else "#71717a"};
                border: 1px solid {"#854d0e" if t_active else "#27272a"};
                border-radius: 4px;
                padding: 0px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {"#d99b38" if t_active else "#3f3f46"};
                color: {"#d99b38" if t_active else "#a1a1aa"};
            }}
        """)

        t2_active = self.undergraph_target2_btn.isChecked()
        self.undergraph_target2_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1c1c1f" if t2_active else "#121214"};
                color: {"#c084fc" if t2_active else "#71717a"};
                border: 1px solid {"#7e22ce" if t2_active else "#27272a"};
                border-radius: 4px;
                padding: 0px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {"#c084fc" if t2_active else "#3f3f46"};
                color: {"#c084fc" if t2_active else "#a1a1aa"};
            }}
        """)

        m_active = self.undergraph_meas_btn.isChecked()
        self.undergraph_meas_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1c1c1f" if m_active else "#121214"};
                color: {"#ededed" if m_active else "#71717a"};
                border: 1px solid {"#3f3f46" if m_active else "#27272a"};
                border-radius: 4px;
                padding: 0px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {"#a1a1aa" if m_active else "#3f3f46"};
                color: {"#ededed" if m_active else "#a1a1aa"};
            }}
        """)

        r_active = self.undergraph_resultant_btn.isChecked()
        self.undergraph_resultant_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1c1c1f" if r_active else "#121214"};
                color: {"#7dd3fc" if r_active else "#71717a"};
                border: 1px solid {"#0369a1" if r_active else "#27272a"};
                border-radius: 4px;
                padding: 0px 8px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                border-color: {"#7dd3fc" if r_active else "#3f3f46"};
                color: {"#7dd3fc" if r_active else "#a1a1aa"};
            }}
        """)

    def _on_undergraph_overlay_toggled(self) -> None:
        self._style_undergraph_toggle_buttons()
        active_preset = self.preset_manager.active_preset_name
        has_meta = self.preset_manager.has_preset_metadata(active_preset)
        has_plot_curves = (
            self.eq_plot._meas_freqs is not None or
            self.eq_plot._target_freqs is not None or
            self.eq_plot._resultant_freqs is not None or
            (self._disp_meas_freqs is not None and len(self._disp_meas_freqs) > 0) or
            bool(self._autoeq_meas_raw[0])
        )
        sender = self.sender()
        if not has_meta and not has_plot_curves and sender and sender.isChecked():
            sender.blockSignals(True)
            sender.setChecked(False)
            sender.blockSignals(False)
            self._style_undergraph_toggle_buttons()

            self._open_assign_curves_dialog(active_preset)
            return

        self.eq_plot.set_overlays_visible(
            self.undergraph_meas_btn.isChecked(),
            self.undergraph_target_btn.isChecked(),
            self.undergraph_resultant_btn.isChecked(),
            show_target2=self.undergraph_target2_btn.isChecked(),
        )

    def _on_graph_legend_clicked(self, curve_type: str) -> None:
        if curve_type == "target":
            self.undergraph_target_btn.toggle()
        elif curve_type == "target2":
            self.undergraph_target2_btn.toggle()
        elif curve_type == "meas":
            self.undergraph_meas_btn.toggle()
        elif curve_type == "resultant":
            self.undergraph_resultant_btn.toggle()

    def _on_add_target_clicked(self) -> None:
        active_preset = self.preset_manager.active_preset_name
        meta = self.preset_manager.get_preset_metadata(active_preset) or {}
        base_url = meta.get("base_url")
        if not base_url:
            self._open_assign_curves_dialog(active_preset)
            return

        targets = self.squig_client.get_targets_for_provider(base_url)
        if not targets:
            self._open_assign_curves_dialog(active_preset)
            return

        menu = QtWidgets.QMenu(self)

        curr_t2 = meta.get("target2_name") or meta.get("target2_file") or ""
        p_name = meta.get("database_name") or "Provider"
        lbl_head = menu.addAction(f"Target 2 ({p_name})")
        lbl_head.setEnabled(False)
        menu.addSeparator()

        for t in targets:
            t_name = t.name
            t_file = t.file_name
            is_active = bool(curr_t2 and (curr_t2.lower() == t_name.lower() or curr_t2.lower() == t_file.lower()))
            cat_tag = f"[{t.category}] " if t.category else ""
            act = menu.addAction(f"{cat_tag}{t_name}")
            act.setCheckable(True)
            act.setChecked(is_active)
            act.triggered.connect(
                lambda checked=False, tf=t_file, tn=t_name: self._set_preset_target2(active_preset, tf, tn)
            )

        if curr_t2:
            menu.addSeparator()
            act_rm = menu.addAction("Remove Target 2")
            act_rm.triggered.connect(lambda: self._set_preset_target2(active_preset, None, None))

        menu.addSeparator()
        act_config = menu.addAction("Configure reference curves...")
        act_config.triggered.connect(lambda: self._open_assign_curves_dialog(active_preset))

        menu.exec(self.undergraph_add_target_btn.mapToGlobal(QtCore.QPoint(0, self.undergraph_add_target_btn.height() + 4)))

    def _set_preset_target2(self, preset_name: str, target_file: Optional[str], target_name: Optional[str]) -> None:
        meta = self.preset_manager.get_preset_metadata(preset_name) or {}
        if not meta:
            return
        meta["target2_file"] = target_file or ""
        meta["target2_name"] = target_name or ""
        self.preset_manager.set_preset_metadata(preset_name, meta)

        if preset_name == self.preset_manager.active_preset_name:
            self._target2_req_id += 1
            req_id = self._target2_req_id
            self._retire_worker(self._target2_worker)
            self._target2_worker = None

            if target_file:
                base_u = meta.get("base_url", "")
                db_id = meta.get("database_id", "") or _resolve_db_id(self.squig_client, base_u)
                off_db = get_offline_db()
                t2_res = off_db.get_target(db_id, target_file, target_name)
                if t2_res and len(t2_res[0]) > 0 and len(t2_res[0]) == len(t2_res[1]):
                    self._apply_target2_curve_data(t2_res[0], t2_res[1], label=target_name or "Target 2")
                elif base_u:
                    worker = CurveFetchWorker(
                        client=self.squig_client,
                        base_url=base_u,
                        model=None,
                        target_mode=0,
                        target=None,
                        target2=SquigTarget(name=target_name or target_file, file_name=target_file),
                        parent=self,
                    )
                    def _on_t2_ready(t2_data: Tuple, rid: int = req_id, p_name: str = preset_name, lbl: str = target_name or "Target 2") -> None:
                        if rid != self._target2_req_id or self.preset_manager.active_preset_name != p_name:
                            return
                        t2f, t2s = t2_data
                        if t2f and t2s:
                            self._apply_target2_curve_data(t2f, t2s, label=lbl)
                    worker.sigTarget2Loaded.connect(_on_t2_ready)
                    self._target2_worker = worker
                    worker.start()
            else:
                self._apply_target2_curve_data(None, None)

    def _revert_db_combo_to_active(self) -> None:
        active_id = self._active_db_info.get("id")
        self.autoeq_db_combo.blockSignals(True)
        for i in range(self.autoeq_db_combo.count()):
            d = self.autoeq_db_combo.itemData(i)
            if isinstance(d, dict) and d.get("id") == active_id:
                self.autoeq_db_combo.setCurrentIndex(i)
                break
        self.autoeq_db_combo.blockSignals(False)

    def _refresh_active_database(self) -> None:
        if self._active_db_info:
            self._load_database_index(self._active_db_info, force_refresh=True)

    def _load_database_index(self, db_info: Dict[str, str], force_refresh: bool = False) -> None:
        base_url = db_info.get("base_url", "")
        if not base_url:
            return

        if self._autoeq_worker and self._autoeq_worker.isRunning():
            self._cancel_autoeq_optimization()

        self.autoeq_status_label.setText(f"Loading database {db_info.get('name', '')}...")
        self.autoeq_status_label.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        self.autoeq_db_combo.setEnabled(False)

        self._db_req_id += 1
        req_id = self._db_req_id
        self._retire_worker(self._db_worker)
        self._db_worker = None

        worker = DatabaseIndexWorker(self.squig_client, base_url, force_refresh, parent=self)
        worker.sigIndexLoaded.connect(
            lambda m, t, r, rid=req_id: self._on_database_index_loaded(m, t, r, rid)
        )
        worker.sigIndexFailed.connect(
            lambda err, rid=req_id: self._on_database_index_failed(err, rid)
        )
        self._db_worker = worker
        worker.start()

    def _on_database_index_loaded(
        self,
        models: List[SquigModel],
        targets: List[SquigTarget],
        rig: str,
        req_id: Optional[int] = None,
    ) -> None:
        if req_id is not None and req_id != self._db_req_id:
            return
        self.autoeq_db_combo.setEnabled(True)
        self._autoeq_models = models
        self._autoeq_targets = targets
        self.autoeq_model_combo.set_models(models)
        rig_clean = rig.replace("clone ", "").strip()
        self.autoeq_rig_badge.setText(f"Rig: {rig} [Isolated]")
        self.autoeq_rig_badge.setToolTip(
            f"Strict Coupler Isolation: Targets and measurements are locked to this {rig_clean} database to avoid measurement mismatch."
        )

        if hasattr(self, "autoeq_target_iem_combo"):
            self.autoeq_target_iem_combo.set_provider_info(
                self._active_db_info.get("base_url", ""),
                self._active_db_info.get("name", ""),
                rig,
            )
            self.autoeq_target_iem_combo.set_models(models)
            self.autoeq_target_iem_badge.setText(f"Rig: {rig_clean} [Matched]")
        self._populate_target_profiles()

        # Repopulate Targets (strictly scoped to this database)
        self.autoeq_target_combo.blockSignals(True)
        self.autoeq_target_combo.clear()
        for t in targets:
            label = f"[{t.category}] {t.name}" if t.category else t.name
            self.autoeq_target_combo.addItem(label, t)

        if targets:
            self.autoeq_target_combo.setCurrentIndex(0)
        self.autoeq_target_combo.blockSignals(False)

        # Check if we have a pending global model selection
        pending = self._pending_global_model
        self._pending_global_model = None

        if pending:
            self._suppress_curve_fetch = False
            matched_model: Optional[SquigModel] = None
            for m in models:
                if m.file_name.lower() == pending.file_name.lower():
                    matched_model = m
                    break
                elif m.name.lower() == pending.name.lower():
                    matched_model = m
                    break
                elif pending.name.lower() in m.display_name.lower():
                    matched_model = m
                    break

            self.autoeq_model_combo.set_selected_model(matched_model)

            # Match target first before fetching curves so both model and target fetch together
            if pending.targets and targets:
                self.autoeq_target_combo.blockSignals(True)
                for p_targ in pending.targets:
                    found = False
                    for t_idx, t in enumerate(targets):
                        if t.name.lower() == p_targ.lower():
                            self.autoeq_target_combo.setCurrentIndex(t_idx)
                            found = True
                            break
                    if found:
                        break
                self.autoeq_target_combo.blockSignals(False)

            if matched_model:
                self._fetch_selected_curves()
        elif self._pending_restore_model_file or self._pending_restore_target_file or self._suppress_curve_fetch:
            self._apply_pending_autoeq_restore()
        else:
            self.autoeq_model_combo.set_selected_model(None)

        self.autoeq_status_label.setText(f"Loaded {len(models)} models, {len(targets)} targets")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")

    def _on_database_index_failed(self, err: str, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._db_req_id:
            return
        self.autoeq_db_combo.setEnabled(True)
        # Ensure curve fetching is never left permanently suppressed on DB index failure
        self._pending_restore_model_file = None
        self._pending_restore_target_file = None
        self._pending_restore_skip_fetch = False
        self._suppress_curve_fetch = False
        self.autoeq_status_label.setText(f"Database error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _on_searchable_model_selected(self, model: SquigModel) -> None:
        self._fetch_selected_curves()

    def _on_autoeq_model_changed(self) -> None:
        self._fetch_selected_curves()

    def _on_autoeq_target_changed(self) -> None:
        self._fetch_selected_curves()

    def _fetch_selected_curves(self) -> None:
        if getattr(self, "_suppress_curve_fetch", False):
            return
        model: Optional[SquigModel] = self.autoeq_model_combo.currentData()
        chan: str = self.autoeq_chan_combo.currentData() or "avg"
        base_url = self._active_db_info.get("base_url", "")
        mode = getattr(self, "_autoeq_target_mode", 0)

        target: Optional[SquigTarget] = None
        target_iem: Optional[SquigModel] = None
        target_iem_chan: str = "avg"
        target_profile_name: str = ""

        if mode == 0:
            target = self.autoeq_target_combo.currentData() if hasattr(self, "autoeq_target_combo") else None
        elif mode == 1:
            target_iem = self.autoeq_target_iem_combo.currentData() if hasattr(self, "autoeq_target_iem_combo") else None
            target_iem_chan = self.autoeq_target_iem_chan_combo.currentData() or "avg" if hasattr(self, "autoeq_target_iem_chan_combo") else "avg"
        elif mode == 2:
            if hasattr(self, "autoeq_target_profile_combo"):
                target_profile_name = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()

        has_target = bool(target or target_iem or target_profile_name)
        if not base_url or not (model or has_target):
            return

        self._curve_req_id += 1
        req_id = self._curve_req_id
        self._retire_worker(self._curve_worker)
        self._curve_worker = None

        worker = CurveFetchWorker(
            client=self.squig_client,
            base_url=base_url,
            model=model,
            target_mode=mode,
            target=target,
            target_iem_model=target_iem,
            target_iem_chan=target_iem_chan,
            target_profile_name=target_profile_name,
            preset_manager=self.preset_manager,
            sample_rate=self.sample_rate,
            channel_mode=chan,
            parent=self,
        )
        worker.sigCurvesLoaded.connect(
            lambda m_data, t_data, rid=req_id: self._on_curves_loaded(m_data, t_data, rid)
        )
        worker.sigCurveFailed.connect(
            lambda err, rid=req_id: self._on_curves_failed(err, rid)
        )
        self._curve_worker = worker
        worker.start()

    def _on_curves_loaded(self, meas_data: Tuple, target_data: Tuple, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._curve_req_id:
            return
        mf, ms = meas_data
        tf, ts = target_data
        self._autoeq_meas_raw = (mf, ms)
        self._autoeq_target_raw = (tf, ts)

        model: Optional[SquigModel] = self.autoeq_model_combo.currentData()
        if model and hasattr(model, "display_name") and model.display_name:
            m_label = model.display_name
        elif hasattr(self, "autoeq_model_combo") and self.autoeq_model_combo.text().strip():
            m_label = self.autoeq_model_combo.text().strip()
        else:
            m_label = "Measurement" if (mf and len(mf) > 0) else ""

        mode = getattr(self, "_autoeq_target_mode", 0)
        if mode == 0:
            target = self.autoeq_target_combo.currentData() if hasattr(self, "autoeq_target_combo") else None
            if target and hasattr(target, "name") and target.name:
                t_label = target.name
            elif hasattr(self, "autoeq_target_combo") and self.autoeq_target_combo.currentText().strip():
                t_label = self.autoeq_target_combo.currentText().strip()
            else:
                t_label = "Target" if (tf and len(tf) > 0) else ""
        elif mode == 1:
            target_iem = self.autoeq_target_iem_combo.currentData() if hasattr(self, "autoeq_target_iem_combo") else None
            t_label = f"{target_iem.display_name} (Target)" if target_iem else "Target IEM"
            if target_iem and hasattr(self, "autoeq_target_iem_badge"):
                t_rig = getattr(target_iem, "rig", "") or self._active_db_info.get("rig", "")
                t_rig_clean = t_rig.replace("clone ", "").strip()
                is_same = getattr(target_iem, "base_url", "").rstrip("/") == self._active_db_info.get("base_url", "").rstrip("/")
                match_tag = "[Matched]" if is_same else "[External]"
                self.autoeq_target_iem_badge.setText(f"Rig: {t_rig_clean} {match_tag}")
        else:  # 2
            prof_name = ""
            if hasattr(self, "autoeq_target_profile_combo"):
                prof_name = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
            t_label = f"{prof_name} (Resultant)" if prof_name else "Target Resultant"

        eff_tf, eff_ts = self._get_effective_target_raw()
        if abs(self._autoeq_bass_boost_db) > 1e-4 or abs(self._autoeq_treble_tilt_db) > 1e-4:
            if t_label:
                t_label = f"{t_label} (Pref)"
        dm, dt, dr = self._compute_normalized_display_curves(mf, ms, eff_tf, eff_ts, self.eq_plot.bands)

        has_dm = bool(dm[0] and len(dm[0]) > 0)
        has_dt = bool(dt[0] and len(dt[0]) > 0)
        has_dr = bool(dr[0] and len(dr[0]) > 0)

        self.undergraph_meas_btn.blockSignals(True)
        self.undergraph_target_btn.blockSignals(True)
        self.undergraph_resultant_btn.blockSignals(True)
        self.undergraph_meas_btn.setChecked(has_dm)
        self.undergraph_target_btn.setChecked(has_dt)
        self.undergraph_resultant_btn.setChecked(has_dr)
        self.undergraph_meas_btn.blockSignals(False)
        self.undergraph_target_btn.blockSignals(False)
        self.undergraph_resultant_btn.blockSignals(False)

        self.eq_plot._is_meas_checked = has_dm
        self.eq_plot._is_target_checked = has_dt
        self.eq_plot._is_resultant_checked = has_dr

        is_sweeping = bool(getattr(self.eq_plot, "_is_sweeping", False))

        if has_dm:
            self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
            self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)
            if is_sweeping:
                self.eq_plot.set_measurement_overlay(dm[0], dm[1], label=m_label)
            else:
                self.eq_plot.animate_overlay_curve("meas", dm[0], dm[1], label=m_label)
        else:
            self._disp_meas_freqs = None
            self._disp_meas_spls = None
            if is_sweeping:
                self.eq_plot.set_measurement_overlay(None, None)
            else:
                self.eq_plot.animate_overlay_curve("meas", None, None)

        if has_dt:
            if is_sweeping:
                self.eq_plot.set_target_overlay(dt[0], dt[1], label=t_label)
            else:
                self.eq_plot.animate_overlay_curve("target", dt[0], dt[1], label=t_label)
        else:
            if is_sweeping:
                self.eq_plot.set_target_overlay(None, None)
            else:
                self.eq_plot.animate_overlay_curve("target", None, None)

        if has_dr:
            r_label = f"{m_label} + EQ" if m_label else "Resultant + EQ"
            if is_sweeping:
                self.eq_plot.set_resultant_overlay(dr[0], dr[1], label=r_label)
            else:
                self.eq_plot.animate_overlay_curve("resultant", dr[0], dr[1], label=r_label)
        else:
            if is_sweeping:
                self.eq_plot.set_resultant_overlay(None, None)
            else:
                self.eq_plot.animate_overlay_curve("resultant", None, None)

        self.eq_plot.set_overlays_visible(
            has_dm, has_dt, has_dr,
            show_target2=self.undergraph_target2_btn.isChecked(),
        )
        self._style_undergraph_toggle_buttons()

        if mf and tf:
            if "Fitted" not in self.autoeq_status_label.text() and "Fitting" not in self.autoeq_status_label.text():
                self.autoeq_status_label.setText("Traces aligned - ready to compute")
                self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")
        elif mf:
            if "Fitted" not in self.autoeq_status_label.text():
                self.autoeq_status_label.setText(f"Measurement loaded for '{m_label}'")
                self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")
        elif tf:
            if model:
                self.autoeq_status_label.setText(f"Measurement unavailable for '{m_label}'")
                self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; background: transparent;")
        else:
            self.autoeq_status_label.setText("No curve data found for this selection")
            self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; background: transparent;")

    def _on_curves_failed(self, err: str, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._curve_req_id:
            return
        self.autoeq_status_label.setText(f"Curve fetch error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; background: transparent;")

    def _on_overlay_toggled(self) -> None:
        self.eq_plot.set_overlays_visible(
            self.undergraph_meas_btn.isChecked(),
            self.undergraph_target_btn.isChecked(),
            self.undergraph_resultant_btn.isChecked(),
            show_target2=self.undergraph_target2_btn.isChecked(),
        )

    def _update_resultant_curve(self) -> None:
        if self._disp_meas_freqs is None or self._disp_meas_spls is None or len(self._disp_meas_freqs) == 0:
            self.eq_plot.set_resultant_overlay(None, None)
            return

        # Compute composite filter response across the measurement frequencies (preamp=0.0 so shape aligns with target)
        eq_response = compute_composite_response(
            self.eq_plot.bands, self._disp_meas_freqs, preamp_db=0.0, fs=self.sample_rate
        )
        resultant_spls = self._disp_meas_spls + eq_response

        meta = self.preset_manager.get_preset_metadata(self.preset_manager.active_preset_name)
        m_name = (meta.get("model_name") if meta else None)
        if not m_name:
            model = self.autoeq_model_combo.currentData()
            m_name = model.name if model else "Resultant"
        label = f"{m_name} + EQ"

        self.eq_plot.set_resultant_overlay(self._disp_meas_freqs, resultant_spls, label=label)
        self.eq_plot.set_overlays_visible(
            self.undergraph_meas_btn.isChecked(),
            self.undergraph_target_btn.isChecked(),
            self.undergraph_resultant_btn.isChecked(),
            show_target2=self.undergraph_target2_btn.isChecked(),
        )

    def _compute_autoeq(self) -> None:
        # If already optimizing, clicking stops/cancels optimization immediately
        if self._autoeq_worker and self._autoeq_worker.isRunning():
            self._cancel_autoeq_optimization()
            return

        mf, ms = self._autoeq_meas_raw
        tf, ts = self._autoeq_target_raw
        mode = getattr(self, "_autoeq_target_mode", 0)

        if not mf or not tf:
            model = self.autoeq_model_combo.currentData()
            if not model:
                self.autoeq_status_label.setText("Select a headphone or IEM model")
                self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                return

            if mode == 0:
                target = self.autoeq_target_combo.currentData() if hasattr(self, "autoeq_target_combo") else None
                if not target:
                    self.autoeq_status_label.setText("Select a target curve")
                    self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                    return
            elif mode == 1:
                target_iem = self.autoeq_target_iem_combo.currentData() if hasattr(self, "autoeq_target_iem_combo") else None
                if not target_iem:
                    self.autoeq_status_label.setText("Select a target IEM or headphone")
                    self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                    return
            elif mode == 2:
                prof_name = ""
                if hasattr(self, "autoeq_target_profile_combo"):
                    prof_name = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
                if not prof_name:
                    self.autoeq_status_label.setText("Select a profile to match")
                    self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                    return

            if not mf:
                if self._curve_worker and self._curve_worker.isRunning():
                    self.autoeq_status_label.setText("Downloading measurement curve...")
                    self.autoeq_status_label.setStyleSheet("color: #d99b38; font-size: 11px; background: transparent;")
                else:
                    m_disp = model.display_name if model else "this model"
                    self.autoeq_status_label.setText(f"Measurement unavailable for '{m_disp}'")
                    self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                return
            if not tf:
                if self._curve_worker and self._curve_worker.isRunning():
                    self.autoeq_status_label.setText("Downloading target curve...")
                    self.autoeq_status_label.setStyleSheet("color: #d99b38; font-size: 11px; background: transparent;")
                else:
                    self.autoeq_status_label.setText("Target response unavailable")
                    self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; font-weight: 500; background: transparent;")
                return

        num_bands = self.autoeq_bands_spin.value()
        use_bass_shelf = self.autoeq_bass_shelf_cb.isChecked()
        max_boost = self.autoeq_max_boost_spin.value()
        max_freq = self.autoeq_max_freq_spin.value()
        rig_type = str(self._active_db_info.get("rig", "") or "")

        self.autoeq_compute_btn.setEnabled(True)
        self.autoeq_compute_btn.setText("Stop optimization")
        self.autoeq_compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #1c1c1f;
                color: #e5484d;
                border: 1px solid #e5484d;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #27272a;
                color: #ededed;
                border-color: #e5484d;
            }
        """)
        self.autoeq_status_label.setText(f"Fitting {num_bands} bands...")
        self.autoeq_status_label.setStyleSheet("color: #a1a1aa; font-size: 11px; background: transparent;")
        if hasattr(self, "autoeq_rms_label"):
            self.autoeq_rms_label.setVisible(False)

        self._autoeq_req_id += 1
        req_id = self._autoeq_req_id
        self._retire_worker(self._autoeq_worker, use_cancel=True)
        self._autoeq_worker = None

        worker = AutoEQWorker(
            meas_freqs=mf,
            meas_spls=ms,
            target_freqs=tf,
            target_spls=ts,
            num_bands=num_bands,
            use_bass_shelf=use_bass_shelf,
            max_boost_db=max_boost,
            max_cut_db=12.0,
            max_freq=max_freq,
            parent=self,
            bass_boost_db=self._autoeq_bass_boost_db,
            treble_tilt_db=self._autoeq_treble_tilt_db,
            rig_type=rig_type,
            auto_prune=True,
        )
        worker.sigOptimized.connect(lambda res, rid=req_id: self._on_autoeq_optimized(res, rid))
        worker.sigFailed.connect(lambda err, rid=req_id: self._on_autoeq_failed(err, rid))
        self._autoeq_worker = worker
        worker.start()

    def _cancel_autoeq_optimization(self) -> None:
        self._autoeq_req_id += 1
        self._retire_worker(self._autoeq_worker, use_cancel=True)
        self._autoeq_worker = None
        self._curve_req_id += 1
        self._retire_worker(self._curve_worker)
        self._curve_worker = None
        self._reset_compute_btn_state()
        self.autoeq_status_label.setText("Optimization stopped")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")

    def _reset_compute_btn_state(self) -> None:
        self.autoeq_compute_btn.setEnabled(True)
        self.autoeq_compute_btn.setText("Compute AutoEQ")
        self.autoeq_compute_btn.setStyleSheet("")

    def _on_autoeq_optimized(self, result: OptimizationResult, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._autoeq_req_id:
            return
        self._record_undo_snapshot()
        self._reset_compute_btn_state()

        preamp_db, clean_bands = self._sanitize_eq_state(result.preamp_db, result.bands)

        # Compute refreshed resultant overlay using preference-adjusted target when present
        mf, ms = self._autoeq_meas_raw
        if getattr(result, "adjusted_target_freqs", None) and getattr(result, "adjusted_target_spl", None):
            tf, ts = result.adjusted_target_freqs, result.adjusted_target_spl
        else:
            tf, ts = self._get_effective_target_raw()
        dm, dt, dr = self._compute_normalized_display_curves(mf, ms, tf, ts, clean_bands)
        preset_name, autoeq_meta = self._build_current_autoeq_preset_meta()
        m_label = autoeq_meta.get("model_name") or "Measurement"
        t_label = autoeq_meta.get("target_name") or "Target"
        if abs(self._autoeq_bass_boost_db) > 1e-4 or abs(self._autoeq_treble_tilt_db) > 1e-4:
            t_label = f"{t_label} (Pref)"

        if dm[0]:
            self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
            self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)

        has_dm = bool(dm[0])
        has_dt = bool(dt[0])
        has_dr = bool(dr[0])

        self.undergraph_meas_btn.blockSignals(True)
        self.undergraph_target_btn.blockSignals(True)
        self.undergraph_resultant_btn.blockSignals(True)
        self.undergraph_meas_btn.setChecked(has_dm)
        self.undergraph_target_btn.setChecked(has_dt)
        self.undergraph_resultant_btn.setChecked(has_dr)
        self.undergraph_meas_btn.blockSignals(False)
        self.undergraph_target_btn.blockSignals(False)
        self.undergraph_resultant_btn.blockSignals(False)
        self.eq_plot._is_meas_checked = has_dm
        self.eq_plot._is_target_checked = has_dt
        self.eq_plot._is_resultant_checked = has_dr
        self._style_undergraph_toggle_buttons()

        # 1. Update plot canvas with smooth sweep transition & overlays
        self.eq_plot.animate_to_preset(
            clean_bands,
            preamp_db,
            target_data=(dt[0], dt[1], t_label) if has_dt else None,
            meas_data=(dm[0], dm[1], m_label) if has_dm else None,
            resultant_data=(dr[0], dr[1], f"{m_label} + EQ") if has_dr else None,
        )

        # 2. Update Preamp controls & headroom badge
        self._apply_preamp_value(preamp_db)

        # 3. Update bottom table
        self._populate_table(clean_bands)

        # 4. Immediate debounced Equalizer APO writer
        self.preset_manager.schedule_apo_export(
            preamp_db, clean_bands, bypassed=self.is_eq_bypassed
        )

        # 5. Save into presets manager with metadata without triggering redundant _on_preset_loaded
        self._suppress_preset_reload = True
        try:
            try:
                self.preset_manager.save_or_update_preset(
                    preset_name,
                    preamp_db,
                    clean_bands,
                    autoeq_metadata=autoeq_meta,
                    emit_changed=False,
                )
            except TypeError:
                self.preset_manager.save_or_update_preset(
                    preset_name,
                    preamp_db,
                    clean_bands,
                    autoeq_metadata=autoeq_meta,
                )
        finally:
            self._suppress_preset_reload = False

        self._last_state = (round(preamp_db, 2), [b.to_dict() for b in clean_bands])
        self._update_preamp_headroom_badge(preamp_db)

        # 6. Status message + tabular-mono Region RMS summary & pruned count
        pruned = int(getattr(result, "pruned_count", 0) or 0)
        pruned_txt = f" · {pruned} pruned" if pruned > 0 else ""
        status = (
            f"Fitted {len(clean_bands)} bands ({result.elapsed_ms:.0f} ms) · "
            f"RMS {result.final_rms_db:.2f} dB · Preamp {preamp_db:+.1f} dB{pruned_txt}"
        )
        self.autoeq_status_label.setText(status)
        self.autoeq_status_label.setStyleSheet(
            f"font-family: {MONO_FONT_FAMILY}; color: #ededed; font-weight: 500; font-size: 10px; background: transparent;"
        )

        rb = getattr(result, "region_rms_before", None) or {}
        ra = getattr(result, "region_rms_after", None) or {}
        if hasattr(self, "autoeq_rms_label"):
            if rb and ra:
                parts = []
                for key, lbl in (("bass", "Bass"), ("mid", "Mid"), ("treble", "Treble")):
                    if key in rb and key in ra:
                        parts.append(f"{lbl} {rb[key]:.1f}->{ra[key]:.1f}")
                if parts:
                    self.autoeq_rms_label.setText("  ".join(parts) + " dB")
                    self.autoeq_rms_label.setVisible(True)
                else:
                    self.autoeq_rms_label.setVisible(False)
            else:
                self.autoeq_rms_label.setVisible(False)

    def _on_autoeq_failed(self, err: str, req_id: Optional[int] = None) -> None:
        if req_id is not None and req_id != self._autoeq_req_id:
            return
        self._reset_compute_btn_state()
        self.autoeq_status_label.setText(f"Optimization error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #e5484d; font-size: 11px; background: transparent;")

    def _save_autoeq_profile(self) -> None:
        default_name, autoeq_meta = self._build_current_autoeq_preset_meta()

        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save AutoEQ profile",
            "Profile name",
            text=default_name,
        )
        if ok and name.strip():
            self.preset_manager.save_or_update_preset(
                name.strip(), self.preamp_spin.value(), self.eq_plot.bands, autoeq_metadata=autoeq_meta
            )
            self.autoeq_status_label.setText(f"Saved '{name.strip()}' to Profiles")
            self.autoeq_status_label.setStyleSheet("color: #4cc38a; font-size: 11px; background: transparent;")

    def _clear_autoeq_overlays(self) -> None:
        self._autoeq_req_id += 1
        self._retire_worker(self._autoeq_worker, use_cancel=True)
        self._autoeq_worker = None
        self._reset_compute_btn_state()
        self.eq_plot.clear_overlays()
        self._disp_meas_freqs = None
        self._disp_meas_spls = None
        self.undergraph_meas_btn.blockSignals(True)
        self.undergraph_target_btn.blockSignals(True)
        self.undergraph_resultant_btn.blockSignals(True)
        self.undergraph_target2_btn.blockSignals(True)
        self.undergraph_meas_btn.setChecked(False)
        self.undergraph_target_btn.setChecked(False)
        self.undergraph_resultant_btn.setChecked(False)
        self.undergraph_target2_btn.setChecked(False)
        self.undergraph_target2_btn.setVisible(False)
        self.undergraph_meas_btn.blockSignals(False)
        self.undergraph_target_btn.blockSignals(False)
        self.undergraph_resultant_btn.blockSignals(False)
        self.undergraph_target2_btn.blockSignals(False)
        self._style_undergraph_toggle_buttons()
        self.autoeq_status_label.setText("Overlays cleared")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px; background: transparent;")
        if hasattr(self, "autoeq_rms_label"):
            self.autoeq_rms_label.setVisible(False)


def main():
    QtCore.qInstallMessageHandler(_qt_message_filter)

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("pinf.cleanapo.desktop.1.0")
        except Exception:
            pass

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Clean APO")
    app.setOrganizationName("Clean APO")

    # Load and apply application icon
    icon_candidates = [
        os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "clean_apo.ico"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "clean_apo.ico"),
        os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "clean_apo.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "clean_apo.png"),
    ]
    app_icon = None
    for p in icon_candidates:
        if p and os.path.exists(p):
            app_icon = QtGui.QIcon(p)
            break
    if app_icon and not app_icon.isNull():
        app.setWindowIcon(app_icon)

    default_font = QtGui.QFont("Segoe UI", 9)
    default_font.setPointSize(9)
    app.setFont(default_font)

    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#000000"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ededed"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#000000"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#0b0b0c"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#121214"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#ededed"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ededed"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#121214"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#ededed"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#1c1c1f"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#ededed"))
    app.setPalette(palette)

    window = MainWindow()
    if app_icon and not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
