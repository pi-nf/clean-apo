"""
Clean APO - Production-Ready Native Desktop Application for Windows.
Minimalist Black & White aesthetic, real-time Equalizer APO integration,
sweeping profile animations, audio device loopback routing, and universal EQ controls.
"""

from __future__ import annotations
import os
import sys
import math
import urllib.parse
from typing import Optional, List, Dict, Tuple
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

def _qt_message_filter(mode, context, message):
    msg = str(message)
    if "setPointSize" in msg or "Point size" in msg or "pointSize" in msg:
        return
    if mode == QtCore.QtMsgType.QtFatalMsg:
        sys.stderr.write(f"Fatal: {message}\n")
        sys.exit(1)
    elif mode == QtCore.QtMsgType.QtCriticalMsg:
        sys.stderr.write(f"Critical: {message}\n")
    elif mode == QtCore.QtMsgType.QtWarningMsg:
        sys.stderr.write(f"Warning: {message}\n")

QtCore.qInstallMessageHandler(_qt_message_filter)

from biquad import EQBand, FilterType, compute_composite_response
from presets import (
    PresetManager,
    DEFAULT_APO_PATH,
    LOCAL_FALLBACK_PATH,
    get_apo_configurator_path,
    launch_apo_configurator,
    clean_apo_device_name,
    is_equalizer_apo_installed,
    get_bundled_apo_installer_path,
    launch_apo_installer,
    ensure_apo_root_config_include,
    grant_apo_directory_permissions,
    create_windows_shortcuts,
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
from squig_client import SquigClient, SquigModel, SquigTarget, GlobalSearchResult, DEFAULT_DATABASES
from autoeq_optimizer import AutoEQWorker, OptimizationResult, optimize_parametric_eq
from offline_db import get_offline_db, OfflineSyncWorker


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
            self.sigIndexLoaded.emit(models, targets, rig)
        except Exception as e:
            self.sigIndexFailed.emit(str(e))


class CurveFetchWorker(QtCore.QThread):
    sigCurvesLoaded = QtCore.pyqtSignal(object, object)  # (meas_f, meas_s), (targ_f, targ_s)
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
        self.preset_manager = preset_manager
        self.sample_rate = sample_rate
        self.channel_mode = channel_mode

    def run(self) -> None:
        try:
            mf, ms = ([], [])
            tf, ts = ([], [])
            if self.model:
                m_url = getattr(self.model, "base_url", "") or self.base_url
                mf, ms = self.client.fetch_iem_measurement(m_url, self.model, self.channel_mode)

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
                if self.target_profile_name and self.preset_manager:
                    p_data = self.preset_manager.presets.get(self.target_profile_name)
                    if p_data:
                        bands = [EQBand.from_dict(b) for b in p_data.get("bands", [])]
                        meta = p_data.get("autoeq") or {}
                        db_id = meta.get("database_id", "")
                        m_file = meta.get("model_file") or meta.get("model_name", "")
                        p_chan = meta.get("channel_mode", "avg")
                        p_base_url = meta.get("base_url", "") or self.base_url

                        meas_res = None
                        if db_id and m_file:
                            try:
                                from offline_db import get_offline_db
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

                        if meas_res and len(meas_res[0]) > 0:
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

            self.sigCurvesLoaded.emit((mf, ms), (tf, ts))
        except Exception as e:
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

    def focusInEvent(self, event: QtGui.QFocusEvent) -> None:
        super().focusInEvent(event)
        QtCore.QTimer.singleShot(0, self._select_numeric_only)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        QtCore.QTimer.singleShot(0, self._select_numeric_only)

    def _select_numeric_only(self) -> None:
        text = self.text()
        suffix = self.suffix()
        num_len = len(text) - len(suffix) if suffix and text.endswith(suffix) else len(text)
        self.lineEdit().setSelection(0, max(0, num_len))


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
    - Active (ON): Solid white circle with bold black band number.
    - Bypassed (OFF): Dark zinc circle with dashed border and muted grey band number.
    - Hover: Subtly glows with PointingHandCursor to indicate clickability.
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
        self.setStyleSheet("QPushButton { background: transparent; border: none; }")
        self._update_tooltip()
        self.toggled.connect(lambda _: self._update_tooltip())

    def _update_tooltip(self) -> None:
        state = "bypass" if self.isChecked() else "enable"
        self.setToolTip(f"Band {self.band_id} — Click to {state}")

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect_f = QtCore.QRectF(self.rect())
        center = rect_f.center()
        radius = 9.5

        if self.underMouse():
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 30)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(center, radius + 2.0, radius + 2.0)

        if self.isChecked():
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
            painter.setPen(QtGui.QPen(QtGui.QColor("#18181b"), 1.0))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#000000")))
        else:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(18, 18, 22)))
            painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 190), 1.4, QtCore.Qt.PenStyle.DashLine))
            painter.drawEllipse(center, radius, radius)
            painter.setPen(QtGui.QPen(QtGui.QColor(160, 160, 170)))

        font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        text_rect = QtCore.QRectF(rect_f)
        text_rect.translate(0, -1.0)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(self.band_id))


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

        # Windows 11 DWM rounded corners attribute (DWMWA_WINDOW_CORNER_PREFERENCE = 33, DWMWCP_ROUND = 2)
        pref = ctypes.c_int(2)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            33,
            ctypes.byref(pref),
            ctypes.sizeof(pref)
        )
    except Exception:
        pass


class WindowControlButton(QtWidgets.QPushButton):
    """
    Vector-rendered minimalist window control button matching Google Antigravity.
    Supports 'min', 'max', 'restore', and 'close' with crisp 1.2px anti-aliased geometry.
    """
    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind  # 'min', 'max', 'restore', 'close'
        self.setFixedSize(32, 26)
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
                bg_color = QtGui.QColor("#dc2626") if not is_pressed else QtGui.QColor("#b91c1c")
                painter.setBrush(QtGui.QBrush(bg_color))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2), 4.0, 4.0)
                icon_color = QtGui.QColor("#ffffff")
            else:
                icon_color = QtGui.QColor("#cccccc")
        else:
            if is_hovered:
                bg_color = QtGui.QColor(255, 255, 255, 22) if not is_pressed else QtGui.QColor(255, 255, 255, 36)
                painter.setBrush(QtGui.QBrush(bg_color))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRoundedRect(QtCore.QRectF(1, 1, self.width() - 2, self.height() - 2), 4.0, 4.0)
                icon_color = QtGui.QColor("#ffffff")
            else:
                icon_color = QtGui.QColor("#cccccc")

        pen = QtGui.QPen(icon_color)
        pen.setWidthF(1.2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        cx = self.width() / 2.0
        cy = self.height() / 2.0

        if self.kind == "min":
            w = 10.5
            painter.drawLine(QtCore.QPointF(cx - w / 2.0, cy), QtCore.QPointF(cx + w / 2.0, cy))

        elif self.kind == "max":
            size = 10.0
            r = QtCore.QRectF(cx - size / 2.0, cy - size / 2.0, size, size)
            painter.drawRoundedRect(r, 2.4, 2.4)

        elif self.kind == "restore":
            size = 8.2
            shift = 2.4

            bx = cx - size / 2.0 + shift / 2.0
            by = cy - size / 2.0 - shift / 2.0
            back_rect = QtCore.QRectF(bx, by, size, size)

            fx = cx - size / 2.0 - shift / 2.0
            fy = cy - size / 2.0 + shift / 2.0
            front_rect = QtCore.QRectF(fx, fy, size, size)

            painter.drawRoundedRect(back_rect, 2.0, 2.0)

            front_bg = QtGui.QColor("#18181b") if is_hovered else QtGui.QColor("#000000")
            painter.setBrush(QtGui.QBrush(front_bg))
            painter.drawRoundedRect(front_rect, 2.0, 2.0)

        elif self.kind == "close":
            half = 4.6
            painter.drawLine(QtCore.QPointF(cx - half, cy - half), QtCore.QPointF(cx + half, cy + half))
            painter.drawLine(QtCore.QPointF(cx - half, cy + half), QtCore.QPointF(cx + half, cy - half))

        painter.end()


class DialogTitleBar(QtWidgets.QFrame):
    """
    Sleek minimalist custom title bar for modal dialogs.
    Features drag-to-move, clean typography, and a minimalist close button.
    """
    def __init__(self, dialog: QtWidgets.QDialog, title_text: str, parent=None):
        super().__init__(parent)
        self.dialog = dialog
        self.setObjectName("dialogTitleBar")
        self.setFixedHeight(38)
        self.setStyleSheet("""
            QFrame#dialogTitleBar {
                background-color: #0c0c0e;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border-bottom: 1px solid #27272a;
            }
        """)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 8, 0)
        layout.setSpacing(8)

        title_lbl = QtWidgets.QLabel(title_text)
        title_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #ffffff; letter-spacing: 0.5px; background: transparent;")
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


def make_smooth_scroll(scroll_area: QtWidgets.QAbstractScrollArea) -> None:
    """Configures scroll area for continuous pixel-smooth scrolling."""
    if not scroll_area:
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
            if isinstance(obj, QtWidgets.QAbstractScrollArea):
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
        self.setPlaceholderText("Search model (e.g. 105, Hexa, HD600)...")
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
        completer_popup.setStyleSheet("""
            QListView {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                font-size: 12px;
                padding: 3px;
                outline: none;
            }
            QListView::item {
                padding: 5px 8px;
                border-radius: 3px;
                min-height: 22px;
                color: #f4f4f5;
            }
            QListView::item:hover, QListView::item:selected {
                background-color: #ffffff;
                color: #000000;
                font-weight: 600;
            }
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
        QtCore.QTimer.singleShot(0, self.selectAll)

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
        self.setPlaceholderText("Search reviewer database (e.g. Crinacle, Super Review)...")
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
        completer_popup.setStyleSheet("""
            QListView {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                font-size: 12px;
                padding: 3px;
                outline: none;
            }
            QListView::item {
                padding: 5px 8px;
                border-radius: 3px;
                min-height: 22px;
                color: #f4f4f5;
            }
            QListView::item:hover, QListView::item:selected {
                background-color: #ffffff;
                color: #000000;
                font-weight: 600;
            }
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
        QtCore.QTimer.singleShot(0, self.selectAll)

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

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_F4):
            if not self.completer.popup().isVisible():
                self._show_suggestions(force_all=False)
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
    Searchable model input specifically for Target IEM/headphone selection.
    Supports:
    - Same provider mode: searches models from the active reviewer database.
    - Global search mode: searches across all 43,000+ catalog models, prioritizing
      measurements from the current active provider at the very top.
    """
    def __init__(self, squig_client: Optional[SquigClient] = None, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.squig_client = squig_client
        self.setPlaceholderText("Search target IEM to clone (e.g. HD600, Chu II)...")
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

    def _do_global_search(self) -> None:
        if self.same_provider_only or not self.squig_client:
            return
        query = self.text().strip()
        if not query:
            return
        results = self.squig_client.search_global_catalog(query, limit=120)
        if not results:
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
            tag = "★ Same Provider" if is_same else r.provider_name
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
    Studio segmented pill selector with smooth cubic horizontal sliding indicator
    using QEasingCurve.Type.InOutCubic.
    """
    sigTabChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        tabs: List[str],
        parent: Optional[QtWidgets.QWidget] = None,
        height: int = 34,
        font_size: int = 10,
    ):
        super().__init__(parent)
        self.tabs = tabs
        self._current_index = 0
        self._pill_pos = 0.0
        self._font_size = font_size
        self.setFixedHeight(height)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(240)
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

        # Outer dark capsule container
        track_radius = 8.0
        painter.setPen(QtGui.QPen(QtGui.QColor("#27272a"), 1))
        painter.setBrush(QtGui.QColor("#18181b"))
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, w - 1.0, h - 1.0), track_radius, track_radius)

        # Sliding white pill indicator
        padding = 3.0
        pill_w = tab_w - (padding * 2.0)
        pill_h = h - (padding * 2.0)
        pill_x = padding + (self._pill_pos * tab_w)
        pill_y = padding
        pill_radius = max(3.0, track_radius - 2.0)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#f4f4f5"))  # Studio clean white/off-white
        painter.drawRoundedRect(QtCore.QRectF(pill_x, pill_y, pill_w, pill_h), pill_radius, pill_radius)

        # Tab text labels
        font = QtGui.QFont("Segoe UI", self._font_size, QtGui.QFont.Weight.Bold)
        font.setLetterSpacing(QtGui.QFont.SpacingType.AbsoluteSpacing, 0.5)
        painter.setFont(font)

        for i, text in enumerate(self.tabs):
            tab_rect = QtCore.QRectF(i * tab_w, 0, tab_w, h)
            dist = abs(self._pill_pos - i)
            if dist < 0.5:
                # Active: solid dark text on white pill
                painter.setPen(QtGui.QColor("#09090b"))
            else:
                # Inactive: muted grey text on dark background
                painter.setPen(QtGui.QColor("#a1a1aa"))
            painter.drawText(tab_rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

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
        self.setWindowTitle("Equalizer APO Device Setup")
        self.setFixedSize(460, 230)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
            QLabel {
                color: #e4e4e7;
                font-family: 'Segoe UI';
            }
            QPushButton {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #f4f4f5;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #27272a;
            }
            QPushButton#primaryLaunchBtn {
                background: #f4f4f5;
                color: #09090b;
                border: 1px solid #ffffff;
            }
            QPushButton#primaryLaunchBtn:hover {
                background: #e4e4e7;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Equalizer APO Device Setup")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        title = QtWidgets.QLabel("Equalizer APO is not yet installed on this device")
        title.setStyleSheet("font-size: 14px; font-weight: 700; color: #ffffff;")
        layout.addWidget(title)

        dev_lbl = QtWidgets.QLabel(f"Device: {device_name}")
        dev_lbl.setStyleSheet("font-size: 12px; font-weight: 600; color: #38bdf8;")
        dev_lbl.setWordWrap(True)
        layout.addWidget(dev_lbl)

        desc = QtWidgets.QLabel(
            "Windows requires Equalizer APO to be attached to each playback device. "
            "Use the Device Selector or Configurator to install Equalizer APO on this device."
        )
        desc.setStyleSheet("font-size: 11px; color: #a1a1aa; line-height: 1.4;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addStretch()

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        launch_btn = QtWidgets.QPushButton("Run APO Configurator")
        launch_btn.setObjectName("primaryLaunchBtn")
        launch_btn.clicked.connect(self._launch_configurator)
        btn_row.addWidget(launch_btn)

        dismiss_btn = QtWidgets.QPushButton("OK")
        dismiss_btn.clicked.connect(self.accept)
        btn_row.addWidget(dismiss_btn)

        layout.addLayout(btn_row)
        self.configurator_path = configurator_path

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)

    def _launch_configurator(self) -> None:
        if not self.configurator_path or not os.path.exists(self.configurator_path):
            if not is_equalizer_apo_installed():
                dlg = EqualizerApoInstallDialog(self.parent())
                dlg.exec()
                return
        ok, msg = launch_apo_configurator(self.configurator_path)
        if ok:
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(self, "Launch Error", f"Could not launch Configurator: {msg}")


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
        self.setWindowTitle("Equalizer APO Setup - Clean APO")
        self.setFixedSize(520, 360)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
            QLabel {
                color: #e4e4e7;
                font-family: 'Segoe UI';
            }
            QPushButton {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #f4f4f5;
                padding: 7px 16px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #27272a;
                border-color: #3f3f46;
            }
            QPushButton#primaryInstallBtn {
                background: #ffffff;
                color: #09090b;
                border: 1px solid #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryInstallBtn:hover {
                background: #e4e4e7;
            }
            QCheckBox {
                color: #71717a;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3f3f46;
                border-radius: 3px;
                background: #18181b;
            }
            QCheckBox::indicator:checked {
                background: #38bdf8;
                border-color: #38bdf8;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Equalizer APO Setup")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 20)
        layout.setSpacing(12)
        main_layout.addWidget(content)

        # Header Title
        title_box = QtWidgets.QHBoxLayout()
        title_box.setSpacing(8)

        self.header_title = QtWidgets.QLabel("Equalizer APO is Required")
        self.header_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        title_box.addWidget(self.header_title)
        title_box.addStretch()
        layout.addLayout(title_box)

        # Description
        self.desc_lbl = QtWidgets.QLabel(
            "Clean APO applies low-latency, real-time parametric EQ across all Windows audio "
            "(Spotify, YouTube, games, voice chat) via the Equalizer APO audio engine.\n\n"
            "Equalizer APO was not detected on this system. The official installer is bundled "
            "directly with Clean APO for instant setup."
        )
        self.desc_lbl.setStyleSheet("font-size: 11.5px; color: #a1a1aa; line-height: 1.45;")
        self.desc_lbl.setWordWrap(True)
        layout.addWidget(self.desc_lbl)

        # Features box
        features_box = QtWidgets.QFrame()
        features_box.setStyleSheet("background: #141417; border: 1px solid #27272a; border-radius: 6px; padding: 6px 10px;")
        f_layout = QtWidgets.QVBoxLayout(features_box)
        f_layout.setSpacing(5)

        b1 = QtWidgets.QLabel("• Official Equalizer APO x64 v1.4.2 bundled")
        b1.setStyleSheet("color: #d4d4d8; font-size: 11px;")
        f_layout.addWidget(b1)

        b2 = QtWidgets.QLabel("• Automatically connects 'clean_apo_generated.txt' into main config")
        b2.setStyleSheet("color: #d4d4d8; font-size: 11px;")
        f_layout.addWidget(b2)

        b3 = QtWidgets.QLabel("• Configurator will open so you can select your Headphones or Speakers")
        b3.setStyleSheet("color: #d4d4d8; font-size: 11px;")
        f_layout.addWidget(b3)
        layout.addWidget(features_box)

        # Status / Progress label
        self.status_msg = QtWidgets.QLabel("")
        self.status_msg.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        self.status_msg.setWordWrap(True)
        self.status_msg.setVisible(False)
        layout.addWidget(self.status_msg)

        layout.addStretch()

        # Checkbox: Don't ask again
        self.chk_dont_ask = QtWidgets.QCheckBox("Do not prompt again on startup")
        layout.addWidget(self.chk_dont_ask)

        # Button row
        self.btn_row = QtWidgets.QHBoxLayout()
        self.btn_row.setSpacing(8)

        self.install_btn = QtWidgets.QPushButton("Install Equalizer APO")
        self.install_btn.setObjectName("primaryInstallBtn")
        self.install_btn.clicked.connect(self._start_installation)
        self.btn_row.addWidget(self.install_btn)

        self.launch_cfg_btn = QtWidgets.QPushButton("Launch Configurator")
        self.launch_cfg_btn.setVisible(False)
        self.launch_cfg_btn.clicked.connect(self._launch_configurator)
        self.btn_row.addWidget(self.launch_cfg_btn)

        self.dismiss_btn = QtWidgets.QPushButton("Continue Offline")
        self.dismiss_btn.clicked.connect(self._on_dismiss)
        self.btn_row.addWidget(self.dismiss_btn)

        layout.addLayout(self.btn_row)

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(1200)
        self._poll_timer.timeout.connect(self._check_installation_polling)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)

    def _start_installation(self) -> None:
        ok, msg = launch_apo_installer()
        if not ok:
            self.status_msg.setText(f"Install Notice: {msg}")
            self.status_msg.setStyleSheet("color: #f87171; font-size: 11px; font-weight: 600;")
            self.status_msg.setVisible(True)
            return

        self.install_btn.setEnabled(False)
        self.install_btn.setText("Installer Running...")
        self.status_msg.setText("Setup launched! Please follow the installer prompts, check your audio devices, and click Finish.")
        self.status_msg.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        self.status_msg.setVisible(True)
        self._poll_timer.start()

    def _check_installation_polling(self) -> None:
        if is_equalizer_apo_installed():
            self._poll_timer.stop()
            self._on_installation_complete()

    def _on_installation_complete(self) -> None:
        ensure_apo_root_config_include(DEFAULT_APO_PATH)
        self.header_title.setText("Equalizer APO Ready!")
        self.desc_lbl.setText(
            "Equalizer APO was installed and linked to Clean APO successfully!\n\n"
            "Clean APO is now actively sending real-time parametric EQ directly to your Windows audio devices."
        )
        self.status_msg.setText("✓ Active & Connected (clean_apo_generated.txt)")
        self.status_msg.setStyleSheet("color: #4ade80; font-size: 12px; font-weight: 700;")
        self.status_msg.setVisible(True)

        self.install_btn.setVisible(False)
        self.launch_cfg_btn.setVisible(True)
        self.dismiss_btn.setText("Done")
        self.dismiss_btn.setObjectName("primaryInstallBtn")
        self.dismiss_btn.setStyleSheet("""
            background: #ffffff;
            color: #09090b;
            border: 1px solid #ffffff;
            font-weight: 700;
        """)
        self.sigInstalled.emit()

    def _launch_configurator(self) -> None:
        cfg = get_apo_configurator_path()
        if cfg:
            launch_apo_configurator(cfg)
        self.accept()

    def _on_dismiss(self) -> None:
        self._poll_timer.stop()
        self.accept()


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
# Minimalist Thin-Line Black & White QSS Stylesheet
# Divided strictly by razor-thin grey lines (#27272a); zero enclosing boxes.
# -----------------------------------------------------------------------------
BW_MINIMALIST_STYLESHEET = """
QMainWindow {
    background-color: transparent;
    border: none;
}

QWidget#centralWidget {
    background-color: #000000;
    border: 1px solid #27272a;
    border-radius: 10px;
}

QDialog {
    background-color: #09090b;
    border: 1px solid #27272a;
    border-radius: 10px;
}

QWidget {
    background-color: #000000;
    color: #ffffff;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    font-size: 12px;
}

/* Dividing lines: Exactly 1px borders dividing header, sidebar, and table */
QFrame#titleBarFrame {
    background-color: #000000;
    border: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    border-bottom: 1px solid #18181b;
}

QFrame#controlBarFrame {
    background-color: #000000;
    border: none;
    border-bottom: 1px solid #27272a;
}

QPushButton#topMenuBtn {
    background-color: transparent;
    color: #d4d4d8;
    border: none;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 500;
}
QPushButton#topMenuBtn:hover {
    background-color: #27272a;
    color: #ffffff;
}
QPushButton#topMenuBtn:pressed {
    background-color: #3f3f46;
    color: #ffffff;
}

QFrame#headerFrame {
    background-color: #000000;
    border: none;
    border-bottom: 1px solid #27272a;
    border-radius: 0px;
}

/* Window Control Buttons (integrated DAW top bar) */
QPushButton#windowControlBtn {
    background: transparent;
    border: none;
    border-radius: 4px;
    color: #a1a1aa;
    font-size: 13px;
    font-weight: 400;
    min-width: 30px;
    max-width: 30px;
    height: 26px;
    padding: 0px;
    margin: 0px 1px;
}
QPushButton#windowControlBtn:hover {
    background-color: #27272a;
    color: #ffffff;
}
QPushButton#windowCloseBtn {
    background: transparent;
    border: none;
    border-radius: 4px;
    color: #a1a1aa;
    font-size: 13px;
    font-weight: 400;
    min-width: 30px;
    max-width: 30px;
    height: 26px;
    padding: 0px;
    margin: 0px 1px;
}
QPushButton#windowCloseBtn:hover {
    background-color: #ef4444;
    color: #ffffff;
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
    background-color: #52525b;
}

/* PushButtons (Sharp Minimalist Black & White with Clean Rounded Corners) */
QPushButton {
    background-color: #121214;
    color: #ffffff;
    border: 1px solid #27272a;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #1e1e24;
    border-color: #52525b;
}
QPushButton:pressed {
    background-color: #000000;
    border-color: #ffffff;
}
QPushButton#primaryBtn {
    background-color: #ffffff;
    color: #000000;
    font-weight: 700;
    border: 1px solid #ffffff;
    border-radius: 5px;
}
QPushButton#primaryBtn:hover {
    background-color: #e4e4e7;
    border-color: #e4e4e7;
}
QPushButton#activeEqBtn {
    background-color: #ffffff;
    color: #000000;
    font-weight: 700;
    border: 1px solid #ffffff;
    border-radius: 5px;
    padding: 6px 10px;
    text-align: center;
}
QPushButton#bypassedEqBtn {
    background-color: #121214;
    color: #71717a;
    font-weight: 600;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    padding: 6px 10px;
    text-align: center;
}
QPushButton#dangerBtn {
    background-color: #121214;
    color: #f87171;
    border: 1px solid #450a0a;
    border-radius: 5px;
}
QPushButton#dangerBtn:hover {
    background-color: #ef4444;
    color: #ffffff;
    border-color: #ef4444;
}

QPushButton#presetCurvesBtn {
    background-color: #121215;
    color: #e4e4e7;
    border: 1px solid #27272a;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 5px 8px;
}
QPushButton#presetCurvesBtn:hover {
    background-color: #1e1e24;
    border-color: #38bdf8;
    color: #38bdf8;
}

/* Minimalist Table Delete Button ("X" turns red on hover) */
QPushButton.tableDeleteBtn {
    background-color: transparent;
    border: none;
    color: #71717a;
    font-size: 13px;
    font-weight: bold;
    padding: 0px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    border-radius: 4px;
}
QPushButton.tableDeleteBtn:hover {
    background-color: transparent;
    color: #ef4444;
}

/* Sidebar Tab Selector Buttons */
QPushButton#sidebarTabBtnActive {
    background-color: #18181b;
    color: #ffffff;
    font-weight: 700;
    border: 1px solid #27272a;
    border-bottom: 2px solid #ffffff;
    border-radius: 5px;
    padding: 6px 10px;
    text-align: center;
}
QPushButton#sidebarTabBtnInactive {
    background-color: #000000;
    color: #71717a;
    font-weight: 600;
    border: 1px solid #27272a;
    border-radius: 5px;
    padding: 6px 10px;
    text-align: center;
}
QPushButton#sidebarTabBtnInactive:hover {
    background-color: #0d0d10;
    color: #d4d4d8;
    border-color: #3f3f46;
}

/* Global Squiglink Search Launch Button */
QPushButton#globalSearchBtn {
    background-color: #121215;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid #3f3f46;
    border-radius: 5px;
    padding: 7px 10px;
    text-align: center;
    letter-spacing: 0.4px;
}
QPushButton#globalSearchBtn:hover {
    background-color: #27272a;
    border-color: #71717a;
    color: #ffffff;
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
    background-color: #09090b;
    color: #ffffff;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 22px;
}
QComboBox {
    background-color: #09090b;
    color: #ffffff;
    border: 1px solid #27272a;
    border-radius: 4px;
    padding: 4px 26px 4px 8px;
    min-height: 22px;
}
QComboBox:hover {
    border-color: #52525b;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #ffffff;
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
    background-color: #09090b;
}
QComboBox:editable QLineEdit {
    background: transparent;
    border: none;
    color: #ffffff;
    padding: 0px;
}
QComboBox QAbstractItemView {
    background-color: #09090b;
    color: #ffffff;
    selection-background-color: #ffffff;
    selection-color: #000000;
    border: 1px solid #27272a;
    border-radius: 4px;
}

/* Table Widget (No outer box, clean subtle inner grid) */
QTableWidget {
    background-color: #000000;
    alternate-background-color: #060608;
    border: none;
    border-radius: 0px;
    gridline-color: #18181b;
    selection-background-color: rgba(255, 255, 255, 18);
    selection-color: #ffffff;
}
QTableWidget QLineEdit,
QTableWidget QSpinBox,
QTableWidget QDoubleSpinBox,
QTableWidget QComboBox {
    border-radius: 0px;
}
QTableWidget QComboBox {
    border: 1px solid transparent;
    background-color: transparent;
}
QTableWidget QComboBox:hover {
    border: 1px solid transparent;
}
QTableWidget QComboBox:focus {
    border: 1px solid transparent;
}
QHeaderView::section {
    background-color: #000000;
    color: #a1a1aa;
    font-weight: 600;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #27272a;
    padding: 6px 4px;
}

/* Checkbox (Black & White with crisp Tick & perfect alignment) */
QCheckBox {
    spacing: 0px;
    margin: 0px;
    padding: 0px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #3f3f46;
    background-color: #000000;
    border-radius: 3px;
}
QCheckBox::indicator:hover {
    border-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #000000;
    border: 1px solid #ffffff;
    image: url('{CHECK_WHITE_PATH}');
}
QCheckBox::indicator:checked:hover {
    background-color: #09090b;
    border-color: #ffffff;
}

/* Context & Dropdown Menus (Antigravity DAW Style) */
QMenu {
    background-color: #121214;
    color: #f4f4f5;
    border: 1px solid #27272a;
    border-radius: 8px;
    padding: 4px;
    font-size: 12px;
}
QMenu::item {
    padding: 6px 24px 6px 28px;
    border-radius: 4px;
    background-color: transparent;
    color: #f4f4f5;
}
QMenu::item:selected {
    background-color: #27272a;
    color: #ffffff;
}
QMenu::item:disabled {
    color: #71717a;
}
QMenu::separator {
    height: 1px;
    background: #27272a;
    margin: 4px 6px;
}
QMenu::indicator {
    width: 14px;
    height: 14px;
    left: 8px;
}
QMenu::indicator:unchecked {
    image: none;
}
QMenu::indicator:checked {
    image: url('{CHECK_WHITE_PATH}');
}
QMenu::indicator:checked:selected {
    image: url('{CHECK_WHITE_PATH}');
}

/* Scrollbars: Pill Handle with Charcoal Track and Arrows */
QScrollBar:vertical {
    border: none;
    background: #18181b;
    width: 14px;
    margin: 16px 0 16px 0;
    border-radius: 0px;
}
QScrollBar::handle:vertical {
    background: #71717a;
    min-height: 28px;
    border-radius: 4px;
    margin: 2px 3px 2px 3px;
}
QScrollBar::handle:vertical:hover {
    background: #a1a1aa;
}
QScrollBar::handle:vertical:pressed {
    background: #d4d4d8;
}
QScrollBar::sub-line:vertical {
    border: none;
    background: #18181b;
    height: 16px;
    subcontrol-position: top;
    subcontrol-origin: margin;
}
QScrollBar::add-line:vertical {
    border: none;
    background: #18181b;
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
    background: #18181b;
    height: 14px;
    margin: 0 16px 0 16px;
    border-radius: 0px;
}
QScrollBar::handle:horizontal {
    background: #71717a;
    min-width: 28px;
    border-radius: 4px;
    margin: 3px 2px 3px 2px;
}
QScrollBar::handle:horizontal:hover {
    background: #a1a1aa;
}
QScrollBar::handle:horizontal:pressed {
    background: #d4d4d8;
}
QScrollBar::sub-line:horizontal {
    border: none;
    background: #18181b;
    width: 16px;
    subcontrol-position: left;
    subcontrol-origin: margin;
}
QScrollBar::add-line:horizontal {
    border: none;
    background: #18181b;
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

/* Preamp Slider */
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
    background: #ffffff;
    margin: 0px 4px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: none;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}
QSlider::handle:horizontal:hover {
    background: #e4e4e7;
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
            "QListWidget::item { padding: 8px 12px; margin: 1px 0px; color: #a1a1aa; border: none; }"
            "QListWidget::item:hover { color: #ffffff; background-color: #09090b; }"
            "QListWidget::item:selected { color: #ffffff; font-weight: bold; background-color: transparent; }"
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
        make_smooth_scroll(self)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, lambda: self.animate_to_current(immediate=True))

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
    Settings dialog allowing configuration of Equalizer APO output file path
    and verification of write permissions.
    """
    sigPathUpdated = QtCore.pyqtSignal(str)

    def __init__(self, current_path: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Settings - Clean APO")
        self.resize(560, 240)
        self.setStyleSheet(BW_MINIMALIST_STYLESHEET)
        self.current_path = current_path
        self._init_ui()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Clean APO Configuration")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 16, 20, 20)

        path_lbl = QtWidgets.QLabel("Equalizer APO Output File Path:")
        path_lbl.setStyleSheet("color: #a1a1aa; font-weight: 600;")
        layout.addWidget(path_lbl)

        path_box = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit(self.current_path)
        path_box.addWidget(self.path_edit)

        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_box.addWidget(browse_btn)
        layout.addLayout(path_box)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("font-weight: 500;")
        layout.addWidget(self.status_label)

        # Equalizer APO Status & Configurator/Installer launch row
        apo_installed = is_equalizer_apo_installed()
        apo_row = QtWidgets.QHBoxLayout()
        if apo_installed:
            stat_dot = QtWidgets.QLabel("●")
            stat_dot.setStyleSheet("color: #4ade80; font-size: 11px;")
            stat_lbl = QtWidgets.QLabel("Equalizer APO: Connected & Active")
            stat_lbl.setStyleSheet("color: #4ade80; font-weight: 600; font-size: 11.5px;")
            apo_row.addWidget(stat_dot)
            apo_row.addWidget(stat_lbl)
            apo_row.addStretch()
            apo_btn = QtWidgets.QPushButton("Launch Configurator")
            apo_btn.setFixedHeight(28)
            apo_btn.setToolTip("Open Equalizer APO Device Selector / Configurator")
            apo_btn.clicked.connect(lambda: launch_apo_configurator())
            apo_row.addWidget(apo_btn)
        else:
            stat_dot = QtWidgets.QLabel("○")
            stat_dot.setStyleSheet("color: #f59e0b; font-size: 11px;")
            stat_lbl = QtWidgets.QLabel("Equalizer APO: Not Installed")
            stat_lbl.setStyleSheet("color: #f59e0b; font-weight: 600; font-size: 11.5px;")
            apo_row.addWidget(stat_dot)
            apo_row.addWidget(stat_lbl)
            apo_row.addStretch()
            apo_btn = QtWidgets.QPushButton("Install Equalizer APO")
            apo_btn.setObjectName("primaryBtn")
            apo_btn.setFixedHeight(28)
            apo_btn.setToolTip("Install bundled Equalizer APO x64 installer")
            apo_btn.clicked.connect(self._open_installer_dialog)
            apo_row.addWidget(apo_btn)
        layout.addLayout(apo_row)

        btn_row = QtWidgets.QHBoxLayout()
        test_btn = QtWidgets.QPushButton("Test Permissions")
        test_btn.clicked.connect(self._test_permissions)
        btn_row.addWidget(test_btn)

        fix_btn = QtWidgets.QPushButton("Grant APO Permissions (UAC)")
        fix_btn.setToolTip("Request Administrator privileges via UAC to grant standard user write access to Equalizer APO folder")
        fix_btn.clicked.connect(self._fix_permissions)
        btn_row.addWidget(fix_btn)

        default_btn = QtWidgets.QPushButton("Reset to Default APO Path")
        default_btn.clicked.connect(lambda: self.path_edit.setText(DEFAULT_APO_PATH))
        btn_row.addWidget(default_btn)

        shortcut_btn = QtWidgets.QPushButton("Create Desktop Shortcut")
        shortcut_btn.setToolTip("Creates or updates Clean APO shortcuts on your Desktop and Start Menu")
        shortcut_btn.clicked.connect(self._create_shortcuts)
        btn_row.addWidget(shortcut_btn)

        btn_row.addStretch()

        save_btn = QtWidgets.QPushButton("Apply")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(save_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)
        main_layout.addWidget(content)

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select Equalizer APO Generated Config File",
            self.path_edit.text(),
            "Equalizer APO Config (*.txt);;All Files (*.*)",
        )
        if path:
            self.path_edit.setText(path)

    def _open_installer_dialog(self) -> None:
        dlg = EqualizerApoInstallDialog(self)
        dlg.exec()
        if is_equalizer_apo_installed():
            self.path_edit.setText(DEFAULT_APO_PATH)

    def _test_permissions(self) -> None:
        path = self.path_edit.text().strip()
        try:
            folder = os.path.dirname(path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                pass
            self.status_label.setText("Success: Write access confirmed!")
            self.status_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        except Exception as e:
            self.status_label.setText(f"Permission Error: {e}")
            self.status_label.setStyleSheet("color: #f87171; font-weight: bold;")

    def _fix_permissions(self) -> None:
        folder = os.path.dirname(self.path_edit.text().strip())
        ok, msg = grant_apo_directory_permissions(folder)
        if ok:
            self.status_label.setText("Permissions granted successfully via UAC!")
            self.status_label.setStyleSheet("color: #ffffff; font-weight: bold;")
            self._test_permissions()
        else:
            self.status_label.setText(f"Could not elevate: {msg}")
            self.status_label.setStyleSheet("color: #f87171; font-weight: bold;")

    def _create_shortcuts(self) -> None:
        ok, msg = create_windows_shortcuts()
        if ok:
            self.status_label.setText("Success: Clean APO shortcuts created on Desktop & Start Menu!")
            self.status_label.setStyleSheet("color: #4ade80; font-weight: bold;")
        else:
            self.status_label.setText(f"Shortcut Error: {msg}")
            self.status_label.setStyleSheet("color: #f87171; font-weight: bold;")

    def _on_save(self) -> None:
        new_path = self.path_edit.text().strip()
        if new_path:
            self.sigPathUpdated.emit(new_path)
        self.accept()


# -----------------------------------------------------------------------------
# Global Squiglink Search Dialog (43,000+ Measurements / 148 Databases)
# -----------------------------------------------------------------------------
class GlobalSearchDialog(QtWidgets.QDialog):
    """
    Global Model Search across all 148 Squiglink databases (43,600+ measurements).
    Displays IEM & Headphone models, measuring reviewer/provider, coupler rig,
    and all target curves provided by that reviewer.
    """
    def __init__(self, squig_client: SquigClient, initial_query: str = "", preset_manager: Optional[PresetManager] = None, parent=None):
        super().__init__(parent)
        self.client = squig_client
        self.preset_manager = preset_manager
        self.selected_result: Optional[GlobalSearchResult] = None
        self.load_role: str = "source"
        self._current_results: List[GlobalSearchResult] = []

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Global Model Search — 43,000+ Measurements (Squiglink)")
        self.resize(1020, 620)
        self.setMinimumSize(860, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #f4f4f5;
            }
            QLineEdit {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                padding: 7px 10px;
                font-size: 13px;
                selection-background-color: #ffffff;
                selection-color: #000000;
            }
            QLineEdit:focus {
                border: 1px solid #ffffff;
            }
            QTableWidget {
                background-color: #09090b;
                alternate-background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 4px;
                gridline-color: #1f1f23;
                color: #f4f4f5;
                font-size: 12px;
                selection-background-color: #27272a;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #18181b;
                color: #a1a1aa;
                border: none;
                border-bottom: 1px solid #27272a;
                border-right: 1px solid #27272a;
                padding: 6px 8px;
                font-weight: 700;
                font-size: 11px;
            }
            QPushButton {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #f4f4f5;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #27272a;
                border-color: #3f3f46;
            }
            QPushButton#primaryBtn {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryBtn:hover {
                background-color: #e4e4e7;
            }
            QPushButton#primaryBtn:disabled {
                background-color: #18181b;
                color: #52525b;
                border-color: #27272a;
            }
            QComboBox {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                padding: 5px 8px;
                font-size: 12px;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Global Model Search — 43,000+ Measurements (Squiglink)")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        main_layout.addWidget(content)

        # Header Title & Search Controls
        model_row = QtWidgets.QHBoxLayout()
        model_row.setSpacing(8)
        model_lbl = QtWidgets.QLabel("MODEL:")
        model_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #a1a1aa;")
        model_row.addWidget(model_lbl)
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search model or brand across all 148 reviewer databases (e.g. Meze 105, HD600, Zero:2, Aria)...")
        self.search_input.setClearButtonEnabled(True)
        if initial_query:
            self.search_input.setText(initial_query)
        model_row.addWidget(self.search_input, stretch=1)
        layout.addLayout(model_row)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(8)
        target_lbl = QtWidgets.QLabel("TARGET:")
        target_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #a1a1aa;")
        filter_row.addWidget(target_lbl)
        self.target_input = QtWidgets.QLineEdit()
        self.target_input.setPlaceholderText("Filter target curves (multiple at once, comma-separated e.g. Harman, KEMAR, DF, IEF)...")
        self.target_input.setClearButtonEnabled(True)
        filter_row.addWidget(self.target_input, stretch=1)

        self.type_filter_combo = QtWidgets.QComboBox()
        self.type_filter_combo.addItem("All Types", "all")
        self.type_filter_combo.addItem("Headphones", "headphones")
        self.type_filter_combo.addItem("IEMs", "iems")
        self.type_filter_combo.currentIndexChanged.connect(self._on_search_query_changed)
        filter_row.addWidget(self.type_filter_combo)
        layout.addLayout(filter_row)

        # Status & Stats Bar
        self.stats_label = QtWidgets.QLabel("Indexed 43,600+ measurements across 148 Squiglink databases.")
        self.stats_label.setStyleSheet("color: #71717a; font-size: 11px;")
        layout.addWidget(self.stats_label)

        # Results Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Model / Measurement",
            "Type",
            "Reviewer / Provider",
            "Coupler Rig",
            "Available Targets for Provider"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 240)
        self.table.setColumnWidth(2, 230)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        make_smooth_scroll(self.table)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, stretch=1)

        # Detail Box for Selected Result
        info_frame = QtWidgets.QFrame()
        info_frame.setStyleSheet("QFrame { background-color: #121215; border: 1px solid #27272a; padding: 6px; }")
        info_layout = QtWidgets.QVBoxLayout(info_frame)
        info_layout.setContentsMargins(10, 8, 10, 8)
        info_layout.setSpacing(4)

        self.selected_model_lbl = QtWidgets.QLabel("Select a measurement above to preview reviewer coupler and targets.")
        self.selected_model_lbl.setStyleSheet("font-size: 12px; font-weight: 700; color: #ffffff;")
        info_layout.addWidget(self.selected_model_lbl)

        self.targets_list_lbl = QtWidgets.QLabel("Targets: None")
        self.targets_list_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        self.targets_list_lbl.setWordWrap(True)
        info_layout.addWidget(self.targets_list_lbl)

        layout.addWidget(info_frame)

        # Bottom Actions
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(10)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.sync_db_btn = QtWidgets.QPushButton("Offline DB Sync")
        self.sync_db_btn.setToolTip("Download and manage complete offline measurement & target library (~60 MB)")
        self.sync_db_btn.clicked.connect(self._open_offline_sync)
        btn_layout.addWidget(self.sync_db_btn)

        btn_layout.addStretch()

        self.load_target_btn = QtWidgets.QPushButton("Load as Target IEM")
        self.load_target_btn.setEnabled(False)
        self.load_target_btn.setToolTip("Load this model into AutoEQ as the target response to clone")
        self.load_target_btn.clicked.connect(self._on_load_target_clicked)
        btn_layout.addWidget(self.load_target_btn)

        self.load_btn = QtWidgets.QPushButton("Load Selected Model into AutoEQ")
        self.load_btn.setObjectName("primaryBtn")
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

    def _on_search_query_changed(self) -> None:
        self._search_timer.start(120)

    def _execute_search(self) -> None:
        query = self.search_input.text().strip()
        target_query = self.target_input.text().strip()
        type_filter = self.type_filter_combo.currentData() or "all"
        results = self.client.search_global_catalog(query, target_query=target_query, limit=150)

        if type_filter != "all":
            results = [r for r in results if r.db_type.lower() == type_filter.lower()]

        # Sort: favorite providers first, then model name
        favs = self.preset_manager.get_favorite_databases() if self.preset_manager else []
        def is_fav(r: GlobalSearchResult) -> bool:
            return any(
                f.lower() in r.base_url.lower() or f.lower() in r.provider_name.lower()
                for f in favs
            )
        results.sort(key=lambda r: (0 if is_fav(r) else 1, r.name.lower()))

        self._current_results = results
        self.table.setRowCount(len(results))
        self.table.blockSignals(True)

        for row, res in enumerate(results):
            # Model Name
            m_text = res.full_name or res.file_name
            item_model = QtWidgets.QTableWidgetItem(m_text)
            item_model.setData(QtCore.Qt.ItemDataRole.UserRole, res)
            self.table.setItem(row, 0, item_model)

            # Type
            item_type = QtWidgets.QTableWidgetItem(res.db_type)
            item_type.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, item_type)

            # Provider (starred if favorite)
            res_fav = is_fav(res)
            prov_disp = f"★ {res.provider_name}" if res_fav else res.provider_name
            item_prov = QtWidgets.QTableWidgetItem(prov_disp)
            if res_fav:
                item_prov.setForeground(QtGui.QBrush(QtGui.QColor("#facc15")))
            self.table.setItem(row, 2, item_prov)

            # Rig
            item_rig = QtWidgets.QTableWidgetItem(res.rig)
            item_rig.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, item_rig)

            # Targets
            if res.targets:
                t_str = ", ".join(res.targets[:6])
                if len(res.targets) > 6:
                    t_str += f" (+{len(res.targets) - 6} more)"
            else:
                t_str = "Standard database targets"
            item_targets = QtWidgets.QTableWidgetItem(t_str)
            item_targets.setToolTip(", ".join(res.targets) if res.targets else t_str)
            self.table.setItem(row, 4, item_targets)

        self.table.blockSignals(False)

        count = len(results)
        filters_applied = []
        if query:
            filters_applied.append(f"model '{query}'")
        if target_query:
            filters_applied.append(f"targets '{target_query}'")
        if type_filter != "all":
            filters_applied.append(f"type '{type_filter}'")

        if filters_applied:
            self.stats_label.setText(f"Found {count} measurement(s) matching {', '.join(filters_applied)}.")
        else:
            self.stats_label.setText(f"Showing {count} measurements. Type above to filter all 43,600+ database entries.")

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
            self.selected_model_lbl.setText(
                f"Selected: {res.full_name or res.file_name}  •  Provider: {res.provider_name} ({res.rig})"
            )
            if res.targets:
                t_preview = "  •  ".join(res.targets)
                self.targets_list_lbl.setText(f"Available Targets ({len(res.targets)}): {t_preview}")
            else:
                self.targets_list_lbl.setText("Available Targets: Standard database target curves (will be fetched on load)")
        else:
            self.selected_result = None
            self.load_btn.setEnabled(False)
            self.load_target_btn.setEnabled(False)
            self.selected_model_lbl.setText("No measurement selected.")
            self.targets_list_lbl.setText("Targets: None")

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


class OfflineSyncDialog(QtWidgets.QDialog):
    """
    Minimalist studio modal dialog for syncing the offline measurement library.
    Allows downloading essential reviewer databases or the complete 43k+ model catalog
    into local SQLite storage for instant 0ms offline access.
    """
    def __init__(self, squig_client: SquigClient, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.client = squig_client
        self.offline_db = get_offline_db()
        target = parent
        while target is not None and not hasattr(target, "offline_sync_worker"):
            target = target.parent()
        self.main_win = target if target is not None else (parent.window() if parent else None)

        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
        self.setWindowTitle("Offline Database Sync")
        self.setFixedSize(620, 570)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 10px;
                color: #f4f4f5;
                font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            }
            QLabel {
                color: #e4e4e7;
                background: transparent;
            }
            QGroupBox {
                border: 1px solid #27272a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: 700;
                color: #a1a1aa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color: #f4f4f5;
            }
            QRadioButton {
                color: #d4d4d8;
                font-size: 11px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #52525b;
                border-radius: 7px;
                background: #18181b;
            }
            QRadioButton::indicator:checked {
                border-color: #ffffff;
                background-color: #ffffff;
            }
            QProgressBar {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 3px;
                height: 12px;
                text-align: center;
                font-size: 10px;
                color: #a1a1aa;
            }
            QProgressBar::chunk {
                background-color: #f4f4f5;
                border-radius: 2px;
            }
            QPushButton {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #f4f4f5;
                padding: 6px 14px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #27272a;
                border-color: #3f3f46;
                color: #ffffff;
            }
            QPushButton#primaryActionBtn {
                background: #f4f4f5;
                color: #09090b;
                border: 1px solid #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryActionBtn:hover {
                background: #ffffff;
                color: #000000;
            }
            QPushButton#stopActionBtn {
                background: #27272a;
                color: #f87171;
                border: 1px solid #3f3f46;
                font-weight: 700;
            }
            QPushButton#stopActionBtn:hover {
                background: #3f3f46;
                color: #ef4444;
            }
            QPushButton#dangerBtn {
                background: transparent;
                color: #f87171;
                border: 1px solid #7f1d1d;
                font-weight: 600;
            }
            QPushButton#dangerBtn:hover {
                background: #450a0a;
                border-color: #ef4444;
                color: #fca5a5;
            }
            QFrame#statCard {
                background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 4px;
                padding: 4px;
            }
            QFrame#statCard QLabel {
                background: transparent;
                border: none;
            }
        """)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        title_bar = DialogTitleBar(self, "Offline Database Sync")
        main_layout.addWidget(title_bar)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(14)
        main_layout.addWidget(content)

        desc = QtWidgets.QLabel(
            "Download measurement curves and targets to local SQLite storage. "
            "Enables instant 0ms AutoEQ loading and full offline capability without network dependence."
        )
        desc.setStyleSheet("font-size: 11px; color: #a1a1aa; line-height: 1.4;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Stats Row
        stats_frame = QtWidgets.QFrame()
        stats_frame.setObjectName("statCard")
        stats_layout = QtWidgets.QHBoxLayout(stats_frame)
        stats_layout.setContentsMargins(16, 12, 16, 12)
        stats_layout.setSpacing(0)

        # Col 1: Cached Models
        col1 = QtWidgets.QVBoxLayout()
        col1.setSpacing(4)
        col1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        meas_lbl = QtWidgets.QLabel("CACHED MODELS")
        meas_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        meas_lbl.setStyleSheet("font-size: 9px; font-weight: 800; color: #71717a; letter-spacing: 0.5px; background: transparent;")
        self.meas_stat_val = QtWidgets.QLabel("0 / 43,614")
        self.meas_stat_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.meas_stat_val.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff; background: transparent;")
        col1.addWidget(meas_lbl)
        col1.addWidget(self.meas_stat_val)
        stats_layout.addLayout(col1, 1)

        # Divider line 1 (Left of "CACHED TARGETS")
        div1 = QtWidgets.QFrame()
        div1.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        div1.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        div1.setStyleSheet("background-color: #27272a; width: 1px; max-width: 1px; border: none; margin: 2px 14px;")
        stats_layout.addWidget(div1)

        # Col 2: Cached Targets
        col2 = QtWidgets.QVBoxLayout()
        col2.setSpacing(4)
        col2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        t_lbl = QtWidgets.QLabel("CACHED TARGETS")
        t_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        t_lbl.setStyleSheet("font-size: 9px; font-weight: 800; color: #71717a; letter-spacing: 0.5px; background: transparent;")
        self.target_stat_val = QtWidgets.QLabel("0 / 1,875")
        self.target_stat_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.target_stat_val.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff; background: transparent;")
        col2.addWidget(t_lbl)
        col2.addWidget(self.target_stat_val)
        stats_layout.addLayout(col2, 1)

        # Divider line 2 (Right of "CACHED TARGETS")
        div2 = QtWidgets.QFrame()
        div2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        div2.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        div2.setStyleSheet("background-color: #27272a; width: 1px; max-width: 1px; border: none; margin: 2px 14px;")
        stats_layout.addWidget(div2)

        # Col 3: Database Size
        col3 = QtWidgets.QVBoxLayout()
        col3.setSpacing(4)
        col3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        size_lbl = QtWidgets.QLabel("DATABASE SIZE")
        size_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        size_lbl.setStyleSheet("font-size: 9px; font-weight: 800; color: #71717a; letter-spacing: 0.5px; background: transparent;")
        self.size_stat_val = QtWidgets.QLabel("0.0 MB")
        self.size_stat_val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.size_stat_val.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff; background: transparent;")
        col3.addWidget(size_lbl)
        col3.addWidget(self.size_stat_val)
        stats_layout.addLayout(col3, 1)

        layout.addWidget(stats_frame)

        # Scope Selection matching Profile List style
        scope_lbl = QtWidgets.QLabel("SYNC SCOPE")
        scope_lbl.setStyleSheet("font-size: 9px; font-weight: 800; color: #71717a; letter-spacing: 0.5px; background: transparent;")
        layout.addWidget(scope_lbl)

        self.scope_list = AnimatedPresetListWidget(self)
        self.scope_list.setFixedHeight(156)
        self.scope_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scope_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scope_list.setStyleSheet(
            "QListWidget { background-color: #060608; border: 1px solid #27272a; border-radius: 6px; outline: none; padding: 2px; }"
            "QListWidget::item { padding: 8px 12px; margin: 1px 0px; color: #a1a1aa; border: none; font-size: 11px; }"
            "QListWidget::item:hover { color: #ffffff; background-color: #121215; }"
            "QListWidget::item:selected { color: #ffffff; font-weight: bold; background-color: transparent; }"
        )

        item1 = QtWidgets.QListWidgetItem("Essential & Popular (Targets + Top ~15 Reviewers: Sharur, Super Review, Timmy, HBB)")
        item1.setSizeHint(QtCore.QSize(0, 46))
        item1.setData(QtCore.Qt.ItemDataRole.UserRole, "essential")
        self.scope_list.addItem(item1)

        item2 = QtWidgets.QListWidgetItem("Full Library (All 43,600+ Models across 148 Databases ~60 MB)")
        item2.setSizeHint(QtCore.QSize(0, 46))
        item2.setData(QtCore.Qt.ItemDataRole.UserRole, "full")
        self.scope_list.addItem(item2)

        item3 = QtWidgets.QListWidgetItem("Target Curves Only (All 1,875 Targets across all databases ~1 MB)")
        item3.setSizeHint(QtCore.QSize(0, 46))
        item3.setData(QtCore.Qt.ItemDataRole.UserRole, "targets_only")
        self.scope_list.addItem(item3)

        self.scope_list.setCurrentItem(item1)
        layout.addWidget(self.scope_list)

        # Progress Section
        prog_box = QtWidgets.QVBoxLayout()
        prog_box.setSpacing(4)
        self.phase_lbl = QtWidgets.QLabel("Ready to sync.")
        self.phase_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #ffffff;")
        prog_box.addWidget(self.phase_lbl)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_box.addWidget(self.progress_bar)

        self.detail_lbl = QtWidgets.QLabel("Click 'Start Sync' to download curves into local storage.")
        self.detail_lbl.setStyleSheet("font-size: 10px; color: #71717a;")
        self.detail_lbl.setWordWrap(True)
        prog_box.addWidget(self.detail_lbl)

        layout.addLayout(prog_box)
        layout.addStretch()

        # Background Download Disclaimer
        self.disclaimer_lbl = QtWidgets.QLabel(
            "Closing this window will not interrupt the download; it will continue in the background."
        )
        self.disclaimer_lbl.setStyleSheet("font-size: 10px; color: #a1a1aa; font-style: italic;")
        self.disclaimer_lbl.setWordWrap(True)
        layout.addWidget(self.disclaimer_lbl)

        # Action Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.delete_btn = QtWidgets.QPushButton("Delete Database")
        self.delete_btn.setObjectName("dangerBtn")
        self.delete_btn.setToolTip("Delete local database cache to free up storage or re-download from scratch")
        self.delete_btn.clicked.connect(self._delete_database)
        btn_row.addWidget(self.delete_btn)

        btn_row.addStretch()

        self.start_btn = QtWidgets.QPushButton("Start Sync")
        self.start_btn.setObjectName("primaryActionBtn")
        self.start_btn.clicked.connect(self._toggle_sync)
        btn_row.addWidget(self.start_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.close_btn)

        layout.addLayout(btn_row)

        self._refresh_stats()

        # Check if main window already has an active background worker
        if self.main_win and getattr(self.main_win, "offline_sync_worker", None):
            w = self.main_win.offline_sync_worker
            if w.isRunning():
                self._attach_worker(w)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)
        QtCore.QTimer.singleShot(0, self._adjust_scope_list_height)

    def _adjust_scope_list_height(self) -> None:
        fm = self.scope_list.fontMetrics()
        row_h = max(44, fm.height() + 24)
        for i in range(self.scope_list.count()):
            self.scope_list.item(i).setSizeHint(QtCore.QSize(0, row_h))
        needed_h = (row_h * self.scope_list.count()) + (self.scope_list.frameWidth() * 2) + 16
        self.scope_list.setFixedHeight(max(156, needed_h))

    def _refresh_stats(self) -> None:
        stats = self.offline_db.get_stats()
        self.meas_stat_val.setText(f"{stats['measurements']:,} / 43,614")
        self.target_stat_val.setText(f"{stats['targets']:,} / 1,875")
        self.size_stat_val.setText(f"{stats['size_mb']} MB")

    def _delete_database(self) -> None:
        worker = getattr(self.main_win, "offline_sync_worker", None) if self.main_win else None
        if worker and worker.isRunning():
            QtWidgets.QMessageBox.warning(
                self,
                "Sync Active",
                "Please stop the current sync before deleting the database.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Database",
            "Are you sure you want to delete all cached measurements and targets?\n\n"
            "This will delete your local offline database cache to free up storage, allowing you to re-download from scratch.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.offline_db.clear_all()
            self._refresh_stats()
            self.progress_bar.setValue(0)
            self.phase_lbl.setText("Database deleted.")
            self.detail_lbl.setText("All local cached curves and targets have been cleared.")
            if self.main_win and hasattr(self.main_win, "_update_offline_db_badge"):
                self.main_win._update_offline_db_badge()

    def _attach_worker(self, worker: OfflineSyncWorker) -> None:
        worker.sigPhaseChanged.connect(self._on_phase_changed)
        worker.sigProgress.connect(self._on_progress)
        worker.sigStatsUpdated.connect(self._on_stats_updated)
        worker.sigFinished.connect(self._on_finished)
        worker.sigError.connect(self._on_error)

        self.start_btn.setText("Stop")
        self.start_btn.setObjectName("stopActionBtn")
        self.start_btn.setStyleSheet("background: #27272a; color: #f87171; border: 1px solid #3f3f46; font-weight: 700;")
        self.phase_lbl.setText("Syncing in progress...")
        self.scope_list.setEnabled(False)

    def _toggle_sync(self) -> None:
        worker = getattr(self.main_win, "offline_sync_worker", None) if self.main_win else None
        if worker and worker.isRunning():
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Stopping...")
            QtWidgets.QApplication.processEvents()
            worker.cancel()
            worker.wait(800)
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start Sync")
            self.start_btn.setObjectName("primaryActionBtn")
            self.start_btn.setStyleSheet("")
            self.phase_lbl.setText("Sync stopped.")
            self.scope_list.setEnabled(True)
            self._refresh_stats()
            if self.main_win and hasattr(self.main_win, "_update_offline_db_badge"):
                self.main_win._update_offline_db_badge()
            return

        cur_item = self.scope_list.currentItem()
        scope = cur_item.data(QtCore.Qt.ItemDataRole.UserRole) if cur_item else "essential"

        # Immediate responsive feedback
        self.start_btn.setText("Stop")
        self.start_btn.setObjectName("stopActionBtn")
        self.start_btn.setStyleSheet("background: #27272a; color: #f87171; border: 1px solid #3f3f46; font-weight: 700;")
        self.phase_lbl.setText("Initializing sync...")
        self.detail_lbl.setText("Connecting to database servers...")
        self.scope_list.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        new_worker = OfflineSyncWorker(self.client, scope=scope, max_workers=24, parent=self.main_win)
        if self.main_win:
            self.main_win.offline_sync_worker = new_worker
            new_worker.sigProgress.connect(self.main_win._on_bg_sync_progress)
            new_worker.sigFinished.connect(self.main_win._on_bg_sync_finished)

        self._attach_worker(new_worker)
        new_worker.start()

    def _on_phase_changed(self, phase: str) -> None:
        self.phase_lbl.setText(phase)

    def _on_progress(self, current: int, total: int, status: str, item_name: str) -> None:
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.detail_lbl.setText(f"{status} • Current: {item_name}")

    def _on_stats_updated(self, meas: int, targets: int, size_mb: float) -> None:
        self.meas_stat_val.setText(f"{meas:,} / 43,614")
        self.target_stat_val.setText(f"{targets:,} / 1,875")
        self.size_stat_val.setText(f"{size_mb} MB")

    def _on_finished(self, new_saved: int, total_meas: int, total_targets: int) -> None:
        self.start_btn.setText("Start Sync")
        self.start_btn.setObjectName("primaryActionBtn")
        self.start_btn.setStyleSheet("")
        self.scope_list.setEnabled(True)
        self.progress_bar.setValue(100)
        self.phase_lbl.setText("Sync complete! Curves available offline.")
        self.phase_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #ffffff;")
        self.detail_lbl.setText(f"Successfully cached {new_saved:,} new curves in local SQLite storage.")
        self._refresh_stats()

    def _on_error(self, err: str) -> None:
        self.start_btn.setText("Start Sync")
        self.start_btn.setObjectName("primaryActionBtn")
        self.start_btn.setStyleSheet("")
        self.scope_list.setEnabled(True)
        self.phase_lbl.setText(f"Error: {err}")
        self.phase_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #f87171;")


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
        self.setWindowTitle(f"Target & Measurement — {preset_name}")
        self.resize(640, 560)
        self.setMinimumSize(560, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 8px;
                color: #f4f4f5;
            }
            QLabel {
                color: #f4f4f5;
                background: transparent;
            }
            QLineEdit {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                padding: 6px 10px;
                font-size: 12px;
                selection-background-color: #ffffff;
                selection-color: #000000;
            }
            QLineEdit:focus {
                border: 1px solid #38bdf8;
            }
            QComboBox {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #ffffff;
                padding: 6px 24px 6px 10px;
                font-size: 12px;
            }
            QComboBox:hover {
                border-color: #3f3f46;
            }
            QComboBox:focus {
                border-color: #38bdf8;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left-width: 0px;
            }
            QComboBox QAbstractItemView {
                background-color: #09090b;
                border: 1px solid #27272a;
                color: #ffffff;
                selection-background-color: #ffffff;
                selection-color: #000000;
                outline: none;
                padding: 2px;
            }
            QPushButton {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #f4f4f5;
                padding: 7px 14px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #27272a;
                border-color: #3f3f46;
            }
            QPushButton#primaryBtn {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryBtn:hover {
                background-color: #e4e4e7;
            }
            QPushButton#dangerBtn {
                background-color: #1c1012;
                color: #f87171;
                border: 1px solid #7f1d1d;
            }
            QPushButton#dangerBtn:hover {
                background-color: #ef4444;
                color: #ffffff;
                border-color: #ef4444;
            }
        """)

        self._init_ui()
        self._populate_providers()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._db_worker and self._db_worker.isRunning():
            self._db_worker.quit()
            self._db_worker.wait(300)
        super().closeEvent(event)

    def _init_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = DialogTitleBar(self, f"Target & Measurement Reference — {self.preset_name}")
        main_layout.addWidget(title_bar)

        # Body container
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(18, 14, 18, 16)
        layout.setSpacing(12)

        # Top banner describing the target & measurement concept
        banner_box = QtWidgets.QFrame()
        banner_box.setStyleSheet("""
            QFrame {
                background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 6px 10px;
            }
        """)
        banner_layout = QtWidgets.QVBoxLayout(banner_box)
        banner_layout.setContentsMargins(4, 4, 4, 4)
        banner_layout.setSpacing(2)

        p_lbl = QtWidgets.QLabel(f"EQ Profile: <span style='color: #ffffff; font-weight: 700;'>{self.preset_name}</span>")
        p_lbl.setStyleSheet("font-size: 13px; color: #a1a1aa;")
        banner_layout.addWidget(p_lbl)

        p_desc = QtWidgets.QLabel(
            "Attach a headphone measurement and compensation target to view the raw curve, "
            "reference target, and live resultant response alongside your EQ filters."
        )
        p_desc.setStyleSheet("font-size: 11px; color: #71717a;")
        p_desc.setWordWrap(True)
        banner_layout.addWidget(p_desc)
        layout.addWidget(banner_box)

        # Provider / Reviewer Database selection
        sec_prov = QtWidgets.QVBoxLayout()
        sec_prov.setSpacing(4)
        prov_head = QtWidgets.QHBoxLayout()
        prov_lbl = QtWidgets.QLabel("PROVIDER / REVIEWER DATABASE:")
        prov_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        prov_head.addWidget(prov_lbl)
        prov_head.addStretch()

        self.btn_global_search = QtWidgets.QPushButton("Search All (43k+)")
        self.btn_global_search.setStyleSheet("font-size: 11px; padding: 2px 8px; font-weight: normal;")
        self.btn_global_search.setToolTip("Search across 43,000+ measurements from 148 reviewers to find your model")
        self.btn_global_search.clicked.connect(self._open_global_search)
        prov_head.addWidget(self.btn_global_search)
        sec_prov.addLayout(prov_head)

        self.provider_combo = SearchableDatabaseInput(self)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        sec_prov.addWidget(self.provider_combo)

        rig_box = QtWidgets.QHBoxLayout()
        rig_box.setSpacing(6)
        self.rig_badge = QtWidgets.QLabel("Rig: IEC 60318-4 (711) [Coupler Isolated]")
        self.rig_badge.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: 600;")
        rig_box.addWidget(self.rig_badge)
        rig_box.addStretch()
        sec_prov.addLayout(rig_box)

        consistency_note = QtWidgets.QLabel(
            "Strict Rig Consistency: Measurement and target are locked to this provider's coupler to guarantee alignment."
        )
        consistency_note.setStyleSheet("font-size: 10px; color: #71717a;")
        sec_prov.addWidget(consistency_note)
        layout.addLayout(sec_prov)

        # Measurement / Model selection
        sec_meas = QtWidgets.QVBoxLayout()
        sec_meas.setSpacing(4)
        meas_lbl = QtWidgets.QLabel("HEADPHONE / IEM MEASUREMENT:")
        meas_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        sec_meas.addWidget(meas_lbl)

        meas_row = QtWidgets.QHBoxLayout()
        meas_row.setSpacing(6)

        self.model_input = SearchableModelInput(self)
        self.model_input.sigModelSelected.connect(self._on_model_selected)
        meas_row.addWidget(self.model_input, stretch=3)

        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItem("Avg", "avg")
        self.chan_combo.addItem("L", "L")
        self.chan_combo.addItem("R", "R")
        self.chan_combo.setToolTip("Channel: Average (L+R), Left (L), or Right (R)")
        self.chan_combo.setFixedWidth(80)
        meas_row.addWidget(self.chan_combo, stretch=1)
        sec_meas.addLayout(meas_row)
        layout.addLayout(sec_meas)

        # Primary Target Curve selection
        sec_targ = QtWidgets.QVBoxLayout()
        sec_targ.setSpacing(4)
        targ_lbl = QtWidgets.QLabel("PRIMARY TARGET CURVE:")
        targ_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        sec_targ.addWidget(targ_lbl)

        self.target_combo = QtWidgets.QComboBox()
        sec_targ.addWidget(self.target_combo)
        layout.addLayout(sec_targ)

        # Secondary Target Curve selection (optional)
        sec_targ2 = QtWidgets.QVBoxLayout()
        sec_targ2.setSpacing(4)
        targ2_lbl = QtWidgets.QLabel("SECONDARY TARGET CURVE (OPTIONAL):")
        targ2_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        sec_targ2.addWidget(targ2_lbl)

        self.target2_combo = QtWidgets.QComboBox()
        self.target2_combo.addItem("(None)", None)
        sec_targ2.addWidget(self.target2_combo)
        layout.addLayout(sec_targ2)

        # Status Label
        self.status_label = QtWidgets.QLabel("Select a provider, model, and target to attach.")
        self.status_label.setStyleSheet("font-size: 11px; color: #71717a;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # Action Buttons row
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.cancel_btn)

        has_curves = bool(
            self.existing_meta and (
                self.existing_meta.get("model_file") or
                self.existing_meta.get("target_file") or
                self.existing_meta.get("is_autoeq")
            )
        )
        self.remove_btn = QtWidgets.QPushButton("Remove Curves")
        self.remove_btn.setObjectName("dangerBtn")
        self.remove_btn.setEnabled(has_curves)
        self.remove_btn.setToolTip("Remove reference curves from this profile")
        self.remove_btn.clicked.connect(self._on_remove)
        btn_row.addWidget(self.remove_btn)

        btn_row.addStretch()

        self.apply_btn = QtWidgets.QPushButton("Apply Curves")
        self.apply_btn.setObjectName("primaryBtn")
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
                    "id": f"{c.get('username') or p_name.lower().replace(' ', '_')}_{cat_type.lower()}",
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
            if db.get("id") in favs:
                name = f"★ {name}"
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
        self.status_label.setText(f"Loading {db_data.get('name', '')} catalog...")
        self.status_label.setStyleSheet("color: #71717a; font-size: 11px;")

        if self._db_worker and self._db_worker.isRunning():
            self._db_worker.quit()
            self._db_worker.wait(300)

        self._db_worker = DatabaseIndexWorker(self.client, base_url, parent=self)
        self._db_worker.sigIndexLoaded.connect(self._on_db_index_loaded)
        self._db_worker.sigIndexFailed.connect(self._on_db_index_failed)
        self._db_worker.start()

    def _on_db_index_loaded(self, models: List[SquigModel], targets: List[SquigTarget], rig: str) -> None:
        self.model_input.setEnabled(True)
        self.target_combo.setEnabled(True)
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
        self.status_label.setText(f"Loaded {len(models)} models and {len(targets)} targets from {db_name}.")
        self.status_label.setStyleSheet("color: #71717a; font-size: 11px;")

    def _on_db_index_failed(self, err: str) -> None:
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
            self._on_db_index_loaded(fallback_models, fallback_targets, db_data.get("rig", "IEC 711"))
        else:
            self.model_input.setEnabled(True)
            self.target_combo.setEnabled(True)
            self.status_label.setText(f"Notice: Offline or error loading index: {err}")
            self.status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _on_model_selected(self, model: SquigModel) -> None:
        if model:
            self.status_label.setText(f"Selected: {model.display_name}")
            self.status_label.setStyleSheet("color: #ffffff; font-size: 11px;")

    def _open_global_search(self) -> None:
        initial_q = self.model_input.text().strip()
        dlg = GlobalSearchDialog(self.client, initial_query=initial_q, preset_manager=self.preset_manager, parent=self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.selected_result:
            res: GlobalSearchResult = dlg.selected_result
            res_url = res.base_url.rstrip("/")
            found_idx = -1
            for i in range(self.provider_combo.count()):
                d = self.provider_combo.itemData(i)
                if isinstance(d, dict) and d.get("base_url", "").rstrip("/") == res_url:
                    found_idx = i
                    break
            if found_idx == -1:
                new_info = {
                    "id": res.provider_name.lower().replace(" ", "_"),
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

        db_id = provider_data.get("id", "")
        db_name = provider_data.get("name", "")
        base_url = provider_data.get("base_url", "")
        rig = provider_data.get("rig", "")
        chan = self.chan_combo.currentData() or "avg"

        meta = {
            "is_autoeq": False,
            "database_id": db_id,
            "database_name": db_name,
            "base_url": base_url,
            "model_name": model.name if model else "",
            "model_file": model.file_name if model else "",
            "target_name": target.name if target else "",
            "target_file": target.file_name if target else (target.name if target else ""),
            "target2_name": target2.name if target2 else "",
            "target2_file": target2.file_name if target2 else "",
            "channel_mode": chan,
            "rig": rig,
        }

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
        self._autoeq_worker: Optional[AutoEQWorker] = None
        self._pending_global_model: Optional[GlobalSearchResult] = None
        self._pending_restore_model_file: Optional[str] = None
        self._pending_restore_target_file: Optional[str] = None

        self._db_switch_timer = QtCore.QTimer(self)
        self._db_switch_timer.setSingleShot(True)
        self._db_switch_timer.setInterval(100)
        self._db_switch_timer.timeout.connect(self._do_autoeq_db_changed)

        # Preamp cubic smoothing animation
        self._preamp_anim = QtCore.QVariantAnimation(self)
        self._preamp_anim.setDuration(320)
        self._preamp_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._preamp_anim.valueChanged.connect(self._on_preamp_anim_step)

        # Device hotplug monitor timer (checks every 2.5s)
        self._device_hotplug_timer = QtCore.QTimer(self)
        self._device_hotplug_timer.setInterval(2500)
        self._device_hotplug_timer.timeout.connect(self._check_device_hotplug)
        self._device_hotplug_timer.start()

        # Undo / Redo Parameter History Stack (DAW-grade parameter undo)
        self._undo_stack: List[Tuple[float, List[Dict[str, Any]]]] = []
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
        QtWidgets.QApplication.instance().installEventFilter(self)

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

        # Check Equalizer APO presence on startup and prompt bundled installer if missing
        QtCore.QTimer.singleShot(750, self._check_equalizer_apo_on_launch)

        # Start background loopback audio worker
        self.spectrum_worker.start()

        # Trigger initial database loading in background (reads disk cache instantly, network fallback)
        self._load_database_index(self._active_db_info, force_refresh=False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._device_hotplug_timer.stop()
        self._db_switch_timer.stop()
        if self._db_worker and self._db_worker.isRunning():
            self._db_worker.terminate()
            self._db_worker.wait(400)
        if self._curve_worker and self._curve_worker.isRunning():
            self._curve_worker.terminate()
            self._curve_worker.wait(400)
        if self._autoeq_worker and self._autoeq_worker.isRunning():
            self._autoeq_worker.terminate()
            self._autoeq_worker.wait(400)
        if self.offline_sync_worker and self.offline_sync_worker.isRunning():
            self.offline_sync_worker.cancel()
            self.offline_sync_worker.wait(400)
        self.spectrum_worker.stop()
        self.spectrum_worker.wait(500)
        super().closeEvent(event)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        apply_frameless_window_shadow(self)

    def changeEvent(self, event: QtCore.QEvent) -> None:
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            is_max = self.isMaximized()
            if hasattr(self, "btn_max"):
                self.btn_max.set_kind("restore" if is_max else "max")
            if hasattr(self, "central_widget"):
                if is_max:
                    self.central_widget.setStyleSheet("QWidget#centralWidget { background-color: #000000; border: none; border-radius: 0px; }")
                else:
                    self.central_widget.setStyleSheet("QWidget#centralWidget { background-color: #000000; border: 1px solid #27272a; border-radius: 10px; }")
        super().changeEvent(event)

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

        # Keyboard shortcuts
        self.rename_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_F2), self)
        self.rename_shortcut.activated.connect(self._rename_preset_inline)

        self.delete_shortcut = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self)
        self.delete_shortcut.activated.connect(self._delete_preset)

    # -------------------------------------------------------------------------
    # DAW-Style Top Title & Menu Bar (Row 1)
    # -------------------------------------------------------------------------
    def _create_top_title_bar(self) -> QtWidgets.QFrame:
        frame = HeaderFrame(self)
        frame.setObjectName("titleBarFrame")
        frame.setFixedHeight(36)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(14, 0, 6, 0)
        layout.setSpacing(4)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Brand Title (clean APO - lowercase clean, uppercase APO)
        self.brand_title_lbl = QtWidgets.QLabel("clean APO")
        self.brand_title_lbl.setStyleSheet(
            "font-size: 13px; font-weight: 700; color: #ffffff; padding-right: 14px; background: transparent;"
        )
        layout.addWidget(self.brand_title_lbl)

        # Menu 1: Settings
        self.btn_menu_settings = QtWidgets.QPushButton("Settings")
        self.btn_menu_settings.setObjectName("topMenuBtn")
        self.btn_menu_settings.setToolTip("Open Clean APO & Equalizer APO Settings")
        self.btn_menu_settings.clicked.connect(self._open_settings)
        layout.addWidget(self.btn_menu_settings)

        # Menu 2: Input (Target Playback Device for Equalizer APO)
        self.btn_menu_input = QtWidgets.QPushButton("Input")
        self.btn_menu_input.setObjectName("topMenuBtn")
        self.btn_menu_input.setToolTip("Select Equalizer APO Playback Target Device")
        self.btn_menu_input.clicked.connect(self._show_input_menu)
        layout.addWidget(self.btn_menu_input)

        # Menu 3: Capture Device (Loopback Capture for Spectrum Analyzer)
        self.btn_menu_capture = QtWidgets.QPushButton("Capture Device")
        self.btn_menu_capture.setObjectName("topMenuBtn")
        self.btn_menu_capture.setToolTip("Select Audio Loopback Capture Device for Spectrum Analyzer")
        self.btn_menu_capture.clicked.connect(self._show_capture_menu)
        layout.addWidget(self.btn_menu_capture)

        # Middle stretch allows window dragging across the entire top bar
        layout.addStretch(1)

        # Far Right: Window Controls (Minimize, Maximize/Restore, Close matching Google Antigravity)
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
    # DAW Control Bar (Row 2 - Operational Controls)
    # -------------------------------------------------------------------------
    def _create_control_bar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("controlBarFrame")
        frame.setFixedHeight(38)
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(10)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        # Left: Universal EQ Toggle Button
        self.eq_universal_toggle = QtWidgets.QPushButton("EQ On")
        self.eq_universal_toggle.setObjectName("activeEqBtn")
        self.eq_universal_toggle.setFixedHeight(28)
        self.eq_universal_toggle.setFixedWidth(75)
        self.eq_universal_toggle.clicked.connect(self._toggle_universal_eq)
        layout.addWidget(self.eq_universal_toggle)

        # Left: Live Spectrum Analyzer Toggle (Right next to EQ On)
        self.spectrum_toggle_btn = QtWidgets.QPushButton("Spectrum: ON")
        self.spectrum_toggle_btn.setCheckable(True)
        self.spectrum_toggle_btn.setChecked(True)
        self.spectrum_toggle_btn.setObjectName("activeEqBtn")
        self.spectrum_toggle_btn.setFixedHeight(28)
        self.spectrum_toggle_btn.setFixedWidth(112)
        self.spectrum_toggle_btn.clicked.connect(self._toggle_spectrum)
        layout.addWidget(self.spectrum_toggle_btn)

        layout.addStretch(1)

        # Right: Preamp Control (Clean unclipped spinbox + smooth slider)
        preamp_box = QtWidgets.QHBoxLayout()
        preamp_box.setSpacing(6)
        preamp_box.setContentsMargins(0, 0, 0, 0)
        preamp_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        preamp_lbl = QtWidgets.QLabel("PREAMP:")
        preamp_lbl.setStyleSheet("font-size: 10px; font-weight: 700; color: #71717a; letter-spacing: 0.5px; padding-right: 2px;")
        preamp_box.addWidget(preamp_lbl)

        self.preamp_spin = CleanSpinBox()
        self.preamp_spin.setRange(-30.0, 15.0)
        self.preamp_spin.setSingleStep(0.5)
        self.preamp_spin.setSuffix(" dB")
        self.preamp_spin.setValue(0.0)
        self.preamp_spin.setFixedHeight(28)
        self.preamp_spin.setFixedWidth(88)
        self.preamp_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        preamp_box.addWidget(self.preamp_spin)

        self.preamp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.preamp_slider.setRange(-300, 150)
        self.preamp_slider.setValue(0)
        self.preamp_slider.setFixedHeight(28)
        self.preamp_slider.setFixedWidth(120)
        preamp_box.addWidget(self.preamp_slider)
        layout.addLayout(preamp_box)

        return frame

    def _show_input_menu(self) -> None:
        self.btn_menu_input.setStyleSheet("background-color: #27272a; color: #ffffff;")
        menu = self._create_input_menu()
        menu.exec(self.btn_menu_input.mapToGlobal(QtCore.QPoint(0, self.btn_menu_input.height() + 3)))
        self.btn_menu_input.setStyleSheet("")

    def _create_input_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        active_name = self.preset_manager.active_device_name
        is_all = clean_apo_device_name(active_name) == "all"

        act_all = menu.addAction("All Endpoints (Default / Global)")
        act_all.setCheckable(True)
        act_all.setChecked(is_all)
        act_all.triggered.connect(lambda: self._on_input_device_selected(None, "All Endpoints"))

        devices = self._get_playback_devices()
        if devices:
            menu.addSeparator()
            for dev_name in devices:
                act = menu.addAction(dev_name)
                act.setCheckable(True)
                checked = not is_all and (clean_apo_device_name(active_name).lower() == clean_apo_device_name(dev_name).lower())
                act.setChecked(checked)
                act.triggered.connect(lambda checked, d=dev_name: self._on_input_device_selected(d, d))

        return menu

    def _get_playback_devices(self) -> List[str]:
        return get_available_playback_devices()

    def _on_input_device_selected(self, dev_id: Optional[str], dev_name: str) -> None:
        self.preset_manager.set_active_device_name(dev_id)
        preamp_val, bands = self._collect_current_eq_state()
        self.preset_manager.schedule_apo_export(
            preamp_val, bands, bypassed=self.is_eq_bypassed
        )
        self._check_device_apo_status(dev_name)

    def _show_capture_menu(self) -> None:
        self.btn_menu_capture.setStyleSheet("background-color: #27272a; color: #ffffff;")
        menu = self._create_capture_menu()
        menu.exec(self.btn_menu_capture.mapToGlobal(QtCore.QPoint(0, self.btn_menu_capture.height() + 3)))
        self.btn_menu_capture.setStyleSheet("")

    def _create_capture_menu(self) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(self)
        current_idx = self.preset_manager.audio_device_index
        devices = get_available_audio_devices()

        act_default = menu.addAction("Default Windows Audio (Auto-Follow)")
        act_default.setCheckable(True)
        act_default.setChecked(current_idx is None)
        act_default.triggered.connect(lambda: self._on_capture_device_selected(None, "Default Windows Audio (Auto-Follow)"))

        non_default = [(idx, name) for idx, name in devices if idx is not None]
        if non_default:
            menu.addSeparator()
            for idx, name in non_default:
                act = menu.addAction(name)
                act.setCheckable(True)
                act.setChecked(current_idx is not None and current_idx == idx)
                act.triggered.connect(lambda checked, i=idx, n=name: self._on_capture_device_selected(i, n))

        return menu

    def _on_capture_device_selected(self, dev_idx: Optional[int], dev_name: str) -> None:
        self.spectrum_worker.set_device_index(dev_idx)
        self.preset_manager.audio_device_index = dev_idx
        self.preset_manager.save_settings()

    # -------------------------------------------------------------------------
    # Left Sidebar: Profile/Preset Management & Squiglink AutoEQ Suite
    # -------------------------------------------------------------------------
    def _create_sidebar(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("sidebarFrame")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Top Navigation Tab Switcher [PROFILES] [AUTOEQ] (iOS/macOS Segmented Capsule)
        self.sidebar_tab_bar = SegmentedPillTabBar(["PROFILES", "AUTOEQ"], self)
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

        # Header with Import button
        head_box = QtWidgets.QHBoxLayout()
        header = QtWidgets.QLabel("PROFILES")
        header.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 1px;")
        head_box.addWidget(header)
        head_box.addStretch()

        import_btn = QtWidgets.QPushButton("Import EQ")
        import_btn.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        import_btn.clicked.connect(self._import_eq_dialog)
        head_box.addWidget(import_btn)
        layout.addLayout(head_box)

        # Search box
        self.preset_search = QtWidgets.QLineEdit()
        self.preset_search.setPlaceholderText("Search profiles...")
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

        # Action Buttons (Swapped: Rename in top-left, Add in bottom-left)
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)

        ren_btn = QtWidgets.QPushButton("Rename (F2)")
        ren_btn.clicked.connect(self._rename_preset_inline)
        btn_grid.addWidget(ren_btn, 0, 0)

        dup_btn = QtWidgets.QPushButton("Duplicate")
        dup_btn.clicked.connect(self._duplicate_preset)
        btn_grid.addWidget(dup_btn, 0, 1)

        add_btn = QtWidgets.QPushButton("+ Add")
        add_btn.setObjectName("primaryBtn")
        add_btn.setToolTip("Create a new flat EQ profile")
        add_btn.clicked.connect(self._add_preset_auto)
        btn_grid.addWidget(add_btn, 1, 0)

        del_btn = QtWidgets.QPushButton("Delete")
        del_btn.setObjectName("dangerBtn")
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
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        # 0. Global Database Search & Offline Database Sync
        search_sync_row = QtWidgets.QHBoxLayout()
        search_sync_row.setSpacing(4)

        self.autoeq_global_search_btn = QtWidgets.QPushButton("Global Search (43k+)")
        self.autoeq_global_search_btn.setObjectName("globalSearchBtn")
        self.autoeq_global_search_btn.setToolTip("Search across 43,000+ IEM & headphone measurements from 148 reviewer databases")
        self.autoeq_global_search_btn.clicked.connect(self._open_global_search)
        search_sync_row.addWidget(self.autoeq_global_search_btn, stretch=3)

        self.autoeq_offline_sync_btn = QtWidgets.QPushButton("Offline DB")
        self.autoeq_offline_sync_btn.setToolTip("Download and manage offline measurement and target curves (~60 MB)")
        self.autoeq_offline_sync_btn.setStyleSheet("""
            QPushButton {
                background: #18181b;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #e4e4e7;
                font-size: 11px;
                font-weight: 600;
                padding: 6px 8px;
            }
            QPushButton:hover {
                background: #27272a;
                border-color: #38bdf8;
                color: #ffffff;
            }
        """)
        self.autoeq_offline_sync_btn.clicked.connect(self._open_offline_sync)
        search_sync_row.addWidget(self.autoeq_offline_sync_btn, stretch=2)

        layout.addLayout(search_sync_row)

        # Database Selection
        db_sec = QtWidgets.QVBoxLayout()
        db_sec.setSpacing(2)
        db_lbl = QtWidgets.QLabel("Database")
        db_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        db_sec.addWidget(db_lbl)

        db_row = QtWidgets.QHBoxLayout()
        db_row.setSpacing(4)

        self.autoeq_db_combo = SearchableDatabaseInput(self)
        self.autoeq_db_combo.currentIndexChanged.connect(self._on_autoeq_db_changed)
        db_row.addWidget(self.autoeq_db_combo, stretch=1)

        self.autoeq_db_star_btn = QtWidgets.QPushButton("☆")
        self.autoeq_db_star_btn.setFixedWidth(26)
        self.autoeq_db_star_btn.setFixedHeight(26)
        self.autoeq_db_star_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.autoeq_db_star_btn.setToolTip("Favorite this database to pin it to top")
        self.autoeq_db_star_btn.setStyleSheet("color: #71717a; font-size: 16px; border: none; background: transparent; padding: 0px; text-align: center;")
        self.autoeq_db_star_btn.clicked.connect(self._toggle_current_db_favorite)
        db_row.addWidget(self.autoeq_db_star_btn)

        db_sec.addLayout(db_row)

        self.autoeq_rig_badge = QtWidgets.QLabel("Rig: clone IEC 60318-4 (711) [Isolated]")
        self.autoeq_rig_badge.setStyleSheet("font-size: 10px; color: #a1a1aa; padding: 0px 2px;")
        self.autoeq_rig_badge.setToolTip("Strict Rig Isolation: Targets and measurements are locked to this reviewer's coupler to prevent measurement mismatch.")
        db_sec.addWidget(self.autoeq_rig_badge)
        layout.addLayout(db_sec)

        # Headphone / IEM Model Selection (Single input field with dropdown menu as you type)
        model_sec = QtWidgets.QVBoxLayout()
        model_sec.setSpacing(2)
        model_lbl = QtWidgets.QLabel("Headphone / IEM Model")
        model_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        model_sec.addWidget(model_lbl)

        model_row = QtWidgets.QHBoxLayout()
        model_row.setSpacing(4)

        self.autoeq_model_combo = SearchableModelInput()
        self.autoeq_model_combo.sigModelSelected.connect(self._on_searchable_model_selected)
        model_row.addWidget(self.autoeq_model_combo, stretch=3)

        self.autoeq_chan_combo = QtWidgets.QComboBox()
        self.autoeq_chan_combo.addItem("Avg", "avg")
        self.autoeq_chan_combo.addItem("L", "L")
        self.autoeq_chan_combo.addItem("R", "R")
        self.autoeq_chan_combo.setToolTip("Channel: Average (L+R), Left (L), or Right (R)")
        self.autoeq_chan_combo.setFixedWidth(80)
        self.autoeq_chan_combo.setStyleSheet("QComboBox { padding: 4px 20px 4px 8px; }")
        self.autoeq_chan_combo.currentIndexChanged.connect(self._on_autoeq_model_changed)
        model_row.addWidget(self.autoeq_chan_combo, stretch=1)

        model_sec.addLayout(model_row)
        layout.addLayout(model_sec)

        # Target Specification (Target Curve vs IEM Tuning vs Resultant Profile)
        target_sec = QtWidgets.QVBoxLayout()
        target_sec.setSpacing(4)
        target_lbl = QtWidgets.QLabel("Target Response")
        target_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        target_sec.addWidget(target_lbl)

        # Segmented pill selector styled like top PROFILES / AUTOEQ bar
        self.autoeq_target_mode_tab_bar = SegmentedPillTabBar(
            ["TARGET CURVE", "IEM TUNING", "RESULTANT"],
            self,
            height=28,
            font_size=8,
        )
        self.autoeq_target_mode_tab_bar.sigTabChanged.connect(self._on_target_mode_changed)
        target_sec.addWidget(self.autoeq_target_mode_tab_bar)

        # Stacked container for target options
        self.autoeq_target_stack = QtWidgets.QStackedWidget()

        # Page 0: Target Curve
        p0_widget = QtWidgets.QWidget()
        p0_layout = QtWidgets.QVBoxLayout(p0_widget)
        p0_layout.setContentsMargins(0, 2, 0, 0)
        p0_layout.setSpacing(2)
        self.autoeq_target_combo = QtWidgets.QComboBox()
        self.autoeq_target_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p0_layout.addWidget(self.autoeq_target_combo)
        self.autoeq_target_info_label = QtWidgets.QLabel("Aligned: Midband MSE (200 - 2,000 Hz)")
        self.autoeq_target_info_label.setStyleSheet("font-size: 10px; color: #71717a;")
        p0_layout.addWidget(self.autoeq_target_info_label)
        self.autoeq_target_stack.addWidget(p0_widget)

        # Page 1: IEM Tuning (Match another IEM / Headphone)
        p1_widget = QtWidgets.QWidget()
        p1_layout = QtWidgets.QVBoxLayout(p1_widget)
        p1_layout.setContentsMargins(0, 2, 0, 0)
        p1_layout.setSpacing(2)
        p1_row = QtWidgets.QHBoxLayout()
        p1_row.setSpacing(4)
        self.autoeq_target_iem_combo = SearchableTargetIEMInput(squig_client=self.squig_client)
        self.autoeq_target_iem_combo.sigModelSelected.connect(lambda _: self._on_autoeq_target_changed())
        p1_row.addWidget(self.autoeq_target_iem_combo, stretch=3)
        self.autoeq_target_iem_chan_combo = QtWidgets.QComboBox()
        self.autoeq_target_iem_chan_combo.addItem("Avg", "avg")
        self.autoeq_target_iem_chan_combo.addItem("L", "L")
        self.autoeq_target_iem_chan_combo.addItem("R", "R")
        self.autoeq_target_iem_chan_combo.setToolTip("Target Channel: Average (L+R), Left (L), or Right (R)")
        self.autoeq_target_iem_chan_combo.setFixedWidth(80)
        self.autoeq_target_iem_chan_combo.setStyleSheet("QComboBox { padding: 4px 20px 4px 8px; }")
        self.autoeq_target_iem_chan_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p1_row.addWidget(self.autoeq_target_iem_chan_combo, stretch=1)
        p1_layout.addLayout(p1_row)

        p1_sub_row = QtWidgets.QHBoxLayout()
        p1_sub_row.setSpacing(6)
        self.autoeq_target_same_provider_cb = QtWidgets.QCheckBox("Same provider only")
        self.autoeq_target_same_provider_cb.setChecked(True)
        self.autoeq_target_same_provider_cb.setToolTip(
            "Checked: search within current reviewer coupler rig (recommended for acoustic consistency).\n"
            "Unchecked: search across 43k+ models with current provider prioritized first."
        )
        self.autoeq_target_same_provider_cb.setStyleSheet("""
            QCheckBox {
                color: #a1a1aa;
                font-size: 10px;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
        """)
        self.autoeq_target_same_provider_cb.toggled.connect(self._on_target_same_provider_toggled)
        p1_sub_row.addWidget(self.autoeq_target_same_provider_cb)

        self.autoeq_target_iem_badge = QtWidgets.QLabel("Rig: IEC 60318-4 [Matched]")
        self.autoeq_target_iem_badge.setStyleSheet("font-size: 10px; color: #71717a;")
        self.autoeq_target_iem_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        p1_sub_row.addWidget(self.autoeq_target_iem_badge, stretch=1)
        p1_layout.addLayout(p1_sub_row)
        self.autoeq_target_stack.addWidget(p1_widget)

        # Page 2: Resultant Profile
        p2_widget = QtWidgets.QWidget()
        p2_layout = QtWidgets.QVBoxLayout(p2_widget)
        p2_layout.setContentsMargins(0, 2, 0, 0)
        p2_layout.setSpacing(2)
        self.autoeq_target_profile_combo = QtWidgets.QComboBox()
        self.autoeq_target_profile_combo.currentIndexChanged.connect(self._on_autoeq_target_changed)
        p2_layout.addWidget(self.autoeq_target_profile_combo)
        self.autoeq_target_profile_info_label = QtWidgets.QLabel("Resultant acoustic response of selected profile")
        self.autoeq_target_profile_info_label.setStyleSheet("font-size: 10px; color: #71717a;")
        p2_layout.addWidget(self.autoeq_target_profile_info_label)
        self.autoeq_target_stack.addWidget(p2_widget)

        target_sec.addWidget(self.autoeq_target_stack)
        layout.addLayout(target_sec)

        # Optimization Settings (Compact 2x2 Grid)
        opt_sec = QtWidgets.QVBoxLayout()
        opt_sec.setSpacing(3)
        opt_lbl = QtWidgets.QLabel("Optimizer Settings")
        opt_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 0.8px;")
        opt_sec.addWidget(opt_lbl)

        opt_grid = QtWidgets.QGridLayout()
        opt_grid.setHorizontalSpacing(6)
        opt_grid.setVerticalSpacing(3)

        # Row 0: Bands & Bass Shelf Checkbox
        bands_lbl = QtWidgets.QLabel("Bands:")
        bands_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        opt_grid.addWidget(bands_lbl, 0, 0)

        self.autoeq_bands_spin = QtWidgets.QSpinBox()
        self.autoeq_bands_spin.setRange(1, 32)
        self.autoeq_bands_spin.setValue(7)
        self.autoeq_bands_spin.setFixedWidth(84)
        self.autoeq_bands_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        opt_grid.addWidget(self.autoeq_bands_spin, 0, 1)

        self.autoeq_bass_shelf_cb = QtWidgets.QPushButton("Bass Shelf")
        self.autoeq_bass_shelf_cb.setCheckable(True)
        self.autoeq_bass_shelf_cb.setChecked(True)
        self.autoeq_bass_shelf_cb.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.autoeq_bass_shelf_cb.setToolTip("Allocates Band 1 as Low Shelf filter for sub-bass target correction")
        self.autoeq_bass_shelf_cb.setFixedHeight(36)
        self.autoeq_bass_shelf_cb.setStyleSheet("""
            QPushButton {
                background-color: #18181b;
                color: #71717a;
                border: 1px solid #27272a;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 26px;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #27272a;
                border-color: #3f3f46;
                color: #a1a1aa;
            }
            QPushButton:checked {
                background-color: #ffffff;
                color: #09090b;
                border: 1px solid #ffffff;
                font-weight: 600;
            }
            QPushButton:checked:hover {
                background-color: #f4f4f5;
                color: #000000;
                border-color: #ffffff;
            }
        """)
        opt_grid.addWidget(self.autoeq_bass_shelf_cb, 0, 2, 1, 2)

        # Row 1: Max Boost & Treble Limit (Borders line up exactly with row 0)
        boost_lbl = QtWidgets.QLabel("Boost:")
        boost_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        opt_grid.addWidget(boost_lbl, 1, 0)

        self.autoeq_max_boost_spin = CleanSpinBox()
        self.autoeq_max_boost_spin.setRange(0.0, 15.0)
        self.autoeq_max_boost_spin.setValue(9.0)
        self.autoeq_max_boost_spin.setSingleStep(0.5)
        self.autoeq_max_boost_spin.setSuffix(" dB")
        self.autoeq_max_boost_spin.setFixedWidth(84)
        self.autoeq_max_boost_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        opt_grid.addWidget(self.autoeq_max_boost_spin, 1, 1)

        treble_lbl = QtWidgets.QLabel("Limit:")
        treble_lbl.setStyleSheet("font-size: 11px; color: #a1a1aa;")
        opt_grid.addWidget(treble_lbl, 1, 2)

        self.autoeq_max_freq_spin = CleanSpinBox()
        self.autoeq_max_freq_spin.setRange(4000.0, 10000.0)
        self.autoeq_max_freq_spin.setDecimals(0)
        self.autoeq_max_freq_spin.setValue(8500.0)
        self.autoeq_max_freq_spin.setSingleStep(500.0)
        self.autoeq_max_freq_spin.setSuffix(" Hz")
        self.autoeq_max_freq_spin.setFixedWidth(96)
        self.autoeq_max_freq_spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.autoeq_max_freq_spin.setToolTip("Restricts aggressive corrections above 8.5–10 kHz due to 711 coupler insertion resonances")
        opt_grid.addWidget(self.autoeq_max_freq_spin, 1, 3)

        opt_sec.addLayout(opt_grid)
        layout.addLayout(opt_sec)

        # Action Button & Live Status
        self.autoeq_compute_btn = QtWidgets.QPushButton("Compute AutoEQ")
        self.autoeq_compute_btn.setObjectName("primaryBtn")
        self.autoeq_compute_btn.setMinimumHeight(30)
        self.autoeq_compute_btn.clicked.connect(self._compute_autoeq)
        layout.addWidget(self.autoeq_compute_btn)

        self.autoeq_status_label = QtWidgets.QLabel("Ready")
        self.autoeq_status_label.setMaximumHeight(20)
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")
        layout.addWidget(self.autoeq_status_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.autoeq_save_preset_btn = QtWidgets.QPushButton("Save Profile")
        self.autoeq_save_preset_btn.clicked.connect(self._save_autoeq_profile)
        btn_row.addWidget(self.autoeq_save_preset_btn)

        self.autoeq_clear_overlays_btn = QtWidgets.QPushButton("Clear Traces")
        self.autoeq_clear_overlays_btn.clicked.connect(self._clear_autoeq_overlays)
        btn_row.addWidget(self.autoeq_clear_overlays_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # -------------------------------------------------------------------------
    # Bottom Filter Table Panel
    # -------------------------------------------------------------------------
    def _create_filter_table_panel(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("tableFrame")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        header_row = QtWidgets.QHBoxLayout()
        self.band_count_label = QtWidgets.QLabel("BANDS (0)")
        self.band_count_label.setStyleSheet("font-size: 10px; font-weight: 800; color: #71717a; letter-spacing: 1px;")
        header_row.addWidget(self.band_count_label)

        header_row.addSpacing(16)

        # Graph curve overlay toggles (left of Flatten)
        self.undergraph_target_btn = QtWidgets.QPushButton("Target")
        self.undergraph_target_btn.setCheckable(True)
        self.undergraph_target_btn.setChecked(False)
        self.undergraph_target_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_target_btn.setToolTip("Toggle Target reference curve overlay on graph")
        self.undergraph_target_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_target_btn)

        self.undergraph_meas_btn = QtWidgets.QPushButton("Measurement")
        self.undergraph_meas_btn.setCheckable(True)
        self.undergraph_meas_btn.setChecked(False)
        self.undergraph_meas_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_meas_btn.setToolTip("Toggle IEM measurement curve overlay on graph")
        self.undergraph_meas_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_meas_btn)

        self.undergraph_resultant_btn = QtWidgets.QPushButton("Resultant")
        self.undergraph_resultant_btn.setCheckable(True)
        self.undergraph_resultant_btn.setChecked(False)
        self.undergraph_resultant_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_resultant_btn.setToolTip("Toggle predicted resultant response curve on graph")
        self.undergraph_resultant_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_resultant_btn)

        self.undergraph_target2_btn = QtWidgets.QPushButton("Target 2")
        self.undergraph_target2_btn.setCheckable(True)
        self.undergraph_target2_btn.setChecked(False)
        self.undergraph_target2_btn.setVisible(False)
        self.undergraph_target2_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_target2_btn.setToolTip("Toggle Secondary Target reference curve overlay on graph")
        self.undergraph_target2_btn.toggled.connect(self._on_undergraph_overlay_toggled)
        header_row.addWidget(self.undergraph_target2_btn)

        self.undergraph_add_target_btn = QtWidgets.QPushButton("+")
        self.undergraph_add_target_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.undergraph_add_target_btn.setToolTip("Add or change secondary target curve from same provider")
        self.undergraph_add_target_btn.setFixedSize(22, 22)
        self.undergraph_add_target_btn.setStyleSheet("""
            QPushButton {
                background-color: #121215;
                border: 1px solid #27272a;
                border-radius: 4px;
                color: #a1a1aa;
                font-size: 13px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #1c1c20;
                border-color: #3f3f46;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #27272a;
            }
        """)
        self.undergraph_add_target_btn.clicked.connect(self._on_add_target_clicked)
        header_row.addWidget(self.undergraph_add_target_btn)

        self._style_undergraph_toggle_buttons()

        header_row.addStretch()

        flatten_btn = QtWidgets.QPushButton("Flatten")
        flatten_btn.setToolTip("Reset all band gains to 0 dB")
        flatten_btn.clicked.connect(self._flatten_all_bands)
        header_row.addWidget(flatten_btn)

        compress_btn = QtWidgets.QPushButton("Compress")
        compress_btn.setToolTip("Scale all band gains down (cancels out with Expand)")
        compress_btn.clicked.connect(self._compress_all_bands)
        header_row.addWidget(compress_btn)

        expand_btn = QtWidgets.QPushButton("Expand")
        expand_btn.setToolTip("Scale all band gains up (cancels out with Compress)")
        expand_btn.clicked.connect(self._expand_all_bands)
        header_row.addWidget(expand_btn)

        add_band_btn = QtWidgets.QPushButton("＋ Add Band")
        add_band_btn.setObjectName("primaryBtn")
        add_band_btn.clicked.connect(self._add_band_default)
        header_row.addWidget(add_band_btn)

        reset_bands_btn = QtWidgets.QPushButton("Reset All")
        reset_bands_btn.clicked.connect(self._reset_all_bands)
        header_row.addWidget(reset_bands_btn)

        layout.addLayout(header_row)

        # Table Widget (Column 5 is explicitly "Delete")
        self.filter_table = QtWidgets.QTableWidget()
        self.filter_table.setColumnCount(6)
        self.filter_table.setHorizontalHeaderLabels([
            "Band", "Filter Type", "Frequency (Hz)", "Gain (dB)", "Q Factor", "Delete"
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

    # -------------------------------------------------------------------------
    # Audio Device Initialization & Dynamic Hotplug Tracking
    # -------------------------------------------------------------------------
    def _init_audio_device(self) -> None:
        active_idx = self.preset_manager.audio_device_index
        self.spectrum_worker.set_device_index(active_idx)
        active_name = self.preset_manager.active_device_name
        self._check_device_apo_status(active_name or "All Endpoints")

    def _check_device_hotplug(self) -> None:
        pass

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
        if self.preset_manager.get_setting("ignore_missing_apo", False):
            return
        if not is_equalizer_apo_installed():
            self._prompt_install_equalizer_apo()

    def _prompt_install_equalizer_apo(self) -> None:
        dlg = EqualizerApoInstallDialog(self)
        dlg.sigInstalled.connect(self._on_apo_installed_success)
        dlg.exec()
        if dlg.chk_dont_ask.isChecked():
            self.preset_manager.set_setting("ignore_missing_apo", True)

    def _on_apo_installed_success(self) -> None:
        self.preset_manager.apo_target_path = DEFAULT_APO_PATH
        self.preset_manager.active_apo_path = DEFAULT_APO_PATH
        ensure_apo_root_config_include(DEFAULT_APO_PATH)
        self.preset_manager.save_settings()
        self.preset_manager.save_active_preset_to_apo()
        if hasattr(self, "status_bar_label"):
            self.status_bar_label.setText("Equalizer APO Connected (clean_apo_generated.txt)")

    def _check_device_apo_status(self, dev_name: str) -> None:
        if not dev_name or "auto-follow" in dev_name.lower():
            return
        if not is_equalizer_apo_installed():
            self._prompt_install_equalizer_apo()
            return
        cfg = get_apo_configurator_path()
        if not cfg:
            return
        if not self._is_device_installed_in_apo(dev_name):
            dlg = ApoDeviceConfigDialog(dev_name, cfg, self)
            dlg.exec()

    def _is_device_installed_in_apo(self, dev_name: str) -> bool:
        if not dev_name or "auto-follow" in dev_name.lower():
            return True
        clean_name = clean_apo_device_name(dev_name).lower()
        if clean_name == "all":
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
                                _, vd, _ = winreg.EnumValue(pk, pi)
                                if isinstance(vd, str) and (clean_name in vd.lower() or vd.lower() in clean_name):
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

        self.preset_manager.sigPresetChanged.connect(self._on_preset_loaded)
        self.preset_manager.sigPresetsListChanged.connect(self._refresh_presets_list)

        self.eq_plot.sigBandModified.connect(self._on_plot_band_modified)
        self.eq_plot.sigBandAdded.connect(self._on_plot_band_added)
        self.eq_plot.sigBandDeleted.connect(self._delete_band)
        self.eq_plot.sigBandDragStarted.connect(self._on_plot_drag_started)

        self.spectrum_worker.sigSpectrumReady.connect(self.eq_plot.update_spectrum_curve)
        self.eq_plot.sigSpectrumToggled.connect(self._on_plot_spectrum_toggled)
        self.eq_plot.sigLegendCurveClicked.connect(self._on_graph_legend_clicked)
        self.eq_plot.sigWheelScrollStarted.connect(self._notify_wheel_scroll)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Wheel:
            if isinstance(watched, (QtWidgets.QAbstractSpinBox, QtWidgets.QSlider)):
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
    def _toggle_universal_eq(self) -> None:
        self.is_eq_bypassed = not self.is_eq_bypassed
        if self.is_eq_bypassed:
            self.eq_universal_toggle.setText("EQ Off")
            self.eq_universal_toggle.setObjectName("bypassedEqBtn")
        else:
            self.eq_universal_toggle.setText("EQ On")
            self.eq_universal_toggle.setObjectName("activeEqBtn")

        self.eq_universal_toggle.setStyle(self.eq_universal_toggle.style())
        self.eq_plot.set_eq_bypassed(self.is_eq_bypassed)
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
        )

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
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)
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
        self._is_undoing_redoing = True
        try:
            restored_bands = [EQBand.from_dict(d) for d in bands_dict]
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
        self.preset_manager.set_active_preset(item.text())

    def _on_preset_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Double clicking profile prompts for inline renaming (no modal window)."""
        self.preset_list_widget.editItem(item)

    def _rename_preset_inline(self) -> None:
        """F2 or Rename button triggers inline renaming without opening any new window."""
        item = self.preset_list_widget.currentItem()
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

            # rename_preset updates presets.json, settings.json, and emits sigPresetsListChanged & sigPresetChanged
            self.preset_manager.rename_preset(old_name, new_name)
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

        act_curves = menu.addAction("Edit Target & Measurement..." if has_curves else "+ Add Target & Measurement...")
        act_curves.triggered.connect(lambda: self._open_assign_curves_dialog(name))

        if has_curves:
            act_remove = menu.addAction("Remove Target & Measurement")
            act_remove.triggered.connect(lambda: self._remove_preset_curves(name))

        menu.addSeparator()

        act_ren = menu.addAction("Rename (F2)")
        act_ren.triggered.connect(self._rename_preset_inline)

        act_dup = menu.addAction("Duplicate")
        act_dup.triggered.connect(self._duplicate_preset)

        act_del = menu.addAction("Delete")
        act_del.triggered.connect(self._delete_preset)

        menu.exec(self.preset_list_widget.mapToGlobal(pos))

    def _open_assign_curves_dialog(self, preset_name: Optional[str] = None) -> None:
        target_preset = preset_name or self.preset_manager.active_preset_name
        dlg = ProfileCurveAssignmentDialog(
            client=self.squig_client,
            preset_manager=self.preset_manager,
            preset_name=target_preset,
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
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
        if mf and tf and len(mf) > 0 and len(tf) > 0 and len(ms) > 0 and len(ts) > 0:
            f_arr = np.asarray(tf, dtype=np.float64)
            t_arr = np.asarray(ts, dtype=np.float64)
            mid_mask = (f_arr >= 200.0) & (f_arr <= 2000.0)
            target_norm = float(np.mean(t_arr[mid_mask])) if np.any(mid_mask) else float(np.mean(t_arr))

            meas_interp = np.interp(tf, mf, ms)
            offset = float(np.mean(t_arr[mid_mask] - meas_interp[mid_mask])) if np.any(mid_mask) else float(np.mean(t_arr - meas_interp))
            meas_aligned = np.asarray(ms, dtype=np.float64) + offset

            disp_target = t_arr - target_norm
            disp_meas = meas_aligned - target_norm

            eq_response = compute_composite_response(
                bands, np.asarray(mf, dtype=np.float64), preamp_db=0.0, fs=self.sample_rate
            )
            disp_res = disp_meas + eq_response

            return (
                (list(mf), list(disp_meas)),
                (list(tf), list(disp_target)),
                (list(mf), list(disp_res)),
            )
        elif mf and len(mf) > 0 and len(ms) > 0:
            m_arr = np.asarray(ms, dtype=np.float64)
            m_norm = float(np.mean(m_arr))
            disp_meas = m_arr - m_norm
            eq_response = compute_composite_response(
                bands, np.asarray(mf, dtype=np.float64), preamp_db=0.0, fs=self.sample_rate
            )
            disp_res = disp_meas + eq_response
            return (
                (list(mf), list(disp_meas)),
                ([], []),
                (list(mf), list(disp_res)),
            )
        elif tf and len(tf) > 0 and len(ts) > 0:
            t_arr = np.asarray(ts, dtype=np.float64)
            t_norm = float(np.mean(t_arr))
            disp_target = t_arr - t_norm
            return (
                ([], []),
                (list(tf), list(disp_target)),
                ([], []),
            )
        return (([], []), ([], []), ([], []))

    def _on_preset_loaded(self, preset_name: str, animate: bool = True) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        for i in range(self.preset_list_widget.count()):
            item = self.preset_list_widget.item(i)
            if item.text() == preset_name:
                self.preset_list_widget.setCurrentItem(item)
                self.preset_list_widget.animate_to_current()
                break

        preamp, bands = self.preset_manager.get_active_preset_data()

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
            db_id = autoeq_meta.get("database_id", "")
            m_file = autoeq_meta.get("model_file", "")
            t_file = autoeq_meta.get("target_file", "")
            t_name = autoeq_meta.get("target_name", "")
            chan = autoeq_meta.get("channel_mode", "avg")
            m_name = autoeq_meta.get("model_name", "")
            t_mode = autoeq_meta.get("target_mode", "target_curve")

            # Query offline database synchronously (ultra-fast SQLite lookup)
            from offline_db import get_offline_db
            off_db = get_offline_db()
            m_res = off_db.get_measurement(db_id, m_file, chan) if m_file else None
            if not m_res and m_file:
                base_u = autoeq_meta.get("base_url")
                if base_u:
                    try:
                        temp_m = SquigModel(
                            brand="",
                            name=m_name or m_file,
                            full_name=m_name or m_file,
                            file_name=m_file,
                            display_name=m_name or m_file,
                        )
                        m_res = self.squig_client.fetch_iem_measurement(base_u, temp_m, chan)
                    except Exception:
                        m_res = None

            t_res = None
            if t_mode == "iem_tuning" or autoeq_meta.get("target_iem_file"):
                t_iem_file = autoeq_meta.get("target_iem_file", "")
                t_iem_name = autoeq_meta.get("target_iem_name", "")
                t_iem_chan = autoeq_meta.get("target_iem_channel", "avg")
                t_iem_base = autoeq_meta.get("target_iem_base_url") or autoeq_meta.get("base_url", "")
                t_res = off_db.get_measurement(db_id, t_iem_file, t_iem_chan) if t_iem_file else None
                if not t_res and t_iem_file and t_iem_base:
                    try:
                        temp_m = SquigModel(
                            brand="",
                            name=t_iem_name or t_iem_file,
                            full_name=t_iem_name or t_iem_file,
                            file_name=t_iem_file,
                            display_name=t_iem_name or t_iem_file,
                        )
                        t_res = self.squig_client.fetch_iem_measurement(t_iem_base, temp_m, t_iem_chan)
                    except Exception:
                        t_res = None
                t_name = f"{t_iem_name or t_iem_file} (Target)"
            elif t_mode == "resultant_profile" or autoeq_meta.get("target_profile_name"):
                target_p_name = autoeq_meta.get("target_profile_name", "")
                t_res = self._compute_preset_resultant_curve(target_p_name)
                t_name = f"{target_p_name} (Resultant)"
            else:
                t_res = off_db.get_target(db_id, t_file, t_name) if t_file else None
                if not t_res and t_file:
                    base_u = autoeq_meta.get("base_url")
                    if base_u:
                        try:
                            t_res = self.squig_client.fetch_target_curve(
                                base_u, SquigTarget(name=t_name or t_file, file_name=t_file)
                            )
                        except Exception:
                            t_res = None

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
                t2_file = autoeq_meta.get("target2_file", "")
                t2_name = autoeq_meta.get("target2_name", "")
                t2_res = off_db.get_target(db_id, t2_file, t2_name) if t2_file else None
                if not t2_res and t2_file:
                    base_u = autoeq_meta.get("base_url")
                    if base_u:
                        try:
                            t2_res = self.squig_client.fetch_target_curve(
                                base_u, SquigTarget(name=t2_name or t2_file, file_name=t2_file)
                            )
                        except Exception:
                            t2_res = None

                if t2_res and len(t2_res[0]) > 0 and len(t2_res[1]) > 0:
                    t2_f, t2_s = t2_res
                    t2_f_arr = np.asarray(t2_f, dtype=np.float64)
                    t2_s_arr = np.asarray(t2_s, dtype=np.float64)
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
                self._restore_autoeq_preset_ui(autoeq_meta, skip_curve_fetch=False)
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
        self._last_state = self._get_current_state()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_curves_button_state()

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
                "id": db_id or "custom",
                "name": db_name,
                "base_url": base_url,
                "rig": meta.get("rig", "IEC 711"),
            }
            ins_idx = max(0, self.autoeq_db_combo.count() - 1)
            self.autoeq_db_combo.blockSignals(True)
            self.autoeq_db_combo.insertItem(ins_idx, db_name, new_db_info)
            self.autoeq_db_combo.blockSignals(False)
            self.autoeq_db_combo.setCurrentIndex(ins_idx)

    def _apply_pending_autoeq_restore(self) -> None:
        skip = getattr(self, "_pending_restore_skip_fetch", False)
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
                self._pending_restore_model_file = None
                if not skip:
                    self._fetch_selected_curves()

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
                    self._pending_restore_target_file = None
                    break

        if not self._pending_restore_model_file and not self._pending_restore_target_file:
            if getattr(self, "_pending_restore_skip_fetch", False):
                self._suppress_curve_fetch = False
            self._pending_restore_skip_fetch = False

    def _on_preamp_anim_step(self, val: float) -> None:
        self._apply_preamp_value(float(val))

    def _apply_preamp_value(self, val: float) -> None:
        self.preamp_spin.blockSignals(True)
        self.preamp_slider.blockSignals(True)
        self.preamp_spin.setValue(val)
        self.preamp_slider.setValue(int(val * 10))
        self.preamp_spin.blockSignals(False)
        self.preamp_slider.blockSignals(False)
        self.eq_plot.set_preamp(val)

    def _add_preset_auto(self) -> None:
        """
        Creates a new, flat EQ profile automatically named 'New Profile [N]'.
        Immediately activates inline editing so user can optionally rename it.
        """
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
            # Select new item and trigger inline editing immediately
            for i in range(self.preset_list_widget.count()):
                item = self.preset_list_widget.item(i)
                if item.text() == name:
                    self.preset_list_widget.setCurrentItem(item)
                    QtCore.QTimer.singleShot(50, lambda it=item: self.preset_list_widget.editItem(it))
                    break

    def _duplicate_preset(self) -> None:
        current = self.preset_manager.active_preset_name
        new_name = self.preset_manager.duplicate_preset(current)
        if new_name:
            for i in range(self.preset_list_widget.count()):
                item = self.preset_list_widget.item(i)
                if item.text() == new_name:
                    self.preset_list_widget.setCurrentItem(item)
                    QtCore.QTimer.singleShot(50, lambda it=item: self.preset_list_widget.editItem(it))
                    break

    def _delete_preset(self) -> None:
        current = self.preset_manager.active_preset_name
        if len(self.preset_manager.get_preset_names()) <= 1:
            QtWidgets.QMessageBox.information(self, "Delete Profile", "Cannot delete the only remaining profile.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Confirm Delete", f"Delete profile '{current}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.preset_manager.delete_preset(current)

    def _import_eq_dialog(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Equalizer Configuration",
            "",
            "EQ Files (*.txt *.peace *.json);;Equalizer APO (*.txt);;Peace (*.peace);;JSON (*.json);;All Files (*.*)",
        )
        if not file_path:
            return

        imported = self.preset_manager.import_eq_from_file(file_path)
        if not imported:
            QtWidgets.QMessageBox.warning(self, "Import Failed", "Could not parse filter data from the selected file.")
            return

        name, preamp, bands = imported
        existing = set(self.preset_manager.get_preset_names())
        imp_name = name
        idx = 2
        while imp_name in existing:
            imp_name = f"{name} ({idx})"
            idx += 1

        self.preset_manager.add_preset(imp_name, preamp, bands)

    # -------------------------------------------------------------------------
    # Preamp Synchronization
    # -------------------------------------------------------------------------
    def _on_preamp_spin_changed(self, val: float) -> None:
        self.preamp_slider.blockSignals(True)
        self.preamp_slider.setValue(int(val * 10))
        self.preamp_slider.blockSignals(False)
        self._apply_preamp_change(val)

    def _on_preamp_slider_changed(self, int_val: int) -> None:
        val = int_val / 10.0
        self.preamp_spin.blockSignals(True)
        self.preamp_spin.setValue(val)
        self.preamp_spin.blockSignals(False)
        self._apply_preamp_change(val)

    def _apply_preamp_change(self, preamp_db: float) -> None:
        self._record_undo_snapshot()
        self.eq_plot.set_preamp(preamp_db)
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
        self.preamp_spin.setValue(0.0)
        self.eq_plot.set_bands(self.eq_plot.bands, preamp_db=0.0)
        self._populate_table(self.eq_plot.bands)
        self.preset_manager.update_active_preset_content(0.0, self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(0.0, self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
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
        self._commit_state()

    # -------------------------------------------------------------------------
    # Table & Plot Bi-directional Binding
    # -------------------------------------------------------------------------
    def _populate_table(self, bands: List[EQBand]) -> None:
        self._updating_table = True
        self.filter_table.setRowCount(len(bands))
        self.band_count_label.setText(f"BANDS ({len(bands)})")

        for row, band in enumerate(bands):
            # 0. Band Badge Button (toggles bypass on click, mirrors plot handle)
            badge_btn = BandBadgeButton(band.band_id, band.enabled)
            badge_btn.toggled.connect(self._on_table_active_toggled)
            badge_container = QtWidgets.QWidget()
            b_layout = QtWidgets.QHBoxLayout(badge_container)
            b_layout.setContentsMargins(4, 2, 4, 2)
            b_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            b_layout.addWidget(badge_btn)
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

            # 5. Delete Button ("X" turns red on hover)
            del_btn = QtWidgets.QPushButton("✕")
            del_btn.setProperty("class", "tableDeleteBtn")
            del_btn.setProperty("band_id", band.band_id)
            del_btn.setToolTip("Delete this filter band")
            del_btn.clicked.connect(self._on_table_delete_clicked)
            
            del_container = QtWidgets.QWidget()
            del_layout = QtWidgets.QHBoxLayout(del_container)
            del_layout.setContentsMargins(0, 0, 0, 0)
            del_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            del_layout.addWidget(del_btn)
            self.filter_table.setCellWidget(row, 5, del_container)

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
            self._commit_state()

    def _on_table_type_changed(self, filter_type: str) -> None:
        if self._updating_table:
            return
        sender = self.sender()
        band_id = sender.property("band_id") if sender else None
        if band_id is not None:
            self._record_undo_snapshot()
            ftype = FilterType.from_str(filter_type)
            gain = 0.0 if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS) else None
            self.eq_plot.update_band_params(band_id, filter_type=filter_type, gain=gain)
            self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
            self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
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
        self.preset_manager.update_active_preset_content(self.preamp_spin.value(), self.eq_plot.bands)
        self.preset_manager.schedule_apo_export(self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed)
        self._update_resultant_curve()
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
            freq=round(freq, 1),
            gain=round(gain, 1),
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
        self._commit_state()

    def _reset_all_bands(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Reset Bands", "Reset all filter bands to default flat settings?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._record_undo_snapshot()
            bands = [
                EQBand(1, True, "Low Shelf", 100.0, 0.0, 0.707),
                EQBand(2, True, "Peak", 1000.0, 0.0, 1.414),
                EQBand(3, True, "High Shelf", 8000.0, 0.0, 0.707),
            ]
            self.preamp_spin.setValue(0.0)
            self.eq_plot.set_bands(bands, preamp_db=0.0)
            self._populate_table(bands)
            self.preset_manager.update_active_preset_content(0.0, bands)
            self.preset_manager.schedule_apo_export(0.0, bands, bypassed=self.is_eq_bypassed)
            self._update_resultant_curve()
            self._commit_state()

    # -------------------------------------------------------------------------
    # Header & Settings Handlers
    # -------------------------------------------------------------------------
    def _toggle_spectrum(self) -> None:
        is_on = self.spectrum_toggle_btn.isChecked()
        self.spectrum_worker.set_enabled(is_on)
        self.eq_plot.spectrum_curve.setVisible(is_on)
        if is_on:
            self.spectrum_toggle_btn.setText("Spectrum: ON")
            self.spectrum_toggle_btn.setObjectName("activeEqBtn")
        else:
            self.spectrum_toggle_btn.setText("Spectrum: OFF")
            self.spectrum_toggle_btn.setObjectName("bypassedEqBtn")
        self.spectrum_toggle_btn.setStyle(self.spectrum_toggle_btn.style())

    def _on_plot_spectrum_toggled(self, visible: bool) -> None:
        self.spectrum_toggle_btn.blockSignals(True)
        self.spectrum_toggle_btn.setChecked(visible)
        self.spectrum_worker.set_enabled(visible)
        if visible:
            self.spectrum_toggle_btn.setText("Spectrum: ON")
            self.spectrum_toggle_btn.setObjectName("activeEqBtn")
        else:
            self.spectrum_toggle_btn.setText("Spectrum: OFF")
            self.spectrum_toggle_btn.setObjectName("bypassedEqBtn")
        self.spectrum_toggle_btn.setStyle(self.spectrum_toggle_btn.style())
        self.spectrum_toggle_btn.blockSignals(False)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(
            self.preset_manager.apo_target_path,
            self
        )
        dialog.sigPathUpdated.connect(self._on_apo_path_changed)
        dialog.exec()

    def _on_apo_path_changed(self, new_path: str) -> None:
        self.preset_manager.apo_target_path = new_path
        self.preset_manager.save_settings()
        self.preset_manager.schedule_apo_export(
            self.preamp_spin.value(), self.eq_plot.bands, bypassed=self.is_eq_bypassed
        )

    def _on_audio_device_changed(self, dev_idx: int) -> None:
        self._on_capture_device_selected(dev_idx, "")

    # -------------------------------------------------------------------------
    # AutoEQ & Squiglink Database Handlers
    # -------------------------------------------------------------------------
    def _switch_sidebar_tab(self, index: int) -> None:
        self.sidebar_stack.slideToIndex(index)
        self.sidebar_tab_bar.set_current_index(index, animate=True)

    def _open_global_search(self) -> None:
        initial_q = self.autoeq_model_combo.currentText().strip()
        dlg = GlobalSearchDialog(self.squig_client, initial_query=initial_q, preset_manager=self.preset_manager, parent=self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.selected_result:
            if getattr(dlg, "load_role", "source") == "target":
                self._on_global_search_applied_as_target(dlg.selected_result)
            else:
                self._on_global_search_applied(dlg.selected_result)

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

        self.autoeq_target_iem_combo.set_selected_model(target_m)
        self.autoeq_target_iem_badge.setText(f"Rig: {res.rig} [{'Matched' if is_same else 'External'}]")
        self._fetch_selected_curves()

    def _on_target_mode_changed(self, idx: int) -> None:
        self._autoeq_target_mode = idx
        self.autoeq_target_stack.setCurrentIndex(idx)
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

    def _compute_preset_resultant_curve(self, preset_name: str) -> Optional[Tuple[List[float], List[float]]]:
        p_data = self.preset_manager.presets.get(preset_name)
        if not p_data:
            return None
        bands = [EQBand.from_dict(b) for b in p_data.get("bands", [])]
        meta = p_data.get("autoeq") or {}
        db_id = meta.get("database_id", "")
        m_file = meta.get("model_file") or meta.get("model_name", "")
        chan = meta.get("channel_mode", "avg")
        base_u = meta.get("base_url", "")

        from offline_db import get_offline_db
        off_db = get_offline_db()
        m_res = off_db.get_measurement(db_id, m_file, chan) if (db_id and m_file) else None
        if not m_res and m_file and base_u:
            try:
                temp_m = SquigModel(name=m_file, file_name=m_file, brand="", full_name=m_file, display_name=m_file)
                m_res = self.squig_client.fetch_iem_measurement(base_u, temp_m, chan)
            except Exception:
                m_res = None

        if m_res and len(m_res[0]) > 0:
            pmf, pms = m_res
            eq_resp = compute_composite_response(bands, np.asarray(pmf, dtype=np.float64), preamp_db=0.0, fs=self.sample_rate)
            res_spls = np.asarray(pms, dtype=np.float64) + eq_resp
            return (list(pmf), res_spls.tolist())
        else:
            f_grid = np.geomspace(20.0, 20000.0, num=300)
            eq_resp = compute_composite_response(bands, f_grid, preamp_db=0.0, fs=self.sample_rate)
            return (f_grid.tolist(), eq_resp.tolist())

    def _build_current_autoeq_preset_meta(self) -> Tuple[str, Dict[str, Any]]:
        model = self.autoeq_model_combo.currentData()
        chan = self.autoeq_chan_combo.currentData() or "avg"
        m_name = model.name if model else "IEM"
        mode = getattr(self, "_autoeq_target_mode", 0)

        if mode == 0:
            target = self.autoeq_target_combo.currentData()
            t_name = target.name if target else "Target"
            preset_name = f"AutoEQ: {m_name} ({t_name})"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "target_curve",
                "database_id": self._active_db_info.get("id", ""),
                "database_name": self._active_db_info.get("name", ""),
                "base_url": self._active_db_info.get("base_url", ""),
                "model_name": m_name,
                "model_file": model.file_name if model else "",
                "target_name": t_name,
                "target_file": target.file_name if target else (target.name if target else ""),
                "channel_mode": chan,
            }
        elif mode == 1:
            target_iem = self.autoeq_target_iem_combo.currentData()
            t_name = target_iem.name if target_iem else "Target IEM"
            t_url = getattr(target_iem, "base_url", "") or self._active_db_info.get("base_url", "")
            t_prov = getattr(target_iem, "provider_name", "") or self._active_db_info.get("name", "")
            t_rig = getattr(target_iem, "rig", "") or self._active_db_info.get("rig", "")
            t_chan = self.autoeq_target_iem_chan_combo.currentData() or "avg"
            preset_name = f"AutoEQ: {m_name} -> {t_name}"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "iem_tuning",
                "database_id": self._active_db_info.get("id", ""),
                "database_name": self._active_db_info.get("name", ""),
                "base_url": self._active_db_info.get("base_url", ""),
                "model_name": m_name,
                "model_file": model.file_name if model else "",
                "channel_mode": chan,
                "target_name": f"{t_name} (Tuning)",
                "target_iem_name": t_name,
                "target_iem_file": target_iem.file_name if target_iem else "",
                "target_iem_channel": t_chan,
                "target_iem_base_url": t_url,
                "target_iem_provider": t_prov,
                "target_iem_rig": t_rig,
            }
        else:  # mode == 2
            target_prof = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
            preset_name = f"AutoEQ: {m_name} -> {target_prof}"
            autoeq_meta = {
                "is_autoeq": True,
                "target_mode": "resultant_profile",
                "database_id": self._active_db_info.get("id", ""),
                "database_name": self._active_db_info.get("name", ""),
                "base_url": self._active_db_info.get("base_url", ""),
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
        self.autoeq_status_label.setText(f"Offline sync complete! {total_meas + total_targets:,} curves available.")
        self.autoeq_status_label.setStyleSheet("color: #ffffff; font-size: 11px;")

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
                "id": res.provider_name.lower().replace(" ", "_"),
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
                    "id": f"custom_{int(time.time())}",
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
                        "id": f"{c.get('username') or p_name.lower().replace(' ', '_')}_{cat_type.lower()}",
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
            self.autoeq_db_star_btn.setText("☆")
            self.autoeq_db_star_btn.setStyleSheet("""
                QPushButton {
                    color: #3f3f46;
                    font-size: 16px;
                    border: none;
                    background: transparent;
                    padding: 0px;
                    text-align: center;
                }
            """)
            return
        self.autoeq_db_star_btn.setEnabled(True)
        db_id = db_data.get("id", "")
        is_fav = self.preset_manager.is_favorite_database(db_id)
        if is_fav:
            self.autoeq_db_star_btn.setText("★")
            self.autoeq_db_star_btn.setStyleSheet("""
                QPushButton {
                    color: #facc15;
                    font-size: 16px;
                    border: none;
                    background: transparent;
                    padding: 0px;
                    text-align: center;
                }
                QPushButton:hover {
                    color: #fde047;
                }
            """)
        else:
            self.autoeq_db_star_btn.setText("☆")
            self.autoeq_db_star_btn.setStyleSheet("""
                QPushButton {
                    color: #71717a;
                    font-size: 16px;
                    border: none;
                    background: transparent;
                    padding: 0px;
                    text-align: center;
                }
                QPushButton:hover {
                    color: #f4f4f5;
                }
            """)

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
                background-color: {"#1c190c" if t_active else "#09090b"};
                color: {"#facc15" if t_active else "#52525b"};
                border: 1px solid {"#854d0e" if t_active else "#27272a"};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: {"600" if t_active else "normal"};
            }}
            QPushButton:hover {{
                border-color: {"#facc15" if t_active else "#3f3f46"};
            }}
        """)

        t2_active = self.undergraph_target2_btn.isChecked()
        self.undergraph_target2_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#1e1428" if t2_active else "#09090b"};
                color: {"#c084fc" if t2_active else "#52525b"};
                border: 1px solid {"#7e22ce" if t2_active else "#27272a"};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: {"600" if t2_active else "normal"};
            }}
            QPushButton:hover {{
                border-color: {"#c084fc" if t2_active else "#3f3f46"};
            }}
        """)

        m_active = self.undergraph_meas_btn.isChecked()
        self.undergraph_meas_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#18181b" if m_active else "#09090b"};
                color: {"#d4d4d8" if m_active else "#52525b"};
                border: 1px solid {"#52525b" if m_active else "#27272a"};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: {"600" if m_active else "normal"};
            }}
            QPushButton:hover {{
                border-color: {"#a1a1aa" if m_active else "#3f3f46"};
            }}
        """)

        r_active = self.undergraph_resultant_btn.isChecked()
        self.undergraph_resultant_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#081d28" if r_active else "#09090b"};
                color: {"#38bdf8" if r_active else "#52525b"};
                border: 1px solid {"#0284c7" if r_active else "#27272a"};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: {"600" if r_active else "normal"};
            }}
            QPushButton:hover {{
                border-color: {"#38bdf8" if r_active else "#3f3f46"};
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

            # No redundant confirmation dialogue - directly open curves assignment dialog!
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
        menu.setStyleSheet("""
            QMenu {
                background-color: #09090b;
                border: 1px solid #27272a;
                border-radius: 6px;
                color: #f4f4f5;
                padding: 4px;
                font-size: 11px;
            }
            QMenu::item {
                padding: 5px 14px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #ffffff;
                color: #000000;
                font-weight: 600;
            }
            QMenu::separator {
                height: 1px;
                background-color: #27272a;
                margin: 4px 6px;
            }
        """)

        curr_t2 = meta.get("target2_name") or meta.get("target2_file") or ""
        p_name = meta.get("database_name") or "Provider"
        lbl_head = menu.addAction(f"Add Target 2 ({p_name}):")
        lbl_head.setEnabled(False)
        menu.addSeparator()

        for t in targets:
            t_name = t.name
            t_file = t.file_name
            is_active = bool(curr_t2 and (curr_t2.lower() == t_name.lower() or curr_t2.lower() == t_file.lower()))
            prefix = "✓  " if is_active else "    "
            cat_tag = f"[{t.category}] " if t.category else ""
            act = menu.addAction(f"{prefix}{cat_tag}{t_name}")
            act.triggered.connect(
                lambda checked=False, tf=t_file, tn=t_name: self._set_preset_target2(active_preset, tf, tn)
            )

        if curr_t2:
            menu.addSeparator()
            act_rm = menu.addAction("✕  Remove Target 2")
            act_rm.triggered.connect(lambda: self._set_preset_target2(active_preset, None, None))

        menu.addSeparator()
        act_config = menu.addAction("Configure All Reference Curves...")
        act_config.triggered.connect(lambda: self._open_assign_curves_dialog(active_preset))

        menu.exec(self.undergraph_add_target_btn.mapToGlobal(QtCore.QPoint(0, self.undergraph_add_target_btn.height() + 2)))

    def _set_preset_target2(self, preset_name: str, target_file: Optional[str], target_name: Optional[str]) -> None:
        meta = self.preset_manager.get_preset_metadata(preset_name) or {}
        if not meta:
            return
        meta["target2_file"] = target_file or ""
        meta["target2_name"] = target_name or ""
        self.preset_manager.set_preset_metadata(preset_name, meta)

        if preset_name == self.preset_manager.active_preset_name:
            if target_file:
                base_u = meta.get("base_url", "")
                from offline_db import get_offline_db
                off_db = get_offline_db()
                t2_res = off_db.get_target(meta.get("database_id", ""), target_file, target_name)
                if not t2_res and base_u:
                    try:
                        t2_res = self.squig_client.fetch_target_curve(
                            base_u, SquigTarget(name=target_name or target_file, file_name=target_file)
                        )
                    except Exception:
                        t2_res = None

                if t2_res and len(t2_res[0]) > 0 and len(t2_res[1]) > 0:
                    t2_f, t2_s = t2_res
                    t2_f_arr = np.asarray(t2_f, dtype=np.float64)
                    t2_s_arr = np.asarray(t2_s, dtype=np.float64)
                    mid_mask2 = (t2_f_arr >= 200.0) & (t2_f_arr <= 2000.0)
                    target2_norm = float(np.mean(t2_s_arr[mid_mask2])) if np.any(mid_mask2) else float(np.mean(t2_s_arr))
                    disp_target2 = t2_s_arr - target2_norm
                    self.eq_plot.set_target2_overlay(t2_f_arr.tolist(), disp_target2.tolist(), label=target_name or "Target 2")
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

        if self._db_worker and self._db_worker.isRunning():
            self._db_worker.quit()
            self._db_worker.wait(400)

        self._db_worker = DatabaseIndexWorker(self.squig_client, base_url, force_refresh, parent=self)
        self._db_worker.sigIndexLoaded.connect(self._on_database_index_loaded)
        self._db_worker.sigIndexFailed.connect(self._on_database_index_failed)
        self._db_worker.start()

    def _on_database_index_loaded(self, models: List[SquigModel], targets: List[SquigTarget], rig: str) -> None:
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
            self.autoeq_target_iem_combo.set_models(models)
            self.autoeq_target_iem_combo.set_provider_info(
                self._active_db_info.get("base_url", ""),
                self._active_db_info.get("name", ""),
                rig,
            )
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
            if matched_model:
                self._fetch_selected_curves()

            # If pending specified any targets, try to match target or keep first
            if pending.targets and targets:
                for p_targ in pending.targets:
                    found = False
                    for t_idx, t in enumerate(targets):
                        if t.name.lower() == p_targ.lower():
                            self.autoeq_target_combo.setCurrentIndex(t_idx)
                            found = True
                            break
                    if found:
                        break
        elif self._pending_restore_model_file or self._pending_restore_target_file:
            self._apply_pending_autoeq_restore()
        else:
            self.autoeq_model_combo.set_selected_model(None)

        self.autoeq_status_label.setText(f"Loaded {len(models)} models, {len(targets)} targets.")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")

    def _on_database_index_failed(self, err: str) -> None:
        self.autoeq_db_combo.setEnabled(True)
        self.autoeq_status_label.setText(f"Database error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _on_searchable_model_selected(self, model: SquigModel) -> None:
        self._fetch_selected_curves()

    def _on_autoeq_model_changed(self) -> None:
        self._fetch_selected_curves()

    def _on_autoeq_target_changed(self) -> None:
        self._fetch_selected_curves()

    def _fetch_selected_curves(self) -> None:
        if getattr(self, "_suppress_curve_fetch", False) or getattr(self.eq_plot, "_is_sweeping", False):
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

        if self._curve_worker and self._curve_worker.isRunning():
            self._curve_worker.quit()
            self._curve_worker.wait(300)

        self._curve_worker = CurveFetchWorker(
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
        self._curve_worker.sigCurvesLoaded.connect(self._on_curves_loaded)
        self._curve_worker.sigCurveFailed.connect(self._on_curves_failed)
        self._curve_worker.start()

    def _on_curves_loaded(self, meas_data: Tuple, target_data: Tuple) -> None:
        if getattr(self.eq_plot, "_is_sweeping", False):
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

        dm, dt, dr = self._compute_normalized_display_curves(mf, ms, tf, ts, self.eq_plot.bands)

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

        if has_dm:
            self._disp_meas_freqs = np.asarray(dm[0], dtype=np.float64)
            self._disp_meas_spls = np.asarray(dm[1], dtype=np.float64)
            self.eq_plot.animate_overlay_curve("meas", dm[0], dm[1], label=m_label)
        else:
            self._disp_meas_freqs = None
            self._disp_meas_spls = None
            self.eq_plot.animate_overlay_curve("meas", None, None)

        if has_dt:
            self.eq_plot.animate_overlay_curve("target", dt[0], dt[1], label=t_label)
        else:
            self.eq_plot.animate_overlay_curve("target", None, None)

        if has_dr:
            r_label = f"{m_label} + EQ" if m_label else "Resultant + EQ"
            self.eq_plot.animate_overlay_curve("resultant", dr[0], dr[1], label=r_label)
        else:
            self.eq_plot.animate_overlay_curve("resultant", None, None)

        self.eq_plot.set_overlays_visible(
            has_dm, has_dt, has_dr,
            show_target2=self.undergraph_target2_btn.isChecked(),
        )
        self._style_undergraph_toggle_buttons()

        if mf and tf:
            if "Fitted" not in self.autoeq_status_label.text() and "Fitting" not in self.autoeq_status_label.text():
                self.autoeq_status_label.setText("Traces aligned. Ready to compute AutoEQ.")
                self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")
        elif mf:
            if "Fitted" not in self.autoeq_status_label.text():
                self.autoeq_status_label.setText(f"Measurement loaded for '{m_label}'.")
                self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")
        elif tf:
            if model:
                self.autoeq_status_label.setText(f"Measurement unavailable for '{m_label}'. Select another provider.")
                self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")
        else:
            self.autoeq_status_label.setText("No curve data found for this selection.")
            self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _on_curves_failed(self, err: str) -> None:
        self.autoeq_status_label.setText(f"Curve fetch error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")

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
                self.autoeq_status_label.setText("Please select a headphone / IEM model.")
                self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                return

            if mode == 0:
                target = self.autoeq_target_combo.currentData() if hasattr(self, "autoeq_target_combo") else None
                if not target:
                    self.autoeq_status_label.setText("Please select a target curve.")
                    self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                    return
            elif mode == 1:
                target_iem = self.autoeq_target_iem_combo.currentData() if hasattr(self, "autoeq_target_iem_combo") else None
                if not target_iem:
                    self.autoeq_status_label.setText("Please select a target IEM / headphone to clone.")
                    self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                    return
            elif mode == 2:
                prof_name = ""
                if hasattr(self, "autoeq_target_profile_combo"):
                    prof_name = self.autoeq_target_profile_combo.currentData() or self.autoeq_target_profile_combo.currentText().strip()
                if not prof_name:
                    self.autoeq_status_label.setText("Please select a previous profile to match.")
                    self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                    return

            if not mf:
                if self._curve_worker and self._curve_worker.isRunning():
                    self.autoeq_status_label.setText("Curve is currently downloading, please wait...")
                    self.autoeq_status_label.setStyleSheet("color: #facc15; font-size: 11px;")
                else:
                    m_disp = model.display_name if model else "this model"
                    self.autoeq_status_label.setText(f"Measurement unavailable for '{m_disp}'. Select another provider.")
                    self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                return
            if not tf:
                if self._curve_worker and self._curve_worker.isRunning():
                    self.autoeq_status_label.setText("Target curve is currently downloading, please wait...")
                    self.autoeq_status_label.setStyleSheet("color: #facc15; font-size: 11px;")
                else:
                    self.autoeq_status_label.setText("Target response unavailable. Please select another target.")
                    self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px; font-weight: bold;")
                return

        num_bands = self.autoeq_bands_spin.value()
        use_bass_shelf = self.autoeq_bass_shelf_cb.isChecked()
        max_boost = self.autoeq_max_boost_spin.value()
        max_freq = self.autoeq_max_freq_spin.value()

        # Turn button into Stop Optimization button (stays active so user can stop anytime)
        self.autoeq_compute_btn.setEnabled(True)
        self.autoeq_compute_btn.setText("Stop Optimization")
        self.autoeq_compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #1c1917;
                color: #f87171;
                border: 1px solid #ef4444;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #27272a;
                color: #ffffff;
                border-color: #f87171;
            }
        """)
        self.autoeq_status_label.setText(f"Fitting {num_bands} bands with SciPy...")
        self.autoeq_status_label.setStyleSheet("color: #a1a1aa; font-size: 11px;")

        self._autoeq_worker = AutoEQWorker(
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
        )
        self._autoeq_worker.sigOptimized.connect(self._on_autoeq_optimized)
        self._autoeq_worker.sigFailed.connect(self._on_autoeq_failed)
        self._autoeq_worker.start()

    def _cancel_autoeq_optimization(self) -> None:
        if self._autoeq_worker and self._autoeq_worker.isRunning():
            self._autoeq_worker.cancel()
        if self._curve_worker and self._curve_worker.isRunning():
            self._curve_worker.quit()
            self._curve_worker.wait(200)
        self._reset_compute_btn_state()
        self.autoeq_status_label.setText("Operation stopped.")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")

    def _reset_compute_btn_state(self) -> None:
        self.autoeq_compute_btn.setEnabled(True)
        self.autoeq_compute_btn.setText("Compute AutoEQ")
        self.autoeq_compute_btn.setStyleSheet("")

    def _on_autoeq_optimized(self, result: OptimizationResult) -> None:
        self._record_undo_snapshot()
        self._reset_compute_btn_state()

        # 1. Update plot canvas with smooth sweep transition & node creation
        self.eq_plot.animate_to_preset(result.bands, result.preamp_db)

        # 2. Update Preamp controls
        self._apply_preamp_value(result.preamp_db)

        # 3. Update bottom table
        self._populate_table(result.bands)

        # 4. Immediate debounced Equalizer APO writer
        preset_name, autoeq_meta = self._build_current_autoeq_preset_meta()

        self.preset_manager.schedule_apo_export(
            result.preamp_db, result.bands, bypassed=self.is_eq_bypassed
        )

        # 5. Automatically save into presets manager with metadata
        self.preset_manager.save_or_update_preset(
            preset_name, result.preamp_db, result.bands, autoeq_metadata=autoeq_meta
        )

        # 6. Update resultant frequency response overlay
        self._update_resultant_curve()
        self.undergraph_meas_btn.setChecked(True)
        self.undergraph_target_btn.setChecked(True)
        self.undergraph_resultant_btn.setChecked(True)
        self._style_undergraph_toggle_buttons()
        self._commit_state()

        # 7. Status message (compact single-line format)
        status = (
            f"Fitted {len(result.bands)} bands ({result.elapsed_ms:.0f}ms) • RMS {result.final_rms_db:.2f}dB • Preamp: {result.preamp_db:+.1f}dB"
        )
        self.autoeq_status_label.setText(status)
        self.autoeq_status_label.setStyleSheet("color: #ffffff; font-weight: 600; font-size: 11px;")

    def _on_autoeq_failed(self, err: str) -> None:
        self._reset_compute_btn_state()
        self.autoeq_status_label.setText(f"Optimization error: {err}")
        self.autoeq_status_label.setStyleSheet("color: #f87171; font-size: 11px;")

    def _save_autoeq_profile(self) -> None:
        default_name, autoeq_meta = self._build_current_autoeq_preset_meta()

        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save AutoEQ Profile",
            "Profile Name:",
            text=default_name,
        )
        if ok and name.strip():
            self.preset_manager.save_or_update_preset(
                name.strip(), self.preamp_spin.value(), self.eq_plot.bands, autoeq_metadata=autoeq_meta
            )
            self.autoeq_status_label.setText(f"Saved profile '{name.strip()}' to Presets!")
            self.autoeq_status_label.setStyleSheet("color: #ffffff; font-size: 11px;")

    def _clear_autoeq_overlays(self) -> None:
        if self._autoeq_worker and self._autoeq_worker.isRunning():
            self._autoeq_worker.cancel()
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
        self.autoeq_status_label.setText("Overlays cleared.")
        self.autoeq_status_label.setStyleSheet("color: #71717a; font-size: 11px;")


def main():
    def qt_message_filter(mode, context, message):
        # Silence benign Qt/pyqtgraph font point size warning when style pixel sizes are used
        if "setPointSize" in message and "<= 0" in message:
            return
        if mode == QtCore.QtMsgType.QtFatalMsg:
            sys.stderr.write(f"Fatal: {message}\n")
            sys.exit(1)
        elif mode == QtCore.QtMsgType.QtCriticalMsg:
            sys.stderr.write(f"Critical: {message}\n")
        elif mode == QtCore.QtMsgType.QtWarningMsg:
            sys.stderr.write(f"Warning: {message}\n")

    QtCore.qInstallMessageHandler(qt_message_filter)

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
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#000000"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#060608"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#09090b"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#121214"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#ffffff"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#000000"))
    app.setPalette(palette)

    window = MainWindow()
    if app_icon and not app_icon.isNull():
        window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
