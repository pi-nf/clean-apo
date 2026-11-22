"""
Interactive pyqtgraph frequency response canvas, draggable filter nodes, live spectrum overlay,
and cubic-smoothed sweeping graph transition animations.
"""

from __future__ import annotations
import html
import sys
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from biquad import EQBand, FilterType, compute_composite_response, compute_filter_response

MIN_FREQ = 20.0
MAX_FREQ = 20000.0
MIN_GAIN = -20.0
MAX_GAIN = 20.0
X_MIN = math.log10(MIN_FREQ)
X_MAX = math.log10(MAX_FREQ)

_NOTE_NAMES: Tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
)


def _freq_to_musical_note(freq_hz: float) -> str:
    """Converts frequency in Hz to nearest equal-temperament musical pitch (A4 = 440 Hz)."""
    if not isinstance(freq_hz, (int, float)) or not math.isfinite(freq_hz) or freq_hz <= 0.0:
        return ""
    midi = round(12.0 * math.log2(freq_hz / 440.0) + 69.0)
    note = _NOTE_NAMES[midi % 12]
    octave = (midi // 12) - 1
    return f"{note}{octave}"


def _format_freq(freq_hz: float) -> str:
    """Formats a frequency value consistently in Hz or kHz."""
    if not isinstance(freq_hz, (int, float)) or not math.isfinite(freq_hz) or freq_hz <= 0.0:
        return "0 Hz"
    if freq_hz >= 10000.0:
        return f"{freq_hz / 1000.0:.1f} kHz"
    if freq_hz >= 1000.0:
        return f"{freq_hz / 1000.0:.2f} kHz"
    if freq_hz >= 100.0:
        return f"{freq_hz:.0f} Hz"
    return f"{freq_hz:.1f} Hz"


def _mono_font(
    point_size: int = 8,
    weight: QtGui.QFont.Weight = QtGui.QFont.Weight.Normal,
) -> QtGui.QFont:
    font = QtGui.QFont("Cascadia Mono", point_size, weight)
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setFamilies(["Cascadia Mono", "Consolas", "Courier New", "monospace"])
    return font


def _sanitize_band(band: EQBand) -> EQBand:
    """Returns a deep copy of band with finite, safe numerical parameters."""
    b = EQBand.from_dict(band.to_dict())
    if not isinstance(b.freq, (int, float)) or not math.isfinite(b.freq) or b.freq <= 0.0:
        b.freq = 1000.0
    if not isinstance(b.gain, (int, float)) or not math.isfinite(b.gain):
        b.gain = 0.0
    if not isinstance(b.q, (int, float)) or not math.isfinite(b.q) or b.q <= 0.0:
        b.q = 1.414
    ftype = FilterType.from_str(b.filter_type)
    if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
        b.gain = 0.0
    return b


def _safe_band_coords(
    band: EQBand,
    y_min: float = MIN_GAIN,
    y_max: float = MAX_GAIN,
) -> Tuple[float, float]:
    """Maps a band's (freq, gain) to clamped plot view coordinates (log10(freq), gain)."""
    freq = float(band.freq) if isinstance(band.freq, (int, float)) and math.isfinite(band.freq) else 1000.0
    freq = max(MIN_FREQ, min(MAX_FREQ, freq))
    ftype = FilterType.from_str(band.filter_type)
    if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
        gain = 0.0
    else:
        gain = float(band.gain) if isinstance(band.gain, (int, float)) and math.isfinite(band.gain) else 0.0
        gain = max(float(y_min), min(float(y_max), gain))
    return math.log10(freq), gain


def _sanitize_curve_data(
    freqs: Optional[Sequence[float]],
    spls: Optional[Sequence[float]],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Validates, filters to finite [MIN_FREQ, MAX_FREQ], and sorts overlay curve arrays ascending."""
    if freqs is None or spls is None:
        return None
    try:
        f_arr = np.asarray(freqs, dtype=np.float64).ravel()
        s_arr = np.asarray(spls, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if f_arr.size == 0 or s_arr.size == 0 or f_arr.size != s_arr.size:
        return None
    valid = np.isfinite(f_arr) & np.isfinite(s_arr) & (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
    if not np.any(valid):
        return None
    f_clean = f_arr[valid]
    s_clean = s_arr[valid]
    if f_clean.size > 1 and np.any(np.diff(f_clean) < 0):
        order = np.argsort(f_clean, kind="mergesort")
        f_clean = f_clean[order]
        s_clean = s_clean[order]
    return f_clean, s_clean


def _get_assets_dir() -> str:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, "assets")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def ensure_check_assets() -> Tuple[str, str]:
    assets_dir = _get_assets_dir()
    white_path = os.path.join(assets_dir, "check_white.png")
    black_path = os.path.join(assets_dir, "check_black.png")

    def _make_icon(color_hex: str, out_path: str):
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img = QtGui.QImage(16, 16, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(color_hex), 2.2)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            path = QtGui.QPainterPath()
            path.moveTo(3.0, 8.5)
            path.lineTo(6.5, 12.0)
            path.lineTo(13.0, 4.0)
            p.drawPath(path)
            p.end()
            img.save(out_path)
        except Exception:
            pass

    if not os.path.exists(white_path):
        _make_icon("#ffffff", white_path)
    if not os.path.exists(black_path):
        _make_icon("#000000", black_path)

    return white_path.replace("\\", "/"), black_path.replace("\\", "/")


def ensure_scrollbar_assets() -> Tuple[str, str, str, str]:
    assets_dir = _get_assets_dir()
    up_path = os.path.join(assets_dir, "scroll_up.png")
    down_path = os.path.join(assets_dir, "scroll_down.png")
    left_path = os.path.join(assets_dir, "scroll_left.png")
    right_path = os.path.join(assets_dir, "scroll_right.png")

    def _make_arrow(points: List[Tuple[float, float]], out_path: str):
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img = QtGui.QImage(12, 12, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(QtCore.Qt.GlobalColor.transparent)
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor("#8e8e93"))
            path = QtGui.QPainterPath()
            path.moveTo(points[0][0], points[0][1])
            for pt in points[1:]:
                path.lineTo(pt[0], pt[1])
            path.closeSubpath()
            p.drawPath(path)
            p.end()
            img.save(out_path)
        except Exception:
            pass

    if not os.path.exists(up_path):
        _make_arrow([(6.0, 3.2), (2.5, 8.8), (9.5, 8.8)], up_path)
    if not os.path.exists(down_path):
        _make_arrow([(2.5, 3.2), (9.5, 3.2), (6.0, 8.8)], down_path)
    if not os.path.exists(left_path):
        _make_arrow([(3.2, 6.0), (8.8, 2.5), (8.8, 9.5)], left_path)
    if not os.path.exists(right_path):
        _make_arrow([(3.2, 2.5), (8.8, 6.0), (3.2, 9.5)], right_path)

    return (
        up_path.replace("\\", "/"),
        down_path.replace("\\", "/"),
        left_path.replace("\\", "/"),
        right_path.replace("\\", "/"),
    )


def ensure_chevron_assets() -> Tuple[str, str]:
    assets_dir = _get_assets_dir()
    down_svg = os.path.join(assets_dir, "chevron_down.svg")
    down_hover_svg = os.path.join(assets_dir, "chevron_down_hover.svg")

    svg_normal = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#a1a1aa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M4 6.5L8 10.5L12 6.5" />
</svg>'''

    svg_hover = '''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M4 6.5L8 10.5L12 6.5" />
</svg>'''

    if not os.path.exists(down_svg) or not os.path.exists(down_hover_svg):
        try:
            os.makedirs(assets_dir, exist_ok=True)
            with open(down_svg, "w", encoding="utf-8") as f:
                f.write(svg_normal)
            with open(down_hover_svg, "w", encoding="utf-8") as f:
                f.write(svg_hover)
        except Exception:
            pass

    return down_svg.replace("\\", "/"), down_hover_svg.replace("\\", "/")


CHECK_WHITE_PATH, CHECK_BLACK_PATH = ensure_check_assets()
SCROLL_UP_PATH, SCROLL_DOWN_PATH, SCROLL_LEFT_PATH, SCROLL_RIGHT_PATH = ensure_scrollbar_assets()
CHEVRON_DOWN_PATH, CHEVRON_DOWN_HOVER_PATH = ensure_chevron_assets()


def _build_context_menu_qss() -> str:
    """Unified quiet graphite context menu QSS (4px grid, 4px radius, three text roles)."""
    return (
        "QMenu {"
        "  background-color: #121214;"
        "  color: #ededed;"
        "  border: 1px solid #27272a;"
        "  border-radius: 4px;"
        "  padding: 4px;"
        "  font-family: \"Segoe UI\", sans-serif;"
        "  font-size: 12px;"
        "}"
        "QMenu::item {"
        "  padding: 6px 20px 6px 28px;"
        "  border-radius: 4px;"
        "  color: #ededed;"
        "}"
        "QMenu::item:disabled {"
        "  color: #71717a;"
        "}"
        "QMenu::item:selected {"
        "  background-color: #1c1c1f;"
        "  color: #ededed;"
        "}"
        "QMenu::separator {"
        "  height: 1px;"
        "  background: #27272a;"
        "  margin: 4px 8px;"
        "}"
        "QMenu::indicator {"
        "  width: 14px;"
        "  height: 14px;"
        "  left: 8px;"
        "}"
        "QMenu::indicator:unchecked { image: none; }"
        f"QMenu::indicator:checked {{ image: url('{CHECK_WHITE_PATH}'); }}"
        f"QMenu::indicator:checked:selected {{ image: url('{CHECK_WHITE_PATH}'); }}"
    )


class EQGridItem(pg.GraphicsObject):
    """
    Subtle background grid item rendered inside the ViewBox at zValue = 0,
    strictly behind curves (zValue = 10) and band nodes (zValue = 25).
    Guarantees grid lines never cut across or through node badges or curves.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, y_step: int = 10, parent=None):
        super().__init__(parent)
        self.setZValue(0)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.y_step = max(1, int(y_step))
        self.show_x_major = True
        self.show_x_minor = True
        self.show_y = True

    def set_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        y_step: Optional[int] = None,
    ) -> None:
        self.prepareGeometryChange()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        if y_step is not None:
            self.y_step = max(1, int(y_step))
        self.update()

    def set_y_step(self, step: int) -> None:
        new_step = max(1, int(step))
        if self.y_step != new_step:
            self.y_step = new_step
            self.update()

    def set_show_x_major(self, show: bool) -> None:
        if self.show_x_major != bool(show):
            self.show_x_major = bool(show)
            self.update()

    def set_show_x_minor(self, show: bool) -> None:
        if self.show_x_minor != bool(show):
            self.show_x_minor = bool(show)
            self.update()

    def set_show_x(self, show: bool) -> None:
        val = bool(show)
        if self.show_x_major != val or self.show_x_minor != val:
            self.show_x_major = val
            self.show_x_minor = val
            self.update()

    @property
    def show_x(self) -> bool:
        return self.show_x_major

    def set_show_y(self, show: bool) -> None:
        if self.show_y != bool(show):
            self.show_y = bool(show)
            self.update()

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(self.x_min, self.y_min, self.x_max - self.x_min, self.y_max - self.y_min)

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        major_pen = QtGui.QPen(QtGui.QColor(39, 39, 42, 200), 1.0)
        major_pen.setCosmetic(True)
        minor_pen = QtGui.QPen(QtGui.QColor(39, 39, 42, 100), 1.0)
        minor_pen.setCosmetic(True)

        # Horizontal grid lines snapped to active dB step (10 dB at ±20, 4 dB at ±12)
        if self.show_y:
            painter.setPen(major_pen)
            step = max(1, int(self.y_step))
            y_start = int(math.ceil(self.y_min / float(step)) * step)
            y_end = int(math.floor(self.y_max / float(step)) * step)
            for g in range(y_start, y_end + 1, step):
                painter.drawLine(QtCore.QPointF(self.x_min, float(g)), QtCore.QPointF(self.x_max, float(g)))

        # Vertical minor/intermediate grid lines (indicating 30, 40, ..., 3k, 4k, 6k, 7k, 8k, 9k)
        if self.show_x_minor:
            painter.setPen(minor_pen)
            major_set = {20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0}
            for decade in [10, 100, 1000, 10000]:
                for m in range(2, 10):
                    f = float(m * decade)
                    if MIN_FREQ <= f <= MAX_FREQ and f not in major_set:
                        x = math.log10(f)
                        painter.drawLine(QtCore.QPointF(x, self.y_min), QtCore.QPointF(x, self.y_max))

        # Vertical major grid lines (20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k)
        if self.show_x_major:
            painter.setPen(major_pen)
            for f in [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]:
                x = math.log10(f)
                painter.drawLine(QtCore.QPointF(x, self.y_min), QtCore.QPointF(x, self.y_max))


class EQNodeItem(pg.GraphicsObject):
    """
    Interactive draggable node plotted at (log10(Fc), Gain).
    - Mastering-room monochrome styling: crisp badge with quiet focus ring (no hover halo glow).
    - Generous circular hitbox (covers entire badge and outer padding) for effortless clicking.
    - Uses inverse parent transform instead of ItemIgnoresTransformations to prevent Qt6 hit-testing clipping.
    - Precision modifiers: Shift+Drag (5x fine), Ctrl+Drag (lock freq), Alt+Drag (lock gain).
    - Middle-click: quick bypass toggle; Double-click: reset gain to 0.0 dB; Mouse wheel: Q factor.
    - Right-click: Context menu.
    """
    sigMoved = QtCore.pyqtSignal(int, float, float)       # band_id, freq, gain
    sigQChanged = QtCore.pyqtSignal(int, float)            # band_id, q
    sigTypeChanged = QtCore.pyqtSignal(int, str)           # band_id, type
    sigBypassToggled = QtCore.pyqtSignal(int, bool)        # band_id, enabled
    sigDeleteRequested = QtCore.pyqtSignal(int)            # band_id
    sigHoverChanged = QtCore.pyqtSignal(object, bool)      # node, is_hovered
    sigDragStarted = QtCore.pyqtSignal(int)                # band_id
    sigWheelScrollStarted = QtCore.pyqtSignal()

    def __init__(
        self,
        band: EQBand,
        viewBox: pg.ViewBox,
        parent=None,
        y_min: float = MIN_GAIN,
        y_max: float = MAX_GAIN,
    ):
        super().__init__(parent)
        self.band = _sanitize_band(band)
        self.vb = viewBox
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # Badge radius: 13.0 px (26 px diameter visible badge)
        self.radius = 13.0
        self.is_dragging = False
        self.is_hovered = False
        self.is_bypassed_globally = False
        self._last_transform = None
        self._transform_sync_pending = False
        self._fade_start_opacity = 1.0

        # Precision drag state
        self._drag_anchor_x = 0.0
        self._drag_anchor_y = 0.0
        self._drag_curr_x = 0.0
        self._drag_curr_y = 0.0
        self._last_drag_view_pos: Optional[QtCore.QPointF] = None
        self._active_drag_modifiers = QtCore.Qt.KeyboardModifier.NoModifier

        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setZValue(25)

        self._update_node_position()

    def set_y_bounds(self, y_min: float, y_max: float) -> None:
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self._update_node_position()
        self.update()

    def updateTransform(self) -> None:
        """
        Maintains constant pixel size without ItemIgnoresTransformations,
        preserving full circular hit detection across the entire badge in Qt6.
        """
        self._transform_sync_pending = False
        if self.scene() is None:
            return
        p = self.parentItem()
        if p is None:
            return
        pt = p.sceneTransform()
        if pt == self._last_transform:
            return
        self._last_transform = pt
        t, ok = pt.inverted()
        if ok:
            t.setMatrix(t.m11(), t.m12(), t.m13(), t.m21(), t.m22(), t.m23(), 0.0, 0.0, t.m33())
            self.setTransform(t)

    def _update_node_position(self) -> None:
        """Positions node in pyqtgraph view coordinates (log10(freq), gain)."""
        x, y = _safe_band_coords(self.band, self.y_min, self.y_max)
        self.setPos(x, y)

    def boundingRect(self) -> QtCore.QRectF:
        r = self.radius + 10.0
        return QtCore.QRectF(-r, -r, 2.0 * r, 2.0 * r)

    def shape(self) -> QtGui.QPainterPath:
        # Generous circular hit area (20px radius / 40px diameter)
        # Guarantees clicking anywhere on or near the circle grabs the node instantly
        path = QtGui.QPainterPath()
        path.addEllipse(QtCore.QPointF(0, 0), self.radius + 7.0, self.radius + 7.0)
        return path

    def paint(self, painter: QtGui.QPainter, option, widget) -> None:
        p = self.parentItem()
        if p is not None and p.sceneTransform() != self._last_transform and not self._transform_sync_pending:
            self._transform_sync_pending = True
            QtCore.QTimer.singleShot(0, self.updateTransform)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # 1. Crisp 1.5px Focus Ring on Hover or Drag (no translucent glow halo)
        if (self.is_dragging or self.is_hovered) and not self.is_bypassed_globally:
            ring_color = QtGui.QColor("#ededed") if self.is_dragging else QtGui.QColor("#a1a1aa")
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(ring_color, 1.5))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius + 3.0, self.radius + 3.0)

        # 2. Main Badge (Mastering-Room Monochrome)
        if self.is_bypassed_globally:
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#121214")))
            painter.setPen(QtGui.QPen(QtGui.QColor("#3f3f46"), 1.2, QtCore.Qt.PenStyle.DashLine))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#71717a")))
        elif self.band.enabled:
            fill_hex = "#ffffff" if (self.is_dragging or self.is_hovered) else "#ededed"
            stroke_hex = "#0b0b0c"
            painter.setBrush(QtGui.QBrush(QtGui.QColor(fill_hex)))
            painter.setPen(QtGui.QPen(QtGui.QColor(stroke_hex), 1.2))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#0b0b0c")))
        else:
            stroke_hex = "#a1a1aa" if (self.is_dragging or self.is_hovered) else "#71717a"
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#121214")))
            painter.setPen(QtGui.QPen(QtGui.QColor(stroke_hex), 1.4, QtCore.Qt.PenStyle.DashLine))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#a1a1aa")))

        # 3. Crisp Band Number Typography (weight 600, centered)
        font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.DemiBold)
        painter.setFont(font)
        text_rect = QtCore.QRectF(-self.radius, -self.radius - 0.5, 2 * self.radius, 2 * self.radius)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter, str(self.band.band_id))

    # -------------------------------------------------------------------------
    # Mouse & Event Handling
    # -------------------------------------------------------------------------
    def hoverEnterEvent(self, event) -> None:
        self.is_hovered = True
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.update()
        self.sigHoverChanged.emit(self, True)
        event.accept()

    def hoverLeaveEvent(self, event) -> None:
        self.is_hovered = False
        self.update()
        self.sigHoverChanged.emit(self, False)
        event.accept()

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.sigDragStarted.emit(self.band.band_id)
            self.is_dragging = True
            curr_pos = self.pos()
            self._drag_anchor_x = float(curr_pos.x())
            self._drag_anchor_y = float(curr_pos.y())
            self._drag_curr_x = self._drag_anchor_x
            self._drag_curr_y = self._drag_anchor_y
            self._active_drag_modifiers = event.modifiers()
            if self.vb is not None:
                vp = self.vb.mapSceneToView(event.scenePos())
                self._last_drag_view_pos = QtCore.QPointF(vp.x(), vp.y())
            else:
                self._last_drag_view_pos = QtCore.QPointF(self._drag_anchor_x, self._drag_anchor_y)
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.update()
            self.sigHoverChanged.emit(self, True)
            event.accept()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.sigDragStarted.emit(self.band.band_id)
            self.sigBypassToggled.emit(self.band.band_id, not self.band.enabled)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.is_dragging and self.vb:
            view_pos = self.vb.mapSceneToView(event.scenePos())
            mods = event.modifiers()
            self._active_drag_modifiers = mods

            is_shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            is_ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            is_alt = bool(mods & QtCore.Qt.KeyboardModifier.AltModifier)

            if self._last_drag_view_pos is None:
                self._last_drag_view_pos = QtCore.QPointF(self._drag_anchor_x, self._drag_anchor_y)

            dx = float(view_pos.x() - self._last_drag_view_pos.x())
            dy = float(view_pos.y() - self._last_drag_view_pos.y())
            self._last_drag_view_pos = QtCore.QPointF(view_pos.x(), view_pos.y())

            scale = 0.2 if is_shift else 1.0
            if is_ctrl:
                new_x = self._drag_anchor_x
            else:
                new_x = self._drag_curr_x + dx * scale

            ftype = FilterType.from_str(self.band.filter_type)
            if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                new_y = 0.0
            elif is_alt:
                new_y = self._drag_anchor_y
            else:
                new_y = self._drag_curr_y + dy * scale

            x = max(X_MIN, min(X_MAX, new_x))
            y = 0.0 if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS) else max(self.y_min, min(self.y_max, new_y))

            self._drag_curr_x = x
            self._drag_curr_y = y

            self.setPos(x, y)
            new_freq = 10.0 ** x
            new_gain = y
            self.band.freq = new_freq
            self.band.gain = new_gain
            self.sigMoved.emit(self.band.band_id, new_freq, new_gain)
            self.sigHoverChanged.emit(self, True)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self._last_drag_view_pos = None
            self._active_drag_modifiers = QtCore.Qt.KeyboardModifier.NoModifier
            self.is_hovered = bool(self.shape().contains(event.pos()))
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.update()
            self.sigHoverChanged.emit(self, self.is_hovered)
            event.accept()
        elif event.button() == QtCore.Qt.MouseButton.MiddleButton:
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_dragging = False
            self._last_drag_view_pos = None
            self._active_drag_modifiers = QtCore.Qt.KeyboardModifier.NoModifier
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.sigDragStarted.emit(self.band.band_id)
            self.band.gain = 0.0
            self._update_node_position()
            self.update()
            self.sigMoved.emit(self.band.band_id, self.band.freq, 0.0)
            if self.is_hovered:
                self.sigHoverChanged.emit(self, True)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtWidgets.QGraphicsSceneWheelEvent) -> None:
        delta = event.delta()
        if delta == 0:
            event.ignore()
            return
        self.sigWheelScrollStarted.emit()
        if delta > 0:
            new_q = min(20.0, self.band.q * 1.1)
        else:
            new_q = max(0.1, self.band.q / 1.1)

        self.band.q = round(new_q, 3)
        self.sigQChanged.emit(self.band.band_id, self.band.q)
        self.sigHoverChanged.emit(self, True)
        self.update()
        event.accept()

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.setStyleSheet(_build_context_menu_qss())

        title_act = menu.addAction(f"Band {self.band.band_id} ({self.band.filter_type})")
        title_act.setEnabled(False)
        menu.addSeparator()

        type_menu = menu.addMenu("Filter type")
        type_menu.setStyleSheet(menu.styleSheet())
        for f_type in [FilterType.PEAK, FilterType.LOW_SHELF, FilterType.HIGH_SHELF, FilterType.HIGH_PASS, FilterType.LOW_PASS]:
            act = type_menu.addAction(f_type.value)
            act.setCheckable(True)
            act.setChecked(FilterType.from_str(self.band.filter_type) == f_type)

            def _on_type_selected(_checked: bool = False, t: str = f_type.value) -> None:
                if self.scene() is None:
                    return
                self.sigDragStarted.emit(self.band.band_id)
                self.sigTypeChanged.emit(self.band.band_id, t)

            act.triggered.connect(_on_type_selected)

        bypass_text = "Enable filter" if not self.band.enabled else "Bypass filter"
        bypass_act = menu.addAction(bypass_text)

        def _on_bypass_selected(_checked: bool = False) -> None:
            if self.scene() is None:
                return
            self.sigDragStarted.emit(self.band.band_id)
            self.sigBypassToggled.emit(self.band.band_id, not self.band.enabled)

        bypass_act.triggered.connect(_on_bypass_selected)

        reset_act = menu.addAction("Reset gain to 0 dB")

        def _on_reset_gain(_checked: bool = False) -> None:
            if self.scene() is None:
                return
            self.sigDragStarted.emit(self.band.band_id)
            self.band.gain = 0.0
            self._update_node_position()
            self.update()
            self.sigMoved.emit(self.band.band_id, self.band.freq, 0.0)
            if self.is_hovered:
                self.sigHoverChanged.emit(self, True)

        reset_act.triggered.connect(_on_reset_gain)

        menu.addSeparator()
        delete_act = menu.addAction("Delete band")

        def _on_delete_selected(_checked: bool = False) -> None:
            if self.scene() is None:
                return
            self.sigDeleteRequested.emit(self.band.band_id)

        delete_act.triggered.connect(_on_delete_selected)

        menu.exec(event.screenPos())
        event.accept()


class EQViewBox(pg.ViewBox):
    sigCanvasDoubleClicked = QtCore.pyqtSignal(float, float)

    def __init__(self, plot_widget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_widget = plot_widget
        self.y_min = MIN_GAIN
        self.y_max = MAX_GAIN
        self.setMouseMode(pg.ViewBox.PanMode)
        self.setLimits(xMin=X_MIN, xMax=X_MAX, yMin=self.y_min, yMax=self.y_max)

    def set_y_bounds(self, y_min: float, y_max: float) -> None:
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.setLimits(xMin=X_MIN, xMax=X_MAX, yMin=self.y_min, yMax=self.y_max)

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(ev.scenePos())
            x = max(X_MIN, min(X_MAX, pos.x()))
            y = max(self.y_min, min(self.y_max, pos.y()))
            freq = 10.0 ** x
            self.sigCanvasDoubleClicked.emit(freq, y)
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.ignore()
            return
        super().mouseClickEvent(ev)

    def contextMenuEvent(self, ev: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        if self.plot_widget is not None:
            pos = self.mapSceneToView(ev.scenePos())
            x = max(X_MIN, min(X_MAX, pos.x()))
            y = max(self.y_min, min(self.y_max, pos.y()))
            freq = 10.0 ** x
            sp = ev.screenPos()
            screen_pt = sp.toPoint() if hasattr(sp, "toPoint") else sp
            self.plot_widget.show_canvas_context_menu(screen_pt, freq, y)
            ev.accept()
        else:
            super().contextMenuEvent(ev)

    def raiseContextMenu(self, ev) -> None:
        if self.plot_widget is not None:
            pos = self.mapSceneToView(ev.scenePos())
            x = max(X_MIN, min(X_MAX, pos.x()))
            y = max(self.y_min, min(self.y_max, pos.y()))
            freq = 10.0 ** x
            sp = ev.screenPos()
            screen_pt = sp.toPoint() if hasattr(sp, "toPoint") else sp
            self.plot_widget.show_canvas_context_menu(screen_pt, freq, y)
        else:
            super().raiseContextMenu(ev)

    def mouseDragEvent(self, ev, axis=None) -> None:
        ev.ignore()

    def wheelEvent(self, ev, axis=None) -> None:
        ev.ignore()


class FrequencyAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(orientation="bottom", *args, **kwargs)
        self.setPen(pg.mkPen("#27272a", width=1))
        self.setTextPen(pg.mkPen("#a1a1aa"))
        self.setStyle(tickTextOffset=8, tickLength=0)
        self.setTickFont(_mono_font(8))
        self.setHeight(28)

        major_freqs = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        major_ticks = [
            (math.log10(f), f"{int(f // 1000)}k" if f >= 1000 else str(int(f)))
            for f in major_freqs
        ]

        minor_ticks = []
        for decade in [10, 100, 1000, 10000]:
            for mult in range(2, 10):
                val = mult * decade
                if MIN_FREQ <= val <= MAX_FREQ and val not in major_freqs:
                    minor_ticks.append((math.log10(val), ""))

        self.setTicks([major_ticks, minor_ticks])

    def setGrid(self, val) -> None:
        # Prevent AxisItem from painting grid on top of the view and points
        self.grid = False

    def boundingRect(self) -> QtCore.QRectF:
        rect = super().boundingRect()
        # Always maintain horizontal margin so 20 Hz and 20 kHz labels are never dropped by pyqtgraph when grid is toggled
        return rect.adjusted(-20, 0, 20, 0)

    def generateDrawSpecs(self, p: QtGui.QPainter):
        axis_spec, tick_specs, text_specs = super().generateDrawSpecs(p)
        if axis_spec is not None and text_specs:
            _, p1, p2 = axis_spec
            x_left = min(p1.x(), p2.x())
            x_right = max(p1.x(), p2.x())
            adjusted_specs = []
            for rect, flags, text in text_specs:
                r = QtCore.QRectF(rect)
                # Keep leftmost label ("20") clear of the bottom-left corner / Y-axis labels
                if r.left() < x_left + 2.0:
                    r.moveLeft(x_left + 2.0)
                elif r.right() > x_right:
                    r.moveRight(x_right)
                adjusted_specs.append((r, flags, text))
            text_specs = adjusted_specs
        return axis_spec, tick_specs, text_specs


class DecibelAxisItem(pg.AxisItem):
    sigDoubleClicked = QtCore.pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(orientation="left", *args, **kwargs)
        self.setPen(pg.mkPen("#27272a", width=1))
        self.setTextPen(pg.mkPen("#a1a1aa"))
        self.setStyle(tickTextOffset=8, tickLength=0)
        self.setTickFont(_mono_font(8))
        self.setWidth(64)
        self.setToolTip("Double-click to switch ±12 dB and ±20 dB scale")
        self._y_range_db = 20
        self.set_y_range_mode(20)

    def set_y_range_mode(self, db_range: int) -> None:
        self._y_range_db = 12 if int(db_range) <= 16 else 20
        step = 4 if self._y_range_db == 12 else 10
        ticks = [
            (float(val), f"{val:+.0f} dB" if val != 0 else "0 dB")
            for val in range(-self._y_range_db, self._y_range_db + 1, step)
        ]
        self.setTicks([ticks])
        self.update()

    def setGrid(self, val) -> None:
        # Prevent AxisItem from painting grid on top of the view and points
        self.grid = False

    def boundingRect(self) -> QtCore.QRectF:
        rect = super().boundingRect()
        # Always maintain margin so top/bottom/wide dB labels are never dropped by pyqtgraph
        return rect.adjusted(-16, -16, 0, 16)

    def generateDrawSpecs(self, p: QtGui.QPainter):
        axis_spec, tick_specs, text_specs = super().generateDrawSpecs(p)
        if axis_spec is not None and text_specs:
            _, p1, p2 = axis_spec
            y_top = min(p1.y(), p2.y())
            y_bottom = max(p1.y(), p2.y())
            adjusted_specs = []
            for rect, flags, text in text_specs:
                r = QtCore.QRectF(rect)
                # Keep bottommost dB label strictly above the bottom axis spine so it never collides with "20" Hz
                if r.bottom() > y_bottom - 2.0:
                    r.moveBottom(y_bottom - 2.0)
                elif r.top() < y_top:
                    r.moveTop(y_top)
                adjusted_specs.append((r, flags, text))
            text_specs = adjusted_specs
        return axis_spec, tick_specs, text_specs

    def mousePressEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self.sigDoubleClicked.emit()
            return
        super().mouseDoubleClickEvent(ev)

    def mouseClickEvent(self, event) -> None:
        if event.double() and event.button() == QtCore.Qt.MouseButton.LeftButton:
            event.accept()
            self.sigDoubleClicked.emit()
            return
        super().mouseClickEvent(event)


class ClickableOverlayLegend(pg.TextItem):
    """Interactive text banner on the graph canvas where clicking a trace label toggles it."""
    sigCurveClicked = QtCore.pyqtSignal(str)  # "target", "target2", "meas", "resultant"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regions: List[Tuple[float, float, str]] = []
        self.setAcceptHoverEvents(True)
        if hasattr(self, "textItem"):
            self.textItem.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)

    def _anchor_at(self, local_pos: QtCore.QPointF) -> str:
        if hasattr(self, "textItem") and self.textItem is not None:
            doc = self.textItem.document()
            if doc is not None and doc.documentLayout() is not None:
                doc_pos = self.textItem.mapFromParent(local_pos)
                anchor = doc.documentLayout().anchorAt(doc_pos)
                if anchor:
                    return anchor
        return ""

    def hoverMoveEvent(self, ev) -> None:
        if self._anchor_at(ev.pos()):
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()
        super().hoverMoveEvent(ev)

    def hoverLeaveEvent(self, ev) -> None:
        self.unsetCursor()
        super().hoverLeaveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            anchor = self._anchor_at(ev.pos())
            if anchor:
                ev.accept()
                self.sigCurveClicked.emit(anchor)
                return
        super().mousePressEvent(ev)


class EQPlotWidget(pg.PlotWidget):
    """
    Main EQ canvas hosting:
    - Logarithmic frequency grid with switchable ±12 dB / ±20 dB Y scale
    - Live audio spectrum analyzer underlay + peak-hold ghost trace
    - Individual hovered/dragged band curve preview + hairline crosshairs
    - Composite frequency response line (crisp white, 60+ FPS)
    - Interactive draggable band nodes with precision gestures
    - Quiet monospace cursor & band HUD readouts
    - Silky smooth 700 ms cubic sweeping transition for the graph curve
    """
    sigBandModified = QtCore.pyqtSignal(int, float, float, float, str, bool)
    sigBandAdded = QtCore.pyqtSignal(float, float)
    sigBandDeleted = QtCore.pyqtSignal(int)
    sigSpectrumToggled = QtCore.pyqtSignal(bool)
    sigLegendCurveClicked = QtCore.pyqtSignal(str)
    sigBandDragStarted = QtCore.pyqtSignal(int)
    sigWheelScrollStarted = QtCore.pyqtSignal()
    sigYRangeChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        self.custom_vb = EQViewBox(plot_widget=self)
        self.bottom_axis = FrequencyAxisItem()
        self.left_axis = DecibelAxisItem()

        super().__init__(
            parent=parent,
            viewBox=self.custom_vb,
            axisItems={"bottom": self.bottom_axis, "left": self.left_axis},
        )

        self.freq_points = np.geomspace(MIN_FREQ, MAX_FREQ, 1000)
        self.x_log_points = np.log10(self.freq_points)
        self._x_norm = np.linspace(0.0, 1.0, len(self.freq_points))
        self.sample_rate = 48000.0

        self._y_range_db: int = 20
        self.y_min: float = MIN_GAIN
        self.y_max: float = MAX_GAIN

        self.preamp_db = 0.0
        self.bands: List[EQBand] = []
        self.node_items: Dict[int, EQNodeItem] = {}
        self.is_bypassed = False
        self._eq_active_factor = 1.0
        self._cached_bypass_active_response: Optional[np.ndarray] = None
        self._last_composite_response = np.zeros(len(self.freq_points), dtype=np.float64)
        self._preview_band_id: Optional[int] = None
        self._peak_hold_visible: bool = False

        # Universal EQ Bypass / Toggle Animation (350 ms, InOutCubic)
        self._bypass_animation = QtCore.QVariantAnimation(self)
        self._bypass_animation.setDuration(350)
        self._bypass_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._bypass_animation.valueChanged.connect(self._on_bypass_anim_progress)
        self._bypass_animation.finished.connect(self._on_bypass_anim_finished)

        # Transition sweep animation (700 ms, InOutCubic, silky smooth wave front & moving nodes)
        self._sweep_animation = QtCore.QVariantAnimation(self)
        self._sweep_animation.setDuration(700)
        self._sweep_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._sweep_animation.valueChanged.connect(self._on_sweep_progress)
        self._sweep_animation.finished.connect(self._on_sweep_finished)
        self._sweep_old_response = np.zeros(1000)
        self._sweep_target_response = np.zeros(1000)
        self._is_sweeping = False

        # Overlay sweep animation state
        self._sweep_meas_animating = False
        self._sweep_old_meas = np.zeros(1000)
        self._sweep_target_meas = np.zeros(1000)
        self._sweep_old_meas_opacity = 0.0
        self._sweep_target_meas_opacity = 0.0
        self._pending_final_meas: Optional[Tuple[Sequence[float], Sequence[float], str]] = None

        self._sweep_target_animating = False
        self._sweep_old_target = np.zeros(1000)
        self._sweep_target_target = np.zeros(1000)
        self._sweep_old_target_opacity = 0.0
        self._sweep_target_target_opacity = 0.0
        self._pending_final_target: Optional[Tuple[Sequence[float], Sequence[float], str]] = None

        self._sweep_resultant_animating = False
        self._sweep_old_resultant = np.zeros(1000)
        self._sweep_target_resultant = np.zeros(1000)
        self._sweep_old_resultant_opacity = 0.0
        self._sweep_target_resultant_opacity = 0.0
        self._pending_final_resultant: Optional[Tuple[Sequence[float], Sequence[float], str]] = None

        # Standalone overlay animation (400 ms, InOutCubic)
        self._overlay_anim = QtCore.QVariantAnimation(self)
        self._overlay_anim.setDuration(400)
        self._overlay_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)
        self._overlay_anim.valueChanged.connect(self._on_overlay_anim_progress)
        self._overlay_anim.finished.connect(self._on_overlay_anim_finished)
        self._anim_overlay_types: Dict[str, Tuple[np.ndarray, np.ndarray, float, float, Any]] = {}

        self._is_meas_checked = True
        self._is_target_checked = True
        self._is_target2_checked = True
        self._is_resultant_checked = True

        # Node animation tracking structures
        self._node_anim_common: Dict[int, Tuple[EQNodeItem, float, float, float, float, EQBand]] = {}
        self._node_anim_removed: List[EQNodeItem] = []
        self._node_anim_added: List[EQNodeItem] = []

        self._init_ui()

    def _init_ui(self) -> None:
        self.setBackground("#000000")
        self.showGrid(x=False, y=False)
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().layout.setContentsMargins(8, 12, 16, 8)

        self.setXRange(X_MIN, X_MAX, padding=0.0)
        self.setYRange(self.y_min, self.y_max, padding=0.0)
        self.disableAutoRange()
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMouseTracking(True)

        # Dedicated background grid inside ViewBox at zValue = 0 (strictly behind curves & nodes)
        self.grid_item = EQGridItem(X_MIN, X_MAX, self.y_min, self.y_max, y_step=10)
        self.grid_item.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.addItem(self.grid_item)

        # Spectrum Analyzer Curve (translucent graphite/white underlay)
        self.spectrum_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(255, 255, 255, 56), width=1.0),
            brush=pg.mkBrush(QtGui.QColor(255, 255, 255, 18)),
            fillLevel=self.y_min,
            antialias=True,
        )
        self.spectrum_curve.setZValue(1)
        self.addItem(self.spectrum_curve)

        # Live Spectrum Peak-Hold Ghost Trace (thin crisp line above spectrum underlay, below EQ)
        self.peak_hold_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(161, 161, 170, 115), width=1.1),
            antialias=True,
        )
        self.peak_hold_curve.setZValue(1.5)
        self.peak_hold_curve.setVisible(False)
        self.addItem(self.peak_hold_curve)

        # 0 dB Reference line
        zero_pen = pg.mkPen(QtGui.QColor(63, 63, 70, 180), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.zero_line = pg.InfiniteLine(pos=0.0, angle=0.0, pen=zero_pen)
        self.zero_line.setZValue(2)
        self.zero_line.setCacheMode(QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.addItem(self.zero_line)

        # Hairline Crosshairs intersecting hovered/dragged node at (Fc, Gain)
        crosshair_pen = pg.mkPen(QtGui.QColor("#3f3f46"), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.crosshair_v_line = pg.InfiniteLine(pos=math.log10(1000.0), angle=90.0, pen=crosshair_pen)
        self.crosshair_v_line.setZValue(3)
        self.crosshair_v_line.setVisible(False)
        self.addItem(self.crosshair_v_line)

        self.crosshair_h_line = pg.InfiniteLine(pos=0.0, angle=0.0, pen=crosshair_pen)
        self.crosshair_h_line.setZValue(3)
        self.crosshair_h_line.setVisible(False)
        self.addItem(self.crosshair_h_line)

        # Raw IEM Measurement Overlay Curve (subtle dashed line)
        self.meas_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(161, 161, 170, 160), width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.meas_curve.setZValue(4)
        self.meas_curve.setVisible(False)
        self.addItem(self.meas_curve)

        # Target Compensation Reference Curve (distinct dashed amber line)
        self.target_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(250, 204, 21, 190), width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.target_curve.setZValue(5)
        self.target_curve.setVisible(False)
        self.addItem(self.target_curve)

        # Secondary Target Reference Curve (distinct dashed violet line)
        self.target2_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(192, 132, 252, 200), width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.target2_curve.setZValue(5)
        self.target2_curve.setVisible(False)
        self.addItem(self.target2_curve)

        # Resultant Predicted Response Curve (Model + EQ in sky-cyan)
        self.resultant_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(56, 189, 248, 220), width=1.8),
            antialias=True,
        )
        self.resultant_curve.setZValue(6)
        self.resultant_curve.setVisible(False)
        self.addItem(self.resultant_curve)

        # Individual Band Response Preview (subtle neutral fill + dashed outline on hover/drag)
        self.band_preview_fill_curve = pg.PlotCurveItem(
            pen=None,
            brush=pg.mkBrush(QtGui.QColor(161, 161, 170, 26)),
            fillLevel=0.0,
            antialias=True,
        )
        self.band_preview_fill_curve.setZValue(7)
        self.band_preview_fill_curve.setVisible(False)
        self.addItem(self.band_preview_fill_curve)

        self.band_preview_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(161, 161, 170, 195), width=1.3, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.band_preview_curve.setZValue(8)
        self.band_preview_curve.setVisible(False)
        self.addItem(self.band_preview_curve)

        # Composite EQ Curve Fill (translucent white wash)
        self.eq_fill_curve = pg.PlotCurveItem(
            pen=None,
            brush=pg.mkBrush(QtGui.QColor(255, 255, 255, 14)),
            fillLevel=0.0,
            antialias=True,
        )
        self.eq_fill_curve.setZValue(9)
        self.addItem(self.eq_fill_curve)

        # Composite EQ Response Curve (pure white, 2.4px width)
        self.eq_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor("#ffffff"), width=2.4),
            antialias=True,
        )
        self.eq_curve.setZValue(10)
        self.addItem(self.eq_curve)

        # HUD Info Text Item (top-left banner, monospace numeric readouts)
        self.hud_text_item = pg.TextItem(text="", color="#ededed", anchor=(0, 0))
        self.hud_text_item.setZValue(30)
        self.hud_text_item.setVisible(False)
        self.addItem(self.hud_text_item)

        # Bypassed Watermark (top-right banner, sentence case)
        self.bypass_text_item = pg.TextItem(text="", color="#71717a", anchor=(1, 0))
        self.bypass_text_item.setHtml(
            "<span style=\"font-family: 'Segoe UI', sans-serif; font-size: 11px; "
            "font-weight: 500; color: #71717a;\">EQ bypassed</span>"
        )
        self.bypass_text_item.setZValue(30)
        self.bypass_text_item.setVisible(False)
        self.addItem(self.bypass_text_item)

        # AutoEQ Overlay Legend (bottom-left banner, interactive clickable; below nodes at z=25)
        self.overlay_legend_item = ClickableOverlayLegend(text="", color="#a1a1aa", anchor=(0, 1))
        self.overlay_legend_item.setZValue(20)
        self.overlay_legend_item.setVisible(False)
        self.overlay_legend_item.sigCurveClicked.connect(self.sigLegendCurveClicked.emit)
        self.addItem(self.overlay_legend_item)

        # Live Cursor Readout (top-left clear area, mutually exclusive with hud_text_item; never collides with bypass/legend)
        self.cursor_text_item = pg.TextItem(text="", color="#a1a1aa", anchor=(0, 0))
        self.cursor_text_item.setZValue(30)
        self.cursor_text_item.setVisible(False)
        self.addItem(self.cursor_text_item)

        self._reposition_corner_items()

        self._meas_freqs: Optional[np.ndarray] = None
        self._meas_spls: Optional[np.ndarray] = None
        self._meas_label: str = ""
        self._target_freqs: Optional[np.ndarray] = None
        self._target_spls: Optional[np.ndarray] = None
        self._target_label: str = ""
        self._target2_freqs: Optional[np.ndarray] = None
        self._target2_spls: Optional[np.ndarray] = None
        self._target2_label: str = ""
        self._resultant_freqs: Optional[np.ndarray] = None
        self._resultant_spls: Optional[np.ndarray] = None
        self._resultant_label: str = ""

        self.custom_vb.sigCanvasDoubleClicked.connect(self.sigBandAdded.emit)
        self.custom_vb.sigRangeChanged.connect(self._on_view_range_changed)
        self.custom_vb.sigTransformChanged.connect(self._on_view_range_changed)
        self.left_axis.sigDoubleClicked.connect(self.toggle_y_range_mode)
        self.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

    def _reposition_corner_items(self) -> None:
        y_margin = (self.y_max - self.y_min) * 0.03
        self.hud_text_item.setPos(X_MIN + 0.05, self.y_max - y_margin)
        self.cursor_text_item.setPos(X_MIN + 0.05, self.y_max - y_margin)
        self.bypass_text_item.setPos(X_MAX - 0.05, self.y_max - y_margin)
        self.overlay_legend_item.setPos(X_MIN + 0.05, self.y_min + y_margin)

    def set_y_range_mode(self, db_range: int) -> None:
        """Switches vertical dB scale between ±12 dB (4 dB grid) and ±20 dB (10 dB grid)."""
        mode = 12 if int(db_range) <= 16 else 20
        changed = (mode != self._y_range_db)
        self._y_range_db = mode
        self.y_min = -float(mode)
        self.y_max = float(mode)
        y_step = 4 if mode == 12 else 10

        self.custom_vb.set_y_bounds(self.y_min, self.y_max)
        self.setYRange(self.y_min, self.y_max, padding=0.0)
        self.left_axis.set_y_range_mode(mode)
        self.grid_item.set_bounds(X_MIN, X_MAX, self.y_min, self.y_max, y_step=y_step)
        self.spectrum_curve.setFillLevel(self.y_min)

        for node in self.node_items.values():
            node.set_y_bounds(self.y_min, self.y_max)
        for node in self._node_anim_removed:
            node.set_y_bounds(self.y_min, self.y_max)

        self._reposition_corner_items()

        if self._preview_band_id is not None and self._preview_band_id in self.node_items:
            self._show_band_preview(self.node_items[self._preview_band_id].band)

        self._on_view_range_changed()
        if changed:
            self.sigYRangeChanged.emit(self._y_range_db)

    def get_y_range_mode(self) -> int:
        """Returns the active vertical dB scale (12 or 20)."""
        return int(self._y_range_db)

    def toggle_y_range_mode(self) -> None:
        """Toggles vertical dB scale between ±12 dB and ±20 dB."""
        self.set_y_range_mode(12 if self._y_range_db == 20 else 20)

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self._on_view_range_changed()

    def resizeEvent(self, ev: Optional[QtGui.QResizeEvent]) -> None:
        super().resizeEvent(ev)
        self._on_view_range_changed()

    def leaveEvent(self, ev: QtCore.QEvent) -> None:
        self.cursor_text_item.setVisible(False)
        super().leaveEvent(ev)

    def _on_view_range_changed(self) -> None:
        if not hasattr(self, "node_items"):
            return
        for node in self.node_items.values():
            node.updateTransform()
        for node in self._node_anim_removed:
            node.updateTransform()

    # -------------------------------------------------------------------------
    # Universal EQ Bypass / Toggle Animation
    # -------------------------------------------------------------------------
    def set_eq_bypassed(self, bypassed: bool) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self.is_bypassed = bypassed
        self._bypass_animation.stop()
        self._cached_bypass_active_response = compute_composite_response(
            self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
        )
        start_factor = self._eq_active_factor
        end_factor = 0.0 if bypassed else 1.0
        self._bypass_animation.setStartValue(start_factor)
        self._bypass_animation.setEndValue(end_factor)
        self._bypass_animation.start()

    def _on_bypass_anim_progress(self, val: float) -> None:
        self._eq_active_factor = float(val)
        factor = self._eq_active_factor

        # Interpolate composite EQ curve towards 0 dB flat line
        active_response = getattr(self, "_cached_bypass_active_response", None)
        if active_response is None:
            active_response = compute_composite_response(
                self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
            )
            self._cached_bypass_active_response = active_response
        current_response = factor * active_response
        self._last_composite_response = np.asarray(current_response, dtype=np.float64)
        self.eq_curve.setData(self.x_log_points, current_response)

        # Interpolate curve pen (white solid -> dim grey dashed)
        r = int(255 * factor + 82 * (1.0 - factor))
        g = int(255 * factor + 82 * (1.0 - factor))
        b = int(255 * factor + 91 * (1.0 - factor))
        width = 2.4 * factor + 1.8 * (1.0 - factor)

        if factor < 0.08:
            pen = pg.mkPen(QtGui.QColor(82, 82, 91), width=1.8, style=QtCore.Qt.PenStyle.DashLine)
            self.eq_fill_curve.setBrush(pg.mkBrush(None))
            self.bypass_text_item.setVisible(True)
        else:
            pen = pg.mkPen(QtGui.QColor(r, g, b), width=width)
            fill_alpha = int(14 * factor)
            self.eq_fill_curve.setBrush(pg.mkBrush(QtGui.QColor(255, 255, 255, fill_alpha)))
            self.eq_fill_curve.setData(self.x_log_points, current_response)
            self.bypass_text_item.setVisible(False)

        self.eq_curve.setPen(pen)

        # Update node appearance
        for node in self.node_items.values():
            node.is_bypassed_globally = (factor < 0.5)
            node.update()

    def _on_bypass_anim_finished(self) -> None:
        if self.is_bypassed:
            self._eq_active_factor = 0.0
            self.eq_curve.setPen(pg.mkPen(QtGui.QColor("#52525b"), width=1.8, style=QtCore.Qt.PenStyle.DashLine))
            self.eq_fill_curve.setBrush(pg.mkBrush(None))
            self.bypass_text_item.setVisible(True)
            flat_response = np.zeros(len(self.freq_points))
            self._last_composite_response = flat_response
            self.eq_curve.setData(self.x_log_points, flat_response)
        else:
            self._eq_active_factor = 1.0
            self.eq_curve.setPen(pg.mkPen(QtGui.QColor("#ffffff"), width=2.4))
            self.eq_fill_curve.setBrush(pg.mkBrush(QtGui.QColor(255, 255, 255, 14)))
            self.bypass_text_item.setVisible(False)
            self.recompute_composite_curve()

        for node in self.node_items.values():
            node.is_bypassed_globally = self.is_bypassed
            node.update()

    # -------------------------------------------------------------------------
    # Animation & Node Lifecycle Helpers
    # -------------------------------------------------------------------------
    def _create_node(self, band: EQBand) -> EQNodeItem:
        node = EQNodeItem(band, self.custom_vb, y_min=self.y_min, y_max=self.y_max)
        node.is_bypassed_globally = self.is_bypassed
        node.sigMoved.connect(self._on_node_moved)
        node.sigQChanged.connect(self._on_node_q_changed)
        node.sigTypeChanged.connect(self._on_node_type_changed)
        node.sigBypassToggled.connect(self._on_node_bypass_toggled)
        node.sigDeleteRequested.connect(self.sigBandDeleted.emit)
        node.sigHoverChanged.connect(self._on_node_hover)
        node.sigDragStarted.connect(self._on_node_drag_started)
        node.sigWheelScrollStarted.connect(self.sigWheelScrollStarted.emit)
        return node

    def _disable_removed_node(self, node: EQNodeItem) -> None:
        was_active = node.is_hovered or node.is_dragging or (self._preview_band_id == node.band.band_id)
        if was_active:
            node.is_hovered = False
            node.is_dragging = False
            if not any(n.is_dragging or n.is_hovered for n in self.node_items.values()):
                self.hud_text_item.setVisible(False)
                self._hide_band_preview()
        node.setAcceptHoverEvents(False)
        node.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        node.setEnabled(False)
        for sig in (
            node.sigMoved,
            node.sigQChanged,
            node.sigTypeChanged,
            node.sigBypassToggled,
            node.sigDeleteRequested,
            node.sigHoverChanged,
            node.sigDragStarted,
            node.sigWheelScrollStarted,
        ):
            try:
                sig.disconnect()
            except TypeError:
                pass

    def _capture_overlay_state(self, cname: str) -> Tuple[np.ndarray, float]:
        curve_item = getattr(self, f"{cname}_curve", None)
        curr_freqs = getattr(self, f"_{cname}_freqs", None)
        curr_spls = getattr(self, f"_{cname}_spls", None)
        overlay_active = cname in self._anim_overlay_types
        if curve_item is not None and (self._is_sweeping or overlay_active):
            _, curr_y = curve_item.getData()
            if curr_y is not None and len(curr_y) == len(self.freq_points):
                op = float(curve_item.opacity()) if curve_item.isVisible() else 0.0
                return np.asarray(curr_y, dtype=np.float64), op
            if overlay_active:
                old_s, _target_s, old_op, _target_op, _final = self._anim_overlay_types[cname]
                return np.copy(old_s), float(old_op)
        if curr_freqs is not None and len(curr_freqs) > 0 and curr_spls is not None and len(curr_spls) == len(curr_freqs):
            op = (float(curve_item.opacity()) if curve_item is not None and curve_item.isVisible() else 0.0)
            return np.interp(self.freq_points, curr_freqs, curr_spls), op
        return np.zeros(len(self.freq_points), dtype=np.float64), 0.0

    def _stop_overlay_anim(self, curve_name: Optional[str] = None) -> None:
        if curve_name is None:
            self._overlay_anim.stop()
            for cname in list(self._anim_overlay_types.keys()):
                curve_item = getattr(self, f"{cname}_curve", None)
                if curve_item is not None:
                    curve_item.setOpacity(1.0)
            self._anim_overlay_types.clear()
        elif curve_name in self._anim_overlay_types:
            self._anim_overlay_types.pop(curve_name, None)
            curve_item = getattr(self, f"{curve_name}_curve", None)
            if curve_item is not None:
                curve_item.setOpacity(1.0)
            if not self._anim_overlay_types:
                self._overlay_anim.stop()

    def _cancel_sweep_animation(self, finalize_overlays: bool = True) -> None:
        if not self._is_sweeping:
            return
        self._sweep_animation.stop()
        self._is_sweeping = False

        for node in self._node_anim_removed:
            try:
                self.removeItem(node)
            except Exception:
                pass
        self._node_anim_removed.clear()

        for _bid, (node, _x0, _y0, _x1, _y1, target_band) in self._node_anim_common.items():
            node.band = _sanitize_band(target_band)
            node._update_node_position()
            node.setOpacity(1.0)
            node.update()
        self._node_anim_common.clear()

        for node in self._node_anim_added:
            node.setOpacity(1.0)
            node.update()
        self._node_anim_added.clear()

        for node in self.node_items.values():
            node.setOpacity(1.0)

        if finalize_overlays:
            self._finalize_sweep_overlays()
        else:
            for flag_attr, curve_attr, pend_attr in (
                ("_sweep_meas_animating", "meas_curve", "_pending_final_meas"),
                ("_sweep_target_animating", "target_curve", "_pending_final_target"),
                ("_sweep_resultant_animating", "resultant_curve", "_pending_final_resultant"),
            ):
                if getattr(self, flag_attr):
                    setattr(self, flag_attr, False)
                    setattr(self, pend_attr, None)
                    getattr(self, curve_attr).setOpacity(1.0)

    def _finalize_sweep_overlays(self) -> None:
        if self._sweep_meas_animating:
            self._sweep_meas_animating = False
            pending = self._pending_final_meas
            self._pending_final_meas = None
            self.meas_curve.setOpacity(1.0)
            if pending:
                mf, ms, mlbl = pending
                self.set_measurement_overlay(mf, ms, label=mlbl)
            else:
                self.set_measurement_overlay(None, None)

        if self._sweep_target_animating:
            self._sweep_target_animating = False
            pending = self._pending_final_target
            self._pending_final_target = None
            self.target_curve.setOpacity(1.0)
            if pending:
                tf, ts, tlbl = pending
                self.set_target_overlay(tf, ts, label=tlbl)
            else:
                self.set_target_overlay(None, None)

        if self._sweep_resultant_animating:
            self._sweep_resultant_animating = False
            pending = self._pending_final_resultant
            self._pending_final_resultant = None
            self.resultant_curve.setOpacity(1.0)
            if pending:
                rf, rs, rlbl = pending
                self.set_resultant_overlay(rf, rs, label=rlbl)
            else:
                self.set_resultant_overlay(None, None)

    def _on_node_drag_started(self, band_id: int) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self.sigBandDragStarted.emit(band_id)

    # -------------------------------------------------------------------------
    # Sweeping Transition Animation with Moving & Fading Points
    # -------------------------------------------------------------------------
    def animate_to_preset(
        self,
        new_bands: List[EQBand],
        new_preamp: float,
        target_data: Optional[Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str]] = None,
        meas_data: Optional[Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str]] = None,
        resultant_data: Optional[Tuple[Optional[Sequence[float]], Optional[Sequence[float]], str]] = None,
    ) -> None:
        """
        Executes a silky smooth 700 ms left-to-right graph sweep animation.
        - Nodes present in both presets smoothly glide to their new positions.
        - Nodes absent in the new preset smoothly fade out and are removed.
        - Newly added nodes smoothly fade in at their target positions.
        - Simultaneously sweeps and transitions explicitly supplied target, measurement, and resultant traces;
          omitted (None) overlay args preserve existing overlays untouched.
        """
        if self._bypass_animation.state() == QtCore.QAbstractAnimation.State.Running:
            self._bypass_animation.stop()
            self._on_bypass_anim_finished()

        # Capture in-flight standalone overlay animations and current visual states before stopping them
        in_flight_overlay = dict(self._anim_overlay_types)
        self._sweep_old_meas, self._sweep_old_meas_opacity = self._capture_overlay_state("meas")
        self._sweep_old_target, self._sweep_old_target_opacity = self._capture_overlay_state("target")
        self._sweep_old_resultant, self._sweep_old_resultant_opacity = self._capture_overlay_state("resultant")
        self._stop_overlay_anim()

        # target2 is not swept, so finalize any in-flight standalone target2 animation immediately
        if "target2" in in_flight_overlay:
            final_t2 = in_flight_overlay["target2"][4]
            if final_t2:
                f2, s2, lbl2 = final_t2
                self.set_target2_overlay(f2, s2, label=lbl2)
            else:
                self.set_target2_overlay(None, None)

        was_sweeping = self._is_sweeping
        if was_sweeping:
            self._sweep_animation.stop()
            _, curr_eq = self.eq_curve.getData()
            if curr_eq is not None and len(curr_eq) == len(self.freq_points):
                self._sweep_old_response = np.asarray(curr_eq, dtype=np.float64)
            else:
                self._sweep_old_response = self._eq_active_factor * compute_composite_response(
                    self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
                )

            for node in self._node_anim_removed:
                try:
                    self.removeItem(node)
                except Exception:
                    pass
            self._node_anim_removed.clear()
        else:
            self._sweep_old_response = self._eq_active_factor * compute_composite_response(
                self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
            )

        self.preamp_db = float(new_preamp) if isinstance(new_preamp, (int, float)) and math.isfinite(new_preamp) else 0.0
        self._cached_bypass_active_response = None

        old_map: Dict[int, EQBand] = {b.band_id: _sanitize_band(b) for b in self.bands}
        new_map: Dict[int, EQBand] = {b.band_id: _sanitize_band(b) for b in new_bands}

        common_ids = set(old_map.keys()) & set(new_map.keys())
        removed_ids = set(old_map.keys()) - set(new_map.keys())
        added_ids = set(new_map.keys()) - set(old_map.keys())

        self._node_anim_common.clear()
        self._node_anim_removed.clear()
        self._node_anim_added.clear()

        # 1. Common points: glide smoothly from current visual position to new target position
        for bid in common_ids:
            if bid in self.node_items:
                node = self.node_items[bid]
                curr_pos = node.pos()
                x0, y0 = float(curr_pos.x()), float(curr_pos.y())
                x1, y1 = _safe_band_coords(new_map[bid], self.y_min, self.y_max)
                node._fade_start_opacity = float(node.opacity())
                node.is_bypassed_globally = self.is_bypassed
                self._node_anim_common[bid] = (node, x0, y0, x1, y1, new_map[bid])

        # 2. Removed points: disable interaction and smoothly fade out from current opacity
        for bid in removed_ids:
            if bid in self.node_items:
                node = self.node_items.pop(bid)
                node._fade_start_opacity = float(node.opacity())
                self._disable_removed_node(node)
                self._node_anim_removed.append(node)

        if self._preview_band_id is not None and self._preview_band_id not in self.node_items:
            self._hide_band_preview()

        # 3. Added points: materialize at new pos and smoothly fade in
        for bid in added_ids:
            target_b = new_map[bid]
            node = self._create_node(target_b)
            node.setOpacity(0.0)
            self.addItem(node)
            node.updateTransform()
            self.node_items[bid] = node
            self._node_anim_added.append(node)

        self.bands = list(new_map.values())
        self._sweep_target_response = self._eq_active_factor * compute_composite_response(
            self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
        )

        # Overlay sweep setup:
        # - Explicit tuple (data is not None): sweep to new data or clear if empty/None
        # - Omitted (None) with in-flight standalone overlay anim: hand off seamlessly to this sweep
        # - Omitted (None) with in-flight prior sweep anim: preserve and continue seamlessly
        # - Otherwise: leave existing static overlay untouched
        for cname, data in (
            ("meas", meas_data),
            ("target", target_data),
            ("resultant", resultant_data),
        ):
            flag_attr = f"_sweep_{cname}_animating"
            old_attr = f"_sweep_old_{cname}"
            target_attr = f"_sweep_target_{cname}"
            target_op_attr = f"_sweep_target_{cname}_opacity"
            pend_attr = f"_pending_final_{cname}"
            is_chk = getattr(self, f"_is_{cname}_checked", True)

            if data is not None:
                f_in, s_in, lbl = data
                cleaned = _sanitize_curve_data(f_in, s_in)
                if cleaned is not None:
                    cf, cs = cleaned
                    setattr(self, target_attr, np.interp(self.freq_points, cf, cs))
                    setattr(self, target_op_attr, 1.0 if is_chk else 0.0)
                    setattr(self, pend_attr, (cf, cs, lbl))
                    setattr(self, f"_{cname}_label", lbl)
                else:
                    setattr(self, target_attr, np.copy(getattr(self, old_attr)))
                    setattr(self, target_op_attr, 0.0)
                    setattr(self, pend_attr, None)
                    setattr(self, f"_{cname}_label", "")
                setattr(self, flag_attr, True)
            elif cname in in_flight_overlay:
                _old_s, target_s, _old_op, _target_op, final_data = in_flight_overlay[cname]
                if final_data is not None:
                    setattr(self, target_attr, target_s)
                    setattr(self, target_op_attr, 1.0 if is_chk else 0.0)
                    setattr(self, pend_attr, final_data)
                    setattr(self, f"_{cname}_label", final_data[2])
                else:
                    setattr(self, target_attr, np.copy(getattr(self, old_attr)))
                    setattr(self, target_op_attr, 0.0)
                    setattr(self, pend_attr, None)
                    setattr(self, f"_{cname}_label", "")
                setattr(self, flag_attr, True)
            elif was_sweeping and getattr(self, flag_attr):
                pending = getattr(self, pend_attr)
                if pending is not None:
                    setattr(self, target_op_attr, 1.0 if is_chk else 0.0)
                else:
                    setattr(self, target_attr, np.copy(getattr(self, old_attr)))
                    setattr(self, target_op_attr, 0.0)
                setattr(self, flag_attr, True)
            else:
                setattr(self, flag_attr, False)

        self._update_overlay_legend()

        # Set sweeping state
        self._is_sweeping = True

        # Synchronously invoke _on_sweep_progress(0.0) so that frame 0 is rendered immediately
        # with zero single-frame delay or stale data before Qt begins timer events
        self._on_sweep_progress(0.0)

        # Start 700 ms sweep animation
        self._sweep_animation.stop()
        self._sweep_animation.setStartValue(0.0)
        self._sweep_animation.setEndValue(1.0)
        self._sweep_animation.start()

    def _on_sweep_progress(self, val: float) -> None:
        p = float(val)

        # 1. Update curve sweep (wave front with cubic hermite easing)
        sigma = 0.20
        center = (1.0 + 2.0 * sigma) * p - sigma
        u = np.clip((center - self._x_norm) / (2.0 * sigma) + 0.5, 0.0, 1.0)
        alpha = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        blended = (1.0 - alpha) * self._sweep_old_response + alpha * self._sweep_target_response

        self._last_composite_response = np.asarray(blended, dtype=np.float64)
        self.eq_curve.setData(self.x_log_points, blended)
        if not self.is_bypassed:
            self.eq_fill_curve.setData(self.x_log_points, blended)

        # Overlays sweep (smooth wave front with cubic hermite easing)
        if self._sweep_meas_animating:
            blended_m = (1.0 - alpha) * self._sweep_old_meas + alpha * self._sweep_target_meas
            self.meas_curve.setData(self.x_log_points, blended_m)
            op_m = (1.0 - p) * self._sweep_old_meas_opacity + p * self._sweep_target_meas_opacity
            self.meas_curve.setOpacity(op_m)
            self.meas_curve.setVisible(op_m > 0.01)

        if self._sweep_target_animating:
            blended_t = (1.0 - alpha) * self._sweep_old_target + alpha * self._sweep_target_target
            self.target_curve.setData(self.x_log_points, blended_t)
            op_t = (1.0 - p) * self._sweep_old_target_opacity + p * self._sweep_target_target_opacity
            self.target_curve.setOpacity(op_t)
            self.target_curve.setVisible(op_t > 0.01)

        if self._sweep_resultant_animating:
            blended_r = (1.0 - alpha) * self._sweep_old_resultant + alpha * self._sweep_target_resultant
            self.resultant_curve.setData(self.x_log_points, blended_r)
            op_r = (1.0 - p) * self._sweep_old_resultant_opacity + p * self._sweep_target_resultant_opacity
            self.resultant_curve.setOpacity(op_r)
            self.resultant_curve.setVisible(op_r > 0.01)

        # 2. Smoothly move common nodes to target positions
        for _bid, (node, x0, y0, x1, y1, _target_band) in self._node_anim_common.items():
            curr_x = (1.0 - p) * x0 + p * x1
            curr_y = (1.0 - p) * y0 + p * y1
            node.setPos(curr_x, curr_y)
            node.band.freq = 10.0 ** curr_x
            node.band.gain = curr_y
            if node._fade_start_opacity < 1.0:
                node.setOpacity(min(1.0, node._fade_start_opacity + p * (1.0 - node._fade_start_opacity)))
            node.update()

        # 3. Smoothly fade out removed nodes
        fade_out_factor = max(0.0, 1.0 - p)
        for node in self._node_anim_removed:
            node.setOpacity(node._fade_start_opacity * fade_out_factor)

        # 4. Smoothly fade in added nodes
        fade_in = min(1.0, p)
        for node in self._node_anim_added:
            node.setOpacity(fade_in)

    def _on_sweep_finished(self) -> None:
        self._is_sweeping = False

        # Cleanly remove faded-out nodes
        for node in self._node_anim_removed:
            try:
                self.removeItem(node)
            except Exception:
                pass
        self._node_anim_removed.clear()

        # Reset opacity and finalize common and added nodes
        for _bid, (node, _x0, _y0, _x1, _y1, target_band) in self._node_anim_common.items():
            node.band = _sanitize_band(target_band)
            node._update_node_position()
            node.setOpacity(1.0)
            node.update()
        self._node_anim_common.clear()

        for node in self._node_anim_added:
            node.setOpacity(1.0)
            node.update()
        self._node_anim_added.clear()

        for node in self.node_items.values():
            node.setOpacity(1.0)

        self._finalize_sweep_overlays()
        self.recompute_composite_curve()

    def animate_overlay_curve(
        self,
        curve_name: str,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
        duration: int = 400,
    ) -> None:
        """Smoothly transitions a single overlay curve (target, target2, meas, or resultant) with InOutCubic easing."""
        if self._is_sweeping:
            if curve_name == "target":
                self.set_target_overlay(freqs, spls, label)
            elif curve_name == "target2":
                self.set_target2_overlay(freqs, spls, label)
            elif curve_name == "meas":
                self.set_measurement_overlay(freqs, spls, label)
            elif curve_name == "resultant":
                self.set_resultant_overlay(freqs, spls, label)
            return

        curve_item = getattr(self, f"{curve_name}_curve", None)
        if not curve_item:
            return

        old_sampled, old_op = self._capture_overlay_state(curve_name)

        is_checked = getattr(self, f"_is_{curve_name}_checked", True)
        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is not None:
            f_clean, s_clean = cleaned
            target_sampled = np.interp(self.freq_points, f_clean, s_clean)
            target_op = 1.0 if is_checked else 0.0
            final_data = (f_clean, s_clean, label)
            setattr(self, f"_{curve_name}_freqs", f_clean)
            setattr(self, f"_{curve_name}_spls", s_clean)
            setattr(self, f"_{curve_name}_label", label)
        else:
            target_sampled = np.copy(old_sampled)
            target_op = 0.0
            final_data = None
            setattr(self, f"_{curve_name}_freqs", None)
            setattr(self, f"_{curve_name}_spls", None)
            setattr(self, f"_{curve_name}_label", "")

        self._anim_overlay_types[curve_name] = (old_sampled, target_sampled, old_op, target_op, final_data)
        curve_item.setVisible((target_op > 0.0) or (old_op > 0.0))
        self._update_overlay_legend()

        self._overlay_anim.stop()
        self._overlay_anim.setDuration(duration)
        self._overlay_anim.setStartValue(0.0)
        self._overlay_anim.setEndValue(1.0)
        self._overlay_anim.start()

    def _on_overlay_anim_progress(self, val: float) -> None:
        p = float(val)
        sigma = 0.20
        center = (1.0 + 2.0 * sigma) * p - sigma
        u = np.clip((center - self._x_norm) / (2.0 * sigma) + 0.5, 0.0, 1.0)
        alpha = 3.0 * (u ** 2) - 2.0 * (u ** 3)

        for cname, (old_s, target_s, old_op, target_op, _) in self._anim_overlay_types.items():
            curve_item = getattr(self, f"{cname}_curve", None)
            if curve_item:
                blended = (1.0 - alpha) * old_s + alpha * target_s
                curve_item.setData(self.x_log_points, blended)
                is_chk = getattr(self, f"_is_{cname}_checked", True)
                cur_target_op = target_op if is_chk else 0.0
                op = (1.0 - p) * old_op + p * cur_target_op
                curve_item.setOpacity(op)
                curve_item.setVisible(op > 0.01)

    def _on_overlay_anim_finished(self) -> None:
        setter_map = {
            "meas": self.set_measurement_overlay,
            "target": self.set_target_overlay,
            "target2": self.set_target2_overlay,
            "resultant": self.set_resultant_overlay,
        }
        pending = list(self._anim_overlay_types.items())
        self._anim_overlay_types.clear()
        for cname, (_, _, _, _, final_data) in pending:
            curve_item = getattr(self, f"{cname}_curve", None)
            if curve_item:
                curve_item.setOpacity(1.0)
                setter = setter_map.get(cname)
                if setter:
                    if final_data:
                        f, s, lbl = final_data
                        setter(f, s, label=lbl)
                    else:
                        setter(None, None)
        self._update_overlay_legend()

    # -------------------------------------------------------------------------
    # Band Synchronization
    # -------------------------------------------------------------------------
    def set_bands(self, bands: List[EQBand], preamp_db: float = 0.0) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self.preamp_db = float(preamp_db) if isinstance(preamp_db, (int, float)) and math.isfinite(preamp_db) else 0.0
        self._cached_bypass_active_response = None
        self.hud_text_item.setVisible(False)
        self._hide_band_preview()

        dedup_map: Dict[int, EQBand] = {b.band_id: _sanitize_band(b) for b in bands}
        self.bands = list(dedup_map.values())

        for item in list(self.node_items.values()):
            self.removeItem(item)
        self.node_items.clear()

        for band in self.bands:
            node = self._create_node(band)
            self.addItem(node)
            node.updateTransform()
            self.node_items[band.band_id] = node

        self.recompute_composite_curve()

    def update_band_params(
        self,
        band_id: int,
        freq: Optional[float] = None,
        gain: Optional[float] = None,
        q: Optional[float] = None,
        filter_type: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self._cached_bypass_active_response = None
        for band in self.bands:
            if band.band_id == band_id:
                if freq is not None and math.isfinite(float(freq)) and float(freq) > 0.0:
                    band.freq = float(freq)
                if gain is not None and math.isfinite(float(gain)):
                    band.gain = float(gain)
                if q is not None and math.isfinite(float(q)) and float(q) > 0.0:
                    band.q = float(q)
                if filter_type is not None:
                    band.filter_type = str(filter_type)
                if FilterType.from_str(band.filter_type) in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                    band.gain = 0.0
                if enabled is not None:
                    band.enabled = bool(enabled)

                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band = _sanitize_band(band)
                    node._update_node_position()
                    node.update()
                    if node.is_hovered or node.is_dragging:
                        self._on_node_hover(node, True)
                break

        self.recompute_composite_curve()

    def set_preamp(self, preamp_db: float) -> None:
        self.preamp_db = float(preamp_db) if isinstance(preamp_db, (int, float)) and math.isfinite(preamp_db) else 0.0
        self._cached_bypass_active_response = None
        if not self._is_sweeping:
            self.recompute_composite_curve()

    def recompute_composite_curve(self) -> None:
        if self.is_bypassed and self._eq_active_factor == 0.0:
            flat = np.zeros(len(self.freq_points), dtype=np.float64)
            self._last_composite_response = flat
            self.eq_curve.setData(self.x_log_points, flat)
            self.eq_fill_curve.setData(self.x_log_points, flat)
            return

        response_db = compute_composite_response(
            self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
        )
        if self._eq_active_factor < 1.0:
            response_db = self._eq_active_factor * response_db

        self._last_composite_response = np.asarray(response_db, dtype=np.float64)
        self.eq_curve.setData(self.x_log_points, response_db)
        if not self.is_bypassed:
            self.eq_fill_curve.setData(self.x_log_points, response_db)

    # -------------------------------------------------------------------------
    # Individual Band Preview, Hairline Crosshairs, HUD & Live Cursor Readout
    # -------------------------------------------------------------------------
    def _show_band_preview(self, band: EQBand) -> None:
        b = _sanitize_band(band)
        resp = compute_filter_response(
            b.filter_type, b.freq, b.gain, b.q, self.freq_points, fs=self.sample_rate
        )
        clip_lo = min(-40.0, self.y_min - 20.0)
        clip_hi = max(40.0, self.y_max + 20.0)
        resp_clean = np.clip(np.nan_to_num(resp, nan=0.0, posinf=clip_hi, neginf=clip_lo), clip_lo, clip_hi)

        if b.enabled and not self.is_bypassed:
            self.band_preview_curve.setPen(
                pg.mkPen(QtGui.QColor(161, 161, 170, 195), width=1.3, style=QtCore.Qt.PenStyle.DashLine)
            )
            self.band_preview_fill_curve.setBrush(pg.mkBrush(QtGui.QColor(161, 161, 170, 26)))
        else:
            self.band_preview_curve.setPen(
                pg.mkPen(QtGui.QColor(161, 161, 170, 110), width=1.1, style=QtCore.Qt.PenStyle.DashLine)
            )
            self.band_preview_fill_curve.setBrush(pg.mkBrush(QtGui.QColor(161, 161, 170, 12)))

        self.band_preview_curve.setData(self.x_log_points, resp_clean)
        self.band_preview_fill_curve.setData(self.x_log_points, resp_clean)
        self.band_preview_curve.setVisible(True)
        self.band_preview_fill_curve.setVisible(True)

        x, y = _safe_band_coords(b, self.y_min, self.y_max)
        self.crosshair_v_line.setPos(x)
        self.crosshair_h_line.setPos(y)
        self.crosshair_v_line.setVisible(True)
        self.crosshair_h_line.setVisible(True)
        self._preview_band_id = b.band_id

    def _hide_band_preview(self) -> None:
        self._preview_band_id = None
        self.band_preview_curve.setVisible(False)
        self.band_preview_fill_curve.setVisible(False)
        self.crosshair_v_line.setVisible(False)
        self.crosshair_h_line.setVisible(False)

    def _eval_composite_at_x(self, x_log: float) -> float:
        if self._last_composite_response.size != self.x_log_points.size:
            return 0.0
        val = float(np.interp(x_log, self.x_log_points, self._last_composite_response))
        if not math.isfinite(val) or abs(val) < 1e-9:
            return 0.0
        return val

    def _on_scene_mouse_moved(self, scene_pos: QtCore.QPointF) -> None:
        if any(n.is_hovered or n.is_dragging for n in self.node_items.values()):
            self.cursor_text_item.setVisible(False)
            return

        if not self.custom_vb.sceneBoundingRect().contains(scene_pos):
            self.cursor_text_item.setVisible(False)
            return

        view_pos = self.custom_vb.mapSceneToView(scene_pos)
        x = float(view_pos.x())
        y = float(view_pos.y())
        if not (X_MIN <= x <= X_MAX and self.y_min <= y <= self.y_max):
            self.cursor_text_item.setVisible(False)
            return

        freq = 10.0 ** x
        freq_str = _format_freq(freq)
        note_str = _freq_to_musical_note(freq)
        cursor_db = 0.0 if abs(y) < 0.05 else y
        eq_db = self._eval_composite_at_x(x)
        if abs(eq_db) < 0.05:
            eq_db = 0.0

        self.cursor_text_item.setHtml(
            "<span style=\"font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 11px; color: #a1a1aa;\">"
            f"{html.escape(freq_str)}&nbsp;&nbsp;"
            f"<span style='color: #71717a;'>{html.escape(note_str)}</span>&nbsp;&nbsp;"
            f"{cursor_db:+.1f} dB&nbsp;&nbsp;"
            f"<span style='color: #71717a;'>EQ</span> "
            f"<span style='color: #ededed;'>{eq_db:+.1f} dB</span>"
            "</span>"
        )
        self.cursor_text_item.setVisible(True)

    def _on_node_hover(self, node: EQNodeItem, is_hovered: bool) -> None:
        if is_hovered:
            self.cursor_text_item.setVisible(False)
            b = node.band
            f_str = _format_freq(b.freq)
            note_str = _freq_to_musical_note(b.freq)
            gain_val = 0.0 if abs(b.gain) < 0.05 else b.gain
            sep = "<span style='color: #3f3f46;'> &nbsp;|&nbsp; </span>"
            parts = [
                f"<span style='color: #ededed;'>Band {b.band_id}</span>",
                f"<span style='color: #a1a1aa;'>{html.escape(str(b.filter_type))}</span>",
                f"<span style='color: #ededed;'>{html.escape(f_str)}</span> <span style='color: #71717a;'>{html.escape(note_str)}</span>",
                f"<span style='color: #ededed;'>{gain_val:+.1f} dB</span>",
                f"<span style='color: #a1a1aa;'>Q {b.q:.2f}</span>",
            ]
            if not b.enabled:
                parts.append("<span style='color: #71717a;'>Bypassed</span>")
            if node.is_dragging:
                mods = node._active_drag_modifiers
                mod_tags: List[str] = []
                if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    mod_tags.append("Fine")
                if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
                    mod_tags.append("Freq lock")
                if mods & QtCore.Qt.KeyboardModifier.AltModifier:
                    mod_tags.append("Gain lock")
                if mod_tags:
                    parts.append(f"<span style='color: #71717a;'>{html.escape(', '.join(mod_tags))}</span>")

            self.hud_text_item.setHtml(
                "<span style=\"font-family: 'Cascadia Mono', 'Consolas', monospace; font-size: 11px;\">"
                + sep.join(parts)
                + "</span>"
            )
            self.hud_text_item.setVisible(True)
            self._show_band_preview(b)
        else:
            active_node = next((n for n in self.node_items.values() if n.is_dragging or n.is_hovered), None)
            if active_node is not None:
                self._on_node_hover(active_node, True)
            else:
                self.hud_text_item.setVisible(False)
                self._hide_band_preview()

    # -------------------------------------------------------------------------
    # Node Callbacks
    # -------------------------------------------------------------------------
    def _on_node_moved(self, band_id: int, freq: float, gain: float) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self._cached_bypass_active_response = None
        for band in self.bands:
            if band.band_id == band_id:
                band.freq = float(freq) if math.isfinite(float(freq)) and float(freq) > 0.0 else band.freq
                if FilterType.from_str(band.filter_type) in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                    band.gain = 0.0
                else:
                    band.gain = float(gain) if math.isfinite(float(gain)) else band.gain
                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band.freq = band.freq
                    node.band.gain = band.gain
                    node._update_node_position()
                    node.update()
                    if node.is_hovered or node.is_dragging:
                        self._on_node_hover(node, True)
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_q_changed(self, band_id: int, q: float) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self._cached_bypass_active_response = None
        for band in self.bands:
            if band.band_id == band_id:
                band.q = float(q) if math.isfinite(float(q)) and float(q) > 0.0 else band.q
                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band.q = band.q
                    node.update()
                    if node.is_hovered or node.is_dragging:
                        self._on_node_hover(node, True)
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_type_changed(self, band_id: int, ftype: str) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self._cached_bypass_active_response = None
        for band in self.bands:
            if band.band_id == band_id:
                band.filter_type = ftype
                if FilterType.from_str(ftype) in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                    band.gain = 0.0
                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band = _sanitize_band(band)
                    node._update_node_position()
                    node.update()
                    if node.is_hovered or node.is_dragging:
                        self._on_node_hover(node, True)
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_bypass_toggled(self, band_id: int, enabled: bool) -> None:
        self._cancel_sweep_animation(finalize_overlays=True)
        self._cached_bypass_active_response = None
        for band in self.bands:
            if band.band_id == band_id:
                band.enabled = enabled
                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band.enabled = enabled
                    node.update()
                    if node.is_hovered or node.is_dragging:
                        self._on_node_hover(node, True)
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _sanitize_spectrum_arrays(
        self,
        freqs: Any,
        mags_db: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if freqs is None or mags_db is None:
            return None
        try:
            y_arr = np.asarray(mags_db, dtype=np.float64).ravel()
        except (TypeError, ValueError):
            return None
        if y_arr.size == 0:
            return None
        if y_arr.size == self.x_log_points.size:
            x_arr = self.x_log_points
        else:
            try:
                f_arr = np.asarray(freqs, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                return None
            if f_arr.size != y_arr.size:
                return None
            valid = np.isfinite(f_arr) & (f_arr > 0.0)
            if not np.any(valid):
                return None
            x_arr = np.log10(np.clip(f_arr[valid], MIN_FREQ, MAX_FREQ))
            y_arr = y_arr[valid]
        if not np.all(np.isfinite(y_arr)):
            y_arr = np.nan_to_num(y_arr, nan=self.y_min, posinf=self.y_max, neginf=self.y_min)
        return x_arr, y_arr

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_spectrum_curve(self, freqs: np.ndarray, mags_db: np.ndarray) -> None:
        if not self.spectrum_curve.isVisible():
            return
        cleaned = self._sanitize_spectrum_arrays(freqs, mags_db)
        if cleaned is None:
            return
        self.spectrum_curve.setData(cleaned[0], cleaned[1])

    @QtCore.pyqtSlot(object, object)
    def update_peak_hold_curve(self, freqs: Any, mags_db: Any) -> None:
        if not self.peak_hold_curve.isVisible():
            return
        cleaned = self._sanitize_spectrum_arrays(freqs, mags_db)
        if cleaned is None:
            self.peak_hold_curve.setData([], [])
            return
        self.peak_hold_curve.setData(cleaned[0], cleaned[1])

    def set_peak_hold_visible(self, visible: bool) -> None:
        """Shows or hides the live spectrum peak-hold ghost trace."""
        vis = bool(visible)
        self._peak_hold_visible = vis
        if not vis:
            self.peak_hold_curve.setData([], [])
            self.peak_hold_curve.setVisible(False)
        else:
            self.peak_hold_curve.setVisible(True)

    # -------------------------------------------------------------------------
    # AutoEQ Curve Overlays (Measurement & Target Reference Traces)
    # -------------------------------------------------------------------------
    def _set_sweeping_overlay(
        self,
        cname: str,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str,
    ) -> None:
        flag_attr = f"_sweep_{cname}_animating"
        old_attr = f"_sweep_old_{cname}"
        old_op_attr = f"_sweep_old_{cname}_opacity"
        target_attr = f"_sweep_target_{cname}"
        target_op_attr = f"_sweep_target_{cname}_opacity"
        pend_attr = f"_pending_final_{cname}"

        if not getattr(self, flag_attr):
            old_y, old_op = self._capture_overlay_state(cname)
            setattr(self, old_attr, old_y)
            setattr(self, old_op_attr, old_op)

        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is not None:
            cf, cs = cleaned
            setattr(self, f"_{cname}_freqs", cf)
            setattr(self, f"_{cname}_spls", cs)
            setattr(self, f"_{cname}_label", label)
            setattr(self, target_attr, np.interp(self.freq_points, cf, cs))
            is_chk = getattr(self, f"_is_{cname}_checked", True)
            setattr(self, target_op_attr, 1.0 if is_chk else 0.0)
            setattr(self, pend_attr, (cf, cs, label))
        else:
            setattr(self, f"_{cname}_freqs", None)
            setattr(self, f"_{cname}_spls", None)
            setattr(self, f"_{cname}_label", "")
            setattr(self, target_attr, np.copy(getattr(self, old_attr)))
            setattr(self, target_op_attr, 0.0)
            setattr(self, pend_attr, None)

        setattr(self, flag_attr, True)
        self._update_overlay_legend()

    def set_measurement_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        self._stop_overlay_anim("meas")
        if self._is_sweeping:
            self._set_sweeping_overlay("meas", freqs, spls, label)
            return

        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is None:
            self.meas_curve.setData([], [])
            self.meas_curve.setVisible(False)
            self._meas_freqs = None
            self._meas_spls = None
            self._meas_label = ""
            self._update_overlay_legend()
            return

        self._meas_freqs, self._meas_spls = cleaned
        self._meas_label = label

        x_log = np.log10(self._meas_freqs)
        self.meas_curve.setData(x_log, self._meas_spls)
        self.meas_curve.setVisible(self._is_meas_checked)
        self._update_overlay_legend()

    def set_target_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        self._stop_overlay_anim("target")
        if self._is_sweeping:
            self._set_sweeping_overlay("target", freqs, spls, label)
            return

        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is None:
            self.target_curve.setData([], [])
            self.target_curve.setVisible(False)
            self._target_freqs = None
            self._target_spls = None
            self._target_label = ""
            self._update_overlay_legend()
            return

        self._target_freqs, self._target_spls = cleaned
        self._target_label = label

        x_log = np.log10(self._target_freqs)
        self.target_curve.setData(x_log, self._target_spls)
        self.target_curve.setVisible(self._is_target_checked)
        self._update_overlay_legend()

    def set_target2_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        self._stop_overlay_anim("target2")
        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is None:
            self.target2_curve.setData([], [])
            self.target2_curve.setVisible(False)
            self._target2_freqs = None
            self._target2_spls = None
            self._target2_label = ""
            self._update_overlay_legend()
            return

        self._target2_freqs, self._target2_spls = cleaned
        self._target2_label = label

        x_log = np.log10(self._target2_freqs)
        self.target2_curve.setData(x_log, self._target2_spls)
        self.target2_curve.setVisible(self._is_target2_checked)
        self._update_overlay_legend()

    def set_resultant_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        """Sets the resultant predicted frequency response overlay (Model + EQ in cyan)."""
        self._stop_overlay_anim("resultant")
        if self._is_sweeping:
            self._set_sweeping_overlay("resultant", freqs, spls, label)
            return

        cleaned = _sanitize_curve_data(freqs, spls)
        if cleaned is None:
            self.resultant_curve.setData([], [])
            self.resultant_curve.setVisible(False)
            self._resultant_freqs = None
            self._resultant_spls = None
            self._resultant_label = ""
            self._update_overlay_legend()
            return

        self._resultant_freqs, self._resultant_spls = cleaned
        self._resultant_label = label

        x_log = np.log10(self._resultant_freqs)
        self.resultant_curve.setData(x_log, self._resultant_spls)
        self.resultant_curve.setVisible(self._is_resultant_checked)
        self._update_overlay_legend()

    def clear_overlays(self) -> None:
        self._stop_overlay_anim()
        # target2 is not part of the preset sweep animation, so always clear it immediately
        self.target2_curve.setData([], [])
        self.target2_curve.setVisible(False)
        self._target2_freqs = None
        self._target2_spls = None
        self._target2_label = ""

        if self._is_sweeping:
            for cname in ("meas", "target", "resultant"):
                curve_item = getattr(self, f"{cname}_curve")
                if (
                    getattr(self, f"_sweep_{cname}_animating")
                    or curve_item.isVisible()
                    or getattr(self, f"_{cname}_freqs") is not None
                ):
                    self._set_sweeping_overlay(cname, None, None, "")
                else:
                    setattr(self, f"_{cname}_label", "")
            self._update_overlay_legend()
            return

        self.meas_curve.setData([], [])
        self.target_curve.setData([], [])
        self.resultant_curve.setData([], [])
        self.meas_curve.setVisible(False)
        self.target_curve.setVisible(False)
        self.resultant_curve.setVisible(False)
        self._meas_freqs = None
        self._meas_spls = None
        self._meas_label = ""
        self._target_freqs = None
        self._target_spls = None
        self._target_label = ""
        self._resultant_freqs = None
        self._resultant_spls = None
        self._resultant_label = ""
        self._update_overlay_legend()

    def set_overlays_visible(
        self,
        show_meas: bool,
        show_target: bool,
        show_resultant: bool = True,
        show_target2: bool = True,
    ) -> None:
        self._is_meas_checked = bool(show_meas)
        self._is_target_checked = bool(show_target)
        self._is_resultant_checked = bool(show_resultant)
        self._is_target2_checked = bool(show_target2)
        overlay_running = self._overlay_anim.state() == QtCore.QAbstractAnimation.State.Running

        if self._is_sweeping:
            if self._sweep_meas_animating:
                self._sweep_target_meas_opacity = 1.0 if (self._is_meas_checked and self._pending_final_meas is not None) else 0.0
            else:
                self.meas_curve.setVisible(self._is_meas_checked and (self._meas_freqs is not None))

            if self._sweep_target_animating:
                self._sweep_target_target_opacity = 1.0 if (self._is_target_checked and self._pending_final_target is not None) else 0.0
            else:
                self.target_curve.setVisible(self._is_target_checked and (self._target_freqs is not None))

            if self._sweep_resultant_animating:
                self._sweep_target_resultant_opacity = 1.0 if (self._is_resultant_checked and self._pending_final_resultant is not None) else 0.0
            else:
                self.resultant_curve.setVisible(self._is_resultant_checked and (self._resultant_freqs is not None))

            self.target2_curve.setVisible(self._is_target2_checked and (self._target2_freqs is not None))
        else:
            if not (overlay_running and "meas" in self._anim_overlay_types):
                self.meas_curve.setVisible(self._is_meas_checked and (self._meas_freqs is not None))
            if not (overlay_running and "target" in self._anim_overlay_types):
                self.target_curve.setVisible(self._is_target_checked and (self._target_freqs is not None))
            if not (overlay_running and "target2" in self._anim_overlay_types):
                self.target2_curve.setVisible(self._is_target2_checked and (self._target2_freqs is not None))
            if not (overlay_running and "resultant" in self._anim_overlay_types):
                self.resultant_curve.setVisible(self._is_resultant_checked and (self._resultant_freqs is not None))
        self._update_overlay_legend()

    def _update_overlay_legend(self) -> None:
        items = []
        base_style = "font-family: 'Segoe UI', sans-serif; font-size: 11px; font-weight: 500;"
        if self._target_label:
            vis = self._is_target_checked
            color = "#facc15" if vis else "#71717a"
            decor = "none" if vis else "line-through"
            safe_txt = html.escape(f"Target: {self._target_label}", quote=True)
            items.append(
                f'<a href="target" style="{base_style} color: {color}; text-decoration: {decor};">{safe_txt}</a>'
            )

        if self._target2_label:
            vis2 = self._is_target2_checked
            color2 = "#c084fc" if vis2 else "#71717a"
            decor2 = "none" if vis2 else "line-through"
            safe_txt2 = html.escape(f"Target 2: {self._target2_label}", quote=True)
            items.append(
                f'<a href="target2" style="{base_style} color: {color2}; text-decoration: {decor2};">{safe_txt2}</a>'
            )

        if self._meas_label:
            vis = self._is_meas_checked
            color = "#a1a1aa" if vis else "#71717a"
            decor = "none" if vis else "line-through"
            safe_txt = html.escape(f"Measurement: {self._meas_label}", quote=True)
            items.append(
                f'<a href="meas" style="{base_style} color: {color}; text-decoration: {decor};">{safe_txt}</a>'
            )

        if self._resultant_label:
            vis = self._is_resultant_checked
            color = "#38bdf8" if vis else "#71717a"
            decor = "none" if vis else "line-through"
            safe_txt = html.escape(f"Resultant: {self._resultant_label}", quote=True)
            items.append(
                f'<a href="resultant" style="{base_style} color: {color}; text-decoration: {decor};">{safe_txt}</a>'
            )

        if items:
            sep = '<span style="color: #3f3f46;"> &nbsp;|&nbsp; </span>'
            self.overlay_legend_item.setHtml(sep.join(items))
            self.overlay_legend_item.setVisible(True)
        else:
            self.overlay_legend_item.setHtml("")
            self.overlay_legend_item.setVisible(False)

    # -------------------------------------------------------------------------
    # Canvas Context Menu & Working Plot Options
    # -------------------------------------------------------------------------
    def _toggle_x_grid(self, checked: bool) -> None:
        self.grid_item.set_show_x(checked)

    def _toggle_x_major_grid(self, checked: bool) -> None:
        self.grid_item.set_show_x_major(checked)

    def _toggle_x_minor_grid(self, checked: bool) -> None:
        self.grid_item.set_show_x_minor(checked)

    def _toggle_y_grid(self, checked: bool) -> None:
        self.grid_item.set_show_y(checked)

    def _toggle_spectrum_overlay(self, visible: bool) -> None:
        vis = bool(visible)
        self.spectrum_curve.setVisible(vis)
        self.peak_hold_curve.setVisible(vis and self._peak_hold_visible)
        self.sigSpectrumToggled.emit(vis)

    def _toggle_overlay_from_menu(self, curve_type: str, checked: bool) -> None:
        if self.receivers(self.sigLegendCurveClicked) > 0:
            self.sigLegendCurveClicked.emit(curve_type)
        else:
            setattr(self, f"_is_{curve_type}_checked", bool(checked))
            self.set_overlays_visible(
                self._is_meas_checked,
                self._is_target_checked,
                self._is_resultant_checked,
                show_target2=self._is_target2_checked,
            )

    def reset_default_view(self) -> None:
        y_step = 4 if self._y_range_db == 12 else 10
        self.setXRange(X_MIN, X_MAX, padding=0.0)
        self.setYRange(self.y_min, self.y_max, padding=0.0)
        self.grid_item.set_bounds(X_MIN, X_MAX, self.y_min, self.y_max, y_step=y_step)

    def export_plot_image(self) -> None:
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export EQ plot image",
            "clean_apo_eq_curve.png",
            "PNG image (*.png);;JPEG image (*.jpg);;All files (*)",
        )
        if path:
            if not os.path.splitext(path)[1]:
                path += ".jpg" if "JPEG" in selected_filter else ".png"
            pix = self.grab()
            if not pix.save(path):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export failed",
                    f"Could not save EQ plot image to:\n{path}",
                )

    def show_canvas_context_menu(self, screen_pos: QtCore.QPoint, freq: float, gain: float) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(_build_context_menu_qss())

        f_str = _format_freq(freq)
        gain_val = 0.0 if abs(gain) < 0.05 else gain
        add_act = menu.addAction(f"Add filter band at {f_str} ({gain_val:+.1f} dB)")
        add_act.triggered.connect(lambda: self.sigBandAdded.emit(freq, gain))

        menu.addSeparator()

        scale_12_act = menu.addAction("Scale: ±12 dB")
        scale_12_act.setCheckable(True)
        scale_12_act.setChecked(self._y_range_db == 12)
        scale_12_act.triggered.connect(lambda: self.set_y_range_mode(12))

        scale_20_act = menu.addAction("Scale: ±20 dB")
        scale_20_act.setCheckable(True)
        scale_20_act.setChecked(self._y_range_db == 20)
        scale_20_act.triggered.connect(lambda: self.set_y_range_mode(20))

        menu.addSeparator()

        plot_opts_menu = menu.addMenu("Plot options")
        plot_opts_menu.setStyleSheet(menu.styleSheet())

        x_maj_act = plot_opts_menu.addAction("Show frequency grid")
        x_maj_act.setCheckable(True)
        x_maj_act.setChecked(self.grid_item.show_x_major)
        x_maj_act.triggered.connect(self._toggle_x_major_grid)

        x_min_act = plot_opts_menu.addAction("Show intermediate grid")
        x_min_act.setCheckable(True)
        x_min_act.setChecked(self.grid_item.show_x_minor)
        x_min_act.triggered.connect(self._toggle_x_minor_grid)

        y_act = plot_opts_menu.addAction("Show decibel grid")
        y_act.setCheckable(True)
        y_act.setChecked(self.grid_item.show_y)
        y_act.triggered.connect(self._toggle_y_grid)

        plot_opts_menu.addSeparator()

        spec_act = plot_opts_menu.addAction("Show live spectrum")
        spec_act.setCheckable(True)
        spec_act.setChecked(self.spectrum_curve.isVisible())
        spec_act.triggered.connect(self._toggle_spectrum_overlay)

        if self._meas_freqs is not None:
            meas_act = plot_opts_menu.addAction("Show IEM measurement curve")
            meas_act.setCheckable(True)
            meas_act.setChecked(self._is_meas_checked)
            meas_act.triggered.connect(lambda c: self._toggle_overlay_from_menu("meas", c))

        if self._target_freqs is not None:
            targ_act = plot_opts_menu.addAction("Show target curve")
            targ_act.setCheckable(True)
            targ_act.setChecked(self._is_target_checked)
            targ_act.triggered.connect(lambda c: self._toggle_overlay_from_menu("target", c))

        if self._target2_freqs is not None:
            targ2_act = plot_opts_menu.addAction("Show secondary target curve")
            targ2_act.setCheckable(True)
            targ2_act.setChecked(self._is_target2_checked)
            targ2_act.triggered.connect(lambda c: self._toggle_overlay_from_menu("target2", c))

        if self._resultant_freqs is not None:
            res_act = plot_opts_menu.addAction("Show resultant curve")
            res_act.setCheckable(True)
            res_act.setChecked(self._is_resultant_checked)
            res_act.triggered.connect(lambda c: self._toggle_overlay_from_menu("resultant", c))

        reset_act = plot_opts_menu.addAction(
            f"Reset view (20 Hz to 20 kHz, ±{self._y_range_db} dB)"
        )
        reset_act.triggered.connect(self.reset_default_view)

        menu.addSeparator()

        export_act = menu.addAction("Export image")
        export_act.triggered.connect(self.export_plot_image)

        menu.exec(screen_pos)
