"""
Interactive pyqtgraph frequency response canvas, draggable filter nodes, live spectrum overlay,
and cubic-smoothed sweeping graph transition animations.
"""

from __future__ import annotations
import sys
import math
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from biquad import EQBand, FilterType, compute_composite_response

MIN_FREQ = 20.0
MAX_FREQ = 20000.0
MIN_GAIN = -20.0
MAX_GAIN = 20.0
X_MIN = math.log10(MIN_FREQ)
X_MAX = math.log10(MAX_FREQ)


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


class EQGridItem(pg.GraphicsObject):
    """
    Subtle background grid item rendered inside the ViewBox at zValue = 0,
    strictly behind curves (zValue = 10) and band nodes (zValue = 25).
    Guarantees grid lines never cut across or through node badges or curves.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, parent=None):
        super().__init__(parent)
        self.setZValue(0)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.show_x_major = True
        self.show_x_minor = True
        self.show_y = True

    def set_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        self.prepareGeometryChange()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
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
        major_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 32), 1.0)
        major_pen.setCosmetic(True)
        minor_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 15), 1.0)
        minor_pen.setCosmetic(True)

        # Horizontal grid lines (every 10 dB: -20, -10, 0, 10, 20)
        if self.show_y:
            painter.setPen(major_pen)
            for g in range(int(self.y_min), int(self.y_max) + 1, 10):
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
    - Monochrome Black & White styling: Crisp white badge with bold numeral.
    - Generous circular hitbox (covers entire badge and outer padding) for effortless clicking.
    - Uses inverse parent transform instead of ItemIgnoresTransformations to prevent Qt6 hit-testing clipping.
    - Hand cursor on hover, closed hand on drag.
    - Mouse wheel: Q factor adjustment.
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

    def __init__(self, band: EQBand, viewBox: pg.ViewBox, parent=None):
        super().__init__(parent)
        self.band = EQBand.from_dict(band.to_dict())
        self.vb = viewBox
        
        # Badge radius: 13.0 px (26 px diameter visible badge)
        self.radius = 13.0
        self.is_dragging = False
        self.is_hovered = False
        self.is_bypassed_globally = False
        self._last_transform = None

        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setZValue(25)

        self._update_node_position()

    def updateTransform(self) -> None:
        """
        Maintains constant pixel size without ItemIgnoresTransformations,
        preserving full circular hit detection across the entire badge in Qt6.
        """
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
        x = math.log10(max(MIN_FREQ, min(MAX_FREQ, self.band.freq)))
        y = max(MIN_GAIN, min(MAX_GAIN, self.band.gain))
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
        self.updateTransform()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # 1. Halo Glow on Hover or Drag
        if (self.is_dragging or self.is_hovered) and not self.is_bypassed_globally:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 45)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius + 7.0, self.radius + 7.0)

        # 2. Main Badge (Monochrome Black & White)
        if self.is_bypassed_globally:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(25, 25, 30)))
            painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 110), 1.2, QtCore.Qt.PenStyle.DashLine))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor(100, 100, 110)))
        elif self.band.enabled:
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
            painter.setPen(QtGui.QPen(QtGui.QColor("#18181b"), 1.2))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor("#000000")))
        else:
            painter.setBrush(QtGui.QBrush(QtGui.QColor(18, 18, 22)))
            painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 190), 1.4, QtCore.Qt.PenStyle.DashLine))
            painter.drawEllipse(QtCore.QPointF(0, 0), self.radius, self.radius)
            painter.setPen(QtGui.QPen(QtGui.QColor(160, 160, 170)))

        # 3. Bold Band Number Typography
        font = QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        text_rect = QtCore.QRectF(-self.radius, -self.radius - 1.0, 2 * self.radius, 2 * self.radius)
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
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            self.prepareGeometryChange()
            self.update()
            if self.scene():
                self.scene().update()
            self.sigHoverChanged.emit(self, True)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self.is_dragging and self.vb:
            view_pos = self.vb.mapSceneToView(event.scenePos())
            x = max(X_MIN, min(X_MAX, view_pos.x()))
            
            ftype = FilterType.from_str(self.band.filter_type)
            if ftype in (FilterType.HIGH_PASS, FilterType.LOW_PASS):
                y = 0.0
            else:
                y = max(MIN_GAIN, min(MAX_GAIN, view_pos.y()))

            self.prepareGeometryChange()
            self.setPos(x, y)
            new_freq = 10.0 ** x
            new_gain = y
            self.band.freq = new_freq
            self.band.gain = new_gain
            self.sigMoved.emit(self.band.band_id, new_freq, new_gain)
            self.sigHoverChanged.emit(self, True)
            
            if self.scene():
                self.scene().update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self.prepareGeometryChange()
            self.update()
            if self.scene():
                self.scene().update()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtWidgets.QGraphicsSceneWheelEvent) -> None:
        self.sigWheelScrollStarted.emit()
        self.sigDragStarted.emit(self.band.band_id)
        delta = event.delta()
        if delta > 0:
            new_q = min(20.0, self.band.q * 1.1)
        else:
            new_q = max(0.1, self.band.q / 1.1)

        self.band.q = round(new_q, 3)
        self.sigQChanged.emit(self.band.band_id, self.band.q)
        self.sigHoverChanged.emit(self, True)
        self.update()
        if self.scene():
            self.scene().update()
        event.accept()

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        menu.setStyleSheet(
            f"QMenu {{ background-color: #000000; color: #ffffff; border: 1px solid #3f3f46; border-radius: 0px; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 20px 6px 28px; border-radius: 0px; }}"
            f"QMenu::item:selected {{ background-color: #ffffff; color: #000000; font-weight: bold; }}"
            f"QMenu::separator {{ height: 1px; background: #27272a; margin: 4px 6px; }}"
            f"QMenu::indicator {{ width: 14px; height: 14px; left: 8px; }}"
            f"QMenu::indicator:unchecked {{ image: none; }}"
            f"QMenu::indicator:checked {{ image: url('{CHECK_WHITE_PATH}'); }}"
            f"QMenu::indicator:checked:selected {{ image: url('{CHECK_BLACK_PATH}'); }}"
        )

        title_act = menu.addAction(f"Band {self.band.band_id} ({self.band.filter_type})")
        title_act.setEnabled(False)
        menu.addSeparator()

        type_menu = menu.addMenu("Filter Type")
        type_menu.setStyleSheet(menu.styleSheet())
        for f_type in [FilterType.PEAK, FilterType.LOW_SHELF, FilterType.HIGH_SHELF, FilterType.HIGH_PASS, FilterType.LOW_PASS]:
            act = type_menu.addAction(f_type.value)
            act.setCheckable(True)
            act.setChecked(FilterType.from_str(self.band.filter_type) == f_type)
            act.triggered.connect(lambda checked, t=f_type.value: self.sigTypeChanged.emit(self.band.band_id, t))

        bypass_text = "Enable Filter" if not self.band.enabled else "Bypass Filter"
        bypass_act = menu.addAction(bypass_text)
        bypass_act.triggered.connect(lambda: self.sigBypassToggled.emit(self.band.band_id, not self.band.enabled))

        reset_act = menu.addAction("Reset Gain to 0 dB")
        reset_act.triggered.connect(lambda: self.sigMoved.emit(self.band.band_id, self.band.freq, 0.0))

        menu.addSeparator()
        delete_act = menu.addAction("Delete Band")
        delete_act.triggered.connect(lambda: self.sigDeleteRequested.emit(self.band.band_id))

        menu.exec(event.screenPos())
        event.accept()


class EQViewBox(pg.ViewBox):
    sigCanvasDoubleClicked = QtCore.pyqtSignal(float, float)

    def __init__(self, plot_widget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_widget = plot_widget
        self.setMouseMode(pg.ViewBox.PanMode)
        self.setLimits(xMin=X_MIN, xMax=X_MAX, yMin=MIN_GAIN, yMax=MAX_GAIN)

    def mouseDoubleClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = self.mapSceneToView(ev.scenePos())
            x = max(X_MIN, min(X_MAX, pos.x()))
            y = max(MIN_GAIN, min(MAX_GAIN, pos.y()))
            freq = 10.0 ** x
            self.sigCanvasDoubleClicked.emit(freq, y)
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            ev.accept()
            self.raiseContextMenu(ev)
        else:
            super().mouseClickEvent(ev)

    def raiseContextMenu(self, ev) -> None:
        if self.plot_widget is not None:
            pos = self.mapSceneToView(ev.scenePos())
            x = max(X_MIN, min(X_MAX, pos.x()))
            y = max(MIN_GAIN, min(MAX_GAIN, pos.y()))
            freq = 10.0 ** x
            self.plot_widget.show_canvas_context_menu(ev.screenPos().toPoint(), freq, y)
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
        self.setHeight(24)

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
        return rect.adjusted(-15, 0, 15, 0)


class DecibelAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(orientation="left", *args, **kwargs)
        self.setPen(pg.mkPen("#27272a", width=1))
        self.setTextPen(pg.mkPen("#a1a1aa"))
        self.setWidth(52)
        ticks = [
            (val, f"{val:+.0f} dB" if val != 0 else "0 dB")
            for val in range(-20, 25, 10)
        ]
        self.setTicks([ticks])

    def setGrid(self, val) -> None:
        # Prevent AxisItem from painting grid on top of the view and points
        self.grid = False

    def boundingRect(self) -> QtCore.QRectF:
        rect = super().boundingRect()
        # Always maintain vertical margin so +20 dB and -20 dB labels are never dropped by pyqtgraph when grid is toggled
        return rect.adjusted(0, -15, 0, 15)


class ClickableOverlayLegend(pg.TextItem):
    """Interactive text banner on the graph canvas where clicking a trace label toggles it."""
    sigCurveClicked = QtCore.pyqtSignal(str)  # "target", "meas", "resultant"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regions: List[Tuple[float, float, str]] = []
        if hasattr(self, "textItem"):
            self.textItem.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            w = self.textItem.boundingRect().width()
            if w > 0:
                click_x = ev.pos().x()
                ratio = max(0.0, min(1.0, click_x / w))
                for r_start, r_end, ctype in self.regions:
                    if r_start <= ratio <= r_end:
                        ev.accept()
                        self.sigCurveClicked.emit(ctype)
                        return
        super().mousePressEvent(ev)


class EQPlotWidget(pg.PlotWidget):
    """
    Main EQ canvas hosting:
    - Logarithmic frequency grid
    - Live audio spectrum analyzer underlay
    - Composite frequency response line (crisp white, 60+ FPS)
    - Interactive draggable band nodes with generous circular hitboxes
    - Floating HUD readouts (top banner, zero trails)
    - Silky smooth 700 ms cubic sweeping transition for the graph curve
    """
    sigBandModified = QtCore.pyqtSignal(int, float, float, float, str, bool)
    sigBandAdded = QtCore.pyqtSignal(float, float)
    sigBandDeleted = QtCore.pyqtSignal(int)
    sigSpectrumToggled = QtCore.pyqtSignal(bool)
    sigLegendCurveClicked = QtCore.pyqtSignal(str)
    sigBandDragStarted = QtCore.pyqtSignal(int)
    sigWheelScrollStarted = QtCore.pyqtSignal()

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
        self.sample_rate = 48000.0

        self.preamp_db = 0.0
        self.bands: List[EQBand] = []
        self.node_items: Dict[int, EQNodeItem] = {}
        self.is_bypassed = False
        self._eq_active_factor = 1.0

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
        self.getPlotItem().layout.setContentsMargins(10, 12, 20, 10)
        
        self.setXRange(X_MIN, X_MAX, padding=0.0)
        self.setYRange(MIN_GAIN, MAX_GAIN, padding=0.0)
        self.disableAutoRange()

        # Dedicated background grid inside ViewBox at zValue = 0 (strictly behind curves & nodes)
        self.grid_item = EQGridItem(X_MIN, X_MAX, MIN_GAIN, MAX_GAIN)
        self.addItem(self.grid_item)

        # 0 dB Reference line
        zero_pen = pg.mkPen(QtGui.QColor(82, 82, 91, 140), width=1.0, style=QtCore.Qt.PenStyle.DashLine)
        self.zero_line = pg.InfiniteLine(pos=0.0, angle=0.0, pen=zero_pen)
        self.zero_line.setZValue(2)
        self.addItem(self.zero_line)

        # Raw IEM Measurement Overlay Curve (subtle dashed line)
        self.meas_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(161, 161, 170, 160), width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.meas_curve.setZValue(4)
        self.meas_curve.setVisible(False)
        self.addItem(self.meas_curve)

        # Target Compensation Reference Curve (distinct dashed amber/yellow line)
        self.target_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(250, 204, 21, 190), width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.target_curve.setZValue(5)
        self.target_curve.setVisible(False)
        self.addItem(self.target_curve)

        # Secondary Target Reference Curve (distinct dashed violet/purple line)
        self.target2_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(192, 132, 252, 200), width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            antialias=True,
        )
        self.target2_curve.setZValue(5)
        self.target2_curve.setVisible(False)
        self.addItem(self.target2_curve)

        # Resultant Predicted Response Curve (Model + EQ in crisp cyan)
        self.resultant_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(56, 189, 248, 220), width=1.8),
            antialias=True,
        )
        self.resultant_curve.setZValue(6)
        self.resultant_curve.setVisible(False)
        self.addItem(self.resultant_curve)

        # Spectrum Analyzer Curve (translucent white underlay)
        self.spectrum_curve = pg.PlotCurveItem(
            pen=pg.mkPen(QtGui.QColor(255, 255, 255, 60), width=1.0),
            brush=pg.mkBrush(QtGui.QColor(255, 255, 255, 20)),
            fillLevel=MIN_GAIN,
            antialias=True,
        )
        self.spectrum_curve.setZValue(1)
        self.addItem(self.spectrum_curve)

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

        # HUD Info Text Item (top-left banner)
        self.hud_text_item = pg.TextItem(text="", color="#ffffff", anchor=(0, 0))
        self.hud_text_item.setPos(X_MIN + 0.05, MAX_GAIN - 1.2)
        self.hud_text_item.setZValue(30)
        self.hud_text_item.setVisible(False)
        self.addItem(self.hud_text_item)

        # Bypassed Watermark (top-right banner)
        self.bypass_text_item = pg.TextItem(
            text="EQ OFF", color="#71717a", anchor=(1, 0)
        )
        self.bypass_text_item.setPos(X_MAX - 0.05, MAX_GAIN - 1.2)
        self.bypass_text_item.setZValue(30)
        self.bypass_text_item.setVisible(False)
        self.addItem(self.bypass_text_item)

        # AutoEQ Overlay Legend (bottom-left banner, interactive clickable)
        self.overlay_legend_item = ClickableOverlayLegend(text="", color="#a1a1aa", anchor=(0, 1))
        self.overlay_legend_item.setPos(X_MIN + 0.05, MIN_GAIN + 1.2)
        self.overlay_legend_item.setZValue(30)
        self.overlay_legend_item.setVisible(False)
        self.overlay_legend_item.sigCurveClicked.connect(self.sigLegendCurveClicked.emit)
        self.addItem(self.overlay_legend_item)

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

    def _on_view_range_changed(self) -> None:
        for node in self.node_items.values():
            node.updateTransform()
        for node in self._node_anim_removed:
            node.updateTransform()

    # -------------------------------------------------------------------------
    # Universal EQ Bypass / Toggle Animation
    # -------------------------------------------------------------------------
    def set_eq_bypassed(self, bypassed: bool) -> None:
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
        - Simultaneously sweeps and transitions target, measurement, and resultant traces!
        """
        if self._is_sweeping:
            self._sweep_animation.stop()
            # Capture currently displayed curves so animation branches seamlessly from current visual state
            _, curr_eq = self.eq_curve.getData()
            if curr_eq is not None and len(curr_eq) == len(self.freq_points):
                self._sweep_old_response = np.asarray(curr_eq, dtype=np.float64)
            else:
                self._sweep_old_response = compute_composite_response(
                    self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
                )

            _, curr_m = self.meas_curve.getData()
            if curr_m is not None and len(curr_m) == len(self.freq_points):
                self._sweep_old_meas = np.asarray(curr_m, dtype=np.float64)
                self._sweep_old_meas_opacity = self.meas_curve.opacity()
            elif self._meas_freqs is not None and len(self._meas_freqs) > 0 and self._meas_spls is not None:
                self._sweep_old_meas = np.interp(self.freq_points, self._meas_freqs, self._meas_spls)
                self._sweep_old_meas_opacity = 1.0 if self.meas_curve.isVisible() else 0.0
            else:
                self._sweep_old_meas = np.zeros(len(self.freq_points))
                self._sweep_old_meas_opacity = 0.0

            _, curr_t = self.target_curve.getData()
            if curr_t is not None and len(curr_t) == len(self.freq_points):
                self._sweep_old_target = np.asarray(curr_t, dtype=np.float64)
                self._sweep_old_target_opacity = self.target_curve.opacity()
            elif self._target_freqs is not None and len(self._target_freqs) > 0 and self._target_spls is not None:
                self._sweep_old_target = np.interp(self.freq_points, self._target_freqs, self._target_spls)
                self._sweep_old_target_opacity = 1.0 if self.target_curve.isVisible() else 0.0
            else:
                self._sweep_old_target = np.zeros(len(self.freq_points))
                self._sweep_old_target_opacity = 0.0

            _, curr_r = self.resultant_curve.getData()
            if curr_r is not None and len(curr_r) == len(self.freq_points):
                self._sweep_old_resultant = np.asarray(curr_r, dtype=np.float64)
                self._sweep_old_resultant_opacity = self.resultant_curve.opacity()
            elif self._resultant_freqs is not None and len(self._resultant_freqs) > 0 and self._resultant_spls is not None:
                self._sweep_old_resultant = np.interp(self.freq_points, self._resultant_freqs, self._resultant_spls)
                self._sweep_old_resultant_opacity = 1.0 if self.resultant_curve.isVisible() else 0.0
            else:
                self._sweep_old_resultant = np.zeros(len(self.freq_points))
                self._sweep_old_resultant_opacity = 0.0

            for node in self._node_anim_removed:
                try:
                    self.removeItem(node)
                except Exception:
                    pass
            self._node_anim_removed.clear()
        else:
            self._sweep_old_response = compute_composite_response(
                self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
            )
            if self._meas_freqs is not None and len(self._meas_freqs) > 0 and self._meas_spls is not None:
                self._sweep_old_meas = np.interp(self.freq_points, self._meas_freqs, self._meas_spls)
                self._sweep_old_meas_opacity = 1.0 if self.meas_curve.isVisible() else 0.0
            else:
                self._sweep_old_meas = np.zeros(len(self.freq_points))
                self._sweep_old_meas_opacity = 0.0

            if self._target_freqs is not None and len(self._target_freqs) > 0 and self._target_spls is not None:
                self._sweep_old_target = np.interp(self.freq_points, self._target_freqs, self._target_spls)
                self._sweep_old_target_opacity = 1.0 if self.target_curve.isVisible() else 0.0
            else:
                self._sweep_old_target = np.zeros(len(self.freq_points))
                self._sweep_old_target_opacity = 0.0

            if self._resultant_freqs is not None and len(self._resultant_freqs) > 0 and self._resultant_spls is not None:
                self._sweep_old_resultant = np.interp(self.freq_points, self._resultant_freqs, self._resultant_spls)
                self._sweep_old_resultant_opacity = 1.0 if self.resultant_curve.isVisible() else 0.0
            else:
                self._sweep_old_resultant = np.zeros(len(self.freq_points))
                self._sweep_old_resultant_opacity = 0.0

        self.preamp_db = new_preamp

        old_map: Dict[int, EQBand] = {b.band_id: EQBand.from_dict(b.to_dict()) for b in self.bands}
        new_map: Dict[int, EQBand] = {b.band_id: EQBand.from_dict(b.to_dict()) for b in new_bands}

        common_ids = set(old_map.keys()) & set(new_map.keys())
        removed_ids = set(old_map.keys()) - set(new_map.keys())
        added_ids = set(new_map.keys()) - set(old_map.keys())

        self._node_anim_common.clear()
        self._node_anim_removed.clear()
        self._node_anim_added.clear()

        # 1. Common points: glide smoothly from old to new positions
        for bid in common_ids:
            if bid in self.node_items:
                node = self.node_items[bid]
                x0 = math.log10(max(MIN_FREQ, min(MAX_FREQ, old_map[bid].freq)))
                y0 = max(MIN_GAIN, min(MAX_GAIN, old_map[bid].gain))
                x1 = math.log10(max(MIN_FREQ, min(MAX_FREQ, new_map[bid].freq)))
                y1 = max(MIN_GAIN, min(MAX_GAIN, new_map[bid].gain))
                node.setOpacity(1.0)
                node.is_bypassed_globally = self.is_bypassed
                self._node_anim_common[bid] = (node, x0, y0, x1, y1, new_map[bid])

        # 2. Removed points: stay at old pos and smoothly fade out
        for bid in removed_ids:
            if bid in self.node_items:
                node = self.node_items.pop(bid)
                node.setOpacity(1.0)
                self._node_anim_removed.append(node)

        # 3. Added points: materialize at new pos and smoothly fade in
        for bid in added_ids:
            target_b = new_map[bid]
            node = EQNodeItem(target_b, self.custom_vb)
            node.is_bypassed_globally = self.is_bypassed
            node.sigMoved.connect(self._on_node_moved)
            node.sigQChanged.connect(self._on_node_q_changed)
            node.sigTypeChanged.connect(self._on_node_type_changed)
            node.sigBypassToggled.connect(self._on_node_bypass_toggled)
            node.sigDeleteRequested.connect(self.sigBandDeleted.emit)
            node.sigHoverChanged.connect(self._on_node_hover)
            node.sigDragStarted.connect(self.sigBandDragStarted.emit)
            node.sigWheelScrollStarted.connect(self.sigWheelScrollStarted.emit)

            node.setOpacity(0.0)
            self.addItem(node)
            node.updateTransform()
            self.node_items[bid] = node
            self._node_anim_added.append(node)

        self.bands = [EQBand.from_dict(b.to_dict()) for b in new_bands]
        self._sweep_target_response = compute_composite_response(
            self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
        )

        # Measurement sweep setup
        if meas_data is not None:
            mf, ms, mlbl = meas_data
            if mf is not None and len(mf) > 0 and ms is not None and len(ms) > 0:
                self._sweep_target_meas = np.interp(self.freq_points, mf, ms)
                self._sweep_target_meas_opacity = 1.0 if getattr(self, "_is_meas_checked", True) else 0.0
                self._pending_final_meas = (mf, ms, mlbl)
            else:
                self._sweep_target_meas = np.copy(self._sweep_old_meas)
                self._sweep_target_meas_opacity = 0.0
                self._pending_final_meas = None
            self._sweep_meas_animating = True
        elif self.meas_curve.isVisible():
            self._sweep_target_meas = np.copy(self._sweep_old_meas)
            self._sweep_target_meas_opacity = 0.0
            self._pending_final_meas = None
            self._sweep_meas_animating = True
        else:
            self._sweep_meas_animating = False

        # Target sweep setup
        if target_data is not None:
            tf, ts, tlbl = target_data
            if tf is not None and len(tf) > 0 and ts is not None and len(ts) > 0:
                self._sweep_target_target = np.interp(self.freq_points, tf, ts)
                self._sweep_target_target_opacity = 1.0 if getattr(self, "_is_target_checked", True) else 0.0
                self._pending_final_target = (tf, ts, tlbl)
            else:
                self._sweep_target_target = np.copy(self._sweep_old_target)
                self._sweep_target_target_opacity = 0.0
                self._pending_final_target = None
            self._sweep_target_animating = True
        elif self.target_curve.isVisible():
            self._sweep_target_target = np.copy(self._sweep_old_target)
            self._sweep_target_target_opacity = 0.0
            self._pending_final_target = None
            self._sweep_target_animating = True
        else:
            self._sweep_target_animating = False

        # Resultant sweep setup
        if resultant_data is not None:
            rf, rs, rlbl = resultant_data
            if rf is not None and len(rf) > 0 and rs is not None and len(rs) > 0:
                self._sweep_target_resultant = np.interp(self.freq_points, rf, rs)
                self._sweep_target_resultant_opacity = 1.0 if getattr(self, "_is_resultant_checked", True) else 0.0
                self._pending_final_resultant = (rf, rs, rlbl)
            else:
                self._sweep_target_resultant = np.copy(self._sweep_old_resultant)
                self._sweep_target_resultant_opacity = 0.0
                self._pending_final_resultant = None
            self._sweep_resultant_animating = True
        elif self.resultant_curve.isVisible():
            self._sweep_target_resultant = np.copy(self._sweep_old_resultant)
            self._sweep_target_resultant_opacity = 0.0
            self._pending_final_resultant = None
            self._sweep_resultant_animating = True
        else:
            self._sweep_resultant_animating = False

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
        x_norm = np.linspace(0.0, 1.0, len(self.freq_points))
        u = np.clip((center - x_norm) / (2.0 * sigma) + 0.5, 0.0, 1.0)
        alpha = 3.0 * (u ** 2) - 2.0 * (u ** 3)
        blended = (1.0 - alpha) * self._sweep_old_response + alpha * self._sweep_target_response

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
        for bid, (node, x0, y0, x1, y1, target_band) in self._node_anim_common.items():
            curr_x = (1.0 - p) * x0 + p * x1
            curr_y = (1.0 - p) * y0 + p * y1
            node.setPos(curr_x, curr_y)
            node.band.freq = 10.0 ** curr_x
            node.band.gain = curr_y
            node.update()

        # 3. Smoothly fade out removed nodes
        fade_out = max(0.0, 1.0 - p)
        for node in self._node_anim_removed:
            node.setOpacity(fade_out)

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
        for bid, (node, x0, y0, x1, y1, target_band) in self._node_anim_common.items():
            node.band = EQBand.from_dict(target_band.to_dict())
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

        # Finalize overlay traces
        if self._sweep_meas_animating:
            self._sweep_meas_animating = False
            self.meas_curve.setOpacity(1.0)
            if self._pending_final_meas:
                mf, ms, mlbl = self._pending_final_meas
                self.set_measurement_overlay(mf, ms, label=mlbl)
            else:
                self.set_measurement_overlay(None, None)

        if self._sweep_target_animating:
            self._sweep_target_animating = False
            self.target_curve.setOpacity(1.0)
            if self._pending_final_target:
                tf, ts, tlbl = self._pending_final_target
                self.set_target_overlay(tf, ts, label=tlbl)
            else:
                self.set_target_overlay(None, None)

        if self._sweep_resultant_animating:
            self._sweep_resultant_animating = False
            self.resultant_curve.setOpacity(1.0)
            if self._pending_final_resultant:
                rf, rs, rlbl = self._pending_final_resultant
                self.set_resultant_overlay(rf, rs, label=rlbl)
            else:
                self.set_resultant_overlay(None, None)

        self.recompute_composite_curve()

    def animate_overlay_curve(
        self,
        curve_name: str,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
        duration: int = 400,
    ) -> None:
        """Smoothly transitions a single overlay curve (target, meas, or resultant) with InOutCubic easing."""
        if self._is_sweeping:
            if curve_name == "target":
                self.set_target_overlay(freqs, spls, label)
            elif curve_name == "meas":
                self.set_measurement_overlay(freqs, spls, label)
            elif curve_name == "resultant":
                self.set_resultant_overlay(freqs, spls, label)
            return

        curve_item = getattr(self, f"{curve_name}_curve", None)
        if not curve_item:
            return

        curr_freqs = getattr(self, f"_{curve_name}_freqs", None)
        curr_spls = getattr(self, f"_{curve_name}_spls", None)

        if curr_freqs is not None and len(curr_freqs) > 0 and curr_spls is not None:
            old_sampled = np.interp(self.freq_points, curr_freqs, curr_spls)
            old_op = 1.0 if curve_item.isVisible() else 0.0
        else:
            old_sampled = np.zeros(len(self.freq_points))
            old_op = 0.0

        is_checked = getattr(self, f"_is_{curve_name}_checked", True)
        if freqs is not None and len(freqs) > 0 and spls is not None and len(spls) > 0:
            f_arr = np.asarray(freqs, dtype=np.float64)
            s_arr = np.asarray(spls, dtype=np.float64)
            valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
            if np.any(valid):
                target_sampled = np.interp(self.freq_points, f_arr[valid], s_arr[valid])
                target_op = 1.0 if is_checked else 0.0
                final_data = (f_arr[valid], s_arr[valid], label)
                setattr(self, f"_{curve_name}_freqs", f_arr[valid])
                setattr(self, f"_{curve_name}_spls", s_arr[valid])
                setattr(self, f"_{curve_name}_label", label)
            else:
                target_sampled = np.copy(old_sampled)
                target_op = 0.0
                final_data = None
                setattr(self, f"_{curve_name}_freqs", None)
                setattr(self, f"_{curve_name}_spls", None)
                setattr(self, f"_{curve_name}_label", "")
        else:
            target_sampled = np.copy(old_sampled)
            target_op = 0.0
            final_data = None
            setattr(self, f"_{curve_name}_freqs", None)
            setattr(self, f"_{curve_name}_spls", None)
            setattr(self, f"_{curve_name}_label", "")

        self._anim_overlay_types[curve_name] = (old_sampled, target_sampled, old_op, target_op, final_data)
        curve_item.setVisible(is_checked and (target_op > 0.0 or old_op > 0.0))
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
        x_norm = np.linspace(0.0, 1.0, len(self.freq_points))
        u = np.clip((center - x_norm) / (2.0 * sigma) + 0.5, 0.0, 1.0)
        alpha = 3.0 * (u ** 2) - 2.0 * (u ** 3)

        for cname, (old_s, target_s, old_op, target_op, _) in self._anim_overlay_types.items():
            curve_item = getattr(self, f"{cname}_curve", None)
            if curve_item:
                blended = (1.0 - alpha) * old_s + alpha * target_s
                curve_item.setData(self.x_log_points, blended)
                is_chk = getattr(self, f"_is_{cname}_checked", True)
                cur_target_op = 1.0 if is_chk else 0.0
                op = (1.0 - p) * old_op + p * cur_target_op
                curve_item.setOpacity(op)
                curve_item.setVisible(is_chk and (op > 0.01))

    def _on_overlay_anim_finished(self) -> None:
        for cname, (_, _, _, _, final_data) in list(self._anim_overlay_types.items()):
            curve_item = getattr(self, f"{cname}_curve", None)
            if curve_item:
                curve_item.setOpacity(1.0)
                setter = getattr(self, f"set_{cname}_overlay", None)
                if setter:
                    if final_data:
                        f, s, lbl = final_data
                        setter(f, s, label=lbl)
                    else:
                        setter(None, None)
        self._anim_overlay_types.clear()
        self._update_overlay_legend()

    # -------------------------------------------------------------------------
    # Band Synchronization
    # -------------------------------------------------------------------------
    def set_bands(self, bands: List[EQBand], preamp_db: float = 0.0) -> None:
        self.preamp_db = preamp_db
        self.bands = [EQBand.from_dict(b.to_dict()) for b in bands]

        for item in list(self.node_items.values()):
            self.removeItem(item)
        self.node_items.clear()

        for band in self.bands:
            node = EQNodeItem(band, self.custom_vb)
            node.is_bypassed_globally = self.is_bypassed
            node.sigMoved.connect(self._on_node_moved)
            node.sigQChanged.connect(self._on_node_q_changed)
            node.sigTypeChanged.connect(self._on_node_type_changed)
            node.sigBypassToggled.connect(self._on_node_bypass_toggled)
            node.sigDeleteRequested.connect(self.sigBandDeleted.emit)
            node.sigHoverChanged.connect(self._on_node_hover)
            node.sigDragStarted.connect(self.sigBandDragStarted.emit)
            node.sigWheelScrollStarted.connect(self.sigWheelScrollStarted.emit)

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
        for band in self.bands:
            if band.band_id == band_id:
                if freq is not None:
                    band.freq = float(freq)
                if gain is not None:
                    band.gain = float(gain)
                if q is not None:
                    band.q = float(q)
                if filter_type is not None:
                    band.filter_type = str(filter_type)
                if enabled is not None:
                    band.enabled = bool(enabled)

                if band_id in self.node_items:
                    node = self.node_items[band_id]
                    node.band = EQBand.from_dict(band.to_dict())
                    node._update_node_position()
                    node.update()
                break

        self.recompute_composite_curve()

    def set_preamp(self, preamp_db: float) -> None:
        self.preamp_db = float(preamp_db)
        if not self._is_sweeping:
            self.recompute_composite_curve()

    def recompute_composite_curve(self) -> None:
        if self.is_bypassed and self._eq_active_factor == 0.0:
            flat = np.zeros(len(self.freq_points))
            self.eq_curve.setData(self.x_log_points, flat)
            self.eq_fill_curve.setData(self.x_log_points, flat)
            return

        response_db = compute_composite_response(
            self.bands, self.freq_points, preamp_db=self.preamp_db, fs=self.sample_rate
        )
        if self._eq_active_factor < 1.0:
            response_db = self._eq_active_factor * response_db + (1.0 - self._eq_active_factor) * 0.0

        self.eq_curve.setData(self.x_log_points, response_db)
        if not self.is_bypassed:
            self.eq_fill_curve.setData(self.x_log_points, response_db)

    # -------------------------------------------------------------------------
    # HUD & Hover Callbacks
    # -------------------------------------------------------------------------
    def _on_node_hover(self, node: EQNodeItem, is_hovered: bool) -> None:
        if is_hovered:
            b = node.band
            f_str = f"{b.freq:.0f} Hz" if b.freq >= 100 else f"{b.freq:.1f} Hz"
            hud_str = f"BAND {b.band_id}   |   {b.filter_type.upper()}   |   {f_str}   |   {b.gain:+.1f} dB   |   Q {b.q:.2f}"
            self.hud_text_item.setText(hud_str)
            self.hud_text_item.setVisible(True)
        else:
            if not any(n.is_dragging for n in self.node_items.values()):
                self.hud_text_item.setVisible(False)

    # -------------------------------------------------------------------------
    # Node Callbacks
    # -------------------------------------------------------------------------
    def _on_node_moved(self, band_id: int, freq: float, gain: float) -> None:
        for band in self.bands:
            if band.band_id == band_id:
                band.freq = freq
                band.gain = gain
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_q_changed(self, band_id: int, q: float) -> None:
        for band in self.bands:
            if band.band_id == band_id:
                band.q = q
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_type_changed(self, band_id: int, ftype: str) -> None:
        for band in self.bands:
            if band.band_id == band_id:
                band.filter_type = ftype
                if ftype in (FilterType.HIGH_PASS.value, FilterType.LOW_PASS.value):
                    band.gain = 0.0
                if band_id in self.node_items:
                    self.node_items[band_id].band = EQBand.from_dict(band.to_dict())
                    self.node_items[band_id]._update_node_position()
                    self.node_items[band_id].update()
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    def _on_node_bypass_toggled(self, band_id: int, enabled: bool) -> None:
        for band in self.bands:
            if band.band_id == band_id:
                band.enabled = enabled
                if band_id in self.node_items:
                    self.node_items[band_id].band.enabled = enabled
                    self.node_items[band_id].update()
                self.recompute_composite_curve()
                self.sigBandModified.emit(band_id, band.freq, band.gain, band.q, band.filter_type, band.enabled)
                break

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_spectrum_curve(self, freqs: np.ndarray, mags_db: np.ndarray) -> None:
        self.spectrum_curve.setData(self.x_log_points, mags_db)

    # -------------------------------------------------------------------------
    # AutoEQ Curve Overlays (Measurement & Target Reference Traces)
    # -------------------------------------------------------------------------
    def set_measurement_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        if self._is_sweeping:
            self._meas_label = label
            if freqs is None or spls is None or len(freqs) == 0:
                self._pending_final_meas = None
                self._sweep_target_meas_opacity = 0.0
            else:
                f_arr = np.asarray(freqs, dtype=np.float64)
                s_arr = np.asarray(spls, dtype=np.float64)
                valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
                if np.any(valid):
                    self._meas_freqs = f_arr[valid]
                    self._meas_spls = s_arr[valid]
                    self._sweep_target_meas = np.interp(self.freq_points, self._meas_freqs, self._meas_spls)
                    self._sweep_target_meas_opacity = 1.0 if getattr(self, "_is_meas_checked", True) else 0.0
                    self._pending_final_meas = (freqs, spls, label)
            self._update_overlay_legend()
            return

        if freqs is None or spls is None or len(freqs) == 0:
            self.meas_curve.setData([], [])
            self.meas_curve.setVisible(False)
            self._meas_freqs = None
            self._meas_spls = None
            self._meas_label = ""
            self._update_overlay_legend()
            return

        f_arr = np.asarray(freqs, dtype=np.float64)
        s_arr = np.asarray(spls, dtype=np.float64)
        valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
        if not np.any(valid):
            return

        self._meas_freqs = f_arr[valid]
        self._meas_spls = s_arr[valid]
        self._meas_label = label

        x_log = np.log10(self._meas_freqs)
        self.meas_curve.setData(x_log, self._meas_spls)
        self.meas_curve.setVisible(getattr(self, "_is_meas_checked", True))
        self._update_overlay_legend()

    def set_target_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        if self._is_sweeping:
            self._target_label = label
            if freqs is None or spls is None or len(freqs) == 0:
                self._pending_final_target = None
                self._sweep_target_target_opacity = 0.0
            else:
                f_arr = np.asarray(freqs, dtype=np.float64)
                s_arr = np.asarray(spls, dtype=np.float64)
                valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
                if np.any(valid):
                    self._target_freqs = f_arr[valid]
                    self._target_spls = s_arr[valid]
                    self._sweep_target_target = np.interp(self.freq_points, self._target_freqs, self._target_spls)
                    self._sweep_target_target_opacity = 1.0 if getattr(self, "_is_target_checked", True) else 0.0
                    self._pending_final_target = (freqs, spls, label)
            self._update_overlay_legend()
            return

        if freqs is None or spls is None or len(freqs) == 0:
            self.target_curve.setData([], [])
            self.target_curve.setVisible(False)
            self._target_freqs = None
            self._target_spls = None
            self._target_label = ""
            self._update_overlay_legend()
            return

        f_arr = np.asarray(freqs, dtype=np.float64)
        s_arr = np.asarray(spls, dtype=np.float64)
        valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
        if not np.any(valid):
            return

        self._target_freqs = f_arr[valid]
        self._target_spls = s_arr[valid]
        self._target_label = label

        x_log = np.log10(self._target_freqs)
        self.target_curve.setData(x_log, self._target_spls)
        self.target_curve.setVisible(getattr(self, "_is_target_checked", True))
        self._update_overlay_legend()

    def set_target2_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        if freqs is None or spls is None or len(freqs) == 0:
            self.target2_curve.setData([], [])
            self.target2_curve.setVisible(False)
            self._target2_freqs = None
            self._target2_spls = None
            self._target2_label = ""
            self._update_overlay_legend()
            return

        f_arr = np.asarray(freqs, dtype=np.float64)
        s_arr = np.asarray(spls, dtype=np.float64)
        valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
        if not np.any(valid):
            return

        self._target2_freqs = f_arr[valid]
        self._target2_spls = s_arr[valid]
        self._target2_label = label

        x_log = np.log10(self._target2_freqs)
        self.target2_curve.setData(x_log, self._target2_spls)
        self.target2_curve.setVisible(getattr(self, "_is_target2_checked", True))
        self._update_overlay_legend()

    def set_resultant_overlay(
        self,
        freqs: Optional[Sequence[float]],
        spls: Optional[Sequence[float]],
        label: str = "",
    ) -> None:
        """Sets the resultant predicted frequency response overlay (Model + EQ in cyan)."""
        if self._is_sweeping:
            self._resultant_label = label
            if freqs is None or spls is None or len(freqs) == 0:
                self._pending_final_resultant = None
                self._sweep_target_resultant_opacity = 0.0
            else:
                f_arr = np.asarray(freqs, dtype=np.float64)
                s_arr = np.asarray(spls, dtype=np.float64)
                valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
                if np.any(valid):
                    self._resultant_freqs = f_arr[valid]
                    self._resultant_spls = s_arr[valid]
                    self._sweep_target_resultant = np.interp(self.freq_points, self._resultant_freqs, self._resultant_spls)
                    self._sweep_target_resultant_opacity = 1.0 if getattr(self, "_is_resultant_checked", True) else 0.0
                    self._pending_final_resultant = (freqs, spls, label)
            self._update_overlay_legend()
            return

        if freqs is None or spls is None or len(freqs) == 0:
            self.resultant_curve.setData([], [])
            self.resultant_curve.setVisible(False)
            self._resultant_freqs = None
            self._resultant_spls = None
            self._resultant_label = ""
            self._update_overlay_legend()
            return

        f_arr = np.asarray(freqs, dtype=np.float64)
        s_arr = np.asarray(spls, dtype=np.float64)
        valid = (f_arr >= MIN_FREQ) & (f_arr <= MAX_FREQ)
        if not np.any(valid):
            return

        self._resultant_freqs = f_arr[valid]
        self._resultant_spls = s_arr[valid]
        self._resultant_label = label

        x_log = np.log10(self._resultant_freqs)
        self.resultant_curve.setData(x_log, self._resultant_spls)
        self.resultant_curve.setVisible(getattr(self, "_is_resultant_checked", True))
        self._update_overlay_legend()

    def clear_overlays(self) -> None:
        if self._is_sweeping:
            self._pending_final_meas = None
            self._pending_final_target = None
            self._pending_final_resultant = None
            self._sweep_target_meas_opacity = 0.0
            self._sweep_target_target_opacity = 0.0
            self._sweep_target_resultant_opacity = 0.0
            self._meas_label = ""
            self._target_label = ""
            self._target2_label = ""
            self._resultant_label = ""
            self._update_overlay_legend()
            return

        self.meas_curve.setData([], [])
        self.target_curve.setData([], [])
        self.target2_curve.setData([], [])
        self.resultant_curve.setData([], [])
        self.meas_curve.setVisible(False)
        self.target_curve.setVisible(False)
        self.target2_curve.setVisible(False)
        self.resultant_curve.setVisible(False)
        self._meas_freqs = None
        self._meas_spls = None
        self._meas_label = ""
        self._target_freqs = None
        self._target_spls = None
        self._target_label = ""
        self._target2_freqs = None
        self._target2_spls = None
        self._target2_label = ""
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
        self._is_meas_checked = show_meas
        self._is_target_checked = show_target
        self._is_resultant_checked = show_resultant
        self._is_target2_checked = show_target2
        if self._is_sweeping:
            self._sweep_target_meas_opacity = 1.0 if show_meas else 0.0
            self._sweep_target_target_opacity = 1.0 if show_target else 0.0
            self._sweep_target_resultant_opacity = 1.0 if show_resultant else 0.0
        else:
            self.meas_curve.setVisible(show_meas and (self._meas_freqs is not None))
            self.target_curve.setVisible(show_target and (self._target_freqs is not None))
            self.target2_curve.setVisible(show_target2 and (self._target2_freqs is not None))
            self.resultant_curve.setVisible(show_resultant and (self._resultant_freqs is not None))
        self._update_overlay_legend()

    def _update_overlay_legend(self) -> None:
        items = []
        if self._target_label:
            vis = self.target_curve.isVisible()
            color = "#facc15" if vis else "#71717a"
            decor = "text-decoration: none;" if vis else "text-decoration: line-through; opacity: 0.55;"
            txt = f"— Target: {self._target_label}"
            items.append(("target", txt, f"<span style='color: {color}; font-weight: 600; {decor}'>{txt}</span>"))

        if self._target2_label:
            vis2 = self.target2_curve.isVisible()
            color2 = "#c084fc" if vis2 else "#71717a"
            decor2 = "text-decoration: none;" if vis2 else "text-decoration: line-through; opacity: 0.55;"
            txt2 = f"— Target 2: {self._target2_label}"
            items.append(("target2", txt2, f"<span style='color: {color2}; font-weight: 600; {decor2}'>{txt2}</span>"))

        if self._meas_label:
            vis = self.meas_curve.isVisible()
            color = "#d4d4d8" if vis else "#71717a"
            decor = "text-decoration: none;" if vis else "text-decoration: line-through; opacity: 0.55;"
            txt = f"┄ Measurement: {self._meas_label}"
            items.append(("meas", txt, f"<span style='color: {color}; font-weight: 600; {decor}'>{txt}</span>"))

        if self._resultant_label:
            vis = self.resultant_curve.isVisible()
            color = "#38bdf8" if vis else "#71717a"
            decor = "text-decoration: none;" if vis else "text-decoration: line-through; opacity: 0.55;"
            txt = f"— Resultant: {self._resultant_label}"
            items.append(("resultant", txt, f"<span style='color: {color}; font-weight: 600; {decor}'>{txt}</span>"))

        if items:
            sep = " &nbsp;&nbsp;|&nbsp;&nbsp; "
            total_chars = sum(len(it[1]) for it in items) + 3 * (len(items) - 1)
            regions = []
            parts = []
            char_accum = 0.0

            for ctype, raw_txt, html_str in items:
                start_r = char_accum / total_chars
                char_accum += len(raw_txt)
                end_r = char_accum / total_chars
                regions.append((start_r, end_r, ctype))
                char_accum += 3  # separator spacing
                parts.append(html_str)

            self.overlay_legend_item.regions = regions
            self.overlay_legend_item.setHtml(sep.join(parts))
            self.overlay_legend_item.setVisible(True)
        else:
            self.overlay_legend_item.regions = []
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
        self.spectrum_curve.setVisible(visible)
        self.sigSpectrumToggled.emit(visible)

    def reset_default_view(self) -> None:
        self.setXRange(X_MIN, X_MAX, padding=0.0)
        self.setYRange(MIN_GAIN, MAX_GAIN, padding=0.0)
        self.grid_item.set_bounds(X_MIN, X_MAX, MIN_GAIN, MAX_GAIN)

    def export_plot_image(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export EQ Plot Image",
            "clean_apo_eq_curve.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )
        if path:
            pix = self.grab()
            pix.save(path)

    def show_canvas_context_menu(self, screen_pos: QtCore.QPoint, freq: float, gain: float) -> None:
        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet(
            f"QMenu {{ background-color: #000000; color: #ffffff; border: 1px solid #3f3f46; border-radius: 0px; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 22px 6px 28px; border-radius: 0px; }}"
            f"QMenu::item:selected {{ background-color: #ffffff; color: #000000; font-weight: bold; }}"
            f"QMenu::separator {{ height: 1px; background: #27272a; margin: 4px 6px; }}"
            f"QMenu::indicator {{ width: 14px; height: 14px; left: 8px; }}"
            f"QMenu::indicator:unchecked {{ image: none; }}"
            f"QMenu::indicator:checked {{ image: url('{CHECK_WHITE_PATH}'); }}"
            f"QMenu::indicator:checked:selected {{ image: url('{CHECK_BLACK_PATH}'); }}"
        )

        f_str = f"{freq:.0f} Hz" if freq >= 100 else f"{freq:.1f} Hz"
        add_act = menu.addAction(f"Add Filter Band at {f_str} ({gain:+.1f} dB)")
        add_act.triggered.connect(lambda: self.sigBandAdded.emit(freq, gain))

        menu.addSeparator()

        plot_opts_menu = menu.addMenu("Plot Options")
        plot_opts_menu.setStyleSheet(menu.styleSheet())

        x_maj_act = plot_opts_menu.addAction("Show Frequency (X) Grid")
        x_maj_act.setCheckable(True)
        x_maj_act.setChecked(self.grid_item.show_x_major)
        x_maj_act.triggered.connect(self._toggle_x_major_grid)

        x_min_act = plot_opts_menu.addAction("Show Intermediate Grid (3k, 4k, etc.)")
        x_min_act.setCheckable(True)
        x_min_act.setChecked(self.grid_item.show_x_minor)
        x_min_act.triggered.connect(self._toggle_x_minor_grid)

        y_act = plot_opts_menu.addAction("Show Decibel (Y) Grid")
        y_act.setCheckable(True)
        y_act.setChecked(self.grid_item.show_y)
        y_act.triggered.connect(self._toggle_y_grid)

        plot_opts_menu.addSeparator()

        spec_act = plot_opts_menu.addAction("Show Live Spectrum")
        spec_act.setCheckable(True)
        spec_act.setChecked(self.spectrum_curve.isVisible())
        spec_act.triggered.connect(self._toggle_spectrum_overlay)

        if self._meas_freqs is not None:
            meas_act = plot_opts_menu.addAction("Show IEM Measurement Curve")
            meas_act.setCheckable(True)
            meas_act.setChecked(self.meas_curve.isVisible())
            meas_act.triggered.connect(lambda c: self.set_overlays_visible(c, self.target_curve.isVisible()))

        if self._target_freqs is not None:
            targ_act = plot_opts_menu.addAction("Show Target Curve")
            targ_act.setCheckable(True)
            targ_act.setChecked(self.target_curve.isVisible())
            targ_act.triggered.connect(lambda c: self.set_overlays_visible(self.meas_curve.isVisible(), c))

        reset_act = plot_opts_menu.addAction("Reset View (20 Hz - 20 kHz, -20 to +20 dB)")
        reset_act.triggered.connect(self.reset_default_view)

        menu.addSeparator()

        export_act = menu.addAction("Export Image...")
        export_act.triggered.connect(self.export_plot_image)

        menu.exec(screen_pos)
