"""
Generates high-resolution Clean APO application icon (.png and .ico)
using PyQt6 QPainter and standard Windows ICO container format.
"""
import os
import struct
from PyQt6 import QtCore, QtGui, QtWidgets

def create_icon_image(size: int) -> QtGui.QImage:
    img = QtGui.QImage(size, size, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    
    painter = QtGui.QPainter(img)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
    
    # 1. Dark squircle background
    margin = size * 0.04
    rect = QtCore.QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
    radius = size * 0.22
    
    # Subtle gradient background
    bg_grad = QtGui.QLinearGradient(0, 0, 0, size)
    bg_grad.setColorAt(0.0, QtGui.QColor("#18181b"))
    bg_grad.setColorAt(1.0, QtGui.QColor("#09090b"))
    painter.setBrush(QtGui.QBrush(bg_grad))
    painter.setPen(QtGui.QPen(QtGui.QColor("#27272a"), max(1.0, size * 0.025)))
    painter.drawRoundedRect(rect, radius, radius)
    
    # 2. Subtle grid lines inside icon
    grid_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 18), max(1.0, size * 0.01))
    painter.setPen(grid_pen)
    
    # Horizontal center line (0 dB)
    cy = size * 0.52
    painter.drawLine(QtCore.QPointF(rect.left() + size * 0.08, cy), QtCore.QPointF(rect.right() - size * 0.08, cy))
    
    # Vertical grid markers
    for fx in [0.28, 0.5, 0.72]:
        x = size * fx
        painter.drawLine(QtCore.QPointF(x, rect.top() + size * 0.12), QtCore.QPointF(x, rect.bottom() - size * 0.12))
        
    # 3. Dynamic EQ Response Curve
    curve_path = QtGui.QPainterPath()
    p0 = (size * 0.15, size * 0.62)
    p1 = (size * 0.32, size * 0.38) # Bass shelf / peak
    p2 = (size * 0.50, size * 0.60) # Mid dip
    p3 = (size * 0.70, size * 0.32) # Upper-mid / treble boost
    p4 = (size * 0.85, size * 0.52) # Air rolloff
    
    curve_path.moveTo(p0[0], p0[1])
    curve_path.cubicTo(size * 0.22, size * 0.62, size * 0.25, size * 0.38, p1[0], p1[1])
    curve_path.cubicTo(size * 0.38, size * 0.38, size * 0.42, size * 0.60, p2[0], p2[1])
    curve_path.cubicTo(size * 0.58, size * 0.60, size * 0.62, size * 0.32, p3[0], p3[1])
    curve_path.cubicTo(size * 0.76, size * 0.32, size * 0.80, size * 0.52, p4[0], p4[1])
    
    # Translucent glow under curve
    glow_path = QtGui.QPainterPath(curve_path)
    glow_path.lineTo(p4[0], cy)
    glow_path.lineTo(p0[0], cy)
    glow_path.closeSubpath()
    
    fill_grad = QtGui.QLinearGradient(0, size * 0.3, 0, cy)
    fill_grad.setColorAt(0.0, QtGui.QColor(255, 255, 255, 40))
    fill_grad.setColorAt(1.0, QtGui.QColor(255, 255, 255, 5))
    painter.setBrush(QtGui.QBrush(fill_grad))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.drawPath(glow_path)
    
    # White EQ line
    line_pen = QtGui.QPen(QtGui.QColor("#ffffff"), max(2.0, size * 0.04))
    line_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    line_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(line_pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.drawPath(curve_path)
    
    # 4. Filter Node points
    node_pen = QtGui.QPen(QtGui.QColor("#000000"), max(1.0, size * 0.02))
    painter.setPen(node_pen)
    painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
    node_r = max(2.5, size * 0.04)
    for pt in [p1, p3]:
        painter.drawEllipse(QtCore.QPointF(pt[0], pt[1]), node_r, node_r)
        
    painter.end()
    return img


def save_ico(images: list[QtGui.QImage], ico_path: str):
    """Encodes images into standard multi-resolution Windows .ico file."""
    # Convert images to PNG byte arrays
    png_buffers = []
    for img in images:
        ba = QtCore.QByteArray()
        buf = QtCore.QBuffer(ba)
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        img.save(buf, "PNG")
        png_buffers.append(bytes(ba))
        
    num_images = len(images)
    header = struct.pack("<HHH", 0, 1, num_images) # Reserved (0), Type (1=ICO), Count
    
    entries = []
    offset = 6 + 16 * num_images
    for i, img in enumerate(images):
        w = img.width() if img.width() < 256 else 0
        h = img.height() if img.height() < 256 else 0
        size_bytes = len(png_buffers[i])
        entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, size_bytes, offset)
        entries.append(entry)
        offset += size_bytes
        
    with open(ico_path, "wb") as f:
        f.write(header)
        for entry in entries:
            f.write(entry)
        for buf in png_buffers:
            f.write(buf)


def main():
    app = QtWidgets.QApplication([])
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    sizes = [256, 128, 64, 48, 32, 16]
    images = [create_icon_image(s) for s in sizes]
    
    # Save primary 256x256 PNG
    png_path = os.path.join(assets_dir, "clean_apo.png")
    images[0].save(png_path, "PNG")
    print(f"Saved PNG icon to {png_path}")
    
    # Save multi-size Windows ICO
    ico_path = os.path.join(assets_dir, "clean_apo.ico")
    save_ico(images, ico_path)
    print(f"Saved ICO icon to {ico_path}")

if __name__ == "__main__":
    main()
