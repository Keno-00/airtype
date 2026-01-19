"""
Keyboard widget with key grid and swipe path visualization.
"""
from typing import List, Optional, Tuple, Callable, Dict
from PyQt5.QtWidgets import (
    QWidget, QGridLayout, QLabel, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QPoint, QRectF, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPainterPath, QBrush

from .layouts import get_layout, get_key_positions


class KeyButton(QLabel):
    """Individual key button."""
    
    def __init__(self, key_data, parent=None):
        label = key_data[0] if isinstance(key_data, tuple) else key_data
        super().__init__(label, parent)
        self.key = label
        self.is_special = key_data[2] if isinstance(key_data, tuple) else False
        self._highlighted = False
        self.setAlignment(Qt.AlignCenter)
        self.setObjectName("KeyButton")
        if self.is_special:
            self.setProperty("special", "true")
        self._update_style()
    
    def set_highlighted(self, highlighted: bool):
        """Set key highlight state."""
        if self._highlighted != highlighted:
            self._highlighted = highlighted
            self._update_style()
    
    def _update_style(self):
        """Update visual style based on state."""
        if self._highlighted:
            self.setProperty("highlighted", "true")
        else:
            self.setProperty("highlighted", "false")
        # Force style refresh
        self.style().unpolish(self)
        self.style().polish(self)


class SwipeCanvas(QWidget):
    """Transparent overlay that draws swipe path and cursor."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._path_points: List[Tuple[float, float]] = []
        self._is_swiping = False
        self._cursor_pos: Optional[Tuple[float, float]] = None
        self._path_fade_timer = QTimer(self)
        self._path_fade_timer.setSingleShot(True)
        self._path_fade_timer.timeout.connect(self._fade_path)
        self._show_path = True
    
    def set_cursor(self, x: float, y: float):
        """Set cursor position (always visible, shows hand position)."""
        self._cursor_pos = (x, y)
        self.update()
    
    def start_swipe(self):
        """Start a new swipe path."""
        self._path_points.clear()
        self._is_swiping = True
        self._show_path = True
        self._path_fade_timer.stop()
        self.update()
    
    def add_point(self, x: float, y: float):
        """Add point to swipe path with distance-based filtering."""
        if self._is_swiping:
            # Distance filter: only add if moved at least 0.5% of screen width/height
            # This prevents hand tremors from creating massive path objects
            if not self._path_points:
                self._path_points.append((x, y))
            else:
                lx, ly = self._path_points[-1]
                dist = ((x - lx)**2 + (y - ly)**2)**0.5
                if dist > 0.005:
                    self._path_points.append((x, y))
                    # Cap path size to prevent UI lag on extremely long/stale swipes
                    if len(self._path_points) > 500:
                        self._path_points.pop(0)
                    self.update()
    
    def end_swipe(self):
        """End current swipe - path fades after delay."""
        self._is_swiping = False
        # Keep path visible for 1 second before clearing
        self._path_fade_timer.start(1000)  # 1 second delay
        self.update()
    
    def _fade_path(self):
        """Clear path after fade delay."""
        self._path_points.clear()
        self._show_path = False
        self.update()
    
    def clear(self):
        """Clear the swipe path."""
        self._path_points.clear()
        self._is_swiping = False
        self._show_path = False
        self.update()
    
    def paintEvent(self, event):
        """Draw the swipe path and cursor."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Draw swipe path
        if self._path_points and self._show_path:
            pen = QPen(QColor(100, 200, 255, 200))
            pen.setWidth(4)
            painter.setPen(pen)
            
            if len(self._path_points) >= 2:
                # Optimized drawing using drawPolyline instead of QPainterPath
                points = [QPoint(int(px * w), int(py * h)) for px, py in self._path_points]
                painter.drawPolyline(*points)
            
            # Draw path endpoint highlight
            if self._path_points:
                x, y = self._path_points[-1]
                painter.setBrush(QColor(100, 200, 255, 120))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(QPoint(int(x * w), int(y * h)), 10, 10)
        
        # Draw cursor (always visible when hand is detected)
        if self._cursor_pos:
            cx, cy = self._cursor_pos
            px, py = int(cx * w), int(cy * h)
            
            if self._is_swiping:
                # Active swipe - solid cursor
                painter.setBrush(QColor(255, 200, 100, 220))
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                # Not swiping - hollow ring cursor
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(255, 200, 100, 200), 3))
            
            painter.drawEllipse(QPoint(px, py), 15, 15)
            
            # Draw crosshair
            painter.setPen(QPen(QColor(255, 200, 100, 150), 1))
            painter.drawLine(px - 20, py, px + 20, py)
            painter.drawLine(px, py - 20, px, py + 20)


class PredictionBar(QWidget):
    """Word prediction display with 3 slots."""
    
    prediction_selected = pyqtSignal(int)  # Emits 0, 1, or 2
    
    def __init__(self, parent_keyboard=None):
        super().__init__(parent_keyboard)
        self.parent_keyboard = parent_keyboard
        self.setObjectName("PredictionBar")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        self._slots: List[QLabel] = []
        for i in range(3):
            slot = QLabel(f"[{i+1}]")
            slot.setObjectName("PredictionSlot")
            slot.setAlignment(Qt.AlignCenter)
            slot.setMinimumHeight(40)
            self._slots.append(slot)
            layout.addWidget(slot)
    
    def set_predictions(self, words: List[str]):
        """Set word predictions (up to 3) on both bar and integrated keys."""
        for i, slot in enumerate(self._slots):
            word = words[i] if i < len(words) else ""
            label = f"[{i+1}] {word}" if word else f"[{i+1}]"
            slot.setText(label)
            
            # Also update integrated layout keys (PRE1, PRE2, PRE3)
            key_name = f"PRE{i+1}"
            if key_name in self.parent_keyboard._keys:
                self.parent_keyboard._keys[key_name].setText(word if word else f"[{i+1}]")
    
    def clear(self):
        """Clear all predictions."""
        for i, slot in enumerate(self._slots):
            slot.setText(f"[{i+1}]")
    
    def highlight_slot(self, index: int):
        """Highlight a prediction slot (0, 1, or 2)."""
        for i, slot in enumerate(self._slots):
            if i == index:
                slot.setProperty("highlighted", "true")
            else:
                slot.setProperty("highlighted", "false")
            slot.style().unpolish(slot)
            slot.style().polish(slot)


class KeyboardWidget(QWidget):
    """
    Main keyboard widget with key grid and swipe logic.
    Optimized for performance by caching key bounding boxes.
    """
    
    key_hovered = pyqtSignal(str)
    swipe_completed = pyqtSignal(list)
    
    def __init__(self, layout_name: str = 'qwerty', parent=None):
        super().__init__(parent)
        self.setObjectName("KeyboardWidget")
        
        self._layout_name = layout_name
        self._layout = get_layout(layout_name)
        self._keys: dict[str, KeyButton] = {}
        self._key_rects_cache: dict[str, QRectF] = {}
        self._current_key: Optional[str] = None
        self._swipe_keys: List[str] = []
        self._swipe_points: List[Tuple[int, int]] = []  # History of pixel points in current swipe
        self._last_reg_index: int = 0  # Point index of the last key registration
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Build the keyboard UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        self.prediction_bar = PredictionBar(self)
        main_layout.addWidget(self.prediction_bar)
        
        self.keyboard_container = QWidget()
        self.keyboard_container.setObjectName("KeyboardContainer")
        self.container_layout = QVBoxLayout(self.keyboard_container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(5)
        
        self._build_layout_ui()
        main_layout.addWidget(self.keyboard_container)
        
        # Cache bounds after layout settles
        QTimer.singleShot(200, self._update_key_rects_cache)

    def _build_layout_ui(self):
        """Build the keys based on current layout."""
        # Clear existing keys from storage (not widget, that happens in update_layout)
        self._keys.clear()
        
        for row in self._layout:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)
            row_layout.addStretch()
            
            for key_data in row:
                char = key_data[0] if isinstance(key_data, tuple) else key_data
                width_mult = key_data[1] if isinstance(key_data, tuple) else 1.0
                
                btn = KeyButton(key_data)
                btn.setMinimumSize(int(45 * width_mult), 50)
                self._keys[char.upper()] = btn
                row_layout.addWidget(btn)
            
            row_layout.addStretch()
            self.container_layout.addWidget(row_widget)

    def update_layout(self, new_layout):
        """Switch to a new keyboard layout."""
        # Remove old rows
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._layout = new_layout
        self._build_layout_ui()
        QTimer.singleShot(200, self._update_key_rects_cache)
    
    def _update_key_rects_cache(self):
        """Pre-calculate and cache key bounding boxes with hit-test margins."""
        self._key_rects_cache.clear()
        margin = 10
        for key_label, key_btn in self._keys.items():
            # Get position relative to KeyboardWidget
            key_pos = key_btn.mapTo(self, QPoint(0, 0))
            rect = QRectF(key_pos.x(), key_pos.y(), 
                          key_btn.width(), key_btn.height())
            # Pre-expand rect so we avoid object creation in hot loop
            self._key_rects_cache[key_label] = rect.adjusted(-margin, -margin, margin, margin)
    
    def resizeEvent(self, event):
        """Recalculate bounds cache on resize."""
        super().resizeEvent(event)
        self._update_key_rects_cache()
    
    def get_key_centers(self) -> Dict[str, Tuple[float, float]]:
        """Return actual normalized (0.0-1.0) centers of keys in the widget."""
        centers = {}
        w = float(self.width())
        h = float(self.height())
        if w < 1 or h < 1:
            return {}
            
        for key, rect in self._key_rects_cache.items():
            center = rect.center()
            centers[key] = (center.x() / w, center.y() / h)
        return centers
    
    def update_hand_position_pixels(self, px: int, py: int):
        """FAST lookup of hovered key using cached rects."""
        hovered_key = None
        
        for key_label, key_rect in self._key_rects_cache.items():
            if key_rect.contains(px, py):
                hovered_key = key_label
                break
        
        if hovered_key != self._current_key:
            if self._current_key and self._current_key in self._keys:
                self._keys[self._current_key].set_highlighted(False)
            
            if hovered_key and hovered_key in self._keys:
                self._keys[hovered_key].set_highlighted(True)
                self.key_hovered.emit(hovered_key)
            
            self._current_key = hovered_key

    @property
    def current_key(self) -> Optional[str]:
        """Return the label of the currently hovered key."""
        return self._current_key
    
    def start_swipe_pixels(self, px: int, py: int):
        """Start tracking a swipe at pixel position."""
        self._swipe_keys.clear()
        self._swipe_points.clear()
        self._swipe_points.append((px, py))
        self._last_reg_index = 0
        
        self.update_hand_position_pixels(px, py)
        if self._current_key and len(self._current_key) == 1 and self._current_key.isalpha():
            self._swipe_keys.append(self._current_key)
            self._last_reg_index = 0
    
    def update_swipe_pixels(self, px: int, py: int):
        """Continue tracking a swipe. Only 'register' key if it's a bend or circle."""
        self.update_hand_position_pixels(px, py)
        self._swipe_points.append((px, py))
        
        if not self._current_key:
            return
            
        # Check for a bend if we have enough points
        if len(self._swipe_points) >= 5:
            p_start = self._swipe_points[-5]
            p_mid = self._swipe_points[-3]
            p_end = self._swipe_points[-1]
            
            if self._is_significant_bend(p_start, p_mid, p_end):
                is_new_key = not self._swipe_keys or self._swipe_keys[-1] != self._current_key
                
                # Filter: ONLY LETTERS are considered for prediction paths
                if len(self._current_key) == 1 and self._current_key.isalpha():
                    # Logic for duplicate letters via "circling":
                    # If we are on the SAME key but we've moved significantly since the last 
                    # registration (at least 15 points / approx 0.5s of motion), register it again.
                    points_since_reg = len(self._swipe_points) - self._last_reg_index
                    
                    if is_new_key or points_since_reg > 15:
                        self._swipe_keys.append(self._current_key)
                        self._last_reg_index = len(self._swipe_points)

    def _is_significant_bend(self, p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int]) -> bool:
        """Calculate if these 3 points form a sharp enough corner (> 45 degrees)."""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        m1 = (v1[0]**2 + v1[1]**2)**0.5
        m2 = (v2[0]**2 + v2[1]**2)**0.5
        
        if m1 < 5 or m2 < 5:  # Ignore very small movements (noise)
            return False
            
        # Dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos_theta = dot / (m1 * m2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        # If cos_theta < 0.8 (approx 36 degrees deviation), it's a "bend"
        # The smaller the value, the sharper the turn needed.
        return cos_theta < 0.85
    
    def end_swipe(self) -> List[str]:
        """Finalize the swipe and return the key path. Always adds the last key."""
        if self._current_key:
            if not self._swipe_keys or self._swipe_keys[-1] != self._current_key:
                self._swipe_keys.append(self._current_key)
                
        path = self._swipe_keys.copy()
        self._swipe_keys.clear()
        self._swipe_points.clear()
        
        if path:
            self.swipe_completed.emit(path)
        return path
    
    def clear_swipe(self):
        """Forcefully clear swipe state without returning data."""
        self._swipe_keys.clear()
        self._swipe_points.clear()
        self._last_reg_index = 0
        self._current_key = None
        for key in self._keys.values():
            key.set_highlighted(False)
    
    def set_predictions(self, words: List[str]):
        """Update the prediction bar text."""
        self.prediction_bar.set_predictions(words)
    
    def highlight_prediction(self, index: int):
        """Highlight a specific slot in prediction bar."""
        self.prediction_bar.highlight_slot(index)

