"""
Overlay window - frameless, transparent, always-on-top.
"""
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QStackedLayout
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QScreen

from .keyboard_widget import KeyboardWidget
from .layouts import get_key_positions, get_layout
from prediction.engine import PredictionEngine


class OverlayWindow(QMainWindow):
    """
    Frameless overlay window that docks to left or right side of screen.
    
    Features:
    - Always on top
    - Semi-transparent
    - 1/4 screen width
    - Draggable
    """
    
    def __init__(
        self, 
        position: str = 'right',
        layout_name: str = 'qwerty',
        theme_path: Optional[Path] = None,
        parent=None
    ):
        super().__init__(parent)
        
        self._position = position
        self._layout_name = layout_name
        self._prediction_index = -1
        self._drag_pos: Optional[QPoint] = None
        
        self._setup_window()
        self._setup_ui()
        self._setup_prediction()
        
        if theme_path:
            self.load_theme(theme_path)
        else:
            self._load_default_theme()
        
        self._position_window()
        
        # After UI settles, update prediction engine with REAL key positions
        QTimer.singleShot(500, self._sync_prediction_coordinates)
    
    def _setup_prediction(self):
        """Initialize the word prediction engine."""
        dict_path = Path(__file__).parent.parent / "prediction" / "dictionary.json"
        # Initial guess from layout math (will be refined by _sync_prediction_coordinates)
        layout_data = get_layout(self._layout_name)
        positions = get_key_positions(layout_data)
        self.prediction_engine = PredictionEngine(dict_path, positions)
        
    def _sync_prediction_coordinates(self):
        """Update prediction engine with actual UI key coordinates."""
        centers = self.keyboard.get_key_centers()
        if centers:
            self.prediction_engine.update_layout(centers)
            print("Prediction coordinates synchronized with UI.")
    
    def _setup_window(self):
        """Configure window flags and attributes."""
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool  # Don't show in taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("OverlayWindow")
    
    def _setup_ui(self):
        """Build the UI."""
        # Central widget with background
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 0, 15, 10)  # Left, top, right, bottom padding
        
        # Title bar for dragging
        title_bar = QWidget()
        title_bar.setObjectName("TitleBar")
        title_bar.setFixedHeight(30)
        title_bar.mousePressEvent = self._title_mouse_press
        title_bar.mouseMoveEvent = self._title_mouse_move
        layout.addWidget(title_bar)
        
        # Container with stacked layout for keyboard + swipe canvas
        container = QWidget()
        stacked = QStackedLayout(container)
        stacked.setStackingMode(QStackedLayout.StackAll)
        
        # Keyboard widget (bottom layer)
        self.keyboard = KeyboardWidget(self._layout_name)
        stacked.addWidget(self.keyboard)
        
        # Swipe canvas (top layer, transparent)
        from .keyboard_widget import SwipeCanvas
        self.swipe_canvas = SwipeCanvas()
        stacked.addWidget(self.swipe_canvas)
        
        layout.addWidget(container)
    
    def _position_window(self):
        """Position window using a formula based on screen dimensions."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        
        # Use full screen geometry to get absolute coordinates
        screen_geo = screen.geometry()
        sw = screen_geo.width()
        sh = screen_geo.height()
        sx = screen_geo.x()
        sy = screen_geo.y()
        
        # Formula: 
        # Width: 25% of screen (Standard 1/4 layout)
        # Height: 50% of screen (Centered vertically)
        # Margin: 2% of screen width from the nearest edge
        w = int(sw * 0.25)
        h = int(sh * 0.50)
        margin = int(sw * 0.02)
        
        # Calculate X based on position
        if self._position == 'left':
            x = sx + margin
        else:  # right
            x = sx + sw - w - margin
            
        # Calculate Y (Centered vertically)
        y = sy + (sh - h) // 2
        
        # Safety clamp: Ensure the window actually fits on the screen
        x = max(sx, min(x, sx + sw - w))
        y = max(sy, min(y, sy + sh - h))
        
        print(f"UI Window Geometry: {w}x{h} at global ({x}, {y})")
        
        self.setFixedSize(w, h)
        self.move(x, y)
    
    def _load_default_theme(self):
        """Load the default dark theme."""
        theme = """
        #CentralWidget {
            background-color: rgba(30, 30, 40, 230);
            border-radius: 10px;
        }
        
        #TitleBar {
            background-color: rgba(50, 50, 60, 200);
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }
        
        #KeyboardWidget {
            background: transparent;
        }
        
        #KeyboardContainer {
            background: transparent;
        }
        
        #KeyButton {
            background-color: rgba(60, 60, 80, 200);
            color: white;
            border: 1px solid rgba(100, 100, 120, 150);
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            padding: 5px;
        }
        
        #KeyButton[special="true"] {
            background-color: rgba(45, 45, 60, 220);
            font-size: 11px;
            color: rgba(255, 255, 255, 200);
        }
        
        /* Integrated prediction slots at the top */
        #KeyButton[text^="[1]"], #KeyButton[text^="[2]"], #KeyButton[text^="[3]"],
        #KeyButton[key="PRE1"], #KeyButton[key="PRE2"], #KeyButton[key="PRE3"] {
            background-color: rgba(80, 70, 60, 220);
            color: #ffcc66;
            font-size: 13px;
        }
        
        #KeyButton[highlighted="true"] {
            background-color: rgba(100, 200, 255, 220);
            color: black;
            border: 2px solid rgba(255, 255, 255, 150);
        }
        
        #PredictionBar {
            background-color: rgba(30, 30, 40, 220);
            border-radius: 8px;
            margin-bottom: 5px;
        }
        
        #PredictionSlot {
            background-color: rgba(50, 50, 70, 150);
            color: white;
            border: 1px solid rgba(100, 100, 120, 80);
            border-radius: 5px;
            font-size: 14px;
            padding: 8px;
        }
        
        #PredictionSlot[highlighted="true"] {
            background-color: rgba(255, 200, 100, 230);
            color: black;
            font-weight: bold;
        }
        """
        self.setStyleSheet(theme)
    
    def load_theme(self, path: Path):
        """Load theme from QSS file."""
        if path.exists():
            with open(path, 'r') as f:
                self.setStyleSheet(f.read())
        else:
            self._load_default_theme()
    
    def _title_mouse_press(self, event):
        """Handle title bar mouse press for dragging."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def _title_mouse_move(self, event):
        """Handle title bar mouse move for dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
    
    def set_position(self, position: str):
        """Set dock position ('left' or 'right')."""
        self._position = position
        self._position_window()
    
    def update_hand_position(self, x: float, y: float):
        """Update keyboard based on hand position."""
        self.swipe_canvas.set_cursor(x, y)
        # Convert to pixel coords using swipe canvas dimensions
        px = int(x * self.swipe_canvas.width())
        py = int(y * self.swipe_canvas.height())
        self.keyboard.update_hand_position_pixels(px, py)
    
    def start_swipe(self, x: float, y: float):
        """Start a swipe gesture."""
        self.swipe_canvas.start_swipe()
        self.swipe_canvas.add_point(x, y)
        px = int(x * self.swipe_canvas.width())
        py = int(y * self.swipe_canvas.height())
        self.keyboard.start_swipe_pixels(px, py)
    
    def update_swipe(self, x: float, y: float):
        """Update swipe path."""
        self.swipe_canvas.add_point(x, y)
        px = int(x * self.swipe_canvas.width())
        py = int(y * self.swipe_canvas.height())
        self.keyboard.update_swipe_pixels(px, py)
    
    def cancel_swipe(self):
        """Abort the current swipe (e.g., hand lost)."""
        self.swipe_canvas.end_swipe()
        self.keyboard.clear_swipe()
        
    def end_swipe(self):
        """End swipe gesture and perform shape-based prediction."""
        points = self.swipe_canvas._path_points.copy()
        raw_keys = self.keyboard.end_swipe()
        
        # If the swipe was ONLY functional keys (like single tap on shift/backspace), 
        # handle that instead of word prediction
        if len(raw_keys) == 1:
            key = raw_keys[0]
            if key in ["SHIFT", "BACKSPACE", "CLOSE", "?123", "ABC", "PRE1", "PRE2", "PRE3"]:
                self.handle_special_key(key)
                return raw_keys

        # Get predictions based on path shape
        top_words = self.prediction_engine.predict(points)
        if top_words:
            self.set_predictions(top_words)
            self._prediction_index = -1
            
        return raw_keys

    def handle_special_key(self, key):
        """Handle functional keys like SHIFT, BACKSPACE, etc."""
        if key == "SHIFT":
            # Just a single tap shift? 
            pass
        elif key == "BACKSPACE":
            print("Action: Backspace")
        elif key == "CLOSE":
            QApplication.quit()
        elif key in ["?123", "ABC"]:
            new_layout = "symbols" if key == "?123" else "qwerty"
            self.switch_layout(new_layout)
        elif key.startswith("PRE"):
            idx = int(key[3:]) - 1
            print(f"Action: Selection Slot {idx+1}")

    def switch_layout(self, layout_name):
        """Switch between QWERTY and Symbols layouts."""
        self._layout_name = layout_name
        self.keyboard.update_layout(get_layout(layout_name))
        QTimer.singleShot(100, self._sync_prediction_coordinates)

    def cycle_prediction(self):
        """Cycle through top 3 prediction results."""
        self._prediction_index = (self._prediction_index + 1) % 3
        self.highlight_prediction(self._prediction_index)

    def toggle_caps(self):
        """Handle double-fist caps lock - triggered if hovering SHIFT."""
        if self.keyboard.current_key == "SHIFT":
            print("Action: CAPS LOCK TOGGLED")
        else:
            print(f"Double-fist ignored (hovering {self.keyboard.current_key}, not SHIFT)")
    
    def set_predictions(self, words):
        """Set word predictions."""
        self.keyboard.set_predictions(words)
    
    def highlight_prediction(self, index: int):
        """Highlight a prediction slot."""
        self.keyboard.highlight_prediction(index)
