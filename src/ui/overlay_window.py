"""
Overlay window - frameless, transparent, always-on-top.
"""
from pathlib import Path
from typing import Optional
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QStackedLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QScreen, QImage, QPixmap
import numpy as np
import time

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
        
        # Text accumulation state
        self._input_text = ""   # Committed text
        self._current_word = "" # Uncommitted partial word
        self._last_predictions = []
        self._last_swipe_time = 0.0 # For smart backspace
        
        # Capitalization state
        self._shift_on = False
        self._caps_lock = False
        self._sentence_start = True
        
        self._setup_window()
        self._setup_ui()
        self._setup_prediction()
        
        if theme_path:
            self.load_theme(theme_path)
        else:
            self._load_default_theme()
        
        self._position_window()
        
        # Initialize native theme after UI is built
        self._apply_native_theme()
        self._update_key_casing()
        
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

        # ---------------------------------------------------------
        # Persisent Input Field (Accumulator)
        # ---------------------------------------------------------
        self.input_container = QWidget()
        self.input_container.setObjectName("InputContainer")
        self.input_container.setFixedHeight(50)
        input_layout = QHBoxLayout(self.input_container)
        input_layout.setContentsMargins(5, 5, 5, 5)
        
        self.input_field = QLineEdit()
        self.input_field.setObjectName("PersistentInputField")
        self.input_field.setReadOnly(True)
        self.input_field.setPlaceholderText("AirType...")
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("→")
        self.send_button.setObjectName("SendButton")
        self.send_button.setFixedSize(40, 40)
        self.send_button.clicked.connect(self._handle_send)
        input_layout.addWidget(self.send_button)
        
        layout.addWidget(self.input_container)
        
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
        
        # Webcam preview below keyboard
        self.webcam_preview = QLabel()
        self.webcam_preview.setObjectName("WebcamPreview")
        self.webcam_preview.setAlignment(Qt.AlignCenter)
        self.webcam_preview.setMinimumHeight(80)
        self.webcam_preview.setMaximumHeight(150)
        self.webcam_preview.setScaledContents(True)
        layout.addWidget(self.webcam_preview)
    
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
        # Width: 32% of screen (Wider for de-smushing)
        # Height: 45% of screen (Better aspect ratio)
        # Margin: 2% of screen width from the nearest edge
        w = int(sw * 0.32)
        h = int(sh * 0.45)
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
    
    def _apply_native_theme(self):
        """Inherit colors and fonts from the system theme with robust dark detection."""
        # Use Application-level palette for more reliable system-wide theme detection
        pal = QApplication.palette()
        
        bg_color = pal.color(pal.Window)
        fg_color = pal.color(pal.WindowText)
        
        # Robust dark detection: check if text is significantly lighter than background
        is_dark = fg_color.lightness() > bg_color.lightness()
        
        # FALLBACK: If the system returns a pure black/white or broken palette
        if is_dark:
            # Force high-contrast dark if system gives us something too bright but says it's dark
            bg = bg_color.name()
            fg = fg_color.name()
            accent = pal.color(pal.Highlight).name()
            if accent.lower() in ["#ffffff", "#000000"]: # Fallback for broken accent detection
                accent = "#3a86ff"
        else:
            bg = bg_color.name()
            fg = fg_color.name()
            accent = pal.color(pal.Highlight).name()
            if accent.lower() in ["#ffffff", "#000000"]:
                accent = "#007aff" # macOS/iOS-like blue fallback
        
        accent_fg = pal.color(pal.HighlightedText).name()
        
        # Build RGBA with appropriate opacity
        r, g, b = bg_color.red(), bg_color.green(), bg_color.blue()
        if is_dark:
            # Dark mode: slightly deeper background for readability
            bg_rgba = f"rgba({max(20, int(r*0.9))}, {max(20, int(g*0.9))}, {max(30, int(b*0.9))}, 210)"
            input_rgba = f"rgba(0, 0, 0, 60)"
        else:
            bg_rgba = f"rgba({r}, {g}, {b}, 220)"
            input_rgba = f"rgba(255, 255, 255, 60)"
            
        font_family = QApplication.font().family()
        
        theme = f"""
        * {{
            font-family: "{font_family}";
        }}
        
        #CentralWidget {{
            background-color: {bg_rgba};
            border: 1px solid rgba(120, 120, 140, 60);
            border-radius: 12px;
        }}
        
        #TitleBar {{
            background-color: rgba(0, 0, 0, 30);
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }}
        
        /* Keys */
        #KeyButton {{
            background-color: rgba({"255, 255, 255" if is_dark else "0, 0, 0"}, 15);
            color: {fg};
            border-radius: 6px;
            font-size: 16px;
            border: 1px solid rgba({"255, 255, 255" if is_dark else "0, 0, 0"}, 10);
        }}
        
        #KeyStatusDot {{
            background-color: {accent};
            border-radius: 3px;
        }}
        
        #KeyButton[special="true"] {{
            background-color: rgba(0, 0, 0, 30);
            font-weight: bold;
        }}
        
        #KeyButton[prediction="true"] {{
            background-color: rgba(0, 0, 0, 20);
            color: {accent};
            font-size: 14px;
            border: 1px solid {accent}40;
        }}
        
        #KeyButton[highlighted="true"] {{
            background-color: {accent};
            color: {accent_fg};
        }}
        
        #InputContainer {{
            background-color: {input_rgba};
            border-radius: 8px;
            margin-bottom: 5px;
            border: 1px solid rgba({"255, 255, 255" if is_dark else "0, 0, 0"}, 15);
        }}
        
        #PersistentInputField {{
            background: transparent;
            border: none;
            color: {fg};
            font-size: 18px;
            padding-left: 10px;
        }}
        
        #SendButton {{
            background-color: {accent};
            color: {accent_fg};
            border-radius: 20px;
            font-size: 20px;
        }}
        
        #WebcamPreview {{
            background-color: rgba(0, 0, 0, 80);
            border: 1px solid rgba(100, 100, 120, 80);
            border-radius: 8px;
        }}
        """
        self.setStyleSheet(theme)
    
    def _load_default_theme(self):
        # We now use _apply_native_theme instead
        self._apply_native_theme()

    def update_theme(self):
        """Update the theme if system colors change."""
        self._apply_native_theme()
    
    def load_theme(self, path: Path):
        """Load theme from QSS file."""
        if path.exists():
            with open(path, 'r') as f:
                self.setStyleSheet(f.read())
        else:
            self._apply_native_theme()
    
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

    def _handle_send(self):
        """Handle the send button - print to terminal and clear field."""
        text = self.input_field.text()
        if text:
            print(f"Action: SENDING TEXT -> '{text}'")
            self._input_text = ""
            self._current_word = ""
            self._update_display()
            self.set_predictions([])
            
    def toggle_caps(self):
        """Handle double-fist caps lock - triggered if hovering SHIFT."""
        if self.keyboard.current_key == "SHIFT":
            self._caps_lock = not self._caps_lock
            print(f"Action: CAPS LOCK {'ON' if self._caps_lock else 'OFF'}")
            self._update_key_casing()
            # Double fist on shift counts as shift intent too
            self._shift_on = False
        else:
            print(f"Double-fist ignored (hovering {self.keyboard.current_key}, not SHIFT)")
            
    def end_swipe(self):
        """End swipe gesture and perform shape or prefix-based prediction."""
        points = self.swipe_canvas._path_points.copy()
        raw_keys = self.keyboard.end_swipe()
        
        if not raw_keys:
            return []
            
        # Is this a single tap (length 1 path)?
        if len(points) < 5 or len(raw_keys) == 1:
            key = raw_keys[0]
            if key in ["SHIFT", "BACKSPACE", "CLOSE", "?123", "ABC", "PRE1", "PRE2", "PRE3", "SPACE"]:
                self.handle_special_key(key)
            elif len(key) == 1:
                # Normal letter/number/symbol tap
                char = self._format_text(key)
                self._current_word += char
                self._update_display()
                
                # Consume capitalization states
                if self._shift_on or self._sentence_start:
                    self._shift_on = False
                    self._sentence_start = False
                    self._update_key_casing()
                
                # Predict completions based on prefix
                if self._layout_name == 'qwerty':
                    top_words = self.prediction_engine.predict_prefix(self._current_word)
                    # Format predictions to match current casing context
                    top_words = [self._format_text(w) for w in top_words]
                    self.set_predictions(top_words)
            return raw_keys

        # It's a swipe (shape-writing)
        top_words = self.prediction_engine.predict(points)
        if top_words:
            # Swipe no longer auto-commits with space. 
            # It sets the word as "pending" for manual confirmation or quick delete.
            word = top_words[0]
            self._current_word = self._format_text(word)
            
            # Consume capitalization states for the whole word
            if self._shift_on or self._sentence_start:
                self._shift_on = False
                self._sentence_start = False
                self._update_key_casing()

            self._last_swipe_time = time.time()
            self._update_display()
            
            # Format all predictions
            formatted_words = [self._format_text(w) for w in top_words]
            self.set_predictions(formatted_words)
            self._prediction_index = 0
            self.highlight_prediction(0)
            
        return raw_keys

    def _format_text(self, text: str) -> str:
        """Apply SHIFT, CAPS LOCK, and Sentence Start casing to text."""
        if not text:
            return ""
            
        # If CAPS LOCK is on, everything is SCREAMING
        if self._caps_lock:
            return text.upper()
            
        # If SHIFT or Sentence Start is on, capitalize the first letter
        if (self._shift_on or self._sentence_start) and text[0].isalpha():
            if len(text) == 1:
                return text.upper()
            return text.capitalize()
            
        return text.lower()

    def _check_sentence_boundary(self):
        """Determine if we are at the start of a new sentence."""
        text = self._input_text.strip()
        if not text:
            self._sentence_start = True
        elif text[-1] in ".!?":
            self._sentence_start = True
        else:
            self._sentence_start = False
        
        self._update_key_casing()

    def _update_key_casing(self):
        """Update keyboard key labels to reflect current casing state."""
        self.keyboard.set_shift_mode(self._shift_on or self._sentence_start, self._caps_lock)
        # Also sync CAPS indicator specifically
        self.keyboard.set_key_active("CAPS", self._caps_lock)
        self.keyboard.set_key_active("SHIFT", self._shift_on)

    def _update_display(self):
        """Update the input field with committed text + current word."""
        full_text = self._input_text + self._current_word
        self.input_field.setText(full_text)
        self.input_field.setCursorPosition(len(full_text))

    def handle_special_key(self, key):
        """Handle functional keys like SHIFT, BACKSPACE, etc."""
        if key == "SHIFT":
            # Toggle SHIFT state (clears after next letter or word)
            self._shift_on = not self._shift_on
            self._update_key_casing()
            
        elif key == "BACKSPACE":
            # Smart Backspace: Delete whole word if immediately after swipe (2s window)
            if self._current_word and (time.time() - self._last_swipe_time < 2.0):
                print(f"Action: Quick Delete word '{self._current_word}'")
                self._current_word = ""
                # Reset sentence start check if we delete a whole word
                self._check_sentence_boundary()
            elif self._current_word:
                self._current_word = self._current_word[:-1]
                if not self._current_word:
                    self._check_sentence_boundary()
            elif self._input_text:
                self._input_text = self._input_text[:-1]
                self._check_sentence_boundary()
            
            self._update_display()
            
            # Update prefix predictions
            if self._current_word:
                top_words = self.prediction_engine.predict_prefix(self._current_word)
                formatted_words = [self._format_text(w) for w in top_words]
                self.set_predictions(formatted_words)
            else:
                self.set_predictions([])
                
        elif key == "SPACE":
            # Committed current word + space
            if self._current_word:
                self._input_text += self._current_word
                self._current_word = ""
            
            self._input_text += " "
            self._shift_on = False # Reset shift on commitment
            self._check_sentence_boundary()
            self._update_display()
            self.set_predictions([])
            
        elif key == "CLOSE":
            QApplication.quit()
        elif key in ["?123", "ABC"]:
            new_layout = "symbols" if key == "?123" else "qwerty"
            self.switch_layout(new_layout)
        elif key.startswith("PRE"):
            idx = int(key[3:]) - 1
            if idx < len(self._last_predictions):
                word = self._last_predictions[idx]
                # Selection replaces current partial word and adds a space
                self._input_text += word + " "
                self._current_word = ""
                self._shift_on = False
                self._check_sentence_boundary()
                self._update_display()
                self.set_predictions([])
                print(f"Action: Selection committed '{word}'")

    def switch_layout(self, layout_name):
        """Switch between QWERTY and Symbols layouts and reset state if needed."""
        self._layout_name = layout_name
        self.keyboard.update_layout(get_layout(layout_name))
        
        # Prediction only active for letters
        if layout_name != 'qwerty':
            self.set_predictions([])
            
        # Ensure coordinates are re-synced after layout settles
        QTimer.singleShot(250, self._sync_prediction_coordinates)

    def cycle_prediction(self):
        """Cycle through top 3 prediction results in the text field."""
        if not self._last_predictions:
            return
            
        self._prediction_index = (self._prediction_index + 1) % len(self._last_predictions)
        self.highlight_prediction(self._prediction_index)
        
        # Update current word in display to show the cycled candidate?
        # Typically shape-writers commit immediately.
        # But for cycling, we could replace the last word.
        # For now, let's just stick to the requested behavior.
    
    def set_predictions(self, words):
        """Set word predictions and store them."""
        self._last_predictions = words
        self.keyboard.set_predictions(words)
        self._prediction_index = -1 # Reset highlight
        self.highlight_prediction(-1)
    
    def highlight_prediction(self, index: int):
        """Highlight a prediction slot."""
        self.keyboard.highlight_prediction(index)
    
    def set_webcam_frame(self, frame: np.ndarray):
        """
        Update the webcam preview - shows only hand landmarks on black.
        
        Args:
            frame: BGR numpy array with landmarks on black from HandTracker
        """
        if frame is None:
            self.webcam_preview.clear()
            return
        
        # Convert BGR to RGB and display directly (no extra processing)
        # Frame is already flipped by HandTracker
        rgb = frame[:, :, ::-1].copy()
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.webcam_preview.setPixmap(pixmap)
