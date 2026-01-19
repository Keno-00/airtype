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
        self._debug = False
        
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
        QTimer.singleShot(1000, self._sync_prediction_coordinates)
        
        # Ensure predictions are hidden at startup
        self.set_predictions([])
    
    def _setup_prediction(self):
        """Initialize the word prediction engine."""
        dict_path = Path(__file__).parent.parent / "prediction" / "dictionary.json"
        # Initial guess from layout math (will be refined by _sync_prediction_coordinates)
        layout_data = get_layout(self._layout_name)
        positions = get_key_positions(layout_data)
        self.prediction_engine = PredictionEngine(dict_path, positions)
        
    def _sync_prediction_coordinates(self):
        """Update prediction engine with actual UI key coordinates."""
        # Normalize relative to swipe_canvas (which covers the whole window)
        centers = self.keyboard.get_key_centers(relative_to=self.swipe_canvas)
        if centers:
            self.prediction_engine.update_layout(centers)
            print("Prediction coordinates synchronized with UI overlay.")
    
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
        central = QWidget()
        central.setObjectName("CentralWidget")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(central)
        
        # 1. Main Content Container
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_container)
        

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
        # Don't set read-only so cursor is visible, but we'll filter key events
        self.input_field.setPlaceholderText("AirType...")
        self.input_field.installEventFilter(self)  # Filter keyboard input
        self.input_field.cursorPositionChanged.connect(self._handle_cursor_changed)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("→")
        self.send_button.setObjectName("SendButton")
        self.send_button.setFixedSize(40, 40)
        self.send_button.clicked.connect(self._handle_send)
        input_layout.addWidget(self.send_button)
        
        layout.addWidget(self.input_container)
        
        # Keyboard widget (Directly in main layout now)
        self.keyboard = KeyboardWidget(self._layout_name)
        layout.addWidget(self.keyboard)
        
        # Webcam preview below keyboard
        self.webcam_preview = QLabel()
        self.webcam_preview.setObjectName("WebcamPreview")
        self.webcam_preview.setAlignment(Qt.AlignCenter)
        self.webcam_preview.setMinimumHeight(80)
        self.webcam_preview.setMaximumHeight(150)
        self.webcam_preview.setScaledContents(True)
        layout.addWidget(self.webcam_preview)

        # 2. Global Overlay Layer (Cursor & Path)
        # Direct child of window, not in layout, to ensure full window coverage
        from .keyboard_widget import SwipeCanvas
        self.swipe_canvas = SwipeCanvas(self)
        self.swipe_canvas.raise_()
    
    def resizeEvent(self, event):
        """Handle window resize to update overlay and caches."""
        super().resizeEvent(event)
        if hasattr(self, 'swipe_canvas'):
            self.swipe_canvas.setGeometry(self.rect())
            
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
        
        #KeyButton[prediction="true"][empty="true"] {{
            background-color: transparent;
            border: 1px solid transparent;
            color: transparent;
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
        
        #SendButton[hover="true"] {{
            background-color: {accent};
            border: 2px solid rgba(255, 255, 255, 200);
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
    
    
    def set_position(self, position: str):
        """Set dock position ('left' or 'right')."""
        self._position = position
        self._position_window()
    
    def update_hand_position(self, x: float, y: float):
        """Update keyboard based on hand position (normalized 0-1 for whole window)."""
        if not self.swipe_canvas or not self.swipe_canvas.window():
            return
            
        self.swipe_canvas.set_cursor(x, y)
        
        # 1. Map to whole window pixel coordinates
        # Since swipe_canvas covers CentralWidget, we use its size
        w = self.swipe_canvas.width()
        h = self.swipe_canvas.height()
        if w < 1 or h < 1:
            return
            
        px = int(x * w)
        py = int(y * h)
        
        # Use global coordinates for robustness across stacked layers
        global_pos = self.swipe_canvas.mapToGlobal(QPoint(px, py))
        
        if self._debug:
            print(f"DEBUG_MAP: x,y=({x:.3f}, {y:.3f}) px,py=({px}, {py}) global=({global_pos.x()}, {global_pos.y()})")
            print(f"DEBUG_GEO: Canvas={self.swipe_canvas.geometry()} Keyboard={self.keyboard.geometry()}")
            print(f"DEBUG_MAP_ROOT: CanvasGlobal={self.swipe_canvas.mapToGlobal(QPoint(0,0))} KeyboardGlobal={self.keyboard.mapToGlobal(QPoint(0,0))}")

        # 2. Check Send Button
        if self.send_button and self.send_button.window():
            btn_pos = self.send_button.mapFromGlobal(global_pos)
            is_hovering_send = self.send_button.rect().contains(btn_pos)
            
            # Only update if property changed to avoid excessive style polish
            old_hover = self.send_button.property("hover") == "true"
            if is_hovering_send != old_hover:
                self.send_button.setProperty("hover", "true" if is_hovering_send else "false")
                self.send_button.style().unpolish(self.send_button)
                self.send_button.style().polish(self.send_button)

        # 3. Update Keyboard Key Highlighting
        if self.keyboard and self.keyboard.window():
            kb_pos = self.keyboard.mapFromGlobal(global_pos)
            if self._debug:
                print(f"DEBUG_HIT: kb_pos=({kb_pos.x()}, {kb_pos.y()})")
            self.keyboard.update_hand_position_pixels(kb_pos.x(), kb_pos.y())
    
    def navigate_key(self, direction: str):
        """
        Navigate to adjacent key using D-pad direction.
        
        Args:
            direction: 'up', 'down', 'left', or 'right'
        """
        self.keyboard.navigate_direction(direction)
    
    def select_current_key(self):
        """Select the currently highlighted key (A button in controller mode)."""
        # 1. Check if hovering Send button
        if self.send_button.property("hover") == "true":
            self._handle_send()
            return

        # 2. Check Keyboard
        current = self.keyboard.current_key
        if not current:
            return
        
        # Handle special keys
        if current in ["SHIFT", "⌫", "CLOSE", "?123", "ABC", "PRE1", "PRE2", "PRE3", "SPACE", "↵"]:
            self.handle_special_key(current)
        elif len(current) == 1:
            # Normal letter/number/symbol - insert at cursor position
            char = self._format_text(current)
            self._insert_at_cursor(char)
            
            # Consume capitalization states
            if self._shift_on or self._sentence_start:
                self._shift_on = False
                self._sentence_start = False
                self._update_key_casing()
            
            # Predict completions based on current word at cursor
            if self._layout_name == 'qwerty':
                # Extract word being typed at cursor
                word = self._get_word_at_cursor()
                if word:
                    top_words = self.prediction_engine.predict_prefix(word)
                    top_words = [self._format_text(w) for w in top_words]
                    self.set_predictions(top_words)
                else:
                    self.set_predictions([])
    
    def _insert_at_cursor(self, text: str):
        """Insert text at the current cursor position."""
        current_text = self.input_field.text()
        pos = self.input_field.cursorPosition()
        new_text = current_text[:pos] + text + current_text[pos:]
        self.input_field.setText(new_text)
        self.input_field.setCursorPosition(pos + len(text))
        # Also update internal state
        self._input_text = new_text
        self._current_word = ""
    
    def _delete_at_cursor(self):
        """Delete character before cursor position."""
        current_text = self.input_field.text()
        pos = self.input_field.cursorPosition()
        if pos > 0:
            new_text = current_text[:pos-1] + current_text[pos:]
            self.input_field.setText(new_text)
            self.input_field.setCursorPosition(pos - 1)
            self._input_text = new_text
            self._current_word = ""

    def _delete_last_word_at_cursor(self, length: int):
        """Delete specific number of characters before cursor position."""
        current_text = self.input_field.text()
        pos = self.input_field.cursorPosition()
        if pos >= length:
            new_text = current_text[:pos-length] + current_text[pos:]
            self.input_field.setText(new_text)
            self.input_field.setCursorPosition(pos - length)
            self._input_text = new_text
            self._current_word = ""
    
    def _get_word_at_cursor(self) -> str:
        """Get the word being typed at the cursor position."""
        text = self.input_field.text()
        pos = self.input_field.cursorPosition()
        # Find word start
        start = pos
        while start > 0 and text[start-1].isalpha():
            start -= 1
        return text[start:pos]
    
    def _replace_word_at_cursor(self, replacement: str):
        """Replace the word at cursor with new text."""
        text = self.input_field.text()
        pos = self.input_field.cursorPosition()
        # Find word boundaries
        start = pos
        while start > 0 and text[start-1].isalpha():
            start -= 1
        end = pos
        while end < len(text) and text[end].isalpha():
            end += 1
            
        # Replace the partial/full word with the full prediction
        new_text = text[:start] + replacement + text[end:]
        self.input_field.setText(new_text)
        self.input_field.setCursorPosition(start + len(replacement))
        self._input_text = new_text
        self._current_word = replacement.strip()
        self._last_swipe_time = time.time()

    def _handle_cursor_changed(self):
        """Handle cursor movement to update predictions context."""
        if self._layout_name == 'qwerty':
            word = self._get_word_at_cursor()
            if word:
                top_words = self.prediction_engine.predict_prefix(word)
                formatted_words = [self._format_text(w) for w in top_words]
                self.set_predictions(formatted_words)
            else:
                self.set_predictions([])

    
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
            self.input_field.clear()
            self._check_sentence_boundary()
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
        
        # 1. Check Send Button interaction
        if self.send_button.property("hover") == "true":
            self._handle_send()
            return ["SEND"]
        
        if not raw_keys:
            return []
            
        # Is this a single tap (length 1 path)?
        if len(points) < 5 or len(raw_keys) == 1:
            key = raw_keys[0]
            if key in ["SHIFT", "BACKSPACE", "CLOSE", "?123", "ABC", "PRE1", "PRE2", "PRE3", "SPACE"]:
                self.handle_special_key(key)
            elif len(key) == 1:
                # Normal letter/number/symbol tap - insert at cursor
                char = self._format_text(key)
                self._insert_at_cursor(char)
                
                # Consume capitalization states
                if self._shift_on or self._sentence_start:
                    self._shift_on = False
                    self._sentence_start = False
                    self._update_key_casing()
            return raw_keys

        # It's a swipe (shape-writing)
        top_words = self.prediction_engine.predict(points)
        if top_words:
            # Swipe sets the word as "pending" for smart backspace
            word = top_words[0]
            formatted_word = self._format_text(word)
            
            # Word-aware insertion
            self._insert_at_cursor(formatted_word)
            self._current_word = formatted_word # Store for smart backspace
            
            # Reset shift/sentence start
            if self._shift_on or self._sentence_start:
                self._shift_on = False
                self._sentence_start = False
                self._update_key_casing()

            self._last_swipe_time = time.time()
            
            # Format and show predictions
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
        text = self.input_field.text().strip()
        if not text:
            self._sentence_start = True
        elif text and text[-1] in ".!?":
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

    def handle_special_key(self, key):
        """Handle functional keys like SHIFT, BACKSPACE, etc."""
        if key == "SHIFT":
            # Toggle SHIFT state (clears after next letter or word)
            self._shift_on = not self._shift_on
            self._update_key_casing()
            
        elif key in ["BACKSPACE", "⌫"]:
            # Delete character before cursor (or whole word if just swiped)
            if self._current_word and (time.time() - self._last_swipe_time < 2.0):
                # Smart Backspace: Delete whole word if immediately after swipe (2s window)
                print(f"Action: Quick Delete word '{self._current_word}'")
                self._delete_last_word_at_cursor(len(self._current_word))
                self._current_word = ""
                self._check_sentence_boundary()
            else:
                # Delete at cursor position
                self._delete_at_cursor()
                self._check_sentence_boundary()
            
            # Predictions handled by _handle_cursor_changed connection
                
        elif key == "SPACE":
            # Insert space at cursor position
            self._insert_at_cursor(" ")
            self._current_word = ""
            self._shift_on = False
            self._check_sentence_boundary()
            self.set_predictions([])
        
        elif key in ["ENTER", "↵"]:
            # Send the text (same as clicking the send button)
            self._handle_send()
            
        elif key == "CLOSE":
            QApplication.quit()
        elif key in ["?123", "ABC"]:
            new_layout = "symbols" if key == "?123" else "qwerty"
            self.switch_layout(new_layout)
        elif key.startswith("PRE"):
            idx = int(key[3:]) - 1
            if idx < len(self._last_predictions):
                word = self._last_predictions[idx]
                # Replace the word at cursor with prediction + space
                self._replace_word_at_cursor(word + " ")
                self._shift_on = False
                self._check_sentence_boundary()
                self.set_predictions([])
                print(f"Action: Selection committed '{word}'")

    def switch_layout(self, layout_name):
        """Switch between QWERTY and Symbols layouts and reset state if needed."""
        self._layout_name = layout_name
        self.keyboard.update_layout(get_layout(layout_name))
        
        # Prediction only active for qwerty; always sync to clear/restore labels
        if layout_name == 'qwerty':
            self.set_predictions(self._last_predictions)
        else:
            self.set_predictions([])
        
        # Re-apply controller hints after layout change
        if hasattr(self, '_controller_mode') and self._controller_mode:
            QTimer.singleShot(100, lambda: self.keyboard.set_controller_hints(True))
            
        # Ensure coordinates are re-synced after layout settles
        QTimer.singleShot(250, self._sync_prediction_coordinates)
    
    def set_controller_mode(self, enabled: bool):
        """Enable or disable controller mode (shows button hints)."""
        self._controller_mode = enabled
        self.keyboard.set_controller_hints(enabled)
        
        # Add hint to send button
        if enabled:
            from pathlib import Path
            from PyQt5.QtSvg import QSvgWidget
            from PyQt5.QtCore import Qt
            
            icon_path = Path(__file__).parent.parent.parent / "assets" / "icons" / "controller" / "xb_start.svg"
            if icon_path.exists():
                if not hasattr(self, '_send_hint') or self._send_hint is None:
                    self._send_hint = QSvgWidget(self.send_button)
                    self._send_hint.setAttribute(Qt.WA_TranslucentBackground)
                self._send_hint.load(str(icon_path))
                self._send_hint.setFixedSize(24, 10)
                self._send_hint.move(8, 2)
                self._send_hint.show()
        elif hasattr(self, '_send_hint') and self._send_hint:
            self._send_hint.hide()
    
    def move_text_cursor(self, direction: int):
        """
        Move the text cursor left or right.
        
        Args:
            direction: -1 for left, 1 for right
        """
        # Get current cursor position
        pos = self.input_field.cursorPosition()
        new_pos = max(0, min(len(self.input_field.text()), pos + direction))
        self.input_field.setCursorPosition(new_pos)

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
    
    def eventFilter(self, obj, event):
        """Filter keyboard events to prevent typing in the input field."""
        from PyQt5.QtCore import QEvent
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            # Block all key presses to the input field
            return True
        return super().eventFilter(obj, event)
