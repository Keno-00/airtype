"""
AirType UI Module

PyQt5 keyboard overlay for swipe-typing.
"""
from .layouts import get_layout, get_key_positions, QWERTY, SYMBOLS
from .keyboard_widget import KeyboardWidget
from .overlay_window import OverlayWindow

__all__ = [
    'get_layout',
    'get_key_positions',
    'QWERTY',
    'SYMBOLS',
    'KeyboardWidget',
    'OverlayWindow',
]
