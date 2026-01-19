"""
Keyboard layout definitions.
Support for QWERTY, DVORAK, and COLEMAK layouts.
"""
from typing import List, Dict, Tuple


# Key layout as rows of keys
# Each key is a string (letter) or tuple (label, width_multiplier)

# Key definitions: just a string for normal letters,
# or (label, width_multiplier, is_special) for functional keys.

QWERTY = [
    [('PRE1', 3.3, True), ('PRE2', 3.4, True), ('PRE3', 3.3, True)],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    [('SHIFT', 1.5, True), 'Z', 'X', 'C', 'V', 'B', 'N', 'M', ('BACKSPACE', 1.5, True)],
    [('?123', 1.5, True), ',', ('SPACE', 4.0, False), '.', ('CLOSE', 1.5, True)],
]

SYMBOLS = [
    [('PRE1', 3.3, True), ('PRE2', 3.4, True), ('PRE3', 3.3, True)],
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['@', '#', '$', '_', '&', '-', '+', '(', ')', '/'],
    [('SHIFT', 1.5, True), '*', '\"', '\'', ':', ';', '!', '?', ('BACKSPACE', 1.5, True)],
    [('ABC', 1.5, True), ',', ('SPACE', 4.0, False), '.', ('CLOSE', 1.5, True)],
]

LAYOUTS: Dict[str, List[List[object]]] = {
    'qwerty': QWERTY,
    'symbols': SYMBOLS,
}


def get_layout(name: str) -> List[List[object]]:
    """Get keyboard layout by name."""
    return LAYOUTS.get(name.lower(), QWERTY)


def get_key_positions(layout: List[List[object]]) -> Dict[str, Tuple[float, float]]:
    """
    Get normalized (0-1) center positions for each key.
    Accounts for width multipliers.
    """
    positions = {}
    num_rows = len(layout)
    
    # First, calculate total width of each row in terms of "units"
    row_widths = []
    for row in layout:
        row_w = 0.0
        for key in row:
            if isinstance(key, tuple):
                row_w += key[1]
            else:
                row_w += 1.0
        row_widths.append(row_w)
    
    max_row_w = max(row_widths)
    
    for row_idx, row in enumerate(layout):
        row_w = row_widths[row_idx]
        # X offset for centering shorter rows
        x_offset = (max_row_w - row_w) / 2.0
        
        current_x = x_offset
        for key in row:
            char = key[0] if isinstance(key, tuple) else key
            width = key[1] if isinstance(key, tuple) else 1.0
            
            center_x = (current_x + width/2.0) / max_row_w
            center_y = (row_idx + 0.5) / num_rows
            
            positions[char.upper()] = (center_x, center_y)
            current_x += width
            
    return positions
