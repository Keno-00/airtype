"""
Controller button hint mappings for keyboard keys.

Maps keyboard key labels to controller button icons that should be
displayed as hints in the top-right corner of the key.
"""
from pathlib import Path
from typing import Dict, Optional

# Path to controller icons
ICONS_DIR = Path(__file__).parent.parent.parent / "assets" / "icons" / "controller"

# Mapping of keyboard key labels to icon filenames (without .svg extension)
CONTROLLER_HINTS: Dict[str, str] = {
    # Action buttons (X=backspace, Y=space)
    "SPACE": "xb_y",
    "⌫": "xb_x",           # Backspace
    "BACKSPACE": "xb_x",   # Alternative label
    "SHIFT": "xb_lt",      # Left trigger for shift
    "↵": "xb_start",       # Enter/Send = Start
    "ENTER": "xb_start",   # Alternative label
    # Prediction slots - RT + D-pad combos
    "PRE1": "xb_rt_left",   # RT + Left
    "PRE2": "xb_rt_up",     # RT + Up
    "PRE3": "xb_rt_right",  # RT + Right
}

# Hint for currently highlighted key (A to select)
SELECT_HINT = "xb_a"


def get_hint_icon_path(key_label: str) -> Optional[Path]:
    """
    Get the path to the controller hint icon for a key.
    
    Args:
        key_label: The keyboard key label (e.g., "SPACE", "⌫")
        
    Returns:
        Path to the SVG icon, or None if no hint exists for this key
    """
    icon_name = CONTROLLER_HINTS.get(key_label.upper())
    if icon_name:
        icon_path = ICONS_DIR / f"{icon_name}.svg"
        if icon_path.exists():
            return icon_path
    return None


def get_all_hint_icons() -> Dict[str, Path]:
    """
    Get all available hint icons.
    
    Returns:
        Dict mapping key labels to icon paths
    """
    result = {}
    for key, icon_name in CONTROLLER_HINTS.items():
        icon_path = ICONS_DIR / f"{icon_name}.svg"
        if icon_path.exists():
            result[key] = icon_path
    return result
