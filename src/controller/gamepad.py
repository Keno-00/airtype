"""
Gamepad handling using python-evdev.

Provides auto-detection, exclusive grab, and input reading for
Xbox, PlayStation, and generic USB/Bluetooth gamepads.
"""
import select
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import IntEnum

try:
    import evdev
    from evdev import ecodes, InputDevice, categorize
    EVDEV_AVAILABLE = True
except ImportError:
    EVDEV_AVAILABLE = False
    evdev = None
    ecodes = None


class Button(IntEnum):
    """Xbox-style button naming for cross-platform consistency."""
    A = 0       # BTN_SOUTH (Xbox A / PS Cross)
    B = 1       # BTN_EAST (Xbox B / PS Circle)
    X = 2       # BTN_WEST (Xbox X / PS Square but often swapped)
    Y = 3       # BTN_NORTH (Xbox Y / PS Triangle)
    LB = 4      # BTN_TL (Left Bumper)
    RB = 5      # BTN_TR (Right Bumper)
    SELECT = 6  # BTN_SELECT
    START = 7   # BTN_START
    HOME = 8    # BTN_MODE
    L3 = 9      # BTN_THUMBL (Left Stick Click)
    R3 = 10     # BTN_THUMBR (Right Stick Click)


# Evdev button code to Button enum mapping
# Different controllers use different codes, so we map all common variants
BUTTON_MAP = {
    # BTN_SOUTH / BTN_A / BTN_GAMEPAD (304)
    304: Button.A,
    # BTN_EAST / BTN_B (305)
    305: Button.B,
    # BTN_C (306) - some controllers use this for X
    306: Button.X,
    # BTN_NORTH / BTN_X (307) - Xbox 360 uses this for X
    307: Button.X,
    # BTN_WEST / BTN_Y (308) - Xbox 360 uses this for Y
    308: Button.Y,
    # BTN_TL (310)
    310: Button.LB,
    # BTN_TR (311)
    311: Button.RB,
    # BTN_SELECT (314)
    314: Button.SELECT,
    # BTN_START (315)
    315: Button.START,
    # BTN_MODE (316)
    316: Button.HOME,
    # BTN_THUMBL (317)
    317: Button.L3,
    # BTN_THUMBR (318)
    318: Button.R3,
}


@dataclass
class GamepadState:
    """Current state of all gamepad inputs."""
    # Thumbsticks: -1.0 to 1.0 (left/down = negative, right/up = positive)
    left_stick: Tuple[float, float] = (0.0, 0.0)   # (x, y)
    right_stick: Tuple[float, float] = (0.0, 0.0)  # (x, y)
    
    # Triggers: 0.0 to 1.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    
    # D-pad: -1, 0, or 1 for each axis
    dpad: Tuple[int, int] = (0, 0)  # (x, y) where up = 1, down = -1
    
    # Button states: True if pressed
    buttons: Dict[Button, bool] = field(default_factory=lambda: {b: False for b in Button})
    
    # Events this frame
    buttons_pressed: List[Button] = field(default_factory=list)
    buttons_released: List[Button] = field(default_factory=list)
    dpad_changed: bool = False


def find_gamepad() -> Optional[str]:
    """
    Auto-detect the first connected gamepad.
    
    Looks for devices with ABS_X, ABS_Y axes and common gamepad buttons.
    Returns the device path (e.g., '/dev/input/event5') or None.
    """
    if not EVDEV_AVAILABLE:
        return None
        
    for path in evdev.list_devices():
        try:
            device = InputDevice(path)
            caps = device.capabilities()
            
            # Check for absolute axes (thumbsticks)
            if ecodes.EV_ABS not in caps:
                continue
                
            abs_codes = [code for code, _ in caps[ecodes.EV_ABS]] if caps[ecodes.EV_ABS] else []
            
            # Must have X and Y axes
            if ecodes.ABS_X not in abs_codes or ecodes.ABS_Y not in abs_codes:
                continue
            
            # Check for gamepad buttons
            if ecodes.EV_KEY not in caps:
                continue
            
            key_codes = caps[ecodes.EV_KEY]
            
            # Look for BTN_SOUTH (A button) or BTN_GAMEPAD as indicator
            gamepad_buttons = {304, 305, 307, 308, 310, 311}  # Common gamepad buttons
            if any(code in key_codes for code in gamepad_buttons):
                print(f"Found gamepad: {device.name} at {path}")
                return path
                
        except (PermissionError, OSError):
            continue
    
    return None


class Gamepad:
    """
    Gamepad input handler with exclusive grab support.
    
    Usage:
        gamepad = Gamepad()
        if gamepad.connect():
            gamepad.grab()  # Exclusive access
            try:
                while running:
                    state = gamepad.update()
                    # Process state...
            finally:
                gamepad.ungrab()
                gamepad.disconnect()
    """
    
    def __init__(self, device_path: Optional[str] = None, deadzone: float = 0.15):
        """
        Initialize gamepad.
        
        Args:
            device_path: Specific device path, or None to auto-detect
            deadzone: Thumbstick deadzone (0.0 - 1.0)
        """
        self._device_path = device_path
        self._device: Optional[InputDevice] = None
        self._grabbed = False
        self._deadzone = deadzone
        
        # Axis info for normalization
        self._axis_info: Dict[int, Tuple[int, int]] = {}  # code -> (min, max)
        
        # Current state
        self._state = GamepadState()
        
        # Raw axis values (before normalization)
        self._raw_axes: Dict[int, int] = {}
    
    @property
    def connected(self) -> bool:
        """Check if gamepad is connected."""
        return self._device is not None
    
    @property
    def grabbed(self) -> bool:
        """Check if we have exclusive access."""
        return self._grabbed
    
    @property
    def name(self) -> str:
        """Get device name."""
        return self._device.name if self._device else "No device"
    
    def connect(self) -> bool:
        """
        Connect to gamepad device.
        
        Returns:
            True if connected successfully
        """
        if not EVDEV_AVAILABLE:
            print("ERROR: python-evdev not installed")
            return False
        
        path = self._device_path or find_gamepad()
        if not path:
            print("ERROR: No gamepad found")
            return False
        
        try:
            self._device = InputDevice(path)
            self._device_path = path
            
            # Cache axis info for normalization
            caps = self._device.capabilities()
            if ecodes.EV_ABS in caps:
                for code, absinfo in caps[ecodes.EV_ABS]:
                    self._axis_info[code] = (absinfo.min, absinfo.max)
                    self._raw_axes[code] = absinfo.value  # Initial value
            
            print(f"Connected to: {self._device.name}")
            return True
            
        except (PermissionError, OSError) as e:
            print(f"ERROR: Cannot open gamepad: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from gamepad."""
        if self._grabbed:
            self.ungrab()
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
    
    def grab(self) -> bool:
        """
        Grab exclusive access to the gamepad.
        Other applications will not receive input from this device.
        
        Returns:
            True if grabbed successfully
        """
        if not self._device:
            return False
        
        try:
            self._device.grab()
            self._grabbed = True
            print(f"Exclusive grab acquired on: {self._device.name}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to grab device: {e}")
            return False
    
    def ungrab(self):
        """Release exclusive access to the gamepad."""
        if self._device and self._grabbed:
            try:
                self._device.ungrab()
                print(f"Released exclusive grab on: {self._device.name}")
            except Exception as e:
                print(f"Warning: Error releasing grab: {e}")
            self._grabbed = False
    
    def update(self, timeout: float = 0.0) -> GamepadState:
        """
        Read and process pending events.
        
        Args:
            timeout: Maximum time to wait for events (0 = non-blocking)
            
        Returns:
            Current GamepadState
        """
        if not self._device:
            return self._state
        
        # Clear per-frame events
        self._state.buttons_pressed.clear()
        self._state.buttons_released.clear()
        self._state.dpad_changed = False
        
        # Use select for non-blocking read
        r, _, _ = select.select([self._device.fd], [], [], timeout)
        
        if r:
            try:
                for event in self._device.read():
                    self._process_event(event)
            except BlockingIOError:
                pass
            except OSError as e:
                print(f"Gamepad read error: {e}")
        
        return self._state
    
    def _process_event(self, event):
        """Process a single evdev event."""
        if event.type == ecodes.EV_ABS:
            self._process_axis(event.code, event.value)
        elif event.type == ecodes.EV_KEY:
            self._process_button(event.code, event.value)
    
    def _process_axis(self, code: int, value: int):
        """Process an axis event (thumbstick, trigger, dpad)."""
        self._raw_axes[code] = value
        
        # Normalize to -1.0 to 1.0 range
        if code in self._axis_info:
            min_val, max_val = self._axis_info[code]
            center = (min_val + max_val) / 2
            half_range = (max_val - min_val) / 2
            
            if half_range > 0:
                normalized = (value - center) / half_range
            else:
                normalized = 0.0
        else:
            normalized = 0.0
        
        # Apply deadzone
        if abs(normalized) < self._deadzone:
            normalized = 0.0
        
        # Map to appropriate state field
        if code == ecodes.ABS_X:
            self._state.left_stick = (normalized, self._state.left_stick[1])
        elif code == ecodes.ABS_Y:
            # Invert Y so up is positive
            self._state.left_stick = (self._state.left_stick[0], -normalized)
        elif code == ecodes.ABS_RX:
            self._state.right_stick = (normalized, self._state.right_stick[1])
        elif code == ecodes.ABS_RY:
            self._state.right_stick = (self._state.right_stick[0], -normalized)
        elif code == ecodes.ABS_Z:
            # Left trigger (0 to 1)
            self._state.left_trigger = max(0.0, normalized + 1.0) / 2.0
        elif code == ecodes.ABS_RZ:
            # Right trigger (0 to 1)
            self._state.right_trigger = max(0.0, normalized + 1.0) / 2.0
        elif code == ecodes.ABS_HAT0X:
            # D-pad X (-1 = left, 0 = center, 1 = right)
            old_dpad = self._state.dpad
            self._state.dpad = (value, old_dpad[1])
            if self._state.dpad != old_dpad:
                self._state.dpad_changed = True
        elif code == ecodes.ABS_HAT0Y:
            # D-pad Y (-1 = up, 0 = center, 1 = down) - invert so up is positive
            old_dpad = self._state.dpad
            self._state.dpad = (old_dpad[0], -value)
            if self._state.dpad != old_dpad:
                self._state.dpad_changed = True
    
    def _process_button(self, code: int, value: int):
        """Process a button event."""
        if code in BUTTON_MAP:
            button = BUTTON_MAP[code]
            was_pressed = self._state.buttons.get(button, False)
            is_pressed = value > 0
            
            self._state.buttons[button] = is_pressed
            
            if is_pressed and not was_pressed:
                self._state.buttons_pressed.append(button)
            elif not is_pressed and was_pressed:
                self._state.buttons_released.append(button)
        elif value > 0:
            # Debug: print unmapped button codes
            print(f"Unmapped button code: {code}")
    
    def __enter__(self):
        """Context manager entry - connect and grab."""
        if self.connect():
            self.grab()
        return self
    
    def __exit__(self, *args):
        """Context manager exit - ungrab and disconnect."""
        self.disconnect()
