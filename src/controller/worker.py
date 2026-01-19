"""
Background worker for controller input handling.
Runs in a separate QThread to avoid blocking the UI.
"""
import time
from enum import Enum, auto
from typing import Optional, Tuple
from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal

from .gamepad import Gamepad, GamepadState, Button


class ControlMode(Enum):
    """Current controller mode."""
    DPAD_NAV = auto()      # D-pad navigation (default)
    CURSOR = auto()        # RT held - free cursor movement
    SWIPE = auto()         # RT+A held - swipe typing


class NavDirection(Enum):
    """D-pad navigation directions."""
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass
class ControllerGestureState:
    """
    Controller gesture state matching webcam GestureState interface.
    
    This allows the OverlayWindow to use the same signal handlers
    for both webcam and controller input.
    """
    # Matches webcam.GestureState fields
    gesture: str = "NONE"  # SWIPE_START, SWIPE_HOLD, SWIPE_END, NONE
    pinch_distance: float = 0.0
    extended_fingers: int = 0
    hand_position: Tuple[float, float] = (0.5, 0.5)  # Normalized 0-1
    swipe_velocity: Tuple[float, float] = (0.0, 0.0)
    handedness: str = "Controller"
    hand_tilt: float = 0.0
    
    # Controller-specific fields
    mode: ControlMode = ControlMode.DPAD_NAV
    nav_direction: Optional[NavDirection] = None
    button_action: Optional[str] = None  # "SPACE", "BACKSPACE", "SHIFT", "ENTER", "SELECT"


class ControllerWorker(QObject):
    """
    Worker class that handles the controller input loop.
    Emits signals compatible with the webcam worker for UI updates.
    """
    # Signals matching WebcamWorker interface
    gesture_detected = pyqtSignal(object)  # Emits ControllerGestureState
    hand_lost = pyqtSignal()  # Emitted when controller disconnects
    frame_ready = pyqtSignal(object)  # Not used for controller, but kept for interface
    error = pyqtSignal(str)
    
    # Controller-specific signals
    nav_event = pyqtSignal(object)  # Emits NavDirection for D-pad navigation
    button_event = pyqtSignal(str)  # Emits button action string
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._gamepad: Optional[Gamepad] = None
        self._is_running = False
        
        # Control state
        self._mode = ControlMode.DPAD_NAV
        self._cursor_pos = (0.5, 0.5)  # Center of screen
        self._was_swiping = False
        
        # D-pad repeat timing
        self._dpad_held_since: Optional[float] = None
        self._dpad_last_repeat: float = 0.0
        self._last_dpad: Tuple[int, int] = (0, 0)
        
        # Config values with defaults
        ctrl_config = getattr(config, 'controller', None)
        if ctrl_config:
            self._deadzone = getattr(ctrl_config, 'deadzone', 0.15)
            self._sensitivity = getattr(ctrl_config, 'cursor_sensitivity', 2.0)
            self._dpad_repeat_delay = getattr(ctrl_config, 'dpad_repeat_delay', 400) / 1000.0
            self._dpad_repeat_rate = getattr(ctrl_config, 'dpad_repeat_rate', 100) / 1000.0
        else:
            self._deadzone = 0.15
            self._sensitivity = 2.0
            self._dpad_repeat_delay = 0.4
            self._dpad_repeat_rate = 0.1
    
    def start_process(self):
        """Main processing loop. Runs in worker thread."""
        self._gamepad = Gamepad(deadzone=self._deadzone)
        
        if not self._gamepad.connect():
            self.error.emit("No gamepad found or permission denied")
            return
        
        if not self._gamepad.grab():
            self.error.emit("Failed to get exclusive access to gamepad")
            self._gamepad.disconnect()
            return
        
        self._is_running = True
        last_time = time.perf_counter()
        target_fps = 60  # Reduced from 120 for less CPU usage
        min_interval = 1.0 / target_fps
        consecutive_errors = 0
        
        try:
            while self._is_running:
                loop_start = time.perf_counter()
                dt = loop_start - last_time
                last_time = loop_start
                
                # Read gamepad state
                try:
                    state = self._gamepad.update(timeout=0.008)
                    consecutive_errors = 0  # Reset on success
                except OSError as e:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        print(f"Controller disconnected: {e}")
                        self.hand_lost.emit()
                        break
                    time.sleep(0.05)
                    continue
                
                # Process input and update mode
                gesture_state = self._process_input(state, dt)
                
                # Emit gesture state (Skip idle NONE state to reduce traffic)
                if gesture_state.gesture != "NONE" or gesture_state.mode == ControlMode.CURSOR:
                    self.gesture_detected.emit(gesture_state)
                
                # Precise timing
                elapsed = time.perf_counter() - loop_start
                sleep_time = min_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            self.error.emit(f"Controller error: {str(e)}")
        finally:
            self._is_running = False
            if self._gamepad:
                self._gamepad.disconnect()
    
    def stop_process(self):
        """Signal the loop to stop."""
        self._is_running = False
    
    def _process_input(self, state: GamepadState, dt: float) -> ControllerGestureState:
        """
        Process gamepad state and determine current gesture.
        """
        result = ControllerGestureState()
        
        # Check modifier states
        rt_held = state.right_trigger > 0.5
        a_held = state.buttons.get(Button.A, False)
        
        # Determine mode
        if rt_held and a_held:
            new_mode = ControlMode.SWIPE
        elif rt_held:
            new_mode = ControlMode.CURSOR
        else:
            new_mode = ControlMode.DPAD_NAV
        
        result.mode = new_mode
        
        # Handle mode transitions
        if new_mode == ControlMode.SWIPE:
            if self._mode != ControlMode.SWIPE:
                result.gesture = "SWIPE_START"
                self._was_swiping = True
            else:
                result.gesture = "SWIPE_HOLD"
            self._update_cursor(state.left_stick, dt)
        elif self._was_swiping:
            result.gesture = "SWIPE_END"
            self._was_swiping = False
        elif new_mode == ControlMode.CURSOR:
            self._update_cursor(state.left_stick, dt)
            result.gesture = "NONE"
        else:
            result.gesture = "NONE"
            self._process_dpad(state, result)
            self._process_stick_nav(state, result, dt)
        
        self._mode = new_mode
        result.hand_position = self._cursor_pos
        
        # Process RT+D-pad
        if rt_held and state.dpad_changed:
            self._process_rt_dpad(state, result)
        
        # Process button actions
        self._process_buttons(state, result, rt_held)
        
        return result
    
    def _update_cursor(self, stick: Tuple[float, float], dt: float):
        """Update cursor position from thumbstick input."""
        dx = stick[0] * self._sensitivity * dt
        dy = -stick[1] * self._sensitivity * dt  # Invert Y for screen coords
        
        x = max(0.0, min(1.0, self._cursor_pos[0] + dx))
        y = max(0.0, min(1.0, self._cursor_pos[1] + dy))
        
        self._cursor_pos = (x, y)
    
    def _process_dpad(self, state: GamepadState, result: ControllerGestureState):
        """
        Process D-pad for navigation with repeat support.
        
        First press: immediate navigation
        Held: delay, then repeat at rate
        """
        now = time.perf_counter()
        dpad = state.dpad
        
        if dpad != self._last_dpad:
            # D-pad state changed
            self._last_dpad = dpad
            if dpad != (0, 0):
                # New direction pressed
                self._dpad_held_since = now
                self._dpad_last_repeat = now
                result.nav_direction = self._dpad_to_direction(dpad)
                if result.nav_direction:
                    self.nav_event.emit(result.nav_direction)
            else:
                # Released
                self._dpad_held_since = None
                
        elif dpad != (0, 0) and self._dpad_held_since is not None:
            # D-pad held - check for repeat
            held_time = now - self._dpad_held_since
            time_since_repeat = now - self._dpad_last_repeat
            
            if held_time > self._dpad_repeat_delay and time_since_repeat > self._dpad_repeat_rate:
                result.nav_direction = self._dpad_to_direction(dpad)
                if result.nav_direction:
                    self.nav_event.emit(result.nav_direction)
                self._dpad_last_repeat = now
    
    def _dpad_to_direction(self, dpad: Tuple[int, int]) -> Optional[NavDirection]:
        """Convert D-pad state to NavDirection."""
        x, y = dpad
        if y > 0:
            return NavDirection.UP
        elif y < 0:
            return NavDirection.DOWN
        elif x < 0:
            return NavDirection.LEFT
        elif x > 0:
            return NavDirection.RIGHT
        return None
    
    def _process_stick_nav(self, state: GamepadState, result: ControllerGestureState, dt: float):
        """
        Process left stick for discrete navigation in D-pad mode.
        Converts analog stick to discrete nav events with rate limiting.
        """
        # Initialize stick nav state if needed
        if not hasattr(self, '_stick_nav_time'):
            self._stick_nav_time = 0.0
            self._stick_nav_delay = 0.25  # Seconds between nav events
        
        stick = state.left_stick
        threshold = 0.5  # Stick must be pushed past this to trigger
        
        # Check if stick is pushed significantly
        direction = None
        if abs(stick[1]) > abs(stick[0]):  # Vertical dominant
            if stick[1] > threshold:
                direction = NavDirection.UP
            elif stick[1] < -threshold:
                direction = NavDirection.DOWN
        else:  # Horizontal dominant
            if stick[0] > threshold:
                direction = NavDirection.RIGHT
            elif stick[0] < -threshold:
                direction = NavDirection.LEFT
        
        now = time.perf_counter()
        if direction and (now - self._stick_nav_time) > self._stick_nav_delay:
            result.nav_direction = direction
            self.nav_event.emit(direction)
            self._stick_nav_time = now
    
    def _process_buttons(self, state: GamepadState, result: ControllerGestureState, rt_held: bool):
        """Process button presses for actions."""
        for button in state.buttons_pressed:
            action = None
            
            if button == Button.X:
                action = "BACKSPACE"  # X = delete
            elif button == Button.Y:
                action = "SPACE"      # Y = space
            elif button == Button.LB:
                action = "CURSOR_LEFT"  # Move text cursor left
            elif button == Button.RB:
                action = "CURSOR_RIGHT"  # Move text cursor right
            elif button == Button.A and not rt_held:
                # A without RT = select current key
                action = "SELECT"
            elif button == Button.B:
                action = "CLOSE"
            elif button == Button.START:
                action = "SEND"
            
            if action:
                print(f"Controller: Button {button.name} -> {action}")
                result.button_action = action
                self.button_event.emit(action)
        
        # Handle left trigger for shift (analog trigger)
        if state.left_trigger > 0.5:
            if not hasattr(self, '_lt_was_pressed') or not self._lt_was_pressed:
                self._lt_was_pressed = True
                print("Controller: LT -> SHIFT")
                result.button_action = "SHIFT"
                self.button_event.emit("SHIFT")
        else:
            self._lt_was_pressed = False
    
    def _process_rt_dpad(self, state: GamepadState, result: ControllerGestureState):
        """Process RT + D-pad for prediction selection."""
        dpad = state.dpad
        action = None
        
        if dpad[0] < 0:  # Left
            action = "PRE1"
        elif dpad[1] > 0:  # Up
            action = "PRE2"
        elif dpad[0] > 0:  # Right
            action = "PRE3"
        
        if action:
            print(f"Controller: RT+D-pad -> {action}")
            result.button_action = action
            self.button_event.emit(action)
