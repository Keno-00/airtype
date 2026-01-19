"""
Gesture recognition from hand landmarks.
Detects pinch, finger count, fist, and swipe gestures.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple
import math

from .config import GestureConfig
from .hand_tracker import HandLandmarks


class Gesture(Enum):
    """Detected gesture types."""
    NONE = auto()
    
    # Pinch gestures (thumb + index)
    PINCH_START = auto()    # Just made pinch
    PINCH_HOLD = auto()     # Maintaining pinch
    PINCH_END = auto()      # Just released pinch
    
    # Finger count (for word selection)
    FINGERS_1 = auto()
    FINGERS_2 = auto()
    FINGERS_3 = auto()
    
    # Fist (for delete word)
    FIST = auto()
    
    # Swipe gestures
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    
    # Interaction states
    SWIPE_START = auto()
    SWIPE_HOLD = auto()
    SWIPE_END = auto()
    
    # Action gestures
    QUICK_CLENCH = auto()   # Fast fist (select word)
    DOUBLE_FIST = auto()    # Two fast fists (caps lock)


@dataclass
class GestureState:
    """Current gesture state with additional info."""
    gesture: Gesture
    pinch_distance: float = 0.0
    extended_fingers: int = 0
    hand_position: Tuple[float, float] = (0.0, 0.0)
    swipe_velocity: Tuple[float, float] = (0.0, 0.0)


class GestureRecognizer:
    """
    Recognizes gestures from hand landmarks.
    
    Gestures detected:
    - Pinch: Thumb tip close to index tip
    - Finger count: Number of extended fingers (1-3)
    - Fist: All fingers curled
    - Swipe: Hand movement while pinching or fisting
    """
    
    def __init__(self, config: GestureConfig):
        """
        Initialize gesture recognizer.
        
        Args:
            config: Gesture detection thresholds
        """
        self._config = config
        
        # Detection state
        self._is_swiping = False
        self._swipe_type = None # "pinch" or "fist"
        
        # Temporal tracking
        self._fist_start_frame = 0
        self._last_fist_end_frame = 0
        self._fist_seq_count = 0
        
        self._pinch_end_grace_counter = 0
        self._pinch_start_confirm_counter = 0
        self._smoothed_pinch_dist: Optional[float] = None
        
        # Swipe tracking
        self._position_history: List[Tuple[float, float, int]] = []  # (x, y, frame)
        self._frame_count = 0
        self._last_swipe_frame = -100
        
        # Previous state for transitions
        self._prev_gesture = Gesture.NONE
        
        # Tracking stability
        self._last_landmarks: Optional[HandLandmarks] = None
        self._hand_lost_counter = 0
        self._HAND_LOST_GRACE_FRAMES = 30  
        
        # Smoothing state (EMA)
        self._smoothed_pos: Optional[Tuple[float, float]] = None
        
        # Relative motion state (Ballistic/Relative mode)
        self._cursor_pos: Tuple[float, float] = (0.5, 0.5)
        self._last_smoothed_pos: Optional[Tuple[float, float]] = None
        self._anchor_pos: Optional[Tuple[float, float]] = None
        self._last_delta: Tuple[float, float] = (0.0, 0.0)
        
    def update(self, landmarks: Optional[HandLandmarks]) -> GestureState:
        """Update gesture detection with smoothing and tracking grace period."""
        self._frame_count += 1
        
        # Handle lost tracking with grace period
        if landmarks is None:
            if self._last_landmarks and self._hand_lost_counter < self._HAND_LOST_GRACE_FRAMES:
                self._hand_lost_counter += 1
                landmarks = self._last_landmarks
            else:
                was_swiping = self._is_swiping
                self._is_swiping = False
                self._position_history.clear()
                self._last_landmarks = None
                self._hand_lost_counter = 0
                self._smoothed_pos = None
                
                if was_swiping:
                    return GestureState(gesture=Gesture.SWIPE_END)
                return GestureState(gesture=Gesture.NONE)
        else:
            self._last_landmarks = landmarks
            self._hand_lost_counter = 0
        
        # Calculate key metrics
        pinch_dist = self._calculate_pinch_distance(landmarks)
        extended = self._count_extended_fingers(landmarks)
        is_fist = self._detect_fist(landmarks)
        thumb_curled = self._is_thumb_curled(landmarks)
        
        # Motion Mapping: Non-linear transformation
        raw_x = landmarks.palm_center[0]
        raw_y = landmarks.palm_center[1]
        
        # 1. Apply EMA Smoothing to raw input
        alpha = self._config.smoothing_factor
        if self._smoothed_pos is None:
            self._smoothed_pos = (raw_x, raw_y)
        else:
            sx, sy = self._smoothed_pos
            self._smoothed_pos = (
                sx * alpha + raw_x * (1.0 - alpha),
                sy * alpha + raw_y * (1.0 - alpha)
            )
        
        # 2. History and history-based detections
        self._position_history.append((raw_x, raw_y, self._frame_count))
        cutoff = self._frame_count - self._config.swipe_max_frames
        self._position_history = [p for p in self._position_history if p[2] > cutoff]
        
        swipe = self._detect_swipe()
        
        # 3. Determine Gesture FIRST so we can reset anchors on transitions
        gesture = self._determine_gesture(
            pinch_dist, extended, is_fist, thumb_curled, swipe
        )
        
        # Reset relative anchor on start of swipe
        if gesture == Gesture.SWIPE_START:
            self._anchor_pos = self._smoothed_pos
            self._last_delta = (0.0, 0.0)
            
        # 4. Map coordinates (Absolute or Relative)
        sx, sy = self._smoothed_pos
        if self._config.motion_mode == "relative":
            palm_pos = self._calculate_relative_motion(sx, sy)
        else:
            palm_pos = self._calculate_absolute_motion(sx, sy)
            
        # Calculate swipe velocity
        velocity = (0.0, 0.0)
        if len(self._position_history) >= 2:
            vx = self._position_history[-1][0] - self._position_history[0][0]
            vy = self._position_history[-1][1] - self._position_history[0][1]
            dt = max(1, self._position_history[-1][2] - self._position_history[0][2])
            velocity = (vx / dt, vy / dt)
        
        return GestureState(
            gesture=gesture,
            pinch_distance=pinch_dist,
            extended_fingers=extended,
            hand_position=palm_pos,
            swipe_velocity=velocity,
        )
    
    def _calculate_absolute_motion(self, sx: float, sy: float) -> Tuple[float, float]:
        """Calculate absolute screen coordinates (0.5 center)."""
        dx = sx - 0.5
        dy = sy - 0.5
        
        exp = self._config.cursor_exponent
        dx_scaled = math.copysign(pow(abs(dx), exp), dx)
        dy_scaled = math.copysign(pow(abs(dy), exp), dy)
        
        gain = self._config.cursor_sensitivity * 1.5
        x = max(0.0, min(1.0, dx_scaled * gain + 0.5))
        y = max(0.0, min(1.0, dy_scaled * gain + 0.5))
        return (x, y)

    def _calculate_relative_motion(self, sx: float, sy: float) -> Tuple[float, float]:
        """
        Calculate relative cursor movement (ballistic).
        Implements deadzone, acceleration from anchor, and bend resets.
        """
        if self._last_smoothed_pos is None:
            self._last_smoothed_pos = (sx, sy)
            self._anchor_pos = (sx, sy)
            return self._cursor_pos

        # 1. Calculate raw delta
        lx, ly = self._last_smoothed_pos
        dx = sx - lx
        dy = sy - ly
        dist = math.sqrt(dx*dx + dy*dy)
        
        # 2. Deadzone and Exponential Scaling
        # Slow movements are scaled down exponentially (pow 1.5)
        # Fast movements maintain linear or higher gain.
        if dist < self._config.motion_deadzone:
            self._last_smoothed_pos = (sx, sy)
            return self._cursor_pos
            
        # Speed scaling: dampen tremors but don't feel "heavy"
        # Using 0.8 power makes it feel much more linear/direct than 0.5
        speed_gain = pow(dist * 100, 0.8) 
        
        # 3. Handle Bend Reset (Sharp turns reset acceleration anchor)
        if self._config.motion_bend_reset:
            curr_delta = (dx, dy)
            if self._last_delta != (0.0, 0.0):
                dot = self._last_delta[0]*dx + self._last_delta[1]*dy
                m1 = math.sqrt(self._last_delta[0]**2 + self._last_delta[1]**2)
                m2 = math.sqrt(dx**2 + dy**2)
                cos_theta = dot / (m1 * m2 + 1e-6)
                
                if cos_theta < 0.7: # Approx 45 degree turn
                    self._anchor_pos = (sx, sy)
            self._last_delta = curr_delta

        # 4. Acceleration (Logarithmic or linear distance from anchor)
        if self._anchor_pos is None:
            self._anchor_pos = (sx, sy)
            
        ax, ay = self._anchor_pos
        dist_from_anchor = math.sqrt((sx - ax)**2 + (sy - ay)**2)
        
        # Gain factor increases as we move further from the anchor
        # Multiply acceleration to make it feel more powerful
        accel = 1.0 + (dist_from_anchor * self._config.motion_accel * 2.0)
        
        # 5. Apply Movement
        total_gain = self._config.cursor_sensitivity * accel * speed_gain
        
        cx, cy = self._cursor_pos
        nx = max(0.0, min(1.0, cx + (dx/dist) * (dist * total_gain)))
        ny = max(0.0, min(1.0, cy + (dy/dist) * (dist * total_gain)))
        
        self._cursor_pos = (nx, ny)
        self._last_smoothed_pos = (sx, sy)
        
        return self._cursor_pos
    
    def _calculate_pinch_distance(self, landmarks: HandLandmarks) -> float:
        """
        Calculate pinch distance normalized by hand width.
        This makes detection scale-invariant (works at any distance from camera).
        
        Returns:
            Pinch distance as a ratio of hand width (0 = touching, 1 = far apart)
        """
        thumb = landmarks.thumb_tip
        index = landmarks.index_tip
        
        # Calculate raw thumb-index distance
        dx = thumb[0] - index[0]
        dy = thumb[1] - index[1]
        dz = thumb[2] - index[2]
        raw_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Calculate hand width (index MCP to pinky MCP) as reference
        index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
        pinky_mcp = landmarks.get(HandLandmarks.PINKY_MCP)
        hand_width = self._distance_2d(index_mcp, pinky_mcp)
        
        # Normalize by hand width (avoid division by zero)
        if hand_width > 0.01:
            return raw_dist / hand_width
        return raw_dist
    
    def _count_extended_fingers(self, landmarks: HandLandmarks) -> int:
        """Count number of extended fingers (excluding thumb)."""
        count = 0
        
        # Check each finger (index, middle, ring, pinky)
        finger_tips = [
            HandLandmarks.INDEX_TIP,
            HandLandmarks.MIDDLE_TIP,
            HandLandmarks.RING_TIP,
            HandLandmarks.PINKY_TIP,
        ]
        finger_pips = [
            HandLandmarks.INDEX_PIP,
            HandLandmarks.MIDDLE_PIP,
            HandLandmarks.RING_PIP,
            HandLandmarks.PINKY_PIP,
        ]
        finger_mcps = [
            HandLandmarks.INDEX_MCP,
            HandLandmarks.MIDDLE_MCP,
            HandLandmarks.RING_MCP,
            HandLandmarks.PINKY_MCP,
        ]
        
        for tip_idx, pip_idx, mcp_idx in zip(finger_tips, finger_pips, finger_mcps):
            tip = landmarks.get(tip_idx)
            pip = landmarks.get(pip_idx)
            mcp = landmarks.get(mcp_idx)
            
            # Finger is extended if tip is above PIP (lower y = higher on screen)
            # Use y-distance relative to finger length
            finger_length = self._distance_2d(mcp, tip)
            tip_to_pip = tip[1] - pip[1]  # Positive if tip is below pip
            
            # Finger is extended if tip is close to or above pip level
            if finger_length > 0:
                ratio = tip_to_pip / finger_length
                if ratio < self._config.finger_curl_threshold - 0.5:
                    count += 1
        
        return count
    
    def _detect_fist(self, landmarks: HandLandmarks) -> bool:
        """Detect if hand is making a fist. Normalized for distance."""
        extended = self._count_extended_fingers(landmarks)
        
        # Calculate reference hand width for normalization
        index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
        pinky_mcp = landmarks.get(HandLandmarks.PINKY_MCP)
        hand_width = self._distance_2d(index_mcp, pinky_mcp)
        
        # Also check thumb
        thumb_tip = landmarks.thumb_tip
        
        # Thumb is curled if tip is close to palm (near index MCP)
        thumb_to_palm = self._distance_2d(thumb_tip, index_mcp)
        
        # Normalize thumb distance by hand width
        if hand_width > 0.01:
            norm_thumb = thumb_to_palm / hand_width
        else:
            norm_thumb = thumb_to_palm
            
        # Fist if no fingers extended and thumb reasonably tucked
        return extended == 0 and norm_thumb < 1.2
    
    def _is_thumb_curled(self, landmarks: HandLandmarks) -> bool:
        """
        Check if thumb is curled/closed.
        Used to distinguish finger count gestures from open hand.
        """
        thumb_tip = landmarks.thumb_tip
        thumb_mcp = landmarks.get(HandLandmarks.THUMB_MCP)
        index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
        
        # Thumb is curled if tip is close to palm (near index MCP)
        # Threshold is strict - thumb must be fully tucked in
        thumb_to_palm = self._distance_2d(thumb_tip, index_mcp)
        return thumb_to_palm < 0.03
    
    def _detect_swipe(self) -> Optional[str]:
        """Detect swipe gesture from position history."""
        if len(self._position_history) < 3:
            return None
        
        # Check cooldown
        if self._frame_count - self._last_swipe_frame < self._config.swipe_cooldown:
            return None
        
        # Only detect horizontal swipe if already swiping
        if not self._is_swiping:
            return None
        
        # Calculate total horizontal movement
        start_x = self._position_history[0][0]
        end_x = self._position_history[-1][0]
        dx = end_x - start_x
        
        if abs(dx) >= self._config.swipe_min_distance:
            self._last_swipe_frame = self._frame_count
            self._position_history.clear()
            return "left" if dx < 0 else "right"
        
        return None
    
    def _determine_gesture(
        self,
        pinch_dist: float,
        extended_fingers: int,
        is_fist: bool,
        thumb_curled: bool,
        swipe: Optional[str],
    ) -> Gesture:
        """Determine primary gesture from all detected features."""
        
        # 1. Smooth the pinch distance signal
        alpha = self._config.pinch_smoothing
        if self._smoothed_pinch_dist is None:
            self._smoothed_pinch_dist = pinch_dist
        else:
            self._smoothed_pinch_dist = (self._smoothed_pinch_dist * alpha + 
                                       pinch_dist * (1.0 - alpha))
        
        spd = self._smoothed_pinch_dist
        
        # 2. Unified Swiping State Logic
        was_swiping = self._is_swiping
        
        # Fist Timing for special gestures
        fist_gesture = Gesture.NONE
        if is_fist:
            if self._fist_start_frame == 0:
                self._fist_start_frame = self._frame_count
        else:
            if self._fist_start_frame > 0:
                duration = self._frame_count - self._fist_start_frame
                # Quick clench: 2 to 12 frames (approx 0.1s - 0.4s)
                if 2 <= duration <= 12:
                    delta_last = self._frame_count - self._last_fist_end_frame
                    if delta_last < 15: # Within 0.5s
                        fist_gesture = Gesture.DOUBLE_FIST
                        self._fist_seq_count = 0
                    else:
                        fist_gesture = Gesture.QUICK_CLENCH
                        self._fist_seq_count = 1
                
                self._last_fist_end_frame = self._frame_count
                self._fist_start_frame = 0

        # Start Swipe?
        if not self._is_swiping:
            if spd < self._config.pinch_threshold:
                self._is_swiping = True
                self._swipe_type = "pinch"
            elif is_fist:
                # To start a swipe with a fist, we wait slightly
                if self._frame_count - self._fist_start_frame > 5:
                    self._is_swiping = True
                    self._swipe_type = "fist"
        
        # End Swipe?
        else:
            if self._swipe_type == "pinch":
                if spd > self._config.pinch_release:
                    self._pinch_end_grace_counter += 1
                    if self._pinch_end_grace_counter >= self._config.pinch_grace:
                        self._is_swiping = False
                        self._pinch_end_grace_counter = 0
                else:
                    self._pinch_end_grace_counter = 0
            elif self._swipe_type == "fist":
                if not is_fist:
                    self._is_swiping = False

        # 3. Determine Final Gesture
        if swipe == "left": return Gesture.SWIPE_LEFT
        if swipe == "right": return Gesture.SWIPE_RIGHT
        if fist_gesture != Gesture.NONE: return fist_gesture
        
        if self._is_swiping:
            if not was_swiping: return Gesture.SWIPE_START
            return Gesture.SWIPE_HOLD
        if was_swiping: return Gesture.SWIPE_END
        
        # Finger count icons...
        
        # Finger count (only when not pinching AND thumb is curled)
        # This prevents open hand from triggering finger gestures
        if thumb_curled:
            if extended_fingers == 1:
                return Gesture.FINGERS_1
            elif extended_fingers == 2:
                return Gesture.FINGERS_2
            elif extended_fingers == 3:
                return Gesture.FINGERS_3
        
        return Gesture.NONE
    
    @staticmethod
    def _distance_2d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate 2D distance between two 3D points (ignoring z)."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def reset(self) -> None:
        """Reset gesture state."""
        self._is_swiping = False
        self._swipe_type = None
        self._position_history.clear()
        self._prev_gesture = Gesture.NONE
