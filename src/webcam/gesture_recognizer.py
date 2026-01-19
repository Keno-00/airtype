"""
Gesture recognition from hand landmarks.
Detects pinch, finger count, fist, and swipe gestures.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Tuple
import math
import time

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
    handedness: str = "Unknown"
    hand_tilt: float = 0.0  # Radians from vertical


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
        self._fist_active = False # Persistent sticky fist state
        
        # Temporal tracking
        self._fist_start_frame = 0
        self._last_fist_end_frame = 0
        self._fist_seq_count = 0
        
        self._pinch_end_grace_counter = 0
        self._fist_end_grace_counter = 0
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
        self._raw_landmark_ema: Optional[Tuple[float, float]] = None  # Level 0 LPF for jitter
        
        # Spring-damper state for "jelly" cursor feel
        self._spring_pos: Tuple[float, float] = (0.5, 0.5)  # Current rendered position
        self._spring_vel: Tuple[float, float] = (0.0, 0.0)  # Current velocity
        self._last_update_time: Optional[float] = None
        
        # Relative motion state (Ballistic/Relative mode)
        self._cursor_pos: Tuple[float, float] = (0.5, 0.5)
        self._last_smoothed_pos: Optional[Tuple[float, float]] = None
        self._anchor_pos: Optional[Tuple[float, float]] = None
        self._last_delta: Tuple[float, float] = (0.0, 0.0)
        # Stillness lock state (shake filtering)
        self._stillness_center: Optional[Tuple[float, float]] = None
        self._stillness_start: Optional[float] = None
        self._is_locked: bool = False
        self._last_hand_width: float = 0.1
        
        # Projection state for synthetic 120Hz feedback
        self._palm_velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_raw_pos: Optional[Tuple[float, float]] = None
        self._last_raw_time: Optional[float] = None
        self._last_pinch_dist: float = 0.0
        self._last_extended: int = 0
        self._last_is_fist: bool = False
        self._last_thumb_curled: bool = False
        self._last_swipe: bool = False
        self._last_gesture: Gesture = Gesture.NONE
        
        # Orientation & Hand Side
        self._handedness: str = "Unknown"
        self._hand_tilt: float = 0.0  # Radians from vertical (Wrist to Middle MCP)
        
        # Distance-normalization stability
        self._smoothed_hand_width: float = 0.1
        self._fist_confirm_counter = 0
        self._fist_release_counter = 0
        
    def update(self, landmarks: Optional[HandLandmarks]) -> GestureState:
        """
        Update gesture detection. 
        Supports synthetic updates (landmarks=None) by projecting current velocity.
        """
        self._frame_count += 1
        
        now = time.perf_counter()
        
        # 1. Handle Landmarks & Projection
        if landmarks is not None:
            # We have a real update from the camera
            self._hand_lost_counter = 0
            self._last_landmarks = landmarks
            self._handedness = landmarks.handedness
            
            # Calculate tilt (Wrist to Middle MCP)
            self._hand_tilt = self._calculate_hand_tilt(landmarks)
            
            raw_x, raw_y = landmarks.palm_center[0], landmarks.palm_center[1]
            
            # Level 0: Adaptive 1-pole Low Pass Filter for landmark jitter
            # Beta increases with distance (hand_scale) to suppress magnified tremor
            base_beta = self._config.landmark_smoothing
            hand_scale = self._get_hand_scale()
            
            # If scale is 2.5 (very far), beta becomes approx 0.90
            # If scale is 0.5 (very close), beta remains around base (0.6)
            # Farther hands need MUCH more smoothing because 1px error = 10px screen movement
            # Increased cap to 0.98 for extreme distance stability
            adaptive_beta = min(0.98, base_beta + (hand_scale - 1.0) * 0.25)
            
            if self._raw_landmark_ema is None:
                self._raw_landmark_ema = (raw_x, raw_y)
            else:
                ex, ey = self._raw_landmark_ema
                self._raw_landmark_ema = (
                    ex * adaptive_beta + raw_x * (1.0 - adaptive_beta),
                    ey * adaptive_beta + raw_y * (1.0 - adaptive_beta)
                )
            
            # Use filtered landmarks for all subsequent logic
            raw_x, raw_y = self._raw_landmark_ema
            
            # Update key metrics
            pinch_dist = self._calculate_pinch_distance(landmarks)
            extended = self._count_extended_fingers(landmarks)
            is_fist = self._detect_fist(landmarks)
            thumb_curled = self._is_thumb_curled(landmarks)
            
            # Update hand width for distance-normalized sensitivity with LPF
            index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
            pinky_mcp = landmarks.get(HandLandmarks.PINKY_MCP)
            raw_width = self._distance_2d(index_mcp, pinky_mcp)
            
            # Smooth hand width to prevent threshold bouncing
            hw_alpha = 0.8
            self._smoothed_hand_width = (self._smoothed_hand_width * hw_alpha + 
                                       max(0.01, raw_width) * (1.0 - hw_alpha))
            self._last_hand_width = self._smoothed_hand_width
            
            # Store for projection (metrics)
            self._last_pinch_dist = pinch_dist
            self._last_extended = extended
            self._last_is_fist = is_fist
            self._last_thumb_curled = thumb_curled
            
            # Motion Mapping: Apply EMA Smoothing to raw input
            alpha = self._config.smoothing_factor
            if self._smoothed_pos is None:
                self._smoothed_pos = (raw_x, raw_y)
            else:
                lx, ly = self._smoothed_pos
                self._smoothed_pos = (
                    lx * alpha + raw_x * (1.0 - alpha),
                    ly * alpha + raw_y * (1.0 - alpha)
                )
            
            # Update velocity based on SMOOTHED time delta for stability
            if self._last_smoothed_pos is not None and self._last_raw_time is not None:
                dt = now - self._last_raw_time
                if dt > 0:
                    sx, sy = self._smoothed_pos
                    lsx, lsy = self._last_smoothed_pos
                    
                    # Velocity = displacement per second (smoothed)
                    v_max = 2.0 # Cap velocity to prevent "fly-away"
                    vx = max(-v_max, min(v_max, (sx - lsx) / dt))
                    vy = max(-v_max, min(v_max, (sy - lsy) / dt))
                    
                    # Target velocity for synthetic steps (8.3ms)
                    self._palm_velocity = (vx * 0.00833, vy * 0.00833)
                    
            self._last_raw_pos = (raw_x, raw_y)
            self._last_smoothed_pos = self._smoothed_pos
            self._last_raw_time = now
            
            pass # Metrics already updated above
        else:
            # Synthetic update (No new camera frame)
            if self._smoothed_pos is None:
                return GestureState(gesture=Gesture.NONE)
            
            # Grace period check
            self._hand_lost_counter += 1
            if self._hand_lost_counter > self._HAND_LOST_GRACE_FRAMES:
                self._smoothed_pos = None
                self._last_raw_pos = None
                return GestureState(gesture=Gesture.NONE)
            
            # PROJECT raw position based on velocity
            # We decay velocity slightly to avoid infinite drifting
            vx, vy = self._palm_velocity
            self._palm_velocity = (vx * 0.8, vy * 0.8) # Stronger decay
            
            sx, sy = self._smoothed_pos
            self._smoothed_pos = (sx + self._palm_velocity[0], sy + self._palm_velocity[1])
            
            # Use last known metrics
            pinch_dist = self._last_pinch_dist
            extended = self._last_extended
            is_fist = self._last_is_fist
            thumb_curled = self._last_thumb_curled
            landmarks = self._last_landmarks # Still used for hand width scaling etc

        # 2. History and history-based detections
        curr_raw = self._smoothed_pos # Use smoothed for history in synthetic mode
        self._position_history.append((curr_raw[0], curr_raw[1], self._frame_count))
        cutoff = self._frame_count - self._config.swipe_max_frames
        self._position_history = [p for p in self._position_history if p[2] > cutoff]
        
        swipe = self._detect_swipe()
        
        # 3. Determine Gesture
        gesture = self._determine_gesture(
            pinch_dist, extended, is_fist, thumb_curled, swipe
        )
        
        # Reset relative anchor on start of swipe
        if gesture == Gesture.SWIPE_START:
            self._anchor_pos = self._smoothed_pos
            self._last_delta = (0.0, 0.0)
            self._is_swiping = True
        elif gesture == Gesture.SWIPE_END:
            self._is_swiping = False
        
        # 4. Map coordinates (Absolute or Relative)
        sx, sy = self._smoothed_pos
        if self._config.motion_mode == "relative":
            target_pos = self._calculate_relative_motion(sx, sy)
        else:
            target_pos = self._calculate_absolute_motion(sx, sy)
        
        # 5. Hand width for scaling
        if landmarks:
            index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
            pinky_mcp = landmarks.get(HandLandmarks.PINKY_MCP)
            hand_width = self._distance_2d(index_mcp, pinky_mcp)
            if hand_width > 0.01:
                self._last_hand_width = hand_width
        
        # 6. Apply stillness lock & Physics
        target_pos = self._apply_stillness_lock(target_pos, self._last_hand_width)
        palm_pos = self._apply_spring_physics(target_pos)
            
        return GestureState(
            gesture=gesture,
            pinch_distance=pinch_dist,
            extended_fingers=extended,
            hand_position=palm_pos,
            swipe_velocity=self._palm_velocity,
            handedness=self._handedness,
            hand_tilt=self._hand_tilt,
        )
    
    def _calculate_hand_tilt(self, landmarks: HandLandmarks) -> float:
        """Calculate hand tilt (angle from vertical) based on wrist-to-palm vector."""
        wrist = landmarks.get(HandLandmarks.WRIST)
        middle_mcp = landmarks.get(HandLandmarks.MIDDLE_MCP)
        
        # Vector from wrist to middle MCP
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        
        # Angle in radians. 0 is up (negative y), pi is down
        # Image coordinates: y increases downwards
        return math.atan2(dx, -dy)
    
    def _apply_spring_physics(self, target: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply spring-damper physics for jelly-like cursor movement.
        Uses a critically-damped or under-damped spring system for smooth trailing.
        
        The physics equation: F = -k*(pos - target) - c*vel
        Where k = spring stiffness, c = damping coefficient
        """
        import time
        
        current_time = time.time()
        if self._last_update_time is None:
            self._last_update_time = current_time
            self._spring_pos = target
            return target
        
        # Calculate delta time (clamped to prevent physics explosion on lag spikes)
        dt = min(current_time - self._last_update_time, 0.1)  # Max 100ms step
        self._last_update_time = current_time
        
        if dt <= 0:
            return self._spring_pos
        
        # Get physics parameters
        k = self._config.cursor_spring   # Spring stiffness
        c = self._config.cursor_damping  # Damping coefficient  
        m = self._config.cursor_mass     # Mass
        
        # Sub-stepping (Run 4 integration steps per frame for stability)
        steps = 4
        sub_dt = dt / steps
        
        px, py = self._spring_pos
        vx, vy = self._spring_vel
        tx, ty = target
        
        for _ in range(steps):
            # Calculate spring force: F = -k * (pos - target) - c * vel
            # Acceleration: a = F / m
            fx = -k * (px - tx) - c * vx
            fy = -k * (py - ty) - c * vy
            ax = fx / m
            ay = fy / m
            
            # Semi-implicit Euler integration (more stable than explicit)
            vx = vx + ax * sub_dt
            vy = vy + ay * sub_dt
            px = px + vx * sub_dt
            py = py + vy * sub_dt
            
            # Clamp to screen bounds during sub-steps
            px = max(0.0, min(1.0, px))
            py = max(0.0, min(1.0, py))
            
        self._spring_pos = (px, py)
        self._spring_vel = (vx, vy)
        
        return self._spring_pos
    
    def _apply_stillness_lock(
        self, 
        target: Tuple[float, float], 
        hand_width: float
    ) -> Tuple[float, float]:
        """
        Apply stillness lock for shake filtering.
        Lock cursor when hand is stationary, require significant movement to unlock.
        All thresholds are scaled by hand_width for distance-invariance.
        """
        # Bypass if disabled in config
        if not self._config.use_stillness_lock:
            self._is_locked = False
            return target
            
        # Scale thresholds by hand width
        stillness_radius = self._config.stillness_radius * hand_width
        unlock_distance = self._config.unlock_distance * hand_width
        
        now = time.perf_counter()
        
        # Initialize stillness center if not set
        if self._stillness_center is None:
            self._stillness_center = target
            self._stillness_start = now
            return target
        
        # Calculate distance from stillness center
        dx = target[0] - self._stillness_center[0]
        dy = target[1] - self._stillness_center[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if self._is_locked:
            # Locked: require significant movement to unlock
            if dist > unlock_distance:
                self._is_locked = False
                self._stillness_center = target
                self._stillness_start = now
                return target
            else:
                # Stay locked at center
                return self._stillness_center
        else:
            # Not locked: check for stillness
            if dist < stillness_radius:
                # Within stillness zone
                elapsed = now - (self._stillness_start or now)
                if elapsed > self._config.stillness_time:
                    # Lock the cursor
                    self._is_locked = True
                    return self._stillness_center
                else:
                    # Still counting down, allow movement
                    return target
            else:
                # Moved outside stillness zone, reset timer
                self._stillness_center = target
                self._stillness_start = now
                return target
    
    def _calculate_absolute_motion(self, sx: float, sy: float) -> Tuple[float, float]:
        """Calculate absolute screen coordinates (0.5 center)."""
        dx = sx - 0.5
        dy = sy - 0.5
        
        # Apply hand-size normalization (Reference width = 0.1)
        # This ensures that standing further away doesn't decrease reach.
        hand_scale = self._get_hand_scale()
        
        exp = self._config.cursor_exponent
        dx_scaled = math.copysign(pow(abs(dx), exp), dx)
        dy_scaled = math.copysign(pow(abs(dy), exp), dy)
        
        # Calibration: Tone down multiplier slightly for long-path precision
        gain = self._config.cursor_sensitivity * hand_scale * 1.2
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
        hand_scale = self._get_hand_scale()
        total_gain = self._config.cursor_sensitivity * accel * hand_scale
        
        cx, cy = self._cursor_pos
        nx = max(0.0, min(1.0, cx + dx * total_gain))
        ny = max(0.0, min(1.0, cy + dy * total_gain))
        
        self._cursor_pos = (nx, ny)
        self._last_smoothed_pos = (sx, sy)
        
    def _get_hand_scale(self) -> float:
        """Calculate normalization scale based on hand size (Ref width = 0.1)."""
        # Smaller hand (further away) -> Higher scale (more reach)
        # Larger hand (closer) -> Lower scale (less tremors)
        scale = 0.1 / max(0.01, self._last_hand_width)
        return max(0.5, min(2.5, scale))
    
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
        """
        Count number of extended fingers (excluding thumb).
        Uses projection onto hand "forward" axis (Wrist to Middle MCP).
        """
        count = 0
        wrist = landmarks.get(HandLandmarks.WRIST)
        middle_mcp = landmarks.get(HandLandmarks.MIDDLE_MCP)
        
        # Hand axis vector
        ax, ay = middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]
        mag_a = math.sqrt(ax*ax + ay*ay)
        if mag_a < 0.001: return 0
        
        # Unit forward vector
        fux, fuy = ax/mag_a, ay/mag_a
        
        # Check each finger tip (index, middle, ring, pinky)
        finger_tips_indices = [8, 12, 16, 20]
        
        for tip_idx in finger_tips_indices:
            tip = landmarks.get(tip_idx)
            # Vector from wrist to tip
            tx, ty = tip[0] - wrist[0], tip[1] - wrist[1]
            
            # Projection length onto forward axis
            projection = tx * fux + ty * fuy
            
            # Adaptive threshold: Farther hands (higher scale) have harder time producing 
            # large projections due to resolution loss. 
            # We ease the threshold as scale increases.
            hand_scale = self._get_hand_scale()
            # scale=1.0 -> 1.35x
            # scale=2.5 -> approx 1.25x (Easier than 1.35x, but tight enough to ignore noise)
            adaptive_thresh = max(1.25, 1.35 - (hand_scale - 1.0) * 0.1)
            
            if projection > mag_a * adaptive_thresh:
                count += 1
        
        return count
    
    def _detect_fist(self, landmarks: HandLandmarks) -> bool:
        """
        Detect if hand is making a fist with hysteresis (Sticky Fist).
        """
        extended = self._count_extended_fingers(landmarks)
        thumb_curled = self._is_thumb_curled(landmarks)
        
        # Activation: Stricter (0 fingers + thumb curled)
        # Added a 3-frame confirmation buffer to prevent false-positives from jitter
        if extended == 0 and thumb_curled:
            self._fist_confirm_counter += 1
            if self._fist_confirm_counter >= 3:
                self._fist_active = True
                self._fist_release_counter = 0 # Reset release counter
        else:
            self._fist_confirm_counter = 0
        
        # Deactivation: Requires intentional open hand (3+ fingers by default)
        # Added a 3-frame release buffer to prevent flickering out of fist at density
        if extended >= self._config.fist_release_fingers:
            self._fist_release_counter += 1
            if self._fist_release_counter >= 3:
                self._fist_active = False
                self._fist_release_counter = 0
        else:
            self._fist_release_counter = 0
            
        return self._fist_active

    def _is_thumb_curled(self, landmarks: HandLandmarks) -> bool:
        """
        Check if thumb is curled/closed using lateral projection.
        """
        thumb_tip = landmarks.thumb_tip
        index_mcp = landmarks.get(HandLandmarks.INDEX_MCP)
        pinky_mcp = landmarks.get(HandLandmarks.PINKY_MCP)
        
        # Lateral axis: Index MCP to Pinky MCP
        lx, ly = pinky_mcp[0] - index_mcp[0], pinky_mcp[1] - index_mcp[1]
        mag_l = math.sqrt(lx*lx + ly*ly)
        if mag_l < 0.001: return False
        
        # Vector from Index MCP to Thumb Tip
        tx, ty = thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1]
        
        # Unit lateral vector
        lux, luy = lx/mag_l, ly/mag_l
        
        # Projection onto lateral axis (how far thumb moved towards pinky side)
        projection = tx * lux + ty * luy
        
        # Threshold normalized by hand width
        return projection > mag_l * 0.35
    
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
                    self._fist_end_grace_counter += 1
                    if self._fist_end_grace_counter >= self._config.fist_release_grace:
                        self._is_swiping = False
                        self._fist_end_grace_counter = 0
                else:
                    self._fist_end_grace_counter = 0

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
