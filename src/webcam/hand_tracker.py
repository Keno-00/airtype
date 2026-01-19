"""
MediaPipe Hand Tracker wrapper using the Tasks API.
Handles camera capture and hand landmark detection with performance optimizations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import time
import cv2
import numpy as np
import mediapipe as mp

from .config import Config, CameraConfig, MediaPipeConfig

# MediaPipe Tasks API imports
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


@dataclass
class HandLandmarks:
    """
    Normalized hand landmarks from MediaPipe.
    
    Attributes:
        landmarks: List of 21 (x, y, z) tuples, normalized 0-1
        handedness: 'Left' or 'Right'
        confidence: Detection confidence 0-1
    """
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    confidence: float
    
    # MediaPipe landmark indices for convenience
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def get(self, index: int) -> Tuple[float, float, float]:
        """Get landmark by index."""
        return self.landmarks[index]
    
    @property
    def thumb_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[self.THUMB_TIP]
    
    @property
    def index_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[self.INDEX_TIP]
    
    @property
    def middle_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[self.MIDDLE_TIP]
    
    @property
    def ring_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[self.RING_TIP]
    
    @property
    def pinky_tip(self) -> Tuple[float, float, float]:
        return self.landmarks[self.PINKY_TIP]
    
    @property
    def wrist(self) -> Tuple[float, float, float]:
        return self.landmarks[self.WRIST]
    
    @property
    def palm_center(self) -> Tuple[float, float, float]:
        """Approximate palm center from MCP joints."""
        mcps = [self.landmarks[i] for i in [5, 9, 13, 17]]  # MCP joints
        x = sum(p[0] for p in mcps) / 4
        y = sum(p[1] for p in mcps) / 4
        z = sum(p[2] for p in mcps) / 4
        return (x, y, z)


# Hand connections for drawing (same as MediaPipe's HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),           # Palm
]


class HandTracker:
    """
    MediaPipe hand tracking wrapper with camera management.
    Uses the new MediaPipe Tasks API (0.10+).
    
    Performance optimizations:
    - Lazy initialization of MediaPipe
    - VIDEO mode for frame-by-frame processing with tracking
    - Efficient landmark extraction
    """
    
    # Default model path relative to project root
    DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "hand_landmarker.task"
    
    def __init__(self, config: Config, model_path: Optional[Path] = None):
        """
        Initialize hand tracker.
        
        Args:
            config: AirType configuration
            model_path: Path to hand_landmarker.task model file
        """
        self._camera_config: CameraConfig = config.camera
        self._mp_config: MediaPipeConfig = config.mediapipe
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        
        # Lazy initialization
        self._cap: Optional[cv2.VideoCapture] = None
        self._landmarker: Optional[HandLandmarker] = None
        
        # State
        self._is_running = False
        self._last_frame: Optional[cv2.Mat] = None
        self._frame_count = 0
        self._start_perf: float = 0
        self._last_timestamp_ms: int = -1
    
    def start(self) -> bool:
        """
        Start camera capture and MediaPipe.
        
        Returns:
            True if started successfully, False otherwise.
        """
        if self._is_running:
            return True
        
        # Check model exists
        if not self._model_path.exists():
            print(f"ERROR: Model file not found: {self._model_path}")
            print("Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            return False
        
        # Initialize camera
        self._cap = cv2.VideoCapture(self._camera_config.device_id)
        if not self._cap.isOpened():
            return False
        
        # Configure camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._camera_config.fps)
        
        # Determine delegate (GPU with CPU fallback)
        try:
            delegate = BaseOptions.Delegate.GPU
            print("Attempting GPU delegate for MediaPipe...")
        except AttributeError:
            delegate = None
            print("GPU delegate not available, using CPU")
        
        # Initialize MediaPipe Hand Landmarker with VIDEO mode
        base_opts = BaseOptions(model_asset_path=str(self._model_path))
        if delegate is not None:
            try:
                base_opts = BaseOptions(
                    model_asset_path=str(self._model_path),
                    delegate=delegate
                )
                print("GPU delegate enabled for MediaPipe")
            except Exception as e:
                print(f"GPU delegate failed: {e}, using CPU")
        
        options = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self._mp_config.max_num_hands,
            min_hand_detection_confidence=self._mp_config.min_detection_confidence,
            min_tracking_confidence=self._mp_config.min_tracking_confidence,
        )
        
        self._landmarker = HandLandmarker.create_from_options(options)
        self._start_perf = time.perf_counter()
        self._last_timestamp_ms = -1
        self._is_running = True
        return True
    
    def stop(self) -> None:
        """Stop camera capture and release resources."""
        self._is_running = False
        
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        self._last_frame = None
    
    def get_landmarks(self) -> Optional[HandLandmarks]:
        """
        Capture frame and detect hand landmarks.
        """
        if not self._is_running or self._cap is None or self._landmarker is None:
            return None
        
        # Standard read
        ret, frame = self._cap.read()
        if not ret:
            return None
        
        self._frame_count += 1
        
        # Mirror horizontally
        frame = cv2.flip(frame, 1)
        self._last_frame = frame
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optimization: Mark the image as not writeable (zero-copy)
        rgb_frame.flags.writeable = False
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate strictly monotonic timestamp
        timestamp_ms = int((time.perf_counter() - self._start_perf) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        
        # Process frame
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if not result.hand_landmarks:
            return None
        
        # Extract first hand
        hand_landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0]
        
        # Convert to our format
        landmarks = [
            (lm.x, lm.y, lm.z) 
            for lm in hand_landmarks
        ]
        
        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness.category_name,
            confidence=handedness.score,
        )
    
    def get_frame_with_landmarks(
        self, 
        landmarks: Optional[HandLandmarks] = None,
        black_background: bool = False
    ) -> Optional[cv2.Mat]:
        """
        Get last frame with optional landmark overlay for debugging.
        
        Args:
            landmarks: If provided, draw landmarks on frame.
            black_background: If True, draw on black instead of camera image.
            
        Returns:
            Frame with landmarks drawn, or None if no frame available.
        """
        if self._last_frame is None:
            return None
        
        if black_background:
            frame = np.zeros_like(self._last_frame)
        else:
            frame = self._last_frame.copy()
        
        if landmarks is not None:
            # Draw landmarks
            h, w = frame.shape[:2]
            for i, (x, y, z) in enumerate(landmarks.landmarks):
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                start = landmarks.landmarks[start_idx]
                end = landmarks.landmarks[end_idx]
                start_pos = (int(start[0] * w), int(start[1] * h))
                end_pos = (int(end[0] * w), int(end[1] * h))
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        
        return frame
    
    @property
    def is_running(self) -> bool:
        return self._is_running
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
