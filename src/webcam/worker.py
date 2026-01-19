"""
Background worker for MediaPipe hand tracking and gesture recognition.
Runs in a separate QThread to avoid blocking the UI.
"""
import time
import threading
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from .hand_tracker import HandTracker
from .gesture_recognizer import GestureRecognizer, GestureState, Gesture

class WebcamWorker(QObject):
    """
    Worker class that handles the MediaPipe processing loop.
    Emits signals for UI updates.
    """
    # Signals
    gesture_detected = pyqtSignal(object)  # Emits GestureState
    hand_lost = pyqtSignal()
    frame_ready = pyqtSignal(object)  # Emits numpy array (BGR frame with landmarks)
    error = pyqtSignal(str)
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._tracker: Optional[HandTracker] = None
        self._recognizer: Optional[GestureRecognizer] = None
        self._is_running = False
        
        # Threading for 120Hz decoupled feedback
        self._latest_landmarks: Optional[object] = None
        self._landmarks_lock = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        
    def _capture_loop(self):
        """Background thread to pull camera frames as fast as possible."""
        while self._is_running:
            try:
                landmarks = self._tracker.get_landmarks()
                with self._landmarks_lock:
                    self._latest_landmarks = landmarks
            except Exception as e:
                print(f"Capture thread error: {e}")
                time.sleep(0.1) # Cool down on error

    def start_process(self):
        """Main processing loop. Runs in worker thread at 120Hz."""
        self._tracker = HandTracker(self._config)
        self._recognizer = GestureRecognizer(self._config.gestures)
        
        if not self._tracker.start():
            self.error.emit("Could not open camera")
            return
            
        self._is_running = True
        
        # Start the background capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        last_emit_time = 0
        last_frame_time = 0
        target_fps = 120 # Synthetic 120Hz target
        frame_fps = 5    # Very low FPS for landmarks preview
        min_interval = 1.0 / target_fps
        frame_interval = 1.0 / frame_fps
        
        try:
            while self._is_running:
                loop_start = time.perf_counter()
                
                # 1. Get latest landmarks (if any) from capture thread
                with self._landmarks_lock:
                    landmarks = self._latest_landmarks
                    self._latest_landmarks = None # Consume it
                
                # 2. Update with interpolation (landmarks might be None)
                state = self._recognizer.update(landmarks)
                
                now = time.perf_counter()
                
                # 3. Emit gesture state at 120Hz
                if landmarks is not None:
                    # We had a real update, emit immediately
                    self.gesture_detected.emit(state)
                    last_emit_time = now
                elif now - last_emit_time >= min_interval:
                    # Synthetic update (interpolated)
                    self.gesture_detected.emit(state)
                    last_emit_time = now
                
                # 4. Emit webcam frame only if preview is enabled
                if self._config.gestures.show_preview:
                    # Note: landmarks variable here is the one we just consumed (might be None)
                    # We use the raw landmarks from the tracker if needed, but for performance
                    # we only draw when we actually have a fresh detection.
                    if now - last_frame_time >= frame_interval and landmarks is not None:
                        frame = self._tracker.get_frame_with_landmarks(landmarks, black_background=True)
                        if frame is not None:
                            self.frame_ready.emit(frame)
                        last_frame_time = now
                
                # 5. Precise timing to hit 120Hz
                elapsed = time.perf_counter() - loop_start
                sleep_time = min_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")
        finally:
            self._is_running = False
            if self._capture_thread:
                self._capture_thread.join(timeout=1.0)
            if self._tracker:
                self._tracker.stop()
                
    def stop_process(self):
        """Signal the loop to stop and release resources in worker thread."""
        self._is_running = False
