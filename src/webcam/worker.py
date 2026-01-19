"""
Background worker for MediaPipe hand tracking and gesture recognition.
Runs in a separate QThread to avoid blocking the UI.
"""
import time
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
    error = pyqtSignal(str)
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self._config = config
        self._tracker: Optional[HandTracker] = None
        self._recognizer: Optional[GestureRecognizer] = None
        self._is_running = False
        
    def start_process(self):
        """Main processing loop. Runs in worker thread."""
        # CRITICAL: Initialize MediaPipe in the worker thread for thread affinity
        self._tracker = HandTracker(self._config)
        self._recognizer = GestureRecognizer(self._config.gestures)
        
        if not self._tracker.start():
            self.error.emit("Could not open camera")
            return
            
        self._is_running = True
        last_emit_time = 0
        target_fps = 60
        min_interval = 1.0 / target_fps
        
        try:
            while self._is_running:
                landmarks = self._tracker.get_landmarks()
                state = self._recognizer.update(landmarks)
                
                now = time.perf_counter()
                # Throttled UI emission (max 60 FPS) to avoid flooding main thread
                if now - last_emit_time >= min_interval:
                    if landmarks:
                        self.gesture_detected.emit(state)
                    else:
                        self.hand_lost.emit()
                    last_emit_time = now
                    
                # Small yield to OS
                time.sleep(0.001) 
                
        except Exception as e:
            self.error.emit(f"Worker Exception: {str(e)}")
        finally:
            self._is_running = False
            if self._tracker:
                self._tracker.stop()
                
    def stop_process(self):
        """Signal the loop to stop and release resources in worker thread."""
        self._is_running = False
