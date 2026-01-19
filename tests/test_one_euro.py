import pytest
import math
from src.webcam.one_euro_filter import OneEuroFilter
from src.webcam.config import GestureConfig
from src.webcam.gesture_recognizer import GestureRecognizer

def test_one_euro_filter_initialization():
    f = OneEuroFilter(t0=0, x0=10.0, dx0=0.0, min_cutoff=1.0, beta=0.0)
    assert f.x_prev == 10.0
    assert f.dx_prev == 0.0
    assert f.t_prev == 0.0

def test_one_euro_filter_smoothing():
    # beta=0 means simple low-pass filter with fixed cutoff
    f = OneEuroFilter(t0=0, x0=0.0, min_cutoff=1.0, beta=0.0)
    
    # Step response: Input jumps from 0 to 1 at t=0.01
    output = f(t=0.01, x=1.0)
    
    # It should not instantly jump to 1.0
    assert 0 < output < 1.0

def test_one_euro_filter_responsiveness():
    # High beta means more responsive to speed
    f_slow = OneEuroFilter(t0=0, x0=0.0, min_cutoff=0.1, beta=0.0)
    f_fast = OneEuroFilter(t0=0, x0=0.0, min_cutoff=0.1, beta=1.0)
    
    # Simulate a fast move
    t = 0.01
    x = 10.0 # Huge jump
    
    out_slow = f_slow(t, x)
    out_fast = f_fast(t, x)
    
    # The adaptive one (fast) should be closer to input than the slow one
    assert abs(x - out_fast) < abs(x - out_slow)

def test_recognizer_filter_integration():
    config = GestureConfig()
    # Enable filter params if needed, though they have defaults
    
    rec = GestureRecognizer(config)
    
    # Initially filters are None
    assert rec._filter_x is None
    assert rec._filter_y is None
    
    # After first update with landmarks, filters should be initialized
    # We need to mock landmarks or pass something valid-ish if possible, 
    # but GestureRecognizer.update is complex and needs a HandLandmarks object.
    # Alternatively, we can check if it initializes on synthetic update? 
    # No, it needs landmarks to init the filters.
    
    # Let's mock a simple object that mimics HandLandmarks
    class MockLandmarks:
        palm_center = (0.5, 0.5, 0.0)
        handedness = "Right"
        
        @property
        def thumb_tip(self): return (0.5, 0.5, 0.0)
        @property
        def index_tip(self): return (0.5, 0.5, 0.0)
        def get(self, idx): return (0.5, 0.5, 0.0)
        
    rec.update(MockLandmarks())
    
    assert rec._filter_x is not None
    assert rec._filter_y is not None
    assert isinstance(rec._filter_x, OneEuroFilter)
