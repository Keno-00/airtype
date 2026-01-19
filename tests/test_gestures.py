import pytest
from src.webcam.gesture_recognizer import GestureRecognizer, Gesture
from src.webcam.config import GestureConfig

@pytest.fixture
def recognizer():
    config = GestureConfig()
    return GestureRecognizer(config)

def test_hand_lost_resets_state(recognizer):
    # Mock a starting swipe
    recognizer._is_swiping = True
    recognizer._swipe_type = "fist"
    recognizer._smoothed_pos = (0.5, 0.5) # Needed to enter synthetic update logic
    
    # Send None landmarks (hand lost) - needs to exceed grace frames (30)
    for _ in range(31):
        state = recognizer.update(None)
    
    # Verify state reset
    assert recognizer._is_swiping is False
    assert recognizer._swipe_type is None
    assert state.gesture == Gesture.SWIPE_END
