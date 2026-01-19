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
    
    # Send None landmarks (hand lost)
    state = recognizer.update(None)
    
    # Verify state reset
    assert recognizer._is_swiping is False
    assert recognizer._swipe_type is None
    assert state.gesture == Gesture.SWIPE_END
