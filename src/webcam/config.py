"""
Config loader for AirType.
Loads YAML configuration with dataclass validation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class MediaPipeConfig:
    model_complexity: int = 1
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class GestureConfig:
    pinch_threshold: float = 0.25
    pinch_release: float = 0.45
    pinch_smoothing: float = 0.5   # EMA smoothing for the pinch distance itself
    pinch_grace: int = 5           # Frames of grace before ending pinch
    finger_curl_threshold: float = 0.6
    fist_threshold: float = 0.7
    swipe_min_distance: float = 0.15
    swipe_max_frames: int = 15
    swipe_cooldown: int = 10
    
    # Cursor movement transformation
    cursor_sensitivity: float = 1.0  # Base gain
    cursor_exponent: float = 1.0     # Power (1.0 = linear, 2.0 = quadratic)
    smoothing_factor: float = 0.5    # EMA smoothing (0 = off, 1 = max)
    
    # Motion dynamics (Relative/Delta based)
    motion_mode: str = "absolute"    # "absolute" or "relative"
    motion_deadzone: float = 0.002   # Min movement to register (0-1 scale)
    motion_accel: float = 2.0        # How much gain increases with distance from anchor
    motion_bend_reset: bool = True   # Reset acceleration at bends


@dataclass
class InputConfig:
    mode: str = "webcam"


@dataclass
class KeyboardConfig:
    layout: str = "qwerty"


@dataclass 
class UIConfig:
    position: str = "right"
    theme: str = "default"
    debug_overlay: bool = False


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    input: InputConfig = field(default_factory=InputConfig)
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a dataclass, ignoring unknown keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
                    in project root.
    
    Returns:
        Config dataclass with all settings.
    """
    if config_path is None:
        # Default to config.yaml in project root
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        # Return defaults if no config file
        return Config()
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    return Config(
        camera=_dict_to_dataclass(CameraConfig, data.get('camera')),
        mediapipe=_dict_to_dataclass(MediaPipeConfig, data.get('mediapipe')),
        gestures=_dict_to_dataclass(GestureConfig, data.get('gestures')),
        input=_dict_to_dataclass(InputConfig, data.get('input')),
        keyboard=_dict_to_dataclass(KeyboardConfig, data.get('keyboard')),
        ui=_dict_to_dataclass(UIConfig, data.get('ui')),
    )
