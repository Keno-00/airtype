
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import math
from prediction.engine import PredictionEngine
from ui.layouts import get_layout, get_key_positions

def test_prediction():
    layout = get_layout("qwerty")
    positions = get_key_positions(layout)
    dict_path = Path(__file__).parent / "dictionary.json"
    
    engine = PredictionEngine(dict_path, positions)
    
    # Mock path for "THE"
    # T: (0.5, 0.16) approx
    # H: (0.55, 0.5) approx
    # E: (0.25, 0.16) approx
    t_pos = positions["T"]
    h_pos = positions["H"]
    e_pos = positions["E"]
    
    # Path: start at T, go to H, end at E
    path = [t_pos, h_pos, e_pos]
    
    predictions = engine.predict(path)
    print(f"Path for 'THE' -> Features: T, H, E")
    print(f"Top predictions: {predictions}")
    
    if "THE" in predictions:
        print("SUCCESS: 'THE' found in predictions")
    else:
        print("FAILURE: 'THE' not found")

    # Mock path for "HI"
    # H -> I
    h = positions["H"]
    i_pos = positions["I"]
    path_hi = [h, i_pos]
    predictions_hi = engine.predict(path_hi)
    print(f"\nPath for 'HI' -> Features: H, I")
    print(f"Top predictions: {predictions_hi}")
    
    if "HI" in predictions_hi:
        print("SUCCESS: 'HI' found in predictions")
    else:
        print("FAILURE: 'HI' not found")

    # Mock path for "HELLO"
    # H -> E -> L -> L -> O
    h = positions["H"]
    e = positions["E"]
    l = positions["L"]
    o = positions["O"]
    path_hello = [h, e, l, o]
    predictions_hello = engine.predict(path_hello)
    print(f"\nPath for 'HELLO' -> Features: H, E, L, O")
    print(f"Top predictions: {predictions_hello}")
    
    if "HELLO" in predictions_hello:
        print("SUCCESS: 'HELLO' found in predictions")
    else:
        print("FAILURE: 'HELLO' not found")

if __name__ == "__main__":
    test_prediction()
