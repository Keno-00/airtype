
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from prediction.engine import PredictionEngine
from ui.layouts import get_layout, get_key_positions

def debug_hi():
    layout = get_layout("qwerty")
    positions = get_key_positions(layout)
    dict_path = Path(__file__).parent / "dictionary.json"
    
    engine = PredictionEngine(dict_path, positions)
    
    # Path for HI
    h = positions["H"]
    i_pos = positions["I"]
    path_hi = [h, i_pos]
    
    # Check "HI" vs "BOOK" vs "BK" vs "NJ"
    words = ["HI", "BOOK", "BK", "NJ"]
    features = engine._extract_features(path_hi)
    
    print(f"Features for path {[(round(x,3), round(y,3)) for x,y in path_hi]}:")
    for f in features:
        print(f"  {engine._get_nearest_key(f)} at {(round(f[0], 3), round(f[1], 3))}")
        
    print("\nScoring comparison:")
    for w in words:
        if w.lower() in engine._dictionary:
            rank = engine._dictionary[w.lower()]
            score = engine._score_word(w, features, rank)
            # Break down score (manual recalculation for debug)
            word_positions = [positions[c.upper()] for c in w if c.upper() in positions]
            start_dist = ((word_positions[0][0] - features[0][0])**2 + (word_positions[0][1] - features[0][1])**2)**0.5
            end_dist = ((word_positions[-1][0] - features[-1][0])**2 + (word_positions[-1][1] - features[-1][1])**2)**0.5
            anchors = (start_dist + end_dist) * 12.0
            
            print(f"'{w}': Score={score:.4f}, Rank={rank}, AnchorPenalty={anchors:.4f}")
        else:
            print(f"'{w}': NOT IN DICTIONARY")

    # Run full predict
    print("\nTop Predict Results:")
    print(engine.predict(path_hi, top_n=5))

if __name__ == "__main__":
    debug_hi()
