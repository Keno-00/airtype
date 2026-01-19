"""
Word prediction engine for swipe-typing.
Calculates likely word candidates based on swipe path shape.
"""
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional

class PredictionEngine:
    """
    Predicts words based on a series of (x, y) coordinates.
    Uses 'Shape Writing' approach:
    1. Extract features (start, corners/bends, end) from path.
    2. Compare word shapes using Euclidean distance between expected key positions.
    3. Rank by distance * word frequency.
    """
    
    def __init__(self, dictionary_path: Path, layout_positions: Dict[str, Tuple[float, float]]):
        self._layout_positions = layout_positions
        self._dictionary: Dict[str, float] = {} # word -> rank (lower is more frequent)
        self._load_dictionary(dictionary_path)
        
    def update_layout(self, layout_positions: Dict[str, Tuple[float, float]]):
        """Update key positions when UI layout changes."""
        self._layout_positions = layout_positions
        
    def _load_dictionary(self, path: Path):
        """Load dictionary from JSON file."""
        if path.exists():
            with open(path, 'r') as f:
                self._dictionary = json.load(f)
        else:
            # Fallback/Debug dictionary if file missing
            self._dictionary = {"the": 1, "to": 2, "and": 3, "a": 4, "of": 5}
            
    def predict(self, raw_points: List[Tuple[float, float]], top_n: int = 3) -> List[str]:
        """
        Predict top N words for a given swipe path.
        raw_points: List of normalized (x, y) coordinates from 0-1.
        """
        if len(raw_points) < 2:
            return []
            
        # 1. Extract path features (start, corners, end)
        features = self._extract_features(raw_points)
        if not features:
            return []
            
        # 2. Downsample features if too long (prevents O(W*P) explosion)
        if len(features) > 25:
            # Keep first, last, and pick 23 spread out features
            step = len(features) / 24.0
            idx = 0.0
            new_features = []
            for _ in range(24):
                new_features.append(features[int(idx)])
                idx += step
            new_features.append(features[-1])
            features = new_features

        start_pt = features[0]
        end_pt = features[-1]
        
        candidates = []
        
        # 3. Filter and score words
        # Aggressive Filter: Only check words starting/ending near the path bounds
        # (Threshold approx 1.5 keys away = 0.15 normalized)
        for word, rank in self._dictionary.items():
            if len(word) < 2: continue
            
            w_start = word[0].upper()
            w_end = word[-1].upper()
            if w_start not in self._layout_positions or w_end not in self._layout_positions:
                continue
                
            kp_start = self._layout_positions[w_start]
            kp_end = self._layout_positions[w_end]
            
            # Start/End keys must be reasonably close to path endpoints
            dist_start = math.sqrt((kp_start[0]-start_pt[0])**2 + (kp_start[1]-start_pt[1])**2)
            dist_end = math.sqrt((kp_end[0]-end_pt[0])**2 + (kp_end[1]-end_pt[1])**2)
            
            if dist_start > 0.18 or dist_end > 0.18:
                continue
                
            # Score this word
            score = self._score_word(word, features, rank)
            candidates.append((word, score))
            
        # 4. Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        return [c[0].upper() for c in candidates[:top_n]]

    def _extract_features(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Identify start, significant bends, and end of path."""
        if len(points) < 3:
            return points
            
        features = [points[0]]
        
        # Curvature detection (looking for sharp changes in direction)
        for i in range(1, len(points) - 1):
            p_prev = points[i-1]
            p_curr = points[i]
            p_next = points[i+1]
            
            # Vectors
            v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            
            # Magnitudes
            m1 = math.sqrt(v1[0]**2 + v1[1]**2)
            m2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if m1 < 0.001 or m2 < 0.001:
                continue
                
            # Normalized dot product (cosine of angle)
            cos_theta = (v1[0]*v2[0] + v1[1]*v2[1]) / (m1 * m2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            
            # If angle is sharp (e.g. < 120 degrees -> cos < -0.5 is very sharp, 
            # but we want "bends" so maybe cos < 0.7 which is ~45 deg turn)
            if cos_theta < 0.8: 
                features.append(p_curr)
                
        features.append(points[-1])
        return features

    def _score_word(self, word: str, path_features: List[Tuple[float, float]], rank: float) -> float:
        """Calculate how well a word matches the path features."""
        # 1. Target positions for letters in the word (consolidate double letters for length)
        word_positions = []
        distinct_word_keys = []
        for char in word.upper():
            if char in self._layout_positions:
                pos = self._layout_positions[char]
                word_positions.append(pos)
                if not distinct_word_keys or distinct_word_keys[-1] != char:
                    distinct_word_keys.append(char)
        
        if not word_positions:
            return float('inf')
            
        # 2. Path distance metrics
        path_len = 0
        for i in range(1, len(path_features)):
            p1, p2 = path_features[i-1], path_features[i]
            path_len += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            
        word_len = 0
        for i in range(1, len(word_positions)):
            k1, k2 = word_positions[i-1], word_positions[i]
            word_len += math.sqrt((k2[0]-k1[0])**2 + (k2[1]-k1[1])**2)
            
        # 3. Anchor Check (Start/End must be very close)
        start_dist = math.sqrt((word_positions[0][0] - path_features[0][0])**2 + (word_positions[0][1] - path_features[0][1])**2)
        end_dist = math.sqrt((word_positions[-1][0] - path_features[-1][0])**2 + (word_positions[-1][1] - path_features[-1][1])**2)
        
        # 4. Sequential Alignment Scoring
        total_dist = 0
        path_idx = 0
        for k_pos in word_positions:
            min_d = float('inf')
            best_idx = path_idx
            for i in range(path_idx, len(path_features)):
                px, py = path_features[i]
                d = (px - k_pos[0])**2 + (py - k_pos[1])**2 # Use squared distance for speed
                if d < min_d:
                    min_d = d
                    best_idx = i
            total_dist += math.sqrt(min_d)
            path_idx = best_idx 
            
        # 5. Combined Scoring
        avg_dist = total_dist / len(word_positions)
        
        # Length penalty based on DISTINCT keys (HELO = 4 keys vs 4 features)
        length_penalty = abs(len(distinct_word_keys) - len(path_features)) * 0.25
        
        # Physical path distance penalty
        len_ratio = path_len / (word_len + 0.05)
        path_len_penalty = 0.5 if (len_ratio < 0.4 or len_ratio > 3.0) else 0.0
        
        anchor_penalty = (start_dist + end_dist) * 12.0
        
        # 6. Frequency weighting
        freq_factor = math.log10(rank + 1) * 0.05
        
        return avg_dist + length_penalty + anchor_penalty + freq_factor + path_len_penalty

    def _get_nearest_key(self, pos: Tuple[float, float]) -> str:
        """Find the key closest to a given normalized position."""
        min_dist = float('inf')
        nearest = ""
        for key, k_pos in self._layout_positions.items():
            dist = math.sqrt((pos[0] - k_pos[0])**2 + (pos[1] - k_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = key
        return nearest
