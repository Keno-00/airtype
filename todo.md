# AirType Development Roadmap - Tomorrow's Fixes

## 🐛 High Priority Bugs
- [ ] **Clench Jitter / Cursor Drift**: When making a fist (clench), the cursor tends to jump upwards or drift. We need to implement a "motion lock" during the clench transition or a better smoothing filter to ignore movement during the fist-formation frames.
- [ ] **Path Smoothing**: The swipe paths are currently a bit "jagged". Implement a moving average or Savitzky-Golay filter to smooth the points recorded in `KeyboardWidget`.
- [ ] **Sensitivity Redux**: Current sensitivity is high. Balance **effortless movement** with **precision**. Consider a dynamic gain that decreases as you get closer to keys.

## 🛠️ Logic & Quality
- [ ] **Direct Selection Stability**: Ensure that clenching on `PRE1-3` slots always triggers the selection, even if tracking is noisy.
- [ ] **Hand-Lost Grace Period**: Optimize the transition when the hand leaves the frame. It resets now, but it could be "softer" to allow for brief obstructions.
- [ ] **Comprehensive Pytests**: 
  - [ ] Add tests for `PredictionEngine` with edge cases (short vs long paths).
  - [ ] Add UI tests using `pytest-qt` to verify key highlighting and layout switching.
- [ ] **Linter Cleanup**: Run `flake8` and fix stylistic warnings (missing docstrings, unused imports etc.).

## ✨ Future Enhancements
- [ ] Implement a "Tutorial Mode" for new users to learn the clench gesture.
- [ ] Add visual feedback for "Double Fist" (Caps Lock).
