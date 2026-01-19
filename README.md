# AirType

**Swipe-Typing Input System for Linux Couch Gaming**

AirType is an open-source text input solution designed for Linux gaming setups where you're sitting on the couch with a controller or using hand gestures. Say goodbye to slow character-by-character typing on your TV screen! AirType brings intuitive swipe-typing to your living room.



## The Problem

Typing text on a TV with a controller is painfully slow. Whether you're chatting with friends, searching for games, or entering credentials, traditional on-screen keyboards require tedious navigation through individual characters.

## The Solution

AirType provides two innovative input methods:

### 1. **Controller Swipe Input**
- Hold a designated button
- Steer your cursor with the thumbstick
- Trace swipe patterns across an on-screen keyboard
- Release to get word predictions

### 2. **Webcam Gesture Input**
- **Start typing**: Pinch your index finger and thumb together, then draw swipe patterns in the air
- **Select predictions**: Show 1, 2, or 3 fingers to choose from word suggestions
- **Backspace**: Pinch and swipe left to delete one letter
- **Delete word**: Make a fist and swipe left to remove the entire word


## Features

- **Dual Input Methods**: Choose between controller or webcam gesture input
- **Smart Word Prediction**: Swipe-typing with intelligent word suggestions
- **Multiple Keyboard Layouts**: QWERTY, DVORAK, and COLEMAK support out of the box
- **Themeable UI**: Customize appearance with QSS stylesheets (riceable!)
- **Compact Overlay**: 1/4 screen panel that docks to left or right side
- **Standalone Design**: Injects text directly into active applications—no need to modify existing virtual keyboards
- **Easy Distribution**: Packaged as a Flatpak for simple installation across Linux distros
- **Extensible**: Community can add new languages and layouts



## Architecture

### Technology Stack

- **UI Framework**: PyQt5 for responsive, customizable interface
- **Controller Input**: python-evdev for gamepad/controller handling
- **Hand Tracking**: MediaPipe for webcam gesture recognition
- **Text Injection**: Standalone overlay that injects text into active applications (Approach 1)



## Installation

### Via Flatpak (Recommended)

```bash
# Coming soon!
flatpak install flathub io.github.airtype.AirType
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Keno-00/airtype.git
cd airtype

# Install dependencies
pip install -r requirements.txt

# Run AirType
python main.py
```

### Dependencies

- Python 3.8+
- PyQt5
- python-evdev
- mediapipe
- opencv-python (for webcam input)

---

## Usage

### Launching AirType

```bash
# Start with controller input
airtype --input controller

# Start with webcam input
airtype --input webcam

# Specify keyboard layout
airtype --layout dvorak

# Choose panel position
airtype --position right
```

### Controller Input

1. Launch AirType
2. Hold the configured trigger button (default: Right Trigger)
3. Use the left thumbstick to trace your word across the keyboard
4. Release the trigger to see word predictions
5. Navigate and select your desired word

### Webcam Gesture Input

1. Ensure your webcam is connected and positioned to see your hand
2. Launch AirType with webcam mode
3. **Type**: Pinch thumb and index finger, then swipe across the virtual keyboard in the air
4. **Select word**: Show 1, 2, or 3 fingers for the 1st, 2nd, or 3rd prediction
5. **Backspace**: Pinch and swipe left
6. **Delete word**: Make a fist and swipe left

---

## Theming

AirType supports custom themes using QSS (Qt Style Sheets). Place your `.qss` files in the `themes/` directory.

```bash
# Apply a custom theme
airtype --theme themes/cyberpunk.qss
```

Example themes will be available in the repository to get you started!

---

## Language Support

**Current**: English only

**Future**: Community contributions welcome! We've designed AirType to be easily extensible for additional languages. Check out our [Contributing Guide](#-contributing) to learn how to add support for your language.

---

## Contributing

We welcome contributions from the community! Whether you want to:

- Add support for new languages
- Create custom keyboard layouts
- Design themes
- Improve gesture recognition
- Fix bugs or add features

Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/Keno-00/airtype.git
cd airtype

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

---

## Roadmap

- [ ] Initial release with controller and webcam input
- [ ] Flatpak packaging
- [ ] Additional keyboard layouts (AZERTY, etc.)
- [ ] Multi-language support
- [ ] Gesture customization
- [ ] Voice input integration
- [ ] Steam Deck optimization
- [ ] Wayland support improvements

---

## License

AirType is licensed under the **GNU General Public License v2.0 (GPL-2.0)**.

This means you're free to use, modify, and distribute this software, but any derivative works must also be open source under the same license.

See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

- MediaPipe team for excellent hand tracking
- The Linux gaming community for inspiration
- All contributors who help make couch gaming more accessible

---

## Contact

- **Issues**: [GitHub Issues](https://github.com/Keno-00/airtype/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Keno-00/airtype/discussions)

---

**Made with ❤️ for the Linux couch gaming community**
