"""
AirType - Swipe-Typing Input System for Linux Couch Gaming

Entry point for the application.
"""
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AirType - Swipe-Typing Input System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        choices=["webcam", "controller"],
        default="webcam",
        help="Input method (default: webcam)",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    
    parser.add_argument(
        "--layout",
        choices=["qwerty", "dvorak", "colemak"],
        default=None,
        help="Keyboard layout (overrides config)",
    )
    
    parser.add_argument(
        "--position",
        choices=["left", "right"],
        default=None,
        help="Panel position (overrides config)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with landmark overlay",
    )
    
    return parser.parse_args()


def run_webcam_debug(config):
    """
    Run webcam in debug mode - shows camera feed with landmarks.
    Useful for testing hand tracking from couch distance.
    """
    import cv2
    from webcam import HandTracker, GestureRecognizer, Gesture
    
    tracker = HandTracker(config)
    recognizer = GestureRecognizer(config.gestures)
    
    print("Starting webcam debug mode...")
    print("Press 'q' to quit")
    print("-" * 40)
    
    if not tracker.start():
        print("ERROR: Could not open camera")
        return 1
    
    try:
        while True:
            # Get landmarks
            landmarks = tracker.get_landmarks()
            
            # Recognize gesture
            state = recognizer.update(landmarks)
            
            # Get frame with overlay
            frame = tracker.get_frame_with_landmarks(landmarks)
            
            if frame is not None:
                # Add gesture text to frame
                gesture_text = f"Gesture: {state.gesture.name}"
                cv2.putText(
                    frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                # Add additional info
                info_lines = [
                    f"Pinch dist: {state.pinch_distance:.3f}",
                    f"Fingers: {state.extended_fingers}",
                    f"Position: ({state.hand_position[0]:.2f}, {state.hand_position[1]:.2f})",
                ]
                for i, line in enumerate(info_lines):
                    cv2.putText(
                        frame, line, (10, 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
                
                # Print gesture changes to console
                if state.gesture != Gesture.NONE:
                    print(f"[{tracker.frame_count:5d}] {state.gesture.name}")
                
                cv2.imshow("AirType Debug", frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        tracker.stop()
        cv2.destroyAllWindows()
    
    return 0


def run_webcam_mode(config):
    """Run AirType in webcam input mode with UI overlay (Multithreaded)."""
    import signal
    import atexit
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread, Qt
    from webcam import WebcamWorker, Gesture
    from ui import OverlayWindow
    
    app = QApplication(sys.argv)
    
    # Create overlay window
    overlay = OverlayWindow(
        position=config.ui.position,
        layout_name=config.keyboard.layout,
    )
    overlay.show()
    
    # State for gesture handling
    is_swiping = [False]  # Use list for mutability in closure
    
    # Setup background worker and thread
    thread = QThread()
    worker = WebcamWorker(config)
    worker.moveToThread(thread)
    
    def cleanup():
        """Ensure camera is released on exit."""
        print("\nCleaning up camera resources...")
        worker.stop_process()
        thread.quit()
        thread.wait(2000)
        print("Cleanup complete.")
    
    # Register cleanup for various exit scenarios
    atexit.register(cleanup)
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C and kill signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down...")
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def handle_gesture(state):
        """Handle gesture signals from background worker."""
        x, y = state.hand_position
        
        if state.gesture == Gesture.SWIPE_START:
            is_swiping[0] = True
            overlay.start_swipe(x, y)
        
        elif state.gesture == Gesture.SWIPE_HOLD and is_swiping[0]:
            overlay.update_swipe(x, y)
        
        elif state.gesture == Gesture.SWIPE_END and is_swiping[0]:
            is_swiping[0] = False
            path = overlay.end_swipe()
            if path:
                print(f"Action: Finalized word path {' -> '.join(path)}")
        
        elif state.gesture == Gesture.DOUBLE_FIST:
            overlay.toggle_caps()
        
        elif not is_swiping[0]:
            overlay.update_hand_position(x, y)

    def handle_hand_lost():
        """Handle signal when no hand is detected."""
        if is_swiping[0]:
            is_swiping[0] = False
            overlay.cancel_swipe()
            print("Action: Swipe aborted (hand lost)")

    # Connect signals (Use QueuedConnection to ensure UI updates happen in main thread)
    thread.started.connect(worker.start_process)
    worker.gesture_detected.connect(handle_gesture, Qt.QueuedConnection)
    worker.hand_lost.connect(handle_hand_lost, Qt.QueuedConnection)
    worker.frame_ready.connect(overlay.set_webcam_frame, Qt.QueuedConnection)
    worker.error.connect(lambda msg: print(f"WORKER ERROR: {msg}"), Qt.QueuedConnection)
    
    # Start thread
    thread.start()
    
    try:
        result = app.exec_()
    finally:
        cleanup()
        atexit.unregister(cleanup)  # Avoid double cleanup
    
    return result


def run_controller_mode(config):
    """Run AirType in controller input mode with UI overlay."""
    import signal
    import atexit
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QThread, Qt
    from controller import ControllerWorker
    from controller.worker import ControlMode, NavDirection
    from ui import OverlayWindow
    
    app = QApplication(sys.argv)
    
    # Create overlay window
    overlay = OverlayWindow(
        position=config.ui.position,
        layout_name=config.keyboard.layout,
    )
    overlay.show()
    overlay._debug = getattr(config.ui, 'debug_overlay', False)
    
    # Enable controller mode (shows button hints on keys)
    overlay.set_controller_mode(True)
    
    # State for gesture handling
    is_swiping = [False]
    
    # Setup background worker and thread
    thread = QThread()
    worker = ControllerWorker(config)
    worker.moveToThread(thread)
    
    def cleanup():
        """Ensure controller is released on exit."""
        print("\nCleaning up controller resources...")
        worker.stop_process()
        thread.quit()
        thread.wait(2000)
        print("Cleanup complete.")
    
    atexit.register(cleanup)
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C and kill signals gracefully."""
        print(f"\nReceived signal {signum}, shutting down...")
        app.quit()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def handle_gesture(state):
        """Handle gesture signals from controller worker."""
        x, y = state.hand_position
        
        if state.gesture == "SWIPE_START":
            is_swiping[0] = True
            overlay.start_swipe(x, y)
        
        elif state.gesture == "SWIPE_HOLD" and is_swiping[0]:
            overlay.update_swipe(x, y)
        
        elif state.gesture == "SWIPE_END" and is_swiping[0]:
            is_swiping[0] = False
            path = overlay.end_swipe()
            if path:
                print(f"Action: Finalized word path {' -> '.join(path)}")
        
        elif state.mode == ControlMode.CURSOR and not is_swiping[0]:
            # Free cursor movement
            overlay.update_hand_position(x, y)
    
    def handle_nav(direction):
        """Handle D-pad navigation events."""
        # Navigate between keys on the keyboard
        overlay.navigate_key(direction.name.lower())
    
    def handle_button(action):
        """Handle button action events."""
        print(f"Button action received: {action}")
        if action == "SPACE":
            overlay.handle_special_key("SPACE")
        elif action == "BACKSPACE":
            overlay.handle_special_key("BACKSPACE")
        elif action == "SHIFT":
            overlay.handle_special_key("SHIFT")
        elif action == "SELECT":
            overlay.select_current_key()
        elif action == "CLOSE":
            print("Action: Closing application")
            app.quit()
        elif action == "SEND":
            overlay.handle_special_key("ENTER")
        elif action == "CURSOR_LEFT":
            overlay.move_text_cursor(-1)
        elif action == "CURSOR_RIGHT":
            overlay.move_text_cursor(1)
        elif action in ["PRE1", "PRE2", "PRE3"]:
            overlay.handle_special_key(action)
    
    # Connect signals (Use QueuedConnection to ensure UI updates happen in main thread)
    thread.started.connect(worker.start_process)
    worker.gesture_detected.connect(handle_gesture, Qt.QueuedConnection)
    worker.nav_event.connect(handle_nav, Qt.QueuedConnection)
    worker.button_event.connect(handle_button, Qt.QueuedConnection)
    worker.error.connect(lambda msg: print(f"CONTROLLER ERROR: {msg}"), Qt.QueuedConnection)
    
    # Start thread
    thread.start()
    
    try:
        result = app.exec_()
    finally:
        cleanup()
        atexit.unregister(cleanup)
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    from webcam import load_config
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.layout:
        config.keyboard.layout = args.layout
    if args.position:
        config.ui.position = args.position
    if args.debug:
        config.ui.debug_overlay = True
    
    print(f"AirType starting...")
    print(f"  Input mode: {args.input}")
    print(f"  Layout: {config.keyboard.layout}")
    print(f"  Debug: {args.debug}")
    print()
    
    # Run appropriate mode
    if args.input == "webcam":
        if args.debug:
            return run_webcam_debug(config)
        else:
            return run_webcam_mode(config)
    else:
        return run_controller_mode(config)


if __name__ == "__main__":
    sys.exit(main())
