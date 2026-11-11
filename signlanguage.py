"""
=============================================================================
ASL Sign Language Translator & Speech Synthesizer
McGill SE Student Project
Translates American Sign Language (ASL) letters into text in real-time.
Features: Live hand tracking, letter recognition, sentence construction,
          hold-to-commit mechanic, text-to-speech, and on-screen sign guide.
=============================================================================
"""

import os
import sys
import time
import math
import threading
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Import the ASL Classifier
from asl_classifier import ASLClassifier

# Optional TTS Engine
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


def speak_text(text):
    """Speaks text in a background thread so the camera feed never lags."""
    if not TTS_AVAILABLE or not text.strip():
        return

    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    t = threading.Thread(target=_speak, daemon=True)
    t.start()


# Skeleton connection pairs for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm base
]


def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=12):
    """Draws a modern rounded rectangle on an image."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

    if thickness == -1:
        # Filled
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        # Outline
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def main():
    print("=" * 60)
    print("       ASL SIGN LANGUAGE TRANSLATOR & SPEECH SYSTEM        ")
    print("=" * 60)

    # Locate model file
    model_path = "./hand_landmarker.task"
    if not os.path.exists(model_path):
        # Check script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, "hand_landmarker.task")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[ERROR] Could not find 'hand_landmarker.task' in {model_path} or {alt_path}")
            print("Please ensure the hand_landmarker.task model is present.")
            input("Press Enter to exit...")
            return

    print(f"[*] Loading MediaPipe Landmarker from: {model_path}")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    print("[OK] MediaPipe Hand Detector initialized!")

    # Initialize Classifier
    classifier = ASLClassifier()

    # Open Camera
    print("[*] Initializing Camera (Index 0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Camera 0 failed, trying Camera 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[ERROR] Cannot access any webcam. Please check your camera connection.")
        input("Press Enter to exit...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Smoothing and translation state
    history = deque(maxlen=7)
    current_letter = None
    current_conf = 0.0
    stable_letter = None
    letter_hold_start = None
    HOLD_COMMIT_TIME = 0.85  # Seconds of steady pose to lock in letter
    commit_cooldown = 0.0

    # Live sentence string
    sentence = ""
    show_guide = False
    cursor_timer = time.time()
    cursor_visible = True

    # Frame timing
    frame_index = 0
    fps = 0.0
    fps_time = time.time()
    fps_counter = 0

    window_name = "ASL Sign Language Translator"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n[READY] Sign Language Translator is running!")
    print("------------------------------------------------------------")
    print(" Controls:")
    print("   [Spacebar]     - Insert Space into sentence")
    print("   [Backspace]    - Delete last letter")
    print("   [C]            - Clear entire text")
    print("   [S]            - Speak sentence aloud (TTS)")
    print("   [H] or [Tab]   - Toggle ASL Reference Cheat Sheet Guide")
    print("   [Q] or [Esc]   - Quit program")
    print("------------------------------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Calculate FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        # Blink cursor every 0.5s
        if now - cursor_timer > 0.45:
            cursor_visible = not cursor_visible
            cursor_timer = now

        # Convert to RGB and MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Monotonic timestamp
        timestamp_ms = int(frame_index * 33.33)
        frame_index += 1

        results = None
        try:
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            pass

        detected_this_frame = None
        detected_conf = 0.0
        hand_pts_px = []
        hand_bbox = None

        if results and results.hand_landmarks:
            landmarks_raw = results.hand_landmarks[0]
            hand_pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_raw]

            # Hand bounding box
            xs = [p[0] for p in hand_pts_px]
            ys = [p[1] for p in hand_pts_px]
            x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
            y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
            hand_bbox = (x_min, y_min, x_max, y_max)

            # Classify hand landmarks
            detected_this_frame, detected_conf = classifier.classify(landmarks_raw)

        # Temporal smoothing
        if detected_this_frame:
            history.append(detected_this_frame)
        else:
            history.append(None)

        # Find most frequent prediction in recent window
        valid_preds = [p for p in history if p is not None]
        if valid_preds and len(valid_preds) >= 4:
            most_common = max(set(valid_preds), key=valid_preds.count)
            count = valid_preds.count(most_common)
            if count >= 3:
                current_letter = most_common
                current_conf = detected_conf
            else:
                current_letter = None
        else:
            current_letter = None
            current_conf = 0.0

        # Handle Hold-to-Commit mechanics
        commit_progress = 0.0
        if current_letter and current_letter != 'DEL':
            if current_letter == stable_letter:
                if letter_hold_start is None:
                    letter_hold_start = now
                held_time = now - letter_hold_start
                commit_progress = min(1.0, held_time / HOLD_COMMIT_TIME)

                if held_time >= HOLD_COMMIT_TIME and (now > commit_cooldown):
                    # Commit letter
                    if current_letter == 'SPACE':
                        sentence += " "
                    else:
                        sentence += current_letter
                    commit_cooldown = now + 0.60  # debounce delay
                    letter_hold_start = now       # reset hold
            else:
                stable_letter = current_letter
                letter_hold_start = now
        elif current_letter == 'DEL':
            # Fast backspace
            if stable_letter == 'DEL':
                if letter_hold_start and (now - letter_hold_start >= 0.70):
                    if len(sentence) > 0 and now > commit_cooldown:
                        sentence = sentence[:-1]
                        commit_cooldown = now + 0.35
            else:
                stable_letter = 'DEL'
                letter_hold_start = now
        else:
            stable_letter = None
            letter_hold_start = None

        # -------------------------------------------------------------
        # DRAWING UI & VISUALIZATIONS
        # -------------------------------------------------------------

        # 1. Draw Hand Skeleton
        if hand_pts_px:
            # Draw connection bones
            for start, end in HAND_CONNECTIONS:
                pt1 = hand_pts_px[start]
                pt2 = hand_pts_px[end]
                cv2.line(frame, pt1, pt2, (255, 180, 0), 2, cv2.LINE_AA)

            # Draw joint nodes
            for idx, pt in enumerate(hand_pts_px):
                if idx in (4, 8, 12, 16, 20):  # Fingertips
                    cv2.circle(frame, pt, 8, (0, 255, 120), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 5, (0, 210, 255), -1, cv2.LINE_AA)

            # Bounding box with letter badge
            if hand_bbox and current_letter:
                bx1, by1, bx2, by2 = hand_bbox
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 180), 2)
                badge_text = f" {current_letter} ({int(current_conf * 100)}%) "
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(frame, (bx1, max(0, by1 - th - 12)), (bx1 + tw + 6, by1), (0, 220, 160), -1)
                cv2.putText(frame, badge_text, (bx1 + 3, by1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

        # 2. Modern Header Bar (Top)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "ASL SIGN LANGUAGE TRANSLATOR", (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)

        fps_text = f"LIVE | {int(fps)} FPS"
        cv2.circle(frame, (w - 150, 30), 6, (0, 255, 0), -1)
        cv2.putText(frame, fps_text, (w - 135, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)

        # 3. Detection Card (Top-Left HUD)
        card_w, card_h = 320, 175
        card_x, card_y = 20, 70
        overlay = frame.copy()
        draw_rounded_rect(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (25, 28, 35), -1, radius=14)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        draw_rounded_rect(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 220, 160), 2, radius=14)

        cv2.putText(frame, "CURRENT DETECTED SIGN", (card_x + 16, card_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        if current_letter:
            disp_char = current_letter
            title, desc = classifier.letter_guide.get(disp_char, ("Recognized Sign", ""))

            # Huge Letter Badge
            badge_color = (0, 220, 120) if disp_char not in ('SPACE', 'DEL') else (255, 170, 0)
            cv2.putText(frame, disp_char, (card_x + 16, card_y + 88),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, badge_color, 3, cv2.LINE_AA)

            conf_str = f"{int(current_conf * 100)}% Match"
            cv2.putText(frame, conf_str, (card_x + 120, card_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, title, (card_x + 120, card_y + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 230, 255), 1, cv2.LINE_AA)

            # Hold to commit progress bar
            cv2.putText(frame, "Hold to Type:", (card_x + 16, card_y + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
            bar_x, bar_y, bar_w, bar_h = card_x + 16, card_y + 130, card_w - 32, 16
            draw_rounded_rect(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1, radius=6)
            fill_w = int(bar_w * commit_progress)
            if fill_w > 0:
                draw_rounded_rect(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 140), -1, radius=6)
        else:
            cv2.putText(frame, "--", (card_x + 20, card_y + 90),
                        cv2.FONT_HERSHEY_DUPLEX, 1.8, (100, 100, 110), 2, cv2.LINE_AA)
            cv2.putText(frame, "Show hand to camera", (card_x + 85, card_y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 150), 1, cv2.LINE_AA)
            cv2.putText(frame, "Signs: A-Z, Space, Del", (card_x + 85, card_y + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 180, 220), 1, cv2.LINE_AA)

        # 4. Transcribed Text Banner (Bottom)
        banner_h = 105
        banner_y = h - banner_h - 15
        banner_x = 20
        banner_w = w - 40

        overlay = frame.copy()
        draw_rounded_rect(overlay, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          (15, 18, 24), -1, radius=14)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        draw_rounded_rect(frame, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h),
                          (70, 75, 90), 2, radius=14)

        cv2.putText(frame, "TRANSLATED TEXT:", (banner_x + 20, banner_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 200), 1, cv2.LINE_AA)

        # Formatted sentence with cursor
        disp_sentence = sentence
        if len(disp_sentence) > 42:
            disp_sentence = "..." + disp_sentence[-40:]
        cursor_str = "_" if cursor_visible else " "
        rendered_text = f"{disp_sentence}{cursor_str}"

        cv2.putText(frame, rendered_text, (banner_x + 20, banner_y + 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom Shortcut hints
        shortcuts = "[Space] Space  |  [Backspace] Undo  |  [C] Clear  |  [S] Speak  |  [H] Sign Guide  |  [Q] Quit"
        cv2.putText(frame, shortcuts, (banner_x + 20, banner_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 170), 1, cv2.LINE_AA)

        # 5. ASL Cheat Sheet Guide Panel (Collapsible with 'H')
        if show_guide:
            guide_w, guide_h = 420, 520
            guide_x, guide_y = w - guide_w - 20, 70
            overlay = frame.copy()
            draw_rounded_rect(overlay, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                              (18, 20, 28), -1, radius=14)
            cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
            draw_rounded_rect(frame, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                              (0, 200, 255), 2, radius=14)

            cv2.putText(frame, "ASL QUICK REFERENCE GUIDE", (guide_x + 20, guide_y + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press 'H' or 'TAB' to hide", (guide_x + 20, guide_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 160), 1, cv2.LINE_AA)

            gy = guide_y + 78
            guide_items = [
                ("A", "Fist, thumb straight up"),
                ("B", "4 fingers up, thumb folded"),
                ("C", "Curved hand like a cup"),
                ("D", "Index up, thumb & mid touch"),
                ("F", "OK sign (Index & Thumb ring)"),
                ("I", "Pinky up, others curled"),
                ("L", "Index up & Thumb out (90 deg)"),
                ("O", "All tips touch in 'O' ring"),
                ("R", "Index & Middle crossed up"),
                ("U", "Index & Middle up together"),
                ("V", "Peace sign (2 up spread)"),
                ("W", "3 fingers up (W shape)"),
                ("X", "Index bent into claw/hook"),
                ("Y", "Thumb & Pinky out (Shaka)"),
                ("SPACE", "Open palm facing camera"),
                ("DEL", "Thumbs down gesture")
            ]

            for letter, desc in guide_items:
                cv2.putText(frame, letter, (guide_x + 20, gy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 180), 1, cv2.LINE_AA)
                cv2.putText(frame, f"- {desc}", (guide_x + 75, gy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
                gy += 27

        # Display Frame
        cv2.imshow(window_name, frame)

        # Key handling
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or Esc
            break
        elif key == ord(' '):  # Spacebar
            sentence += " "
        elif key in (8, 127):  # Backspace
            if len(sentence) > 0:
                sentence = sentence[:-1]
        elif key in (ord('c'), ord('C')):  # Clear
            sentence = ""
        elif key in (ord('s'), ord('S')):  # Speak
            if sentence.strip():
                print(f"[SPEAK] '{sentence.strip()}'")
                speak_text(sentence.strip())
        elif key in (ord('h'), ord('H'), 9):  # 'H' or Tab
            show_guide = not show_guide

    print("\n[*] Shutting down ASL Translator...")
    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Program closed cleanly. Goodbye!")


if __name__ == "__main__":
    main()
