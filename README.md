# ASL Real-Time Sign Language Translator & Speech Synthesizer

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-1.2%2B-green.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-red.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time American Sign Language (ASL) alphabet translator and text-to-speech synthesizer built with Google MediaPipe and OpenCV. Translates manual alphabet signs directly into typed text and spoken audio.

---

## 📽️ Features

- **Real-Time 3D Hand Skeleton Tracking**: 21 high-precision landmarks powered by MediaPipe.
- **Invariant ASL Alphabet Recognition**: Robust geometric classifier measuring 3D finger curl, joint angles, inter-finger spacing, thumb orientation, and hand pose across scale and distance.
- **Full Alphabet Coverage**: Detects letters **A through Y** with confidence percentages.
- **Hold-to-Commit Text Typing**: Hold any sign steady for ~0.85 seconds to commit the letter into the live sentence. An animated progress bar visualizes your hold duration.
- **Control Gestures**:
  - 🖐️ **Open Palm** (5 fingers spread): Inserts a `SPACE`.
  - 👎 **Thumbs Down** (Fist with thumb down): Backspace / `DELETE`.
- **Speech Synthesis (TTS)**: Press **'S'** to speak the constructed sentence aloud using a non-blocking background speech synthesizer.
- **On-Screen ASL Reference Cheat Sheet**: Press **'H'** or **'Tab'** anytime to toggle an in-window reference guide with instructions for every letter.
- **Cyberpunk / Modern HUD**:
  - Live FPS counter & camera connection status.
  - Hand bounding box with dynamic letter badge.
  - Dedicated recognition card with confidence percentage and anatomical description.
  - Transcribed sentence banner with a blinking typing cursor.

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Zaidmcuni/asl-sign-language-translator.git
cd asl-sign-language-translator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Windows (Quick Launcher)
Simply double-click `run.bat` in the project folder.

### Command Line
```bash
python signlanguage.py
```

---

## ⌨️ Controls & Shortcuts

| Key / Gesture | Action |
|---|---|
| **Hold Sign (~0.85s)** | Commits the letter to the sentence |
| **Open Palm** / **`Spacebar`** | Inserts a space |
| **Thumbs Down** / **`Backspace`** | Deletes the last character |
| **`C`** | Clears the entire sentence |
| **`S`** | Speaks the sentence aloud (Text-to-Speech) |
| **`H`** or **`Tab`** | Toggles the on-screen ASL Cheat Sheet Guide |
| **`Q`** or **`Esc`** | Quits the program |

---

## 🖐️ ASL Alphabet Quick Reference

- **A**: Fist with thumb straight upright resting alongside index finger.
- **B**: 4 fingers straight up together, thumb folded across palm.
- **C**: Hand curved into a cup / "C" shape profile.
- **D**: Index pointing straight UP, thumb touching middle fingertip.
- **E**: All fingers curled down tight into palm, thumb tucked underneath fingertips.
- **F**: Thumb & index tips touching (OK sign), other 3 fingers straight up.
- **G**: Hand held sideways, index finger & thumb pointing forward horizontally.
- **H**: Hand held sideways, index & middle extended horizontally side-by-side.
- **I**: Pinky pointing straight UP, other fingers curled into fist.
- **K**: Index pointing UP, middle angled forward/up, thumb upright between them.
- **L**: Index pointing UP and thumb extended outward at 90° ("L" shape).
- **M**: Thumb tucked under 3 fingers (Index, Middle, Ring).
- **N**: Thumb tucked under 2 fingers (Index, Middle).
- **O**: All 5 fingertips touching thumb in an "O" ring.
- **P**: Like K/H, but pointing downward.
- **Q**: Like G, but pointing downward.
- **R**: Index & Middle extended UP and crossed over each other.
- **S**: Tight fist with thumb folded across the front of fingers (over knuckles).
- **T**: Thumb tucked between index and middle knuckles.
- **U**: Index & Middle extended UP pressed together side-by-side.
- **V**: Peace sign / Victory (Index & Middle UP and spread apart).
- **W**: Index, Middle, Ring extended UP spread apart (3 fingers up).
- **X**: Index finger bent into a claw/hook, other fingers curled into fist.
- **Y**: Thumb & Pinky extended outward ("Hang Loose" / "Shaka").
- **SPACE**: Flat open palm facing camera.
- **DEL**: Thumbs down gesture.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
