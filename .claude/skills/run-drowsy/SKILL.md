---
name: run-drowsy
description: Run the drowsiness detection app (drownsy.py) with pre-flight checks for dependencies, the dlib landmark model, and the Anthropic API key.
---

Launch the drowsiness detector with pre-flight checks:

1. Verify required packages import cleanly in one command:
   `python -c "import cv2, dlib, imutils, scipy, anthropic; print('deps ok')"`
   (`pyttsx3` is optional — the app falls back to print-only alerts without it.)
2. Verify the landmark model exists: `shape_predictor_68_face_landmarks.dat` in the project root (~95 MB).
3. Check `$env:ANTHROPIC_API_KEY` is set. If it is missing, warn the user that Claude-generated alert messages will fall back to a fixed phrase, but the app still works.
4. Run `python drownsy.py` from the project root (D:\project). It opens a webcam window; the user quits with `q` in the video window.
5. If the webcam fails to open, suggest running /camtest to diagnose.

This is an interactive GUI app — run it in the foreground. On launch it runs a ~10 s EAR calibration (user looks at the camera, eyes open) before detection starts. Key tunables are at the top of drownsy.py: CALIB_SECONDS, CALIB_RATIO, EYE_CLOSED_ALERT_SEC, PERCLOS_THRESH, POSE_GATE_PITCH_DEG, SLUMP_PITCH_DEG, CLAUDE_MODEL, ALERT_COOLDOWN.
