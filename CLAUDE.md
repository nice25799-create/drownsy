# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file, real-time driver drowsiness detector (`drownsy.py`). It reads the webcam, tracks facial landmarks with dlib, and raises spoken alerts when the driver's eyes stay closed too long or they yawn repeatedly. Alert wording is generated on the fly by the Claude API. Windows-targeted (PowerShell, DirectShow/MSMF camera backends).

## Commands

Run everything from the project root (`D:\project`).

- Run the app: `python drownsy.py` — opens a webcam window; quit with `q` **in the video window** (not Ctrl-C).
- Webcam diagnostic: `python camtest.py` — probes camera indices 0–2 across DirectShow / Media Foundation / default backends and prints which combo works.
- Dependency check: `python -c "import cv2, dlib, imutils, scipy, anthropic; print('deps ok')"` (`pyttsx3` is optional; without it alerts are print-only).
- Re-extract reference text from PDFs: `python reference\_extract.py` (writes `reference\_text\*.txt`).

There is no build step, no linter, and no test suite. Dependencies are installed ad hoc with `pip install` (no requirements.txt). Python 3.11.

## Required assets & environment

- `shape_predictor_68_face_landmarks.dat` must sit in the project root (~95 MB, dlib's 68-point model). The app will not start without it.
- `ANTHROPIC_API_KEY` env var enables Claude-generated alert lines. If unset, the app still runs and falls back to a fixed wake-up phrase.

## Architecture

`drownsy.py` is the whole application — one `main()` loop plus helpers. Key pieces:

- **Two independent detection paths** run per frame inside the face loop:
  - *Eye closure*: Eye Aspect Ratio (EAR) averaged over both eyes. EAR below `EYE_AR_THRESH` for `EYE_AR_CONSEC_FRAMES` consecutive frames fires a drowsiness alert (one per closure episode).
  - *Yawning*: Mouth Aspect Ratio (MAR) from the 8 inner-lip landmarks (dlib indices `60:68`). MAR above `MOUTH_AR_THRESH` for enough frames counts as one yawn; `YAWN_ALERT_COUNT` yawns inside a rolling `YAWN_WINDOW_SEC` window (tracked in a `deque` of timestamps) fires an alert.
- **Alert pipeline is asynchronous.** When either path triggers, a `daemon` thread runs `handle_alert` → `generate_alert_message` (Claude call) → `speak` (pyttsx3 TTS). The main video loop never blocks on the network or speech. Two guards prevent alert spam: a `threading.Event` (`alert_busy`) ensures only one alert is in flight, and `ALERT_COOLDOWN` enforces a minimum gap between alerts.
- **Claude integration** lives in `generate_alert_message`: it asks for a short spoken wake-up line, passing the episode count and a `reason` string (e.g. eye-closure duration or yawn count). Any exception falls back to the fixed phrase, so a missing key or network failure degrades gracefully. Model is set by `CLAUDE_MODEL`.
- **Tunables** are a labeled config block at the top of `drownsy.py` (`EYE_AR_THRESH`, `EYE_AR_CONSEC_FRAMES`, `MOUTH_AR_THRESH`, `MOUTH_AR_CONSEC_FRAMES`, `YAWN_WINDOW_SEC`, `YAWN_ALERT_COUNT`, `CLAUDE_MODEL`, `ALERT_COOLDOWN`). Threshold tuning is the most common edit.

`camtest.py` matters because `drownsy.py` hardcodes `cv2.VideoCapture(0)`. If the working camera turns out to be a different index/backend, that line in `drownsy.py` needs to match.

## Research context & roadmap

Full background lives in `DROWSINESS_RESEARCH.md` (concepts, ~60-paper library, datasets). Condensed essentials:

- **Headline research angle: posture.** The project's distinguishing interest is drowsiness ↔ **posture** (head pose pitch/yaw/roll, body/sitting posture, seat-pressure), on top of the usual eye/face cues. This is **not yet implemented** — `drownsy.py` currently covers only EAR (eyes) + MAR (yawning). Head-pose and posture features are the roadmap, so new feature work likely heads in that direction.
- **Standard pipeline:** capture → detect/track (Dlib 68-landmarks, MediaPipe, YOLO) → extract features → classify (threshold rules, classic ML, or deep learning) → alert.
- **Metric vocabulary to use:** EAR (drowsy ~0.2–0.25 sustained), MAR (rises when yawning), **PERCLOS** (% eye closure over time, flagged ≥~0.3 — the most-cited vision measure, not yet computed here), head-pose angles, posture/seat-pressure indices (COP, fractal dimension), HRV. Modalities group as behavioral / vehicular / physiological.
- **Benchmark datasets** (for any ML/validation work): NTHU-DDD (most-cited IR benchmark), UTA-RLDD, YawDD (yawning), DROZY (physiological), MRL Eye (eye-state). Models trained on one dataset often don't generalize — validate cross-dataset.
- **Caveat:** vision-only is cheap and non-intrusive but lighting/occlusion-sensitive; combining eye + mouth + head/posture cues is the robustness path. This is a research reference, not safety-certified.

## Reference papers

`reference\` holds the research PDFs behind the thresholds and approach, with pre-extracted plain text in `reference\_text\*.txt`. Search the `_text` files (not the PDFs) when looking up what a paper says — the `ref-search` skill automates this and includes a filename→paper key. The broader (mostly online-only) paper library is catalogued in `DROWSINESS_RESEARCH.md` §4.

## Project skills

`.claude/skills/` defines three workflows: `run-drowsy` (launch with pre-flight checks), `camtest` (webcam diagnostics + privacy-setting troubleshooting), and `ref-search` (topic search across the reference papers). Prefer these for the tasks they cover.
