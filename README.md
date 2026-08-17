# Drowsy — real-time driver drowsiness detector

Watches you through a webcam and speaks a wake-up alert when you show signs of
drowsiness. It tracks four things at once:

| Signal | What it measures | Fires when |
| --- | --- | --- |
| **EAR** | eye openness, calibrated to *your* eyes at startup | eyes shut ≥ 1 s |
| **PERCLOS** | % of the last 60 s spent with eyes closed | ≥ 30 % |
| **MAR** | mouth openness | 3 yawns within 60 s |
| **Head pose** | pitch / yaw / roll of your head | head slumped forward ≥ 4 s |

Head pose also acts as a *gate*: while you're looking down or away, the eye
checks pause. Otherwise glancing at your keyboard reads as closed eyes and
fires a false alarm.

Tested on Windows 11 with Python 3.11.

---

## Setup

### 1. Python

Python **3.11** is what this is developed against. Check yours:

```powershell
python --version
```

### 2. Install the dependencies

```powershell
pip install -r requirements.txt
```

> **If you've tried `pip install dlib` before and it failed** — that's normal.
> The plain `dlib` package compiles from source and needs CMake plus the Visual
> Studio C++ Build Tools. `requirements.txt` uses **`dlib-bin`** instead, which
> is the identical library as a prebuilt wheel. No compiler needed.

### 3. Download the face landmark model

The detector needs dlib's 68-point landmark model. It's ~95 MB, so it isn't
stored in this repo — download it once:

1. Get <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2>
2. Extract the `.bz2` (7-Zip works, or use the PowerShell snippet below)
3. Put `shape_predictor_68_face_landmarks.dat` in the **project root**, next to
   `drownsy.py`

Extract without installing anything:

```powershell
python -c "import bz2,shutil; shutil.copyfileobj(bz2.open('shape_predictor_68_face_landmarks.dat.bz2'), open('shape_predictor_68_face_landmarks.dat','wb'))"
```

The app refuses to start without this file.

### 4. (Optional) Claude-generated alert wording

Set `ANTHROPIC_API_KEY` and each alert gets a freshly written spoken line
instead of a fixed phrase:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Entirely optional. Without it you'll see a `[WARN] Claude alert generation
failed` line and it falls back to a fixed wake-up phrase — everything else
works normally.

---

## Running it

```powershell
python drownsy.py
```

A webcam window opens. **Quit with `q` while the video window is focused** —
not Ctrl-C in the terminal.

### What happens first: a 10-second calibration

The app spends its first 10 seconds measuring your neutral face — your normal
eye openness becomes your personal EAR threshold, and your normal head angle
becomes the baseline the slump detector compares against.

**Sit the way you normally would, facing the camera square-on, and don't look
away.** This matters more than it sounds: calibrating while turned toward a
second monitor once produced a neutral yaw ~30° off, which left the pose gate
stuck on and silently disabled *all* eye detection for the whole session.

You should see something like:

```
[INFO] Calibrated EAR threshold: 0.205 (from 139 samples)
[INFO] Neutral head pose: pitch +5.9, yaw +0.5 deg
```

If the sample count is 0, the app didn't see your face — check the webcam
troubleshooting below.

### Reading the window

- `EAR` / `MAR` / `PERCLOS` / `Yawns(60s)` — live metric values
- `P +12  Y -3  R +1` — head pitch / yaw / roll relative to your neutral pose
- `POSE GATE` in orange — you're looking down or away, eye checks are paused
- `DROWSINESS ALERT!` in red — an alert is firing

Alerts also print to the console, e.g.:

```
[TRIGGER eyes] Eyes were closed for about 1 seconds. (EAR 0.156, th 0.205, PERCLOS 34%)
[TRIGGER slump] Driver's head slumped forward for about 4 seconds. (pitch +11 deg)
```

### Trying it out

- **Eye closure** — close your eyes ~2 s while facing the camera
- **Yawning** — three exaggerated yawns within a minute
- **Slump** — let your head drop forward and hold it ~5 s
- **The gate** — look down at your keyboard; `POSE GATE` should appear and no
  eye alert should fire

---

## Webcam trouble

If the window never opens or the frame is black:

```powershell
python camtest.py
```

This probes camera indices 0–2 across the DirectShow, Media Foundation and
default backends, and prints which combination works.

`drownsy.py` hardcodes `cv2.VideoCapture(0)`. If `camtest.py` reports a
different index working, change that line to match.

Also worth checking on Windows 11: **Settings → Privacy & security → Camera**,
and confirm *"Let desktop apps access your camera"* is on. A camera already in
use by Zoom/Teams/OBS will also fail to open.

---

## Tests

```powershell
python -m pytest tests/ -v
```

Pure-logic tests — threshold derivation, the PERCLOS window, head-pose maths
and the pose gate. No webcam or network needed, runs in about 2 seconds.

---

## Tuning

Thresholds live in a labeled config block at the top of `drownsy.py`. The ones
worth touching first:

| Setting | Default | Effect |
| --- | --- | --- |
| `EYE_CLOSED_ALERT_SEC` | 1.0 | how long eyes must stay shut |
| `PERCLOS_THRESH` | 0.3 | slow-droop sensitivity |
| `MOUTH_AR_THRESH` | 0.5 | how wide a yawn counts |
| `POSE_GATE_PITCH_DEG` | 10.0 | how far down before eye checks pause |
| `SLUMP_PITCH_DEG` | 10.0 | how far down counts as slumping |
| `SLUMP_ALERT_SEC` | 4.0 | how long a slump must last |
| `FRAME_WIDTH` | 800 | preview window size |

The EAR threshold is *not* here — it's calibrated per person at startup.

A note on `SLUMP_PITCH_DEG`: it looks low, and that's deliberate. dlib's face
detector loses your face entirely once your head goes far enough down, so past
roughly +15° there's no angle left to measure. The slump episode has to *start*
while your face is still trackable, and `SLUMP_ALERT_SEC` does the work of
telling a glance apart from a real slump.

---

## Caveats

Vision-only detection is cheap and non-intrusive but sensitive to lighting and
occlusion — glasses, hair over the face, backlighting from a window, or a dim
room all degrade landmark tracking. Expect a fair number of dropped frames.

**This is a research and coursework project, not a safety-certified system.**
Don't rely on it while actually driving.

Background reading, dataset notes and the ~60-paper reference library are in
`DROWSINESS_RESEARCH.md`.
