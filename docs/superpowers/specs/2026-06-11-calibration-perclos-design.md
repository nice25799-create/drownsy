# Design: Per-user EAR calibration, PERCLOS, and time-based detection

Date: 2026-06-11
Status: approved (user, 2026-06-11)
Target: `drownsy.py` (single file, approach B: small helpers + class, no module split)

## Context

`drownsy.py` detects drowsiness with a fixed EAR threshold (0.25) over a fixed
frame count (20), plus MAR-based yawn counting. Three accuracy problems,
all literature-backed:

1. Fixed EAR thresholds fail across people (eye shape, glasses). The papers
   recommend personalized calibration (`2408.05836v1` lists it as the explicit
   accuracy fix).
2. No PERCLOS — the standard drowsiness measure (% of time per minute the eyes
   are closed, `sensors-22-02069`). Consecutive-frame logic misses slow eyelid
   droop that never crosses the consecutive-closure bar.
3. Frame-count thresholds depend on camera FPS: "20 frames" is 0.67 s at
   30 fps but 1.33 s at 15 fps. Detection sensitivity silently varies by
   machine.

User decisions during brainstorming: calibration runs automatically on every
launch (no saved state); balanced sensitivity (1.0 s closure, PERCLOS 0.30);
single-file structure with testable helper units.

## Goals

- Personal EAR threshold from a startup calibration, every launch.
- PERCLOS over a rolling 60 s window as a second, independent eye signal.
- All detection thresholds expressed in seconds, not frames.
- New logic unit-testable without a webcam.

## Non-goals (future steps, not this change)

- Accuracy evaluation harness / labeled datasets (next roadmap step).
- Head pose, posture features, ML classifier (jimaging RF recipe).
- Saving calibration to disk; recalibration hotkey. Restarting the app
  recalibrates.
- Changes to yawn-window logic, Claude alert generation, or TTS.

## New configuration block

| Constant | Value | Meaning |
|---|---|---|
| `CALIB_SECONDS` | 10.0 | Duration of startup calibration capture |
| `CALIB_RATIO` | 0.75 | Threshold = ratio x median open-eye EAR |
| `CALIB_THRESH_MIN` | 0.15 | Lower clamp for derived threshold |
| `CALIB_THRESH_MAX` | 0.30 | Upper clamp for derived threshold |
| `MIN_CALIB_SAMPLES` | 30 | Fewer face-frames than this -> use default |
| `EAR_DEFAULT_THRESH` | 0.25 | Fallback threshold when calibration fails |
| `EYE_CLOSED_ALERT_SEC` | 1.0 | Continuous closure that fires the alert |
| `MOUTH_OPEN_YAWN_SEC` | 0.5 | Continuous open-mouth time = one yawn |
| `PERCLOS_WINDOW_SEC` | 60.0 | PERCLOS rolling window |
| `PERCLOS_THRESH` | 0.30 | PERCLOS value that fires the alert |

Removed: `EYE_AR_THRESH`, `EYE_AR_CONSEC_FRAMES`, `MOUTH_AR_CONSEC_FRAMES`,
and the `COUNTER` / `MOUTH_COUNTER` frame counters.

Kept unchanged: `MOUTH_AR_THRESH` (0.5), `YAWN_WINDOW_SEC` (60), `YAWN_ALERT_COUNT`
(3), `CLAUDE_MODEL`, `ALERT_COOLDOWN` (6), the alert pipeline
(`generate_alert_message` -> `speak`, `alert_busy` event, daemon threads).

## Components

### 1. `derive_threshold(samples) -> (threshold, used_default)` — pure function

- `samples`: list of EAR floats collected during calibration.
- If `len(samples) < MIN_CALIB_SAMPLES`: return `(EAR_DEFAULT_THRESH, True)`.
- Else: `open_ear = median(samples)` (median is robust to blink dips);
  `threshold = clamp(CALIB_RATIO * open_ear, CALIB_THRESH_MIN, CALIB_THRESH_MAX)`;
  return `(threshold, False)`.
- Pure (list in, tuple out): unit-testable.

### 2. `calibrate_ear(vs, detector, predictor, l_idx, r_idx) -> (threshold, used_default) | None`

- Runs after the camera opens, before the detection loop.
- For `CALIB_SECONDS`, per frame: same pipeline as the main loop (resize to
  width 450, grayscale, detect, landmarks; EAR is scale-invariant but the
  pipeline stays identical for consistent detection behavior). Append the
  mean-of-both-eyes EAR for the first detected face; frames with no face
  contribute nothing.
- HUD during calibration: live frame plus
  `"CALIBRATING - look at camera, eyes open ({remaining:.0f}s)"` and the live EAR.
- `q` pressed -> return `None`; caller releases the camera and exits cleanly.
- On completion calls `derive_threshold`. If `used_default`, print
  `[WARN]` that calibration failed and the default 0.25 applies. Print the
  derived threshold either way. Returns `(threshold, used_default)` so the
  main loop can render the `(default)` HUD suffix.

### 3. `PerclosTracker` — class

```python
class PerclosTracker:
    def __init__(self, window_sec: float): ...
    def update(self, closed: bool, t: float) -> None   # append, evict > window
    def value(self) -> float    # fraction of samples with closed=True (0.0 if empty)
    def ready(self) -> bool     # newest_t - oldest_t >= 0.5 * window_sec
    def clear(self) -> None     # restart window (after an alert)
```

- Sample-fraction is the practical webcam approximation of PERCLOS ("closed" =
  EAR below the calibrated threshold, standing in for the strict 80%-eyelid
  definition). Assumes roughly uniform frame intervals; acceptable since the
  same camera produces all samples.
- `ready()` prevents false alarms before the window has >= 30 s of history.

### 4. Time-based detection logic (main loop changes)

- **Eye closure:** on open->closed transition store `closure_start = now`.
  While closed: alert fires when `now - closure_start >= EYE_CLOSED_ALERT_SEC`,
  subject to the existing guards (`ALARM_ON` once per episode, `alert_busy`,
  `ALERT_COOLDOWN`). On closed->open: reset `ALARM_ON`.
  The on-screen `DROWSINESS ALERT!` text appears under the same duration
  condition (was: frame-count condition).
- **Yawn:** when the mouth transitions from closed to open
  (`mar > MOUTH_AR_THRESH` begins) store `mouth_open_start = now`;
  one yawn is counted when the mouth stays open `>= MOUTH_OPEN_YAWN_SEC`
  (`yawn_counted` flag unchanged). The 3-yawns-in-60 s deque logic is already
  time-based and stays.
- **PERCLOS:** each frame with a face: `tracker.update(ear < threshold, now)`.
  If `tracker.ready() and tracker.value() >= PERCLOS_THRESH` and the shared
  alert guards pass: fire an alert with reason
  `"Driver's eyes were closed {value:.0%} of the last minute."`, then
  `tracker.clear()` (same restart-the-window anti-spam pattern as the yawn
  deque).

Frames where no face is detected update nothing (same as current behavior);
PERCLOS samples simply don't accumulate during face-loss.

### 5. HUD

- `EAR: 0.27 (th 0.21)` — live EAR plus calibrated threshold; threshold text
  gets a `(default)` suffix when calibration fell back.
- `PERCLOS: 12%` — yellow normally, red when `>= PERCLOS_THRESH`.
- Existing MAR and yawn-count lines unchanged.

## Error handling

- Calibration with no/poor face detection -> default threshold, warn, run.
- Camera read failure during calibration or main loop -> existing error + exit
  path.
- `q` quits during calibration as well as during detection.
- No new exception sources in the alert path (pipeline untouched).

## Testing

New `tests/test_drowsy_logic.py` (pytest, no webcam, no network):

- `derive_threshold`: median robustness against blink dips (samples with ~10%
  low values barely move the result); ratio math; clamps at 0.15 / 0.30;
  fewer-than-30-samples fallback returns `(0.25, True)`.
- `PerclosTracker`: eviction of samples older than the window; `value()`
  fraction math (empty -> 0.0); `ready()` false before half-window, true
  after; `clear()` resets.
- Importing `drownsy` must stay side-effect-free (`__main__` guard already
  present).

Manual verification (webcam): calibration countdown appears and prints a
personal threshold; closing eyes ~1 s triggers the alert; PERCLOS HUD rises
during repeated closures; yawning still counts; `q` works in both phases.

## Out-of-scope follow-ups (recorded roadmap)

1. Evaluation harness + labeled data (NTHU-DDD request / self-recorded clips).
2. Head pose via `cv2.solvePnP` on the 68 landmarks.
3. 15-frame-window feature vector -> Random Forest per `jimaging-09-00091-v3`
   (99% benchmark).
