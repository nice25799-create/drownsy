# Calibration + PERCLOS + Time-Based Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed EAR thresholds and frame-count logic in `drownsy.py` with per-user startup calibration, a rolling PERCLOS metric, and seconds-based detection windows.

**Architecture:** Single-file app (`drownsy.py`) gains two testable units — a pure `derive_threshold()` function and a `PerclosTracker` class — plus an interactive `calibrate_ear()` startup phase. The main loop's frame counters become timestamp comparisons. New pytest suite covers the pure logic; webcam behavior is verified manually at the end.

**Tech Stack:** Python 3.11, OpenCV, dlib, imutils, pytest (new dev dependency). No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-06-11-calibration-perclos-design.md` (approved 2026-06-11)

**Working directory:** `D:\project` (all commands run from here)

---

## File Structure

- **Modify** `drownsy.py` — config block changes; add `derive_threshold()`, `PerclosTracker`, `calibrate_ear()`; rewrite the eye/mouth trigger logic inside `main()` to be time-based; add PERCLOS alert + HUD lines.
- **Create** `conftest.py` (project root, empty) — makes pytest put `D:\project` on `sys.path` so tests can `import drownsy`.
- **Create** `tests/test_drowsy_logic.py` — all unit tests (no webcam, no network).
- **Modify** `CLAUDE.md` and `.claude/skills/run-drowsy/SKILL.md` — tunables lists go stale after the config rename (Task 6).

Tasks 1–2 are TDD (pure logic). Tasks 3–5 are integration edits to the interactive loop — not unit-testable, so each ends with the full suite + an import smoke test to prove nothing broke. Task 6 is manual webcam verification + doc updates. The app must run after every task's commit.

---

### Task 1: pytest setup + `derive_threshold()`

**Files:**
- Create: `conftest.py` (empty, project root)
- Create: `tests/test_drowsy_logic.py`
- Modify: `drownsy.py` (imports + config block + new function)

- [ ] **Step 1: Ensure pytest is installed**

Run: `python -m pytest --version`
Expected: a version line like `pytest 8.x.x`. If instead you get `No module named pytest`, run: `pip install pytest` and re-check.

- [ ] **Step 2: Create empty `conftest.py` in the project root**

Create `D:\project\conftest.py` with no content (empty file). Without it, pytest does not put the project root on `sys.path`, and `import drownsy` fails inside `tests/`.

- [ ] **Step 3: Write the failing tests**

Create `tests/test_drowsy_logic.py`:

```python
import pytest

import drownsy


# ---------- derive_threshold ----------

def test_too_few_samples_falls_back_to_default():
    samples = [0.30] * (drownsy.MIN_CALIB_SAMPLES - 1)
    thresh, used_default = drownsy.derive_threshold(samples)
    assert thresh == drownsy.EAR_DEFAULT_THRESH
    assert used_default is True


def test_threshold_is_ratio_of_median():
    samples = [0.32] * 100
    thresh, used_default = drownsy.derive_threshold(samples)
    assert thresh == pytest.approx(drownsy.CALIB_RATIO * 0.32)  # 0.24
    assert used_default is False


def test_median_robust_to_blink_dips():
    # 10% of frames are blinks (very low EAR); median must ignore them
    samples = [0.32] * 90 + [0.05] * 10
    thresh, _ = drownsy.derive_threshold(samples)
    assert thresh == pytest.approx(drownsy.CALIB_RATIO * 0.32)


def test_threshold_clamped_low():
    samples = [0.10] * 100  # 0.75 * 0.10 = 0.075 -> clamp to 0.15
    thresh, _ = drownsy.derive_threshold(samples)
    assert thresh == drownsy.CALIB_THRESH_MIN


def test_threshold_clamped_high():
    samples = [0.50] * 100  # 0.75 * 0.50 = 0.375 -> clamp to 0.30
    thresh, _ = drownsy.derive_threshold(samples)
    assert thresh == drownsy.CALIB_THRESH_MAX
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_drowsy_logic.py -v`
Expected: 5 failures/errors, each `AttributeError: module 'drownsy' has no attribute ...` (the constants and function don't exist yet). The import of `drownsy` itself must succeed (cv2/dlib/etc. are installed).

- [ ] **Step 5: Add config constants and `derive_threshold` to `drownsy.py`**

Add to the imports block (after `from collections import deque`):

```python
from statistics import median
```

In the `# -------------------- CONFIGURATION --------------------` block, add after the `MOUTH_AR_*` lines (do NOT remove any old constants yet — the main loop still uses them until Tasks 3–4):

```python
CALIB_SECONDS = 10.0          # startup calibration capture duration
CALIB_RATIO = 0.75            # threshold = ratio x median open-eye EAR
CALIB_THRESH_MIN = 0.15       # clamp for the derived threshold
CALIB_THRESH_MAX = 0.30
MIN_CALIB_SAMPLES = 30        # fewer face-frames than this -> use default
EAR_DEFAULT_THRESH = 0.25     # fallback when calibration fails
```

Add this function directly after `mouth_aspect_ratio()`:

```python
def derive_threshold(samples):
    """Personal EAR threshold from calibration samples: ratio of the median."""
    if len(samples) < MIN_CALIB_SAMPLES:
        return EAR_DEFAULT_THRESH, True

    open_ear = median(samples)
    threshold = CALIB_RATIO * open_ear
    threshold = max(CALIB_THRESH_MIN, min(CALIB_THRESH_MAX, threshold))

    return threshold, False
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_drowsy_logic.py -v`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add conftest.py tests/test_drowsy_logic.py drownsy.py
git commit -m "test: add pytest setup and derive_threshold for personal EAR calibration"
```

---

### Task 2: `PerclosTracker`

**Files:**
- Modify: `tests/test_drowsy_logic.py` (append tests)
- Modify: `drownsy.py` (config + new class)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_drowsy_logic.py`:

```python
# ---------- PerclosTracker ----------

def make_tracker():
    return drownsy.PerclosTracker(window_sec=60.0)


def test_empty_tracker_is_zero_and_not_ready():
    tr = make_tracker()
    assert tr.value() == 0.0
    assert tr.ready() is False


def test_value_is_fraction_of_closed_samples():
    tr = make_tracker()
    for i in range(10):
        tr.update(closed=(i < 3), t=float(i))  # 3 closed out of 10
    assert tr.value() == pytest.approx(0.3)


def test_samples_older_than_window_are_evicted():
    tr = make_tracker()
    tr.update(True, 0.0)
    tr.update(False, 30.0)
    tr.update(False, 61.0)  # t=0 sample is now 61 s old -> evicted
    assert tr.value() == 0.0
    assert len(tr.samples) == 2


def test_ready_requires_half_window_of_history():
    tr = make_tracker()
    tr.update(False, 0.0)
    tr.update(False, 29.0)
    assert tr.ready() is False   # 29 s span < 30 s
    tr.update(False, 30.0)
    assert tr.ready() is True    # 30 s span


def test_clear_resets_window():
    tr = make_tracker()
    tr.update(True, 0.0)
    tr.update(True, 40.0)
    tr.clear()
    assert tr.value() == 0.0
    assert tr.ready() is False
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_drowsy_logic.py -v`
Expected: 5 passed (Task 1), 5 errors with `AttributeError: module 'drownsy' has no attribute 'PerclosTracker'`

- [ ] **Step 3: Add PERCLOS config and the class to `drownsy.py`**

In the configuration block, add after the `EAR_DEFAULT_THRESH` line:

```python
PERCLOS_WINDOW_SEC = 60.0     # rolling window for PERCLOS
PERCLOS_THRESH = 0.30         # PERCLOS value that fires an alert
```

Add this class directly after `derive_threshold()`:

```python
class PerclosTracker:
    """Rolling-window PERCLOS: fraction of recent frames with eyes closed."""

    def __init__(self, window_sec):
        self.window_sec = window_sec
        self.samples = deque()   # (timestamp, closed)

    def update(self, closed, t):
        self.samples.append((t, closed))

        while self.samples and (t - self.samples[0][0]) > self.window_sec:
            self.samples.popleft()

    def value(self):
        if not self.samples:
            return 0.0

        closed_count = sum(1 for _, closed in self.samples if closed)
        return closed_count / len(self.samples)

    def ready(self):
        if len(self.samples) < 2:
            return False

        span = self.samples[-1][0] - self.samples[0][0]
        return span >= 0.5 * self.window_sec

    def clear(self):
        self.samples.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_drowsy_logic.py -v`
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/test_drowsy_logic.py drownsy.py
git commit -m "feat: add PerclosTracker rolling-window eye-closure metric"
```

---

### Task 3: `calibrate_ear()` startup phase

Interactive webcam code — no unit tests; correctness is verified by the suite still passing (import safety) plus the manual checks in Task 6.

**Files:**
- Modify: `drownsy.py` (new function + wiring in `main()`; threshold variable replaces `EYE_AR_THRESH`)

- [ ] **Step 1: Add `calibrate_ear()` after `PerclosTracker`**

```python
def calibrate_ear(vs, detector, predictor, l_idx, r_idx):
    """Collect open-eye EAR samples for CALIB_SECONDS, then derive a threshold.

    Returns (threshold, used_default), or None if the user pressed q.
    """
    (lStart, lEnd) = l_idx
    (rStart, rEnd) = r_idx

    samples = []
    start = time.time()

    while True:
        remaining = CALIB_SECONDS - (time.time() - start)

        if remaining <= 0:
            break

        ret, frame = vs.read()

        if not ret:
            print("[ERROR] Failed to read frame during calibration.")
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        ear = None

        if rects:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)

            leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
            rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
            ear = (leftEAR + rightEAR) / 2.0
            samples.append(ear)

        cv2.putText(
            frame,
            f"CALIBRATING - look at camera, eyes open ({remaining:.0f}s)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2
        )

        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2
            )

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None

    threshold, used_default = derive_threshold(samples)

    if used_default:
        print(f"[WARN] Calibration failed ({len(samples)} samples); "
              f"using default EAR threshold {threshold:.2f}.")
    else:
        print(f"[INFO] Calibrated EAR threshold: {threshold:.3f} "
              f"(from {len(samples)} samples)")

    return threshold, used_default
```

- [ ] **Step 2: Wire calibration into `main()` and switch the loop to the calibrated threshold**

In `main()`, directly after `time.sleep(2.0)` and before the state variables, insert:

```python
    calib = calibrate_ear(
        vs, detector, predictor, (lStart, lEnd), (rStart, rEnd)
    )

    if calib is None:
        print("[INFO] Calibration aborted; exiting.")
        vs.release()
        cv2.destroyAllWindows()
        return

    ear_thresh, ear_thresh_is_default = calib
```

Then change the detection condition in the main loop from:

```python
            if ear < EYE_AR_THRESH:
```

to:

```python
            if ear < ear_thresh:
```

Finally, delete the now-unused `EYE_AR_THRESH = 0.25` line from the configuration block.

- [ ] **Step 3: Run the suite + import smoke test**

Run: `python -m pytest tests/test_drowsy_logic.py -v && python -c "import drownsy; print('import ok')"`
Expected: `10 passed`, then `import ok`

- [ ] **Step 4: Commit**

```bash
git add drownsy.py
git commit -m "feat: run per-user EAR calibration at startup"
```

---

### Task 4: Time-based eye and yawn logic

**Files:**
- Modify: `drownsy.py` (config block + state variables + the two trigger blocks in `main()`)

- [ ] **Step 1: Swap the frame-count constants for seconds**

In the configuration block, delete these two lines:

```python
EYE_AR_CONSEC_FRAMES = 20     # Consecutive frames below threshold to trigger alert
MOUTH_AR_CONSEC_FRAMES = 15   # ~0.5 s at 30 fps; filters out talking/laughing
```

and add in their places:

```python
EYE_CLOSED_ALERT_SEC = 1.0    # continuous eye closure that fires the alert
MOUTH_OPEN_YAWN_SEC = 0.5     # continuous open mouth that counts as one yawn
```

- [ ] **Step 2: Replace the state variables in `main()`**

Delete these lines (currently just before/inside the pre-loop state setup):

```python
    COUNTER = 0
    ALARM_ON = False

    MOUTH_COUNTER = 0
    yawn_counted = False        # current mouth-open episode already counted
    yawn_times = deque()        # timestamps of recent yawns

    drowsy_start = 0.0           # when the current eye-closure began
```

Replace with:

```python
    closure_start = None        # when the current eye-closure began
    ALARM_ON = False

    mouth_open_start = None     # when the current mouth-open episode began
    yawn_counted = False        # current mouth-open episode already counted
    yawn_times = deque()        # timestamps of recent yawns
```

(`episode_count`, `last_alert_time`, `alert_busy` stay unchanged.)

- [ ] **Step 3: Compute `now` once per face, right after MAR**

In the per-face section, after `mar = mouth_aspect_ratio(mouth)` and the mouth hull drawing, the existing yawn block starts. Insert one line *before* the `if mar > MOUTH_AR_THRESH:` line:

```python
            now = time.time()
```

and delete the later duplicate `now = time.time()` line that currently sits under the `# Drop yawns that fell out of the rolling window` comment (the eviction loop keeps using `now`). Also delete the second `now = time.time()` inside the drowsiness block (after the `DROWSINESS ALERT!` putText) — everything in this iteration uses the single `now`.

- [ ] **Step 4: Make the yawn trigger time-based**

Replace this block:

```python
            if mar > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1

                if MOUTH_COUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    # Count each open-mouth episode as one yawn
                    if not yawn_counted:
                        yawn_counted = True
                        yawn_times.append(time.time())
```

with:

```python
            if mar > MOUTH_AR_THRESH:
                if mouth_open_start is None:
                    mouth_open_start = now

                if (now - mouth_open_start) >= MOUTH_OPEN_YAWN_SEC:
                    # Count each open-mouth episode as one yawn
                    if not yawn_counted:
                        yawn_counted = True
                        yawn_times.append(now)
```

and replace its `else` branch:

```python
            else:
                MOUTH_COUNTER = 0
                yawn_counted = False
```

with:

```python
            else:
                mouth_open_start = None
                yawn_counted = False
```

(The `YAWNING` putText inside the block stays exactly where it is.)

- [ ] **Step 5: Make the eye-closure trigger time-based**

Replace the drowsiness block:

```python
            if ear < ear_thresh:
                if COUNTER == 0:
                    drowsy_start = time.time()   # episode begins

                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
```

with:

```python
            if ear < ear_thresh:
                if closure_start is None:
                    closure_start = now   # episode begins

                closed_dur = now - closure_start

                if closed_dur >= EYE_CLOSED_ALERT_SEC:
```

Inside that block, the alert code currently reads:

```python
                    now = time.time()

                    # Fire one spoken alert per episode, respecting the cooldown,
                    # and only if no alert is already being spoken.
                    if (not ALARM_ON
                            and not alert_busy.is_set()
                            and (now - last_alert_time) > ALERT_COOLDOWN):
                        ALARM_ON = True
                        last_alert_time = now
                        episode_count += 1
                        duration = now - drowsy_start
                        reason = (f"Eyes were closed for about "
                                  f"{duration:.0f} seconds.")
```

Change it to (drop the inner `now =` line per Step 3, use `closed_dur`):

```python
                    # Fire one spoken alert per episode, respecting the cooldown,
                    # and only if no alert is already being spoken.
                    if (not ALARM_ON
                            and not alert_busy.is_set()
                            and (now - last_alert_time) > ALERT_COOLDOWN):
                        ALARM_ON = True
                        last_alert_time = now
                        episode_count += 1
                        reason = (f"Eyes were closed for about "
                                  f"{closed_dur:.0f} seconds.")
```

And its `else` branch:

```python
            else:
                COUNTER = 0
                ALARM_ON = False
```

becomes:

```python
            else:
                closure_start = None
                ALARM_ON = False
```

- [ ] **Step 6: Run the suite + import smoke test**

Run: `python -m pytest tests/test_drowsy_logic.py -v && python -c "import drownsy; print('import ok')"`
Expected: `10 passed`, then `import ok`

- [ ] **Step 7: Commit**

```bash
git add drownsy.py
git commit -m "feat: replace frame-count detection with time-based thresholds"
```

---

### Task 5: PERCLOS alert + HUD

**Files:**
- Modify: `drownsy.py` (`main()` only)

- [ ] **Step 1: Create the tracker before the loop**

In `main()`, next to the state variables added in Task 4, add:

```python
    perclos_tracker = PerclosTracker(PERCLOS_WINDOW_SEC)
```

- [ ] **Step 2: Feed the tracker every face-frame**

Directly after the `now = time.time()` line (Task 4 Step 3), add:

```python
            perclos_tracker.update(ear < ear_thresh, now)
            perclos = perclos_tracker.value()
```

- [ ] **Step 3: Fire the PERCLOS alert**

Directly after the existing yawn-frequency alert block (the one ending with the `threading.Thread(...).start()` for yawns), add:

```python
            # Sustained partial closure over the window -> spoken alert
            if (perclos_tracker.ready()
                    and perclos >= PERCLOS_THRESH
                    and not alert_busy.is_set()
                    and (now - last_alert_time) > ALERT_COOLDOWN):
                last_alert_time = now
                episode_count += 1
                reason = (f"Driver's eyes were closed "
                          f"{perclos:.0%} of the last minute.")
                perclos_tracker.clear()   # restart the window after alerting

                alert_busy.set()
                threading.Thread(
                    target=handle_alert,
                    args=(client, episode_count, reason, alert_busy),
                    daemon=True,
                ).start()
```

- [ ] **Step 4: Update the HUD**

Replace the EAR putText:

```python
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
```

with:

```python
            thresh_note = " default" if ear_thresh_is_default else ""

            cv2.putText(
                frame,
                f"EAR: {ear:.2f} (th {ear_thresh:.2f}{thresh_note})",
                (230, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )
```

After the MAR putText block, add:

```python
            perclos_color = (
                (0, 0, 255) if perclos >= PERCLOS_THRESH else (0, 255, 255)
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.0%}",
                (300, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                perclos_color,
                2
            )
```

- [ ] **Step 5: Run the suite + import smoke test**

Run: `python -m pytest tests/test_drowsy_logic.py -v && python -c "import drownsy; print('import ok')"`
Expected: `10 passed`, then `import ok`

- [ ] **Step 6: Commit**

```bash
git add drownsy.py
git commit -m "feat: add PERCLOS alert and HUD readout"
```

---

### Task 6: Manual webcam verification + doc updates

**Files:**
- Modify: `CLAUDE.md` (tunables sentence)
- Modify: `.claude/skills/run-drowsy/SKILL.md` (tunables sentence)

- [ ] **Step 1: Manual verification (needs the user at the webcam)**

Run: `python drownsy.py` — then check, in order:

1. Calibration phase: countdown text appears for ~10 s; console then prints `[INFO] Calibrated EAR threshold: ...` (a personal value, usually 0.18–0.26).
2. HUD shows `EAR: x.xx (th y.yy)` with the calibrated threshold, and `PERCLOS: n%`.
3. Close eyes ~1 s → `DROWSINESS ALERT!` text + one spoken alert.
4. Yawn (or hold mouth wide open ≥0.5 s) three times within a minute → yawn alert still works.
5. Half-close/blink heavily for ~30+ s → PERCLOS % climbs and turns red at 30%, then a PERCLOS alert fires.
6. `q` quits cleanly during calibration; `q` quits cleanly during detection (run twice to check both).

If any check fails: STOP, debug with the systematic-debugging skill before continuing.

- [ ] **Step 2: Update `CLAUDE.md`**

In the Architecture section, replace the tunables sentence:

```
- **Tunables** are a labeled config block at the top of `drownsy.py` (`EYE_AR_THRESH`, `EYE_AR_CONSEC_FRAMES`, `MOUTH_AR_THRESH`, `MOUTH_AR_CONSEC_FRAMES`, `YAWN_WINDOW_SEC`, `YAWN_ALERT_COUNT`, `CLAUDE_MODEL`, `ALERT_COOLDOWN`). Threshold tuning is the most common edit.
```

with:

```
- **Tunables** are a labeled config block at the top of `drownsy.py` (calibration: `CALIB_SECONDS`, `CALIB_RATIO`, clamps, `EAR_DEFAULT_THRESH`; detection: `EYE_CLOSED_ALERT_SEC`, `MOUTH_AR_THRESH`, `MOUTH_OPEN_YAWN_SEC`, `PERCLOS_WINDOW_SEC`, `PERCLOS_THRESH`, `YAWN_WINDOW_SEC`, `YAWN_ALERT_COUNT`; alerts: `CLAUDE_MODEL`, `ALERT_COOLDOWN`). Threshold tuning is the most common edit. The EAR threshold itself is calibrated per user at startup (10 s), not hardcoded.
```

Also update the "two independent detection paths" bullet to mention PERCLOS: change "*Eye closure*: Eye Aspect Ratio (EAR) averaged over both eyes. EAR below `EYE_AR_THRESH` for `EYE_AR_CONSEC_FRAMES` consecutive frames fires a drowsiness alert (one per closure episode)." to "*Eye closure*: Eye Aspect Ratio (EAR) averaged over both eyes, against a per-user threshold calibrated at startup. Continuous closure ≥ `EYE_CLOSED_ALERT_SEC` fires an alert (one per episode); separately, PERCLOS (fraction of the last 60 s spent closed) ≥ `PERCLOS_THRESH` fires a slow-droop alert."

- [ ] **Step 3: Update `.claude/skills/run-drowsy/SKILL.md`**

Replace its closing line:

```
This is an interactive GUI app — run it in the foreground. Key tunables are at the top of drownsy.py: EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, CLAUDE_MODEL, ALERT_COOLDOWN.
```

with:

```
This is an interactive GUI app — run it in the foreground. On launch it runs a ~10 s EAR calibration (user looks at the camera, eyes open) before detection starts. Key tunables are at the top of drownsy.py: CALIB_SECONDS, CALIB_RATIO, EYE_CLOSED_ALERT_SEC, PERCLOS_THRESH, CLAUDE_MODEL, ALERT_COOLDOWN.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md .claude/skills/run-drowsy/SKILL.md
git commit -m "docs: update tunables for calibration + PERCLOS"
```

---

## Spec coverage map

| Spec section | Task |
|---|---|
| Config block (add/remove constants) | 1, 2, 4 |
| `derive_threshold` (median, ratio, clamps, fallback) | 1 |
| `PerclosTracker` (update/value/ready/clear) | 2 |
| `calibrate_ear` (UI, q-abort, warn-on-default) | 3 |
| Time-based eye closure + yawn hold | 4 |
| PERCLOS alert (guards, clear-after-alert) | 5 |
| HUD (threshold display, PERCLOS %, colors) | 5 |
| Error handling (camera fail, abort, fallback) | 3 (code), 6 (verified) |
| Tests | 1, 2 |
| Manual verification checklist | 6 |
