# Head-Pose Gating + Slump Alert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop downward-gaze false eye-closure alarms by gating EAR/PERCLOS on head pose (solvePnP pitch/yaw vs a per-user neutral baseline), and add a dedicated alert for a sustained head slump.

**Architecture:** Three new pure units in `drownsy.py` (same single-file pattern as `derive_threshold`/`PerclosTracker`): `estimate_head_pose` (solvePnP → Euler degrees, positive pitch = looking down), `PoseGate` (EMA smoothing, baseline-relative gate + slump state machine), `derive_pose_baseline` (medians from calibration). The main loop pauses the eye paths while gated and fires a `[TRIGGER slump]` alert through the existing alert pipeline. Spec: `docs/superpowers/specs/2026-06-12-head-pose-design.md`.

**Tech Stack:** Python 3.11, OpenCV (`cv2.solvePnP`, `cv2.Rodrigues`, `cv2.projectPoints` in tests), numpy, dlib 68-landmarks (already in use), pytest.

---

## Context for a zero-context engineer

- `drownsy.py` is the entire app: config constants at top, pure helpers, then `main()` with a webcam loop. Run tests with `python -m pytest -v` from `D:\project` (a one-line `conftest.py` at the repo root makes `import drownsy` resolve). Tests must never need a webcam or network.
- Line numbers below refer to `drownsy.py` at commit `e0b632b` and shift as tasks land — anchor on the quoted code, not the numbers.
- dlib landmark indices used for pose: nose tip 30, chin 8, left eye outer corner 36, right eye outer corner 45, mouth corners 48/54.
- Sign convention (pinned by tests): **positive pitch = looking down**, ~0° = facing the camera. The raw Euler decomposition of this 3D model yields pitch near ±180° for a frontal face; `estimate_head_pose` wraps it.
- The webcam frame is resized to width 450 before processing, so the camera intrinsics are approximated from the *resized* frame size.

## File structure

- **Modify:** `drownsy.py`
  - imports: add `math`, `numpy as np`
  - config block: 5 new tunables
  - after `mouth_aspect_ratio` (~line 58): `POSE_LANDMARK_IDXS`, `POSE_MODEL_POINTS`, `estimate_head_pose`
  - after `derive_threshold` (~line 69): `derive_pose_baseline`
  - after `PerclosTracker` (~line 100): `PoseGate`
  - `calibrate_ear` (~lines 103–176): also collect pitch/yaw samples, return baseline
  - `main()` (~lines 232–515): construct `PoseGate`, per-frame pose + gating, slump trigger, face-lost reset, HUD line
- **Create:** `tests/test_head_pose.py` — all new tests (pose math round-trip, `PoseGate`, `derive_pose_baseline`). `tests/test_drowsy_logic.py` stays untouched.

---

### Task 0: Branch

- [x] **Step 1: Create the feature branch**

```bash
git checkout -b feature/head-pose-gating
```

Run: `git branch --show-current`
Expected: `feature/head-pose-gating`

---

### Task 1: Pose math — `estimate_head_pose`

**Files:**
- Modify: `drownsy.py` (imports, config block, new constants + function after `mouth_aspect_ratio`)
- Create: `tests/test_head_pose.py`

- [x] **Step 1: Write the failing tests**

Create `tests/test_head_pose.py` with a synthetic-projection helper and round-trip tests. The helper builds a rotation the same way the camera sees a head (frontal face = 180° about X for this model), projects the 3D model points with `cv2.projectPoints`, and plants them in a fake 68-point landmark array.

```python
import math

import cv2
import numpy as np
import pytest

import drownsy


# ---------- estimate_head_pose ----------

FRAME_W, FRAME_H = 450, 340


def project_pose(pitch_down_deg, yaw_deg):
    """Fake 68-landmark array for a head at a known pitch/yaw.

    Rotation convention matches the app: a frontal face is a 180-deg
    rotation about X (model +Y is up, camera +Y is down); looking down
    adds positive pitch.
    """
    a = math.radians(180.0 + pitch_down_deg)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(a), -math.sin(a)],
        [0.0, math.sin(a), math.cos(a)],
    ])

    b = math.radians(yaw_deg)
    ry = np.array([
        [math.cos(b), 0.0, math.sin(b)],
        [0.0, 1.0, 0.0],
        [-math.sin(b), 0.0, math.cos(b)],
    ])

    rot = ry @ rx
    rvec, _ = cv2.Rodrigues(rot)
    tvec = np.array([[0.0], [0.0], [1000.0]])

    focal = float(FRAME_W)
    cam = np.array([
        [focal, 0.0, FRAME_W / 2.0],
        [0.0, focal, FRAME_H / 2.0],
        [0.0, 0.0, 1.0],
    ])

    pts, _ = cv2.projectPoints(
        drownsy.POSE_MODEL_POINTS, rvec, tvec, cam, np.zeros((4, 1))
    )
    pts = pts.reshape(-1, 2)

    shape = np.zeros((68, 2))
    for idx, pt in zip(drownsy.POSE_LANDMARK_IDXS, pts):
        shape[idx] = pt

    return shape


def estimate(pitch_down_deg, yaw_deg):
    shape = project_pose(pitch_down_deg, yaw_deg)
    pose = drownsy.estimate_head_pose(shape, FRAME_W, FRAME_H)
    assert pose is not None
    return pose


def test_frontal_face_reads_near_zero():
    pitch, yaw, roll = estimate(0.0, 0.0)
    assert pitch == pytest.approx(0.0, abs=2.0)
    assert yaw == pytest.approx(0.0, abs=2.0)
    assert roll == pytest.approx(0.0, abs=2.0)


def test_looking_down_is_positive_pitch():
    pitch, yaw, _ = estimate(20.0, 0.0)
    assert pitch == pytest.approx(20.0, abs=2.0)
    assert yaw == pytest.approx(0.0, abs=2.0)


def test_looking_up_is_negative_pitch():
    pitch, _, _ = estimate(-15.0, 0.0)
    assert pitch == pytest.approx(-15.0, abs=2.0)


def test_yaw_recovered_with_opposite_signs():
    _, yaw_a, _ = estimate(0.0, 20.0)
    _, yaw_b, _ = estimate(0.0, -20.0)
    assert abs(yaw_a) == pytest.approx(20.0, abs=2.0)
    assert abs(yaw_b) == pytest.approx(20.0, abs=2.0)
    assert (yaw_a > 0) != (yaw_b > 0)


def test_combined_pitch_and_yaw():
    pitch, yaw, _ = estimate(15.0, 20.0)
    assert pitch == pytest.approx(15.0, abs=2.0)
    assert abs(yaw) == pytest.approx(20.0, abs=2.0)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_head_pose.py -v`
Expected: collection error / failures with `AttributeError: module 'drownsy' has no attribute 'POSE_MODEL_POINTS'` (the module imports fine; the new names don't exist yet).

- [x] **Step 3: Implement the pose function**

Three edits to `drownsy.py`.

**3a — imports.** The top of the file currently reads:

```python
import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import os
import threading
```

Change to (add `math` and `numpy`):

```python
import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import math
import time
import os
import threading
```

**3b — config block.** After the `YAWN_ALERT_COUNT = 3` line and before the `CLAUDE_MODEL` line, insert:

```python
POSE_GATE_PITCH_DEG = 10.0    # look down past this (vs neutral) -> gate eye metrics
POSE_GATE_YAW_DEG = 15.0      # turn past this (either way) -> gate eye metrics
SLUMP_PITCH_DEG = 25.0        # pitch-down past this = possible head slump
SLUMP_ALERT_SEC = 2.0         # slump sustained this long -> alert
POSE_EMA_ALPHA = 0.3          # EMA smoothing factor for head-pose angles
```

**3c — constants + function.** Immediately after `mouth_aspect_ratio` (after its `return` line, before `def derive_threshold`), insert:

```python
POSE_LANDMARK_IDXS = (30, 8, 36, 45, 48, 54)

POSE_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # nose tip            (landmark 30)
    (0.0, -330.0, -65.0),       # chin                (landmark 8)
    (-225.0, 170.0, -135.0),    # left eye outer corner   (landmark 36)
    (225.0, 170.0, -135.0),     # right eye outer corner  (landmark 45)
    (-150.0, -150.0, -125.0),   # left mouth corner       (landmark 48)
    (150.0, -150.0, -125.0),    # right mouth corner      (landmark 54)
], dtype=np.float64)


def estimate_head_pose(shape, frame_w, frame_h):
    """Head pose from 6 landmarks via solvePnP.

    Returns (pitch, yaw, roll) in degrees, where ~0 means facing the
    camera and positive pitch means looking down, or None on failure.
    """
    image_points = np.array(
        [shape[i] for i in POSE_LANDMARK_IDXS], dtype=np.float64
    )

    focal = float(frame_w)
    camera_matrix = np.array([
        [focal, 0.0, frame_w / 2.0],
        [0.0, focal, frame_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    try:
        ok, rvec, _ = cv2.solvePnP(
            POSE_MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None

    if not ok:
        return None

    rot, _ = cv2.Rodrigues(rvec)

    sy = math.hypot(rot[0, 0], rot[1, 0])

    if sy < 1e-6:   # gimbal lock; bail out rather than guess
        return None

    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], sy))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))

    # A frontal face decomposes to pitch near +-180 with this model;
    # wrap so straight-ahead reads ~0 and looking down is positive.
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0

    return pitch, yaw, roll
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_head_pose.py -v`
Expected: 5 passed.

Run: `python -m pytest -v`
Expected: 15 passed (10 existing + 5 new); proves `drownsy.py` still imports cleanly.

- [x] **Step 5: Commit**

```bash
git add drownsy.py tests/test_head_pose.py
git commit -m "feat: add solvePnP head-pose estimation"
```

---

### Task 2: `PoseGate` — gate + slump state machine

**Files:**
- Modify: `drownsy.py` (new class after `PerclosTracker`)
- Test: `tests/test_head_pose.py` (append)

- [x] **Step 1: Write the failing tests**

Append to `tests/test_head_pose.py`. `ema_alpha=1.0` disables smoothing so most tests are deterministic; one test exercises the EMA explicitly.

```python
# ---------- PoseGate ----------

def make_gate(**kw):
    defaults = dict(
        gate_pitch_deg=10.0,
        gate_yaw_deg=15.0,
        slump_pitch_deg=25.0,
        slump_alert_sec=2.0,
        ema_alpha=1.0,
        lazy_baseline_frames=5,
    )
    defaults.update(kw)
    return drownsy.PoseGate(**defaults)


def test_ungated_before_any_update():
    g = make_gate()
    g.set_baseline(0.0, 0.0)
    assert g.is_gated() is False
    assert g.slump_alert_due(0.0) is False


def test_gate_on_pitch_down_only():
    g = make_gate()
    g.set_baseline(5.0, 0.0)

    g.update((16.0, 0.0, 0.0), 0.0)   # rel pitch +11 -> past 10
    assert g.is_gated() is True

    g.update((14.0, 0.0, 0.0), 1.0)   # rel pitch +9 -> under 10
    assert g.is_gated() is False

    g.update((-7.0, 0.0, 0.0), 2.0)   # rel pitch -12 = looking UP
    assert g.is_gated() is False


def test_gate_on_yaw_either_direction():
    g = make_gate()
    g.set_baseline(0.0, -2.0)

    g.update((0.0, 14.0, 0.0), 0.0)   # rel yaw +16 -> past 15
    assert g.is_gated() is True

    g.update((0.0, -18.0, 0.0), 1.0)  # rel yaw -16 -> past 15
    assert g.is_gated() is True

    g.update((0.0, 10.0, 0.0), 2.0)   # rel yaw +12 -> under 15
    assert g.is_gated() is False


def test_ema_smoothing_blends_samples():
    g = make_gate(ema_alpha=0.5)
    g.set_baseline(0.0, 0.0)

    g.update((0.0, 0.0, 0.0), 0.0)    # first sample taken raw
    g.update((10.0, 0.0, 0.0), 1.0)   # smoothed = 0.5*10 + 0.5*0 = 5
    assert g.rel_pitch == pytest.approx(5.0)


def test_none_pose_ungates_and_resets_slump():
    g = make_gate()
    g.set_baseline(0.0, 0.0)

    g.update((30.0, 0.0, 0.0), 0.0)   # slump episode starts
    assert g.is_gated() is True

    g.update(None, 1.0)               # solvePnP failed this frame
    assert g.is_gated() is False
    assert g.slump_alert_due(10.0) is False   # timer did not keep running


def test_slump_fires_once_after_duration():
    g = make_gate()
    g.set_baseline(0.0, 0.0)

    g.update((30.0, 0.0, 0.0), 0.0)
    assert g.slump_alert_due(1.9) is False    # under 2.0 s
    g.update((30.0, 0.0, 0.0), 2.0)
    assert g.slump_alert_due(2.0) is True     # fires at the threshold
    g.update((30.0, 0.0, 0.0), 3.0)
    assert g.slump_alert_due(3.0) is False    # once per episode


def test_slump_rearms_after_recovery():
    g = make_gate()
    g.set_baseline(0.0, 0.0)

    g.update((30.0, 0.0, 0.0), 0.0)
    assert g.slump_alert_due(2.5) is True

    g.update((0.0, 0.0, 0.0), 3.0)            # head back up
    g.update((30.0, 0.0, 0.0), 4.0)           # new episode
    assert g.slump_alert_due(6.5) is True


def test_no_baseline_never_gates():
    g = make_gate(lazy_baseline_frames=100)

    for i in range(50):
        g.update((40.0, 0.0, 0.0), float(i))  # extreme pitch, no baseline yet
        assert g.is_gated() is False
        assert g.slump_alert_due(float(i)) is False


def test_lazy_baseline_from_first_frames():
    g = make_gate(lazy_baseline_frames=5)

    for i in range(5):
        g.update((10.0, 2.0, 0.0), float(i))  # neutral-ish frames

    assert g.baseline == pytest.approx((10.0, 2.0))

    g.update((25.0, 2.0, 0.0), 5.0)           # rel pitch +15 vs lazy baseline
    assert g.is_gated() is True


def test_reset_transient_keeps_baseline():
    g = make_gate()
    g.set_baseline(0.0, 0.0)
    g.update((30.0, 0.0, 0.0), 0.0)

    g.reset_transient()                       # face lost

    assert g.baseline == (0.0, 0.0)
    assert g.is_gated() is False
    assert g.slump_alert_due(10.0) is False

    g.update((30.0, 0.0, 0.0), 11.0)          # face back: fresh episode
    assert g.is_gated() is True
    assert g.slump_alert_due(13.0) is True
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_head_pose.py -v`
Expected: the 5 pose-math tests pass; the 10 new tests fail with `AttributeError: module 'drownsy' has no attribute 'PoseGate'`.

- [x] **Step 3: Implement `PoseGate`**

In `drownsy.py`, immediately after the `PerclosTracker` class (after its `clear` method, before `def calibrate_ear`), insert:

```python
class PoseGate:
    """Head-pose gate + slump detector, relative to a neutral baseline.

    Feed update() one (pitch, yaw, roll) per frame (or None when the
    solve failed). Angles are EMA-smoothed; gate/slump decisions use
    smoothed-minus-baseline values. Positive pitch = looking down.
    """

    def __init__(self, gate_pitch_deg, gate_yaw_deg, slump_pitch_deg,
                 slump_alert_sec, ema_alpha, lazy_baseline_frames=60):
        self.gate_pitch_deg = gate_pitch_deg
        self.gate_yaw_deg = gate_yaw_deg
        self.slump_pitch_deg = slump_pitch_deg
        self.slump_alert_sec = slump_alert_sec
        self.ema_alpha = ema_alpha
        self.lazy_baseline_frames = lazy_baseline_frames

        self.baseline = None      # (pitch, yaw) neutral, once known
        self.pitch = None         # EMA-smoothed absolute angles
        self.yaw = None
        self.roll = None
        self.rel_pitch = None     # smoothed minus baseline (None until valid)
        self.rel_yaw = None
        self._lazy_samples = []
        self._slump_start = None
        self._slump_alerted = False

    def set_baseline(self, pitch, yaw):
        self.baseline = (pitch, yaw)

    def update(self, pose, now):
        if pose is None:
            self.rel_pitch = None
            self.rel_yaw = None
            self._slump_start = None
            self._slump_alerted = False
            return

        pitch, yaw, roll = pose

        if self.pitch is None:
            self.pitch, self.yaw, self.roll = pitch, yaw, roll
        else:
            a = self.ema_alpha
            self.pitch = a * pitch + (1.0 - a) * self.pitch
            self.yaw = a * yaw + (1.0 - a) * self.yaw
            self.roll = a * roll + (1.0 - a) * self.roll

        if self.baseline is None:
            self._lazy_samples.append((pitch, yaw))

            if len(self._lazy_samples) < self.lazy_baseline_frames:
                return   # still collecting; stay ungated

            self.baseline = (
                median([p for p, _ in self._lazy_samples]),
                median([y for _, y in self._lazy_samples]),
            )

        self.rel_pitch = self.pitch - self.baseline[0]
        self.rel_yaw = self.yaw - self.baseline[1]

        if self.rel_pitch > self.slump_pitch_deg:
            if self._slump_start is None:
                self._slump_start = now
        else:
            self._slump_start = None
            self._slump_alerted = False

    def is_gated(self):
        if self.rel_pitch is None:
            return False

        return (self.rel_pitch > self.gate_pitch_deg
                or abs(self.rel_yaw) > self.gate_yaw_deg)

    def slump_elapsed(self, now):
        if self._slump_start is None:
            return 0.0

        return now - self._slump_start

    def slump_alert_due(self, now):
        """True exactly once per slump episode; only call when ready to alert."""
        if self._slump_start is None or self._slump_alerted:
            return False

        if (now - self._slump_start) >= self.slump_alert_sec:
            self._slump_alerted = True
            return True

        return False

    def reset_transient(self):
        """On face loss: drop smoothing + slump state, keep the baseline."""
        self.pitch = self.yaw = self.roll = None
        self.rel_pitch = self.rel_yaw = None
        self._slump_start = None
        self._slump_alerted = False
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_head_pose.py -v`
Expected: 15 passed.

Run: `python -m pytest -v`
Expected: 25 passed.

- [x] **Step 5: Commit**

```bash
git add drownsy.py tests/test_head_pose.py
git commit -m "feat: add PoseGate head-pose gate and slump detector"
```

---

### Task 3: Calibration baseline — `derive_pose_baseline` + wiring

**Files:**
- Modify: `drownsy.py` (`derive_pose_baseline` after `derive_threshold`; `calibrate_ear`; `main()` setup)
- Test: `tests/test_head_pose.py` (append)

After this task the app still runs exactly as before — `main()` constructs a `PoseGate` and prints the baseline, but the detection loop doesn't use it yet (that's Task 4).

- [x] **Step 1: Write the failing tests**

Append to `tests/test_head_pose.py`:

```python
# ---------- derive_pose_baseline ----------

def test_pose_baseline_is_median_of_samples():
    pitch = [4.0] * 90 + [40.0] * 10   # a few look-down frames must not skew it
    yaw = [-1.0] * 100
    baseline = drownsy.derive_pose_baseline(pitch, yaw)
    assert baseline == pytest.approx((4.0, -1.0))


def test_pose_baseline_none_when_too_few_samples():
    n = drownsy.MIN_CALIB_SAMPLES - 1
    assert drownsy.derive_pose_baseline([0.0] * n, [0.0] * n) is None
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_head_pose.py -v`
Expected: 15 pass, 2 fail with `AttributeError: module 'drownsy' has no attribute 'derive_pose_baseline'`.

- [x] **Step 3: Implement and wire calibration**

Three edits to `drownsy.py`.

**3a — pure function.** Immediately after `derive_threshold` (after its `return threshold, False` line, before `class PerclosTracker`), insert:

```python
def derive_pose_baseline(pitch_samples, yaw_samples):
    """Neutral (pitch, yaw) from calibration samples, or None if too few."""
    if len(pitch_samples) < MIN_CALIB_SAMPLES:
        return None

    return (median(pitch_samples), median(yaw_samples))
```

**3b — collect pose during calibration.** In `calibrate_ear`:

The docstring's return description changes from `Returns (threshold, used_default), or None if the user pressed q.` to:

```python
    Returns (threshold, used_default, pose_baseline) — pose_baseline is
    (pitch, yaw) or None — or None overall if the user pressed q.
```

Below `samples = []` add:

```python
    pitch_samples = []
    yaw_samples = []
```

Inside the `if rects:` block, the EAR code currently ends with `samples.append(ear)`. After that line (still inside `if rects:`), add:

```python
            pose = estimate_head_pose(
                shape, frame.shape[1], frame.shape[0]
            )

            if pose is not None:
                pitch_samples.append(pose[0])
                yaw_samples.append(pose[1])
```

The function currently ends with:

```python
    threshold, used_default = derive_threshold(samples)

    if used_default:
        print(f"[WARN] Calibration failed ({len(samples)} samples); "
              f"using default EAR threshold {threshold:.2f}.")
    else:
        print(f"[INFO] Calibrated EAR threshold: {threshold:.3f} "
              f"(from {len(samples)} samples)")

    return threshold, used_default
```

Replace with:

```python
    threshold, used_default = derive_threshold(samples)
    pose_baseline = derive_pose_baseline(pitch_samples, yaw_samples)

    if used_default:
        print(f"[WARN] Calibration failed ({len(samples)} samples); "
              f"using default EAR threshold {threshold:.2f}.")
    else:
        print(f"[INFO] Calibrated EAR threshold: {threshold:.3f} "
              f"(from {len(samples)} samples)")

    if pose_baseline is None:
        print(f"[WARN] No head-pose baseline ({len(pitch_samples)} pose "
              "samples); will capture one during detection.")
    else:
        print(f"[INFO] Neutral head pose: pitch {pose_baseline[0]:+.1f}, "
              f"yaw {pose_baseline[1]:+.1f} deg")

    return threshold, used_default, pose_baseline
```

**3c — unpack in `main()`.** The setup currently reads:

```python
    ear_thresh, ear_thresh_is_default = calib
```

Replace with:

```python
    ear_thresh, ear_thresh_is_default, pose_baseline = calib
```

And right after the `perclos_tracker = PerclosTracker(PERCLOS_WINDOW_SEC)` line, add:

```python
    pose_gate = PoseGate(
        POSE_GATE_PITCH_DEG,
        POSE_GATE_YAW_DEG,
        SLUMP_PITCH_DEG,
        SLUMP_ALERT_SEC,
        POSE_EMA_ALPHA,
    )

    if pose_baseline is not None:
        pose_gate.set_baseline(*pose_baseline)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -v`
Expected: 27 passed.

Also verify the module still imports (catches `main()` typos without a webcam):
Run: `python -c "import drownsy; print('import ok')"`
Expected: `import ok`

- [x] **Step 5: Commit**

```bash
git add drownsy.py tests/test_head_pose.py
git commit -m "feat: capture neutral head-pose baseline during calibration"
```

---

### Task 4: Main-loop integration — gate the eye paths, fire the slump alert

**Files:**
- Modify: `drownsy.py` (`main()` detection loop only)

No new unit tests: this task is webcam-loop glue around units already tested in Tasks 1–3 (consistent with how the rest of `main()` is covered). Verification = full suite green + clean import; live behavior is checked in Task 6.

- [x] **Step 1: Compute frame size once per frame**

In the `while True:` loop, right after `frame = imutils.resize(frame, width=450)`, add:

```python
        frame_h, frame_w = frame.shape[:2]
```

- [x] **Step 2: Per-frame pose + gated PERCLOS sampling**

Inside the face loop, the code currently reads:

```python
            # Yawn detection: mouth wide open for enough consecutive time
            now = time.time()
            perclos_tracker.update(ear < ear_thresh, now)
            perclos = perclos_tracker.value()
```

Replace with:

```python
            now = time.time()

            # Head pose: gate the eye metrics while looking away/down
            pose = estimate_head_pose(shape, frame_w, frame_h)
            pose_gate.update(pose, now)
            gated = pose_gate.is_gated()

            if not gated:
                perclos_tracker.update(ear < ear_thresh, now)

            perclos = perclos_tracker.value()

            # Yawn detection: mouth wide open for enough consecutive time
```

(The replacement re-homes the `# Yawn detection` comment so it sits directly above the `if mar > MOUTH_AR_THRESH:` block that follows.)

- [x] **Step 3: Slump alert block**

After the PERCLOS alert block (the one ending with the `threading.Thread(...).start()` that follows `perclos_tracker.clear()`), and before the `# Drowsiness detection` comment, insert:

```python
            # Head slumped far down for long enough -> spoken alert.
            # Busy/cooldown checks come FIRST: slump_alert_due() consumes
            # the episode's single shot, so only ask when we can alert.
            if (not alert_busy.is_set()
                    and (now - last_alert_time) > ALERT_COOLDOWN
                    and pose_gate.slump_alert_due(now)):
                last_alert_time = now
                episode_count += 1
                reason = (f"Driver's head slumped forward for about "
                          f"{pose_gate.slump_elapsed(now):.0f} seconds.")
                print(f"[TRIGGER slump] {reason} "
                      f"(pitch {pose_gate.rel_pitch:+.0f} deg)", flush=True)

                alert_busy.set()
                threading.Thread(
                    target=handle_alert,
                    args=(client, episode_count, reason, alert_busy),
                    daemon=True,
                ).start()
```

- [x] **Step 4: Gate the eye-closure path**

The eye-closure path currently starts with:

```python
            # Drowsiness detection
            if ear < ear_thresh:
```

Replace those two lines with:

```python
            # Drowsiness detection (paused while the pose gate is active)
            if ear < ear_thresh and not gated:
```

Leave the body and the `else:` branch untouched — when gated, control now falls into the existing `else`, which resets `closure_start`/`ALARM_ON`. That is exactly the spec behavior.

- [x] **Step 5: Reset pose state on face loss**

After the `for rect in rects:` loop body ends (dedent back to the `while` level, just before `# Show frame` / `cv2.imshow("Frame", frame)`), add:

```python
        if not rects:
            pose_gate.reset_transient()
```

- [x] **Step 6: Verify**

Run: `python -m pytest -v`
Expected: 27 passed.

Run: `python -c "import drownsy; print('import ok')"`
Expected: `import ok`

- [ ] **Step 7: Commit**

```bash
git add drownsy.py
git commit -m "feat: gate eye metrics on head pose and add slump alert"
```

---

### Task 5: HUD — pose readout + gate indicator

**Files:**
- Modify: `drownsy.py` (`main()` display section only)

- [x] **Step 1: Add the pose line and gate tag**

In the display section of the face loop, after the `cv2.putText(... f"PERCLOS: {perclos:.0%}" ...)` call and before the `cv2.putText(... f"Yawns(60s): ..." ...)` call, insert:

```python
            if pose_gate.rel_pitch is not None:
                pose_text = (f"P {pose_gate.rel_pitch:+.0f}  "
                             f"Y {pose_gate.rel_yaw:+.0f}  "
                             f"R {pose_gate.roll:+.0f}")
            else:
                pose_text = "P --  Y --  R --"

            pose_color = (0, 165, 255) if gated else (255, 255, 255)

            cv2.putText(
                frame,
                pose_text,
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                pose_color,
                2
            )

            if gated:
                cv2.putText(
                    frame,
                    "POSE GATE",
                    (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )
```

(`P`/`Y` are baseline-relative degrees; `R` is the smoothed absolute roll, display-only per the spec. Positions (10, 105)/(10, 130) sit under the existing left-column text — Yawns is at (10, 55), YAWNING at (10, 80).)

- [x] **Step 2: Verify**

Run: `python -m pytest -v`
Expected: 27 passed.

Run: `python -c "import drownsy; print('import ok')"`
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add drownsy.py
git commit -m "feat: show head pose and gate state on HUD"
```

---

### Task 6: Docs + short live verification

**Files:**
- Modify: `CLAUDE.md` (architecture + roadmap bullets)
- Modify: `.claude/skills/run-drowsy/SKILL.md` (tunables list)

- [x] **Step 1: Update CLAUDE.md**

Two edits:

**1a.** In the Architecture section, the "Two independent detection paths" bullet intro reads `**Two independent detection paths** run per frame inside the face loop:`. Change the word "Two" to "Three" and add this sub-bullet after the *Yawning* one:

```markdown
  - *Head pose*: pitch/yaw/roll via `cv2.solvePnP` on 6 landmarks against a generic 3D head model, baseline-relative (neutral pose captured during calibration, or lazily from the first valid frames). Looking down/away past `POSE_GATE_PITCH_DEG`/`POSE_GATE_YAW_DEG` *gates* the eye paths (closure timer resets, PERCLOS stops sampling) so downward gaze can't fire false eye alerts; pitch-down past `SLUMP_PITCH_DEG` sustained `SLUMP_ALERT_SEC` fires its own slump alert.
```

**1b.** In the Research-context bullet, the "Known limitation (verified live 2026-06-12)" sentence says the lookdown false-alarm fix is planned. Rewrite that sentence to past tense, e.g.:

```markdown
**Resolved limitation:** EAR conflates downward gaze with eye closure (verified live 2026-06-12); fixed by head-pose pitch/yaw gating (solvePnP) — the eye paths pause while the pose gate is active, and a separate slump alert covers genuinely head-down sleep.
```

Also update the sentence "This is **not yet implemented**" in the same bullet — head-pose is now implemented; body/seat posture remains roadmap.

**1c.** In the Tunables sentence of the Architecture section, add the new names to the detection list: `POSE_GATE_PITCH_DEG`, `POSE_GATE_YAW_DEG`, `SLUMP_PITCH_DEG`, `SLUMP_ALERT_SEC`, `POSE_EMA_ALPHA`.

- [x] **Step 2: Update run-drowsy skill tunables**

In `.claude/skills/run-drowsy/SKILL.md`, the last paragraph lists key tunables (`CALIB_SECONDS, CALIB_RATIO, EYE_CLOSED_ALERT_SEC, PERCLOS_THRESH, CLAUDE_MODEL, ALERT_COOLDOWN`). Add `POSE_GATE_PITCH_DEG, SLUMP_PITCH_DEG` to that list.

- [x] **Step 3: Live verification (one run, three checks)**

Run `python drownsy.py` in the foreground (webcam GUI; quit with `q` in the video window). Sit normally through the 10 s calibration, then:

1. Look down at the keyboard ~5 s → HUD shows `POSE GATE`, **no** `[TRIGGER eyes]` (this was the false-alarm scenario).
2. Slump the head far down for ~3 s → console prints `[TRIGGER slump] ...`.
3. Look at the camera and hold eyes closed ~2 s → `[TRIGGER eyes]` still fires.

Note pass/fail per check. If a check fails, stop and debug (superpowers:systematic-debugging) before committing.

**Outcome (2026-08-17): all three checks pass.** It took several runs and
surfaced three defects, all fixed:

1. *Slump alert never fired.* Two causes. `reset_transient()` cleared the
   slump timer on face loss, and a deep slump takes the face out of dlib's
   view for ~15 s — so the episode reset every frame and the alert was
   unreachable. The slump check also lived *inside* the face loop, so it
   never ran during the blackout. Timer now survives face loss; check moved
   outside the loop.
2. *Bogus multi-second eye alerts.* `closure_start` is only cleared in the
   `else` branch inside the face loop, so a slump-induced blackout left the
   closure timer running and billed the whole gap as one closure the moment
   the face returned (observed: "Eyes were closed for about 8 seconds").
   Pre-existing bug, unreachable until head-pose made the user leave frame.
   It also consumed `ALERT_COOLDOWN`, blocking the slump alert — one bug,
   both symptoms. Face loss now resets `closure_start`/`ALARM_ON`.
3. *Gate jammed on for an entire session.* Calibrating while angled at a
   second monitor produced a neutral yaw of −31.7° (vs −2.4° facing square),
   leaving the yaw arm engaged permanently and silently disabling all eye
   detection. Not a code fix — documented in CLAUDE.md.

Observed angles: neutral pitch varies −0.9 to +17.8 between runs; a
deliberate look-down tracks at rel_pitch +9 to +15; beyond that the face is
lost rather than reading higher. A single +32.7 sample was an outlier.
`SLUMP_PITCH_DEG` was accordingly set to 10 (not the 20–25 originally
guessed) so the episode starts while the face is still trackable, with
`SLUMP_ALERT_SEC` raised to 4 to separate a glance from a slump. Final run:
5 slump alerts across both the visible-pitch and face-lost paths, eye and
PERCLOS alerts firing normally, no false eye alerts.

Gap: the eye-closure reset on face loss lives inline in `main()` and has no
unit test, unlike `PoseGate`.

- [x] **Step 4: Annotate this plan**

Tick all checkboxes; note the live-verification outcome under Task 6 (which checks passed, observed angles).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md .claude/skills/run-drowsy/SKILL.md docs/superpowers/plans/2026-06-12-head-pose-gating.md
git commit -m "docs: record head-pose gating in architecture and tunables"
```

---

## Self-review notes (spec → tasks)

- Gate on pitch-down + |yaw|, pause closure timer + PERCLOS → Task 4 steps 2/4.
- Slump alert, once per episode, existing pipeline + `[TRIGGER slump]` → Task 2 (`slump_alert_due`) + Task 4 step 3.
- Baseline from calibration medians, lazy 60-frame fallback → Task 3 + Task 2 (`lazy_baseline_frames`).
- solvePnP failure ⇒ ungated, slump non-accruing → Task 1 (`None` returns) + Task 2 (`update(None, ...)` test).
- Face lost ⇒ `reset_transient()` → Task 4 step 5; tested in Task 2.
- HUD pitch/yaw/roll + `POSE GATE` → Task 5.
- Yawn path stays ungated; roll display-only → no gating code touches MAR (Task 4) and roll is HUD-only (Task 5).
- Spec testing list 1–4 → Tasks 1, 2, 2, 3 respectively; manual 3-check protocol → Task 6 step 3.

