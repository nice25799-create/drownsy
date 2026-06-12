# Head-Pose Gating + Slump Alert — Design Spec

**Date:** 2026-06-12
**Status:** Approved (brainstormed and section-approved in session)

## Problem

Verified live on 2026-06-12: EAR cannot distinguish downward gaze from eye
closure. Looking at the keyboard or a phone collapses EAR to 0.13–0.16 —
below any reasonable calibrated threshold (observed 0.195–0.22) — and fires
false eye-closure alerts. Threshold tuning cannot fix this because the
look-down and eyes-closed EAR distributions overlap. The fix is to know
where the head is pointing.

This also advances the project's headline research angle (drowsiness ↔
posture): head-pose pitch/yaw is the first posture feature.

## Goals

1. **Gate:** pause the eye metrics (EAR-closure timer and PERCLOS sampling)
   while the head is turned away from the camera — moderately pitched down
   or yawed left/right — so downward gaze stops causing false eye alerts.
2. **Slump alert:** an extreme, sustained pitch-down is itself a drowsiness
   sign (head dropping). Fire a dedicated alert for it. This is also the
   safety counterweight to the gate: gating alone would mask someone who
   falls asleep head-down.
3. **HUD readout:** show pitch/yaw/roll and a visible `POSE GATE` indicator.

## Non-goals

- Roll-based gating (roll is computed and displayed only).
- Gating the yawn/MAR path — it produced no false alarms, and a yawn while
  looking down is still a yawn.
- Body posture, seat pressure, ML classifiers, new dependencies (still
  later roadmap).

## Approach chosen

**solvePnP with the existing dlib landmarks** (option A), over a nose-offset
proxy (B) and MediaPipe (C). Reasons: true angles in degrees (comparable to
the literature's ±7° reference, jimaging-09-00091), no new dependency, and
the math is unit-testable synthetically. Soukupová & Čech note EAR is only
"partially head pose insensitive", supporting pose as the missing input.

## Architecture

Everything stays in `drownsy.py` (single-file pattern), implemented as pure
testable units like `derive_threshold` and `PerclosTracker`.

### Config block additions

| Tunable | Default | Meaning |
|---|---|---|
| `POSE_GATE_PITCH_DEG` | 10.0 | baseline-relative pitch-down beyond this → gate eye metrics |
| `POSE_GATE_YAW_DEG` | 15.0 | baseline-relative \|yaw\| beyond this → gate eye metrics |
| `SLUMP_PITCH_DEG` | 25.0 | baseline-relative pitch-down beyond this = possible slump |
| `SLUMP_ALERT_SEC` | 2.0 | slump sustained this long → alert |
| `POSE_EMA_ALPHA` | 0.3 | EMA smoothing factor for raw angles |

(Literature reference: jimaging paper used ±7° absolute on MediaPipe; our
gate defaults are slightly looser because they are baseline-relative and
tunable.)

### Components

1. **`estimate_head_pose(shape, frame_w, frame_h)`** — pure function.
   - Image points: dlib landmark indices 30 (nose tip), 8 (chin),
     36 (left eye outer corner), 45 (right eye outer corner),
     48 (left mouth corner), 54 (right mouth corner).
   - 3D generic head model (standard OpenCV-tutorial values, nose at
     origin): nose (0, 0, 0); chin (0, −330, −65);
     left eye (−225, 170, −135); right eye (225, 170, −135);
     left mouth (−150, −150, −125); right mouth (150, −150, −125).
   - Camera intrinsics approximated: focal length ≈ `frame_w`, principal
     point = frame center, zero distortion.
   - `cv2.solvePnP` (iterative) → `cv2.Rodrigues` → rotation matrix →
     Euler angles via decomposition.
   - **Sign/normalization convention (pinned by test):** output is
     normalized/wrapped so a face looking straight at the camera reads
     near 0°, and **positive pitch = looking down**. (Raw decomposition of
     this model often yields pitch near ±180°; the wrap is part of the
     function's contract and the synthetic round-trip test enforces it.)
   - Returns `(pitch, yaw, roll)` in degrees, or `None` when solvePnP
     fails or the geometry is degenerate.

2. **`PoseGate` class** — owns baseline, smoothing, gate and slump state.
   - `set_baseline(pitch, yaw)` / baseline from calibration medians.
   - `update(pitch, yaw, roll, now)` — EMA-smooths raw angles, computes
     baseline-relative values.
   - `is_gated()` — True when relative pitch-down > `POSE_GATE_PITCH_DEG`
     or |relative yaw| > `POSE_GATE_YAW_DEG`. False whenever there is no
     baseline yet or no valid pose this frame.
   - `slump_alert_due(now)` — returns True exactly once per slump
     episode, at the moment relative pitch-down has continuously exceeded
     `SLUMP_PITCH_DEG` for ≥ `SLUMP_ALERT_SEC`. The episode (and its
     fired flag) resets when pitch recovers below `SLUMP_PITCH_DEG`.
   - `reset_transient()` — clears EMA state and slump timer (called on
     face loss). Baseline is kept.
   - **Lazy baseline fallback:** if calibration produced no usable pose
     baseline, `PoseGate` collects the first 60 valid pose frames during
     detection and uses their median. Gate and slump are inactive until a
     baseline exists.

3. **Calibration extension** — the existing 10 s EAR calibration loop also
   calls `estimate_head_pose` per frame and collects pitch/yaw samples;
   the medians become the neutral baseline. No change to EAR threshold
   derivation.

4. **Main-loop integration (per frame, inside the face loop):**

   ```
   landmarks → EAR/MAR (unchanged)
            → estimate_head_pose → PoseGate.update (EMA, baseline-relative)
            → gated?  ──yes──> reset eye-closure state (closure_start=None,
                               ALARM_ON=False), skip PERCLOS sample
                      ──no───> eye-closure + PERCLOS paths run as today
            → PoseGate.slump_alert_due(now)?
                      → [TRIGGER slump] + handle_alert (existing pipeline:
                        alert_busy, ALERT_COOLDOWN, Claude message, TTS)
   ```

   Face lost → `PoseGate.reset_transient()`; gate off while no face.

5. **HUD:** one added line, e.g. `P +12  Y -3  R +1`, plus a `POSE GATE`
   tag while gated. Slump alert uses the existing alert banner.

6. **Trigger logging:** console print matching the existing style:
   `[TRIGGER slump] {reason} (pitch {rel_pitch:+.0f} deg, {elapsed:.1f}s)`.
   Reason string for Claude: "Head slumped forward for about N seconds."

## Error handling

- **solvePnP failure on a frame** → `estimate_head_pose` returns `None` →
  frame treated as *ungated*, slump timer does not advance. The feature
  degrades to "no head pose"; it can never block eye detection.
- **No baseline from calibration** → lazy baseline (above); eye detection
  unaffected meanwhile.
- **Face lost** → `reset_transient()` so a fresh detection doesn't inherit
  stale smoothed angles or a stale slump timer.
- **Alert spam** → one slump alert per episode + existing `alert_busy` and
  `ALERT_COOLDOWN` guards.

## Testing

Pure-logic tests, no webcam, extending `tests/test_drowsy_logic.py`
(or a sibling `tests/test_head_pose.py`):

1. **Pose math round-trip:** rotate the 3D model points by a known
   pitch/yaw, project to 2D with the same intrinsics, feed the points to
   `estimate_head_pose`, assert recovered angles match within ~2°. Also
   pins the sign convention (positive pitch = down) and near-0° neutral.
2. **Gate logic:** synthetic angle sequences → gate turns on past either
   threshold (pitch and yaw independently), off when back under, always
   off without a baseline.
3. **Slump timer:** synthetic timestamps → `slump_alert_due` stays False
   below `SLUMP_ALERT_SEC`, returns True at the threshold, only once per
   episode, resets on recovery and on `reset_transient()`.
4. **Baseline:** median derivation from calibration samples; lazy-fallback
   path activates after 60 valid frames.

### Manual verification (one short run, three checks)

1. Look down at the keyboard → expect `POSE GATE` on HUD and **no** eye
   alert (the original false-alarm scenario).
2. Slump head down ~3 s → expect `[TRIGGER slump]` alert.
3. Normal long blink looking at the camera → expect the eye alert still
   fires.

## References

- `jimaging-09-00091-v3.txt` (Albadawi 2023): head-pose feature with ±7°
  thresholds relative to initial nose reference; describes drowsy head
  pose as random tilting associated with eye closure lasting a few
  seconds.
- `05 (1).txt` (Soukupová & Čech): EAR is only partially head-pose
  insensitive.
- Live debugging session 2026-06-12: look-down EAR 0.13–0.16 vs calibrated
  thresholds 0.195–0.22; PERCLOS baseline 4–13% while sitting normally.
