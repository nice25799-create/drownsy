import math

import cv2
import numpy as np
import pytest

import drownsy


# ---------- estimate_head_pose ----------

FRAME_W, FRAME_H = 450, 340


def project_pose(pitch_down_deg, yaw_deg, roll_deg=0.0):
    """Fake 68-landmark array for a head at a known pitch/yaw/roll.

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

    c = math.radians(roll_deg)
    rz = np.array([
        [math.cos(c), -math.sin(c), 0.0],
        [math.sin(c), math.cos(c), 0.0],
        [0.0, 0.0, 1.0],
    ])

    rot = rz @ ry @ rx
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


def test_roll_recovered_alongside_pitch_and_yaw():
    shape = project_pose(10.0, 5.0, 8.0)
    pose = drownsy.estimate_head_pose(shape, FRAME_W, FRAME_H)
    assert pose is not None
    pitch, yaw, roll = pose
    assert pitch == pytest.approx(10.0, abs=2.0)
    assert abs(yaw) == pytest.approx(5.0, abs=2.0)
    assert roll == pytest.approx(8.0, abs=2.0)


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
    assert g.slump_elapsed(0.0) == 0.0        # no episode yet

    g.update((30.0, 0.0, 0.0), 0.0)
    assert g.slump_elapsed(1.9) == pytest.approx(1.9)
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


# ---------- derive_pose_baseline ----------

def test_pose_baseline_is_median_of_samples():
    pitch = [4.0] * 90 + [40.0] * 10   # a few look-down frames must not skew it
    yaw = [-1.0] * 100
    baseline = drownsy.derive_pose_baseline(pitch, yaw)
    assert baseline == pytest.approx((4.0, -1.0))


def test_pose_baseline_none_when_too_few_samples():
    n = drownsy.MIN_CALIB_SAMPLES - 1
    assert drownsy.derive_pose_baseline([0.0] * n, [0.0] * n) is None
