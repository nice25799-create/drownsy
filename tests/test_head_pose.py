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
