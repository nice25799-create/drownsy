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
