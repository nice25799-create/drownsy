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
