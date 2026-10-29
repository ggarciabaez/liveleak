import numpy as np
import pytest

from modules.kfilter import KalmanFilter


def test_gain_is_between_0_and_1():
    _, _, k = KalmanFilter().fuse(0.3, 0.7, 0.2, 0.5)
    assert 0.0 <= k <= 1.0


def test_agreeing_sources_return_same_value():
    x, _, _ = KalmanFilter().fuse(0.4, 0.4, 0.1, 0.1)
    assert x == pytest.approx(0.4)


def test_fused_lies_between_inputs():
    x, _, _ = KalmanFilter().fuse(0.2, 0.8, 0.1, 0.1)
    assert 0.2 <= x <= 0.8


def test_trusts_lower_variance_source():
    # detng far more certain -> output near detng
    x, _, _ = KalmanFilter(process_noise=0.0).fuse(0.0, 1.0, 1.0, 0.001)
    assert x > 0.9
    # stgnn far more certain -> output near stgnn
    x, _, _ = KalmanFilter(process_noise=0.0).fuse(0.0, 1.0, 0.001, 1.0)
    assert x < 0.1


def test_posterior_variance_shrinks():
    _, p, _ = KalmanFilter().fuse(0.5, 0.5, 0.2, 0.5)
    assert p < 0.2 + 0.01  # below prior variance + Q


def test_state_is_stored():
    kf = KalmanFilter()
    x, p, _ = kf.fuse(0.1, 0.9, 0.3, 0.3)
    assert kf.x == x and kf.P == p


def test_finite_with_numpy_inputs():
    out = KalmanFilter().fuse(np.float64(0.5), np.float32(0.5), 0.1, 0.1)
    assert all(np.isfinite(out))