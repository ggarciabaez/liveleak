"""Cross-module contract tests: what RiskDetector assumes about its parts."""
import inspect

import numpy as np
import pytest
import torch

from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter
from modules.stgnn import STGNN


def test_detng_to_kalman_handoff(history):
    score, *_ = DetNG().compile(get_rules()).execute(history)
    x, p, k = KalmanFilter().fuse(0.2, score, 0.1, 0.1)
    assert 0.0 <= x <= 1.0 and p >= 0 and 0 <= k <= 1


def test_load_state_dict_roundtrip(tmp_path, history):
    a, b = STGNN(50.0, 16), STGNN(50.0, 16)
    path = tmp_path / "stgnn.pth"
    torch.save(a.state_dict(), path)
    b.load_state_dict(torch.load(path))
    np.testing.assert_allclose(a.predict_kinematics(history), b.predict_kinematics(history), rtol=1e-5)


@pytest.mark.xfail(reason="RiskDetector assigns the result of load_state_dict() (a key report) to self.model", strict=True)
def test_risk_detector_keeps_model_not_load_result():
    from modules.risk import RiskDetector
    src = inspect.getsource(RiskDetector.__init__)
    assert "STGNN(graph_rad).load_state_dict" not in src.replace(" ", "")