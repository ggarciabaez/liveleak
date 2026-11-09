"""Cross-module contract tests: what RiskDetector assumes about its parts."""
import inspect

import numpy as np

from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter


def test_detng_to_kalman_handoff(history):
    mask = np.ones(history.shape[:2], dtype=bool)
    score, *_ = DetNG().compile(get_rules()).execute(history, mask)
    x, p, k = KalmanFilter().fuse(0.2, score, 0.1, 0.1)
    assert 0.0 <= x <= 1.0 and p >= 0 and 0 <= k <= 1


def test_risk_detector_keeps_model_not_load_result():
    """__init__ must assign the STGNN itself to self.model, not a load call's return value."""
    from modules.risk import RiskDetector
    src = inspect.getsource(RiskDetector.__init__).replace(" ", "")
    assert "STGNN(graph_rad).load_state_dict" not in src
    assert "self.model=STGNN(graph_rad)\n" in src