"""Cross-module contract tests: what RiskDetector assumes about its parts."""
import inspect

import numpy as np
import torch

from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter
from modules.stgnn import STGNN


def test_detng_to_kalman_handoff(history):
    mask = np.ones(history.shape[:2], dtype=bool)
    score, *_ = DetNG().compile(get_rules()).execute(history, mask)
    x, p, k = KalmanFilter().fuse(0.2, score, 0.1, 0.1)
    assert 0.0 <= x <= 1.0 and p >= 0 and 0 <= k <= 1


def test_save_load_weights_roundtrip(tmp_path, history):
    mask = np.ones(history.shape[:2], dtype=bool)
    a, b = STGNN(50.0, 16), STGNN(50.0, 16)
    a.eval()
    b.eval()

    def predict(model):
        with torch.inference_mode():
            return model(history, mask).numpy()

    # Sanity: independently initialised models must disagree before loading.
    assert not np.allclose(predict(a), predict(b))

    path = tmp_path / "stgnn.pth"
    a.save_weights(path)
    b.load_weights(path)
    np.testing.assert_allclose(predict(a), predict(b), rtol=1e-5)


def test_risk_detector_keeps_model_not_load_result():
    """__init__ must assign the STGNN itself to self.model, not a load call's return value."""
    from modules.risk import RiskDetector
    src = inspect.getsource(RiskDetector.__init__).replace(" ", "")
    assert "STGNN(graph_rad).load_state_dict" not in src
    assert "self.model=STGNN(graph_rad)\n" in src