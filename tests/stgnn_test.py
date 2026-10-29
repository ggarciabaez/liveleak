import numpy as np
import pytest
import torch

from modules.stgnn import STGNN, MotionAnomalyCalibrator


@pytest.fixture
def model():
    return STGNN(graph_radius=50.0, hidden_dim=16)


# ---- calibrator ----
def test_calibrator_requires_fit():
    with pytest.raises(RuntimeError):
        MotionAnomalyCalibrator().score([0.1])


@pytest.mark.parametrize("bad", [[], [-1.0], [np.nan], [np.inf]])
def test_calibrator_rejects_bad_fit_data(bad):
    with pytest.raises(ValueError):
        MotionAnomalyCalibrator().fit(bad)


def test_calibrator_percentiles():
    cal = MotionAnomalyCalibrator().fit([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(cal.score([0.0, 2.0, 10.0]), [0.0, 0.5, 1.0])


def test_calibrator_score_monotonic():
    cal = MotionAnomalyCalibrator().fit(np.random.rand(100))
    s = cal.score(np.linspace(0, 2, 50))
    assert np.all(np.diff(s) >= 0) and s.min() >= 0 and s.max() <= 1


# ---- model ----
@pytest.mark.parametrize("kw", [{"graph_radius": 0}, {"graph_radius": -1}, {"graph_radius": 1, "hidden_dim": 0}])
def test_constructor_validates(kw):
    with pytest.raises(ValueError):
        STGNN(**kw)


def test_forward_shape_and_finite(model, history):
    out = model(history)
    assert out.shape == (history.shape[1], 4)
    assert torch.isfinite(out).all()


def test_forward_accepts_tensor_and_numpy(model, history):
    a = model.predict_kinematics(history)
    b = model.predict_kinematics(torch.as_tensor(history))
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_forward_empty_scene(model):
    assert model(np.zeros((3, 0, 13))).shape == (0, 4)


@pytest.mark.parametrize("shape", [(5, 2, 12), (2, 13), (0, 2, 13)])
def test_forward_rejects_bad_shapes(model, shape):
    with pytest.raises(ValueError):
        model(np.zeros(shape))


def test_forward_rejects_misaligned_ids(model, history):
    bad = history.copy()
    bad[-1, :, 0] = [9, 8]
    with pytest.raises(ValueError):
        model(bad)


def test_forward_rejects_duplicate_ids(model, history):
    bad = history.copy()
    bad[:, :, 0] = 1
    with pytest.raises(ValueError):
        model(bad)


def test_predict_kinematics_restores_train_mode(model, history):
    model.train()
    model.predict_kinematics(history)
    assert model.training


def test_isolated_entities_still_work(history):
    tiny_radius = STGNN(graph_radius=1e-3, hidden_dim=16)  # no edges at all
    assert torch.isfinite(tiny_radius(history)).all()


def test_gradients_flow(model, history):
    loss = model(history).pow(2).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


# ---- scoring ----
def test_score_forecast_contract(model, history):
    cal = MotionAnomalyCalibrator().fit(np.random.rand(50))
    observed = history[-1]  # (N, 13) with matching IDs
    out = model.score_forecast(history, observed, cal)
    assert set(out) == {"scene_score", "entity_scores", "forecast_errors", "entity_ids"}
    assert 0.0 <= out["scene_score"] <= 1.0
    assert out["entity_scores"].shape == (history.shape[1],)


def test_score_forecast_accepts_4col_observation(model, history):
    cal = MotionAnomalyCalibrator().fit(np.random.rand(50))
    out = model.score_forecast(history, history[-1, :, 5:9], cal)
    assert out["scene_score"] is not None


def test_score_forecast_rejects_id_mismatch(model, history):
    cal = MotionAnomalyCalibrator().fit(np.random.rand(50))
    bad = history[-1].copy()
    bad[:, 0] = [7, 8]
    with pytest.raises(ValueError):
        model.score_forecast(history, bad, cal)


# ---- training smoke test ----
@pytest.mark.slow
def test_overfits_tiny_constant_velocity_set(history):
    torch.manual_seed(0)
    m = STGNN(graph_radius=50.0, hidden_dim=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    last = history[-1, :, 5:9]
    target = torch.as_tensor(
        np.concatenate([last[:, :2] + last[:, 2:], last[:, 2:]], axis=1), dtype=torch.float32
    )  # constant-velocity next state
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = (m(history) - target).pow(2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first