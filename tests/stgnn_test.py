"""STGNN + calibrator tests for the masked (T, N, 13) window format.

Format recap: slot n is one track across the whole window, `mask[t, n]` says whether
that track was detected in frame t. Padding rows are id -1 / zeros, but the model must
never depend on what is inside them: only the mask counts.
"""
import numpy as np
import pytest
import torch

from modules.stgnn import (
    STGNN,
    MotionAnomalyCalibrator,
    fit_motion_anomaly_calibrator,
)

PERSON, MACHINE, CONE = 1, 2, 4  # type codes used by conftest.make_vec
T = 6


@pytest.fixture
def model():
    return STGNN(graph_radius=50.0, hidden_dim=16)


def _window(make_frame, slots, frames=T):
    """slots: (track_id, type, t_first, t_last, px0, py0, vx, vy) per slot.
    Returns feats (T, N, 13) with id -1 / zero padding, and mask (T, N) bool."""
    feats = np.zeros((frames, len(slots), 13), np.float32)
    feats[..., 0] = -1
    mask = np.zeros((frames, len(slots)), bool)
    for n, (tid, typ, t0, t1, x, y, vx, vy) in enumerate(slots):
        for t in range(t0, t1 + 1):
            feats[t, n] = make_frame((tid, typ, x + vx * t, y + vy * t, vx, vy))[0]
            mask[t, n] = True
    return feats, mask


@pytest.fixture
def win(make_frame):
    """4 slots, all inside each other's graph radius:
    0 person, whole window | 1 machine, whole window
    2 person, leaves after t=2 | 3 person, appears at t=3"""
    return _window(make_frame, [
        (1, PERSON, 0, 5, 50.0, 50.0, -2.0, 0.0),
        (2, MACHINE, 0, 5, 30.0, 50.0, 0.0, 0.0),
        (3, PERSON, 0, 2, 60.0, 60.0, 0.5, 0.0),
        (4, PERSON, 3, 5, 45.0, 55.0, -1.0, 0.0),
    ])


def _garbage_in_padding(feats, mask):
    """Fill padding rows with junk placed right on top of real entities, so an
    unmasked model would wire them into the graph."""
    bad = feats.copy()
    rng = np.random.default_rng(1)
    junk = rng.normal(size=(int((~mask).sum()), 13)) * 100
    junk[:, 5:7] = [50.0, 50.0]
    bad[~mask] = junk
    return bad


# ================= calibrator =================
def test_calibrator_requires_fit():
    with pytest.raises(RuntimeError):
        MotionAnomalyCalibrator().score([0.1])


@pytest.mark.parametrize("bad", [[], [-1.0], [np.nan], [np.inf]])
def test_calibrator_rejects_bad_fit_data(bad):
    with pytest.raises(ValueError):
        MotionAnomalyCalibrator().fit(bad)


@pytest.mark.parametrize("bad", [[-1.0], [np.nan], [np.inf]])
def test_calibrator_rejects_bad_scoring_input(bad):
    cal = MotionAnomalyCalibrator().fit([1.0, 2.0])
    with pytest.raises(ValueError):
        cal.score(bad)


def test_calibrator_scores_percentile_of_a_single_error():
    cal = MotionAnomalyCalibrator().fit([1.0, 2.0, 3.0, 4.0])
    assert cal.score(0.0) == 0.0
    assert cal.score(2.0) == 0.5   # side="right": ties count as <=
    assert cal.score(10.0) == 1.0


def test_calibrator_score_of_many_errors_is_the_max():
    cal = MotionAnomalyCalibrator().fit([1.0, 2.0, 3.0, 4.0])
    assert cal.score([0.0, 2.0, 10.0]) == 1.0
    assert cal.score([0.0, 2.0]) == 0.5
    assert isinstance(cal.score([0.0, 2.0]), float)


def test_calibrator_empty_errors_score_zero():
    cal = MotionAnomalyCalibrator().fit([1.0, 2.0])
    assert cal.score(np.empty(0)) == 0.0


def test_calibrator_score_monotonic_and_bounded():
    cal = MotionAnomalyCalibrator().fit(np.random.rand(100))
    s = np.array([cal.score(e) for e in np.linspace(0, 2, 50)])
    assert np.all(np.diff(s) >= 0) and s.min() >= 0 and s.max() <= 1


def test_calibrator_save_load_roundtrip(tmp_path):
    errors = np.random.rand(40)
    path = tmp_path / "calib.npz"
    np.savez(path, errors=MotionAnomalyCalibrator().fit(errors).normal_errors)
    loaded = MotionAnomalyCalibrator()
    loaded.load(path)
    ref = MotionAnomalyCalibrator().fit(errors)
    for e in (0.0, 0.3, 0.9, 5.0):
        assert loaded.score(e) == ref.score(e)


def test_calibrator_load_none_is_noop():
    cal = MotionAnomalyCalibrator()
    cal.load(None)
    assert cal.normal_errors is None


# ================= constructor / weights =================
@pytest.mark.parametrize("kw", [{"graph_radius": 0}, {"graph_radius": -1}, {"graph_radius": 1, "hidden_dim": 0}])
def test_constructor_validates(kw):
    with pytest.raises(ValueError):
        STGNN(**kw)


def test_weights_roundtrip(tmp_path, model, win):
    feats, mask = win
    other = STGNN(graph_radius=50.0, hidden_dim=16)
    path = tmp_path / "w.pth"
    model.save_weights(path)
    other.load_weights(path)
    with torch.no_grad():
        np.testing.assert_allclose(model(feats, mask), other(feats, mask), rtol=1e-5)


def test_load_weights_none_is_noop(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_weights(None)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


# ================= forward: contract =================
def test_forward_shape_and_finite(model, history):
    out = model(history)  # mask=None, all rows real
    assert out.shape == (history.shape[1], 4)
    assert torch.isfinite(out).all()


def test_forward_masked_shape_and_finite(model, win):
    out = model(*win)
    assert out.shape == (4, 4) and torch.isfinite(out).all()


def test_forward_accepts_tensor_and_numpy(model, win):
    feats, mask = win
    a = model(feats, mask)
    b = model(torch.as_tensor(feats), torch.as_tensor(mask))
    np.testing.assert_allclose(a.detach(), b.detach(), rtol=1e-5)


def test_forward_empty_scene(model):
    assert model(np.zeros((3, 0, 13))).shape == (0, 4)


@pytest.mark.parametrize("shape", [(5, 2, 12), (2, 13)])
def test_forward_rejects_bad_shapes(model, shape):
    with pytest.raises(ValueError):
        model(np.zeros(shape))


@pytest.mark.xfail(strict=True, reason="T=0 reaches feats[-1] and raises IndexError instead of ValueError")
def test_forward_rejects_zero_frames(model):
    with pytest.raises(ValueError):
        model(np.zeros((0, 2, 13)))


@pytest.mark.xfail(strict=True, reason="mask shape is never validated; a (T, N') mask raises IndexError")
def test_forward_rejects_mismatched_mask_shape(model, win):
    feats, mask = win
    with pytest.raises(ValueError):
        model(feats, mask[:, :2])


def test_mask_none_infers_presence_from_padding_id(model, win):
    feats, mask = win
    with torch.no_grad():
        np.testing.assert_allclose(model(feats), model(feats, mask), rtol=1e-5)


# ================= forward: the mask is what counts =================
def _present_last(win_mask):
    return torch.as_tensor(win_mask[-1])


def test_padding_contents_never_matter(model, win):
    feats, mask = win
    sel = _present_last(mask)
    with torch.no_grad():
        clean = model(feats, mask)[sel]
        dirty = model(_garbage_in_padding(feats, mask), mask)[sel]
    np.testing.assert_allclose(clean, dirty, atol=1e-5)


def test_padded_rows_get_no_gradient(model, win):
    feats, mask = win
    x = torch.tensor(_garbage_in_padding(feats, mask), requires_grad=True)
    model(x, mask)[_present_last(mask)].pow(2).sum().backward()
    assert torch.count_nonzero(x.grad[~torch.as_tensor(mask)][:, 1:]) == 0


def test_late_appearance_equals_fresh_start(model, win):
    """Slot 3 shows up at t=3; its output must equal a run that only sees t>=3."""
    feats, mask = win
    with torch.no_grad():
        full = model(feats, mask)[3]
        fresh = model(feats[3:], mask[3:])[3]
    np.testing.assert_allclose(full, fresh, atol=1e-5)


def test_empty_middle_frame_is_skipped(model, win):
    feats, mask = win
    mask2 = mask.copy()
    mask2[2] = False  # nobody detected at t=2
    keep = [0, 1, 3, 4, 5]
    with torch.no_grad():
        with_gap = model(feats, mask2)
        without = model(feats[keep], mask2[keep])
    sel = _present_last(mask)
    np.testing.assert_allclose(with_gap[sel], without[sel], atol=1e-5)


def test_slot_order_is_irrelevant(model, win):
    feats, mask = win
    perm = np.array([2, 0, 3, 1])  # new slot j holds old slot perm[j]
    present = mask[-1][perm]
    with torch.no_grad():
        base = model(feats, mask)
        permuted = model(feats[:, perm], mask[:, perm])
    np.testing.assert_allclose(permuted[present], base[perm][present], atol=1e-5)


def test_single_present_entity(model, win):
    feats, mask = win
    only = np.zeros_like(mask)
    only[:, 0] = True
    assert torch.isfinite(model(feats, only)).all()


def test_nobody_present(model, win):
    feats, _ = win
    assert torch.isfinite(model(feats, np.zeros((T, 4), bool))).all()


def test_isolated_entities_still_work(win):
    tiny_radius = STGNN(graph_radius=1e-3, hidden_dim=16)  # no edges at all
    assert torch.isfinite(tiny_radius(*win)).all()


def test_gradients_flow_with_partial_mask(model, win):
    feats, mask = win
    model(feats, mask)[_present_last(mask)].pow(2).mean().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.xfail(strict=True, reason="forward keeps cones in the graph; filter them upstream or inside forward")
def test_cones_do_not_affect_moving_entities(model, make_frame):
    base_slots = [
        (1, PERSON, 0, 5, 50.0, 50.0, -2.0, 0.0),
        (2, MACHINE, 0, 5, 30.0, 50.0, 0.0, 0.0),
    ]
    cone = (9, CONE, 0, 5, 45.0, 52.0, 0.0, 0.0)
    f0, m0 = _window(make_frame, base_slots)
    f1, m1 = _window(make_frame, base_slots + [cone])
    with torch.no_grad():
        np.testing.assert_allclose(model(f0, m0), model(f1, m1)[:2], atol=1e-5)


# ================= compute_window_error =================
def test_window_error_one_value_per_entity_present_in_last_two_frames(model, win):
    err = model.compute_window_error(*win)
    # slots 0, 1 (whole window) and 3 (appeared t=3); slot 2 is gone
    assert err.shape == (3,)
    assert torch.isfinite(err).all() and (err >= 0).all()


def test_window_error_matches_manual_rmse(model, win):
    feats, mask = win
    with torch.no_grad():
        err = model.compute_window_error(feats, mask)
        pred = model(feats[:-1], mask[:-1])
    elig = torch.as_tensor(mask[-2] & mask[-1])
    obs = torch.as_tensor(feats[-1, :, 5:9])
    expected = (pred[elig] - obs[elig]).pow(2).mean(-1).sqrt()
    np.testing.assert_allclose(err, expected, atol=1e-6)


def test_window_error_is_zero_when_target_equals_forecast(model, win):
    feats, mask = win
    with torch.no_grad():
        pred = model(feats[:-1], mask[:-1])
    hit = feats.copy()
    hit[-1, :, 5:9] = pred.numpy()
    with torch.no_grad():
        err = model.compute_window_error(hit, mask)
    assert torch.allclose(err, torch.zeros_like(err), atol=1e-5)


def test_window_error_never_sees_the_target_frame(model, win):
    """Target frame content only enters through the observation, not the forecast."""
    feats, mask = win
    moved = feats.copy()
    moved[-1, :, 5:9] += 10.0
    with torch.no_grad():
        a = model.compute_window_error(feats, mask)
        b = model.compute_window_error(moved, mask)
    assert not torch.allclose(a, b)
    # each error moves by at most the injected offset (rmse of a +10 shift on 4 dims)
    assert (b - a).abs().max() <= 10.0 + 1e-4


def test_window_error_drops_entity_that_vanishes_at_target(model, make_frame):
    feats, mask = _window(make_frame, [
        (1, PERSON, 0, 5, 50.0, 50.0, -2.0, 0.0),
        (2, MACHINE, 0, 4, 30.0, 50.0, 0.0, 0.0),  # absent at t=5
    ])
    assert model.compute_window_error(feats, mask).shape == (1,)


def test_window_error_drops_entity_that_appears_at_target(model, make_frame):
    feats, mask = _window(make_frame, [
        (1, PERSON, 0, 5, 50.0, 50.0, -2.0, 0.0),
        (2, PERSON, 5, 5, 40.0, 50.0, 0.0, 0.0),  # first seen at t=5
    ])
    assert model.compute_window_error(feats, mask).shape == (1,)


def test_window_error_ignores_padding_contents(model, win):
    feats, mask = win
    with torch.no_grad():
        clean = model.compute_window_error(feats, mask)
        dirty = model.compute_window_error(_garbage_in_padding(feats, mask), mask)
    np.testing.assert_allclose(clean, dirty, atol=1e-5)


@pytest.mark.parametrize("case", ["one_frame", "no_overlap"])
@pytest.mark.xfail(strict=True, reason="returns a fake [0.0] sample; it would pollute the calibration set. Should be empty")
def test_window_error_with_nothing_to_evaluate_is_empty(model, win, case):
    feats, mask = win
    if case == "one_frame":
        feats, mask = feats[:1], mask[:1]
    else:
        mask = mask.copy()
        mask[-1] = False
    assert model.compute_window_error(feats, mask).numel() == 0


def test_window_error_feeds_the_calibrator(model, win):
    cal = MotionAnomalyCalibrator().fit(np.random.rand(50))
    with torch.no_grad():
        err = model.compute_window_error(*win).numpy()
    assert 0.0 <= cal.score(err) <= 1.0


# ================= fit_motion_anomaly_calibrator (fully-present sequences) =================
def test_fit_calibrator_collects_one_error_per_entity_per_window(model, history):
    cal = fit_motion_anomaly_calibrator(model, history, window_size=3, calib_file_save=None)
    frames, entities = history.shape[:2]
    assert cal.normal_errors.size == (frames - 3) * entities
    assert np.all(np.diff(cal.normal_errors) >= 0)  # stored sorted


def test_fit_calibrator_restores_train_mode_and_writes_file(model, history, tmp_path):
    model.train()
    path = tmp_path / "calib.npz"
    cal = fit_motion_anomaly_calibrator(model, history, window_size=3, calib_file_save=path)
    assert model.training
    np.testing.assert_array_equal(np.load(path)["errors"], cal.normal_errors)


def test_fit_calibrator_rejects_short_sequences(model, history):
    with pytest.raises(ValueError):
        fit_motion_anomaly_calibrator(model, history, window_size=history.shape[0], calib_file_save=None)


# ================= training smoke test =================
@pytest.mark.slow
def test_overfits_tiny_constant_velocity_set(win):
    torch.manual_seed(0)
    feats, mask = win
    m = STGNN(graph_radius=50.0, hidden_dim=16)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    sel = torch.as_tensor(mask[-1])
    last = torch.as_tensor(feats[-1, :, 5:9])
    target = torch.cat([last[:, :2] + last[:, 2:], last[:, 2:]], dim=1)  # constant velocity
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = (m(feats, mask)[sel] - target[sel]).pow(2).mean()
        first = first if first is not None else loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first