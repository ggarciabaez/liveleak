import numpy as np
import pytest

from modules.detng import (
    DetNG,
    gen_random_data,
    gen_valid_feature,
    get_rules,
    rule_cone_zone,
    rule_kinematic_ttc,
    rule_ppe_compliance,
    rule_vectorized_proximity,
)
from tests.conftest import CONE, MACHINE, PERSON, VEHICLE


def ctx_for(frame):
    """Run DetNG's own context builder via a one-rule engine to capture ctx."""
    captured = {}

    def spy(ctx):
        captured["ctx"] = ctx
        return 0.0
    DetNG().compile([(spy, 1.0)]).execute(np.atleast_3d(frame)[None] if frame.ndim == 2 else frame)
    return captured["ctx"]


# ---- feature generation ----
@pytest.mark.parametrize("obj_type", [1, 2, 3, 4])
def test_gen_valid_feature_one_hot_and_shape(obj_type):
    v = gen_valid_feature(obj_type, [1, 2, 3, 4], 0b111)
    assert v.shape == (13,)
    assert v[1:5].sum() == 1.0 and v[obj_type] == 1.0


def test_gen_valid_feature_cone_is_static_and_na():
    v = gen_valid_feature(CONE, [1, 2, 3, 4], 0b111)
    assert v[7] == 0 and v[8] == 0 and v[-1] == 1.0


def test_gen_valid_feature_ppe_bitmask():
    v = gen_valid_feature(PERSON, [0, 0, 0, 0], 0b101)
    assert list(v[9:12]) == [1, 0, 1]


def test_gen_random_data_shape_and_continuity():
    d = gen_random_data(s=4, n=3, t=5)
    assert d.shape == (4, 5, 3, 13)
    # overlapping windows: sample i frame 1 == sample i+1 frame 0
    np.testing.assert_array_equal(d[0, 1], d[1, 0])


# ---- engine ----
def test_compile_returns_self_and_counts_rules():
    eng = DetNG()
    assert eng.compile(get_rules()) is eng
    assert eng.num_rules == 4


@pytest.mark.xfail(reason="empty-engine path returns a 2-tuple; callers unpack 3 values", strict=True)
def test_execute_without_rules_matches_3_tuple_contract():
    score, _, _ = DetNG().execute(np.zeros((1, 1, 13)))
    assert score == 0.0


def test_execute_score_bounded_and_names_violations(history):
    eng = DetNG().compile(get_rules())
    score, violations, scores = eng.execute(history)
    assert 0.0 <= score <= 1.0
    assert len(scores) == eng.num_rules
    assert all(isinstance(v, str) for v in violations)


def test_execute_empty_scene_is_zero():
    eng = DetNG().compile(get_rules())
    score, violations, _ = eng.execute(np.empty((3, 0, 13)))
    assert score == 0.0 and violations == []


def test_higher_weights_never_lower_score(history):
    lo = DetNG().compile(get_rules((0.1, 0.1, 0.1, 0.1))).execute(history)[0]
    hi = DetNG().compile(get_rules((1.0, 1.0, 1.0, 1.0))).execute(history)[0]
    assert hi >= lo


# ---- individual rules (each takes a ctx list; last frame is evaluated) ----
def test_ppe_compliance_full_ppe_is_zero(make_frame):
    f = make_frame((1, PERSON, 0, 0, 0, 0, 0b111))
    assert rule_ppe_compliance(ctx_for(f[None])) == 0.0


def test_ppe_compliance_no_ppe_is_one(make_frame):
    f = make_frame((1, PERSON, 0, 0, 0, 0, 0b000))
    assert rule_ppe_compliance(ctx_for(f[None])) == 1.0


def test_proximity_zero_when_far_and_one_when_touching(make_frame):
    far = make_frame((1, PERSON, 0, 0), (2, MACHINE, 100, 100))
    near = make_frame((1, PERSON, 0, 0), (2, MACHINE, 1, 0))
    assert rule_vectorized_proximity(ctx_for(far[None])) == 0.0
    assert rule_vectorized_proximity(ctx_for(near[None])) == 1.0


def test_proximity_treats_vehicles_as_hazards(make_frame):
    f = make_frame((1, PERSON, 0, 0), (2, VEHICLE, 1, 0))
    assert rule_vectorized_proximity(ctx_for(f[None])) == 1.0


def test_ttc_converging_scores_positive_diverging_zero(make_frame):
    converge = make_frame((1, PERSON, 10, 0, -2, 0), (2, MACHINE, 0, 0, 0, 0))
    diverge = make_frame((1, PERSON, 10, 0, 2, 0), (2, MACHINE, 0, 0, 0, 0))
    assert rule_kinematic_ttc(ctx_for(converge[None])) > 0.0
    assert rule_kinematic_ttc(ctx_for(diverge[None])) == 0.0


def test_cone_zone_needs_three_cones(make_frame):
    two = make_frame((1, PERSON, 5, 5), (2, CONE, 0, 0), (3, CONE, 10, 10))
    three = make_frame((1, PERSON, 5, 5), (2, CONE, 0, 0), (3, CONE, 10, 0), (4, CONE, 10, 10))
    assert rule_cone_zone(ctx_for(two[None])) == 0.0
    assert rule_cone_zone(ctx_for(three[None])) == 1.0


def test_cone_zone_person_outside_is_zero(make_frame):
    f = make_frame((1, PERSON, 500, 500), (2, CONE, 0, 0), (3, CONE, 10, 0), (4, CONE, 10, 10))
    assert rule_cone_zone(ctx_for(f[None])) == 0.0