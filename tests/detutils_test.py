"""Pure-function tests for Detector.py. YOLO weights are never loaded."""
import numpy as np
import pytest

sv = pytest.importorskip("supervision")
pytest.importorskip("ultralytics")
pytest.importorskip("trackers")

from modules.Detector import CLASSES, gyr_gradient, match_ppe


def dets(boxes, classes, ids=None):
    d = sv.Detections(
        xyxy=np.array(boxes, dtype=float).reshape(-1, 4),
        class_id=np.array(classes, dtype=int),
    )
    if ids is not None:
        d.tracker_id = np.array(ids)
    return d


def test_class_table_matches_constants():
    assert CLASSES[5] == "Person" and CLASSES[6] == "Safety Cone"
    assert CLASSES[8] == "machinery" and CLASSES[9] == "vehicle"


def test_match_ppe_empty_inputs():
    assert match_ppe(dets([], []), dets([[0, 0, 1, 1]], [0])) == {}
    assert match_ppe(dets([[0, 0, 10, 10]], [5], [1]), dets([], [])) == {}


def test_match_ppe_assigns_by_box_containment():
    person = dets([[0, 0, 100, 200], [300, 0, 400, 200]], [5, 5], [1, 2])
    ppe = dets([[40, 5, 60, 25], [40, 80, 60, 150], [340, 5, 360, 25]], [0, 7, 0])  # hh, vest, hh
    out = match_ppe(person, ppe)
    assert list(out[1]) == [1, 0, 1]   # hardhat + vest
    assert list(out[2]) == [1, 0, 0]   # hardhat only


def test_negative_ppe_classes_do_not_count_as_present():
    person = dets([[0, 0, 100, 200]], [5], [1])
    ppe = dets([[40, 5, 60, 25]], [2])  # NO-Hardhat
    assert list(match_ppe(person, ppe)[1]) == [0, 0, 0]


@pytest.mark.parametrize("t", [-5.0, 0.0, 0.5, 1.0, 5.0])
def test_gyr_gradient_valid_bgr_and_clamped(t):
    b, g, r = gyr_gradient(t)
    assert b == 0 and 0 <= g <= 255 and 0 <= r <= 255


def test_gyr_gradient_endpoints():
    assert gyr_gradient(0.0) == (0, 255, 0)
    assert gyr_gradient(1.0) == (0, 0, 255)