"""Focused tests for application-level identity continuity."""
import numpy as np
import pytest

sv = pytest.importorskip("supervision")
pytest.importorskip("ultralytics")
pytest.importorskip("trackers")
pytest.importorskip("scipy")

from modules.Detector import Detector, match_ppe


def detections(boxes, classes, ids=None):
    out = sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        class_id=np.asarray(classes, dtype=int),
    )
    if ids is not None:
        out.tracker_id = np.asarray(ids)
    return out


def resolver(max_age=3):
    detector = Detector.__new__(Detector)
    detector.identity_tracks = {}
    detector.identity_frame = 0
    detector.identity_max_age = max_age
    detector.next_object_id = 1
    detector.next_cone_id = -2
    detector.current_image = None
    return detector


def test_reuses_person_id_after_tracker_id_changes():
    detector = resolver()
    detector.identity_frame = 1
    first_id = detector._resolve_ids(
        detections([[0, 0, 20, 40]], [5], [11]), type_idx=0
    )[0]

    detector.identity_frame = 2
    second_id = detector._resolve_ids(
        detections([[1, 0, 21, 40]], [5], [99]), type_idx=0
    )[0]

    assert second_id == first_id


def test_reuses_cone_ids_when_detection_order_changes():
    detector = resolver()
    detector.identity_frame = 1
    first_ids = detector._resolve_ids(
        detections([[5, 5, 15, 15], [95, 95, 105, 105]], [6, 6]), type_idx=3
    )

    detector.identity_frame = 2
    second_ids = detector._resolve_ids(
        detections([[95, 95, 105, 105], [5, 5, 15, 15]], [6, 6]), type_idx=3
    )

    assert second_ids == first_ids[::-1]


def test_does_not_reconnect_after_identity_timeout():
    detector = resolver(max_age=2)
    detector.identity_frame = 1
    first_id = detector._resolve_ids(
        detections([[0, 0, 20, 40]], [8], [1]), type_idx=1
    )[0]

    detector.identity_frame = 4
    later_id = detector._resolve_ids(
        detections([[1, 0, 21, 40]], [8], [2]), type_idx=1
    )[0]

    assert later_id != first_id


def test_ambiguous_person_match_gets_new_id():
    detector = resolver()
    detector.identity_frame = 1
    detector._resolve_ids(
        detections([[0, 0, 20, 40], [2, 0, 22, 40]], [5, 5], [1, 2]),
        type_idx=0,
    )

    detector.identity_frame = 2
    ambiguous_id = detector._resolve_ids(
        detections([[1, 0, 21, 40]], [5], [3]), type_idx=0
    )[0]

    assert ambiguous_id not in {1, 2}


def test_untracked_people_keep_separate_ppe_matches():
    people = detections([[0, 0, 100, 200], [200, 0, 300, 200]], [5, 5], [-1, -1])
    ppe = detections([[40, 5, 60, 25], [240, 5, 260, 25]], [0, 7])

    matches = match_ppe(people, ppe)

    assert matches[("row", 0)] == [1, 0, 0]
    assert matches[("row", 1)] == [0, 0, 1]
