"""Detector utility and tracker-handoff tests; no model weights required."""

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from modules.Detector import Detector, match_ppe  # noqa: E402 - import after dependency check

def detections(boxes, classes, ids=None):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        class_id=np.asarray(classes, dtype=int),
        tracker_id=None if ids is None else np.asarray(ids, dtype=int),
    )


def test_ppe_is_associated_with_each_tracked_person():
    people = detections(
        [[0, 0, 100, 200], [200, 0, 300, 200]],
        [5, 5],
        [11, 22],
    )
    ppe = detections(
        [[40, 5, 60, 25], [240, 5, 260, 25]],
        [0, 7],
    )

    assert match_ppe(people, ppe) == {
        11: [1, 0, 0],
        22: [0, 0, 1],
    }


def test_ppe_outside_person_is_not_associated():
    people = detections([[0, 0, 100, 200]], [5], [11])
    ppe = detections([[200, 5, 220, 25]], [0])

    assert match_ppe(people, ppe) == {11: [0, 0, 0]}


def test_no_people_returns_empty_ppe_map():
    people = detections([], [], [])
    ppe = detections([[40, 5, 60, 25]], [0])

    assert match_ppe(people, ppe) == {}


def test_no_ppe_returns_empty_ppe_map():
    people = detections([[0, 0, 100, 200]], [5], [11])
    ppe = detections([], [])

    assert match_ppe(people, ppe) == {}


def test_only_track_sends_selected_classes_to_tracker():
    class FakeTracker:
        def __init__(self):
            self.received = None

        def update(self, selected):
            self.received = selected
            return selected

    # Avoid initializing YOLO or loading model weights.
    detector = Detector.__new__(Detector)
    detector.tracker = FakeTracker()

    scene = detections(
        [
            [0, 0, 100, 200],
            [200, 0, 300, 200],
            [400, 0, 500, 200],
            [600, 0, 620, 20],
            [40, 5, 60, 25],
        ],
        [5, 8, 9, 6, 0],
    )

    tracked, untracked = detector.only_track(scene, ids=(5, 8, 9))

    np.testing.assert_array_equal(
        detector.tracker.received.class_id, [5, 8, 9]
    )
    np.testing.assert_array_equal(tracked.class_id, [5, 8, 9])
    np.testing.assert_array_equal(untracked.class_id, [6, 0])