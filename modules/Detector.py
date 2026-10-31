from ultralytics import YOLO
from trackers import ByteTrackTracker as Tracker
import supervision as sv
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


CLASSES = ('Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle')
PPE_IDS = (0, 1, 2, 3, 4, 7)
ID_CONE = 6
ID_PERSON = 5
ID_MACHINERY = 8
ID_VEHICLE = 9


def match_ppe(person_dets: sv.Detections, ppe_dets: sv.Detections) -> dict:
    ppe_map = {}
    if len(person_dets) == 0 or len(ppe_dets) == 0:
        return ppe_map

    ppe_cx = (ppe_dets.xyxy[:, 0] + ppe_dets.xyxy[:, 2]) / 2.0
    ppe_cy = (ppe_dets.xyxy[:, 1] + ppe_dets.xyxy[:, 3]) / 2.0

    for i in range(len(person_dets)):
        t_id = person_dets.tracker_id[i]

        x1, y1, x2, y2 = person_dets.xyxy[i]

        # Boolean mask of PPE centers that fall inside this person's bounding box
        inside_x = (ppe_cx >= x1) & (ppe_cx <= x2)
        inside_y = (ppe_cy >= y1) & (ppe_cy <= y2)
        inside_mask = inside_x & inside_y

        # Extract classes of the enclosed PPE
        person_ppe_classes = ppe_dets.class_id[inside_mask]

        # Map specific PPE classes (0: Hardhat, 1: Mask, 7: Vest)
        hh = 1 if 0 in person_ppe_classes else 0
        mask = 1 if 1 in person_ppe_classes else 0
        vest = 1 if 7 in person_ppe_classes else 0

        # Keep the existing tracker-ID key for valid IDs. Use a row key for
        # missing IDs so multiple people with tracker_id == -1 cannot collide.
        key = t_id if t_id is not None and int(t_id) >= 0 else ("row", i)
        ppe_map[key] = [hh, mask, vest]

    return ppe_map

class Detector:
    def __init__(self, model: str, framerate=30, track_life = 30, **kwargs):
        self.model = YOLO(model, task='detect')
        self.tracker = Tracker(track_life, framerate, **kwargs)
        self.track_dict = {}
        self.identity_tracks = {}
        self.identity_frame = 0
        self.identity_max_age = max(1, int(track_life))
        self.next_object_id = 1
        self.next_cone_id = -2
        self.current_image = None

    @staticmethod
    def _box_iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    def _appearance_descriptor(self, box):
        if self.current_image is None:
            return None
        height, width = self.current_image.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = self.current_image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        return histogram.astype(np.float32).reshape(-1)

    def _resolve_ids(self, dets: sv.Detections, type_idx: int) -> list[int]:
        """Map per-frame tracker IDs to stable application IDs.

        Short gaps and tracker ID resets are bridged by one-to-one matching
        within the same object class using predicted center, box overlap,
        relative box size, and the incoming tracker ID as supporting evidence.
        People also use a lightweight color-histogram comparison. Cones use
        tighter spatial matching because they are stationary.
        """
        count = len(dets)
        if count == 0:
            return []

        boxes = np.asarray(dets.xyxy, dtype=np.float32)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        appearances = [
            self._appearance_descriptor(box) if type_idx == 0 else None
            for box in boxes
        ]
        if dets.tracker_id is None:
            raw_ids = [-1] * count
        else:
            raw_ids = [
                -1 if value is None else int(value)
                for value in np.asarray(dets.tracker_id).tolist()
            ]

        candidates = [
            (track_id, state)
            for track_id, state in self.identity_tracks.items()
            if state["type"] == type_idx
            and 0 < self.identity_frame - state["last_frame"] <= self.identity_max_age
        ]
        costs = np.full((len(candidates), count), 1e6, dtype=np.float64)
        for row, (_, state) in enumerate(candidates):
            gap = self.identity_frame - state["last_frame"]
            displacement = state["velocity"] * gap
            predicted_center = state["center"] + displacement
            predicted_box = state["box"] + np.tile(displacement, 2)
            box_area = max(
                1.0,
                (state["box"][2] - state["box"][0])
                * (state["box"][3] - state["box"][1]),
            )
            box_scale = np.sqrt(box_area)
            speed = float(np.linalg.norm(state["velocity"]))
            if type_idx == 3:
                max_distance = max(24.0, 0.6 * box_scale)
            else:
                max_distance = min(300.0, max(40.0, 0.9 * box_scale + 1.25 * speed * gap))

            for column, (box, center, raw_id, appearance) in enumerate(
                zip(boxes, centers, raw_ids, appearances)
            ):
                distance = float(np.linalg.norm(center - predicted_center))
                if distance > max_distance:
                    continue
                overlap = self._box_iou(predicted_box, box)
                current_area = max(
                    1.0,
                    (box[2] - box[0]) * (box[3] - box[1]),
                )
                size_distance = min(
                    1.0,
                    abs(np.log(current_area / box_area)) / np.log(4.0),
                )
                if (
                    type_idx == 0
                    and state["appearance"] is not None
                    and appearance is not None
                ):
                    appearance_distance = cv2.compareHist(
                        state["appearance"].reshape(8, 8),
                        appearance.reshape(8, 8),
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                    cost = (
                        0.40 * (1.0 - overlap)
                        + 0.30 * (distance / max_distance)
                        + 0.10 * size_distance
                        + 0.20 * appearance_distance
                    )
                else:
                    cost = (
                        0.50 * (1.0 - overlap)
                        + 0.35 * (distance / max_distance)
                        + 0.15 * size_distance
                    )
                if raw_id >= 0 and raw_id == state["raw_id"]:
                    cost = max(0.0, cost - 0.15)
                costs[row, column] = cost

        resolved = [None] * count
        if costs.size:
            track_rows, detection_columns = linear_sum_assignment(costs)
            for row, column in zip(track_rows, detection_columns):
                assigned_cost = costs[row, column]
                if assigned_cost > 0.82:
                    continue
                # Avoid forcing an identity when two previous objects are
                # equally plausible matches for the current detection.
                column_costs = np.sort(costs[:, column])
                if len(column_costs) > 1 and column_costs[1] - column_costs[0] < 0.06:
                    continue
                resolved[column] = candidates[row][0]

        for index, track_id in enumerate(resolved):
            if track_id is None:
                if type_idx == 3:
                    track_id = self.next_cone_id
                    self.next_cone_id -= 1
                else:
                    track_id = self.next_object_id
                    self.next_object_id += 1

            previous = self.identity_tracks.get(track_id)
            if type_idx == 3:
                velocity = np.zeros(2, dtype=np.float32)
            elif previous is not None:
                gap = max(1, self.identity_frame - previous["last_frame"])
                velocity = (centers[index] - previous["center"]) / gap
            else:
                velocity = np.zeros(2, dtype=np.float32)
            appearance = appearances[index]
            if previous is not None and previous["appearance"] is not None and appearance is not None:
                appearance = 0.8 * previous["appearance"] + 0.2 * appearance
                appearance /= max(float(appearance.sum()), 1e-12)
            self.identity_tracks[track_id] = {
                "type": type_idx,
                "raw_id": raw_ids[index],
                "box": boxes[index].copy(),
                "center": centers[index].copy(),
                "velocity": velocity.astype(np.float32, copy=False),
                "appearance": appearance,
                "last_frame": self.identity_frame,
            }
            resolved[index] = int(track_id)

        expired = [
            track_id
            for track_id, state in self.identity_tracks.items()
            if self.identity_frame - state["last_frame"] > self.identity_max_age
        ]
        for track_id in expired:
            del self.identity_tracks[track_id]
        return resolved

    def __call__(self, frame, **kwargs):
        self.current_image = np.asarray(frame)
        return sv.Detections.from_ultralytics(self.model.predict(frame, verbose=False, **kwargs)[0])

    def only_track(self, frame, ids=(), **kwargs):
        if not isinstance(frame, sv.Detections):
            results = self(frame, **kwargs)
        else:
            results = frame
        # TODO: add protections for empty results
        tracked_dets = self.tracker.update(results[np.isin(results.class_id, ids)])
        untracked_dets = results[np.isin(results.class_id, ids, invert=True)]
        return tracked_dets, untracked_dets

    def separate(self, dets=None, frame=None, **kwargs):
        if dets is None:
            if frame is None:
                raise ValueError("Either dets or frame must be provided.")
            dets = self(frame, **kwargs)
        ppe_dets = dets[np.isin(dets.class_id, (0, 1, 2, 3, 4, 7))]
        cone_dets = dets[dets.class_id == 6]

        person_dets = dets[dets.class_id == 5]
        machinery_dets = dets[dets.class_id == 8]
        vehicle_dets = dets[dets.class_id == 9]

        return {"ppe": ppe_dets, "person": person_dets, "cone": cone_dets,
                "machinery": machinery_dets, "vehicle": vehicle_dets}

    def get_vectors(self, frame, ud = None) -> list[np.ndarray]:
        """
        Orchestrates inference, tracking, separation, and vector formatting.
        Returns a list of 4 matrices [Persons, Machinery, Vehicles, Cones].
        Vector format: [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
        """
        if ud is None:
            td, ud = self.only_track(frame, ids=(5, 8, 9))
        else:
            td = frame
        self.identity_frame += 1
        moving = self.separate(td)
        static = self.separate(ud)

        ppe_map = match_ppe(moving["person"], static["ppe"])

        final_vectors = [[], [], [], []]
        current_centroids = {}

        def process_subset(dets: sv.Detections, type_idx: int):
            if len(dets) == 0:
                return
            type_onehot = [0, 0, 0, 0]
            type_onehot[type_idx] = 1
            xyxy = np.asarray(dets.xyxy)
            cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
            cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
            persistent_ids = self._resolve_ids(dets, type_idx)

            for i in range(len(dets)):
                raw_id = (
                    int(dets.tracker_id[i])
                    if dets.tracker_id is not None and dets.tracker_id[i] is not None
                    else -1
                )
                t_id = persistent_ids[i]

                icx, icy = cx[i], cy[i]
                vx, vy = 0.0, 0.0

                if t_id >= 0:
                    current_centroids[t_id] = (icx, icy)
                    # Derive velocity from this ID's previous observed center.
                    # If the ID was absent on the preceding frame, start at zero
                    # instead of bridging a gap and creating a false jump.
                    if t_id in self.track_dict:
                        previous_x, previous_y = self.track_dict[t_id]
                        vx, vy = icx - previous_x, icy - previous_y

                # Default PPE state for non-persons
                hh, mask, vest, na = 0, 0, 0, 1

                if type_idx == 0:  # Person logic
                    ppe_key = raw_id if raw_id >= 0 else ("row", i)
                    hh, mask, vest = ppe_map.get(ppe_key, [0, 0, 0])
                    na = 0  # NA is 0 for persons as they are eligible for PPE

                vec = [t_id, *type_onehot, icx, icy, vx, vy, hh, mask, vest, na]
                final_vectors[type_idx].append(vec)

        # 0: Persons, 1: Machinery, 2: Vehicles, 3: Cones
        # I don't like this redundancy but it really is optimal.
        process_subset(moving["person"], 0)
        process_subset(moving["machinery"], 1)
        process_subset(moving["vehicle"], 2)
        process_subset(static["cone"], 3)

        self.track_dict = current_centroids

        # Convert lists to 2D numpy arrays, return empty 0x13 arrays if no objects found
        return [np.array(v) if len(v) > 0 else np.empty((0, 13)) for v in final_vectors]

    # Vector format: [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
    def draw(self, frame, vectors):
        colors = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 255)}
        for idx, vector in enumerate(vectors):
            for det in vector:
                if det.size == 0:
                    continue

                cv2.circle(frame, (int(det[5]), int(det[6])), 5, colors[idx], -1)
                cv2.putText(frame, str(det[0]), (int(det[5]), int(det[6])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                t = np.linalg.norm(det[7:9])/10.0
                C = gyr_gradient(t)
                cv2.arrowedLine(frame, (int(det[5]), int(det[6])), (int(det[5] + 10*det[7]), int(det[6] + 10*det[8])), C, 2)
        return frame


def gyr_gradient(t: float) -> tuple[int, int, int]:
    # Enforce strict bounds between 0.0 and 1.0
    t = max(0.0, min(1.0, t))

    # Red rises quickly in the first half and stays fully saturated at 255
    r = int(255 * min(1.0, t * 2.0))

    # Green stays fully saturated at 255 and drops off in the second half
    g = int(255 * min(1.0, (1.0 - t) * 2.0))

    # Blue remains entirely absent throughout this gradient
    b = 0

    return b, g, r


if __name__ == "__main__":  # No se toca!!!
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    cam = cv2.VideoCapture(0)
    det = Detector("../assets/ppe_50ep.engine")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        dets = det(frame)
        vecs = det.get_vectors(dets)
        frame = det.draw(frame, vecs)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
