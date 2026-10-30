from ultralytics import YOLO
from trackers import ByteTrackTracker as Tracker
import supervision as sv
import cv2
import numpy as np


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
        if t_id is None:
            continue

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

        ppe_map[t_id] = [hh, mask, vest]

    return ppe_map

class Detector:
    def __init__(self, model: str, framerate=30, track_life = 30, **kwargs):
        self.model = YOLO(model, task='detect')
        self.tracker = Tracker(track_life, framerate, **kwargs)
        self.track_dict = {}

    def __call__(self, frame, **kwargs):
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

            for i in range(len(dets)):
                t_id = dets.tracker_id[i] if dets.tracker_id is not None else -1
                if t_id is None:
                    t_id = -1

                # Untracked items (like cones) get ID -1, tracked must have an ID
                if type_idx != 3 and (t_id is None or t_id == -1):
                    continue

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
                    hh, mask, vest = ppe_map.get(t_id, [0, 0, 0])
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
    def draw(self, frame, dets, vectors):
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
        frame = det.draw(frame, dets, vecs)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
