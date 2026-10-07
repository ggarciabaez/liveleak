from ultralytics import YOLO
from trackers import ByteTrackTracker as Tracker
import supervision as sv
import numpy as np


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
        inside_x = (ppe_cx >= x1) & (ppe_cx <= x2)
        inside_y = (ppe_cy >= y1) & (ppe_cy <= y2)
        inside_mask = inside_x & inside_y

        person_ppe_classes = ppe_dets.class_id[inside_mask]

        hh = 1 if 0 in person_ppe_classes else 0
        mask = 1 if 1 in person_ppe_classes else 0
        vest = 1 if 7 in person_ppe_classes else 0

        ppe_map[t_id] = [hh, mask, vest]

    return ppe_map


class BatchedDetector:
    def __init__(self, model: str, batch_size: int, framerate=30, track_life=30, **kwargs):
        # Load the TRT engine ONCE to save VRAM
        self.model = YOLO(model, task='detect')
        self.batch_size = batch_size

        # Instantiate B independent trackers and state dictionaries to prevent cross-contamination
        self.trackers = [Tracker(track_life, framerate, **kwargs) for _ in range(batch_size)]
        self.track_dicts = [{} for _ in range(batch_size)]

    def separate(self, dets: sv.Detections):
        ppe = dets[np.isin(dets.class_id, (0, 1, 2, 3, 4, 7))]
        cone = dets[dets.class_id == 6]
        person = dets[dets.class_id == 5]
        machinery = dets[dets.class_id == 8]
        vehicle = dets[dets.class_id == 9]
        return {"ppe": ppe, "person": person, "cone": cone, "machinery": machinery, "vehicle": vehicle}

    def _extract_vectors(self, moving, static, ppe_map, batch_idx):
        final_vectors = []
        current_centroids = {}
        track_dict = self.track_dicts[batch_idx]

        def process_subset(dets: sv.Detections, type_idx: int, type_onehot: list):
            if len(dets) == 0:
                return np.empty((0, 13))

            cx = (dets.xyxy[:, 0] + dets.xyxy[:, 2]) / 2.0
            cy = (dets.xyxy[:, 1] + dets.xyxy[:, 3]) / 2.0

            subset_vecs = []

            for i in range(len(dets)):
                t_id = dets.tracker_id[i] if dets.tracker_id is not None else -1

                if type_idx != 3 and (t_id is None or t_id == -1):
                    continue

                icx, icy = cx[i], cy[i]
                vx, vy = 0.0, 0.0

                if t_id != -1:
                    current_centroids[t_id] = (icx, icy)
                    if t_id in track_dict:
                        px, py = track_dict[t_id]
                        vx, vy = icx - px, icy - py

                hh, mask, vest, na = 0, 0, 0, 1

                if type_idx == 0:
                    hh, mask, vest = ppe_map.get(t_id, [0, 0, 0])
                    na = 0

                vec = [t_id, *type_onehot, icx, icy, vx, vy, hh, mask, vest, na]
                subset_vecs.append(vec)

            return np.array(subset_vecs) if len(subset_vecs) > 0 else np.empty((0, 13))

        # Compile subsets
        final_vectors.append(process_subset(moving["person"], 0, [1, 0, 0, 0]))
        final_vectors.append(process_subset(moving["machinery"], 1, [0, 1, 0, 0]))
        final_vectors.append(process_subset(moving["vehicle"], 2, [0, 0, 1, 0]))
        final_vectors.append(process_subset(static["cone"], 3, [0, 0, 0, 1]))

        self.track_dicts[batch_idx] = current_centroids

        # Stack subsets to yield an (N, 13) array for this frame
        valid_vecs = [v for v in final_vectors if v.size > 0]
        return np.vstack(valid_vecs) if valid_vecs else np.empty((0, 13))

    def process_batch(self, frames: list) -> list[np.ndarray]:
        """
        Accepts a list of B frames.
        Returns a list of B numpy arrays, each of shape (N, 13).
        """
        batch_outputs = []

        for b, frame in enumerate(frames):
            # Gracefully handle exhausted streams padded with None from VideoDataset
            if frame is None:
                batch_outputs.append(np.empty((0, 13)))
                continue

            # 1. Inference (isolated loop to bypass TensorRT batched input failure)
            res = self.model.predict(frame, verbose=False)[0]
            dets = sv.Detections.from_ultralytics(res)

            # 2. Tracking (route to stream-specific tracker)
            track_mask = np.isin(dets.class_id, [5, 8, 9])
            tracked_dets = self.trackers[b].update(dets[track_mask])
            untracked_dets = dets[~track_mask]

            # 3. Separation
            moving = self.separate(tracked_dets)
            static = self.separate(untracked_dets)

            # 4. PPE Association
            ppe_map = match_ppe(moving["person"], static["ppe"])

            # 5. Vectorization
            frame_vectors = self._extract_vectors(moving, static, ppe_map, b)
            batch_outputs.append(frame_vectors)

        return batch_outputs

    def reset(self):
        for tracker in self.trackers:
            tracker.reset()