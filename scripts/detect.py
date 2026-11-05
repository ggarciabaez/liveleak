import json
from pathlib import Path
from glob import glob
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from trackers import OCSORTTracker, ByteTrackTracker

# --- Shared Pipeline & Tracker Configuration ---
PIPELINE_CFG = {
    "model_path": "../assets/ppe_50ep.engine",
    "conf": 0.25,
    "target_fps": 30.0,  # Target effective FPS on Jetson Orin
    "tracker": {
        "type": "OCSORTTracker",
        "track_activation_threshold": 0.25,
        "minimum_matching_threshold": 0.8,
        "minimum_consecutive_frames": 1,
        "track_life_seconds": 2.0,  # Converted to buffer frames via effective_fps
    },
}

CLASSES = (
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
    "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
)
PPE_IDS = (0, 1, 2, 3, 4, 7)
ID_PERSON = 5
ID_CONE = 6
ID_MACHINERY = 8
ID_VEHICLE = 9
TRACKED = (ID_PERSON, ID_MACHINERY, ID_VEHICLE)


def make_tracker(cfg: dict, effective_fps: float):
    t_cfg = cfg["tracker"]
    fps_int = max(1, round(effective_fps))
    buffer_frames = max(1, round(t_cfg["track_life_seconds"] * effective_fps))
    TRACKER = ByteTrackTracker if t_cfg["type"] == "ByteTrackTracker" else OCSORTTracker
    return TRACKER(buffer_frames, fps_int,
                    minimum_consecutive_frames=t_cfg["minimum_consecutive_frames"],
                    minimum_iou_threshold=t_cfg["minimum_matching_threshold"],
    )


def get_video_props(path: str, target_fps: float) -> tuple[int, int, float, int, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    vid_stride = max(1, round(src_fps / target_fps))
    eff_fps = src_fps / vid_stride
    return w, h, src_fps, vid_stride, eff_fps


def process_result(results: sv.Detections, tracker):
    is_tracked = np.isin(results.class_id, TRACKED)
    tracked_dets = tracker.update(results[is_tracked])
    untracked_dets = results[~is_tracked]

    def separate(dets: sv.Detections):
        return {
            "ppe": dets[np.isin(dets.class_id, PPE_IDS)],
            "person": dets[dets.class_id == ID_PERSON],
            "cone": dets[dets.class_id == ID_CONE],
            "machinery": dets[dets.class_id == ID_MACHINERY],
            "vehicle": dets[dets.class_id == ID_VEHICLE],
        }

    return separate(tracked_dets), separate(untracked_dets)


def match_ppe(person_dets: sv.Detections, ppe_dets: sv.Detections) -> dict:
    ppe_map = {}
    if len(person_dets) == 0 or len(ppe_dets) == 0 or person_dets.tracker_id is None:
        return ppe_map

    ppe_cx = (ppe_dets.xyxy[:, 0] + ppe_dets.xyxy[:, 2]) * 0.5
    ppe_cy = (ppe_dets.xyxy[:, 1] + ppe_dets.xyxy[:, 3]) * 0.5

    for i in range(len(person_dets)):
        t_id = person_dets.tracker_id[i]
        if t_id is None or t_id == -1:
            continue

        x1, y1, x2, y2 = person_dets.xyxy[i]
        inside_mask = (ppe_cx >= x1) & (ppe_cx <= x2) & (ppe_cy >= y1) & (ppe_cy <= y2)
        person_ppe_classes = ppe_dets.class_id[inside_mask]

        hh = 1.0 if 0 in person_ppe_classes else 0.0
        mask = 1.0 if 1 in person_ppe_classes else 0.0
        vest = 1.0 if 7 in person_ppe_classes else 0.0
        ppe_map[int(t_id)] = (hh, mask, vest)

    return ppe_map


def extract_frame_rows(
    results: sv.Detections, tracker: OCSORTTracker | ByteTrackTracker, track_dict: dict
) -> np.ndarray:
    """Extracts and merges all entities in a single frame into an (n_t, 13) float32 array.
    Note: Cones keep ID = -1 here; resolution-independent spatial hashing happens at load time."""
    moving, static = process_result(results, tracker)
    ppe_map = match_ppe(moving["person"], static["ppe"])
    # very stingy btw
    tracks = {track.tracker_id: (track.state_estimator.kf.state, track.get_state_bbox()) for track in tracker.tracks if (track.tracker_id >= 0 and track.time_since_update == 0)}

    rows = []
    current_centroids = {}

    def process_subset(dets: sv.Detections, type_idx: int):
        if len(dets) == 0:
            return
        type_onehot = [0.0, 0.0, 0.0, 0.0]
        type_onehot[type_idx] = 1.0

        for i in range(len(dets)):
            t_id = int(dets.tracker_id[i]) if (dets.tracker_id is not None and dets.tracker_id[i] is not None) else -1

            # Tracked classes must have a valid tracker ID; cones (type_idx == 3) use -1
            if type_idx != 3 and t_id == -1:
                continue
            # TODO: use this for the actual detector
            if t_id == -1:
                cx = cy = vx = vy = 0.0
            else:
                state, (x1, y1, x2, y2) = tracks[t_id]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                vx, vy = np.asarray(state).reshape(-1)[4:6]
                current_centroids[t_id] = (cx, cy)

            if type_idx == 0:
                hh, mask, vest = ppe_map.get(t_id, (0.0, 0.0, 0.0))
                na = 0.0
            else:
                hh, mask, vest, na = 0.0, 0.0, 0.0, 1.0

            rows.append([float(t_id), *type_onehot, cx, cy, vx, vy, hh, mask, vest, na])

    process_subset(moving["person"], 0)
    process_subset(moving["machinery"], 1)
    process_subset(moving["vehicle"], 2)
    process_subset(static["cone"], 3)

    # Update velocity memory (keep previous tracks alive across brief occlusions)
    track_dict.update(current_centroids)

    if not rows:
        return np.empty((0, 13), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def process_video(video_path: str, model: YOLO, out_path: Path, cfg: dict):
    w, h, src_fps, vid_stride, eff_fps = get_video_props(video_path, cfg["target_fps"])
    tracker = make_tracker(cfg, eff_fps)
    tracks = {}

    frame_chunks = []
    frame_counts = [0]

    results = model(
        video_path,
        stream=True,
        conf=cfg["conf"],
        vid_stride=vid_stride,
        verbose=False,
    )

    for result in results:
        dets = sv.Detections.from_ultralytics(result)
        f_rows = extract_frame_rows(dets, tracker, tracks)
        if len(f_rows) > 0:
            frame_chunks.append(f_rows)
        frame_counts.append(len(f_rows))

    # Build contiguous (R, 13) array and (F + 1,) CSR pointer array
    if frame_chunks:
        rows = np.ascontiguousarray(np.concatenate(frame_chunks, axis=0), dtype=np.float32)
    else:
        rows = np.empty((0, 13), dtype=np.float32)

    frame_ptr = np.ascontiguousarray(np.cumsum(frame_counts, dtype=np.int64))

    metadata = {
        "source_video": Path(video_path).name,
        "width": w,
        "height": h,
        "src_fps": src_fps,
        "vid_stride": vid_stride,
        "effective_fps": eff_fps,
        "num_frames": len(frame_counts) - 1,
        "total_rows": int(rows.shape[0]),
        "config": cfg,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        rows=rows,
        frame_ptr=frame_ptr,
        metadata=np.array(json.dumps(metadata)),
    )
    print(f"Saved {out_path.name}: {metadata['num_frames']} frames, {rows.shape[0]} rows ({w}x{h} @ {eff_fps:.1f} fps)")


if __name__ == "__main__":
    model = YOLO(PIPELINE_CFG["model_path"], task="detect")
    # videos = ["../data/construction/videos/pexels-akin-victor-10202777-3842x2160-30fps.mp4"]
    videos = glob("../data/construction/videos/*.mp4")
    out_dir = Path("results")

    for idx, v in enumerate(videos):
        process_video(v, model, out_dir / f"{idx:04d}.npz", PIPELINE_CFG)



