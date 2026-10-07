"""Prepare and load per-video STGNN motion sequences

The public data layout is one compressed ``.npz`` file per source video. Each
file stores a flat ``(detections, 13)`` feature array, frame offsets, and JSON
metadata; no Python objects are pickled. ``Dataset`` keeps videos separate
when making train/validation/test splits and builds only windows whose tracked
entities are present in every frame (including the prediction target frame).

Feature columns are ``[track_id, person, machine, vehicle, cone, x, y, vx,
vy, hardhat, mask, vest, na]``. Positions are normalized by image width and
height; velocities are normalized in the same units per processed frame.
Untracked ``-1`` detections are removed because they cannot be aligned safely
across time
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


FEATURE_COUNT = 13
DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def normalize_frame_vectors(
    vectors: Sequence[np.ndarray],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Combine Detector output, remove untracked rows, and normalize geometry

    The detector reports pixel coordinates and per-update pixel displacement.
    This conversion must also be used by live inference if it consumes these
    prepared features
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")

    nonempty = [np.asarray(group) for group in vectors if len(group)]
    if not nonempty:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)

    features = np.concatenate(nonempty, axis=0).astype(np.float32, copy=False)
    if features.ndim != 2 or features.shape[1] != FEATURE_COUNT:
        raise ValueError("each detection row must contain exactly 13 features")
    if not np.isfinite(features).all():
        raise ValueError("detector features must be finite")

    # Cones and other detections without a persistent identity cannot form a
    # temporal training target, so omit them rather than inventing identities.
    features = features[features[:, 0] >= 0].copy()
    if features.size == 0:
        return np.empty((0, FEATURE_COUNT), dtype=np.float32)
    if np.unique(features[:, 0]).size != features.shape[0]:
        raise ValueError("track IDs must be unique within each frame")

    features[:, 5] /= image_width
    features[:, 6] /= image_height
    features[:, 7] /= image_width
    features[:, 8] /= image_height
    return features


def align_window(frames: Sequence[np.ndarray]) -> np.ndarray | None:
    """Align a frame window by persistent ID; return ``None`` if empty/invalid"""
    if not frames:
        return None

    frame_ids = []
    frame_maps = []
    for frame in frames:
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 2 or frame.shape[1] != FEATURE_COUNT:
            raise ValueError("each frame must have shape (entities, 13)")
        if not np.isfinite(frame).all():
            raise ValueError("frame features must be finite")
        tracked = frame[frame[:, 0] >= 0]
        ids = tracked[:, 0]
        if np.unique(ids).size != ids.size:
            raise ValueError("track IDs must be unique within each frame")
        frame_ids.append(ids)
        frame_maps.append({int(row[0]): row for row in tracked})

    if any(ids.size == 0 for ids in frame_ids):
        return None
    shared_ids = frame_ids[0]
    for ids in frame_ids[1:]:
        shared_ids = np.intersect1d(shared_ids, ids, assume_unique=True)
    if shared_ids.size == 0:
        return None

    # Sorting the IDs gives each entity the same row position in every frame.
    aligned = np.stack(
        [np.stack([mapping[int(track_id)] for track_id in shared_ids]) for mapping in frame_maps]
    )
    return aligned.astype(np.float32, copy=False)


def _load_video_file(path: Path) -> tuple[list[np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["features"], dtype=np.float32)
        offsets = np.asarray(archive["offsets"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata"].item()))

    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT:
        raise ValueError(f"{path} has an invalid feature array")
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError(f"{path} has invalid frame offsets")
    if offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{path} has inconsistent frame offsets")

    frames = [values[offsets[i] : offsets[i + 1]] for i in range(offsets.size - 1)]
    return frames, metadata


def prepare_video(
    video_path: str | Path,
    detector_weights: str | Path,
    output_dir: str | Path,
    frame_step: int = 1,
) -> Path:
    """Run the project's detector/tracker and cache normalized frames for one clip"""
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")

    # Keep heavyweight video dependencies optional when merely importing the
    # Dataset class to read already-prepared feature files.
    import cv2
    from modules.Detector import Detector

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {video_path}")

    detector = Detector(str(detector_weights))
    frames: list[np.ndarray] = []
    frame_index = 0
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break

            # Update tracking on every decoded frame; only save selected frames.
            detections = detector(image)
            vectors = detector.get_vectors(detections)
            if frame_index % frame_step == 0:
                frames.append(normalize_frame_vectors(vectors, image.shape[1], image.shape[0]))
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"video contained no decodable frames: {video_path}")

    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(frame) for frame in frames])
    features = np.concatenate(frames, axis=0) if offsets[-1] else np.empty((0, FEATURE_COUNT), dtype=np.float32)
    metadata = {
        "source_video": video_path.name,
        "source_fps": source_fps,
        "processed_fps": source_fps / frame_step if source_fps > 0 else None,
        "frame_step": frame_step,
        "decoded_frame_count": frame_index,
        "saved_frame_count": len(frames),
        "width": width,
        "height": height,
        "feature_columns": [
            "track_id", "person", "machine", "vehicle", "cone", "x", "y",
            "vx", "vy", "hardhat", "mask", "vest", "na",
        ],
        "coordinate_units": "image fractions; velocity is image fraction per saved frame",
    }
    output_path = output_dir / f"{video_path.stem}.npz"
    np.savez_compressed(
        output_path,
        features=features,
        offsets=offsets,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return output_path


def prepare_videos(
    video_root: str | Path,
    detector_weights: str | Path,
    output_dir: str | Path,
    frame_step: int = 1,
) -> list[Path]:
    """Prepare all supported videos recursively, one independent tracker per clip."""
    video_root = Path(video_root)
    paths = sorted(
        path for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in DEFAULT_VIDEO_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"no supported videos found under {video_root}")
    return [prepare_video(path, detector_weights, output_dir, frame_step) for path in paths]


class Dataset(TorchDataset):
    """Sliding STGNN examples from cached clips, split by video to prevent leakage

    Each item is ``(history, target)`` where history has shape ``(T, N, 13)``
    and target has shape ``(N, 4)``. Since the entity count may vary, use a
    batch size of one (or provide a custom collate function) when training.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        window_size: int = 8,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        seed: int = 42,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0 < train_fraction < 1 or not 0 <= validation_fraction < 1:
            raise ValueError("split fractions must be between 0 and 1")
        if train_fraction + validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be less than 1")

        self.cache_dir = Path(cache_dir)
        self.split = split
        self.window_size = window_size
        files = sorted(self.cache_dir.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no prepared .npz clips found in {self.cache_dir}")

        # Deterministically allocate complete videos to splits, never individual frames.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(files))
        shuffled = [files[index] for index in order]
        if len(files) == 1:
            counts = (1, 0, 0)
        elif len(files) == 2:
            counts = (1, 0, 1)
        else:
            train_count = max(1, int(len(files) * train_fraction))
            validation_count = max(1, int(len(files) * validation_fraction))
            # Reserve at least one whole clip for test as well as validation.
            while train_count + validation_count > len(files) - 1:
                train_count -= 1
            counts = (
                train_count,
                validation_count,
                len(files) - train_count - validation_count,
            )
        train_end = counts[0]
        validation_end = train_end + counts[1]
        assignments = {
            "train": shuffled[:train_end],
            "validation": shuffled[train_end:validation_end],
            "test": shuffled[validation_end:],
        }

        self._videos: dict[Path, list[np.ndarray]] = {}
        self._windows: list[tuple[Path, int]] = []
        for path in assignments[split]:
            frames, _ = _load_video_file(path)
            self._videos[path] = frames
            for start in range(max(0, len(frames) - window_size)):
                # The target frame is also included for identity alignment.
                if align_window(frames[start : start + window_size + 1]) is not None:
                    self._windows.append((path, start))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, start = self._windows[index]
        aligned = align_window(
            self._videos[path][start : start + self.window_size + 1]
        )
        if aligned is None:  # Guard against accidental mutation after indexing.
            raise RuntimeError(f"window no longer has aligned entities: {path}")
        history = torch.from_numpy(aligned[:-1].copy())
        target = torch.from_numpy(aligned[-1, :, 5:9].copy())
        return history, target


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", required=True, help="folder containing extracted videos")
    parser.add_argument("--detector-weights", required=True, help="YOLO detector checkpoint")
    parser.add_argument("--output-dir", default="dataset_bundle/local_data/prepared")
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()
    outputs = prepare_videos(args.video_root, args.detector_weights, args.output_dir, args.frame_step)
    print(f"Prepared {len(outputs)} clips in {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    _main()
