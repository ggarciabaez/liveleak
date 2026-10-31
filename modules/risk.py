"""Connect tracked, aligned feature windows to the risk estimators."""

from collections import deque

import numpy as np
import torch
from torch import load as tload

from modules.Detector import Detector
from modules.dataset import align_window, normalize_frame_vectors
from modules.detng import DetNG, get_rules
from modules.kfilter import KalmanFilter
from modules.stgnn import MotionAnomalyCalibrator, STGNN


class RiskDetector:
    def __init__(
        self,
        yolo_path,
        stgnn_path,
        winsize=5,
        std_detng=0.1,
        std_stgnn=0.1,
        graph_rad=0.25,
        ruleset=None,
        calibrator=None,
        calibrator_path=None,
    ):
        if winsize < 1:
            raise ValueError("winsize must be positive")
        if calibrator is not None and calibrator_path is not None:
            raise ValueError("provide either calibrator or calibrator_path, not both")

        self.detector = Detector(yolo_path)
        self.detng = DetNG().compile(get_rules() if ruleset is None else ruleset)

        checkpoint = tload(stgnn_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("STGNN checkpoint must contain a state dictionary")
        model_config = checkpoint.get("model_config", {})
        self.model = STGNN(
            model_config.get("graph_radius", graph_rad),
            hidden_dim=model_config.get("hidden_dim", 64),
        )
        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("model", checkpoint)
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.kf = KalmanFilter()
        self.results = {}
        self.winsize = winsize
        # Keep one extra frame so each current observation can score the
        # forecast made from the preceding winsize frames.
        self.model_frames = deque(maxlen=winsize + 1)
        if calibrator_path is not None:
            calibrator = MotionAnomalyCalibrator().load(calibrator_path)
        self.calibrator = calibrator
        self.sigdet = std_detng
        self.sigstg = std_stgnn

    @staticmethod
    def _combine_vectors(vector_groups):
        nonempty = [np.asarray(group) for group in vector_groups if len(group)]
        if not nonempty:
            return np.empty((0, 13), dtype=np.float32)
        return np.concatenate(nonempty, axis=0).astype(np.float32, copy=False)

    def _append_model_frame(self, image, vector_groups):
        previous_frame = self.model_frames[-1] if self.model_frames else None
        model_frame = normalize_frame_vectors(
            vector_groups,
            image.shape[1],
            image.shape[0],
            previous_frame,
        )
        self.model_frames.append(model_frame)
        return model_frame

    def preload(self, frames):
        """Warm tracker and feature history with frames preceding live scoring."""
        for frame in frames:
            if frame is None:
                continue
            detections = self.detector(frame)
            vector_groups = self.detector.get_vectors(detections)
            self._append_model_frame(frame, vector_groups)

    def __call__(self, frame, draw=False):
        detections = self.detector(frame)
        vector_groups = self.detector.get_vectors(detections)
        current_vectors = self._combine_vectors(vector_groups)

        # Rules consume raw detections; Detector has already assigned
        # persistent IDs to movers and spatially matched the cone detections.
        rule_result = self.detng.execute(current_vectors[np.newaxis, ...])
        if len(rule_result) == 2:
            rule_score, violations = rule_result
            rule_outputs = np.empty(0, dtype=np.float32)
        else:
            rule_score, violations, rule_outputs = rule_result
        self._append_model_frame(frame, vector_groups)

        model_score = None
        forecast_errors = None
        unavailable_reason = "insufficient_history"
        if len(self.model_frames) == self.winsize + 1:
            aligned = align_window(list(self.model_frames))
            if aligned is None:
                unavailable_reason = "no_track_ids_persist_across_window"
            else:
                # Cones have stable IDs for data alignment, but are not motion
                # targets and must not hide a broken moving-object ID.
                moving_entities = aligned[-1, :, 4] <= 0.5
                if not moving_entities.any():
                    unavailable_reason = "no_tracked_moving_entities"
                else:
                    aligned = aligned[:, moving_entities]
                    feature_history = torch.from_numpy(aligned[:-1].copy())
                    observed_next_frame = aligned[-1].copy()
                    with torch.inference_mode():
                        prediction = self.model(feature_history)
                        observed_kinematics = torch.as_tensor(
                            observed_next_frame[:, 5:9], dtype=torch.float32
                        )
                        forecast_errors = torch.sqrt(
                            torch.mean(
                                torch.square(prediction.cpu() - observed_kinematics),
                                dim=1,
                            )
                        ).cpu().numpy()

                    if (
                        self.calibrator is not None
                        and getattr(self.calibrator, "normal_errors", None) is not None
                    ):
                        calibrated = self.calibrator.score(forecast_errors)
                        model_score = float(np.max(calibrated))
                        unavailable_reason = None
                    else:
                        unavailable_reason = "motion_calibrator_not_fitted"

        fused_score = None
        covariance = gain = None
        if model_score is not None:
            fused_score, covariance, gain = self.kf.fuse(
                model_score, rule_score, self.sigstg, self.sigdet
            )

        if draw:
            frame = self.detector.draw(frame, vector_groups)

        self.results = {
            "scores": (model_score, rule_score),
            "fused_score": fused_score,
            "model_available": model_score is not None,
            "model_unavailable_reason": unavailable_reason,
            "forecast_errors": forecast_errors,
            "kf": (covariance, gain) if model_score is not None else None,
            "detng": (violations, rule_outputs),
            "detect": (detections, vector_groups),
        }
        return fused_score, frame


class RecursiveRiskDetector:  # TODO
    def __init__(self):
        pass
