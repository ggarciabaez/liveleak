"""Spatial-temporal motion forecasting and anomaly-score calibration.

The model predicts the next [px, py, vx, vy] for each tracked entity. Once
the next frame is observed, forecast errors can be converted to 0–1 motion
anomaly scores using a calibrator fitted on held-out normal footage
"""

import numpy as np
import torch
import torch.nn as nn


class MotionAnomalyCalibrator:
    """Map forecast errors to empirical percentiles from normal footage"""

    def __init__(self):
        self.normal_errors = None

    def fit(self, normal_errors):
        """Fit on errors from normal sequences not used to train the model"""
        
        normal_errors = np.asarray(normal_errors, dtype=np.float64).reshape(-1)
        
        if normal_errors.size == 0:
            raise ValueError("normal_errors must contain at least one value")
        if not np.isfinite(normal_errors).all() or np.any(normal_errors < 0):
            raise ValueError("normal_errors must be finite and non-negative")

        self.normal_errors = np.sort(normal_errors)
        return self

    def score(self, forecast_errors):
        """Return empirical anomaly percentiles in [0, 1]"""

        if self.normal_errors is None:
            raise RuntimeError("calibrator must be fitted before scoring")

        forecast_errors = np.asarray(forecast_errors, dtype=np.float64)
        if not np.isfinite(forecast_errors).all() or np.any(forecast_errors < 0):
            raise ValueError("forecast_errors must be finite and non-negative")

        ranks = np.searchsorted(self.normal_errors, forecast_errors, side="right")
        return ranks / self.normal_errors.size


class STGNN(nn.Module):
    def __init__(self, graph_radius, hidden_dim=64):
        super().__init__()

        if graph_radius <= 0:
            raise ValueError("graph_radius must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        # Positions determine which entities are connected in each frame
        # graph_radius must use the same units as px and py in the input
        self.graph_radius = graph_radius

        # Exclude the ID and encode the other 12 features
        self.node_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
        )

        # Encode receiver, sender, and relative position for each connection
        self.message_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Combine each entity's representation with its neighbors' messages
        self.spatial_update = nn.GRUCell(hidden_dim, hidden_dim)

        # Read each entity's sequence of frame representations over time
        self.temporal_network = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Predict a change from the latest observed [px, py, vx, vy]
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _encode_frame(self, frame):
        """Build one frame's spatial graph and encode its entities"""

        # Feature layout: [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
        node_features = frame[:, 1:]
        node_states = self.node_encoder(node_features)

        positions = frame[:, 5:7]
        edges = torch.cdist(positions, positions) <= self.graph_radius
        edges.fill_diagonal_(False)
        receiver_indices, sender_indices = edges.nonzero(as_tuple=True)

        relative_positions = (
            positions[receiver_indices] - positions[sender_indices]
        ) / self.graph_radius
        message_features = torch.cat(
            [
                node_states[receiver_indices],
                node_states[sender_indices],
                relative_positions,
            ],
            dim=-1,
        )
        messages = self.message_network(message_features)

        aggregated_messages = torch.zeros_like(node_states)
        aggregated_messages.index_add_(0, receiver_indices, messages)
        neighbor_count = torch.bincount(
            receiver_indices, minlength=frame.shape[0]
        ).clamp_min(1)
        aggregated_messages = aggregated_messages / neighbor_count.unsqueeze(-1)

        return self.spatial_update(aggregated_messages, node_states)

    def forward(self, feature_history):
        """Predict the next kinematics as a Torch tensor"""

        # Accept either a NumPy array or a tensor and use the model's device
        model_device = next(self.parameters()).device
        feature_history = torch.as_tensor(
            feature_history,
            dtype=torch.float32,
            device=model_device,
        )

        if feature_history.ndim != 3 or feature_history.shape[-1] != 13:
            raise ValueError("feature_history must have shape (T, N, 13)")
        if feature_history.shape[0] == 0:
            raise ValueError("feature_history must contain at least one frame")

        entity_count = feature_history.shape[1]
        if entity_count == 0:
            return feature_history.new_empty((0, 4))

        # This implementation expects row i to refer to the same track in every frame 
        track_ids = feature_history[:, :, 0]
        expected_ids = track_ids[:1].expand_as(track_ids)
        if not torch.equal(track_ids, expected_ids):
            raise ValueError("entities must be aligned by track ID across frames")
        if torch.unique(track_ids[0]).numel() != entity_count:
            raise ValueError("track IDs must be unique within each frame")

        # Build one spatial graph representation for each frame
        # Each item has shape (N, H); stacking gives (N, T, H)
        frame_embeddings = [
            self._encode_frame(frame)
            for frame in feature_history
        ]
        temporal_input = torch.stack(frame_embeddings, dim=1)

        # Process each entity's sequence across the T frames
        temporal_output, _ = self.temporal_network(temporal_input)
        latest_embedding = temporal_output[:, -1, :]

        # Predict a change, then add it to the latest observed state
        latest_kinematics = feature_history[-1, :, 5:9]
        predicted_change = self.motion_head(latest_embedding)
        return latest_kinematics + predicted_change

    def predict_kinematics(self, feature_history):
        """Return predictions as a NumPy array for the demo loops"""
        
        was_training = self.training
        self.eval()

        try:
            with torch.inference_mode():
                prediction = self(feature_history)
            return prediction.cpu().numpy()
        finally:
            self.train(was_training)

    def score_forecast(self, feature_history, observed_next_frame, calibrator):
        """Score checked forecasts after the next frame has been observed

        `observed_next_frame` may have shape (N, 13), using the shared feature layout, or (N, 4), containing [px, py, vx, vy]. 
        Rows must correspond to the same entities and ordering as the input history
        The returned scene score is the maximum entity anomaly percentile
        """
        history = torch.as_tensor(feature_history).detach().cpu().numpy()
        observed = torch.as_tensor(observed_next_frame).detach().cpu().numpy()

        if history.ndim != 3 or history.shape[-1] != 13:
            raise ValueError("feature_history must have shape (T, N, 13)")
        if history.shape[0] == 0:
            raise ValueError("feature_history must contain at least one frame")

        entity_count = history.shape[1]
        if observed.shape == (entity_count, 13):
            if not np.array_equal(observed[:, 0], history[-1, :, 0]):
                raise ValueError("observed entities must match history track IDs and row order")
            observed_kinematics = observed[:, 5:9]
        elif observed.shape == (entity_count, 4):
            observed_kinematics = observed
        else:
            raise ValueError("observed_next_frame must have shape (N, 13) or (N, 4)")

        if entity_count == 0:
            return {
                "scene_score": None,
                "entity_scores": np.empty(0, dtype=np.float64),
                "forecast_errors": np.empty(0, dtype=np.float64),
                "entity_ids": history[-1, :, 0],
            }

        predicted_kinematics = self.predict_kinematics(history)
        forecast_errors = np.sqrt(
            np.mean(np.square(predicted_kinematics - observed_kinematics), axis=1)
        )
        entity_scores = calibrator.score(forecast_errors)

        return {
            "scene_score": float(np.max(entity_scores)),
            "entity_scores": entity_scores,
            "forecast_errors": forecast_errors,
            "entity_ids": history[-1, :, 0],
        }


# TODO: it'd be sick af to turn this into a decorator, hella niche usage
def fit_motion_anomaly_calibrator(
    model,
    normal_sequences,
    window_size=8,
    calibrator=None,
):
    """Fit a score calibrator from one-step errors on held-out normal clips."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    sequences = _prepare_feature_sequences(
        normal_sequences,
        window_size,
        "normal_sequences",
    )
    if calibrator is None:
        calibrator = MotionAnomalyCalibrator()

    model_device = next(model.parameters()).device
    was_training = model.training
    normal_errors = []
    model.eval()

    try:
        with torch.inference_mode():
            for sequence in sequences:
                for start in range(sequence.shape[0] - window_size):
                    feature_history = sequence[start : start + window_size].to(
                        model_device
                    )
                    observed = sequence[start + window_size, :, 5:9].to(
                        model_device
                    )
                    prediction = model(feature_history)
                    entity_errors = torch.sqrt(
                        torch.mean(torch.square(prediction - observed), dim=1)
                    )
                    normal_errors.append(entity_errors.cpu().numpy())
    finally:
        model.train(was_training)

    calibrator.fit(np.concatenate(normal_errors))
    return calibrator
