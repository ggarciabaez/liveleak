"""Spatial-temporal motion forecasting and anomaly-score calibration.

The model predicts the next [px, py, vx, vy] for each tracked entity. Once
the next frame is observed, forecast errors can be converted to 0–1 motion
anomaly scores using a calibrator fitted on held-out normal footage.
"""

import numpy as np
import torch
import torch.nn as nn
print("Working on the right one ffs")

class MotionAnomalyCalibrator:
    """Maps forecast RMSE values to empirical percentiles [0, 1] from normal footage."""

    def __init__(self):
        self.normal_errors = None

    def fit(self, normal_errors):
        normal_errors = np.asarray(normal_errors, dtype=np.float64).reshape(-1)
        if normal_errors.size == 0:
            raise ValueError("normal_errors must contain at least one value")
        if not np.isfinite(normal_errors).all() or np.any(normal_errors < 0):
            raise ValueError("normal_errors must be finite and non-negative")

        self.normal_errors = np.sort(normal_errors)
        return self

    def score(self, forecast_errors):
        if self.normal_errors is None:
            raise RuntimeError("Calibrator must be fitted or loaded before scoring")

        forecast_errors = np.asarray(forecast_errors, dtype=np.float64)
        if forecast_errors.size == 0:
            return 0.0
        if not np.isfinite(forecast_errors).all() or np.any(forecast_errors < 0):
            raise ValueError("forecast_errors must be finite and non-negative")

        ranks = np.searchsorted(self.normal_errors, forecast_errors, side="right")
        scores = ranks / self.normal_errors.size
        return float(np.max(scores)) if scores.ndim > 0 else float(scores)

    def load(self, calib_file_load):
        if calib_file_load is None:
            return
        calib = np.load(calib_file_load)
        self.normal_errors = calib["errors"]


class STGNN(nn.Module):
    # Feature indices for readability
    FEAT_SLICE = slice(1, 13)   # 12 features (excluding track ID at col 0)
    POS_SLICE = slice(5, 7)     # [px, py]
    KIN_SLICE = slice(5, 9)     # [px, py, vx, vy]
    CONE_COL = 4
    def __init__(self, graph_radius: float, hidden_dim: int = 64):
        super().__init__()
        if graph_radius <= 0 or hidden_dim <= 0:
            raise ValueError("graph_radius and hidden_dim must be positive")

        self.graph_radius = graph_radius
        self.hidden_dim = hidden_dim

        # 1. Node Encoder: 12 entity features -> hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
        )

        # 2. Edge Message Network: [h_receiver, h_sender, dx, dy] -> hidden_dim
        self.message_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 3. Spatial & Temporal Recurrence
        self.spatial_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.temporal_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # 4. Kinematics Residual Head: hidden_dim -> [d_px, d_py, d_vx, d_vy]
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def _encode_frame(self, frame: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Builds one frame's spatial graph only among active (`frame_mask == True`) entities.
        frame: (N, 13), frame_mask: (N,)
        Returns: (N, hidden_dim)
        """
        N = frame.shape[0]
        device = frame.device
        out_states = torch.zeros((N, self.hidden_dim), dtype=frame.dtype, device=device)

        # Isolate only entities present in this frame
        valid_idx = frame_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            return out_states

        valid_frame = frame[valid_idx]
        node_states = self.node_encoder(valid_frame[:, self.FEAT_SLICE])

        # If only 1 entity is present, no edges exist; still run spatial_update with zero messages
        if valid_idx.numel() == 1:
            zero_msg = torch.zeros_like(node_states)
            out_states[valid_idx] = self.spatial_update(zero_msg, node_states)
            return out_states

        # Find directed edges within graph_radius
        positions = valid_frame[:, self.POS_SLICE]
        edges = torch.cdist(positions, positions) <= self.graph_radius
        edges.fill_diagonal_(False)
        receivers, senders = edges.nonzero(as_tuple=True)

        aggregated = torch.zeros_like(node_states)
        if receivers.numel() > 0:
            rel_pos = (positions[receivers] - positions[senders]) / self.graph_radius
            msg_input = torch.cat([node_states[receivers], node_states[senders], rel_pos], dim=-1)
            messages = self.message_network(msg_input)

            aggregated.index_add_(0, receivers, messages)
            counts = torch.bincount(receivers, minlength=valid_frame.shape[0]).clamp_min(1)
            aggregated = aggregated / counts.unsqueeze(-1)

        out_states[valid_idx] = self.spatial_update(aggregated, node_states)
        return out_states

    def _prepare(self, features, mask):
        """Coerce and validate (T, N, 13) features and the (T, N) presence mask.

        Returns (feats, valid, active). `valid` is presence (mask, or inferred from
        padding id -1); `active` is valid AND not a cone, i.e. the entities that
        take part in the graph and the temporal state.
        """
        device = next(self.parameters()).device
        feats = torch.as_tensor(features, dtype=torch.float32, device=device).contiguous()

        if feats.ndim != 3 or feats.shape[-1] != 13:
            raise ValueError(f"Expected features of shape (T, N, 13), got {tuple(feats.shape)}")

        T, N, _ = feats.shape
        if T == 0:
            raise ValueError("features must contain at least one frame")

        if mask is None:
            # Infer presence from padding (padded slots have ID == -1)
            valid = feats[:, :, 0] != -1
        else:
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device).contiguous()
            if valid.shape != (T, N):
                raise ValueError(f"mask must have shape (T, N) = {(T, N)}, got {tuple(valid.shape)}")
        # A slot must hold one track ID in every frame where it is present,
        # and no two slots may share an ID
        if N > 0:
            ids = feats[:, :, 0]
            hi = ids.masked_fill(~valid, float("-inf")).amax(dim=0)
            lo = ids.masked_fill(~valid, float("inf")).amin(dim=0)
            seen = valid.any(dim=0)
            if not torch.equal(hi[seen], lo[seen]):
                raise ValueError("entities must be aligned by track ID across frames")
            if torch.unique(hi[seen]).numel() != int(seen.sum()):
                raise ValueError("track IDs must be unique across slots")

        # Only moving entities participate; cones are static context for DetNG
        active = valid & ~(feats[:, :, self.CONE_COL] > 0.5)
        return feats, valid, active


    def forward(self, features, mask=None):
        """
        Given a sequence of frames (T, N, 13) and presence mask (T, N),
        predicts the next step's kinematics (N, 4) from the final frame T-1.
        Cones are ignored. Rows of slots that are absent at T-1 (or are cones)
        are meaningless: select with `mask[-1]` and the type column.
        """
        feats, _, active = self._prepare(features, mask)
        T, N, _ = feats.shape
        if N == 0:
            return feats.new_empty((0, 4))

        # Step through time: only update an entity's temporal GRU state on frames where it is visible
        h_temporal = torch.zeros((N, self.hidden_dim), dtype=feats.dtype, device=feats.device)
        for t in range(T):
            spatial_emb = self._encode_frame(feats[t], active[t])
            present = active[t]
            if present.any():
                h_temporal[present] = self.temporal_cell(spatial_emb[present], h_temporal[present])

        latest_kinematics = feats[-1, :, self.KIN_SLICE]
        delta_kinematics = self.motion_head(h_temporal)
        return latest_kinematics + delta_kinematics

    def compute_window_error(self, features, mask) -> torch.Tensor:
        """
        Uses frames [0 ... T-2] to predict kinematics at frame T-1, and returns
        the per-entity RMSE for entities visible in both T-2 and T-1.
        """
        feats, _, active = self._prepare(features, mask)
        T, N, _ = feats.shape
        empty = feats.new_empty((0,))
        if T < 2 or N == 0:
            return empty

        # Only evaluate entities present in the last history frame (T-2) AND target frame (T-1)
        eval_entities = active[-2] & active[-1]
        if not eval_entities.any():
            return empty

        pred_kin = self.forward(feats[:-1], active[:-1])
        obs_kin = feats[-1, :, self.KIN_SLICE]

        return torch.sqrt(torch.mean((pred_kin[eval_entities] - obs_kin[eval_entities]) ** 2, dim=-1))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        if path is None:
            return
        self.load_state_dict(torch.load(path, weights_only=True))


def _prepare_feature_sequences(feature_sequences, window_size, name):
    if isinstance(feature_sequences, (np.ndarray, torch.Tensor)):
        if feature_sequences.ndim == 3:
            sequences = [feature_sequences]
        elif feature_sequences.ndim == 4:
            sequences = list(feature_sequences)
        else:
            raise ValueError(f"{name} must contain sequences shaped (F, N, 13)")
    else:
        sequences = list(feature_sequences)

    if not sequences:
        raise ValueError(f"{name} must contain at least one sequence")

    prepared_sequences = []
    for sequence_index, sequence in enumerate(sequences):
        sequence = torch.as_tensor(sequence, dtype=torch.float32).detach().cpu()
        if sequence.ndim != 3 or sequence.shape[-1] != 13:
            raise ValueError(f"{name}[{sequence_index}] must have shape (F, N, 13)")
        if sequence.shape[0] <= window_size:
            raise ValueError(
                f"{name}[{sequence_index}] needs more frames than window_size"
            )
        if sequence.shape[1] == 0:
            raise ValueError(f"{name}[{sequence_index}] must contain entities")
        if not torch.isfinite(sequence).all():
            raise ValueError(f"{name}[{sequence_index}] must contain finite values")

        track_ids = sequence[:, :, 0]
        if not torch.equal(track_ids, track_ids[0:1, :].expand_as(track_ids)):
            raise ValueError(
                f"{name}[{sequence_index}] must keep each track in the same row"
            )
        if any(torch.unique(frame_ids).numel() != sequence.shape[1] for frame_ids in track_ids):
            raise ValueError(
                f"{name}[{sequence_index}] needs unique track IDs in every frame"
            )
        prepared_sequences.append(sequence)

    return prepared_sequences


# TODO: it'd be sick af to turn this into a decorator, hella niche usage
def fit_motion_anomaly_calibrator(
    model,
    normal_sequences,
    window_size=8,
    calibrator=None,
    calib_file_save="./calib.npz"
):
    """Fit a score calibrator from one-step errors on held-out normal clips."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    sequences = _prepare_feature_sequences(
        normal_sequences,
        window_size,
        "normal_sequences",
    )


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

    if calibrator is None:
        calibrator = MotionAnomalyCalibrator()

    calibrator = calibrator.fit(np.concatenate(normal_errors))
    if calib_file_save is not None:
        np.savez(calib_file_save, errors=calibrator.normal_errors)

    return calibrator

if __name__ == '__main__':
    calib = MotionAnomalyCalibrator()
