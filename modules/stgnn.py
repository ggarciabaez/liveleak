"""Spatial-temporal motion forecasting and anomaly-score calibration.

The model predicts the next [px, py, vx, vy] for each tracked entity. Once
the next frame is observed, forecast errors can be converted to 0–1 motion
anomaly scores using a calibrator fitted on held-out normal footage.
"""

import numpy as np
import torch
from torch import nn

# going against everything i stand for with this one

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
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)

    def _encode_frames_dense(self, frames: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Dense batched spatial graph encoding across K frames (where K = B or B * T).
        frames: (K, N, 13), frame_mask: (K, N)
        Returns: (K, N, hidden_dim)
        """
        K, N, _ = frames.shape
        if K == 0 or N == 0:
            return frames.new_zeros((K, N, self.hidden_dim))

        # Sanitize padded inputs so NaNs/Infs in inactive slots don't corrupt gradients via 0 * NaN
        safe_feats = torch.where(frame_mask.unsqueeze(-1), frames[:, :, self.FEAT_SLICE], 0.0)
        node_states = self.node_encoder(safe_feats)  # (K, N, H)

        if N == 1:
            zero_msg = torch.zeros_like(node_states)
            updated = self.spatial_update(
                zero_msg.reshape(K, self.hidden_dim),
                node_states.reshape(K, self.hidden_dim),
            ).reshape(K, 1, self.hidden_dim)
            return torch.where(frame_mask.unsqueeze(-1), updated, 0.0)

        # Build dense (K, N, N) adjacency matrix: adj[k, i, j] == True means edge j -> i
        positions = torch.where(frame_mask.unsqueeze(-1), frames[:, :, self.POS_SLICE], 0.0)
        dists = torch.cdist(positions, positions)  # (K, N, N)

        eye = torch.eye(N, dtype=torch.bool, device=frames.device).unsqueeze(0)
        valid_pairs = frame_mask.unsqueeze(2) & frame_mask.unsqueeze(1) & ~eye
        adj = (dists <= self.graph_radius) & valid_pairs  # (K, N, N)

        # Pairwise relative positions: receiver (i, dim 1) - sender (j, dim 2)
        rel_pos = (positions.unsqueeze(2) - positions.unsqueeze(1)) / self.graph_radius  # (K, N, N, 2)

        h_rec = node_states.unsqueeze(2).expand(K, N, N, self.hidden_dim)
        h_send = node_states.unsqueeze(1).expand(K, N, N, self.hidden_dim)
        msg_input = torch.cat([h_rec, h_send, rel_pos], dim=-1)  # (K, N, N, 2H + 2)

        # Compute messages everywhere, mask out non-edges, and mean-aggregate over senders (dim=2)
        messages = self.message_network(msg_input)  # (K, N, N, H)
        messages = messages * adj.unsqueeze(-1).to(messages.dtype)

        counts = adj.sum(dim=2, keepdim=True).clamp_min(1).to(messages.dtype)  # (K, N, 1)
        aggregated = messages.sum(dim=2) / counts  # (K, N, H)

        # Update node states via GRUCell and zero out inactive slots
        updated = self.spatial_update(
            aggregated.reshape(K * N, self.hidden_dim),
            node_states.reshape(K * N, self.hidden_dim),
        ).reshape(K, N, self.hidden_dim)

        return torch.where(frame_mask.unsqueeze(-1), updated, 0.0)

    def _prepare(self, features, mask, validate: bool = True):
        """Coerce and validate (T, N, 13) or (B, T, N, 13) features and presence mask.

        Returns (feats, valid, active, was_unbatched) where tensors have batch dim B.
        """
        device = next(self.parameters()).device
        feats = torch.as_tensor(features, dtype=torch.float32, device=device).contiguous()

        was_unbatched = feats.ndim == 3
        if was_unbatched:
            feats = feats.unsqueeze(0)
        elif feats.ndim != 4:
            raise ValueError(
                f"Expected features of shape (T, N, 13) or (B, T, N, 13), got {tuple(feats.shape)}"
            )

        if feats.shape[-1] != 13:
            raise ValueError(
                f"Expected last dimension of size 13, got {tuple(feats.shape)}"
            )

        B, T, N, _ = feats.shape
        if T == 0:
            raise ValueError("features must contain at least one frame")

        if mask is None:
            # Infer presence from padding (padded slots have ID == -1)
            valid = feats[:, :, :, 0] != -1
        else:
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device).contiguous()
            if was_unbatched and valid.ndim == 2:
                valid = valid.unsqueeze(0)
            if valid.shape != (B, T, N):
                expected = (T, N) if was_unbatched else (B, T, N)
                actual = tuple(valid.squeeze(0).shape) if was_unbatched and valid.shape[0] == 1 else tuple(valid.shape)
                raise ValueError(f"mask must have shape {expected}, got {actual}")

        if validate and N > 0 and B > 0:
            ids = feats[:, :, :, 0]
            hi = ids.masked_fill(~valid, float("-inf")).amax(dim=1)  # (B, N)
            lo = ids.masked_fill(~valid, float("inf")).amin(dim=1)   # (B, N)
            seen = valid.any(dim=1)                                  # (B, N)

            if not torch.equal(hi[seen], lo[seen]):
                raise ValueError("entities must be aligned by track ID across frames")

            # Vectorized uniqueness check across active slots per batch item: sort and check adjacent
            sorted_hi, _ = torch.where(seen, hi, float("inf")).sort(dim=-1)
            sorted_seen = torch.isfinite(sorted_hi)
            dup = (sorted_hi[:, 1:] == sorted_hi[:, :-1]) & sorted_seen[:, 1:] & sorted_seen[:, :-1]
            if dup.any():
                raise ValueError("track IDs must be unique across slots")

        # Only moving entities participate; cones are static context for DetNG
        active = valid & ~(feats[:, :, :, self.CONE_COL] > 0.5)
        return feats, valid, active, was_unbatched

    def forward(self, features, mask=None, validate: bool = True):
        """
        Given a sequence of frames (T, N, 13) or batch (B, T, N, 13) and optional
        presence mask (T, N) or (B, T, N), predicts the next step's kinematics
        (N, 4) or (B, N, 4) from the final frame T-1.
        """
        feats, _, active, was_unbatched = self._prepare(features, mask, validate=validate)
        B, T, N, _ = feats.shape
        if N == 0:
            out = feats.new_empty((B, 0, 4))
            return out.squeeze(0) if was_unbatched else out

        # 1. Encode all (B * T) frames simultaneously in one dense pass
        spatial_all = self._encode_frames_dense(
            feats.reshape(B * T, N, 13),
            active.reshape(B * T, N),
        ).reshape(B, T, N, self.hidden_dim)

        # 2. Step through T with a dense (B * N) GRU update, masking inactive slots
        h_temporal = feats.new_zeros((B * N, self.hidden_dim))
        for t in range(T):
            present_flat = active[:, t].reshape(B * N, 1)
            spatial_flat = spatial_all[:, t].reshape(B * N, self.hidden_dim)
            h_next = self.temporal_cell(spatial_flat, h_temporal)
            h_temporal = torch.where(present_flat, h_next, h_temporal)

        h_temporal = h_temporal.reshape(B, N, self.hidden_dim)
        latest_kinematics = feats[:, -1, :, self.KIN_SLICE]
        delta_kinematics = self.motion_head(h_temporal)
        pred = latest_kinematics + delta_kinematics

        return pred.squeeze(0) if was_unbatched else pred

    def compute_window_error(self, features, mask=None, return_mask: bool = False):
        """
        Uses frames [0 ... T-2] to predict kinematics at frame T-1.
        - If `return_mask=False` (default): returns a 1D tensor of valid entity RMSEs
          (flattened across B if batched), preserving backward compatibility.
        - If `return_mask=True`: returns `(rmse, eval_entities)` of shape `(B, N)`
          (or `(N,)` if unbatched) for per-sample loss reduction without synchronization.
        """
        feats, _, active, was_unbatched = self._prepare(features, mask)
        B, T, N, _ = feats.shape

        if T < 2 or N == 0:
            if return_mask:
                empty_err = feats.new_zeros((B, N))
                empty_mask = torch.zeros((B, N), dtype=torch.bool, device=feats.device)
                if was_unbatched:
                    return empty_err.squeeze(0), empty_mask.squeeze(0)
                return empty_err, empty_mask
            return feats.new_empty((0,))

        # Only evaluate entities present in the last history frame (T-2) AND target frame (T-1)
        eval_entities = active[:, -2] & active[:, -1]  # (B, N)

        if not return_mask and not eval_entities.any():
            return feats.new_empty((0,))

        # Skip redundant validation inside forward since _prepare already ran
        pred_kin = self.forward(feats[:, :-1], active[:, :-1], validate=False)
        obs_kin = feats[:, -1, :, self.KIN_SLICE]

        sq_err = (pred_kin - obs_kin) ** 2
        rmse = torch.sqrt(torch.mean(sq_err, dim=-1))  # (B, N)

        if return_mask:
            rmse = torch.where(eval_entities, rmse, 0.0)
            if was_unbatched:
                return rmse.squeeze(0), eval_entities.squeeze(0)
            return rmse, eval_entities

        return rmse[eval_entities]

    def predict_kinematics(self, feature_history):
        """Predict next-step [px, py, vx, vy] values as a NumPy array."""
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                prediction = self(feature_history)
            return prediction.detach().cpu().numpy()
        finally:
            self.train(was_training)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        if path is None:
            return
        self.load_state_dict(torch.load(path, weights_only=True))


def collate(batch):
    xs, vs = [], []
    for f, v in batch:                               # f: (W,N,13), v: (W,N)
        cone = ((f[..., 4] > 0.5) & v).any(0)        # slots that are cones
        xs.append(f[:, ~cone]); vs.append(v[:, ~cone])
    N = max(x.shape[1] for x in xs)
    X = torch.zeros(len(xs), xs[0].shape[0], N, 13); X[..., 0] = -1
    V = torch.zeros(len(xs), xs[0].shape[0], N, dtype=torch.bool)
    for i, (x, v) in enumerate(zip(xs, vs)):
        X[i, :, :x.shape[1]] = x; V[i, :, :x.shape[1]] = v
    return X, V

@torch.no_grad()
def fit_calibrator(model, loader, device, path="calib.npz"):
    model.eval(); worst = []
    for X, V in loader:                              # held-out *normal* videos only
        rmse, ev = model.compute_window_error(X.to(device), V.to(device), return_mask=True)
        has = ev.any(1)
        worst.append(rmse.amax(1)[has].cpu())        # rmse is 0 where not evaluated
    cal = MotionAnomalyCalibrator().fit(torch.cat(worst).numpy())
    np.savez(path, errors=cal.normal_errors)
    return cal


if __name__ == '__main__':
    calib = MotionAnomalyCalibrator()