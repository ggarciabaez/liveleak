import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from collections import deque

# ==========================================
# 1. Architecture: PyG & Stateful Inference
# ==========================================
DIM_IN = 12  # [Type(4), Pos(2), Vel(2), PPE(4)]
DIM_HIDDEN = 32
DIM_OUT = 4  # Forecasted Pos(2) + Vel(2)
T_WINDOW = 5  # Historical frames needed to forecast


class SpatialEncoder(nn.Module):
    """Processes a single frame's sparse graph using Graph Attention."""

    def __init__(self):
        super().__init__()
        # GATConv dynamically handles any number of nodes (N) and edges
        self.gat = GATConv(DIM_IN, DIM_HIDDEN, heads=1, concat=False)

    def forward(self, x, edge_index):
        return F.relu(self.gat(x, edge_index))


class TemporalForecaster(nn.Module):
    """Forecasts kinematics from a tracked sequence of spatial embeddings."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(DIM_HIDDEN, DIM_HIDDEN, batch_first=True)
        self.head = nn.Linear(DIM_HIDDEN, DIM_OUT)

    def forward(self, track_sequences):
        # track_sequences shape: (Num_Valid_Tracks, T_WINDOW, DIM_HIDDEN)
        out, _ = self.gru(track_sequences)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class TrackStateBuffer:
    """Maintains the temporal sequence for each active Tracker ID."""

    def __init__(self, max_len=T_WINDOW):
        self.max_len = max_len
        self.buffer = {}  # track_id -> deque(spatial_embeddings)

    def update_and_fetch(self, active_ids, spatial_embeddings):
        """
        Appends new embeddings to active tracks.
        Returns sequences only for tracks that have a full T_WINDOW of history.
        """
        ready_sequences = []
        ready_ids = []
        ready_indices = []  # Keeps track of which current-frame nodes are ready

        for idx, track_id in enumerate(active_ids):
            if track_id not in self.buffer:
                self.buffer[track_id] = deque(maxlen=self.max_len)

            # Append the detached embedding (no backprop through time across frames in inference)
            self.buffer[track_id].append(spatial_embeddings[idx].detach())

            # Only yield if the track has survived for the full window
            if len(self.buffer[track_id]) == self.max_len:
                seq = torch.stack(list(self.buffer[track_id]))
                ready_sequences.append(seq)
                ready_ids.append(track_id)
                ready_indices.append(idx)

        if ready_sequences:
            return torch.stack(ready_sequences), ready_ids, ready_indices
        return None, [], []


# ==========================================
# 2. Graph Utility: Sparse Edge Index
# ==========================================
def build_edge_index(pos, threshold=3.0):
    """Creates a sparse edge list [2, Num_Edges] for PyTorch Geometric."""
    # Compute pairwise distances
    dist = torch.cdist(pos, pos)
    # Find all pairs within the distance threshold
    edges = (dist < threshold).nonzero(as_tuple=False).t()
    return edges


# ==========================================
# 3. Live Streaming Simulation
# ==========================================
print("Initializing Streaming ST-GNN Pipeline...\n")
encoder = SpatialEncoder()
forecaster = TemporalForecaster()
state_buffer = TrackStateBuffer()


# Simulate a video feed where the number of detections (N) changes
def simulate_frame(frame_idx):
    if frame_idx < 3:
        N = 3
        active_ids = [101, 102, 103]
    elif frame_idx < 7:
        N = 5
        active_ids = [101, 102, 103, 104, 105]  # Two new workers enter
    else:
        N = 4
        active_ids = [101, 103, 104, 105]  # Worker 102 leaves the frame

    # Generate dummy features and positions
    x = torch.rand(N, DIM_IN)
    pos = torch.rand(N, 2) * 10.0
    x[:, 4:6] = pos
    return N, active_ids, x, pos


# Run the inference loop for 10 frames
for t in range(10):
    N_nodes, active_ids, x_t, pos_t = simulate_frame(t)

    # 1. Build dynamic sparse graph for current frame
    edge_index = build_edge_index(pos_t, threshold=4.0)

    # 2. Encode spatial relationships (Handles variable N natively)
    spatial_embeds = encoder(x_t, edge_index)

    # 3. Update state buffer and fetch tracks with enough history
    sequences, ready_ids, valid_indices = state_buffer.update_and_fetch(active_ids, spatial_embeds)

    print(f"Frame {t} | Detections: {N_nodes} | Active IDs: {active_ids}")

    # 4. Forecast risk if we have mature tracks
    if sequences is not None:
        preds = forecaster(sequences)
        print(f"   -> Forecasted {len(ready_ids)} mature tracks: {ready_ids}")
    else:
        print("   -> Buffering history... waiting for mature tracks.")