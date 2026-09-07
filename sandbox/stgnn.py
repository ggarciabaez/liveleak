"""
Synthetic proof-of-concept for the ST-GNN fork of the workplace risk
detection pipeline.

Point of this script: show that a small spatio-temporal GNN can be trained
self-supervised -- forecast next-frame node state, trained ONLY on "normal"
synthetic sequences -- and then produce a visibly higher forecasting error
on a synthetic "anomalous" sequence it never saw in training. Same spirit
as an XOR test for a plain feedforward net, just for this architecture.

NOT RUN. torch isn't available in this environment, and this is a one-shot
per request -- expect to debug shapes/dtypes locally before it's useful.

No PyTorch Geometric here on purpose. The graph convolution is hand-rolled
with plain tensor ops so the mechanics are visible instead of hidden behind
a library API. Swap in torch_geometric later once this makes sense and you
want the convenience/performance.
"""

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Toy world: 4 fixed-identity nodes. Feature layout matches the handbook
#    schema:
#
#      [ type_onehot(4) | position(2) | velocity(2) | ppe_onehot(4) ]
#        person/machinery/vehicle/cone   x,y            vx,vy         hardhat/mask/vest/NA
#
#    Node 0 = Person, Node 1 = Vehicle, Node 2 = Machinery, Node 3 = Cone.
#    PPE only applies to the Person node; everyone else gets NA=1.
#    (Position is kept as an absolute node feature here on the assumption
#    of a single fixed safety camera -- see the handbook's note on
#    position. If the camera moves between sites, swap this for relative
#    displacement/edge-encoded distance instead.)
# ---------------------------------------------------------------------------

NUM_NODES = 4
FEATURE_DIM = 4 + 2 + 2 + 4  # = 12
SEQ_LEN = 12          # timesteps fed to the encoder
FORECAST_HORIZON = 1  # predict 1 step ahead

TYPE_ONEHOT = torch.tensor([
    [1, 0, 0, 0],  # person
    [0, 0, 1, 0],  # vehicle
    [0, 1, 0, 0],  # machinery
    [0, 0, 0, 1],  # cone
], dtype=torch.float32)

NA_PPE = torch.tensor([0, 0, 0, 1], dtype=torch.float32)  # hardhat, mask, vest, NA


def make_ppe(compliant: bool):
    """PPE one-hot for the person node only; NA is used for every other node."""
    if compliant:
        return torch.tensor([1, 1, 1, 0], dtype=torch.float32)
    return torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # non-compliant, still not NA


def build_sequence(person_positions, vehicle_positions, machinery_pos, cone_pos, compliant=True):
    """
    Turn raw per-node (x, y) trajectories into the full feature-vector sequence.
    All position lists are length SEQ_LEN + FORECAST_HORIZON (the extra step
    at the end is the forecasting label, not part of the encoder input).

    Returns:
        features:  (T, NUM_NODES, FEATURE_DIM)
        positions: (T, NUM_NODES, 2)
    """
    T = len(person_positions)
    positions = torch.zeros(T, NUM_NODES, 2)
    velocities = torch.zeros(T, NUM_NODES, 2)

    all_pos = [person_positions, vehicle_positions,
               [machinery_pos] * T, [cone_pos] * T]

    for n in range(NUM_NODES):
        for t in range(T):
            positions[t, n] = torch.tensor(all_pos[n][t], dtype=torch.float32)

    for t in range(1, T):
        velocities[t] = positions[t] - positions[t - 1]
    velocities[0] = velocities[1]  # pad the first step

    ppe = make_ppe(compliant)
    features = torch.zeros(T, NUM_NODES, FEATURE_DIM)
    for t in range(T):
        for n in range(NUM_NODES):
            ppe_vec = ppe if n == 0 else NA_PPE
            features[t, n] = torch.cat([TYPE_ONEHOT[n], positions[t, n], velocities[t, n], ppe_vec])

    return features, positions


# ---------------------------------------------------------------------------
# 2. Synthetic trajectory generators
# ---------------------------------------------------------------------------

def normal_sample():
    """
    'Normal' operation: vehicle cruises along a fixed lane (y ~ 8), machinery
    and cone are static reference points, person wanders slowly and stays
    clear of the vehicle's lane (y in [0, 4]). PPE compliance varies -- the
    dynamics don't, which is the point: this fork should learn motion
    patterns, not compliance state (the rule-based fork already owns that).
    """
    T = SEQ_LEN + FORECAST_HORIZON
    t_arr = np.arange(T)

    vehicle_positions = [(2.0 + 0.5 * t, 8.0 + np.random.normal(0, 0.05)) for t in t_arr]
    machinery_pos = (5.0, 2.0)
    cone_pos = (5.0, 5.0)

    px, py = 1.0, 1.0
    person_positions = []
    for _ in t_arr:
        px += np.random.normal(0, 0.15)
        py += np.random.normal(0, 0.15)
        px, py = np.clip(px, 0, 4), np.clip(py, 0, 4)
        person_positions.append((px, py))

    compliant = np.random.rand() > 0.3
    return build_sequence(person_positions, vehicle_positions, machinery_pos, cone_pos, compliant)


def anomalous_sample():
    """
    'Anomalous' precursor: partway through the window, the person switches
    from slow wandering to a sharp, sustained move toward the vehicle's
    lane -- a closing-distance pattern that never appeared in training.
    """
    T = SEQ_LEN + FORECAST_HORIZON
    t_arr = np.arange(T)

    vehicle_positions = [(2.0 + 0.5 * t, 8.0 + np.random.normal(0, 0.05)) for t in t_arr]
    machinery_pos = (5.0, 2.0)
    cone_pos = (5.0, 5.0)

    px, py = 1.0, 1.0
    person_positions = []
    for t in t_arr:
        if t > SEQ_LEN // 2:
            py += 0.9  # fast, sustained closing speed -- well outside normal wander size
        else:
            py += np.random.normal(0, 0.15)
        px += np.random.normal(0, 0.1)
        person_positions.append((px, py))

    return build_sequence(person_positions, vehicle_positions, machinery_pos, cone_pos, compliant=True)


# ---------------------------------------------------------------------------
# 3. Hand-rolled graph convolution (no torch_geometric)
#
#    H' = relu( normalize(A + I) @ H @ W )
#
#    A is built per-timestep from inverse-distance weights between all node
#    pairs. With only 4 nodes this is a small dense matrix rather than a
#    sparse k-NN structure, but the math is the same at any scale -- swap
#    this for a proper k-NN/threshold sparse adjacency once there are more
#    than a handful of tracked entities per frame.
# ---------------------------------------------------------------------------

def build_adjacency(positions_t):
    """positions_t: (NUM_NODES, 2) -> row-normalized adjacency (NUM_NODES, NUM_NODES)"""
    diff = positions_t.unsqueeze(0) - positions_t.unsqueeze(1)  # (N, N, 2)
    dist = torch.norm(diff, dim=-1) + 1e-3
    A = 1.0 / dist
    A.fill_diagonal_(0.0)
    A = A + torch.eye(NUM_NODES)  # self-loops
    D_inv = 1.0 / A.sum(dim=1, keepdim=True)
    return D_inv * A


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, H, A):
        # H: (N, in_dim), A: (N, N) -> (N, out_dim)
        return torch.relu(self.linear(A @ H))


# ---------------------------------------------------------------------------
# 4. ST-GNN: per-frame spatial encoder -> GRU over time -> forecast head
# ---------------------------------------------------------------------------

class STGNNForecaster(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=16):
        super().__init__()
        self.gc1 = GraphConv(feature_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.temporal = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, 4)  # predict next-step [x, y, vx, vy]

    def forward(self, features, positions):
        """
        features:  (T, N, feature_dim)
        positions: (T, N, 2)  -- rebuilds the adjacency each frame
        returns:   (N, 4) predicted next-step [x, y, vx, vy] per node
        """
        T, N, _ = features.shape
        spatial_embeds = []
        for t in range(T):
            A = build_adjacency(positions[t])
            h = self.gc1(features[t], A)
            h = self.gc2(h, A)
            spatial_embeds.append(h)
        spatial_seq = torch.stack(spatial_embeds, dim=0)   # (T, N, hidden)
        spatial_seq = spatial_seq.permute(1, 0, 2)         # (N, T, hidden) -- N as GRU batch dim
        _, h_n = self.temporal(spatial_seq)                # h_n: (1, N, hidden)
        return self.decoder(h_n.squeeze(0))                # (N, 4)


# ---------------------------------------------------------------------------
# 5. Training loop -- normal sequences only
# ---------------------------------------------------------------------------

def get_target(features, positions):
    """Ground truth for the forecast step: [x, y, vx, vy] at t = SEQ_LEN."""
    pos_target = positions[SEQ_LEN]           # (N, 2)
    vel_target = features[SEQ_LEN, :, 6:8]    # velocity slice of the feature layout
    return torch.cat([pos_target, vel_target], dim=-1)  # (N, 4)


def train(model, n_epochs=1000, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        features, positions = normal_sample()
        pred = model(features[:SEQ_LEN], positions[:SEQ_LEN])
        target = get_target(features, positions)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")
    return model


# ---------------------------------------------------------------------------
# 6. Calibration + inference: normal vs. anomalous error
#
#    This is the Part 2D "calibration" step from the handbook, done the
#    cheap way -- mean/std + sigmoid -- instead of a full EVT fit, since
#    this is a proof of concept, not the production calibration step.
# ---------------------------------------------------------------------------

def sample_error(model, sample_fn):
    features, positions = sample_fn()
    with torch.no_grad():
        pred = model(features[:SEQ_LEN], positions[:SEQ_LEN])
        target = get_target(features, positions)
        return torch.mean((pred - target) ** 2).item()


def calibrate_and_score(model, n_calib=100):
    normal_errors = [sample_error(model, normal_sample) for _ in range(n_calib)]
    mu, sigma = np.mean(normal_errors), np.std(normal_errors) + 1e-6

    def to_probability(err):
        z = (err - mu) / sigma
        return 1 / (1 + np.exp(-z))

    fresh_normal_err = sample_error(model, normal_sample)
    anomaly_err = sample_error(model, anomalous_sample)

    print(f"\ncalibration: mu={mu:.4f}  sigma={sigma:.4f}")
    print(f"fresh normal sample  -> error {fresh_normal_err:.4f}  risk_prob {to_probability(fresh_normal_err):.3f}")
    print(f"anomalous sample     -> error {anomaly_err:.4f}  risk_prob {to_probability(anomaly_err):.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    model = STGNNForecaster()
    train(model)
    calibrate_and_score(model)