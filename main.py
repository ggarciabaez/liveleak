import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
import numpy as np
import scipy.stats as stats

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Info Vector Schema & Constants
# ==========================================
# Features: [Type(4), Pos(2), Vel(2), PPE(4)] = 12
DIM_TYPE = 4
DIM_POS = 2
DIM_VEL = 2
DIM_PPE = 4
DIM_TOTAL = DIM_TYPE + DIM_POS + DIM_VEL + DIM_PPE

T_WINDOW = 5  # T past frames
N_NODES = 8  # Detections per frame


# ==========================================
# 2. Synthetic Data Generator
# ==========================================
def generate_synthetic_data(num_samples=200, T=T_WINDOW, N=N_NODES):
    """Generates varied normal kinematics (smooth movement)."""
    data = torch.zeros(num_samples, T + 1, N, DIM_TOTAL)

    for b in range(num_samples):
        # Assign Node Types: 0,1=Person, 2=Machinery, 3=Vehicle, 4=Cone
        data[b, :, 0:2, 0] = 1.0  # Person
        data[b, :, 2, 1] = 1.0  # Machinery
        data[b, :, 3, 2] = 1.0  # Vehicle
        data[b, :, 4, 3] = 1.0  # Cone

        # Assign PPE: Persons compliant, others NA
        data[b, :, 0:2, 8:11] = 1.0  # Hardhat, Mask, Vest
        data[b, :, 0:2, 11] = 0.0  # NA = 0
        data[b, :, 2:5, 8:11] = 0.0  # No PPE
        data[b, :, 2:5, 11] = 1.0  # NA = 1

        # Generate smooth kinematic trajectories
        for n in range(N):
            pos = torch.rand(2) * 10.0
            vel = torch.randn(2) * 0.2

            for t in range(T + 1):
                data[b, t, n, 4:6] = pos
                data[b, t, n, 6:8] = vel

                # Step forward smoothly for the next frame
                pos = pos + vel
                vel = vel + torch.randn(2) * 0.05
                vel = torch.clamp(vel, -1.0, 1.0)

    return data


# ==========================================
# 3. ST-GNN Architecture
# ==========================================
class SpatialGraphConv(nn.Module):
    """Simple spatial aggregation over threshold-distance graphs."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_self = nn.Linear(in_features, out_features)
        self.W_neigh = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        self_out = self.W_self(x)
        # Row-normalize adjacency
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / degree
        neigh_out = self.W_neigh(torch.bmm(adj_norm, x))
        return F.relu(self_out + neigh_out)


class ST_GNN(nn.Module):
    def __init__(self, in_dim=DIM_TOTAL, hidden_dim=32, out_dim=4):
        super().__init__()
        self.dist_threshold = 3.0
        self.spatial_conv = SpatialGraphConv(in_dim, hidden_dim)
        # Temporal aggregator
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # Forecasts next Pos(2) and Vel(2) = 4 dims
        self.forecaster = nn.Linear(hidden_dim, out_dim)

    def compute_adj(self, pos):
        """Dynamic graph construction per frame based on proximity."""
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = diff.norm(dim=-1)
        return (dist < self.dist_threshold).float()

    def forward(self, x):
        B, T, N, F = x.shape
        spatial_seq = []

        # 1. Spatial pass per frame
        for t in range(T):
            x_t = x[:, t, :, :]
            pos_t = x_t[:, :, 4:6]
            adj_t = self.compute_adj(pos_t)
            s_t = self.spatial_conv(x_t, adj_t)
            spatial_seq.append(s_t)

        spatial_seq = torch.stack(spatial_seq, dim=1)  # (B, T, N, H)

        # 2. Temporal pass per node
        gru_in = spatial_seq.transpose(1, 2).reshape(B * N, T, -1)
        gru_out, _ = self.gru(gru_in)
        last_hidden = gru_out[:, -1, :]

        # 3. Forecast step K=1
        pred = self.forecaster(last_hidden)
        return pred.reshape(B, N, -1)

    # ==========================================


# 4. Training, Calibration, & Inference
# ==========================================
print("Generating synthetic 'normal' training data...")
train_data = generate_synthetic_data(num_samples=250)
X_train = train_data[:, :T_WINDOW, :, :]  # Past T frames
Y_train = train_data[:, T_WINDOW, :, 4:8]  # Target next frame (Pos, Vel)

model = ST_GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print("Training ST-GNN on self-supervised objective...")
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = criterion(preds, Y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1}/100 | Loss: {loss.item():.4f}")

# --- Calibration ---
print("\nCalibrating Risk distribution on held-out normal data...")
model.eval()
val_data = generate_synthetic_data(num_samples=50)
print(val_data.shape)
with torch.no_grad():
    val_preds = model(val_data[:, :T_WINDOW, :, :])
    val_targets = val_data[:, T_WINDOW, :, 4:8]
    # Get MSE per node per sample
    node_mse = F.mse_loss(val_preds, val_targets, reduction='none').mean(dim=-1)

mu_error = node_mse.mean().item()
sigma_error = node_mse.std().item()
print(f"Normal MSE -> Mean: {mu_error:.4f}, Std: {sigma_error:.4f}")

# --- Anomaly Demo ---
print("\nSimulating ST-GNN Anomaly (Node 0 abruptly accelerates towards Machinery)...")
test_data = generate_synthetic_data(num_samples=1)
# Inject anomaly on the target frame for Node 0
test_data[0, T_WINDOW, 0, 6:8] = torch.tensor([8.0, -8.0])  # Massive unnatural velocity
test_data[0, T_WINDOW, 0, 4:6] += test_data[0, T_WINDOW, 0, 6:8]

with torch.no_grad():
    test_X = test_data[:, :T_WINDOW, :, :]
    test_Y = test_data[:, T_WINDOW, :, 4:8]
    pred_Y = model(test_X)
    test_mse = F.mse_loss(pred_Y, test_Y, reduction='none').mean(dim=-1).squeeze()
    print(test_Y, pred_Y, sep="\n")
    print(test_Y.shape, pred_Y.shape)
    print(test_mse)
    for i in range(N_NODES):
        mse = test_mse[i].item()
        # CDF mapping: How statistically unlikely is this error?
        z_score = (mse - mu_error) / sigma_error
        risk_prob = stats.norm.cdf(z_score)

        status = "CRITICAL RISK" if risk_prob > 0.95 else "Normal"
        label = "Person" if i < 2 else "Machinery/Veh/Cone"
        print(f"Node {i} ({label}) | MSE: {mse:.4f} | Risk Score: {risk_prob:.3f} | {status}")