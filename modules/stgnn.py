"""
Spatial-temporal model for one-step entity motion forecasting

Input:  (T, N, 13) feature history
Output: (N, 4) predicted [px, py, vx, vy]

This defines the model architecture. It needs training on tracked sequences before its predictions are useful
"""


# import numpy as np
import torch
import torch.nn as nn

class STGNN(nn.Module):

    def __init__(self, graph_radius, hidden_dim=64):
        super().__init__()

        if graph_radius <= 0:
            raise ValueError("graph_radius must be positive")


        # Positions are used to decide which entities are connected
        # This value must use the same units as px and py in the input
        self.graph_radius = graph_radius

        # Each entity has 13 values, but its ID is for tracking, not learning
        # Encode the other 12 values into a learned hidden representation
        self.node_encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
        )

        # For every connected pair, create a message from:
        # receiver representation, sender representation, relative position
        self.message_network = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Combine each entity's own representation with messages from neighbors
        self.spatial_update = nn.GRUCell(hidden_dim, hidden_dim)


        # Read each entity's sequence of frame representations over time
        self.temporal_network = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=True,
        )

        # Predict a change from the latest observed [px, py, vx, vy]
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )


    def _encode_frame(self, frame):
        """Build one frame's spatial graph and encode its entities

        Feature layout: [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
        
        Exclude id from the learned node features """
        
        node_features = frame[:, 1:]
        node_states = self.node_encoder(node_features)

        # Create a pairwise position-difference matrix with shape (N, N, 2)
        positions = frame[:, 5:7]
        relative_positions = (
            positions[:, None, :] - positions[None, :, :]
        )
        
        # Distances and connections have shape (N, N)
        distances = torch.linalg.vector_norm(relative_positions, dim=-1)
        edges = distances <= self.graph_radius

        # An entity should not send a message to itself
        no_self_edges = ~torch.eye(
            edges.size(0), 
            dtype=torch.bool, 
            device=edges.device
        )
        edges = edges & no_self_edges
        edge_weights = edges.to(node_states.dtype)

        # Make all receiver/sender pairs so the message network can process them
        entity_count = frame.shape[0]
        receivers = node_states[:, None, :].expand(
            entity_count, entity_count, -1
        )
        senders = node_states[None, :, :].expand(
            entity_count, entity_count, -1
        )

        # Scale relative positions so their size is comparable across scenes
        scaled_relative_positions = relative_positions / self.graph_radius    
    
        messages = self.message_network(
            torch.cat(
                [receivers, senders, scaled_relative_positions],
                dim=-1,
            )
        )

        #Ignore pairs that are outside the graph radius, then average neighbors
        messages = messages * edge_weights.unsqueeze(-1)
        neighbor_count = edge_weights.sum(dim=1, keepdim=True).clamp_min(1)
        aggregated_messages = messages.sum(dim=1) / neighbor_count

        # Return one spartially informed representation per entity: shape (N, H)
        return self.spatial_update(aggregated_messages, node_states)

    def forward(self, feature_history):
        """Predict the next kinematics; used when training the model."""

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
        # Align or pad tracks by ID before passing the history in  
        track_ids = feature_history[:, :, 0]
        expected_ids = track_ids[0:1:, :].expand_as(track_ids)
        if not torch.equal(track_ids, expected_ids):
            raise ValueError("entities must be aligned by track ID across frames")

        # Build one spatial graph representation for each frame
        # Each item has shape (N, H); stacking gives (N, T, H)
        frame_embeddings = [
            self._encode_frame(frame)
            for frame in feature_history
        ]
        temporal_input = torch.stack(frame_embeddings, dim=1)


        # GRU processes each entity's history across the T frames
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

    
""" Legacy mock implementation for quick testing 
class MockSTGNN:
    def __init__(self):
        pass

    def __call__(self, feature_buffer):
        state = feature_buffer[-1, ..., 5:9]  # get the last frame's states
        nv = np.random.normal(state[..., 2:], 0.1)  # add noise to velocities
        return np.concatenate([state[..., :2] + nv, nv], axis=-1)  # return new states
"""