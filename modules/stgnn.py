"""
This module currently provides a mock prediction interface for the fusion demo
A trained ST-GNN should eventually replace the mock prediction
"""

import numpy as np


class MockSTGNN:
    """
    Predicts one-step-ahead kinematics for each entity.

    Input feature vectors use the shared DetNG layout:
    [id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]

    The feature history has shape (T, N, 13). Predictions have shape (N, 4) and contain [px, py, vx, vy]
    """

    def __init__(self, noise_std=0.1, seed=None):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def __call__(self, feature_buffer):
        """
        Predict the next position and velocity from the latest frame

        :param feature_buffer: History of entity features with shape (T, N, 13)
        :return: Predicted [px, py, vx, vy] values with shape (N, 4)
        """
        feature_buffer = np.asarray(feature_buffer)

        if feature_buffer.ndim != 3 or feature_buffer.shape[-1] != 13:
            raise ValueError("feature_buffer must have shape (T, N, 13)")

        # Read position and velocity from the latest frame.
        last_state = feature_buffer[-1, :, 5:9]
        position = last_state[:, :2]
        velocity = last_state[:, 2:]

        # Add noise to the current velocity as a stand-in for model prediction.
        predicted_velocity = velocity + self.rng.normal(
            0.0,
            self.noise_std,
            size=velocity.shape,
        )

        # Advance position by one time step using the predicted velocity.
        predicted_position = position + predicted_velocity

        # Return the same [px, py, vx, vy] layout expected by the demo.
        return np.concatenate(
            [predicted_position, predicted_velocity],
            axis=-1,
        )

""" Legacy mock implementation for quick testing 
class MockSTGNN:
    def __init__(self):
        pass

    def __call__(self, feature_buffer):
        state = feature_buffer[-1, ..., 5:9]  # get the last frame's states
        nv = np.random.normal(state[..., 2:], 0.1)  # add noise to velocities
        return np.concatenate([state[..., :2] + nv, nv], axis=-1)  # return new states
"""