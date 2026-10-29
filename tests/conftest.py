"""Shared fixtures. Feature layout everywhere:
[id, person, machine, vehicle, cone, px, py, vx, vy, hh, mask, vest, na]
"""
import numpy as np
import pytest
import torch

from modules.detng import gen_valid_feature

PERSON, MACHINE, VEHICLE, CONE = 1, 2, 3, 4


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)


def make_vec(track_id, obj_type, px, py, vx=0.0, vy=0.0, ppe=0b101):
    """ppe bitmask: bit0=hardhat, bit1=mask, bit2=vest."""
    v = gen_valid_feature(obj_type, [px, py, vx, vy], ppe)
    v[0] = track_id
    return v


@pytest.fixture
def make_frame():
    """Build an (N, 13) frame from (id, type, px, py[, vx, vy, ppe]) tuples."""
    def _make(*entities):
        return np.stack([make_vec(*e) for e in entities])
    return _make


@pytest.fixture
def history(make_frame):
    """(T, N, 13) history: one person walking toward a stationary machine."""
    frames = []
    for t in range(5):
        frames.append(make_frame(
            (1, PERSON, 50.0 - 2 * t, 50.0, -2.0, 0.0, 0b111),
            (2, MACHINE, 30.0, 50.0, 0.0, 0.0),
        ))
    return np.stack(frames)