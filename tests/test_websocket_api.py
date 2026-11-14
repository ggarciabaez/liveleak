"""Real WebSocket transport tests with deterministic inference substitutes."""

import asyncio
import base64
import pickle

import cv2
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from app import app as backend


class FakeDetector:
    def __init__(self):
        self.results = {}
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        self.results = {
            "scores": (None, 0.25),
            "model_unavailable_reason": "insufficient_history",
        }
        return None, frame


def image_bytes():
    ok, encoded = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


async def exchange(messages):
    async with serve(backend.handle_connection, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}", open_timeout=5) as client:
            replies = []
            for message in messages:
                await client.send(message)
                raw = await asyncio.wait_for(client.recv(), timeout=5)
                # Only unpickle the local server created by this test.
                replies.append(pickle.loads(raw))
            return replies


@pytest.mark.parametrize("as_base64", [False, True])
def test_valid_frame_api(monkeypatch, as_base64):
    detector = FakeDetector()
    monkeypatch.setattr(backend, "make_detector", lambda: detector)
    payload = image_bytes()
    if as_base64:
        payload = base64.b64encode(payload).decode("ascii")
    reply, = asyncio.run(exchange([payload]))
    assert reply == {
        "model_score": None,
        "rule_score": 0.25,
        "fused_score": None,
        "model_unavailable_reason": "insufficient_history",
    }
    assert detector.calls == 1


@pytest.mark.parametrize("invalid", [b"not-an-image", "%%%", b""])
def test_invalid_input_then_valid_frame_api(monkeypatch, invalid):
    detector = FakeDetector()
    monkeypatch.setattr(backend, "make_detector", lambda: detector)
    error, good, continued = asyncio.run(exchange([invalid, image_bytes(), image_bytes()]))
    assert "JPEG or PNG" in error["error"]
    assert "error" not in good
    assert good["rule_score"] == 0.25
    assert continued == good
    assert detector.calls == 2
