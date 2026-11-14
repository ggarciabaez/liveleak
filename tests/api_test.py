"""API integration tests for the WebSocket backend (backend/backend.py).

A real WebSocket server is started on a free local port and a real client talks
to it, so framing, base64/binary handling, validation and the reply format are
exercised end to end. Only the model is replaced by a deterministic stub, so no
GPU or weights are needed (CI-friendly).
"""
import asyncio
import base64
import pickle

import cv2
import numpy as np
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from backend import backend


class StubDetector:
    """Stands in for RiskDetector; `empty=True` mimics a frame with no tracked entities."""

    def __init__(self, empty=False):
        self.results = {}
        self.empty = empty
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        if self.empty:
            return None, None
        self.results["scores"] = (0.30, 0.50, 0.40)  # (model, rule, fused-input) layout used by infer()
        self.results["model_unavailable_reason"] = None
        self.results["kf"] = (0.42, 0.05, 0.5)
        self.results["detng"] = (0.5, [], [0.5])
        self.results["detect"] = None
        return 0.42, None


def jpeg_bytes():
    frame = np.random.default_rng(0).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def run_session(monkeypatch, messages, empty=False):
    """Send `messages` over one connection to a live server; return the unpickled replies."""
    detector = StubDetector(empty=empty)
    monkeypatch.setattr(backend, "make_detector", lambda: detector)

    async def go():
        async with serve(backend.handle_connection, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            replies = []
            async with connect(f"ws://127.0.0.1:{port}") as ws:
                for message in messages:
                    await ws.send(message)
                    replies.append(pickle.loads(await asyncio.wait_for(ws.recv(), timeout=20)))
            return replies

    return asyncio.run(go()), detector


def test_valid_frame_returns_scores(monkeypatch):
    """Happy path: a valid JPEG yields model, rule and fused scores."""
    (reply,), detector = run_session(monkeypatch, [jpeg_bytes()])
    assert "error" not in reply
    assert reply["fused_score"] == pytest.approx(0.42)
    assert reply["rule_score"] == pytest.approx(0.50)
    assert reply["model_score"] == pytest.approx(0.40)
    assert reply["model_unavailable_reason"] is None
    assert detector.calls == 1


def test_base64_text_frame_is_accepted(monkeypatch):
    """The protocol also allows a base64 text message."""
    (reply,), _ = run_session(monkeypatch, [base64.b64encode(jpeg_bytes()).decode()])
    assert "error" not in reply
    assert reply["fused_score"] == pytest.approx(0.42)


def test_invalid_frame_returns_error_and_connection_survives(monkeypatch):
    """Error case: malformed bytes get a clear error, the model is not called, and the
    same connection still serves a valid frame afterwards."""
    bad, good = run_session(monkeypatch, [b"definitely not an image", jpeg_bytes()])[0]
    assert "JPEG or PNG" in bad["error"]
    assert "fused_score" not in bad
    assert "error" not in good
    assert good["fused_score"] == pytest.approx(0.42)


def test_invalid_base64_text_returns_error(monkeypatch):
    (reply,), detector = run_session(monkeypatch, ["!!! not base64 !!!"])
    assert "error" in reply
    assert detector.calls == 0


def test_empty_message_returns_error(monkeypatch):
    (reply,), _ = run_session(monkeypatch, [b""])
    assert "error" in reply


def test_frame_without_detections_reports_reason(monkeypatch):
    """A valid frame with no tracked entities returns null scores and `no_detections`."""
    (reply,), _ = run_session(monkeypatch, [jpeg_bytes()], empty=True)
    assert reply["fused_score"] is None
    assert reply["model_score"] is None
    assert reply["rule_score"] is None
    assert reply["model_unavailable_reason"] == "no_detections"


def test_missing_model_file_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_FILES", "ppe.pt,stgnn.pth,calib.npz")
    with pytest.raises(FileNotFoundError, match="Required model file is missing"):
        backend.make_detector()
