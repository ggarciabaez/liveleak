"""Background QThread worker for OpenCV capture + WebSocket inference streaming."""

from __future__ import annotations

import asyncio
import pickle
import time
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


def normalize_unpacked_results(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Standardize the payload returned by backend.py, including the new `results` field
    containing RiskDetector.results (`scores`, `kf`, `detng`, `detect`).
    """
    if "error" in raw:
        return {"error": str(raw["error"])}

    results_dict = raw.get("results") or {}
    if not isinstance(results_dict, dict):
        results_dict = {}

    fused = raw.get("fused_score")
    model = raw.get("model_score")
    rule = raw.get("rule_score")
    reason = raw.get("model_unavailable_reason")

    # Extract Kalman Filter covariance P (fuse returns (x, P, K))
    covariance = 0.0
    kalman_gain = 0.0
    kf_data = results_dict.get("kf")
    if isinstance(kf_data, (tuple, list)) and len(kf_data) >= 2:
        try:
            covariance = float(kf_data[1])
            if len(kf_data) >= 3:
                kalman_gain = float(kf_data[2])
        except (TypeError, ValueError):
            covariance = 0.0
    elif isinstance(kf_data, dict):
        covariance = float(kf_data.get("P", kf_data.get("covariance", 0.0)) or 0.0)

    # Extract DetNG rule violations: DetNG.execute returns (total_risk, violations, scores)
    violations: list[str] = []
    rule_breakdown: list[float] = []
    detng_data = results_dict.get("detng")
    if isinstance(detng_data, (tuple, list)):
        if len(detng_data) == 2:
            raw_viols, raw_scores = detng_data[0], detng_data[1]
        elif len(detng_data) >= 3:
            raw_viols, raw_scores = detng_data[1], detng_data[2]
        else:
            raw_viols, raw_scores = (detng_data[0] if detng_data else []), []

        if isinstance(raw_viols, (list, tuple, np.ndarray)):
            violations = [str(v) for v in raw_viols]
        if raw_scores is not None:
            try:
                rule_breakdown = [float(x) for x in np.asarray(raw_scores, dtype=np.float64).ravel()]
            except Exception:
                pass
    elif isinstance(detng_data, dict):
        violations = [str(v) for v in detng_data.get("violations", [])]
        if "scores" in detng_data and detng_data["scores"] is not None:
            rule_breakdown = [float(x) for x in np.asarray(detng_data["scores"], dtype=np.float64).ravel()]

    # Extract 13-col feature vectors from `detect`
    # SSOT §4.2: get_vectors returns a list of 4 arrays [(n_k, 13)] or merged (N, 13)
    vectors = np.empty((0, 13), dtype=np.float32)
    detect_data = results_dict.get("detect")
    if detect_data is not None:
        try:
            if isinstance(detect_data, np.ndarray):
                if detect_data.ndim == 2 and detect_data.shape[1] == 13:
                    vectors = detect_data.astype(np.float32, copy=False)
                elif detect_data.ndim == 3 and detect_data.shape[2] == 13:
                    # If window tensor (T, N, 13) was stored, grab latest frame
                    last_frame = detect_data[-1]
                    vectors = last_frame[last_frame[:, 0] != -1]
            elif isinstance(detect_data, (list, tuple)):
                valid_parts = [
                    np.asarray(arr, dtype=np.float32)
                    for arr in detect_data
                    if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 13 and arr.shape[0] > 0
                ]
                if valid_parts:
                    vectors = np.vstack(valid_parts)
        except Exception:
            vectors = np.empty((0, 13), dtype=np.float32)

    # Count entities per one-hot columns 1..4 (person, machine, vehicle, cone)
    counts = {"person": 0, "machine": 0, "vehicle": 0, "cone": 0}
    if vectors.shape[0] > 0:
        counts["person"] = int(np.sum(vectors[:, 1] > 0.5))
        counts["machine"] = int(np.sum(vectors[:, 2] > 0.5))
        counts["vehicle"] = int(np.sum(vectors[:, 3] > 0.5))
        counts["cone"] = int(np.sum(vectors[:, 4] > 0.5))

    return {
        "fused_score": float(fused) if fused is not None else None,
        "model_score": float(model) if model is not None else None,
        "rule_score": float(rule) if rule is not None else None,
        "covariance": covariance,
        "kalman_gain": kalman_gain,
        "reason": str(reason) if reason else "ok",
        "violations": violations,
        "rule_breakdown": rule_breakdown,
        "vectors": vectors,
        "counts": counts,
    }


class StreamWorker(QThread):
    """Streams frames from cv2.VideoCapture to backend.py over WebSocket."""

    frame_ready = Signal(np.ndarray, dict, float)  # (bgr_frame, parsed_telemetry, rtt_ms)
    status_changed = Signal(str, str)  # (state_level, human_message)

    def __init__(
            self,
            uri: str,
            source: str | int,
            jpeg_quality: int = 85,
            loop_video: bool = True,
    ) -> None:
        super().__init__()
        self.uri = uri.strip()
        self.source: str | int = int(source) if str(source).strip().isdigit() else str(source).strip()
        self.jpeg_quality = int(np.clip(jpeg_quality, 10, 100))
        self.loop_video = loop_video
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            asyncio.run(self._stream_loop())
        except Exception as exc:
            self.status_changed.emit("error", f"Stream error: {exc}")

    async def _stream_loop(self) -> None:
        self.status_changed.emit("connecting", f"Opening capture source '{self.source}'...")
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status_changed.emit("error", f"Failed to open video source: {self.source}")
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        try:
            self.status_changed.emit("connecting", f"Connecting to {self.uri} (loading TensorRT/ST-GNN)...")
            async with connect(self.uri, max_size=16 * 1024 * 1024) as ws:
                self.status_changed.emit("monitoring", f"Connected to {self.uri}")

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        if isinstance(self.source, str) and self.loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        self.status_changed.emit("stopped", "End of video stream reached.")
                        break

                    ok, buf = cv2.imencode(".jpg", frame, encode_params)
                    if not ok:
                        continue

                    t0 = time.perf_counter()
                    await ws.send(buf.tobytes())
                    raw_reply = await ws.recv()
                    rtt_ms = (time.perf_counter() - t0) * 1000.0

                    raw_dict = pickle.loads(raw_reply)
                    parsed = normalize_unpacked_results(raw_dict)

                    self.frame_ready.emit(frame, parsed, rtt_ms)
                    await asyncio.sleep(0)

        except ConnectionClosed as exc:
            self.status_changed.emit("error", f"Server closed connection: {exc}")
        except OSError as exc:
            self.status_changed.emit("error", f"Connection failed: {exc}")
        finally:
            cap.release()
            if self._running:
                self._running = False
            else:
                self.status_changed.emit("stopped", "Monitoring stopped.")
