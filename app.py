"""Local monitoring application for the construction-risk pipeline.

The server uses only Python's standard HTTP library. OpenCV, Ultralytics,
Supervision, and the tracker are loaded by ``modules.Detector`` when the camera
worker starts.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
DEFAULT_MODEL_PATH = ROOT / "assets" / "ppe_50ep.engine"


@dataclass
class MonitorState:
    camera_connected: bool = False
    detector_ready: bool = False
    frame_number: int = 0
    fps: float = 0.0
    entity_counts: dict[str, int] = field(
        default_factory=lambda: {
            "persons": 0,
            "machinery": 0,
            "vehicles": 0,
            "cones": 0,
        }
    )
    risk_score: float | None = None
    covariance: float | None = None
    error: str | None = None


class CameraWorker:
    """Capture frames, run Detector, and publish the latest JPEG and metadata."""

    def __init__(self, camera_source: str, model_path: Path):
        self.camera_source = camera_source
        self.model_path = model_path
        self.state = MonitorState()
        self._latest_jpeg: bytes | None = None
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run,
            name="camera-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return asdict(self.state)

    def wait_for_jpeg(self, previous_frame: int, timeout: float = 2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self.state.frame_number > previous_frame
                    or self._stop_event.is_set()
                ),
                timeout=timeout,
            )
            return self.state.frame_number, self._latest_jpeg

    @staticmethod
    def _parse_camera_source(source: str):
        return int(source) if source.isdigit() else source

    @staticmethod
    def _draw_detections(frame, detections, cv2):
        class_names = {
            0: "hardhat",
            1: "mask",
            5: "person",
            6: "cone",
            7: "vest",
            8: "machinery",
            9: "vehicle",
        }
        tracker_ids = detections.tracker_id
        for index, box in enumerate(detections.xyxy.astype(int)):
            class_id = int(detections.class_id[index])
            confidence = float(detections.confidence[index])
            tracker_id = None if tracker_ids is None else tracker_ids[index]
            label = class_names.get(class_id, f"class {class_id}")
            if tracker_id is not None:
                label += f" #{int(tracker_id)}"
            label += f" {confidence:.2f}"
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (81, 203, 145), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (232, 238, 243),
                1,
                cv2.LINE_AA,
            )
        return frame

    def _set_error(self, message: str) -> None:
        with self._condition:
            self.state.error = message
            self._condition.notify_all()

    def _run(self) -> None:
        try:
            import cv2
        except ImportError:
            self._set_error("OpenCV is not installed. Run `uv sync` first.")
            return

        detector = None
        if self.model_path.is_file():
            try:
                from modules.Detector import Detector

                detector = Detector(str(self.model_path))
                self.state.detector_ready = True
            except Exception as exc:
                self._set_error(f"Detector initialization failed: {exc}")
        else:
            self._set_error(
                f"YOLO model not found at {self.model_path}. "
                "The raw camera will still be shown."
            )

        capture = cv2.VideoCapture(self._parse_camera_source(self.camera_source))
        if not capture.isOpened():
            self._set_error(f"Could not open camera source: {self.camera_source}")
            return

        self.state.camera_connected = True
        last_frame_time = time.perf_counter()

        try:
            while not self._stop_event.is_set():
                success, frame = capture.read()
                if not success:
                    self._set_error("The camera stopped returning frames.")
                    break

                counts = {key: 0 for key in self.state.entity_counts}
                if detector is not None:
                    try:
                        raw_detections = detector(frame)
                        tracked, untracked = detector.only_track(
                            raw_detections,
                            ids=(5, 8, 9),
                        )
                        vectors = detector.get_vectors(tracked, untracked)
                        counts = {
                            "persons": len(vectors[0]),
                            "machinery": len(vectors[1]),
                            "vehicles": len(vectors[2]),
                            "cones": len(vectors[3]),
                        }
                        frame = self._draw_detections(frame, tracked, cv2)
                        frame = self._draw_detections(frame, untracked, cv2)
                        self.state.error = None
                    except Exception as exc:
                        self._set_error(f"Detector frame processing failed: {exc}")

                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 82],
                )
                if not encoded:
                    continue

                now = time.perf_counter()
                elapsed = max(now - last_frame_time, 1e-6)
                last_frame_time = now

                with self._condition:
                    self._latest_jpeg = jpeg.tobytes()
                    self.state.frame_number += 1
                    self.state.fps = round(1.0 / elapsed, 1)
                    self.state.entity_counts = counts
                    self._condition.notify_all()
        finally:
            capture.release()
            with self._condition:
                self.state.camera_connected = False
                self._condition.notify_all()


class MonitoringServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, camera_worker):
        super().__init__(address, handler)
        self.camera_worker = camera_worker


class MonitoringHandler(BaseHTTPRequestHandler):
    server: MonitoringServer

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/camera/stream":
            self._stream_camera()
        elif path in {"/api/status", "/api/assessment"}:
            self._send_json(self.server.camera_worker.snapshot())
        elif path == "/":
            self._send_file(WEB_ROOT / "index.html")
        elif path.startswith("/static/"):
            requested = (WEB_ROOT / path.removeprefix("/static/")).resolve()
            if WEB_ROOT.resolve() not in requested.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(requested)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _stream_camera(self):
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        frame_number = -1
        try:
            while True:
                frame_number, jpeg = self.server.camera_worker.wait_for_jpeg(
                    frame_number
                )
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local risk monitor")
    parser.add_argument(
        "--host",
        default=os.getenv("LIVELEAK_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("LIVELEAK_PORT", "8000")),
    )
    parser.add_argument(
        "--camera",
        default=os.getenv("LIVELEAK_CAMERA_SOURCE", "0"),
        help="Camera index, video path, or RTSP URL",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.getenv("LIVELEAK_MODEL_PATH", DEFAULT_MODEL_PATH)),
        help="Path to the YOLO .pt or TensorRT .engine model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    worker = CameraWorker(args.camera, args.model.resolve())
    worker.start()
    server = MonitoringServer((args.host, args.port), MonitoringHandler, worker)
    print(f"Monitoring application: http://{args.host}:{args.port}")
    print(f"Camera source: {args.camera}")
    print(f"Detector model: {args.model.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        worker.stop()


if __name__ == "__main__":
    main()
