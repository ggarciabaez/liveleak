"""Lightweight WebSocket test client for streaming video/webcam frames to backend.py."""

import argparse
import asyncio
import pickle
import time

import cv2
from dotenv import load_dotenv
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

load_dotenv()


def draw_hud(frame, result: dict, rtt_ms: float) -> None:
    """Overlay inference scores and round-trip latency onto the preview frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (390, 155), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if "error" in result:
        lines = [
            f"RTT:   {rtt_ms:.1f} ms",
            f"Error: {result['error']}",
        ]
        color = (0, 0, 255)
    else:
        fused = result.get("fused_score")
        model = result.get("model_score")
        rule = result.get("rule_score")
        reason = result.get("model_unavailable_reason")

        fused_str = f"{fused:.3f}" if fused is not None else "None"
        model_str = f"{model:.3f}" if model is not None else "None"
        rule_str = f"{rule:.3f}" if rule is not None else "None"

        lines = [
            f"Fused Score: {fused_str}",
            f"Model Score: {model_str}",
            f"Rule Score:  {rule_str}",
            f"Status:      {reason or 'ok'}",
            f"RTT:         {rtt_ms:.1f} ms ({1000.0 / max(rtt_ms, 1e-3):.1f} FPS)",
        ]
        # Highlight high risk scores in red/orange
        if fused is not None and fused >= 0.7:
            color = (0, 60, 255)
        elif fused is not None and fused >= 0.4:
            color = (0, 200, 255)
        else:
            color = (80, 230, 80)

    for idx, text in enumerate(lines):
        y = 35 + idx * 24
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color if idx == 0 else (235, 235, 235),
            2 if idx == 0 else 1,
            cv2.LINE_AA,
        )


async def stream_client(uri: str, source: str | int, jpeg_quality: int, headless: bool) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    try:
        print(f"Connecting to {uri} (initializing detector on server)...")
        async with connect(uri, max_size=16 * 1024 * 1024) as ws:
            print("Connected. Press 'q' in the preview window (or Ctrl+C) to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop if reading from a video file
                    if isinstance(source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                ok, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                t0 = time.perf_counter()
                # Send binary JPEG bytes directly (avoids base64 overhead; backend accepts both)
                await ws.send(buf.tobytes())
                raw_reply = await ws.recv()
                rtt_ms = (time.perf_counter() - t0) * 1000.0

                result = pickle.loads(raw_reply)

                if headless:
                    print(f"[{rtt_ms:6.1f} ms] {result}")
                else:
                    draw_hud(frame, result, rtt_ms)
                    cv2.imshow("Workplace Risk Detector - Test Client", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Yield briefly so the event loop stays responsive
                await asyncio.sleep(0)
    except ConnectionClosed as exc:
        print(f"Server closed connection: {exc}")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test frontend for RiskDetector WebSocket server.")
    parser.add_argument(
        "--uri",
        default="ws://localhost:9160",
        help="WebSocket server URI (default: ws://localhost:9160)",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index (e.g., 0) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG encoding quality 1-100 (default: 85)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print results to stdout without opening an OpenCV GUI window",
    )
    args = parser.parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(stream_client(args.uri, args.source, args.quality, args.headless))
    except KeyboardInterrupt:
        print("\nStopped.")
