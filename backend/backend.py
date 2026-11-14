"""WebSocket entry point for streaming frames into the risk detector."""
# TODO: test this and hook to the actual app. Also rename ts, it's not fair.
import asyncio
import base64
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from modules.risk import RiskDetector

load_dotenv()

def make_detector():  # TODO: correct risk detector usage
    MODEL_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
    paths = [MODEL_DIR / Path(x) for x in os.environ.get("MODEL_FILES").split(",")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path}")
    return RiskDetector(str(paths[0]), paths[1], paths[2])


def infer(detector, message):
    try:
        encoded = (  # TODO: remember this expects a JPEG frame! So Camera->JPG->B64 <--> B64->JPG->Frame
            base64.b64decode(message, validate=True) if isinstance(message, str) else message
        )
        frame = None
        if encoded:
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, cv2.error):
        frame = None
    if frame is None:
        return {"error": "Expected a JPEG or PNG frame as binary bytes or base64 text"}

    # main returns early on empty frames without updating its diagnostics.
    detector.results.clear()  # TODO: only clear on handshake
    fused_score, _ = detector(frame)
    scores = detector.results.get("scores")
    if scores is None:
        model_score = rule_score = None
        reason = "no_detections"
    else:
        rule_score = scores[1]
        model_score = scores[2] if len(scores) == 3 else scores[0]
        reason = detector.results.get("model_unavailable_reason")
    kf_res = detector.results.get("kf")  # (x, P, K) tuple from KalmanFilter.fuse
    detng_res = detector.results.get("detng")  # (total_risk, violations, scores) from DetNG.execute
    detect_res = detector.results.get("detect")
    return {
        "model_score": float(model_score) if model_score is not None else None,
        "rule_score": float(rule_score) if rule_score is not None else None,
        "fused_score": float(fused_score) if fused_score is not None else None,
        "results": {
            "kf": [float(v) for v in kf_res] if kf_res is not None else None,
            "detng": detng_res,
            "detect": detect_res,
        },
        "model_unavailable_reason": reason,  # reasons! TODO
    }


async def handle_connection(websocket):
    loop = asyncio.get_running_loop()
    try:
        # Keep each stream's tracker and GPU inference on the same worker.
        with ThreadPoolExecutor(max_workers=1) as worker:
            detector = await loop.run_in_executor(worker, make_detector)
            async for message in websocket:
                result = await loop.run_in_executor(worker, infer, detector, message)
                await websocket.send(pickle.dumps(result, protocol=4))
    except ConnectionClosed:
        pass


async def main():
    async with serve(handle_connection, "0.0.0.0", 9160, max_size=16 * 1024 * 1024) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
