"""Smoke test for a RUNNING backend (the real container, with real weights).

    docker compose up --build        # in one terminal, wait until models are loaded
    python scripts/smoke_backend.py  # in another; exit code 0 = pass

Checks: the server accepts a connection; a valid JPEG returns a well-formed score
reply; malformed bytes return a clear error; the connection still works afterwards.
Override the address with --uri or SMOKE_URI. Needs only numpy, opencv and websockets.
"""
import argparse
import asyncio
import os
import pickle
import sys

import cv2
import numpy as np
from websockets.asyncio.client import connect

REQUIRED_KEYS = {"model_score", "rule_score", "fused_score", "model_unavailable_reason"}


def test_frame():
    img = np.full((480, 640, 3), 127, dtype=np.uint8)
    cv2.rectangle(img, (200, 120), (320, 400), (40, 90, 200), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


async def exchange(ws, payload, timeout):
    await ws.send(payload)
    return pickle.loads(await asyncio.wait_for(ws.recv(), timeout))


async def smoke(uri, timeout):
    failures = []

    def check(name, cond, detail=""):
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" - {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    async with connect(uri, max_size=16 * 1024 * 1024, open_timeout=30) as ws:
        check("connects to backend", True)
        # First reply waits for the detector (and weights) to load, hence the long timeout.
        ok = await exchange(ws, test_frame(), timeout)
        check("valid frame returns a score reply", "error" not in ok and REQUIRED_KEYS <= ok.keys(), str(ok)[:200])
        bad = await exchange(ws, b"not an image", timeout)
        check("malformed frame returns a clear error", "JPEG or PNG" in str(bad.get("error", "")), str(bad)[:200])
        again = await exchange(ws, test_frame(), timeout)
        check("connection still works after an error", "error" not in again and REQUIRED_KEYS <= again.keys(),
              str(again)[:200])
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default=os.environ.get("SMOKE_URI", "ws://localhost:5020"))
    ap.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for each reply")
    args = ap.parse_args()
    try:
        failures = asyncio.run(smoke(args.uri, args.timeout))
    except (TimeoutError, OSError) as exc:
        print(f"[FAIL] backend not reachable or timed out at {args.uri}: {exc!r}")
        return 1
    print("SMOKE TEST " + ("FAILED: " + ", ".join(failures) if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
