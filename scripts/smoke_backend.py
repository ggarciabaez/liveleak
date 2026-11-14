"""Check a separately running local backend with its real model artifacts."""

import argparse
import asyncio
import json
import math
import pickle
from pathlib import Path

from websockets.asyncio.client import connect


def validate(result):
    if not isinstance(result, dict) or "error" in result:
        raise ValueError(f"Unexpected backend result: {result}")
    required = {"model_score", "rule_score", "fused_score", "model_unavailable_reason"}
    if not required <= result.keys():
        raise ValueError("Missing response fields")
    for key in ("model_score", "rule_score", "fused_score"):
        value = result[key]
        if value is not None and (
            not isinstance(value, (int, float)) or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid {key}: {value}")
    if result["model_score"] is None and not result["model_unavailable_reason"]:
        raise ValueError("Unavailable learned score has no explanation")


async def smoke(payload, frames):
    # Fixed loopback endpoint: pickle is unsuitable for untrusted responses.
    async with connect("ws://127.0.0.1:9160", open_timeout=30) as client:
        for _ in range(frames):
            await client.send(payload)
            result = pickle.loads(await asyncio.wait_for(client.recv(), timeout=60))
            validate(result)
        await client.send(b"invalid-image")
        error = pickle.loads(await asyncio.wait_for(client.recv(), timeout=60))
        if not isinstance(error, dict) or "error" not in error:
            raise ValueError("Invalid input did not produce an error")
        await client.send(payload)
        validate(pickle.loads(await asyncio.wait_for(client.recv(), timeout=60)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("evidence/smoke.json"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(smoke(args.frame.read_bytes(), args.frames))
    except Exception as exc:
        args.output.write_text(json.dumps({"result": "FAIL", "error": str(exc)}, indent=2))
        raise
    args.output.write_text(json.dumps({
        "result": "PASS", "frames": args.frames,
        "scope": "Real local backend, response contract, invalid input, continued use",
        "limitations": "Repeated image; does not validate motion accuracy, UI or persistence",
    }, indent=2))
    print("Backend smoke test: PASS")


if __name__ == "__main__":
    main()
