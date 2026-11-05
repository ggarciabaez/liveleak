# Run the inference backend

The default image runs with CPU-only PyTorch. Place these three files in a
local `models` folder before running the service.

- `ppe_50ep.pt`, the trained PyTorch YOLO weights with the class order in SSOT
- `stgnn.pth` on `main` or `stgnn_checkpoint.pth` on `ambar-ml`, the trained motion model for that branch
- `calib.npz`, the calibration data fitted for that exact checkpoint and detector pipeline

From the repository root in PowerShell, run:

```powershell
docker build -t workplace-risk .
docker run --rm -p 127.0.0.1:9160:9160 --mount "type=bind,source=$((Resolve-Path ./models).Path),target=/models,readonly" workplace-risk
```

## NVIDIA deployment host

Install a CUDA 13-compatible NVIDIA driver and configure Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the Linux GPU host. Build and run from the repository root:

```sh
docker build --build-arg DEVICE=gpu -t workplace-risk:gpu .
docker run --rm --gpus all -p 127.0.0.1:9160:9160 --mount "type=bind,source=$(pwd)/models,target=/models,readonly" workplace-risk:gpu
```

## WebSocket protocol

The container uses `PPE_MODEL`, `STGNN_CHECKPOINT`, and `STGNN_CALIBRATION`
to override the default file paths under `/models`.

Connect to `ws://localhost:9160` and keep one connection open per camera so
the tracker and motion model can build history. Send one JPEG or PNG per
message, either as raw binary or as plain base64 text. Each message can be
up to 16 MiB.

The server sends one binary pickle per frame. Load it with
`pickle.loads(reply)` from a trusted server. The returned dictionary contains
`rule_score`, `model_score`, `fused_score`, and `model_unavailable_reason`.
During warmup and whenever motion scoring is unavailable, `model_score` and
`fused_score` may be `None`, depending on the detector version. On `main`, a
frame without tracked entities returns null scores and the reason
`no_detections`. An invalid frame returns a dictionary with an `error` key.
