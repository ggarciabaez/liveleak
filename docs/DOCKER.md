# Run the inference backend

The backend is a single container serving a WebSocket on port 5020. Model
files are not in git: on start the container downloads any missing ones from a
GitHub Release into `./models` and reuses them on later runs.

## Quick start (NVIDIA GPU host)

Install a CUDA 13-compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then from the repository root:

```sh
docker compose up --build
```

Use a different model set with `MODEL_RELEASE_TAG=models-v2 docker compose up`.

## Model files

The container needs three files in `/models` (the `./models` folder):

- `ppe_50ep.pt`, the trained PyTorch YOLO weights with the class order in SSOT
- `stgnn.pth`, the trained motion model
- `calib.npz`, the calibration data fitted for that exact checkpoint and detector pipeline

`stgnn.pth` and `calib.npz` are a matched pair, so ship them in the same
release. One tag is one consistent set.

### Publishing a set

1. On GitHub, create a release with a tag such as `models-v1` (Releases ->
   Draft a new release). Mark it as a pre-release if you don't want it shown
   as the project's latest.
2. Attach the three files as release assets (2 GB limit per file).
3. Optional: attach a `SHA256SUMS` asset (`sha256sum ppe_50ep.pt stgnn.pth calib.npz > SHA256SUMS`).
   When present, downloads are verified and a mismatch is deleted and fails the start.

Files already in `./models` are never overwritten, so delete a file to force a
re-download after changing tags. To skip downloading, put the files in
`./models` yourself.

| Variable            | Default                           | Purpose                                           |
|---------------------|-----------------------------------|---------------------------------------------------|
| `MODEL_RELEASE_TAG` | `models-v1`                       | Release tag to download from                      |
| `MODEL_REPO`        | `ggarciabaez/liveleak`            | `owner/repo` hosting the release (must be public) |
| `MODEL_FILES`       | `ppe_50ep.pt,stgnn.pth,calib.npz` | Asset names to fetch                              |

`PPE_MODEL`, `STGNN_CHECKPOINT`, and `STGNN_CALIBRATION` override the paths
the server loads, as before.

## CPU only (no GPU)

The CPU image uses CPU PyTorch and does not need the toolkit. Compose reserves
a GPU, so use plain Docker for this:

```sh
docker build -t workplace-risk:cpu .
docker run --rm -p 127.0.0.1:5020:5020 -v "$(pwd)/models:/models" workplace-risk:cpu
```

(PowerShell: use `${PWD}/models` in place of `$(pwd)/models`.)

## WebSocket protocol

Connect to `ws://localhost:5020` and keep one connection open per camera so
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
