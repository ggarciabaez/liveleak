# liveleak — Single Source of Truth

**Read this fully before changing anything in `modules/`, `scripts/`, or `main.py`. If you are an LLM assistant: this file is binding context. Do not infer contracts from file names, `docs/`, or your own assumptions.**

- Verified against commit `87035f4` (`main`, 2026-10-29), by reading the code. Tests were **not executed** for this pass (see §10).
- Precedence when sources disagree: **code > tests > this file > `docs/` (HANDBOOK, diagrams, TODO)**. `docs/` is partly stale; §9 lists where.
- If you change a contract in §4–§6, update this file **in the same PR**. A PR that changes a contract without touching SSOT.md should be rejected.

---

## 1. What this project is

A real-time computer-vision pipeline that estimates **workplace incident risk** (a score in [0, 1]) from video of a worksite. It detects and tracks people, machinery, vehicles, and safety cones, checks PPE (hardhat, mask, vest), and scores risk with two independent methods that are then fused:

1. **DetNG** — a deterministic, hand-written rule engine (interpretable, no training).
2. **ST-GNN** — a self-supervised spatio-temporal graph network that forecasts each entity's next-frame motion; large forecast error = "unusual motion" = risk.

A **Kalman-style fuser** combines both scores into the final risk.

**Hard constraint:** OSHA/ethics rules prevent using footage of injuries. The ST-GNN therefore must stay **self-supervised on incident-free footage**. Never add a labeled-injury classifier path. The demo need not display the risk score.

Repo name is `liveleak`; `pyproject.toml` package name is `ppe` (don't "fix" this casually).

---

## 2. Status at a glance

| Area | State |
|---|---|
| YOLO detection (finetuned YOLO26m, TensorRT engine) | Works. Weights are **not in the repo** (§8). |
| Tracking (ByteTrack via `trackers` package) | Works. |
| Feature-vector extraction (`Detector.get_vectors`) | Works but has suspected bugs (§9). |
| DetNG rule engine | Works. 4 rules. |
| ST-GNN model + calibrator | Model, scoring, calibrator implemented. **No real training code.** |
| Kalman fusion | Works, but is a per-frame weighted blend, not a temporal filter (§6.5). |
| `RiskDetector` end-to-end loop (`modules/risk.py`) | Works; used by `main.py` with a webcam. |
| Training pipeline (`scripts/`) | **Scratch / non-functional** (§7). |
| `../scripts/dataset.py`, `Dataset` in `train_tools.py` | **Empty stubs.** |
| `app/app.py`, `app/config.toml` | **Empty files.** |
| Server / desktop app / SQLite / reports (in `docs/diagrams/`) | **Not implemented — design only** (§11). |
| `RecursiveRiskDetector` | Empty stub. |

---

## 3. Architecture and data flow

```
frame (BGR np.ndarray)
  └─ Detector.__call__          YOLO → sv.Detections (all 10 classes)
      └─ Detector.get_vectors   track {Person, machinery, vehicle} with ByteTrack;
                                associate PPE to persons; build vectors
          → list of 4 arrays, each (n_k, 13): [persons, machinery, vehicles, cones]
      └─ RiskDetector.merge_frame   concat → (n, 13); rewrite cone IDs
      └─ deque of last `winsize` frames (default 5)
      └─ RiskDetector.build_window  → feats (T, N, 13) float32, valid (T, N) bool
            ├─ DetNG.execute(feats, valid)             → rule score  ∈ [0, 1)
            └─ STGNN.compute_window_error(feats, valid)→ per-entity RMSE
                  └─ MotionAnomalyCalibrator.score     → percentile ∈ [0, 1]
      └─ KalmanFilter.fuse(calib_score, rule_score, σ_stgnn, σ_detng) → fused risk
```

Everything runs **in-process, synchronously, per frame**, passing NumPy arrays / floats. There is no network, queue, or database in the working code.

---

## 4. Data contracts

### 4.1 Classes (YOLO output) — `modules/Detector.py`

```
CLASSES = ('Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest',
           'Person','Safety Cone','Safety Vest','machinery','vehicle')
          #   0        1        2            3         4
          #   5        6           7             8          9
PPE_IDS = (0,1,2,3,4,7)   ID_PERSON=5  ID_CONE=6  ID_MACHINERY=8  ID_VEHICLE=9
```

- **Only Person (5), machinery (8), vehicle (9) are tracked.** Cones (6) and all PPE classes are untracked.
- PPE presence is **hardhat=0, mask=1, vest=7 only**. `NO-*` classes (2, 3, 4) are detected but **ignored**: "no hardhat" means "no hardhat box found".
- PPE → person association: a PPE box belongs to a person if the PPE box **center lies inside the person box**. Overlapping persons can both claim the same PPE.
- Class order is baked into the trained engine. **Changing it requires retraining YOLO and updating every constant above and `separate()` in both detectors.**

### 4.2 Feature vector — 13 columns, fixed order

```
col:   0   1       2        3        4     5   6   7   8   9   10    11    12
name:  id  person  machine  vehicle  cone  px  py  vx  vy  hh  mask  vest  na
```

| Cols | Meaning |
|---|---|
| 0 `id` | Track ID. Special values in §4.4. Stored as float. |
| 1–4 | One-hot entity type: person, machine, vehicle, cone. Exactly one is 1. |
| 5–6 `px, py` | Bounding-box centroid, **pixels in the source frame** (not normalized, not meters). |
| 7–8 `vx, vy` | Velocity, **pixels per frame**. 0 on a track's first frame and for cones. (Definition differs between detectors — §9, item 1.) |
| 9–11 `hh, mask, vest` | PPE present = 1. Only meaningful for persons. |
| 12 `na` | 1 for non-persons (PPE not applicable), 0 for persons. |

- A "frame" is an `(n, 13)` array. `Detector.get_vectors` returns a **list of 4** such arrays in order `[persons, machinery, vehicles, cones]`; empty groups are `(0, 13)`.
- **Never reorder, add, or remove columns** without retraining ST-GNN (it consumes cols 1–12 via `FEAT_SLICE = slice(1, 13)`, with `POS_SLICE = 5:7`, `KIN_SLICE = 5:9`) and rewriting DetNG's `IDX` dict, `gen_valid_feature`, `tests/conftest.py`, and the notes in `Detector.draw`.
- Test/synthetic helper: `gen_valid_feature(obj_type, [px,py,vx,vy], ppe_bitmask)` — `obj_type` 1..4 = column index of the one-hot; PPE bitmask bit0=hardhat, bit1=mask, bit2=vest.

### 4.3 Window tensor — the ST-GNN / DetNG input

```
feats : (T, N, 13) float32     slot n = ONE track across the whole window
valid : (T, N)     bool        valid[t, n] = track n was detected in frame t
```

- Built by `RiskDetector.build_window(frames)` from the deque of per-frame `(n_t, 13)` arrays. Slots are sorted by ID; absent entries are zero rows with `id = -1` and `valid = False`.
- **Rows where `valid` is False must never influence anything.** The model enforces this (tested: padding contents never matter). Don't "optimize" by dropping the mask.
- `STGNN._prepare` enforces: shape `(T, N, 13)`, `T ≥ 1`, mask shape `(T, N)`, **the same ID in a slot in every frame where it's present**, and **unique IDs across slots**. Violations raise `ValueError`. If `mask is None`, presence is inferred as `id != -1`.
- **Active** entities (those in the graph and temporal state) = `valid AND NOT cone`. Cones are static context for DetNG only; STGNN ignores them.
- N varies per window (entities appear/disappear). Anything that assumes fixed N is wrong.
- Default window length `winsize = 5` (a `deque(maxlen=5)`). ST-GNN uses frames `[0..T-2]` to predict frame `T-1`, so it needs `T ≥ 2` and effectively sees 4 frames of history at the default.

### 4.4 ID conventions

| ID | Meaning |
|---|---|
| real track ID | From ByteTrack. Persons, machinery, vehicles. |
| `-1` | Padding slot in a window. **Also** what `Detector.get_vectors` emits for raw cones. |
| `≤ -2` | Cone after `RiskDetector.merge_frame`: `-(2 + gx*10000 + gy)` where `gx, gy = centroid // GRID` (`GRID = 32` px). |

- Raw `get_vectors` output **must** pass through `merge_frame` before `build_window`; otherwise cones look like padding and collide.
- Known limitation (code TODO in `build_window`): two cones in the same 32 px grid cell share an ID and overwrite each other.
- Cone IDs are a spatial hash, **not** tracks. Don't treat them as identities over time.

### 4.5 Scores

| Value | Range | Where |
|---|---|---|
| DetNG score | `[0, 1 − e⁻¹ ≈ 0.632]` with default weights (sum = 1.0) | `DetNG.execute` |
| ST-GNN raw error | ≥ 0, RMSE over `[px, py, vx, vy]` (pixels and px/frame mixed) | `compute_window_error` |
| Calibrated ST-GNN score | `[0, 1]`, percentile of error vs. normal footage; **max over entities** | `MotionAnomalyCalibrator.score` |
| Fused risk | Weighted blend of the two above | `KalmanFilter.fuse` |

The DetNG ceiling (~0.632) is a property of the current weights, not a bug in the aggregation — but it means a fixed threshold like 0.9 (as drawn in `ai_pipeline.mermaid`) is not reachable from DetNG alone. Do not "fix" it silently; it's an open decision (§12).

---

## 5. Artifacts (not in the repo)

`main.py` expects, relative to the working directory:

| File                     | What                                                  | Produced by                                                                                                                         |
|--------------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `assets/ppe_50ep.engine` | TensorRT YOLO26m engine (needs an NVIDIA GPU)         | `sandbox/export_model.py` from `assets/ppe_50ep.pt` (imgsz 640, conf 0.35, iou 0.65, max_det 100, FP16 requested via `quantize=16`) |
| `../assets/v3/stgnn.pth` | `STGNN.state_dict()`, loaded with `weights_only=True` | `STGNN.save_weights`                                                                                                                |
| `../assets/v3/calib.npz` | key `errors`: sorted normal forecast errors           | `fit_motion_anomaly_calibrator`                                                                                                     |

`assets/` is in `.gitignore`. `.gitattributes` marks `*.engine`/`*.pt` as Git LFS, but nothing under `assets/` is committed. Where the team's canonical copies live is an open question (§12). The training notebook writes checkpoints to Google Drive (`MyDrive/liveleak/checkpoints/model_v1`).

**Coupling rules — break these and results are silently wrong:**

1. `stgnn.pth` and `calib.npz` are a **matched pair**. Any retrain of the ST-GNN (or any change to the feature columns, `hidden_dim`, `graph_radius`, or the detector's velocity/position definition) requires **refitting the calibrator** and shipping both together.
2. `graph_radius` is **not** stored in the state dict. It must be the same at train and inference (`RiskDetector` default `graph_rad=250`, in pixels). `hidden_dim` defaults to 64 and `RiskDetector` doesn't pass it, so checkpoints must be trained with 64 or `load_state_dict` fails.
3. The YOLO engine's class order must equal `CLASSES` (§4.1).

---

## 6. Module contracts

### 6.1 `modules/Detector.py` — `Detector` (inference-time)

- `Detector(model_path, framerate=30, track_life=5)` → `YOLO(model, task='detect')`; ByteTrack (`trackers.ByteTrackTracker`) with `lost_track_buffer = track_life * framerate` (= 150 at defaults; the library defines this as a count of 30-FPS frames, scaled by `frame_rate`, so ≈ 5 s).
- Tracks that haven't yet survived `minimum_consecutive_frames` come back with `tracker_id = -1`; `get_vectors` **drops** non-cone entities with ID -1, so a new object only appears in the vectors once its track is confirmed.
- `__call__(frame) -> sv.Detections` (raw, all classes).
- `get_vectors(frame_or_dets, ud=None) -> list[np.ndarray]` — see §4.2. Accepts raw frames or `sv.Detections` (`RiskDetector` passes detections).
- Velocity here = **mean of the two corner velocities from the ByteTrack Kalman state** (`(va+vb)/2`), and zero when the ID wasn't in the previous frame's `track_dict`.
- `draw(frame, vectors)` is visualization only.

### 6.2 `scripts/batchdet.py` — `BatchedDetector` (offline / dataset building)

- Same vector layout, **different internals**: velocity = **raw centroid delta between consecutive frames**; position = raw bbox centroid; `track_life` is passed **directly** as `lost_track_buffer` (default 30 ≈ 1 s at 30 FPS, versus 150 ≈ 5 s in `Detector`); one tracker per stream.
- `process_batch(frames) -> list[(n,13)]`; `None` frames (exhausted streams) yield `(0, 13)`. Despite the name, YOLO runs **one frame at a time** (TensorRT batch workaround noted in code).
- `match_ppe` is duplicated in both detector files. Keep them identical until extracted.

### 6.3 `modules/detng.py` — `DetNG`

```python
DetNG().compile([(rule_fn, weight), ...]) -> self
DetNG.execute(fvecs, mask) -> (total_risk: float, violations: list[str], scores: np.ndarray)
```

- `fvecs` is `(T, N, 13)`, `mask` is `(T, N)` bool. **`mask` is a required argument.**
- For each frame it builds a context dict (`frame, P, V_P, H, V_H, persons, cones`) and passes **the list of T contexts** to every rule. **Rules read `ctx[-1]` (the latest frame).**
- A rule is `rule(ctx_list) -> float in [0, 1]`; `rule.__name__` is reported in `violations` when its score > 0.
- Aggregation: `total_risk = 1 − exp(−Σ wᵢ·sᵢ)` (weights are absolute, not normalized).
- "Hazards" = machinery **or** vehicle.
- Default rules (`get_rules()`), weights `(0.1, 0.25, 0.5, 0.15)`:
  1. `rule_ppe_compliance` — fraction of missing hardhat/vest among persons (ignores mask).
  2. `rule_vectorized_proximity` — `d_safe=10`, `d_danger=2`, max over person–hazard pairs.
  3. `rule_kinematic_ttc` — `T_horizon=5`, `d_thresh=3`, closest-approach severity.
  4. `rule_cone_zone` — person inside the bounding box (+`buffer=1`) of ≥ 3 cones.
- **Units:** distance thresholds are unitless in the code but applied to **pixel** coordinates in the real pipeline (synthetic tests use a 0–100 space). `d_safe=10` is therefore ~10 px at real resolutions. See §9, item 5.
- Empty-engine path returns a **2-tuple**; callers unpack 3 (marked `xfail` in tests). Don't rely on it.

### 6.4 `modules/stgnn.py` — `STGNN`, `MotionAnomalyCalibrator`

- `STGNN(graph_radius, hidden_dim=64)`: node encoder (12 → hidden) → radius graph per frame (`cdist(px,py) ≤ graph_radius`, edge input `[h_recv, h_send, Δpos/graph_radius]`, mean-aggregated) → `spatial_update` GRUCell → `temporal_cell` GRUCell (updated only on frames where the entity is active) → `motion_head` predicts **residuals** `[Δpx, Δpy, Δvx, Δvy]` added to the last observed kinematics.
- `forward(features, mask=None) -> (N, 4)`: next-step kinematics for every slot. Rows for slots absent at `T-1` or cones are **meaningless** — select with `mask[-1]` and the type columns.
- `compute_window_error(features, mask) -> (k,)`: per-entity RMSE for entities active at both `T-2` and `T-1` (history `[:-1]` predicts the last frame). Never sees the target frame as input.
- Objective is **one-step** kinematics forecasting, not K-step (HANDBOOK says K-step; the code is one-step).
- `MotionAnomalyCalibrator`: `fit(errors)` stores sorted normal errors; `score(errors)` = `searchsorted(side="right") / n`, returning the **max** over the given errors (0.0 for empty); `load(path)` reads `errors` from an `.npz`.
- `fit_motion_anomaly_calibrator(model, normal_sequences, window_size=8, ...)` calls `model(history)` **without a mask** and requires every track to be present, in the same row, in every frame (`_prepare_feature_sequences`). It therefore **cannot ingest real tracked data** with entities appearing/disappearing (§9, item 3).

### 6.5 `modules/kfilter.py` — `KalmanFilter`

```python
fuse(stgnn_risk, detng_risk, stgnn_variance, detng_variance) -> (x, P, K)
```
- `P_pred = stgnn_variance + Q`; `K = P_pred / (P_pred + detng_variance)`; `x = stgnn_risk + K·(detng_risk − stgnn_risk)`; `P = (1−K)·P_pred`.
- **Argument order matters**: ST-GNN first, DetNG second (risk, then variance).
- `self.x` and `self.P` are stored but **never read on the next call**; each call is an independent blend with no memory. The "temporal smoothing" described in HANDBOOK is not implemented.
- `RiskDetector` parameters named `std_detng` / `std_stgnn` are passed as **variances**.

### 6.6 `modules/risk.py` — `RiskDetector`

- `RiskDetector(yolo_path, stgnn_path, calib_path, winsize=5, std_detng=0.1, std_stgnn=0.1, graph_rad=250, ruleset=None)`.
- `__call__(frame, draw=False) -> (fused_score | None, frame)`. Returns `None` (and does **not** enqueue) when there are zero detections.
- Side channel: `self.results` dict is overwritten every call with keys `scores`, `kf`, `detng`, `detect`. Treat it as debug output, not an API.
- `preload` / `preload_direct` prime the window queue (`_prime_run` fuses with ST-GNN risk 0.0).
- Uses CUDA for the ST-GNN if available. The window is built from a `deque`, so memory/latency are bounded by `winsize`.

---

## 7. Training / data pipeline (planned, not working)

Intended flow: input videos → `BatchedDetector` (detect + track per video) → `merge_frame` → store per-frame arrays → a `Dataset` builds fixed-length windows with validity masks → train ST-GNN (GRU-based) on incident-free footage → hold out normal clips → `fit_motion_anomaly_calibrator` → save `stgnn.pth` + `calib.npz`.

Current reality:
- `scripts/compress_dataset.py` is scratch: it uses `Detector` without importing it, reads `../assets` and `../data/videos`, and **executes at import**. Don't import it.
- `../scripts/dataset.py` and `Dataset` in `scripts/train_tools.py` are empty stubs.
- `scripts/train.ipynb` is a Colab notebook cloned from the **VLA** project; it still contains VLA/MetaWorld leftovers (`%cd VLA`, `ep*.npz`, a `backbone` LR split). It does not train this ST-GNN yet.
- `scripts/train_tools.py` helpers (shard copy/verify, AdamW builder, cosine scheduler, checkpointing) are generic and reusable; `save_checkpoint` expects `model.cfg`, which `STGNN` does not have.
- `sandbox/` holds experiments (alternate multivariate Kalman, PyG prototype, interactive loops). **Not supported; do not import from it in `modules/`.**

---

## 8. Environment, tooling, tests, CI

- Python ≥ 3.12 (`pyproject.toml`), managed with **uv** (`uv.lock`). CI uses 3.13.
- Real inference needs an NVIDIA GPU + TensorRT. CI and unit tests do not.
- Run tests: `pytest` (config in `pytest.ini`: `testpaths=tests`, `pythonpath=.`, marker `slow`).
- CI (`.github/workflows/ci.yml`): on PR and push to `main`: `ruff check --isolated tests` (lint covers **only `tests/`**) and `pytest --cov=modules` on CPU torch.
- Unused-but-declared deps: `filterpy` (the Kalman filter is hand-rolled), `pytorch-geometric`/`torch-geometric` (only the sandbox prototype touches PyG).
- Demo entry point: `python main.py` (webcam 0, window shows boxes/IDs/velocity arrows and the score).

---

## 9. Known divergences and bugs (verified by reading; fix or consciously accept)

Severity: **H** silently wrong results, **M** wrong in some conditions, **L** hygiene.

| # | Sev | Issue |
|---|---|---|
| 1 | H | **Train/inference skew in the detectors.** `Detector` velocity = mean corner velocity from the ByteTrack Kalman state, position from the tracker state; `BatchedDetector` velocity = centroid delta, position = raw bbox centroid; lost-track buffer is 150 frames vs 30. Data built with `BatchedDetector` is not distributed like what `Detector` feeds at inference. |
| 2 | H | **Likely index misalignment in `Detector.get_vectors`.** `cx, cy, v` are computed from *all* tracks' Kalman states, then indexed by `i` over a *per-class subset* of detections (`process_subset`). Untracked cones also index this array; with no active tracks the function returns early and drops cones. Needs a targeted test before anyone trusts `px/py/vx/vy` in multi-class scenes. (Static reading; not run.) |
| 3 | H | **Variable-N data.** Entities appear/disappear, so real data is non-contiguous. `build_window` + masks handle this at inference, but `fit_motion_anomaly_calibrator` and `_prepare_feature_sequences` require full presence in every frame. Calibration and training on real data are blocked until this is reconciled. |
| 4 | M | **Kalman fuser has no memory** (§6.5). |
| 5 | M | **Pixel units vs. rule thresholds.** `d_safe=10`, `d_danger=2`, `d_thresh=3`, `buffer=1` were written against a 0–100 synthetic space; the pipeline feeds pixels. `graph_rad=250` is tuned in pixels. Resolution changes alter behavior silently. |
| 6 | M | **DetNG ceiling ≈ 0.632** with default weights (Σw = 1). |
| 7 | M | Cone ID collisions in the same 32 px cell (`build_window` TODO). |
| 8 | M | `DetNG.execute` empty-engine path returns 2 values instead of 3. |
| 9 | M | **Tests are out of sync with the code (static reading).** `tests/detng_test.py` and `tests/pipeline_test.py` call `execute(history)` without the now-required `mask`; `pipeline_test.py` calls `STGNN.predict_kinematics`, which does not exist. `tests/stgnn_test.py` matches the current masked API. Run the suite before trusting it. |
| 10 | L | `stgnn.py` prints a debug line at import; `risk.py` imports `torch`/`np` through `modules.stgnn`; `Detector.py` still has `TODO: add protections for empty results`. |
| 11 | L | Docs drift (below). |

**Stale/aspirational docs:**
- `docs/HANDBOOK.md` — says SORT/BoT-SORT (code uses **ByteTrack via the `trackers` package**, not Ultralytics `model.track`); describes a graph builder with edge features `(dx, dy, distance)`, GraphSAGE/GAT, K-step forecasting, Deep SVDD, and a stateful Kalman filter. **None of these exist.** The code uses a radius graph with relative-position messages, GRUCells, one-step forecasting, and a stateless blend. Its checklist is mostly unchecked even where done. The class-rubric/deployment items live in `docs/DEPLOY_CHECK.md`.
- `docs/diagrams/*` — describe a monitoring **server + desktop app** system (WebSocket API, `AccessControl`, `CameraWorker`, `MonitorState`, `EventHistoryStore`, SQLite `risk.db`, weekly reports, MJPEG stream). **Not in the repo** (§11). `ai_pipeline.mermaid` shows a `> 0.9` warning and a "queue full" gate; the code only prints the score and runs once the first frame is queued.
- `README.md` — a single sentence; says nothing about usage.

---

## 10. Rules for changing code (humans and LLMs)

**Never, without an explicit team decision and a same-PR update to this file:**
1. Reorder/add/remove feature columns or change what a column means (§4.2).
2. Change class IDs/order (§4.1) or which classes are tracked.
3. Change the `(T, N, 13)` + `(T, N)` window convention, the ID rules (§4.4), or let padding rows affect results.
4. Change `STGNN` layers, `hidden_dim`, `FEAT_SLICE`, `graph_radius` semantics, or the detector's position/velocity definition **without** retraining and refitting the calibrator (§5).
5. Change `fuse(...)` argument order or the score ranges in §4.5.
6. Add a labeled-injury training path (§1).
7. Make `modules/` import from `sandbox/` or `scripts/`.

**Before opening a PR:**
- Run `pytest` locally; if a test fails because the code is newer than the test, fix the test and say so in the PR.
- If you touched `Detector.py` **or** `scripts/batchdet.py`, check whether the other needs the same change (they must produce identical vectors for identical inputs; see §9.1).
- Add or update a test for any contract you touched (shape/column/ID invariants belong in `tests/`).
- State in the PR description which sections of this file you changed.

**LLM-assisted work:** paste §4–§6 and §10 into the assistant's context. Treat any claim the assistant makes about shapes, columns, or units as unverified until it points to a line of code or a test. Have it list the files and contracts it intends to change **before** it edits.

---

## 11. Interfaces and communication

**Today (working code):** in-process Python calls with NumPy arrays; the only I/O is the OpenCV capture/window and the weight/calibration files in §5.

**Designed, not implemented** (from `docs/diagrams/`): a GPU-host monitoring service pulling frames over a TCP video stream; a WebSocket API (JSON + MJPEG) to a desktop app; `AccessControl`; a single-writer `EventHistoryStore` over SQLite (WAL) with `USER → SESSION → RISK_ESTIMATE(score, covariance)`; weekly report service; Docker on a cloud GPU VM. Treat these as a target architecture. Before building any of it, confirm it's still in scope (§12).

**Team process (intended, per `docs/DEPLOY_CHECK.md`; not yet enforced/verified):** peer-reviewed PRs with branch protection, CI on every PR, Mermaid diagrams as living docs, Docker + cloud deployment. Which channel the team uses to announce contract changes and who owns each module are **not recorded anywhere** — fill in §12.

---

## 12. Open decisions (need a human answer; then move into the sections above)

1. **Velocity/position definition:** tracker Kalman state or raw centroid delta? Pick one and make both detectors share a single implementation (§9.1–2).
2. **Units:** keep pixels (and rescale rule thresholds / pin an input resolution) or normalize coordinates to frame size/meters?
3. **Fixed-N vs. masked data** for calibration and training: adopt validity masks end-to-end (`Dataset` + calibrator) — confirm and specify the window/stride.
4. **DetNG scale:** normalize weights so the max score is 1, or keep absolute weights and adjust downstream thresholds?
5. **Kalman fuser:** keep as a per-frame blend (rename it) or implement real temporal filtering?
6. **Artifact location:** where do the canonical `ppe_50ep.pt/.engine`, `stgnn.pth`, `calib.npz` live (LFS, Drive, release assets)?
7. **Scope of the server/desktop-app/SQLite design:** still planned, or dropped? Delete or mark the corresponding diagrams accordingly.
8. **Team workflow:** branch/PR rules actually in force, module owners, where contract changes get announced.
9. Remove `sandbox/` code that's superseded, and fix or delete the broken scripts in `scripts/`.