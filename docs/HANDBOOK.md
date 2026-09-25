# Workplace Risk Detection — Project Handbook

---

## Part 1 — Components (What / How / Why)

### 1. Detector — YOLO26m (finetuned)
- **What:** Finetuned YOLO26m detecting Person, Hardhat, Mask, Safety Vest (each with a NO-* counterpart), machinery, vehicles, safety cones.
- **How:** Runs per-frame, outputs bounding boxes + class + confidence.
- **Why:** A single-stage detector gives PPE state directly as a class label instead of needing a second classification pass — cheaper, and it's already built.
- **Status:** Done.

### 2. Tracker — currently SORT, evaluate BoT-SORT / ByteTrack
- **What:** Assigns persistent IDs to detections across frames.
- **How:** One correction worth flagging — Ultralytics doesn't ship literal DeepSORT as a built-in tracker. Its native options are **BoT-SORT** (default, optional ReID + camera-motion compensation) and **ByteTrack** (faster, no ReID), both reachable directly via `model.track(tracker="botsort.yaml")` or `"bytetrack.yaml"`. What you were thinking of is BoT-SORT, not DeepSORT — it's the closest built-in equivalent (appearance-based re-ID).
- **Why it matters here specifically:** the ST-GNN's forecasting objective needs stable IDs across a whole window — an ID switch mid-sequence corrupts a training example (looks like a random teleport to the model). BoT-SORT's ReID buys resistance to that in crowded/occluded scenes at some compute cost; ByteTrack is fine if scenes aren't badly crowded. Worth a quick benchmark, not a rewrite — it's a one-line swap.

### 3. Info Vector Builder
- **What:** Converts each frame's tracked detections into the node feature schema (ID, one-hot type, velocity, one-hot PPE).
- **How:** Velocity from centroid delta between consecutive frames of the same ID. PPE one-hot is only meaningful for Person nodes — needs an explicit "not applicable" convention for machinery/vehicle/cone nodes so the GNN doesn't read "no PPE fields" as "unsafe machinery."
- **Why:** This is the seam between detection and both forks — get the NA/encoding convention wrong here and the bug propagates silently into the rule engine and the GNN both.

### 4. Graph Builder
- **What:** Turns each frame's info vectors into a spatial graph, and chains frames into a spatio-temporal window.
- **How:** Spatial edges via k-NN or distance threshold (optionally inverse-distance weighted); temporal edges link same-ID nodes across consecutive frames within a sliding window (e.g. 1–2s).
- **Why:** This is what makes "person near moving vehicle" a structural graph feature instead of something each fork has to re-derive from raw coordinates independently.

### 5. Rule-Based Fork
- **What:** Deterministic scorer — explicit thresholds (proximity to vehicle without hardhat, speed near machinery, PPE non-compliance).
- **How:** Hand-written rules over info vectors + graph edges, output normalized to [0,1].
- **Why:** Interpretable, auditable, and independent of any learned model — a decorrelated second signal for fusion, and your only source of "risk" ground truth to sanity-check the GNN against, since there's no real incident data.

### 6. ST-GNN Fork (self-supervised)
- **What:** Learns typical spatiotemporal dynamics from incident-free footage, flags deviation from that as risk.
- **How:** Covered in Part 2.
- **Why:** OSHA/ethics rules out training on labeled injury data, so it can't be a classifier — a forecasting/anomaly framing sidesteps that entirely by never needing a positive class.

### 7. Kalman Filter Fusion
- **What:** Combines the rule score and GNN probability into one final risk estimate over time.
- **How:** Treat both scores as noisy measurements of one hidden "true risk" state; per-source noise covariances.
- **Why:** Smooths frame-to-frame noise in both signals and lets the fused estimate weight the more reliable source, instead of a fixed static average.

### 8. Demo / Visualization Layer
- **What:** Renders detections, IDs, PPE compliance, and (optionally) the fused risk score.
- **How:** Overlay boxes/IDs/PPE status on video.
- **Why:** The interesting thing to demo is the pipeline correctly flagging staged non-compliance — not a simulated worst case. That's also a *better* demo: it shows the system catching the precursor, which is the actual point of a risk detector, without needing to fabricate or depict an injury.

---

## Part 2 — Architecture: Building, Training, Reasoning

### A. Detection & Tracking
Already functional. Only open item: benchmark BoT-SORT and ByteTrack against the current SORT setup on a representative clip, watching specifically for ID switches during occlusion (person walking behind machinery) — that's the failure mode that will hurt everything downstream.

### B. Info Vector / Graph Contract
Define once, in code, as the shared interface both forks consume:

```
Node:
  id: int
  type: onehot[person, machinery, vehicle, cone]
  velocity: [vx, vy]            (or speed, heading)
  ppe: onehot[hardhat, mask, vest] | NA   (person-only; NA for all other types)
  position: [x, y]              (OPTIONAL on the node — see note)

Edge (per frame):
  (node_i, node_j, dx, dy, distance) for j in k-NN(i) or within distance_threshold

Temporal link:
  node id=k at frame t  ->  node id=k at frame t+1
```

**Note on position (revised):** "how close are two nodes" should be carried as an **edge feature** (dx, dy, distance for every connected pair), not just used to decide which edges exist. That's the actual fix — closeness needs to be something the model can read a magnitude/direction from, not just a binary "there's an edge here." Edge-relative position also keeps the model translation-invariant: it reasons about relationships between entities, not where they happen to sit in a specific camera's frame, which matters if the model ever needs to generalize across sites or a repositioned camera. Absolute (x, y) as a *node* feature is only worth adding on top if the deployment camera is fixed and won't move between training and inference (true for most single-site safety cameras) — otherwise skip it and let exclusion-zone logic live in the rule-based fork, which can hard-code zone coordinates for a specific fixed camera without that assumption leaking into the learned model.

Lock this down early — it's the contract the rule engine, the GNN, and eventually the fusion stage all depend on. Changing it later means retraining the GNN and rewriting the rule engine both.

### C. Rule-Based Fork
Start simple: a handful of weighted conditions (unprotected-person-near-vehicle distance, exclusion-zone entry, speed-vs-proximity product) summed and squashed to [0,1]. Doesn't need heavy design up front — its main job besides scoring is giving a second opinion to compare the GNN against during development.

### D. ST-GNN Fork — training design
The part needing the most care:

1. **Objective — multi-step forecasting.** Given a window of T past frames' graphs, predict node states (position, velocity) K steps ahead. Loss = masked Smooth-L1/MSE between predicted and actual future states, trained only on incident-free footage.
2. **Encoder.** Per-frame spatial GNN (GraphSAGE or GAT over the k-NN graph) → temporal aggregator (GRU or 1D temporal conv over the window) → decoder head predicting the next K frames' node states.
3. **Data requirement.** The "normal" footage needs to be *varied* normal — different site densities, different PPE-compliance mixes, different machinery/vehicle behavior — or the model flags benign-but-rare situations as anomalous just because they're underrepresented, not because they're risky.
4. **Calibration.** Raw prediction error isn't a probability. Fit the error distribution on a held-out slice of normal validation data, map new errors to [0,1] via empirical CDF or a fitted tail (Gaussian/EVT) + sigmoid. Keep this a separate step from training so it can be recalibrated per-site without retraining.
5. **Optional second signal.** A Deep SVDD-style one-class loss on the window embedding (pull normal embeddings toward a learned center) — ensemble its distance-from-center score with the forecasting error for a more robust anomaly signal.

### E. Fusion — Kalman Filter
- **State:** scalar (or low-dim) "true risk level."
- **Measurements:** [rule_score, gnn_probability], each frame.
- **Measurement noise (R):** start as fixed per-source constants tuned on validation data (how much each source jitters frame-to-frame on footage you'd call "stable"); consider scaling the GNN's R dynamically by its own calibrated uncertainty later.
- **Process noise (Q):** how fast true risk is allowed to change frame-to-frame — small for a smooth output, larger for a fused score that reacts fast to a sudden event.
- Treat Q/R as empirically tuned against staged scenarios, not analytically derived — there's no ground-truth statistics to derive them from, and that's fine.

### F. Demo Strategy
Build the demo around *precursor detection*, not incident severity. Run the full pipeline on live or recorded footage with a couple of staged non-compliance moments (someone briefly removes a hardhat, walks near a stationary vehicle without a vest) and show the fused score rising in response. A complete, honest demo of the system doing its job — nothing to fabricate or dramatize.

---

## Part 3 — Checklist

**Detection & Tracking**
- [x] YOLO26m finetuned detector (Person, Hardhat/Mask/Vest ± NO-*, machinery, vehicle, cone)
- [x] Baseline tracking with SORT
- [ ] Benchmark BoT-SORT and ByteTrack via `model.track()` against current SORT, especially on occlusion/crowding

**Info Vector & Graph**
- [ ] Lock the node/edge schema (Part 2B) as a shared contract
- [ ] Build detection+track → info-vector conversion layer, including the person-only PPE NA convention
- [ ] Build spatial graph construction (k-NN/threshold) + sliding temporal window

**Rule-Based Fork**
- [ ] Define initial rule set and weights
- [ ] Normalize output to [0,1]

**ST-GNN Fork**
- [ ] Collect/curate incident-free training footage — varied site density, PPE mixes, machinery behavior
- [ ] Build spatial encoder + temporal aggregator + forecasting head
- [ ] Train self-supervised forecasting objective
- [ ] Calibrate prediction error → risk probability on held-out normal data
- [ ] (optional) add Deep SVDD-style secondary anomaly signal

**Fusion**
- [ ] Implement Kalman filter (state/measurement model)
- [ ] Tune Q/R against staged validation scenarios

**Evaluation (no real incident data available)**
- [ ] Build staged near-miss / non-compliance test clips, held out from training
- [ ] Build synthetic anomaly injection tests (perturbed trajectories)
- [ ] Sanity-check rule-engine vs. GNN agreement

**Demo**
- [ ] Build visualization overlay (boxes, IDs, PPE status, fused score)
- [ ] Script 2–3 staged non-compliance moments for the live demo

*Deployment & rollout (CI/CD, Docker, cloud deploy, PR review) tracked separately — see `workplace-risk-detection-deployment.md`.*