# Project TODO

## Model and risk pipeline

- [x] Normalize live ST-GNN inputs, reconcile short tracker-ID breaks using motion/box matching (plus a lightweight appearance cue for people), spatially match cones, and skip motion-scoring windows without persistent moving-object IDs.
- [ ] Fit and provide a motion-error calibrator from held-out normal clips. Until it is fitted, the live model score and fused score remain unavailable (`None`), while the deterministic rule score is still reported.
- [ ] Train and validate the ST-GNN with the dataset pipeline; no trained checkpoint is currently committed.
- [ ] Review and tune DetNG rule weights using representative clips.
- [ ] Profile the Kalman Filter fusion and the end-to-end inference pipeline for bottlenecks.

## CI and demo

- [ ] Extend CI with model-pipeline smoke or integration checks when a suitable deterministic fixture or checkpoint is available. Basic Ruff and CPU unit-test CI already exists.
- [ ] Finalize the frontend/demo flow and decide how the demo will be served.
