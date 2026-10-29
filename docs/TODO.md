# Project TODO

## Model and risk pipeline

- [ ] Connect the ST-GNN to live `RiskDetector` inference using track-aligned feature history, forecast scoring, and a calibrated risk score before fusion.
- [ ] Train and validate the ST-GNN with the dataset pipeline; no trained checkpoint is currently committed.
- [ ] Review and tune DetNG rule weights using representative clips.
- [ ] Profile the Kalman Filter fusion and the end-to-end inference pipeline for bottlenecks.

## CI and demo

- [ ] Extend CI with model-pipeline smoke or integration checks when a suitable deterministic fixture or checkpoint is available. Basic Ruff and CPU unit-test CI already exists.
- [ ] Finalize the frontend/demo flow and decide how the demo will be served.
