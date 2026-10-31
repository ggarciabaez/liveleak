# Deployment & Rollout

This checklist tracks the deployment work that remains. The repository already has a baseline GitHub Actions workflow in `.github/workflows/ci.yml`: it runs Ruff on pull requests and CPU-based pytest with coverage on pull requests and pushes to `main`. That workflow does not yet type-check the project or exercise a complete live model pipeline.

## Checklist

- [ ] Configure repository branch protection and required peer review before merge.
- [x] Add baseline CI for Ruff and CPU unit tests with coverage.
- [ ] Add type checking and integration tests for the end-to-end pipeline.
- [ ] Add a model-pipeline smoke test when a deterministic fixture or trained checkpoint is available.
- [ ] Containerize the detector, tracker, risk fusion, and supporting services.
- [ ] Choose a deployment target and deploy the demo.
- [x] Keep the Mermaid design diagrams in `docs/diagrams/` as living project documentation.
- [ ] Train and validate the ST-GNN, then connect its calibrated risk score to live inference.

## Open questions

- Which hosting target should the team use?
- Should the demo serve live inference or a pre-recorded run?
- What model fixture can CI use without committing large video data or model weights?
