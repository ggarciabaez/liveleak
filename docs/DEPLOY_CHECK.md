# Deployment & Rollout — Stub

Split out from the main handbook so it doesn't compete with the ST-GNN work for attention. Light on detail for now — flesh out once the core pipeline (Parts 1–3 of the handbook) is further along.

## From the class rubric
- [ ] Peer-reviewed PRs — branch protection + required review before merge
- [ ] CI pipeline — linter/type-checker + unit & integration tests on every PR
- [ ] Docker containerization of the pipeline (detector + tracker + forks + fusion)
- [ ] Public cloud deployment
- [ ] Mermaid.js diagrams as living documentation in the repo
- [ ] At least one AI feature — covered by the ST-GNN fork

## Open questions (not yet decided)
- [ ] Which cloud target
- [ ] What the CI test suite actually covers for this project (schema contract tests? rule-engine unit tests? GNN smoke test?)
- [ ] How the demo gets served (live inference endpoint vs. pre-recorded run)