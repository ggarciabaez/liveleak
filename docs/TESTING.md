# Milestone 4 testing guide

## Setup

Keep the existing component tests. Install requirements-test.txt into the same
Python environment as the inference dependencies:

```powershell
uv pip install -r requirements-test.txt
```

Do not use project synchronization to install all GPU dependencies on a CPU-only
machine. The existing GitHub Actions workflow shows the CPU dependency subset.

## Run everything

First start the real Docker backend as described in docs/DOCKER.md, with matching
YOLO weights, ST-GNN checkpoint and fitted calibration artifact. Keep it running.
Choose a valid local JPEG or PNG fixture. From the repository root:

```powershell
uv run --no-project python scripts/run_checks.py --smoke-frame .\fixtures\valid.png
```

Replace the example path with your real image; no fixture is supplied by this change.
The command runs Ruff, mypy, all pytest tests, and a live-backend smoke test.
It returns nonzero if any stage fails and saves evidence/*.txt, tests.xml,
coverage.xml, and smoke.json. A missing backend or artifact is a failure, not a skip.
The backend must be bound to local port 9160. Never unpickle an untrusted server's output.

## Scope and limitations

- Ruff checks tests/, app/, and the two new check/smoke scripts.
- Mypy checks app/app.py, including untyped function bodies, with imported module
  internals skipped. It is an initial backend check, not full-project type checking.
- Existing tests cover components and cross-module contracts.
- tests/test_websocket_api.py contains five parameterized API cases: binary/base64
  happy paths and three invalid-input cases. It uses real local WebSocket transport
  and a fake detector; it does not prove model accuracy or artifact loading.
- scripts/smoke_backend.py contacts the separately running real backend, validates
  responses, sends invalid input, and verifies continued operation. Unavailable
  learned scoring with an explicit reason is allowed. Repeating one image does not
  validate motion behavior, UI, database persistence, or NFR latency.
- Complete docs/manual-tests.md with real observations. Unimplemented scenarios
  must not be marked Pass.

## Counts and report evidence

Use the executed pytest/JUnit output for counts; source functions are not case counts.
Report the five API cases separately. Classify pipeline_test.py as cross-module
integration tests; remaining existing cases are component tests (review individual
tests before labeling all of them unit tests). Include actual pass/fail/skip totals.
Attach lint.txt, types.txt, pytest.txt and smoke.json or screenshots showing their
results. Do not write 'all passing' until all four stages exit successfully.

Full NFR acceptance is separate: NFR-11 is p95 <= 100 ms on the declared GPU server,
excluding network transit. Authorization, latest-frame processing, camera recovery,
usability and persistence need their own implemented behavior and acceptance evidence.

Every new feature and its tests should be submitted in the same reviewed PR.
