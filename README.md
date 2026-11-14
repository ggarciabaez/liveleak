# liveleak

## Testing

Install the inference dependencies and `requirements-test.txt` in the same Python
environment. Start the Docker backend using [Docker instructions](docs/DOCKER.md).
Then run all automated checks with a real local image:

```powershell
uv run --no-project python scripts/run_checks.py --smoke-frame .\fixtures\valid.png
```

Replace the example image path with an existing JPEG or PNG. This runs lint, initial
backend type checking, pytest (including WebSocket API tests), and a real-backend
smoke test. Results are written to `evidence/`; failures return a nonzero exit code.
See [the test guide](docs/TESTING.md) for setup, scope, and report evidence, and
complete [the manual checklist](docs/manual-tests.md) with actual Pass/Fail results.

A Multi-tiered, mission-critical, blazing-fast, throughput-boosted vision pipeline to solve this problem: https://www.reddit.com/r/CreateMod/comments/1s08vel/lets_hope_osha_doesnt_see_this_incident/
