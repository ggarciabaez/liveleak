"""One command for static checks, existing tests, API tests and live smoke."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-frame", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frame = args.smoke_frame.resolve()
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    commands = [
        ("lint", ["ruff", "check", "tests", "app", "scripts/run_checks.py", "scripts/smoke_backend.py"]),
        ("types", ["mypy", "--config-file", "mypy.ini", "app/app.py"]),
        ("pytest", ["pytest", "--cov=modules", "--cov=app", "--cov-report=term-missing",
                    "--cov-report=xml:evidence/coverage.xml", "--junitxml=evidence/tests.xml"]),
        ("smoke", ["scripts/smoke_backend.py", "--frame", str(frame)]),
    ]
    failed = []
    for name, arguments in commands:
        command = [sys.executable] + ([] if name == "smoke" else ["-m"]) + arguments
        result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        output = result.stdout or ""
        (evidence / f"{name}.txt").write_text(output, encoding="utf-8")
        print(output)
        if result.returncode:
            failed.append(name)
    print("Failed checks: " + ", ".join(failed) if failed else "All automated checks passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
