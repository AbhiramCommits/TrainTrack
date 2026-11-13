"""Run the test suites and write docs/results/test_results.json.

Backend results come from pytest's JUnit XML and coverage JSON; frontend
results come from Vitest's JSON reporter. All figures are measured, never typed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "docs" / "results"
JUNIT = RESULTS / "junit.xml"
COVERAGE = RESULTS / "coverage.json"
FRONTEND_JSON = RESULTS / "frontend_vitest.json"
OUTPUT = RESULTS / "test_results.json"


def run_backend() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--cov=traintrack",
        f"--cov-report=json:{COVERAGE}",
        "--cov-fail-under=80",
        f"--junitxml={JUNIT}",
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit("backend tests failed")

    totals: dict[str, Any] = {}
    if JUNIT.exists():
        tree = ET.parse(JUNIT)
        suite = tree.getroot().find("testsuite") or tree.getroot()
        attributes = suite.attrib
        tests = int(attributes.get("tests", 0))
        failures = int(attributes.get("failures", 0))
        errors = int(attributes.get("errors", 0))
        skipped = int(attributes.get("skipped", 0))
        totals = {
            "tests": tests,
            "passed": tests - failures - errors - skipped,
            "failed": failures + errors,
            "skipped": skipped,
            "duration_seconds": round(float(attributes.get("time", 0)), 3),
        }

    coverage_percent = None
    if COVERAGE.exists():
        coverage = json.loads(COVERAGE.read_text())
        coverage_percent = round(float(coverage["totals"]["percent_covered"]), 2)

    return {
        **totals,
        "coverage_percent": coverage_percent,
        "coverage_scope": "traintrack/ (training entry points, preprocess, and k8s executor excluded)",
        "command": " ".join(command[1:]),
    }


def run_frontend() -> dict[str, Any] | None:
    frontend = PROJECT_ROOT / "frontend"
    if not (frontend / "node_modules").exists():
        return None
    if FRONTEND_JSON.exists():
        FRONTEND_JSON.unlink()
    completed = subprocess.run(
        [
            "npm",
            "test",
            "--",
            "--reporter=json",
            f"--outputFile={FRONTEND_JSON}",
        ],
        cwd=frontend,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit("frontend tests failed")
    if not FRONTEND_JSON.exists():
        return None
    report = json.loads(FRONTEND_JSON.read_text())
    start_time = report.get("startTime")
    end_times = [
        result.get("endTime", 0)
        for result in report.get("testResults", [])
        if isinstance(result, dict)
    ]
    duration = None
    if isinstance(start_time, int | float) and end_times:
        duration = round((max(end_times) - start_time) / 1000, 3)
    return {
        "tests": int(report.get("numTotalTests", 0)),
        "passed": int(report.get("numPassedTests", 0)),
        "failed": int(report.get("numFailedTests", 0)),
        "suites": int(report.get("numTotalTestSuites", 0)),
        "duration_seconds": duration,
        "command": "npm test",
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    backend = run_backend()
    frontend = run_frontend()
    totals = {
        "tests": backend.get("tests", 0) + (frontend["tests"] if frontend else 0),
        "passed": backend.get("passed", 0) + (frontend["passed"] if frontend else 0),
        "failed": backend.get("failed", 0) + (frontend["failed"] if frontend else 0),
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/export_test_results.py",
        "backend": backend,
        "frontend": frontend,
        "totals": totals,
        "backend_coverage_file": "docs/results/coverage.json",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    for temporary in (JUNIT, FRONTEND_JSON):
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
