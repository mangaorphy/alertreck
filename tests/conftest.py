"""
AlertReck pytest configuration
================================
Shared fixtures and markers.

Markers:
  @pytest.mark.hardware  — requires physical hardware (Pi only, skipped in CI)

Usage:
    pytest tests/                          # all tests, skips hardware
    pytest tests/ -m "not hardware"        # explicitly exclude hardware
    pytest tests/ -m hardware              # only hardware tests (on Pi)
    pytest tests/test_unit.py -v -k "not Model"   # skip ONNX tests if model absent
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: test requires physical hardware (microphone, SIM808). "
        "Run only on the Raspberry Pi with wired components."
    )


def pytest_collection_modifyitems(config, items):
    # Auto-skip hardware tests unless --run-hardware is passed
    if not config.getoption("--run-hardware", default=False):
        skip_hw = pytest.mark.skip(reason="Hardware not available. Use --run-hardware to enable.")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hw)


def pytest_addoption(parser):
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require physical hardware (Pi only).",
    )
