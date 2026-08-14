#!/usr/bin/env python3
"""Run the focused reservoir tests without importing the full vLLM package.

The production test still imports the canonical module path.  This runner is
only for minimal CUDA-container qualification where optional vLLM test
dependencies are intentionally absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "vllm.v1.spec_decode.e2etv_event_reservoir"
MODULE_PATH = ROOT / "vllm/v1/spec_decode/e2etv_event_reservoir.py"
TEST_PATH = ROOT / "tests/v1/spec_decode/test_e2etv_event_reservoir.py"


for package_name in ("vllm", "vllm.v1", "vllm.v1.spec_decode"):
    package = types.ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(package_name, package)

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {MODULE_PATH}")
module = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = module
spec.loader.exec_module(module)

raise SystemExit(
    pytest.main(
        [
            "-q",
            "--confcutdir",
            str(TEST_PATH.parent),
            str(TEST_PATH),
        ]
    )
)
