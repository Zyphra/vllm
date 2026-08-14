#!/usr/bin/env python3
"""Run exact-event runtime tests without importing the full vLLM package."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    (
        "vllm.v1.spec_decode.e2etv_event_inputs",
        ROOT / "vllm/v1/spec_decode/e2etv_event_inputs.py",
    ),
    (
        "vllm.v1.spec_decode.e2etv_event_reservoir",
        ROOT / "vllm/v1/spec_decode/e2etv_event_reservoir.py",
    ),
    (
        "vllm.v1.spec_decode.e2etv_runtime",
        ROOT / "vllm/v1/spec_decode/e2etv_runtime.py",
    ),
)
TEST_PATHS = (
    ROOT / "tests/v1/spec_decode/test_e2etv_event_inputs.py",
    ROOT / "tests/v1/spec_decode/test_e2etv_runtime.py",
)


for package_name in ("vllm", "vllm.v1", "vllm.v1.spec_decode"):
    package = types.ModuleType(package_name)
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules.setdefault(package_name, package)

for module_name, module_path in MODULES:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

raise SystemExit(
    pytest.main(
        [
            "-q",
            "--confcutdir",
            str(TEST_PATHS[0].parent),
            *(str(path) for path in TEST_PATHS),
        ]
    )
)
