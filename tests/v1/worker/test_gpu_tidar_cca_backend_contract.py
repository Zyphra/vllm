# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.smoe import _validate_tidar_cca_backend


class _Spec:
    def __init__(self, use_tidar: bool) -> None:
        self._use_tidar = use_tidar

    def use_tidar(self) -> bool:
        return self._use_tidar


def _config(spec) -> SimpleNamespace:
    return SimpleNamespace(speculative_config=spec)


def test_tidar_rejects_native_cca_backend() -> None:
    with pytest.raises(RuntimeError, match="VLLM_CCA_TRITON=1"):
        _validate_tidar_cca_backend(
            _config(_Spec(True)),
            use_triton=False,
        )


@pytest.mark.parametrize("spec", [_Spec(True)])
def test_tidar_accepts_triton_cca_backend(spec) -> None:
    _validate_tidar_cca_backend(
        _config(spec),
        use_triton=True,
    )


@pytest.mark.parametrize("spec", [None, _Spec(False)])
def test_non_tidar_accepts_native_cca_backend(spec) -> None:
    _validate_tidar_cca_backend(
        _config(spec),
        use_triton=False,
    )
