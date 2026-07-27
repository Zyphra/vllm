# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_hipify_source_classifier_matches_only_file_suffixes():
    source = REPO_ROOT.joinpath("cmake", "utils.cmake").read_text()

    assert source.count('REGEX "\\\\.(cc|cpp|hip)$"') == 2
