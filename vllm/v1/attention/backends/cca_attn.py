# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import torch

from vllm.v1.attention.backend import AttentionBackend
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (PAD_SLOT_ID,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class CCAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CCA_ATTN"

    @staticmethod
    def get_builder_cls() -> type["CCAAttentionMetadataBuilder"]:
        return CCAAttentionMetadataBuilder


@dataclass
class CCAAttentionMetadata(BaseMambaAttentionMetadata):
    pass

class CCAAttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[CCAAttentionMetadata]):
    metadata_cls = CCAAttentionMetadata
    supports_update_block_table: bool = False
