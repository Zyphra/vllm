# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


def get_max_num_blocks_per_req(
    kv_cache_specs: Sequence[KVCacheSpec],
    max_model_len: int,
    enable_prefix_caching: bool,
) -> list[int]:
    """Return safe V2 block-table row widths for every KV cache group."""
    widths: list[int] = []
    for spec in kv_cache_specs:
        base_blocks = cdiv(max_model_len, spec.block_size)
        max_num_blocks = base_blocks
        if isinstance(spec, MambaSpec):
            # In cache mode none a recurrent layer normally needs one state
            # page, but speculative decoding appends K scratch pages. Mirror
            # V0's established sizing rule so each request owns all 1 + K
            # columns instead of spilling them into adjacent request rows.
            mamba_state_blocks = (
                base_blocks if enable_prefix_caching else 1
            ) + spec.num_speculative_blocks
            max_num_blocks = max(base_blocks, mamba_state_blocks)
        widths.append(max_num_blocks)
    return widths


class BlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_blocks_per_req: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_model_len: int,
        device: torch.device,
    ):
        if len(max_num_blocks_per_req) != len(block_sizes):
            raise ValueError(
                "max_num_blocks_per_req must have one entry per KV cache group; "
                f"got {len(max_num_blocks_per_req)} for {len(block_sizes)} groups"
            )
        self.block_sizes = block_sizes
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_model_len = max_model_len
        self.device = device

        self.num_kv_cache_groups = len(self.block_sizes)
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = self.max_num_blocks_per_req[i]
            if max_num_blocks < cdiv(self.max_model_len, self.block_sizes[i]):
                raise ValueError(
                    f"KV cache group {i} needs at least "
                    f"{cdiv(self.max_model_len, self.block_sizes[i])} block-table "
                    f"columns, got {max_num_blocks}"
                )
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks),
                dtype=torch.int32,
                device=device,
            )
            self.block_tables.append(block_table)
        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            start = self.num_blocks.np[i, req_index] if not overwrite else 0
            block_ids = new_block_ids[i]
            end = start + len(block_ids)
            row_width = self.block_tables[i].gpu.shape[1]
            if end > row_width:
                raise RuntimeError(
                    "KV block-table row overflow: "
                    f"group={i} request_index={req_index} start={start} "
                    f"new_blocks={len(block_ids)} end={end} row_width={row_width}. "
                    "The model runner must reserve speculative Mamba columns."
                )
            self.block_tables[i].stage_write(req_index, start, block_ids)
            self.num_blocks.np[i, req_index] = end

    def apply_staged_writes(self) -> None:
        # TODO(woosuk): This can be inefficient since it launches one kernel per
        # block table. Implement a kernel to handle all block tables at once.
        for block_table in self.block_tables:
            block_table.apply_write()
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self, idx_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        num_reqs = idx_mapping.shape[0]
        # Pass typed tensors directly. Device-side pointer-table indirection can
        # produce misaligned accesses for hybrid cache groups at large batches.
        for group_idx, (src, dst) in enumerate(
            zip(self.block_tables, self.input_block_tables)
        ):
            _gather_block_table_kernel[(num_reqs,)](
                idx_mapping,
                src.gpu,
                dst,
                self.num_blocks.gpu[group_idx],
                src.gpu.stride(0),
                dst.stride(0),
                BLOCK_SIZE=1024,  # type: ignore
            )
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        num_tokens = positions.shape[0]
        # Keep one graph-capturable launch per group for the same reason as the
        # block-table gather above.
        for group_idx, (block_table, block_size) in enumerate(
            zip(self.block_tables, self.block_sizes)
        ):
            _compute_slot_mapping_kernel[(num_reqs + 1,)](
                num_tokens,
                self.max_num_batched_tokens,
                idx_mapping,
                query_start_loc,
                positions,
                block_table.gpu,
                block_table.gpu.stride(0),
                block_size,
                self.slot_mappings[group_idx],
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=1024,  # type: ignore
            )
        return self.slot_mappings[:, :num_tokens]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]


@triton.jit
def _gather_block_table_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table,  # [max_num_reqs, max_num_blocks]
    dst_block_table,  # [max_num_reqs, max_num_blocks]
    num_blocks,  # [max_num_reqs]
    src_stride,
    dst_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    req_num_blocks = tl.load(num_blocks + req_idx)

    src_row_ptr = src_block_table + req_idx * src_stride
    dst_row_ptr = dst_block_table + batch_idx * dst_stride

    for i in tl.range(0, req_num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < req_num_blocks
        block_ids = tl.load(src_row_ptr + offset, mask=mask)
        tl.store(dst_row_ptr + offset, block_ids, mask=mask)


@triton.jit
def _compute_slot_mapping_kernel(
    num_tokens,
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table,  # [max_num_reqs, max_num_blocks]
    block_table_stride,
    block_size,
    slot_mapping,  # [max_num_tokens]
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    if batch_idx == tl.num_programs(0) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        for i in range(num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)
        block_indices = positions // block_size
        block_numbers = tl.load(
            block_table + req_state_idx * block_table_stride + block_indices
        )
        slot_ids = block_numbers * block_size + positions % block_size
        tl.store(slot_mapping + offset, slot_ids, mask=offset < end_idx)
