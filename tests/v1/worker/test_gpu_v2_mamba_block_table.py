# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.block_table import (
    BlockTables,
    get_max_num_blocks_per_req,
)


def test_v2_tidar_mamba_row_reserves_speculative_columns() -> None:
    max_model_len = 33_824
    spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=max_model_len,
        mamba_cache_mode="none",
        num_speculative_blocks=16,
    )
    assert get_max_num_blocks_per_req(
        [spec], max_model_len, enable_prefix_caching=False
    ) == [17]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_tidar_mamba_rows_do_not_overlap() -> None:
    device = torch.device("cuda")
    tables = BlockTables(
        block_sizes=[33_824],
        max_num_blocks_per_req=[17],
        max_num_reqs=2,
        max_num_batched_tokens=34,
        max_model_len=33_824,
        device=device,
    )
    row0 = list(range(17))
    row1 = list(range(100, 117))
    tables.append_block_ids(0, (row0,), overwrite=True)
    tables.append_block_ids(1, (row1,), overwrite=True)
    tables.apply_staged_writes()
    torch.cuda.synchronize()
    assert tables.block_tables[0].gpu.cpu().tolist() == [row0, row1]
    gathered = tables.gather_block_tables(
        torch.tensor([1, 0], dtype=torch.int32, device=device)
    )[0]
    torch.cuda.synchronize()
    assert gathered.cpu().tolist() == [row1, row0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v2_tidar_mamba_row_overflow_fails_before_write() -> None:
    tables = BlockTables(
        block_sizes=[33_824],
        max_num_blocks_per_req=[17],
        max_num_reqs=1,
        max_num_batched_tokens=17,
        max_model_len=33_824,
        device=torch.device("cuda"),
    )
    with pytest.raises(RuntimeError, match="block-table row overflow"):
        tables.append_block_ids(0, (list(range(18)),), overwrite=True)
