# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

from vllm.v1.outputs import LogprobsTensors


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    prob_token_probs: torch.Tensor | None = None
    logsumexp: torch.Tensor | None = None
    processed_logits: torch.Tensor | None = None
