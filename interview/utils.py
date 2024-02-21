from typing import TypedDict
import torch

class TokeniserOutput(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
