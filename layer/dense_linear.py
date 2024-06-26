import math
from torch import Tensor, nn
from torch.autograd import Function
import torch

import custom_linear


class DenseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    # we only override the forward to map to our custom kernel
    # backward will fallback to nn.Linear.backward()

    def forward(self, input: Tensor) -> Tensor:
        print("[Info]: Entering custom linear implementation")
        return custom_linear.dense_linear(input, self.weight, self.bias)

    @classmethod
    def from_linear(cls, nn_linear_inst: nn.Linear):
        # TODO: not an efficient implementation, we will create another copy of parameters, imagine Billion scale llm!
        # see llm_pipeline.py for a "Wrapper" way
        new_inst = cls(nn_linear_inst.in_features, 
                       nn_linear_inst.out_features, 
                       nn_linear_inst.bias is not None, 
                       next(nn_linear_inst.parameters()).device, 
                       next(nn_linear_inst.parameters()).dtype)
        
        new_inst.weight.data = nn_linear_inst.weight.data
        if new_inst.bias is not None:
            new_inst.bias.data = nn_linear_inst.bias.data
        return new_inst