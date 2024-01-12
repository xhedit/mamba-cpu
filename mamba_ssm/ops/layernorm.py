# Copyright (c) 2023, Tri Dao.
# Implement residual + rms_norm.

import math
import torch
import torch.nn.functional as F

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None):
        weight = self.weight
        bias = self.bias
        eps = self.eps
        if residual is not None:
            x = (x + residual).to(x.dtype)
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
        out = out.to(x.dtype)
        return out
