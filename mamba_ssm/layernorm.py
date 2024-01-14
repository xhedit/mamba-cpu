# Copyright (c) 2023, Tri Dao.
# Implement rms_norm

import torch

class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))

    def forward(self, x):
        rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * rstd * self.weight
