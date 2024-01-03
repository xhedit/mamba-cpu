# Copyright (c) 2023, Tri Dao.
# Implement residual + layer_norm / rms_norm.

# Based on the Triton LayerNorm tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

import math

import torch
import torch.nn.functional as F
#from torch.cuda.amp import custom_fwd, custom_bwd

#import triton
#import triton.language as tl


def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)


def rms_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


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

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            #residual_in_fp32=residual_in_fp32,
        )
