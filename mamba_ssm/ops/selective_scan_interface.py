# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def selective_scan(u, delta, A, B, C, D=None, z=None):
    """
    u: r(D L)
    delta: r(D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(N L) or r(N 2L) or r(G N L) or (G N L)
    C: c(D N) or r(N L) or r(N 2L) or r(G N L) or (G N L)
    D: r(D)
    z: r(D L)
    delta_bias: r(D), fp32

    out: r(D L)
    last_state (optional): r(D dstate) or c(D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    B = B.float()
    C = C.float()
    delta = delta.float()

    seqlen, dim, dstate = u.shape[0], A.shape[0], A.shape[1]

    x = A.new_zeros((dim, dstate))

    deltaA = torch.exp(torch.einsum('dl,dn->dln', delta, A))
    deltaB_u = torch.einsum('dl,ln,ld->dln', delta, B, u)

    ys = []
    for i in range(seqlen):
        x = deltaA[:, i] * x + deltaB_u[:, i]
        y = torch.einsum('dn,n->d', x, C[i])
        #if y.is_complex():
        #    y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=0) # (l d)

    out = y + u * D
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return (out, x)
