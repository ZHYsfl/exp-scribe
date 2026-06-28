"""Pure-PyTorch ``apply_rotary`` for the flash_attn stub.

vLLM's ``ApplyRotaryEmb`` layer probes ``find_spec("flash_attn")`` and, when it
succeeds, does::

    from flash_attn.ops.triton.rotary import apply_rotary

then calls ``apply_rotary(x, cos, sin, interleaved=...)`` on every forward.
Because we ship a *stub* flash_attn (no triton kernels), that import would
normally fail and crash vLLM model construction.  We can't make flash_attn
un-importable (verl imports ``flash_attn.bert_padding`` at module load), so
instead we provide a correct PyTorch implementation here.

It is numerically validated against vLLM's own ``ApplyRotaryEmb.forward_static``
in scripts (see the run script's preflight check).  Semantics mirror the real
flash_attn triton kernel for the arguments vLLM actually passes:

    x:        [..., seqlen, nheads, headdim]
    cos/sin:  [seqlen, rotary_dim // 2]   (rotary_dim may be < headdim)
    interleaved: False -> NeoX style (split-half); True -> GPT-J (even/odd)

The trailing ``headdim - rotary_dim`` channels are passed through unrotated,
matching flash_attn's partial-rotary behaviour.
"""
from __future__ import annotations

import torch


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets=0,
    cu_seqlens=None,
    max_seqlen=None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    # vLLM never uses the varlen / offset path here; guard so a future caller
    # that does isn't silently mis-rotated.
    if cu_seqlens is not None or (isinstance(seqlen_offsets, int) and seqlen_offsets != 0):
        raise NotImplementedError(
            "flash_attn stub apply_rotary: varlen/seqlen_offsets path is not "
            "implemented (vLLM does not use it)."
        )

    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1], (rotary_dim, x.shape)

    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    if conjugate:
        sin = -sin
    # cos/sin: [seqlen, rotary_dim//2] -> [seqlen, 1, rotary_dim//2] so they
    # broadcast across the nheads axis (and any leading batch axis) of x.
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    if not interleaved:  # NeoX: first half / second half
        x1, x2 = x_rot.chunk(2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out_rot = torch.cat((o1, o2), dim=-1)
    else:  # GPT-J: even / odd interleaved
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        out_rot = torch.stack((o1, o2), dim=-1).flatten(-2)

    if x_pass.shape[-1] == 0:
        return out_rot
    return torch.cat((out_rot, x_pass), dim=-1)
