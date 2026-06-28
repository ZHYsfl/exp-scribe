"""Stub flash_attn package used when no flash-attn wheel exists for the
torch/cu combo we're running on.

The verl trainer imports flash_attn at module-load time even when
``use_remove_padding=False`` (the actual unpad/pad helpers are then never
called).  We only need the import lines to succeed.  Any attempt to USE
these helpers will raise loudly so we notice if a code path slips back
into the flash_attn-required branch.
"""
def _missing(*args, **kwargs):  # pragma: no cover
    raise RuntimeError(
        "flash_attn is stubbed (no wheel installed); set "
        "actor_rollout_ref.model.use_remove_padding=False and avoid "
        "code paths that call flash_attn functions."
    )

flash_attn_func = _missing
flash_attn_varlen_func = _missing
