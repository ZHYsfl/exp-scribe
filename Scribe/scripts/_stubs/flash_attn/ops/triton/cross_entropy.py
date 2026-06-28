"""Make ``from flash_attn.ops.triton.cross_entropy import cross_entropy_loss``
fail with ImportError.

verl's ``torch_functional.py`` does that import inside a try/except to set
``FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE``.  We have no real flash-attn triton
kernels (stub package), so we want that flag to be False — then verl computes
log-probs with its pure-PyTorch ``logprobs_from_logits_v2`` fallback instead of
calling a kernel that doesn't exist.  Raising ImportError here is the signal
verl's except clause is waiting for.

(Other flash_attn submodules like ``bert_padding`` stay importable so verl's
*module-load* imports still succeed; those helpers are only ever *called* when
use_remove_padding=True, which we keep False.)
"""
raise ImportError("flash_attn cross_entropy kernels are stubbed out; use the pure-torch fallback")
