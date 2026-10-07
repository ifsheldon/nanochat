"""
Unified attention interface with automatic FA3/SDPA selection and opt-in FA4.

Set NANOCHAT_ATTENTION_IMPL=fa4 to require the separately installed FA4 backend.
The default auto selection uses FA3 when available and PyTorch SDPA otherwise.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
from enum import Enum
import importlib
import os
from types import SimpleNamespace
import torch
import torch.nn.functional as F


# =============================================================================
# Backend selection
# =============================================================================
class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FA3 = "fa3"
    FA4 = "fa4"


try:
    _requested_backend = {
        "auto": None,
        "sdpa": AttentionBackend.SDPA,
        "fa4": AttentionBackend.FA4,
    }[os.environ.get("NANOCHAT_ATTENTION_IMPL", "auto")]
except KeyError as exc:
    raise ValueError("NANOCHAT_ATTENTION_IMPL must be 'auto', 'sdpa', or 'fa4'") from exc


def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if _requested_backend is not None or not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # Preserve the existing automatic FA3 discovery. FA4 requires explicit opt-in.
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    except Exception:
        return None


def _load_flash_attention_4():
    """Load an explicitly requested FA4 backend, failing instead of falling back."""
    from nanochat.common import COMPUTE_DTYPE

    if not torch.cuda.is_available():
        raise RuntimeError("NANOCHAT_ATTENTION_IMPL=fa4 requires a CUDA GPU")
    major, minor = torch.cuda.get_device_capability()
    # The pinned FA4 release permits SM8 forward, but its backward entry point
    # accepts only SM9 through SM12. Require a backend suitable for training.
    if major not in (9, 10, 11, 12):
        raise RuntimeError(f"FA4 training does not support CUDA compute capability {major}.{minor}")
    if COMPUTE_DTYPE not in (torch.float16, torch.bfloat16):
        raise RuntimeError("FA4 requires NANOCHAT_DTYPE=bfloat16 or float16")
    try:
        return importlib.import_module("flash_attn.cute")
    except ImportError as exc:
        raise RuntimeError(
            "NANOCHAT_ATTENTION_IMPL=fa4 requires flash-attn-4 and its dependencies; "
            "install the project's fa4 extra"
        ) from exc


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None
_fa4 = _load_flash_attention_4() if _requested_backend is AttentionBackend.FA4 else None


def _resolve_backend():
    """Resolve the configured backend once, before compiling the model."""
    if _requested_backend is not None:
        return _requested_backend
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return AttentionBackend.FA3
    return AttentionBackend.SDPA

ATTENTION_BACKEND = _resolve_backend()


def _fa4_attention(q, k, v, causal, window_size):
    """Adapt FA4's unlimited-window markers and (output, LSE) return contract."""
    left, right = window_size
    # The largest possible distance to an earlier key is Tk - 1, including
    # bottom-right aligned queries during cached inference.
    left = None if left < 0 or left >= k.size(1) - 1 else left
    right = None if right < 0 else right
    output, _lse = _fa4.flash_attn_func(
        q, k, v, causal=causal, window_size=(left, right)
    )
    return output


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if ATTENTION_BACKEND is AttentionBackend.FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if ATTENTION_BACKEND is AttentionBackend.FA4:
        return _fa4_attention(q, k, v, causal, window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    All backends update k_cache/v_cache in-place. FA4 attends to the occupied
    cache using its dense API, whose causal mask is aligned to the last query.
    SDPA and FA4 require uniform cache positions, as used by nanochat's KVCache.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if ATTENTION_BACKEND is AttentionBackend.FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA and FA4 share cache ownership; their attention kernels read only the
    # occupied prefix. Validate before mutating either cache.
    B, T_new, _, _ = q.shape
    if cache_seqlens is None or cache_seqlens.shape != (B,):
        raise ValueError("cache_seqlens must contain one position per batch item")
    pos = int(cache_seqlens[0].item())
    if not torch.all(cache_seqlens == pos).item():
        raise ValueError("SDPA and FA4 require uniform KV cache positions across the batch")
    if (k is None) != (v is None):
        raise ValueError("New KV cache keys and values must be supplied together")
    if k_cache.shape != v_cache.shape or k_cache.size(0) != B:
        raise ValueError("Key and value caches must have matching batch and tensor shapes")
    if k is not None:
        expected_shape = (B, T_new, k_cache.size(2), k_cache.size(3))
        if k.shape != expected_shape or v.shape != expected_shape:
            raise ValueError(f"New KV cache keys and values must have shape {expected_shape}")
    end_pos = pos + (T_new if k is not None else 0)
    if pos < 0 or end_pos > k_cache.size(1):
        raise ValueError("KV cache insertion exceeds the allocated sequence capacity")
    if T_new > end_pos:
        raise ValueError("KV cache must contain at least as many keys as query tokens")

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]
    return flash_attn_func(q, k_full, v_full, causal=causal, window_size=window_size)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
