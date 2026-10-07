"""CPU tests of FA4 adaptation and cache contracts; kernel validation is separate."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import nanochat.common as common
import nanochat.flash_attention as attention


@pytest.fixture
def fa4_backend(monkeypatch):
    """Model FA4's tuple return, None bounds and bottom-right causal alignment."""
    calls = []

    def flash_attn_func(q, k, v, *, causal, window_size):
        left, right = window_size
        assert left is None or left >= 0, "FA4 requires None for unlimited left context"
        assert right is None or right >= 0, "FA4 requires None for unlimited right context"
        calls.append((q.shape[1], k.shape[1], window_size))
        row = torch.arange(q.shape[1])[:, None] + k.shape[1] - q.shape[1]
        col = torch.arange(k.shape[1])[None, :]
        mask = torch.ones(q.shape[1], k.shape[1], dtype=torch.bool)
        if causal:
            mask &= col <= row
        if left is not None:
            mask &= col >= row - left
        if right is not None:
            mask &= col <= row + right
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, enable_gqa=q.shape[2] != k.shape[2],
        ).transpose(1, 2)
        return out, None

    monkeypatch.setattr(attention, "_fa4", SimpleNamespace(flash_attn_func=flash_attn_func))
    monkeypatch.setattr(attention, "ATTENTION_BACKEND", attention.AttentionBackend.FA4)
    return calls


def reference(q, k, v, window):
    """Independent full-sequence causal reference in the public wrapper's layout."""
    scores = torch.einsum("bthd,bshd->bhts", q, k) / q.shape[-1] ** 0.5
    pos = torch.arange(q.shape[1])
    distance = pos[:, None] - pos[None, :]
    mask = distance >= 0
    if window >= 0:
        mask &= distance <= window
    scores = scores.masked_fill(~mask, -torch.inf)
    return torch.einsum("bhts,bshd->bthd", scores.softmax(dim=-1), v)


@pytest.mark.parametrize("window", [(-1, -1), (-1, 0), (9, 0), (3, 0)])
def test_training_output_and_gradients(fa4_backend, window):
    torch.manual_seed(17)
    inputs = [torch.randn(2, 9, 2, 8, requires_grad=True) for _ in range(3)]
    dout = torch.randn_like(inputs[0])
    expected = reference(*inputs, window[0])
    expected_grads = torch.autograd.grad(expected, inputs, dout)
    actual = attention.flash_attn_func(*inputs, causal=True, window_size=window)
    actual_grads = torch.autograd.grad(actual, inputs, dout)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("window", [-1, 8, 2])
def test_cached_prefill_chunk_and_decode_match_full_attention(fa4_backend, window):
    torch.manual_seed(29)
    q, k, v = [torch.randn(2, 8, 2, 8) for _ in range(3)]
    expected = reference(q, k, v, window)
    # NaNs in unused capacity reveal any accidental attention to future cache slots.
    k_cache, v_cache = [torch.full((2, 12, 2, 8), torch.nan) for _ in range(2)]
    positions = torch.zeros(2, dtype=torch.int32)
    outputs = []
    start = 0
    for length in (3, 2, 1, 2):
        end = start + length
        outputs.append(attention.flash_attn_with_kvcache(
            q[:, start:end], k_cache, v_cache,
            k=k[:, start:end], v=v[:, start:end], cache_seqlens=positions,
            causal=True, window_size=(window, 0),
        ))
        # The model, not the attention backend, owns cache-position advancement.
        torch.testing.assert_close(positions, torch.full_like(positions, start))
        positions.add_(length)
        start = end
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    torch.testing.assert_close(k_cache[:, :8], k)
    torch.testing.assert_close(v_cache[:, :8], v)
    assert k_cache[:, 8:].isnan().all() and v_cache[:, 8:].isnan().all()
    # Querying existing cache contents must not append or expose unused slots.
    repeated = attention.flash_attn_with_kvcache(
        q[:, -1:], k_cache, v_cache, cache_seqlens=positions,
        causal=True, window_size=(window, 0),
    )
    torch.testing.assert_close(repeated, expected[:, -1:])


@pytest.mark.parametrize("positions,error", [
    ([0, 1], "uniform"),
    ([-1, -1], "capacity"),
    ([7, 7], "capacity"),
])
def test_invalid_cache_positions_do_not_mutate_cache(fa4_backend, positions, error):
    q, k, v = [torch.randn(2, 2, 2, 8) for _ in range(3)]
    k_cache, v_cache = [torch.randn(2, 8, 2, 8) for _ in range(2)]
    originals = [k_cache.clone(), v_cache.clone()]
    with pytest.raises(ValueError, match=error):
        attention.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=torch.tensor(positions, dtype=torch.int32), causal=True,
        )
    torch.testing.assert_close(k_cache, originals[0])
    torch.testing.assert_close(v_cache, originals[1])
    assert not fa4_backend


def test_explicit_fa4_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        attention._load_flash_attention_4()


def test_explicit_fa4_rejects_float32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(common, "COMPUTE_DTYPE", torch.float32)
    with pytest.raises(RuntimeError, match="requires NANOCHAT_DTYPE"):
        attention._load_flash_attention_4()


def test_explicit_fa4_rejects_forward_only_architecture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(common, "COMPUTE_DTYPE", torch.bfloat16)
    with pytest.raises(RuntimeError, match="FA4 training does not support.*8.0"):
        attention._load_flash_attention_4()


def test_explicit_fa4_missing_dependency_does_not_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(common, "COMPUTE_DTYPE", torch.bfloat16)

    def missing_dependency(name):
        raise ModuleNotFoundError("Missing FA4 dependency")

    monkeypatch.setattr(attention.importlib, "import_module", missing_dependency)
    with pytest.raises(RuntimeError, match="requires flash-attn-4") as error:
        attention._load_flash_attention_4()
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_fa4_kernel_errors_do_not_fall_back(fa4_backend, monkeypatch):
    def unsupported_kernel(*args, **kwargs):
        raise RuntimeError("Unsupported FA4 kernel shape")

    monkeypatch.setattr(attention._fa4, "flash_attn_func", unsupported_kernel)
    q = torch.randn(1, 4, 2, 8)
    with pytest.raises(RuntimeError, match="Unsupported FA4 kernel shape"):
        attention.flash_attn_func(q, q, q, causal=True)


@pytest.fixture
def real_fa4_backend(monkeypatch):
    """Opt into the real backend for these tests without changing default dispatch."""
    if not torch.cuda.is_available():
        pytest.skip("FA4 kernel tests require CUDA")
    if torch.cuda.get_device_capability()[0] not in (9, 10, 11, 12):
        pytest.skip("The pinned FA4 release has no backward support for this GPU")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("FA4 kernel tests require BF16 support")
    try:
        module = attention.importlib.import_module("flash_attn.cute")
    except ImportError as exc:
        pytest.skip(f"Optional FA4 dependency is unavailable: {exc}")
    monkeypatch.setattr(attention, "_fa4", module)
    monkeypatch.setattr(attention, "ATTENTION_BACKEND", attention.AttentionBackend.FA4)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def sdpa_causal_reference(q, k, v, window):
    """Equal-length reference for real kernels, using full or explicit local masking."""
    mask = None
    if 0 <= window < q.shape[1] - 1:
        pos = torch.arange(q.shape[1], device=q.device)
        distance = pos[:, None] - pos[None, :]
        mask = (distance >= 0) & (distance <= window)
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=mask is None,
    ).transpose(1, 2)


@pytest.mark.parametrize("window", [256, 32])
def test_real_fa4_output_and_backward_against_fp32(real_fa4_backend, window):
    generator = torch.Generator(device="cuda").manual_seed(37)
    inputs = [torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16,
                          generator=generator, requires_grad=True) for _ in range(3)]
    dout = torch.randn(inputs[0].shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    ref_inputs = [x.detach().float().requires_grad_(True) for x in inputs]
    with sdpa_kernel(SDPBackend.MATH):
        expected = sdpa_causal_reference(*ref_inputs, window)
        expected_grads = torch.autograd.grad(expected, ref_inputs, dout.float())
    baseline = sdpa_causal_reference(*inputs, window)
    baseline_grads = torch.autograd.grad(baseline, inputs, dout)
    actual = attention.flash_attn_func(*inputs, causal=True, window_size=(window, 0))
    actual_grads = torch.autograd.grad(actual, inputs, dout)
    for name, got, ref, base in zip(
        ("output", "dq", "dk", "dv"),
        (actual, *actual_grads), (expected, *expected_grads), (baseline, *baseline_grads),
    ):
        assert torch.isfinite(got).all(), f"Nonfinite FA4 {name}"
        error = got.float() - ref
        baseline_error = base.float() - ref
        relative_l2 = error.norm() / ref.norm().clamp_min(1e-12)
        assert relative_l2.item() < 0.01, f"FA4 {name} exceeds 1% relative L2 error"
        assert error.abs().max().item() <= 2 * baseline_error.abs().max().item() + 1e-3, (
            f"FA4 {name} exceeds twice the BF16 SDPA maximum error"
        )


@pytest.mark.parametrize("window", [-1, 2048, 16])
@torch.no_grad()
def test_real_fa4_cached_attention_matches_sdpa(real_fa4_backend, window):
    generator = torch.Generator(device="cuda").manual_seed(41)
    q, k, v = [torch.randn(1, 48, 2, 128, device="cuda", dtype=torch.bfloat16,
                          generator=generator) for _ in range(3)]
    expected = sdpa_causal_reference(q, k, v, window)
    k_cache, v_cache = [torch.full((1, 64, 2, 128), torch.nan,
                                  device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    positions = torch.zeros(1, device="cuda", dtype=torch.int32)
    outputs = []
    start = 0
    for length in (32, 15, 1):
        end = start + length
        outputs.append(attention.flash_attn_with_kvcache(
            q[:, start:end], k_cache, v_cache,
            k=k[:, start:end], v=v[:, start:end], cache_seqlens=positions,
            causal=True, window_size=(window, 0),
        ))
        positions.add_(length)
        start = end
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_cache[:, :48], k)
    torch.testing.assert_close(v_cache[:, :48], v)
    assert k_cache[:, 48:].isnan().all() and v_cache[:, 48:].isnan().all()
