"""An explicit FA3 benchmark must never silently measure the SDPA fallback."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import nanochat.common as common_module
from nanochat.rlm_policy import AttentionBackend


def import_attention(monkeypatch, backend, *, cuda=True, dtype=torch.bfloat16, kernel=None):
    monkeypatch.setenv("NANOCHAT_ATTENTION_IMPL", backend)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(common_module, "COMPUTE_DTYPE", dtype)
    monkeypatch.setitem(sys.modules, "kernels", SimpleNamespace(get_kernel=kernel, has_kernel=lambda name: True))
    path = Path(__file__).resolve().parents[1] / "nanochat" / "flash_attention.py"
    spec = importlib.util.spec_from_file_location("attention_selection_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", ["no_cuda", "download"])
def test_explicit_fa3_loading_failure_aborts_but_auto_can_fall_back(monkeypatch, failure):
    def broken_kernel(name):
        raise OSError("kernel download failed")

    options = {"cuda": failure != "no_cuda", "kernel": broken_kernel}
    with pytest.raises(RuntimeError, match="could not be loaded") as error:
        import_attention(monkeypatch, "fa3", **options)
    expected = "compatible CUDA GPU" if failure == "no_cuda" else "kernel download failed"
    assert expected in str(error.value.__cause__)
    automatic = import_attention(monkeypatch, "auto", **options)
    assert automatic.attention_backend() is AttentionBackend.SDPA


def test_explicit_fa3_runtime_error_is_not_retried_with_sdpa(monkeypatch):
    calls = []

    def fail_forward(*args, **kwargs):
        calls.append("fa3")
        raise RuntimeError("FA3 operator failed")

    def kernel(name):
        assert name == "varunneal/flash-attention-3"
        return SimpleNamespace(flash_attn_interface=SimpleNamespace(flash_attn_func=fail_forward))

    attention = import_attention(monkeypatch, "fa3", kernel=kernel)
    assert attention.attention_backend() is AttentionBackend.FA3
    q = torch.zeros(1, 2, 1, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="FA3 operator failed"):
        attention.flash_attn_func(q, q, q, causal=True)
    assert calls == ["fa3"]


def test_explicit_fa3_rejects_unsupported_dtype_and_sdpa_skips_discovery(monkeypatch):
    calls = []

    def kernel(name):
        calls.append(name)
        return SimpleNamespace(flash_attn_interface=object())

    with pytest.raises(RuntimeError, match="requires bfloat16"):
        import_attention(monkeypatch, "fa3", dtype=torch.float32, kernel=kernel)
    before = calls.copy()
    attention = import_attention(monkeypatch, "sdpa", kernel=kernel)
    assert calls == before
    q = torch.randn(1, 2, 1, 8, requires_grad=True)
    attention.flash_attn_func(q, q, q, causal=True).sum().backward()
    assert q.grad.isfinite().all()
