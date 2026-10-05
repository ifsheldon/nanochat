import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from nanochat.training_runtime import configure_cuda_memory, require_finite_loss, check_training_memory


@pytest.mark.parametrize("limit", ["0", "-1", "nan", "inf", "oops", "81"])
def test_invalid_cuda_limits_fail_before_setting_allocator(monkeypatch, limit):
    monkeypatch.setenv("NANOCHAT_CUDA_MEMORY_LIMIT_GB", limit)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=80_000_000_000))
    def unexpected(*args):
        pytest.fail("invalid limit reached allocator")
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", unexpected)
    with pytest.raises(ValueError):
        configure_cuda_memory(torch.device("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_small_cuda_limit_rejects_real_allocation():
    env = {**os.environ, "NANOCHAT_CUDA_MEMORY_LIMIT_GB": "0.01"}
    subprocess.run([sys.executable, "-c", """
import torch
from nanochat.common import compute_init
compute_init('cuda')
try:
    torch.empty(20_000_000, device='cuda', dtype=torch.float32)
except torch.OutOfMemoryError:
    pass
else:
    raise AssertionError('allocation exceeded configured ceiling')
"""], env=env, check=True, timeout=60)


def test_nonfinite_loss_aborts_before_update():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(FloatingPointError, match="update aborted"):
        loss = parameter * torch.tensor(float("nan"))
        require_finite_loss(loss, 7)
        loss.backward()
        optimizer.step()
    assert parameter.item() == 1.0


def test_system_memory_guard(monkeypatch):
    monkeypatch.setenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    monkeypatch.setattr("nanochat.training_runtime.psutil.virtual_memory", lambda: SimpleNamespace(available=15_000_000_000))
    with pytest.raises(RuntimeError, match="Insufficient system memory at step 12"):
        check_training_memory(torch.device("cpu"), 12)


def test_explicit_sdpa_bypasses_remote_kernel_discovery():
    subprocess.run([sys.executable, "-c", """
import sys
from types import ModuleType
import torch
fake = ModuleType('kernels')
def unexpected(*args):
    raise SystemExit('remote kernel discovery must not run')
fake.get_kernel = fake.has_kernel = unexpected
sys.modules['kernels'] = fake
import nanochat.flash_attention as attention
assert not attention.HAS_FA3 and not attention.USE_FA3
q = torch.randn(1, 4, 2, 8, requires_grad=True)
attention.flash_attn_func(q, q, q, causal=True).sum().backward()
assert q.grad.isfinite().all()
"""], env={**os.environ, "NANOCHAT_ATTENTION_IMPL": "sdpa"}, check=True, timeout=60)
