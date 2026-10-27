"""Resource limits and health checks shared by training and evaluation."""

import math
import os

import psutil
import torch
import torch.distributed as dist


def positive_gb(value, name):
    try:
        amount = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive finite number of decimal GB") from exc
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError(f"{name} must be a positive finite number of decimal GB")
    return amount * 1_000_000_000


def configure_cuda_memory(device):
    value = os.environ.get("NANOCHAT_CUDA_MEMORY_LIMIT_GB")
    if value is None:
        return
    limit = positive_gb(value, "NANOCHAT_CUDA_MEMORY_LIMIT_GB")
    total = torch.cuda.get_device_properties(device).total_memory
    if limit > total:
        raise ValueError("NANOCHAT_CUDA_MEMORY_LIMIT_GB exceeds the selected CUDA device capacity")
    torch.cuda.set_per_process_memory_fraction(limit / total, device)
    print(f"CUDA allocator ceiling: {limit / 1e9:.3f} GB (CPU and external CUDA allocations are additional)", flush=True)


def memory_status(device):
    cuda = device.type == "cuda"
    return {
        "cuda_allocated_gb": torch.cuda.memory_allocated(device) / 1e9 if cuda else 0,
        "cuda_reserved_gb": torch.cuda.memory_reserved(device) / 1e9 if cuda else 0,
        "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9 if cuda else 0,
        "system_available_gb": psutil.virtual_memory().available / 1e9,
    }


def check_training_memory(device, step, *, synchronize_workers=True):
    """Check headroom locally when the caller already coordinates phase errors."""
    value = os.environ.get("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
    if value is None:
        return
    minimum = positive_gb(value, "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB")
    status = memory_status(device)
    low = torch.tensor(status["system_available_gb"] * 1e9 < minimum, device=device, dtype=torch.int32)
    if synchronize_workers and dist.is_initialized():
        dist.all_reduce(low, op=dist.ReduceOp.MAX)
    if low.item():
        raise RuntimeError(f"Insufficient system memory at step {step}: {status}; required available GB: {minimum / 1e9}")
    if step % 10 == 0:
        print(f"Memory at step {step}: {status}", flush=True)


def require_finite_loss(loss, step, *, synchronize_workers=True):
    """Reject nonfinite loss, optionally leaving worker coordination to the caller."""
    finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
    if synchronize_workers and dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise FloatingPointError(f"Non-finite training loss at optimizer step {step}; update aborted")
