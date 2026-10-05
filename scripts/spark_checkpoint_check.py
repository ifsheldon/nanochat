"""Check a validation checkpoint's tensors, optimizer state, and resume metadata."""

import sys
import math

import torch

from nanochat.checkpoint_manager import load_checkpoint
from nanochat.common import compute_init, compute_cleanup


def main():
    directory, step = sys.argv[1], int(sys.argv[2])
    *_, device = compute_init("cuda")
    model, optimizer, meta = load_checkpoint(directory, step, device, load_optimizer=True)
    if meta["step"] != step or not optimizer["state"] or "loop_state" not in meta:
        raise RuntimeError("Missing completed step, optimizer, or scheduler state")
    adam_steps = [int(state["step"]) for state in optimizer["state"].values() if "step" in state]
    if not adam_steps or any(value != step for value in adam_steps):
        raise RuntimeError(f"Optimizer update counts did not resume correctly: {adam_steps}")
    warmup = meta["user_config"]["warmup_steps"]
    if step <= warmup:
        for group in optimizer["param_groups"]:
            if not math.isclose(group["lr"], group["initial_lr"] * step / warmup, rel_tol=1e-6):
                raise RuntimeError("Saved optimizer learning rate does not match the resumed warmup position")
    def check(value):
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
            raise FloatingPointError("Non-finite checkpoint tensor")
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
    check(model)
    check(optimizer)
    print(f"Checkpoint {step}: all model and optimizer tensors finite; resume metadata present")
    compute_cleanup()


if __name__ == "__main__":
    main()
