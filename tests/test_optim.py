"""
Test the combined MuonAdamW optimizer (single rank, no process group).
The AdamW half is checked against torch.optim.AdamW as a reference;
the Muon half is checked behaviorally (determinism + convergence).

Requires a GPU (the fused kernels are compiled on first use, which
dominates the runtime of this file).

python -m pytest tests/test_optim.py -v
"""

import pytest
import torch

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="optimizer tests require CUDA")

if cuda_available:
    from nanochat.optim import MuonAdamW

DEVICE = "cuda"

# keep shapes small and reuse them across tests so the fused kernels compile once
ADAMW_SMALL_SHAPE = (48,)        # numel < 1024 path
ADAMW_LARGE_SHAPE = (1024, 4)    # numel >= 1024 path
MUON_WIDE_SHAPE = (32, 64)
MUON_TALL_SHAPE = (64, 32)


def make_params_and_groups(seed=1337, offset=0.0, targets=None):
    """The menagerie: small/large AdamW params, wide/tall Muon stacks."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def rand(shape):
        base = torch.randn(shape, generator=gen, device=DEVICE) * 0.05
        return torch.nn.Parameter(base + offset)
    params = [rand(ADAMW_SMALL_SHAPE), rand(ADAMW_LARGE_SHAPE)]
    params += [rand(MUON_WIDE_SHAPE) for _ in range(3)]
    params += [rand(MUON_TALL_SHAPE) for _ in range(2)]
    groups = [
        dict(kind="adamw", params=params[0:1], lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="adamw", params=params[1:2], lr=0.02, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0),
        dict(kind="muon", params=params[2:5], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
        dict(kind="muon", params=params[5:7], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
    ]
    return params, groups


def test_adamw_matches_torch_reference():
    """Our fused AdamW must agree with torch.optim.AdamW (both decoupled wd)."""
    hypers = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    p_ours = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
    p_ref = torch.nn.Parameter(p_ours.detach().clone())
    opt_ours = MuonAdamW([dict(kind="adamw", params=[p_ours], **hypers)])
    opt_ref = torch.optim.AdamW([p_ref], **hypers)
    for step in range(10):
        grad = torch.randn(p_ours.shape, generator=torch.Generator(device=DEVICE).manual_seed(step), device=DEVICE)
        p_ours.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_ours.step()
        opt_ref.step()
    torch.testing.assert_close(p_ours, p_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shape", [ADAMW_SMALL_SHAPE, ADAMW_LARGE_SHAPE])
def test_adamw_frozen_parameter_skips_decay_and_initializes_state_on_unfreeze(shape):
    hypers = dict(lr=0.02, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.1)
    active = torch.nn.Parameter(torch.ones(ADAMW_SMALL_SHAPE, device=DEVICE))
    frozen = torch.nn.Parameter(torch.full(shape, 0.5, device=DEVICE), requires_grad=False)
    active_ref = torch.nn.Parameter(active.detach().clone())
    frozen_ref = torch.nn.Parameter(frozen.detach().clone(), requires_grad=False)
    initial_frozen = frozen.detach().clone()
    optimizer = MuonAdamW([dict(kind="adamw", params=[active, frozen], **hypers)])
    reference = torch.optim.AdamW([active_ref, frozen_ref], **hypers)

    for step in range(4):
        active.grad = torch.full_like(active, 0.1 + step * 0.05)
        active_ref.grad = active.grad.clone()
        if step >= 2:
            frozen.requires_grad_(True)
            frozen_ref.requires_grad_(True)
            frozen.grad = torch.full_like(frozen, 0.2 + step * 0.05)
            frozen_ref.grad = frozen.grad.clone()
        optimizer.step()
        reference.step()
        torch.testing.assert_close(active, active_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(frozen, frozen_ref, rtol=1e-5, atol=1e-6)
        if step < 2:
            assert torch.equal(frozen, initial_frozen)
            assert frozen not in optimizer.state
        else:
            assert not torch.equal(frozen, initial_frozen)
            assert optimizer.state[frozen]["step"] == step - 1
            torch.testing.assert_close(
                optimizer.state[frozen]["exp_avg"], reference.state[frozen_ref]["exp_avg"],
            )
            torch.testing.assert_close(
                optimizer.state[frozen]["exp_avg_sq"], reference.state[frozen_ref]["exp_avg_sq"],
            )
        optimizer.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)


@pytest.mark.parametrize("failure", ["missing_trainable_gradient", "stale_frozen_gradient"])
def test_adamw_gradient_contract_failure_aborts_before_any_update(failure):
    active = torch.nn.Parameter(torch.ones(ADAMW_SMALL_SHAPE, device=DEVICE))
    invalid = torch.nn.Parameter(torch.full(ADAMW_SMALL_SHAPE, 0.5, device=DEVICE))
    active.grad = torch.full_like(active, 0.2)
    if failure == "stale_frozen_gradient":
        invalid.requires_grad_(False)
        invalid.grad = torch.full_like(invalid, 0.3)
    before = [parameter.detach().clone() for parameter in (active, invalid)]
    optimizer = MuonAdamW([dict(
        kind="adamw", params=[active, invalid], lr=0.02,
        betas=(0.8, 0.995), eps=1e-10, weight_decay=0.1,
    )])

    with pytest.raises(RuntimeError, match="[Gg]radient"):
        optimizer.step()
    for parameter, original in zip((active, invalid), before):
        assert torch.equal(parameter, original)
        assert parameter not in optimizer.state


def test_determinism():
    """Two identical runs must produce bitwise identical parameters."""
    results = []
    for _ in range(2):
        params, groups = make_params_and_groups()
        opt = MuonAdamW(groups)
        for step in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(step)
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * 0.01
            opt.step()
        results.append([p.detach().clone() for p in params])
    for pa, pb in zip(*results):
        assert torch.equal(pa, pb)


def test_convergence():
    """Optimizing distance-to-target must actually approach the target."""
    targets_params, _ = make_params_and_groups(seed=999)
    targets = [p.detach().clone() for p in targets_params]
    params, groups = make_params_and_groups(seed=999, offset=0.1) # start offset from the targets
    opt = MuonAdamW(groups)
    def distances():
        return [(p.detach() - t).norm().item() for p, t in zip(params, targets)]
    initial = distances()
    for _ in range(50):
        for p, t in zip(params, targets):
            p.grad = 2 * (p.detach() - t) # gradient of ||p - t||^2
        opt.step()
    final = distances()
    for i, (d0, d1) in enumerate(zip(initial, final)):
        assert d1 < 0.5 * d0, f"param {i} did not converge: {d0:.4f} -> {d1:.4f}"
        assert torch.isfinite(params[i]).all()


def test_muon_update_is_orthogonalized():
    """
    The very first Muon update (zero momentum state) of a full-rank gradient
    should be (near) semi-orthogonal after the polar iteration: its nonzero
    singular values land in a band around 1, rather than being spread out.
    """
    p = torch.nn.Parameter(torch.zeros(MUON_WIDE_SHAPE, device=DEVICE))
    group = dict(kind="muon", params=[p], lr=1.0, momentum=0.0, ns_steps=5, beta2=1.0, weight_decay=0.0)
    opt = MuonAdamW([group])
    p.grad = torch.randn(p.shape, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)
    opt.step()
    # with lr=1, wd=0: p_new = -update, so the update is just -p
    svals = torch.linalg.svdvals(-p.detach().float())
    # NorMuon variance reduction rescales the magnitude, so normalize by the mean
    svals = svals / svals.mean()
    assert svals.max() / svals.min() < 4.0, f"update far from semi-orthogonal: {svals}"
