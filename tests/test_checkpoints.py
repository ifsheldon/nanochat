import json

import pytest
import torch

from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_last_step, checkpoint_is_complete


def test_interrupted_save_does_not_publish_checkpoint(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, 1, {"x": torch.ones(1)}, {"state": {}}, {"step": 1})
    original = torch.save
    def fail_optimizer(value, path):
        if "optim_" in str(path):
            raise OSError("interrupted optimizer write")
        original(value, path)
    monkeypatch.setattr(torch, "save", fail_optimizer)
    with pytest.raises(RuntimeError, match="not published"):
        save_checkpoint(tmp_path, 2, {"x": torch.zeros(1)}, {"state": {}}, {"step": 2})
    assert find_last_step(tmp_path, load_optimizer=True) == 1
    assert not checkpoint_is_complete(tmp_path, 2)
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        load_checkpoint(tmp_path, 2, "cpu", load_optimizer=True)
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_restores_optimizer_and_schedule_position(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    def update(p, opt, step):
        opt.param_groups[0]["lr"] = 0.1 * (1 - step / 4)
        p.square().backward()
        opt.step()
        opt.zero_grad()
    for step in range(2):
        update(parameter, optimizer, step)
    save_checkpoint(tmp_path, 2, {"parameter": parameter}, optimizer.state_dict(), {"step": 2, "horizon": 4})
    for step in range(2, 4):
        update(parameter, optimizer, step)
    expected = parameter.detach().clone()
    model, state, meta = load_checkpoint(tmp_path, 2, "cpu", load_optimizer=True)
    restored = torch.nn.Parameter(model["parameter"].detach())
    resumed_optimizer = torch.optim.AdamW([restored], lr=9)
    resumed_optimizer.load_state_dict(state)
    for step in range(meta["step"], meta["horizon"]):
        update(restored, resumed_optimizer, step)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)


def test_missing_optimizer_and_bad_metadata_are_rejected(tmp_path):
    torch.save({}, tmp_path / "model_000001.pt")
    (tmp_path / "meta_000001.json").write_text(json.dumps({"step": 1}))
    assert checkpoint_is_complete(tmp_path, 1)
    assert not checkpoint_is_complete(tmp_path, 1, load_optimizer=True)
    (tmp_path / "meta_000001.json").write_text("{")
    assert not checkpoint_is_complete(tmp_path, 1)


def test_disk_reserve_failure_retains_previous_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    save_checkpoint(tmp_path, 1, {}, {}, {"step": 1})
    monkeypatch.setenv("NANOCHAT_CHECKPOINT_RESERVE_GB", "100")
    monkeypatch.setattr("nanochat.checkpoint_manager.shutil.disk_usage", lambda path: SimpleNamespace(free=99_000_000_000))
    with pytest.raises(RuntimeError, match="reserve"):
        save_checkpoint(tmp_path, 2, {}, {}, {"step": 2})
    assert find_last_step(tmp_path, load_optimizer=True) == 1
