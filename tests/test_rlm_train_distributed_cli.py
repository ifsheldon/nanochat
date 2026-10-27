"""Exercise the actual trainer entrypoint and shared artifacts with CPU workers."""

from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat.rlm_latest import LatestCheckpoint
from scripts import rlm_train
from test_rlm_distributed_training import _cpu_execution
from test_rlm_train_cli import TinyTokenizer


def _episodes(tokenizer, batch_size, sequence_length, *, split, device, resume_state_dict):
    assert split == "train" and device == "cpu"
    consumed = 0 if resume_state_dict is None else resume_state_dict["consumed"]
    rank = int(os.environ["RANK"])
    while True:
        generator = torch.Generator().manual_seed(300 + 100 * rank + consumed)
        tokens = torch.randint(1, tokenizer.get_vocab_size(), (batch_size, sequence_length + 1),
                              generator=generator)
        tokens[:, 0] = tokenizer.get_bos_token_id()
        consumed += 1
        yield tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), {
            "epoch": 1, "pq_idx": 0, "rg_idx": consumed, "consumed": consumed,
        }


def _main_worker(rank, world_size, directory, stop_step, resume, inject_oom, close_failure, mode):
    import nanochat.dataloader as dataloader_module
    import nanochat.rlm as rlm_module
    import nanochat.rlm_training as training_module
    import nanochat.tokenizer as tokenizer_module

    directory = Path(directory)
    output = directory / (mode if mode != "train" else "failed-close" if close_failure else "failed" if inject_oom else "train")
    environment = {
        "RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world_size),
        "NANOCHAT_BASE_DIR": str(directory / "cache"), "NANOCHAT_DTYPE": "float32",
        "NANOCHAT_ATTENTION_IMPL": "sdpa",
        "NANOCHAT_MIN_SYSTEM_AVAILABLE_GB": "0.001", "NANOCHAT_CHECKPOINT_RESERVE_GB": "0.001",
    }
    argv = [
        "rlm_train", "--output-dir", str(output), "--mode", mode, "--device-type", "cpu",
        "--cycles", "2", "--device-batch-size", "1",
        "--total-batch-token-size", "32", "--num-iterations", "4", "--warmup-steps", "0",
        "--no-compile",
    ]
    if mode == "train":
        argv.extend(("--stop-after-step", str(stop_step)))
    if resume:
        argv.extend(("--resume-latest", "--model-profile", str(output / "checkpoints" / "model-profile.jsonc")))
    else:
        argv.extend(("--model-profile", str(directory / "profile.jsonc")))
    initialize_group = dist.init_process_group

    def bounded_group(backend):
        return initialize_group(
            backend, init_method=f"file://{directory / f'rendezvous-{stop_step}-{inject_oom}-{close_failure}'}",
            rank=rank, world_size=world_size, timeout=timedelta(seconds=20),
        )

    actual_forward = rlm_module.RecurrentGPT.forward_cycle

    def forward(model, *args, **kwargs):
        if rank == 1:
            raise torch.cuda.OutOfMemoryError("injected trainer rank-one OOM")
        return actual_forward(model, *args, **kwargs)

    with _cpu_execution(), ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, environment))
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(dist, "init_process_group", bounded_group))
        stack.enter_context(patch.object(tokenizer_module, "get_tokenizer", TinyTokenizer))
        stack.enter_context(patch.object(dataloader_module, "tokenizing_distributed_data_loader_with_state_bos_bestfit", _episodes))
        stack.enter_context(patch.object(rlm_train, "dataset_inventory", return_value={"sha256": "test-data", "files": []}))
        stack.enter_context(patch.object(training_module, "source_manifest", return_value={"sha256": "test-source", "files": {}}))
        if close_failure:
            from nanochat.rlm_distributed import DistributedRunError
            from nanochat.rlm_distributed_checkpoint import CheckpointStateStager

            close_stager = CheckpointStateStager.close

            def fail_close(stager):
                close_stager(stager)
                if rank == 1:
                    raise RuntimeError("injected rank-one checkpoint stager close failure")

            stack.enter_context(patch.object(CheckpointStateStager, "close", fail_close))
            with pytest.raises(DistributedRunError, match="close checkpoint writer.*rank 1.*stager close failure"):
                rlm_train.main()
        elif inject_oom:
            stack.enter_context(patch.object(rlm_module.RecurrentGPT, "forward_cycle", forward))
            with pytest.raises(torch.cuda.OutOfMemoryError, match="rank 1.*trainer rank-one OOM"):
                rlm_train.main()
        else:
            rlm_train.main()


def _launch(world_size, directory, stop_step, *, resume=False, inject_oom=False, close_failure=False, mode="train"):
    context = mp.spawn(
        _main_worker, args=(world_size, str(directory), stop_step, resume, inject_oom, close_failure, mode),
        nprocs=world_size, join=False,
    )
    deadline = time.monotonic() + 30
    try:
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "trainer workers did not finish within 30 seconds"
            if context.join(timeout=remaining, grace_period=1):
                break
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)


def _tokenizer(directory):
    tokenizer = directory / "cache" / "tokenizer" / "tokenizer.pkl"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"distributed trainer tokenizer fixture")
    (directory / "profile.jsonc").write_text("// Distributed GQA experiment\n" + json.dumps({
        "n_layer": 3, "n_embd": 32, "n_head": 2, "n_kv_head": 1,
        "text_length": 4, "memory_tokens": 2, "eom_tokens": 1,
    }))


def test_main_two_four_one_workers_resume_and_publish_one_shared_attempt_each(tmp_path):
    _tokenizer(tmp_path)
    output = tmp_path / "train"
    for index, (world_size, step) in enumerate(((2, 2), (4, 3), (1, 4))):
        _launch(world_size, tmp_path, step, resume=index > 0)
        attempts = sorted((output / "attempts").iterdir())
        assert len(attempts) == index + 1
        attempt = attempts[-1]
        report = json.loads((output / "result.json").read_text())
        assert report["world_size"] == world_size
        assert report["completed_updates"] == step
        assert report["text_tokens"] == step * 32
        assert report["supervised_tokens"] == step * 40
        assert json.loads((attempt / "result.json").read_text()) == report
        assert len((attempt / "updates.jsonl").read_text().splitlines()) == (2 if index == 0 else 1)
        assert sorted(path.name for path in attempt.glob("rank-*")) == (
            [f"rank-{rank:05d}" for rank in range(world_size)] if world_size > 1 else []
        )
        for rank_directory in attempt.glob("rank-*"):
            assert not (rank_directory / "result.json").exists()
            assert not (rank_directory / "updates.jsonl").exists()
        metadata = LatestCheckpoint(output / "checkpoints").metadata()
        assert metadata["model_config"]["n_kv_head"] == 1
        saved_profile = output / "checkpoints" / "model-profile.jsonc"
        if index == 0:
            assert saved_profile.read_bytes() == (tmp_path / "profile.jsonc").read_bytes()
            (tmp_path / "profile.jsonc").unlink()
        assert metadata["step"] == step
        assert metadata["internal_positions"] == step * 56
        assert metadata["run_contract"]["world_size"] == world_size
        assert "world_size" not in metadata["training_config"]
        if world_size > 1:
            assert metadata["distributed"]["world_size"] == world_size
            assert len(metadata["distributed"]["rank_states"]) == world_size


def test_main_worker_oom_records_all_diagnostics_and_no_checkpoint(tmp_path):
    _tokenizer(tmp_path)
    _launch(2, tmp_path, 2, inject_oom=True)
    output = tmp_path / "failed"
    attempts = list((output / "attempts").iterdir())
    assert len(attempts) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["world_size"] == 2
    assert report["failure_kind"] == "cuda_out_of_memory"
    assert "rank 1" in report["error"]
    assert not (output / "checkpoints" / "latest.pt").exists()
    assert not (output / "checkpoints" / "model-profile.jsonc").exists()
    for rank in range(2):
        diagnostic = attempts[0] / f"rank-{rank:05d}"
        assert (diagnostic / "failed-update.json").is_file()
        assert json.loads((diagnostic / "failure.json").read_text())["failure_kind"] == "cuda_out_of_memory"


def test_main_worker_close_failure_stops_peers_and_retains_complete_checkpoint(tmp_path):
    _tokenizer(tmp_path)
    _launch(2, tmp_path, 2, close_failure=True)
    output = tmp_path / "failed-close"
    attempts = list((output / "attempts").iterdir())
    assert len(attempts) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["world_size"] == 2
    assert report["failure_kind"] == "error"
    assert "rank 1" in report["error"]
    assert "stager close failure" in report["error"]
    model, optimizer, metadata = LatestCheckpoint(output / "checkpoints").load_cpu_mapped()
    assert metadata["step"] == 2
    assert model and optimizer["state"]
    for rank in range(2):
        diagnostic = attempts[0] / f"rank-{rank:05d}" / "failure.json"
        assert "stager close failure" in json.loads(diagnostic.read_text())["error"]


@pytest.mark.parametrize("mode", ["capacity-probe", "attention-probe"])
def test_main_multiworker_probe_publishes_one_global_result(tmp_path, mode):
    _tokenizer(tmp_path)
    _launch(2, tmp_path, 1, mode=mode)
    output = tmp_path / mode
    attempts = list((output / "attempts").iterdir())
    assert len(attempts) == 1
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "completed"
    assert report["mode"] == mode
    assert report["world_size"] == 2
    assert report["completed_updates"] == 1 + len(report.get("warmup_updates", []))
    assert len(report["updates"]) == 1
    assert json.loads((attempts[0] / "result.json").read_text()) == report
    assert not (output / "checkpoints").exists()
