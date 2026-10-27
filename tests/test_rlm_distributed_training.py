"""Real recurrent updates preserve a global batch and portable optimizer history."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import timedelta
from itertools import chain
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nanochat.common as common_module
import nanochat.gpt as gpt_module
import nanochat.optim as optim_module
import nanochat.rlm as rlm_module
from nanochat.optim_checkpoint import OptimizerStateLayout
from nanochat.rlm import RecurrentGPT, RLMConfig
from nanochat.rlm_distributed import DistributedRuntime
from nanochat.rlm_distributed import DistributedRunError
from nanochat.rlm_distributed_checkpoint import CheckpointStateStager, DistributedCheckpointWriter
from nanochat.rlm_latest import LatestCheckpoint
from nanochat.rlm_training import TrainingConfig, checkpoint_metadata, measured_update, train_update


class _GlooWork:
    """Gloo lacks the Work Future API used by the NCCL production optimizer."""

    def __init__(self, work):
        self.work = work

    def get_future(self):
        return self

    def wait(self):
        return self.work.wait()


def _gloo_collective(function, reduction_input=None):
    """Retain real CPU collectives while adapting NCCL-only AVG/Future APIs."""
    def call(*args, **kwargs):
        if reduction_input is not None and kwargs.get("op") == dist.ReduceOp.AVG:
            args[reduction_input].div_(dist.get_world_size(kwargs.get("group")))
            kwargs["op"] = dist.ReduceOp.SUM
        work = function(*args, **kwargs)
        return _GlooWork(work) if kwargs.get("async_op") else work
    return call


@contextmanager
def _cpu_execution():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with ExitStack() as stack:
        for module in (common_module, gpt_module, rlm_module, optim_module):
            stack.enter_context(patch.object(module, "COMPUTE_DTYPE", torch.float32))
        for name in ("adamw_step_fused", "muon_step_fused"):
            function = getattr(optim_module, name)
            stack.enter_context(patch.object(optim_module, name, function._torchdynamo_orig_callable))
        for name, reduction_input in (("all_reduce", 0), ("reduce_scatter_tensor", 1),
                                      ("all_gather_into_tensor", None)):
            stack.enter_context(patch.object(dist, name, _gloo_collective(getattr(dist, name), reduction_input)))
        try:
            yield
        finally:
            torch.set_num_threads(previous_threads)


def _model_optimizer():
    torch.manual_seed(812)
    model = RecurrentGPT(RLMConfig(
        sequence_len=9, text_length=4, memory_tokens=2, vocab_size=32,
        # Three layers keep backout's cached residual distinct from the output.
        # At depth two that scalar only rescales the final RMSNorm input, so its
        # near-zero gradient amplifies harmless reduction-order roundoff in Adam.
        n_layer=3, n_head=2, n_kv_head=2, n_embd=32, window_pattern="L",
    ))
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    optimizer = model.setup_optimizer(
        embedding_lr=0.003, unembedding_lr=0.003, matrix_lr=0.003,
        scalar_lr=0.003, weight_decay=0.01,
    )
    return model, optimizer


def _config():
    return TrainingConfig(cycles=2, device_batch_size=1, total_batch_size=32,
                          num_iterations=4, compile=False)


def _loader(step, rank=0, world_size=1):
    generator = torch.Generator().manual_seed(901 + step)
    tokens = torch.randint(1, 32, (4, 9), generator=generator)
    tokens[:, 0] = 0
    for episode in range(rank, 4, world_size):
        yield tokens[episode:episode + 1, :-1].clone(), tokens[episode:episode + 1, 1:].clone(), {
            "epoch": 1, "pq_idx": step, "rg_idx": episode,
        }


def _metadata(model, config, step, loader_state):
    contract = {"model_config": asdict(model.config), "training_config": asdict(config),
                "tokenizer_sha256": "distributed-training-test", "device_type": "cpu"}
    return checkpoint_metadata(model, config, step, loader_state, contract, {
        "text_tokens": step * 32, "supervised_tokens": step * 40,
        "internal_positions": step * 56,
    })


def _reference_updates(directory):
    with _cpu_execution():
        model, optimizer = _model_optimizer()
        config = _config()
        for step in range(4):
            metrics, _ = measured_update(model, model.forward_cycle, optimizer, _loader(step),
                                         config, step, 0)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "metrics": metrics}, directory / f"reference-{step + 1}.pt")


def _runtime(rank, world_size, directory, step):
    timeout = timedelta(seconds=20)
    dist.init_process_group("gloo", init_method=f"file://{directory / f'rendezvous-{step}'}",
                            rank=rank, world_size=world_size, timeout=timeout)
    control = dist.new_group(backend="gloo", timeout=timeout)
    checkpoint = dist.new_group(backend="gloo", timeout=timeout)
    return DistributedRuntime(torch.device("cpu"), rank, world_size, control, checkpoint)


def _assert_tensor_state(actual, expected, context, *, atol=1e-6):
    assert actual.keys() == expected.keys(), context
    for key, value in expected.items():
        if isinstance(value, dict):
            _assert_tensor_state(actual[key], value, f"{context}.{key}", atol=atol)
        elif isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=1e-5, atol=atol,
                                       msg=lambda message: f"{context}.{key}: {message}")
        else:
            assert actual[key] == value, f"{context}.{key}"


def _resume_worker(rank, world_size, directory, step):
    directory = Path(directory)
    runtime = _runtime(rank, world_size, directory, step)
    writer = None
    try:
        with _cpu_execution():
            model, optimizer = _model_optimizer()
            config = _config()
            if step:
                saved_model, saved_optimizer, metadata = LatestCheckpoint(directory).load_cpu_mapped()
                assert metadata["step"] == step
                model.load_state_dict(saved_model)
                optimizer.load_state_dict(OptimizerStateLayout.from_optimizer(optimizer).shard(
                    saved_optimizer, rank, world_size,
                ))
            writer = DistributedCheckpointWriter(
                directory, mode="async", keep_last=1, optimizer=optimizer,
                checkpoint_group=runtime.checkpoint_group, control_group=runtime.control_group,
                latest_step=step or None,
            )
            metrics, loader_state = measured_update(
                model, model.forward_cycle, optimizer, _loader(step, rank, world_size),
                config, step, 0, runtime=runtime,
            )
            reference = torch.load(directory / f"reference-{step + 1}.pt", weights_only=True)
            # FP32 reduction order differs across world sizes. After four
            # updates Muon's polynomial amplifies ~1e-7 prior weight roundoff
            # to ~5e-6 in one matrix; moment comparisons remain at atol=1e-6.
            _assert_tensor_state(model.state_dict(), reference["model"], f"rank{rank}.model", atol=1e-5)
            assert metrics["loss"] == pytest.approx(reference["metrics"]["loss"], rel=1e-6)
            assert metrics["cycle_losses"] == pytest.approx(reference["metrics"]["cycle_losses"], rel=1e-6)
            assert metrics["text_targets"] == 32
            assert metrics["prediction_targets"] == 40
            assert metrics["internal_positions"] == 56
            writer.submit(model.state_dict(), optimizer.state_dict(),
                          _metadata(model, config, step + 1, loader_state), numbered=False)
            writer.wait(step + 1)
            if rank == 0:
                _, complete_optimizer, _ = LatestCheckpoint(directory).load_cpu_mapped()
                _assert_tensor_state(complete_optimizer["state"], reference["optimizer"]["state"], "optimizer")
    finally:
        if writer is not None:
            writer.abort()
        runtime.close()


def test_training_resumes_one_two_four_one_workers_with_same_global_updates(tmp_path):
    _reference_updates(tmp_path)
    for step, world_size in enumerate((1, 2, 4, 1)):
        mp.spawn(_resume_worker, args=(world_size, str(tmp_path), step), nprocs=world_size, join=True)
    metadata = LatestCheckpoint(tmp_path).metadata()
    assert metadata["step"] == 4
    assert metadata["distributed"]["world_size"] == 1


def _oom_worker(rank, world_size, directory):
    runtime = _runtime(rank, world_size, Path(directory), "oom")
    try:
        with _cpu_execution():
            model, optimizer = _model_optimizer()
            original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            steps = []
            actual_step = optimizer.step

            def record_step(**kwargs):
                steps.append(True)
                return actual_step(**kwargs)

            def forward(*args, **kwargs):
                if rank == 1:
                    raise torch.cuda.OutOfMemoryError("injected rank-one forward OOM")
                return model.forward_cycle(*args, **kwargs)

            optimizer.step = record_step
            with pytest.raises(torch.cuda.OutOfMemoryError, match="rank 1.*forward OOM"):
                train_update(model, forward, optimizer, _loader(0, rank, world_size),
                             _config(), 0, 0, runtime=runtime)
            assert not steps
            assert not optimizer.state
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    finally:
        runtime.close()


@pytest.mark.parametrize("world_size", [2, 4])
def test_worker_forward_oom_stops_every_worker_before_optimizer_mutation(tmp_path, world_size):
    mp.spawn(_oom_worker, args=(world_size, str(tmp_path)), nprocs=world_size, join=True)


def _attention_worker(rank, directory, cleanup_failure):
    import nanochat.rlm_attention_probe as probe_module

    directory = Path(directory)
    runtime = _runtime(rank, 2, directory, "attention")
    try:
        with _cpu_execution():
            model, optimizer = _model_optimizer()
            # Only rank one compiles on the first warmup. Every worker must
            # complete that warmup plus a stable warmup before the timed update.
            snapshots = iter([0, 1, 1, 1, 1, 1] if rank == 1 else [0] * 6)
            loader = chain.from_iterable(_loader(step, rank, 2) for step in range(3))
            with ExitStack() as stack:
                stack.enter_context(patch.object(
                    probe_module, "compiler_snapshot", side_effect=lambda: {"events": next(snapshots)},
                ))
                if cleanup_failure and rank == 1:
                    stack.enter_context(patch.object(CheckpointStateStager, "close", side_effect=RuntimeError(
                        "injected staging release failure",
                    )))
                if cleanup_failure:
                    with pytest.raises(DistributedRunError, match="staging release failure"):
                        probe_module.attention_probe(
                            model, model.forward_cycle, optimizer, loader, _config(), 0.01,
                            0, directory, runtime=runtime,
                        )
                    assert not (directory / "latest.pt").exists()
                    return
                report = probe_module.attention_probe(model, model.forward_cycle, optimizer, loader,
                                                      _config(), 0.01, 0, directory, runtime=runtime)
            assert report["completed_updates"] == 3
            assert len(report["warmup_updates"]) == 2
            assert report["attention_probe"]["compilation_excluded"]
            assert report["attention_probe"]["checkpoint_staging_preallocated"]
            assert len(report["updates"]) == 1
            assert report["updates"][0]["text_targets"] == 32
            assert not (directory / "latest.pt").exists()
    finally:
        runtime.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_attention_warmup_and_cleanup_are_coordinated(tmp_path, cleanup_failure):
    mp.spawn(_attention_worker, args=(str(tmp_path), cleanup_failure), nprocs=2, join=True)
