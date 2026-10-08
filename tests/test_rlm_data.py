import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat import dataset
from nanochat.rlm import RLMConfig, RecurrentGPT
from nanochat.rlm_data import contiguous_validation_batches
from nanochat.rlm_inference import evaluate_memory_controls


class IntegerTokenizer:
    """Expose document token boundaries without a trained tokenizer artifact."""

    def get_bos_token_id(self):
        return 0

    def encode(self, documents, prepend, num_threads):
        return [[prepend, *map(int, document.split())] for document in documents]


def write_dataset(tmp_path, monkeypatch, documents):
    root = tmp_path / "data"
    root.mkdir()
    # Training content must never enter this held-out iterator.
    pq.write_table(pa.table({"text": ["90 91 92 93 94 95 96"]}), root / "shard_00000.parquet")
    pq.write_table(pa.table({"text": documents}), root / "shard_00001.parquet", row_group_size=2)
    monkeypatch.setattr(dataset, "DATA_DIR", str(root))


def tiny_model():
    model = RecurrentGPT(RLMConfig(
        sequence_len=8, vocab_size=64, n_layer=1, n_embd=32,
        n_head=2, n_kv_head=2, text_length=3, memory_tokens=2,
    ))
    model.init_weights()
    return model.eval()


def test_contiguous_prefixes_never_pack_or_repeat_documents(tmp_path, monkeypatch):
    write_dataset(tmp_path, monkeypatch, [
        "1 2 3 4 5 6 7 8 9 10 11 12",  # Long enough for more than one episode.
        "40 41",                        # Short documents cannot fill another row.
        "11 12 13 14 15 16",
        "21 22 23 24 25 26",           # One leftover row must not form a partial batch.
    ])
    batches = list(contiguous_validation_batches(
        IntegerTokenizer(), batch_size=2, episode_length=6,
        device=torch.device("cpu"), tokenizer_batch_size=1,
    ))
    assert len(batches) == 1
    inputs, targets = batches[0]
    torch.testing.assert_close(inputs, torch.tensor([[0, 1, 2, 3, 4, 5], [0, 11, 12, 13, 14, 15]]))
    torch.testing.assert_close(targets, torch.tensor([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]]))
    assert torch.equal((inputs == 0).sum(dim=1), torch.ones(2, dtype=torch.long))
    # Shifted-text evaluation scores the new cycle's first token in the prior
    # cycle; EOM also trains on that token, outside this text-only metric.
    torch.testing.assert_close(targets[:, 2], inputs[:, 3])
    assert not torch.equal(targets[:, 2], inputs[:, 2])


def test_short_filter_includes_the_extra_shifted_target_and_preserves_prior_batches(tmp_path, monkeypatch):
    write_dataset(tmp_path, monkeypatch, [
        "31 32 33 34 35",               # Six tokens with BOS, but seven are required.
        "1 2 3 4 5 6",                  # Exactly the required length.
        "11 12 13 14 15 16 17 18",
    ])
    batches = contiguous_validation_batches(IntegerTokenizer(), 1, 6, torch.device("cpu"))
    first_inputs, first_targets = next(batches)
    saved_inputs, saved_targets = first_inputs.clone(), first_targets.clone()
    second_inputs, second_targets = next(batches)
    torch.testing.assert_close(first_inputs, saved_inputs)
    torch.testing.assert_close(first_targets, saved_targets)
    torch.testing.assert_close(first_targets, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    torch.testing.assert_close(second_inputs, torch.tensor([[0, 11, 12, 13, 14, 15]]))
    torch.testing.assert_close(second_targets, torch.tensor([[11, 12, 13, 14, 15, 16]]))
    with pytest.raises(StopIteration):
        next(batches)


def test_finite_document_exhaustion_is_an_evaluation_error(tmp_path, monkeypatch):
    write_dataset(tmp_path, monkeypatch, ["1 2 3 4 5 6", "11 12 13 14 15 16", "21 22 23 24 25 26"])
    batches = contiguous_validation_batches(IntegerTokenizer(), 2, 6, torch.device("cpu"))
    with pytest.raises(ValueError, match="loader ended before num_batches"):
        evaluate_memory_controls(tiny_model(), batches, num_batches=2, cycles=2)


def test_eval_cli_selects_contiguous_documents_and_records_sampling_contract(tmp_path, monkeypatch):
    from scripts import rlm_eval

    write_dataset(tmp_path, monkeypatch, ["1 2 3", "1 2 3 4 5 6", "11 12 13 14 15 16"])
    output = tmp_path / "evaluation.json"
    model = tiny_model()
    monkeypatch.setattr("nanochat.rlm_inference.load_rlm_checkpoint", lambda *args: (model, IntegerTokenizer(), {"step": 6}))
    monkeypatch.setenv("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "0.001")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(sys, "argv", [
        "rlm_eval", "--device", "cpu", "--checkpoint-dir", str(tmp_path),
        "--batch-size", "2", "--batches", "1", "--cycles", "2",
        "--sampler", "contiguous_documents", "--output", str(output),
    ])
    rlm_eval.main()
    result = json.loads(output.read_text())
    assert result["episodes"] == 2
    assert result["sampler"] == "contiguous_documents"
    assert result["sampler_details"]["episode_selection"] == "one_prefix_per_document"
    assert result["sampler_details"]["minimum_tokens_including_bos"] == 7
    assert [cycle["targets"] for cycle in result["cycles"]] == [6, 6]
