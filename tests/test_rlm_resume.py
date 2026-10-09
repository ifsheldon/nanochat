"""A workflow receipt authorizes one exact checkpoint-to-source transition."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from nanochat.rlm_resume import ResumeTransition, ResumeTransitionError


def manifest(files):
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.fixture
def transition_data(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    old_source = manifest({"nanochat/rlm.py": "1" * 64, "scripts/rlm_train.py": "2" * 64})
    new_source = manifest({
        "nanochat/rlm.py": "1" * 64, "scripts/rlm_train.py": "3" * 64,
        "nanochat/rlm_latest.py": "4" * 64,
    })
    contract = {
        "model_config": {"n_embd": 32, "text_length": 4, "memory_tokens": 2},
        "training_config": {"num_iterations": 5574, "cycles": 4, "seed": 42},
        "initialization": {"kind": "pretrained_input_token_embedding", "embedding_sha256": "5" * 64},
        "checkpoint_policy": {"early_step": 2, "save_every": 100, "keep_last": 10},
        "source_sha256": old_source["sha256"], "tokenizer_sha256": "6" * 64,
        "data_inventory_sha256": "7" * 64, "compute_dtype": "torch.bfloat16",
        "device_type": "cuda", "attention_impl": "sdpa",
        "runtime": {"python": "3.10.21", "torch": "2.14.0+cu132", "cuda": "13.2", "triton": "3.8.0"},
    }
    metadata = {
        "model_type": "rlm", "step": 300, "run_contract": contract,
        "rng_state": {"cpu": [42, 0, 1], "cuda": [42, 0, 2]},
        "dataloader_state_dict": {"pq_idx": 6, "rg_idx": 42, "epoch": 1},
    }
    current = deepcopy(contract)
    current["source_sha256"] = new_source["sha256"]
    current["checkpoint_policy"]["rolling_latest"] = "every_update"
    return run, metadata, current, old_source, new_source


def prepare(data):
    return ResumeTransition.prepare(*data, reason="Reviewed synchronous rolling persistence; training computation is unchanged.")


def save_receipt(path, receipt):
    path.write_text(json.dumps(receipt, indent=2))
    return path


def test_reviewed_transition_roundtrip_binds_metadata_contracts_and_source(transition_data, tmp_path):
    run, metadata, current, old_source, new_source = transition_data
    before = deepcopy((metadata, current, old_source, new_source))
    receipt = prepare(transition_data)
    record = receipt.to_dict()
    assert record["checkpoint_metadata_sha256"] == canonical_digest(metadata)
    assert record["previous_contract_sha256"] == canonical_digest(metadata["run_contract"])
    assert record["current_contract_sha256"] == canonical_digest(current)
    loaded = ResumeTransition.from_file(save_receipt(tmp_path / "receipt.json", record))
    # Whitespace and dictionary insertion order do not alter logical identity.
    reordered_metadata = json.loads(json.dumps(metadata, sort_keys=True, indent=4))
    proof = loaded.validate(run, reordered_metadata, current)
    assert proof["receipt_sha256"] == canonical_digest(record)
    assert proof["previous_source"] == old_source
    assert proof["current_source"] == new_source
    assert (metadata, current, old_source, new_source) == before
    # Exported dictionaries cannot mutate the authorization held in memory.
    record["current_contract"]["training_config"]["num_iterations"] = 1
    loaded.validate(run, metadata, current)


@pytest.mark.parametrize("field", [
    "model_config", "training_config", "data_inventory_sha256", "tokenizer_sha256",
    "compute_dtype", "device_type", "attention_impl", "runtime", "initialization",
])
def test_semantic_changes_rejected_during_preparation_and_even_with_rehashed_receipt(transition_data, tmp_path, field):
    _, metadata, current, _, _ = transition_data
    valid_record = prepare(transition_data).to_dict()
    changed = deepcopy(current)
    if isinstance(changed[field], dict):
        changed[field]["changed"] = True
    else:
        changed[field] = "changed"
    bad_data = (*transition_data[:2], changed, *transition_data[3:])
    with pytest.raises(ResumeTransitionError, match="cannot change"):
        prepare(bad_data)
    valid_record["current_contract"] = changed
    valid_record["current_contract_sha256"] = canonical_digest(changed)
    with pytest.raises(ResumeTransitionError, match="cannot change"):
        ResumeTransition.from_file(save_receipt(tmp_path / "bad.json", valid_record))
    assert metadata["run_contract"] != changed


def test_unknown_contract_fields_are_not_exempt_from_semantic_equality(transition_data):
    run, metadata, current, old, new = transition_data
    current["future_training_semantic"] = "new behavior"
    with pytest.raises(ResumeTransitionError, match="cannot change"):
        ResumeTransition.prepare(run, metadata, current, old, new, "must reject")


@pytest.mark.parametrize("change", ["run", "step", "metadata", "source", "policy", "model"])
def test_receipt_cannot_be_reused_for_another_checkpoint_or_target(transition_data, tmp_path, change):
    receipt = prepare(transition_data)
    run, metadata, current, _, _ = deepcopy(transition_data)
    if change == "run":
        run = tmp_path / "another-run"
        run.mkdir()
    elif change == "step":
        metadata["step"] += 1
    elif change == "metadata":
        metadata["rng_state"]["cpu"][0] += 1
    elif change == "source":
        current["source_sha256"] = "a" * 64
    elif change == "policy":
        current["checkpoint_policy"]["keep_last"] = 20
    else:
        current["model_config"]["n_embd"] = 64
    with pytest.raises(ResumeTransitionError):
        receipt.validate(run, metadata, current)


def test_manifest_proof_must_match_file_map_and_corresponding_contract(transition_data):
    run, metadata, current, old, new = transition_data
    altered = deepcopy(new)
    altered["files"]["nanochat/rlm.py"] = "f" * 64
    with pytest.raises(ResumeTransitionError, match="fingerprint"):
        ResumeTransition.prepare(run, metadata, current, old, altered, "invalid proof")
    altered = manifest(altered["files"])
    with pytest.raises(ResumeTransitionError, match="contract identities"):
        ResumeTransition.prepare(run, metadata, current, old, altered, "wrong target")
    with pytest.raises(ResumeTransitionError, match="contract identities"):
        ResumeTransition.prepare(run, metadata, current, new, old, "swapped source proofs")


@pytest.mark.parametrize("name", ["../rlm.py", "/tmp/rlm.py", "nanochat/../rlm.py", "nanochat//rlm.py", "nanochat\\rlm.py"])
def test_unsafe_source_manifest_paths_are_rejected(transition_data, name):
    run, metadata, current, old, _ = transition_data
    new = manifest({name: "1" * 64})
    current["source_sha256"] = new["sha256"]
    with pytest.raises(ResumeTransitionError, match="path"):
        ResumeTransition.prepare(run, metadata, current, old, new, "unsafe archive reference")


@pytest.mark.parametrize("field", ["checkpoint_metadata_sha256", "previous_contract_sha256", "current_contract_sha256"])
def test_tampered_digests_cannot_authorize_resume(transition_data, tmp_path, field):
    run, metadata, current, _, _ = transition_data
    record = prepare(transition_data).to_dict()
    record[field] = "0" * 64
    with pytest.raises(ResumeTransitionError):
        ResumeTransition.from_file(save_receipt(tmp_path / "tampered.json", record)).validate(run, metadata, current)


@pytest.mark.parametrize("kind", ["file_symlink", "parent_symlink", "hardlink", "fifo"])
def test_receipt_file_cannot_redirect_or_block_reading(transition_data, tmp_path, kind):
    owned = tmp_path / "owned"
    owned.mkdir()
    original = save_receipt(owned / "receipt.json", prepare(transition_data).to_dict())
    supplied = tmp_path / "receipt.json"
    if kind == "file_symlink":
        supplied.symlink_to(original)
    elif kind == "parent_symlink":
        parent = tmp_path / "redirect"
        parent.symlink_to(owned, target_is_directory=True)
        supplied = parent / "receipt.json"
    elif kind == "hardlink":
        os.link(original, supplied)
    else:
        os.mkfifo(supplied)
    with pytest.raises(ResumeTransitionError):
        ResumeTransition.from_file(supplied)


@pytest.mark.parametrize("kind", ["relative", "parent_traversal", "run_symlink", "ancestor_symlink"])
def test_run_directory_must_be_exact_owned_absolute_path(transition_data, tmp_path, kind):
    run, metadata, current, old, new = transition_data
    if kind == "relative":
        path = Path("run")
    elif kind == "parent_traversal":
        path = run / ".." / run.name
    elif kind == "run_symlink":
        path = tmp_path / "alias"
        path.symlink_to(run, target_is_directory=True)
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / run.name
    with pytest.raises(ResumeTransitionError):
        ResumeTransition.prepare(path, metadata, current, old, new, "unsafe run path")


def test_receipt_rejects_duplicate_keys_and_nonfinite_values(transition_data, tmp_path):
    record = prepare(transition_data).to_dict()
    path = tmp_path / "ambiguous.json"
    text = json.dumps(record)
    path.write_text('{"version": 1,' + text[1:])
    with pytest.raises(ResumeTransitionError, match="Duplicate JSON key"):
        ResumeTransition.from_file(path)
    record["current_contract"]["checkpoint_policy"]["rolling_latest"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ResumeTransitionError, match="finite JSON"):
        ResumeTransition.from_file(path)


def test_receipt_is_rejected_when_transition_is_unnecessary(transition_data):
    run, metadata, _, old, _ = transition_data
    with pytest.raises(ResumeTransitionError, match="redundant"):
        ResumeTransition.prepare(run, metadata, metadata["run_contract"], old, old, "not a transition")


@pytest.mark.parametrize("field,value", [("version", True), ("version", 2), ("checkpoint_step", True), ("checkpoint_step", 0)])
def test_receipt_version_and_checkpoint_step_are_strict(transition_data, tmp_path, field, value):
    record = prepare(transition_data).to_dict()
    record[field] = value
    with pytest.raises(ResumeTransitionError):
        ResumeTransition.from_file(save_receipt(tmp_path / "bad.json", record))
