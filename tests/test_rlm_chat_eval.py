import json
import sys

import pytest
import torch

from tests.test_rlm_engine import ToyChatTokenizer, active_tiny_rlm


def test_rlm_chat_eval_computes_chatcore_and_publishes_json(tmp_path, monkeypatch, capsys):
    from scripts import rlm_chat_eval

    model = active_tiny_rlm()
    tokenizer = ToyChatTokenizer()
    output_path = tmp_path / "chatcore.json"

    class FakeCategoricalTask:
        eval_type = "categorical"

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {
                "messages": [{"role": "user", "content": "1+1"}, {"role": "assistant", "content": "A"}],
                "letters": ("A", "B", "C", "D"),
            }

        def evaluate(self, conversation, predicted):
            return predicted in ("A", "B", "C", "D")

    class FakeGenerativeTask:
        eval_type = "generative"

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"messages": [{"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}]}

        def evaluate(self, conversation, completion):
            return isinstance(completion, str)

    def fake_build_task(name):
        return FakeCategoricalTask() if name in ("ARC-Easy", "ARC-Challenge", "MMLU") else FakeGenerativeTask()

    monkeypatch.setattr(sys, "argv", [
        "rlm_chat_eval", "--checkpoint-dir", str(tmp_path), "--device", "cpu",
        "--batch-size", "2", "--max-new-tokens", "3", "--output", str(output_path),
    ])
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("nanochat.common.compute_init", lambda dev: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr("nanochat.common.compute_cleanup", lambda: None)
    monkeypatch.setattr("nanochat.training_runtime.check_training_memory", lambda dev, step: None)
    monkeypatch.setattr(
        "nanochat.rlm_inference.load_rlm_checkpoint",
        lambda ckpt_dir, step, dev: (model, tokenizer, {"step": 504, "stage": "sft"}),
    )
    monkeypatch.setattr(rlm_chat_eval, "_build_task", fake_build_task)

    rlm_chat_eval.main()
    persisted = output_path.read_text()
    assert capsys.readouterr().out == persisted
    payload = json.loads(persisted)
    assert payload["step"] == 504
    assert payload["chatcore"] == pytest.approx(1.0)
    assert set(payload["results"]) == set(rlm_chat_eval.ALL_TASKS)


def test_rlm_chat_eval_rejects_unknown_task_before_loading(tmp_path, monkeypatch):
    from scripts import rlm_chat_eval

    monkeypatch.setattr(sys, "argv", [
        "rlm_chat_eval", "--checkpoint-dir", str(tmp_path), "--task-name", "ARC-Easy|NotATask",
    ])
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(SystemExit) as error:
        rlm_chat_eval.main()
    assert error.value.code == 2
