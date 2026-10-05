import pytest

from scripts.spark_d24 import Pipeline, write_json


def test_existing_production_marker_blocks_fresh_launch(tmp_path, monkeypatch):
    pipeline = Pipeline(tmp_path)
    monkeypatch.setattr(pipeline, "verify_inventory", lambda: None)
    monkeypatch.setattr(pipeline, "config", lambda: {"device_batch_size": 16})
    write_json(tmp_path / "production.json", {"device_batch_size": 16})
    def unexpected(*args):
        pytest.fail("existing run was launched again")
    monkeypatch.setattr(pipeline, "run", unexpected)
    with pytest.raises(FileExistsError, match="Production already started"):
        pipeline.pretrain()


def test_modified_training_source_invalidates_validation(tmp_path, monkeypatch):
    write_json(tmp_path / "validated.json", {"source_sha256": "previous-source"})
    pipeline = Pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="changed since validation"):
        pipeline.config()


def test_incomplete_resume_never_launches_training(tmp_path, monkeypatch):
    pipeline = Pipeline(tmp_path, resume_step=100)
    monkeypatch.setattr(pipeline, "verify_inventory", lambda: None)
    monkeypatch.setattr(pipeline, "config", lambda: {"device_batch_size": 16})
    with pytest.raises(FileNotFoundError, match="incomplete"):
        pipeline.pretrain()


def test_changed_data_inventory_blocks_training(tmp_path, monkeypatch):
    write_json(tmp_path / "data-inventory.json", {"shard": {"sha256": "original"}})
    pipeline = Pipeline(tmp_path)
    monkeypatch.setattr(pipeline, "inventory", lambda: {"shard": {"sha256": "changed"}})
    with pytest.raises(RuntimeError, match="inventory changed"):
        pipeline.verify_inventory()
