"""Protect project-local storage selection and cache setup before library imports."""

import os
from pathlib import Path

import pytest

from nanochat import paths


RUNTIME_VARIABLES = (
    "UV_CACHE_DIR", "TORCH_HOME", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
    "HF_HUB_CACHE", "HF_XET_CACHE", "TIKTOKEN_CACHE_DIR", "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR",
)


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    project = tmp_path / "repository"
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    # Register both operations so variables added by configure_runtime_paths are
    # removed at fixture teardown even when they were originally absent.
    for name in (*RUNTIME_VARIABLES, "NANOCHAT_BASE_DIR", "NANOCHAT_RUN_NAME", "NANOCHAT_DATASETS_DIR"):
        monkeypatch.setenv(name, "temporary-test-value")
        monkeypatch.delenv(name)
    return project


def test_defaults_remain_in_repository_when_current_directory_changes(isolated_paths, tmp_path, monkeypatch):
    other_directory = tmp_path / "elsewhere"
    other_directory.mkdir()
    initial = paths.get_run_dir(), paths.get_datasets_dir()
    monkeypatch.chdir(other_directory)
    assert (paths.get_run_dir(), paths.get_datasets_dir()) == initial
    assert initial == (isolated_paths / "training_run" / "pretrain-1", isolated_paths / "datasets")
    assert not isolated_paths.exists()
    assert list(other_directory.iterdir()) == []


def test_named_runs_and_explicit_overrides_keep_datasets_independent(isolated_paths, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_RUN_NAME", "pretrain-2")
    assert paths.get_run_dir() == isolated_paths / "training_run" / "pretrain-2"
    assert paths.get_datasets_dir() == isolated_paths / "datasets"
    output = tmp_path / "external-run"
    datasets = tmp_path / "external-datasets"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(output))
    monkeypatch.setenv("NANOCHAT_DATASETS_DIR", str(datasets))
    assert paths.get_run_dir() == output
    assert paths.get_datasets_dir() == datasets
    # An explicitly selected run name takes precedence over environment roots.
    assert paths.get_run_dir(paths.RunName("pretrain-3")) == isolated_paths / "training_run" / "pretrain-3"
    assert paths.get_datasets_dir() == datasets
    assert not output.exists() and not datasets.exists() and not isolated_paths.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/run", "nested\\run", " run", "run ", "bad\0name"])
def test_invalid_run_names_cannot_escape_the_artifact_root(isolated_paths, monkeypatch, name):
    with pytest.raises(ValueError, match="Run name"):
        if "\0" in name:
            paths.get_run_dir(paths.RunName(name))
        else:
            monkeypatch.setenv("NANOCHAT_RUN_NAME", name)
            paths.get_run_dir()
    assert not isolated_paths.exists()


@pytest.mark.parametrize("variable,resolver", [
    ("NANOCHAT_BASE_DIR", paths.get_run_dir), ("NANOCHAT_DATASETS_DIR", paths.get_datasets_dir),
])
def test_explicit_directories_reject_parent_traversal_before_normalization(isolated_paths, tmp_path, monkeypatch, variable, resolver):
    monkeypatch.setenv(variable, str(tmp_path / "scratch" / ".." / "outside"))
    with pytest.raises(ValueError, match="parent traversal"):
        resolver()
    assert not (tmp_path / "outside").exists()
    assert not isolated_paths.exists()


@pytest.mark.parametrize("variable,resolver", [
    ("NANOCHAT_BASE_DIR", paths.get_run_dir), ("NANOCHAT_DATASETS_DIR", paths.get_datasets_dir),
])
def test_explicit_directories_reject_symlink_ancestors_and_file_destinations(isolated_paths, tmp_path, monkeypatch, variable, resolver):
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv(variable, str(alias / "nested"))
    with pytest.raises(ValueError, match="symlinks"):
        resolver()
    assert list(outside.iterdir()) == []
    regular_file = tmp_path / "file"
    regular_file.write_text("preserved")
    monkeypatch.setenv(variable, str(regular_file))
    with pytest.raises(ValueError, match="must be a directory"):
        resolver()
    assert regular_file.read_text() == "preserved"


def test_runtime_defaults_follow_selected_run_without_creating_directories(isolated_paths, monkeypatch):
    monkeypatch.setenv("NANOCHAT_RUN_NAME", "pretrain-4")
    run = paths.get_run_dir()
    paths.configure_runtime_paths()
    for name in RUNTIME_VARIABLES:
        destination = Path(os.environ[name])
        assert destination == run or run in destination.parents
        assert not destination.exists()
    assert not isolated_paths.exists()


def test_runtime_setup_preserves_explicit_library_overrides_and_is_idempotent(isolated_paths, tmp_path, monkeypatch):
    run = tmp_path / "selected-run"
    custom_compiler = tmp_path / "custom-compiler-cache"
    custom_tracking = tmp_path / "custom-tracking"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(custom_compiler))
    monkeypatch.setenv("WANDB_DIR", str(custom_tracking))
    paths.configure_runtime_paths(run)
    first = {name: os.environ[name] for name in RUNTIME_VARIABLES}
    assert first["TORCHINDUCTOR_CACHE_DIR"] == str(custom_compiler)
    assert first["WANDB_DIR"] == str(custom_tracking)
    for name in set(RUNTIME_VARIABLES) - {"TORCHINDUCTOR_CACHE_DIR", "WANDB_DIR"}:
        assert run in Path(first[name]).parents
    paths.configure_runtime_paths(run)
    assert {name: os.environ[name] for name in RUNTIME_VARIABLES} == first
    assert not any(path.exists() for path in (run, custom_compiler, custom_tracking, isolated_paths))


def test_invalid_runtime_root_does_not_partially_install_environment(isolated_paths, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="symlinks"):
        paths.configure_runtime_paths(alias / "run")
    assert dict(os.environ) == before
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("ancestor", ["cache", "wandb"])
def test_default_runtime_cache_cannot_follow_nested_symlinks_or_partially_mutate_environment(isolated_paths, tmp_path, ancestor):
    run = tmp_path / "run"
    outside = tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    (run / ancestor).symlink_to(outside, target_is_directory=True)
    before = dict(os.environ)
    with pytest.raises(ValueError, match="symlinks"):
        paths.configure_runtime_paths(run)
    assert dict(os.environ) == before
    assert list(outside.iterdir()) == []


def test_default_runtime_cache_rejects_regular_file_ancestor_before_environment_changes(isolated_paths, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    blocked = run / "cache"
    blocked.write_text("owned existing file")
    before = dict(os.environ)
    with pytest.raises(ValueError, match="must be a directory"):
        paths.configure_runtime_paths(run)
    assert dict(os.environ) == before
    assert blocked.read_text() == "owned existing file"
