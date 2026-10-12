"""Shared datasets must stay separate from run artifacts and selected shard counts."""

import io
import json
import multiprocessing
from pathlib import Path
import runpy
import stat
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import nanochat.dataset as dataset
from scripts import base_eval
from tasks.common import load_hub_dataset


@pytest.fixture(autouse=True)
def explicit_shard_selection(monkeypatch):
    monkeypatch.delenv("NANOCHAT_TRAIN_SHARDS", raising=False)


@pytest.fixture
def storage_dirs(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    run = tmp_path / "training_run" / "test"
    monkeypatch.setenv("NANOCHAT_DATASETS_DIR", str(datasets))
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(run))
    return datasets, run


@pytest.fixture
def shared_shards(tmp_path, monkeypatch):
    root = tmp_path / "base_data_climbmix"
    root.mkdir()
    for index in (*range(173), dataset.MAX_SHARD, 99999):
        pq.write_table(pa.table({"text": [f"document-{index}"]}), root / dataset.index_to_filename(index))
    (root / "shard_00001.parquet.tmp").write_bytes(b"unfinished download")
    monkeypatch.setattr(dataset, "DATA_DIR", str(root))
    return root


@pytest.mark.parametrize("count", [8, 170])
@pytest.mark.parametrize("explicit_directory", [False, True])
def test_shared_shards_preserve_requested_training_subset_and_validation(shared_shards, monkeypatch, count, explicit_directory):
    monkeypatch.setenv("NANOCHAT_TRAIN_SHARDS", str(count))
    paths = dataset.list_parquet_files(shared_shards if explicit_directory else None)
    assert [Path(path).name for path in paths] == [
        *(dataset.index_to_filename(index) for index in range(count)),
        dataset.index_to_filename(dataset.MAX_SHARD),
    ]
    training = [text for batch in dataset.parquets_iter_batched("train") for text in batch]
    validation = [text for batch in dataset.parquets_iter_batched("val") for text in batch]
    assert training == [f"document-{index}" for index in range(count)]
    assert validation == [f"document-{dataset.MAX_SHARD}"]


@pytest.mark.parametrize("missing", [0, 7, dataset.MAX_SHARD])
def test_missing_selected_shard_never_substitutes_an_extra_cached_shard(shared_shards, monkeypatch, missing):
    monkeypatch.setenv("NANOCHAT_TRAIN_SHARDS", "8")
    path = shared_shards / dataset.index_to_filename(missing)
    path.unlink()
    with pytest.raises(FileNotFoundError, match=path.name):
        dataset.list_parquet_files()


@pytest.mark.parametrize("count", ["", "0", "-1", "6543", "all", "1.5"])
def test_invalid_training_shard_count_fails_before_loading(tmp_path, monkeypatch, count):
    monkeypatch.setenv("NANOCHAT_TRAIN_SHARDS", count)
    with pytest.raises(ValueError, match="NANOCHAT_TRAIN_SHARDS"):
        dataset.list_parquet_files(tmp_path / "does-not-exist")


def test_explicit_dataset_directory_without_selection_keeps_sorted_parquets(tmp_path):
    for name in ("z.parquet", "a.parquet", "a.parquet.tmp", "readme.txt"):
        (tmp_path / name).write_bytes(b"fixture")
    assert dataset.list_parquet_files(tmp_path) == [str(tmp_path / "a.parquet"), str(tmp_path / "z.parquet")]


def test_missing_climbmix_directory_does_not_fall_back_to_old_dataset(tmp_path, monkeypatch):
    old = tmp_path / "base_data"
    old.mkdir()
    (old / "shard_00000.parquet").write_bytes(b"obsolete data")
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path / "base_data_climbmix"))
    with pytest.raises(FileNotFoundError, match="base_data_climbmix"):
        dataset.list_parquet_files()


def test_downloader_stores_shards_in_shared_datasets_directory(storage_dirs, monkeypatch):
    datasets, run = storage_dirs
    requested = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"downloaded parquet fixture"

    class SerialPool:
        def __init__(self, processes):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, indices):
            return [function(index) for index in indices]

    def download(url, **kwargs):
        requested.append(url)
        return Response()

    monkeypatch.setattr(dataset.requests, "get", download)
    monkeypatch.setattr(multiprocessing, "Pool", SerialPool)
    monkeypatch.setattr(sys, "argv", ["nanochat.dataset", "-n", "2", "-w", "1"])
    runpy.run_path(dataset.__file__, run_name="__main__")
    directory = datasets / "base_data_climbmix"
    expected = [dataset.index_to_filename(index) for index in (0, 1, dataset.MAX_SHARD)]
    assert sorted(path.name for path in directory.glob("*.parquet")) == expected
    assert len(requested) == 3
    assert not run.exists()


def test_task_download_cache_is_shared_across_run_directories(storage_dirs, monkeypatch):
    datasets, run = storage_dirs
    blob = io.BytesIO()
    pq.write_table(pa.table({"question": ["q"], "answer": ["a"]}), blob)
    listing_url = "https://huggingface.co/api/datasets/example/task/parquet/default/train"
    shard_url = "https://example.test/shard.parquet"
    requested = []

    def fetch(url):
        requested.append(url)
        if url == listing_url:
            return io.BytesIO(json.dumps([shard_url]).encode())
        assert url == shard_url
        return io.BytesIO(blob.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    first = load_hub_dataset("example/task")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(run.with_name("second")))
    second = load_hub_dataset("example/task")
    assert first[0] == second[0] == {"question": "q", "answer": "a"}
    assert requested == [listing_url, shard_url]
    assert (datasets / "task_data" / "example--task" / "default" / "train" / "manifest.json").is_file()
    assert not run.exists()
    assert not run.with_name("second").exists()


@pytest.fixture
def core_bundle_bytes():
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("eval_bundle/core.yaml", json.dumps({"icl_tasks": [{
            "label": "example", "icl_task_type": "multiple_choice", "dataset_uri": "example.jsonl",
            "num_fewshot": [0], "continuation_delimiter": " ",
        }]}))
        archive.writestr("eval_bundle/eval_meta_data.csv", "Eval Task,Random baseline\nexample,0\n")
        archive.writestr("eval_bundle/eval_data/example.jsonl", '{"prompt":"q","answer":"a"}\n')
    return blob.getvalue()


def test_core_download_and_extraction_use_dataset_root_but_results_use_run_root(storage_dirs, core_bundle_bytes, monkeypatch):
    datasets, run = storage_dirs
    requested = []
    extraction_paths = []
    extractall = zipfile.ZipFile.extractall

    def fetch(url):
        assert url == base_eval.EVAL_BUNDLE_URL
        requested.append(url)
        return io.BytesIO(core_bundle_bytes)

    def extract(archive, path, *args, **kwargs):
        extraction_paths.append(Path(path))
        return extractall(archive, path, *args, **kwargs)

    def score(model, tokenizer, data, device, task_meta):
        assert data == [{"prompt": "q", "answer": "a"}]
        return 0.75

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    monkeypatch.setattr(base_eval, "evaluate_task", score)
    monkeypatch.setattr(base_eval, "compute_init", lambda device: (False, 0, 0, 1, torch.device("cpu")))
    monkeypatch.setattr(base_eval, "compute_cleanup", lambda: None)
    monkeypatch.setattr(base_eval, "load_model", lambda *args, **kwargs: (
        object(), object(), {"model_config": {"sequence_len": 8}, "step": 7},
    ))
    monkeypatch.setattr(base_eval, "get_token_bytes", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["base_eval", "--eval", "core", "--device-type", "cpu"])
    base_eval.main()
    assert (run / "base_eval" / "base_model_000007.csv").read_text().count("0.750000") >= 1
    assert (datasets / "eval_bundle.zip").is_file()
    assert (datasets / "eval_bundle" / "eval_data" / "example.jsonl").is_file()
    assert extraction_paths and all(path.parent == datasets and not path.exists() for path in extraction_paths)
    assert not (run / "eval_bundle").exists()
    assert not (datasets / "base_eval").exists()
    base_eval.evaluate_core(object(), object(), torch.device("cpu"))
    assert requested == [base_eval.EVAL_BUNDLE_URL]


def test_concurrent_core_evaluations_wait_for_complete_bundle(storage_dirs, core_bundle_bytes, monkeypatch):
    datasets, _ = storage_dirs
    extraction_started, release_extraction, second_started = Event(), Event(), Event()
    requested, extraction_paths = [], []
    extractall = zipfile.ZipFile.extractall
    download_file = base_eval.download_file_with_lock

    def fetch(url):
        requested.append(url)
        return io.BytesIO(core_bundle_bytes)

    def extract(archive, path, *args, **kwargs):
        extraction_paths.append(Path(path))
        extraction_started.set()
        assert release_extraction.wait(timeout=5)
        return extractall(archive, path, *args, **kwargs)

    def download(*args, **kwargs):
        if extraction_started.is_set():
            second_started.set()
        return download_file(*args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    monkeypatch.setattr(base_eval, "download_file_with_lock", download)
    monkeypatch.setattr(base_eval, "evaluate_task", lambda *args: 0.75)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(base_eval.evaluate_core, object(), object(), torch.device("cpu"))
        try:
            assert extraction_started.wait(timeout=5)
            assert (datasets / "eval_bundle.zip").is_file()
            assert not (datasets / "eval_bundle").exists()
            second = executor.submit(base_eval.evaluate_core, object(), object(), torch.device("cpu"))
            assert second_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release_extraction.set()
        assert first.result(timeout=5)["core_metric"] == second.result(timeout=5)["core_metric"] == 0.75
    assert requested == [base_eval.EVAL_BUNDLE_URL]
    assert len(extraction_paths) == 1
    assert not (datasets / "eval_bundle" / "eval_bundle").exists()


def test_core_retries_interrupted_extraction_from_cached_archive(storage_dirs, core_bundle_bytes, monkeypatch):
    datasets, _ = storage_dirs
    requested, extraction_paths = [], []
    extractall = zipfile.ZipFile.extractall

    def fetch(url):
        requested.append(url)
        return io.BytesIO(core_bundle_bytes)

    def extract(archive, path, *args, **kwargs):
        extraction_paths.append(Path(path))
        if len(extraction_paths) == 1:
            (Path(path) / "partial").write_bytes(b"unfinished")
            raise OSError("interrupted extraction")
        return extractall(archive, path, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", fetch)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", extract)
    monkeypatch.setattr(base_eval, "evaluate_task", lambda *args: 0.75)
    with pytest.raises(OSError, match="interrupted extraction"):
        base_eval.evaluate_core(object(), object(), torch.device("cpu"))
    assert (datasets / "eval_bundle.zip").is_file()
    assert not (datasets / "eval_bundle").exists()
    result = base_eval.evaluate_core(object(), object(), torch.device("cpu"))
    assert result["core_metric"] == 0.75
    assert requested == [base_eval.EVAL_BUNDLE_URL]
    assert len(extraction_paths) == 2
    assert all(not path.exists() for path in extraction_paths)


@pytest.mark.parametrize("member_name", [
    "../escaped", "/absolute", "eval_bundle/../escaped", "eval_bundle\\escaped", "C:/escaped", "other/file", "eval_bundle/link",
])
def test_core_rejects_unsafe_archive_before_extracting_any_files(storage_dirs, core_bundle_bytes, monkeypatch, member_name):
    datasets, _ = storage_dirs
    datasets.mkdir()
    archive_path = datasets / "eval_bundle.zip"
    archive_path.write_bytes(core_bundle_bytes)
    member = zipfile.ZipInfo(member_name)
    if member_name == "eval_bundle/link":
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr(member, b"unsafe")
    monkeypatch.setattr(zipfile.ZipFile, "extractall", lambda *args, **kwargs: pytest.fail("Unsafe archive was extracted"))
    with pytest.raises(ValueError, match="Unsafe CORE bundle archive entry"):
        base_eval.place_eval_bundle(archive_path)
    assert not (datasets / "eval_bundle").exists()
    assert list(datasets.iterdir()) == [archive_path]


def test_concurrent_runs_publish_each_shared_shard_once(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    first_started, release_first = Event(), Event()
    requested = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"first half"
            first_started.set()
            assert release_first.wait(timeout=5)
            yield b"second half"

    def download(*args, **kwargs):
        requested.append(args[0])
        return Response()

    monkeypatch.setattr(dataset.requests, "get", download)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(dataset.download_single_file, 0)
        try:
            assert first_started.wait(timeout=5)
            second = executor.submit(dataset.download_single_file, 0)
            assert not (tmp_path / dataset.index_to_filename(0)).exists()
        finally:
            release_first.set()
        assert first.result(timeout=5) and second.result(timeout=5)
    assert len(requested) == 1
    assert (tmp_path / dataset.index_to_filename(0)).read_bytes() == b"first halfsecond half"
    assert not list(tmp_path.glob("*.tmp"))
