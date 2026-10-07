"""Staged, local-only d24 training on one DGX Spark. See dev/SPARK_D24.md."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

from filelock import FileLock


TAG = "spark-d24-bf16"
TOTAL_BATCH = 1_048_576
FINAL_STEP = 5568
STAGES = ("prepare", "validate", "pretrain", "base-eval", "sft", "chat-eval")
TASKS = ("ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval")


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def source_digest():
    paths = [*Path("nanochat").glob("*.py"), *Path("scripts").glob("*.py"),
             Path("runs/runspark_d24.sh"), Path("pyproject.toml"), Path("uv.lock")]
    fingerprint = hashlib.sha256()
    for path in sorted(paths):
        fingerprint.update(str(path).encode())
        fingerprint.update(bytes.fromhex(digest(path)))
    return fingerprint.hexdigest()


class CommandFailure(RuntimeError):
    def __init__(self, command, metrics):
        self.metrics = metrics
        super().__init__(f"Command failed: {shlex.join(command)}; details: {metrics}")


class Pipeline:
    def __init__(self, root, resume_step=None):
        self.root = root
        self.resume_step = resume_step
        self.log_dir = root / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.log_dir.mkdir(parents=True)
        self.command_number = 0
        self.current_stage = None

    def audit(self):
        import importlib.metadata
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--binary"], text=True)
        (self.log_dir / "working.diff").write_text(diff)
        # Git diff does not include untracked implementation files.
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"]).decode().split("\0")
        source = self.log_dir / "untracked-source"
        for name in filter(None, untracked):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, target)
        write_json(self.log_dir / "invocation.json", {
            "argv": sys.argv, "revision": revision, "python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "triton", "rustbpe", "pyarrow")},
            "root": str(self.root), "tag": TAG, "global_batch": TOTAL_BATCH,
            "final_step": FINAL_STEP, "cuda_limit_gb": 80,
            "resume_step": self.resume_step,
            "source_sha256": source_digest(),
        })

    def run(self, module, arguments=()):
        self.command_number += 1
        stem = f"{self.command_number:02d}-{module}"
        metrics_path = self.log_dir / f"{stem}.json"
        log = self.log_dir / f"{stem}.log"
        command = [sys.executable, "-m", "scripts.run_with_metrics", module, str(metrics_path), *map(str, arguments)]
        print(f"Running: {shlex.join(command)}\nLog: {log}", flush=True)
        with log.open("w") as output:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                code = process.wait()
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"error": "process exited without metrics"}
        metrics["log"] = str(log)
        if code or not metrics.get("success"):
            raise CommandFailure(command, metrics)
        return metrics

    def checkpoint_dir(self, source, tag=TAG):
        return self.root / ("base_checkpoints" if source == "base" else "chatsft_checkpoints") / tag

    def complete_step(self, source, tag=TAG):
        from nanochat.checkpoint_manager import find_last_step
        return find_last_step(self.checkpoint_dir(source, tag), load_optimizer=True)

    def inventory(self):
        import pyarrow.parquet as pq
        from nanochat.dataset import MAX_SHARD, index_to_filename
        result = {}
        expected = [index_to_filename(i) for i in [*range(170), MAX_SHARD]]
        directory = self.root / "base_data_climbmix"
        if sorted(p.name for p in directory.glob("*.parquet")) != expected:
            raise RuntimeError("Training directory must contain exactly shards 0..169 and the designated validation shard")
        for name in expected:
            path = directory / name
            parquet = pq.ParquetFile(path)
            if not parquet.metadata.num_rows or "text" not in parquet.schema.names:
                raise ValueError(f"Empty or invalid training shard: {path}")
            result[str(path.relative_to(self.root))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        for name in ("tokenizer.pkl", "token_bytes.pt"):
            path = self.root / "tokenizer" / name
            result[str(path.relative_to(self.root))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        return result

    def verify_inventory(self):
        saved = json.loads((self.root / "data-inventory.json").read_text())
        if self.inventory() != saved:
            raise RuntimeError("Tokenizer or data inventory changed; refusing to continue training")

    def prepare(self):
        from nanochat.dataset import MAX_SHARD, index_to_filename
        old = Path.home() / ".cache/nanochat/spark"
        if (self.root / "data-inventory.json").exists():
            self.verify_inventory()
            return
        for directory in ("tokenizer", "task_data", "eval_bundle"):
            if (old / directory).exists() and not (self.root / directory).exists():
                shutil.copytree(old / directory, self.root / directory)
        data = self.root / "base_data_climbmix"
        data.mkdir(exist_ok=True)
        for i in [*range(170), MAX_SHARD]:
            name = index_to_filename(i)
            if (old / "base_data_climbmix" / name).exists() and not (data / name).exists():
                shutil.copyfile(old / "base_data_climbmix" / name, data / name)
        self.run("nanochat.dataset", ["-n", "170"])
        from nanochat.tokenizer import get_tokenizer
        if get_tokenizer().get_vocab_size() != 32768:
            raise ValueError("d24 requires the validated 32768-token tokenizer")
        from tasks.smoltalk import SmolTalk
        from tasks.mmlu import MMLU
        from tasks.gsm8k import GSM8K
        from tasks.arc import ARC
        from tasks.humaneval import HumanEval
        from scripts.base_eval import EVAL_BUNDLE_URL, place_eval_bundle
        from nanochat.common import download_file_with_lock
        for split in ("train", "test"):
            SmolTalk(split=split)
            GSM8K(subset="main", split=split)
        for split in ("auxiliary_train", "test"):
            MMLU(subset="all", split=split)
        for subset in ("ARC-Easy", "ARC-Challenge"):
            ARC(subset=subset, split="test")
        HumanEval()
        if not (self.root / "eval_bundle").exists():
            download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        write_json(self.root / "data-inventory.json", self.inventory())

    def base_args(self, batch, tag):
        return ["--device-type=cuda", "--depth=24", "--head-dim=128", "--window-pattern=L",
                "--max-seq-len=2048", f"--device-batch-size={batch}", f"--total-batch-size={TOTAL_BATCH}",
                "--target-param-data-ratio=8", f"--model-tag={tag}", "--run=dummy",
                "--eval-every=100", "--eval-tokens=1048576", "--save-every=100",
                "--core-metric-every=-1", "--sample-every=-1"]

    def validate(self):
        self.verify_inventory()
        if (self.root / "validated.json").exists():
            raise FileExistsError("Validation already exists; preserve it or use a new artifact root")
        from nanochat.flash_attention import ATTENTION_BACKEND, AttentionBackend
        if ATTENTION_BACKEND is not AttentionBackend.SDPA:
            raise RuntimeError("This workflow requires the validated SDPA environment")
        for batch in (16, 8, 4, 2, 1):
            tag = f"validate-d24-b{batch}-{self.log_dir.name}"
            arguments = self.base_args(batch, tag) + ["--num-iterations=4", "--eval-every=2", "--eval-tokens=32768", "--save-every=2"]
            try:
                first = self.run("scripts.base_train", arguments + ["--stop-after-step=2"])
                if first["cuda_peak_reserved_gb"] > 72:
                    continue
                if self.complete_step("base", tag) != 2:
                    raise RuntimeError("Validation did not pause at completed update 2")
                resumed = self.run("scripts.base_train", arguments + ["--resume-from-step=2"])
                if self.complete_step("base", tag) != 4:
                    raise RuntimeError("Validation resume did not complete update 4")
                self.run("scripts.spark_checkpoint_check", [str(self.checkpoint_dir("base", tag)), "4"])
                inference = self.run("scripts.base_eval", ["--device-type=cuda", f"--model-tag={tag}", "--step=4", "--eval=sample"])
                sft = self.run("scripts.chat_sft", ["--device-type=cuda", f"--model-tag={tag}", "--model-step=4",
                    "--num-iterations=2", "--eval-every=2", "--eval-tokens=32768", "--chatcore-every=-1", "--run=dummy"])
                if self.complete_step("sft", tag) != 2:
                    raise RuntimeError("SFT validation did not perform exactly two optimizer updates")
                peaks = [record["cuda_peak_reserved_gb"] for record in (first, resumed, inference, sft)]
                if max(peaks) > 72:
                    continue
                size = sum(path.stat().st_size for path in self.checkpoint_dir("base", tag).glob("*000004*"))
                projected = size * ((FINAL_STEP // 100) + 2)
                if shutil.disk_usage(self.root).free < projected + 100_000_000_000:
                    raise OSError("Insufficient storage for all production checkpoints plus 100 GB reserve")
                write_json(self.root / "validated.json", {"device_batch_size": batch, "global_batch": TOTAL_BATCH,
                    "final_step": FINAL_STEP, "validation_tag": tag, "peak_reserved_gb": max(peaks),
                    "checkpoint_bytes": size, "projected_checkpoint_bytes": projected,
                    "source_sha256": source_digest()})
                return
            except CommandFailure as exc:
                if not exc.metrics.get("oom"):
                    raise
                print(f"Validation OOM at device batch {batch}; trying the next smaller batch", flush=True)
        raise RuntimeError("d24 could not pass validation under the memory budget, even at device batch 1")

    def config(self):
        config = json.loads((self.root / "validated.json").read_text())
        if config["source_sha256"] != source_digest():
            raise RuntimeError("Training source or dependencies changed since validation; revalidate before production")
        return config

    def pretrain(self):
        self.verify_inventory()
        config = self.config()
        directory = self.checkpoint_dir("base")
        arguments = self.base_args(config["device_batch_size"], TAG)
        if self.resume_step is None:
            # The marker also protects a run that failed before its first checkpoint.
            marker = self.root / "production.json"
            if marker.exists() or (directory.exists() and any(directory.iterdir())):
                raise FileExistsError("Production already started; specify --resume-from-step explicitly")
            write_json(marker, config)
        else:
            from nanochat.checkpoint_manager import checkpoint_is_complete
            if not checkpoint_is_complete(directory, self.resume_step, load_optimizer=True):
                raise FileNotFoundError("Requested resume checkpoint is incomplete")
            meta = json.loads((directory / f"meta_{self.resume_step:06d}.json").read_text())
            if json.loads((self.root / "production.json").read_text()) != config:
                raise ValueError("Production configuration differs from the validated configuration")
            saved = meta["user_config"]
            expected = {"depth": 24, "head_dim": 128, "window_pattern": "L", "max_seq_len": 2048,
                        "device_batch_size": config["device_batch_size"], "target_param_data_ratio": 8,
                        "num_iterations": -1, "fp8": False}
            if any(saved[key] != value for key, value in expected.items()) or meta["total_batch_size"] != TOTAL_BATCH:
                raise ValueError("Resume configuration differs from the production configuration")
            if self.resume_step >= FINAL_STEP:
                raise ValueError("Pretraining is already complete; run the evaluation/SFT stages")
            arguments.append(f"--resume-from-step={self.resume_step}")
        self.run("scripts.base_train", arguments)
        if self.complete_step("base") != FINAL_STEP:
            raise RuntimeError("Pretraining ended before the agreed horizon")

    def require_base(self):
        if self.complete_step("base") != FINAL_STEP:
            raise RuntimeError("This stage requires the completed production base checkpoint")

    def base_eval(self):
        self.require_base()
        self.run("scripts.base_eval", ["--device-type=cuda", f"--model-tag={TAG}", f"--step={FINAL_STEP}",
            "--device-batch-size=8", "--max-per-task=-1", "--split-tokens=20971520"])
        files = list((self.root / "base_eval").glob("*.csv"))
        if len(files) != 1:
            raise RuntimeError("Expected exactly one production CORE result")
        with files[0].open() as f:
            score = next(float(row[2]) for row in csv.reader(f) if row and row[0].strip() == "CORE")
        write_json(self.root / "base-results.json", {"core": score, "gpt2_threshold": 0.256525,
            "gpt2_target_met": score > 0.256525, "results_csv": str(files[0])})

    def sft(self):
        self.require_base()
        self.verify_inventory()
        directory = self.checkpoint_dir("sft")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Production SFT output exists; preserve it before explicitly restarting SFT")
        self.run("scripts.chat_sft", ["--device-type=cuda", f"--model-tag={TAG}", f"--model-step={FINAL_STEP}",
            "--num-iterations=-1", "--eval-every=100", "--eval-tokens=1048576", "--chatcore-every=-1", "--run=dummy"])

    def chat_eval(self):
        step = self.complete_step("sft")
        scores = {}
        for task in TASKS:
            for batch in (8, 4, 2, 1):
                try:
                    result = self.run("scripts.chat_eval", ["--device-type=cuda", "--source=sft", f"--model-tag={TAG}",
                        f"--step={step}", f"--task-name={task}", f"--batch-size={batch}",
                        "--temperature=0", "--num-samples=1", "--max-new-tokens=512"])
                    text = Path(result["log"]).read_text()
                    match = re.search(re.escape(task) + r" accuracy: ([\d.]+)%", text)
                    if match is None:
                        raise RuntimeError(f"Missing final accuracy for {task}")
                    scores[task] = float(match[1]) / 100
                    write_json(self.root / "chat-results.json", scores)
                    break
                except CommandFailure as exc:
                    if not exc.metrics.get("oom") or batch == 1 or task in ("GSM8K", "HumanEval"):
                        raise
        centered = [(value - (0.25 if task in TASKS[:3] else 0)) / (0.75 if task in TASKS[:3] else 1)
                    for task, value in scores.items()]
        base = json.loads((self.root / "base-results.json").read_text())
        report = {"operational_success": True, **base, "chat_accuracy": scores,
            "chatcore": sum(centered) / len(centered), "base_step": FINAL_STEP, "sft_step": step,
            "base_checkpoint": str(self.checkpoint_dir("base")), "sft_checkpoint": str(self.checkpoint_dir("sft")),
            "chat_command": f"NANOCHAT_BASE_DIR={shlex.quote(str(self.root))} NANOCHAT_CUDA_MEMORY_LIMIT_GB=80 NANOCHAT_ATTENTION_IMPL=sdpa {shlex.quote(sys.executable)} -m scripts.chat_cli --device-type=cuda --model-tag={TAG}"}
        write_json(self.root / "report.json", report)
        print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"), nargs="?", default="all")
    parser.add_argument("--resume-from-step", type=int)
    args = parser.parse_args()
    if args.resume_from_step is not None and args.stage != "pretrain":
        parser.error("--resume-from-step is only valid for the pretrain stage")
    root = Path(os.environ["NANOCHAT_BASE_DIR"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / "pipeline.lock", timeout=0):
        pipeline = Pipeline(root, args.resume_from_step)
        pipeline.audit()
        stages = STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            pipeline.current_stage = stage
            write_json(root / "status.json", {"stage": stage, "status": "running", "logs": str(pipeline.log_dir)})
            try:
                getattr(pipeline, stage.replace("-", "_"))()
            except BaseException as exc:
                write_json(root / "status.json", {"stage": stage, "status": "failed", "error": str(exc), "logs": str(pipeline.log_dir)})
                raise
            write_json(root / "status.json", {"stage": stage, "status": "completed", "logs": str(pipeline.log_dir)})


if __name__ == "__main__":
    main()
