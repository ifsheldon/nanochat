"""Run one pipeline command and persist CUDA peaks and structured failure status."""

import json
import runpy
import sys
from pathlib import Path

import torch

from nanochat.training_runtime import memory_status


def main():
    module, output, *arguments = sys.argv[1:]
    sys.argv = [module, *arguments]
    result = {"module": module, "success": False}
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        result["success"] = True
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), oom=isinstance(exc, torch.OutOfMemoryError))
        raise
    finally:
        try:
            result.update(memory_status(torch.device("cuda")))
        finally:
            Path(output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
