"""Generate text with an RLM checkpoint, carrying latent memory across chunks."""

import argparse
import math
import os
from pathlib import Path

from nanochat.paths import configure_runtime_paths, get_run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=get_run_dir() / "train" / "checkpoints",
                        help="checkpoint directory (default: selected run's train/checkpoints)")
    parser.add_argument("--step", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt", default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_tokens < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Require max-tokens >= 0 and a finite temperature >= 0")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("top-k must be positive when supplied")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("RLM sampling supports one process; do not launch with torchrun")
    os.environ["NANOCHAT_ATTENTION_IMPL"] = "sdpa"
    os.environ.setdefault("NANOCHAT_DTYPE", "float32" if args.device == "cpu" else "bfloat16")
    os.environ.setdefault("NANOCHAT_CUDA_MEMORY_LIMIT_GB", "80")
    os.environ.setdefault("NANOCHAT_MIN_SYSTEM_AVAILABLE_GB", "16")
    configure_runtime_paths()

    from nanochat.common import compute_cleanup, compute_init
    from nanochat.rlm_inference import RLMGenerator, load_rlm_checkpoint
    from nanochat.training_runtime import check_training_memory

    _, _, _, _, device = compute_init(args.device)
    try:
        check_training_memory(device, 0)
        model, tokenizer, _ = load_rlm_checkpoint(args.checkpoint_dir.expanduser(), args.step, device)
        check_training_memory(device, 0)
        tokens = tokenizer.encode(args.prompt, prepend=tokenizer.get_bos_token_id())
        generated = []
        stream = RLMGenerator(model).generate(
            tokens, max_tokens=args.max_tokens, temperature=args.temperature,
            top_k=args.top_k, seed=args.seed,
            stop_token_ids=frozenset({tokenizer.get_bos_token_id()}),
        )
        for index, token in enumerate(stream, start=1):
            check_training_memory(device, index)
            generated.append(token)
        print(args.prompt + tokenizer.decode(generated))
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
