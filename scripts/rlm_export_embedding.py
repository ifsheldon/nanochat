"""Export a verified input embedding and tokenizer without the full checkpoint."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New bundle directory in an existing parent; never overwrite")
    args = parser.parse_args()
    if args.step < 0:
        parser.error("--step must be nonnegative")

    from nanochat.rlm_initialization import TokenEmbeddingInitialization

    output_dir = TokenEmbeddingInitialization.export_from_checkpoint(
        args.checkpoint_dir, args.step, args.tokenizer_dir, args.output_dir,
    )
    size = sum(path.stat().st_size for path in output_dir.iterdir())
    print(f"Published input embedding bundle: {output_dir}")
    print(f"Bundle size: {size:,} bytes; original model and optimizer are not included")


if __name__ == "__main__":
    main()
