"""Finite held-out RLM episodes that stay within individual documents."""

from collections.abc import Iterator

import torch

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer


def contiguous_validation_batches(
    tokenizer: RustBPETokenizer,
    batch_size: int,
    episode_length: int,
    device: torch.device,
    tokenizer_batch_size: int = 128,
    tokenizer_threads: int = 4,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield one BOS-prefixed episode per sufficiently long validation document.

    Documents are visited once in parquet order. Each contributes at most its
    first episode_length + 1 tokens, including BOS and the final shifted target.
    Short documents are skipped, and a final incomplete batch is discarded.
    Each yielded batch owns its storage; later batches cannot overwrite it.
    """
    if batch_size <= 0 or episode_length <= 0:
        raise ValueError("Contiguous evaluation needs positive batch_size and episode_length")
    if tokenizer_batch_size <= 0 or tokenizer_threads <= 0:
        raise ValueError("Tokenizer batch size and thread count must be positive")
    required_tokens = episode_length + 1
    bos = tokenizer.get_bos_token_id()
    rows = []
    for documents in parquets_iter_batched("val"):
        for start in range(0, len(documents), tokenizer_batch_size):
            encoded = tokenizer.encode(
                documents[start:start + tokenizer_batch_size],
                prepend=bos, num_threads=tokenizer_threads,
            )
            for tokens in encoded:
                if len(tokens) < required_tokens:
                    continue
                rows.append(tokens[:required_tokens])
                if len(rows) == batch_size:
                    batch = torch.tensor(rows, dtype=torch.long, device=device)
                    yield batch[:, :-1], batch[:, 1:]
                    rows = []
