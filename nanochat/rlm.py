"""A shared GPT backbone with differentiable memory between bounded text chunks."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig


@dataclass
class RLMConfig(GPTConfig):
    window_pattern: str = "L"
    text_length: int = 1023
    memory_tokens: int = 512
    activation_checkpointing: bool = False

    def __post_init__(self):
        dimensions = (
            "sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head",
            "n_embd", "text_length", "memory_tokens",
        )
        for name in dimensions:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"RLM {name} must be a positive integer")
        if self.n_embd < 24:
            raise ValueError("RLM n_embd must be at least 24 for the smear gate")
        if self.n_embd % self.n_head:
            raise ValueError("RLM n_embd must be divisible by n_head")
        if self.n_kv_head > self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("RLM n_kv_head must divide n_head and cannot exceed it")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("RLM attention head dimension must be even for rotary embeddings")
        self.window_pattern = self.window_pattern.upper()
        if not self.window_pattern or any(char != "L" for char in self.window_pattern):
            raise ValueError("RLM requires full causal attention (window_pattern='L')")
        required_positions = self.text_length + 1 + 2 * self.memory_tokens
        if self.sequence_len < required_positions:
            raise ValueError(
                f"RLM sequence_len must cover EOM, text, and both memory blocks: "
                f"{self.sequence_len} < {required_positions}"
            )


@dataclass
class CycleOutput:
    """Optional outgoing memory plus joint [EOM, text] logits or their loss."""

    memory: torch.Tensor | None
    logits: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class RecurrentGPT(GPT):
    """Reset between episodes; preserve the complete gradient path within an episode."""

    def __init__(self, config: RLMConfig, pad_vocab_size_to=64):
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        self.summary_embedding = nn.Parameter(torch.empty(config.n_embd))
        self.eom_embedding = nn.Parameter(torch.empty(config.n_embd, dtype=torch.float32))

    @torch.no_grad()
    def init_weights(self):
        super().init_weights()
        # One shared query, kept in fp32 like the other small model-owned parameters.
        nn.init.normal_(self.summary_embedding, mean=0.0, std=0.8)
        nn.init.normal_(self.eom_embedding, mean=0.0, std=0.8)

    def _extra_embedding_parameters(self):
        return [self.summary_embedding, self.eom_embedding]

    @staticmethod
    def cycle_targets(text_ids, targets):
        """Label EOM with the first input, followed by all shifted text targets."""
        if text_ids.ndim != 2 or text_ids.shape != targets.shape:
            raise ValueError("RLM targets must match the text_ids shape")
        if text_ids.size(1) == 0:
            raise ValueError("An empty RLM text chunk cannot have supervised targets")
        return torch.cat((text_ids[:, :1], targets), dim=1)

    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """Generate with recurrent rollover; requires model.eval()."""
        # Imported lazily because the inference module also loads RLM models.
        from nanochat.rlm_inference import RLMGenerator

        return RLMGenerator(self).generate(tokens, max_tokens, temperature, top_k, seed)

    def forward_cycle(self, text_ids, memory=None, targets=None, loss_reduction='mean', *, write_memory=True):
        """Read prior memory through EOM, score next tokens, and write the next state.

        Text may be shorter than text_length for bounded-context generation.
        Logits cover EOM followed by every text position, in one projection.
        The caller supplies shifted text targets; EOM predicts the first input
        in every cycle, including initial BOS, without masking or reweighting.
        Empty text is allowed only without targets for an EOM boundary readout;
        its outgoing summary must not replace the caller's memory state.
        Set write_memory=False when outgoing memory has no consumer: summary
        inputs are omitted and the returned memory is None. An empty initial
        readout without incoming memory requires write_memory=True.
        With reduction='none', loss is flattened in batch-major order, like GPT.
        """
        if text_ids.ndim != 2 or text_ids.size(0) == 0:
            raise ValueError("RLM text_ids must have shape (nonempty batch, text length)")
        batch, text_length = text_ids.shape
        if not 0 <= text_length <= self.config.text_length:
            raise ValueError(f"RLM cycle text length must be between 0 and {self.config.text_length}")
        if text_ids.device != self.get_device():
            raise ValueError("RLM text_ids and model must be on the same device")
        if targets is not None and targets.shape != text_ids.shape:
            raise ValueError("RLM targets must match the text_ids shape")
        if not write_memory and text_length == 0 and memory is None:
            raise ValueError("An empty initial EOM readout requires write_memory=True")
        joint_targets = None if targets is None else self.cycle_targets(text_ids, targets)

        text = self.transformer.wte(text_ids).to(COMPUTE_DTYPE)
        eom = self.eom_embedding.to(text.dtype).view(1, 1, -1).expand(batch, 1, -1)
        parts = [eom, text]
        summary_tokens = self.config.memory_tokens if write_memory else 0
        if write_memory:
            summary = self.summary_embedding.to(text.dtype).view(1, 1, -1)
            parts.append(summary.expand(batch, summary_tokens, -1))
        prefix_tokens = 0
        if memory is not None:
            expected_shape = (batch, self.config.memory_tokens, self.config.n_embd)
            if memory.shape != expected_shape:
                raise ValueError(f"RLM memory must have shape {expected_shape}")
            if memory.device != text.device or memory.dtype != text.dtype:
                raise ValueError("RLM memory must match the text embedding device and compute dtype")
            prefix_tokens = self.config.memory_tokens
            parts.insert(0, memory)
        x = torch.cat(parts, dim=1)

        positions = x.size(1)
        cos_sin = self.cos[:, :positions], self.sin[:, :positions]
        hidden = self._forward_embeddings(
            x, text_ids, cos_sin,
            prefix_tokens=prefix_tokens + 1,
            suffix_tokens=summary_tokens,
            activation_checkpointing=self.config.activation_checkpointing,
        )
        next_memory = hidden[:, -self.config.memory_tokens:] if write_memory else None
        logits = self._compute_logits(hidden[:, prefix_tokens:prefix_tokens + 1 + text_length])
        if targets is None:
            return CycleOutput(memory=next_memory, logits=logits)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), joint_targets.reshape(-1),
            ignore_index=-1, reduction=loss_reduction,
        )
        return CycleOutput(memory=next_memory, loss=loss)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        """Run one complete episode per row, without storing recurrent module state.

        Episode length must be a positive multiple of text_length.
        Returns concatenated [EOM, text] logits for each cycle, or
        mean/sum/flattened per-target loss. EOM and all valid shifted text
        targets receive equal weight, including masked text episodes.
        The final cycle omits its summary because this episode discards it.
        """
        if idx.ndim != 2 or idx.size(1) == 0 or idx.size(1) % self.config.text_length:
            raise ValueError("RLM episodes must contain complete text_length chunks")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("RLM episode targets must match the input shape")
        if loss_reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"Unsupported RLM loss reduction: {loss_reduction}")
        memory = None
        outputs = []
        for start in range(0, idx.size(1), self.config.text_length):
            stop = start + self.config.text_length
            cycle = self.forward_cycle(
                idx[:, start:stop], memory,
                None if targets is None else targets[:, start:stop],
                loss_reduction='none',
                write_memory=stop < idx.size(1),
            )
            memory = cycle.memory
            outputs.append(cycle.logits if targets is None else cycle.loss.reshape(idx.size(0), -1))
        result = torch.cat(outputs, dim=1)
        if targets is None:
            return result
        if loss_reduction == 'none':
            return result.reshape(-1)
        if loss_reduction == 'sum':
            return result.sum()
        eom_targets = idx.size(0) * (idx.size(1) // self.config.text_length)
        return result.sum() / ((targets != -1).sum() + eom_targets)
