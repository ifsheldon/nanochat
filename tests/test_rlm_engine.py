from types import SimpleNamespace

import pytest
import torch

from nanochat.engine import KVCache
from nanochat.rlm import RLMConfig, RecurrentGPT
from nanochat.rlm_inference import RLMGenerator, RLMEngine, predict_next_logits


class ToyChatTokenizer:
    """Byte/ID tokenizer with nanochat chat and calculator special tokens in [0, vocab_size)."""

    SPECIALS = {
        "<|python_start|>": 56,
        "<|python_end|>": 57,
        "<|output_start|>": 58,
        "<|output_end|>": 59,
        "<|assistant_start|>": 60,
        "<|assistant_end|>": 61,
        "<|user_start|>": 62,
        "<|bos|>": 63,
    }
    CHARS = "0123456789+-*/(). ABCD="

    def __init__(self):
        self._char_to_id = {ch: idx + 1 for idx, ch in enumerate(self.CHARS)}
        self._id_to_char = {idx + 1: ch for idx, ch in enumerate(self.CHARS)}

    def encode_special(self, token: str) -> int:
        return self.SPECIALS[token]

    def get_bos_token_id(self) -> int:
        return self.SPECIALS["<|bos|>"]

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        ids = [self._char_to_id[ch] for ch in text]
        return [prepend, *ids] if prepend is not None else ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._id_to_char.get(token, "") for token in ids if token in self._id_to_char)

    def render_for_completion(self, conversation: dict) -> list[int]:
        user_tokens = self.encode(conversation["messages"][0]["content"])
        return [self.get_bos_token_id(), self.SPECIALS["<|user_start|>"], *user_tokens, self.SPECIALS["<|assistant_start|>"]]


def active_tiny_rlm(*, n_kv_head: int = 2) -> RecurrentGPT:
    torch.manual_seed(7)
    config = RLMConfig(
        sequence_len=11,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_kv_head=n_kv_head,
        n_embd=32,
        text_length=4,
        memory_tokens=3,
    )
    model = RecurrentGPT(config)
    model.init_weights()
    with torch.no_grad():
        # Activate smear, x0 blending, backout, and projections so cached decode
        # exercises every cross-position pathway.
        model.smear_lambda.fill_(0.35)
        model.smear_gate.weight.normal_(std=0.2)
        model.backout_lambda.fill_(0.25)
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
            if block.attn.ve_gate is not None:
                block.attn.ve_gate.weight.normal_(std=0.2)
    return model.eval()


@pytest.mark.parametrize("n_kv_head", [4, 2])
@pytest.mark.parametrize("prompt_len", [1, 3, 4, 5, 8, 9])
def test_rlm_engine_matches_uncached_generator_across_rollovers(n_kv_head, prompt_len):
    model = active_tiny_rlm(n_kv_head=n_kv_head)
    tokenizer = ToyChatTokenizer()
    # Make tokens 0..54 strongly preferred over special tokens 55..63 (which stay at 0)
    # so top_k=10 never samples python_start (56), assistant_end (61), or bos (63),
    # exercising all 9-10 steps across multiple chunk rollovers uninterrupted.
    with torch.no_grad():
        model.lm_head.weight[:55].normal_(std=1.5)
        model.lm_head.weight[55:].zero_()
    prompt = [(i * 3 + 1) % 50 for i in range(prompt_len)]

    greedy_ref = list(RLMGenerator(model).generate(prompt, max_tokens=10, temperature=0.0))
    greedy_cached, masks = RLMEngine(model, tokenizer).generate_batch(
        prompt, num_samples=1, max_tokens=10, temperature=0.0,
    )
    assert greedy_cached[0][len(prompt):] == greedy_ref
    assert masks[0] == [0] * len(prompt) + [1] * len(greedy_ref)

    sampled_ref = list(RLMGenerator(model).generate(prompt, max_tokens=9, temperature=0.8, top_k=10, seed=123))
    sampled_cached, _ = RLMEngine(model, tokenizer).generate_batch(
        prompt, num_samples=1, max_tokens=9, temperature=0.8, top_k=10, seed=123,
    )
    assert sampled_cached[0][len(prompt):] == sampled_ref


def test_intra_cycle_kv_cache_reset_prevents_stale_leakage():
    model = active_tiny_rlm()
    cfg = model.config
    cache = KVCache(
        batch_size=1,
        num_heads=cfg.n_kv_head,
        seq_len=cfg.memory_tokens + 1 + cfg.text_length,
        head_dim=cfg.n_embd // cfg.n_head,
        num_layers=cfg.n_layer,
        device=model.get_device(),
        dtype=model.cos.dtype,
    )
    chunk = torch.tensor([[2, 5, 8, 11]])
    memory = model.forward_cycle(chunk, memory=None, write_memory=True).memory.clone()

    # Poison cache buffers and smear state, then reset before prefilling cycle 2.
    cache.k_cache.fill_(999.0)
    cache.v_cache.fill_(-999.0)
    cache.prev_embedding = torch.full((1, 1, cfg.n_embd), 777.0, dtype=model.cos.dtype)
    cache.advance(6)
    cache.reset()

    empty = torch.empty((1, 0), dtype=torch.long)
    eom_logits = model.prefill_cycle(empty, memory=memory, kv_cache=cache)
    ref_eom = model.forward_cycle(empty, memory=memory, write_memory=False).logits[:, 0]
    torch.testing.assert_close(eom_logits, ref_eom, atol=1e-5, rtol=1e-5)

    step1 = model.decode_token(torch.tensor([[14]]), kv_cache=cache)
    step2 = model.decode_token(torch.tensor([[19]]), kv_cache=cache)
    ref_two = model.forward_cycle(torch.tensor([[14, 19]]), memory=memory, write_memory=False).logits
    torch.testing.assert_close(step1, ref_two[:, 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(step2, ref_two[:, 2], atol=1e-5, rtol=1e-5)


def test_rlm_engine_injects_calculator_output_across_chunk_rollover():
    tokenizer = ToyChatTokenizer()
    py_start = tokenizer.encode_special("<|python_start|>")
    py_end = tokenizer.encode_special("<|python_end|>")
    out_start = tokenizer.encode_special("<|output_start|>")
    out_end = tokenizer.encode_special("<|output_end|>")
    end_tok = tokenizer.encode_special("<|assistant_end|>")
    expr_ids = tokenizer.encode("6*7")
    expected_out_ids = [out_start, *tokenizer.encode("42"), out_end]

    scripted = [py_start, *expr_ids, py_end, *tokenizer.encode("9"), end_tok]

    class ScriptedRLM:
        def __init__(self):
            self.config = SimpleNamespace(
                vocab_size=64, text_length=4, memory_tokens=2,
                n_kv_head=2, n_head=2, n_embd=16, n_layer=2,
            )
            self.training = False
            self.sampled_cursor = 0
            self.consolidations = []

        def get_device(self):
            return torch.device("cpu")

        def _next_logits(self, batch: int) -> torch.Tensor:
            logits = torch.full((batch, 64), -20.0)
            target = scripted[min(self.sampled_cursor, len(scripted) - 1)]
            logits[:, target] = 20.0
            return logits

        def forward_cycle(self, text_ids, memory=None, *, write_memory=True):
            assert write_memory
            assert text_ids.shape[1] == self.config.text_length
            self.consolidations.append(text_ids.tolist())
            mem = torch.zeros((text_ids.size(0), 2, 16))
            return SimpleNamespace(memory=mem, logits=None)

        def prefill_cycle(self, text_ids, memory=None, *, kv_cache):
            kv_cache.advance((2 if memory is not None else 0) + 1 + text_ids.size(1))
            return self._next_logits(text_ids.size(0))

        def decode_token(self, token_ids, *, kv_cache):
            kv_cache.advance(1)
            return self._next_logits(token_ids.size(0))

    model = ScriptedRLM()
    engine = RLMEngine(model, tokenizer)
    prompt = [63, 1, 2]
    generated = []
    masks = []
    for col, mask_col in engine.generate(prompt, num_samples=1, max_tokens=20, temperature=0.0):
        if mask_col[0] == 1:
            model.sampled_cursor += 1
        if col[0] == end_tok:
            break
        generated.append(col[0])
        masks.append(mask_col[0])

    assert generated == [py_start, *expr_ids, py_end, *expected_out_ids, *tokenizer.encode("9")]
    assert masks == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
    assert len(model.consolidations) >= 2


def test_predict_next_logits_matches_individual_prompts_across_chunk_boundaries():
    model = active_tiny_rlm()
    prompts = [
        [3],                            # < L
        [4, 7, 11],                     # < L
        [2, 6, 9, 13],                  # == L (exact boundary -> EOM readout)
        [5, 8, 12, 15, 19],             # > L
        [1, 2, 3, 4, 5, 6, 7, 8],       # == 2 * L (second exact boundary)
        [9, 8, 7, 6, 5, 4, 3, 2, 10],   # > 2 * L
    ]
    batched = predict_next_logits(model, prompts, pad_token_id=63)
    assert batched.shape == (len(prompts), model.config.vocab_size)

    for idx, prompt in enumerate(prompts):
        single = predict_next_logits(model, [prompt], pad_token_id=0)[0]
        torch.testing.assert_close(batched[idx], single, atol=1e-5, rtol=1e-5)
        ref_top1 = next(RLMGenerator(model).generate(prompt, max_tokens=1, temperature=0.0))
        assert int(batched[idx].argmax().item()) == ref_top1
