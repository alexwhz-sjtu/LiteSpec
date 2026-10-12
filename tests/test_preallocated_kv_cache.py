"""Cache rollback, storage stability, and evaluation decode parity."""

import time

import pytest
import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from evaluation import benchmark
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.modeling.draft.dflash2 import DFlash2DraftModel
from litespec.modeling.draft.dlite import DLiteDraftModel
from litespec.modeling.draft.dspark import DSparkDraftModel
from litespec.modeling.kv_cache import PreallocatedLayer, make_target_cache


def tiny_config(**kwargs):
    return Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        **kwargs,
    )


def test_layer_rollback_reset_and_capacity():
    layer = PreallocatedLayer(12)
    key = torch.randn(2, 2, 8, 4)
    value = torch.randn_like(key)
    layer.update(key, value)
    pointers = (layer.keys.data_ptr(), layer.values.data_ptr())
    layer.crop(-5)
    assert layer.get_seq_length() == 3
    replacement = torch.randn(2, 2, 1, 4)
    keys, values = layer.update(replacement, -replacement)
    torch.testing.assert_close(keys, torch.cat((key[:, :, :3], replacement), dim=2))
    torch.testing.assert_close(values, torch.cat((value[:, :, :3], -replacement), dim=2))
    assert layer.get_mask_sizes(torch.arange(8)) == (12, 0)
    layer.update(key, value)  # exact capacity
    with pytest.raises(ValueError, match="capacity exceeded"):
        layer.update(replacement, replacement)
    assert layer.get_seq_length() == 12
    layer.reset()
    keys, values = layer.update(replacement, -replacement)
    torch.testing.assert_close(keys, replacement)
    torch.testing.assert_close(values, -replacement)
    assert pointers == (layer.keys.data_ptr(), layer.values.data_ptr())


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@torch.inference_mode()
def test_target_logits_parity(backend, batch_size, device):
    torch.manual_seed(0)
    config = tiny_config()
    config._attn_implementation = backend
    target = Qwen3ForCausalLM(config).to(device).eval()
    dynamic = DynamicCache()
    fixed = make_target_cache(target, 32)
    tokens = torch.randint(0, 31, (batch_size, 24), device=device)

    def forward(start, length):
        positions = torch.arange(start, start + length, device=device)
        kwargs = dict(
            input_ids=tokens[:, start : start + length],
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
        )
        expected = target(**kwargs, past_key_values=dynamic).logits
        actual = target(**kwargs, past_key_values=fixed).logits
        torch.testing.assert_close(actual, expected)
        assert torch.equal(actual.argmax(-1), expected.argmax(-1))

    forward(0, 15)
    pointers = fixed.storage_ptrs()
    forward(15, 8)
    dynamic.crop(20)
    fixed.crop(20)
    forward(20, 1)
    dynamic = DynamicCache()
    fixed.reset()
    forward(0, 1)
    assert fixed.storage_ptrs() == pointers


@pytest.mark.parametrize("draft_class", [DLiteDraftModel, DFlashDraftModel, DFlash2DraftModel, DSparkDraftModel])
@torch.inference_mode()
def test_speculative_decode_parity(draft_class):
    torch.manual_seed(5)
    target = Qwen3ForCausalLM(tiny_config()).eval()
    method = dict(block_size=4, target_layer_ids=[1], mask_token_id=31)
    if draft_class is DLiteDraftModel:
        method.update(
            architecture_version="dlite_v1", model_role="student",
            chs_num_layers=1, sequential_head="rnn", sequential_rank=8,
        )
    elif draft_class is DFlash2DraftModel:
        method.update(
            conv_kernel_size=2, conv_group_size=4,
            selector_rank=8, selector_top_k=4,
        )
    config = tiny_config(
        architectures=[draft_class.__name__], num_target_layers=2,
        block_size=4, **{draft_class.method_config_key: method},
    )
    draft = draft_class(config).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    kwargs = dict(
        target=target, input_ids=tokens, max_new_tokens=9,
        stop_token_ids=[], temperature=0.0,
    )
    expected = draft.spec_generate(**kwargs)
    expected_accepts = draft.get_last_decode_stats()["accept_lengths"]
    fixed = make_target_cache(target, tokens.shape[1] + 9 + 4)
    pointers = []
    hook = target.register_forward_hook(lambda *_: pointers.append(fixed.storage_ptrs()))
    try:
        actual = draft.spec_generate(**kwargs, target_cache=fixed)
    finally:
        hook.remove()
    assert torch.equal(actual, expected)
    assert draft.get_last_decode_stats()["accept_lengths"] == expected_accepts
    assert len(fixed.storage_ptrs()) == len(target.model.layers)
    assert len(pointers) > 2
    assert all(value == pointers[0] for value in pointers)


@torch.inference_mode()
def test_baseline_uses_preallocated_cache(monkeypatch):
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(tiny_config()).eval()
    monkeypatch.setattr(benchmark, "cuda_time", time.perf_counter)
    kwargs = dict(
        target=target, input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=9, stop_token_ids=[],
    )
    caches = []

    def record_cache(target, max_length):
        cache = make_target_cache(target, max_length)
        caches.append(cache)
        return cache

    monkeypatch.setattr(benchmark, "make_target_cache", record_cache)
    actual = benchmark.target_generate(**kwargs)
    assert caches[0].get_seq_length() == 11
    monkeypatch.setattr(benchmark, "make_target_cache", lambda *_: DynamicCache())
    expected = benchmark.target_generate(**kwargs)
    assert torch.equal(actual.output_ids, expected.output_ids)
