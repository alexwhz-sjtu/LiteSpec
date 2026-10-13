import argparse
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from litespec.training.recipes._dflash_common import (
    add_family_args,
    build_family_config,
    selector_alpha,
    validate_family_args,
)
from litespec.training.setup import select_tp_rank_batch
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dspark import OnlineDSparkModel
from litespec.core._dflash_common import _DFlashSelectorDiagnostics
from litespec.core._dflash_common import _DFlashUnaryDiagnostics
from litespec.distributed.transport import FamilyBatchPacket, FamilyPacketSpec
from litespec.modeling.config_utils import _qwen35_text_dict_to_qwen3_config
from litespec.modeling.draft import dflash as dflash_module
from litespec.modeling.draft import dlite as dlite_module
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.modeling.draft.dflash2 import DFlash2DraftModel, DFlashGroupedConv
from litespec.modeling.draft.dspark import (
    DSparkDraftModel,
    GatedMarkovHead,
    RNNMarkovHead,
    VanillaMarkovHead,
)


def tiny_config(architecture="DFlashDraftModel", **method_overrides):
    method = {
        "block_size": 4,
        "target_layer_ids": [1],
        "mask_token_id": 31,
        "attention_mode": "gqa",
        **method_overrides,
    }
    return Qwen3Config(
        architectures=[architecture],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        block_size=4,
        layer_types=["full_attention"],
        **{({"DFlashDraftModel": "dflash_config", "DFlash2DraftModel": "dflash2_config", "DSparkDraftModel": "dspark_config"}[architecture]): method},
    )


def test_dflash_and_dlite_share_flex_attention_runtime():
    assert (
        dflash_module.compile_friendly_flex_attention
        is dlite_module.compile_friendly_flex_attention
    )


def test_family_algorithms_accept_tp_draft_sharding():
    for algorithm in ("dflash", "dflash2", "dspark"):
        parser = argparse.ArgumentParser()
        add_family_args(parser, algorithm)
        args = parser.parse_args(
            [
                "--target-model-path",
                "target",
                "--target-model-backend",
                "sglang",
                "--train-data-path",
                "train.jsonl",
                "--output-dir",
                "out",
                "--tp-size",
                "2",
                "--batch-size",
                "2",
                "--shard-draft-by-tp",
            ]
        )
        validate_family_args(parser, args, algorithm)
        assert args.shard_draft_by_tp is True

    batch = torch.arange(12).view(3, 4)
    torch.testing.assert_close(select_tp_rank_batch(batch, 1), batch[1:2])


def test_dflash_and_dflash2_build_expected_modules():
    base = DFlashDraftModel(tiny_config())
    assert base.block_size == 4
    model = DFlash2DraftModel(
        tiny_config(
            "DFlash2DraftModel",
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=3,
        )
    )
    assert isinstance(model.layers[0].attention_conv, DFlashGroupedConv)
    assert model.candidate_selector.top_k == 3
    assert torch.count_nonzero(model.candidate_selector.successor_codebook) == 0


def test_dflash2_checkpoint_round_trip(tmp_path):
    model = DFlash2DraftModel(
        tiny_config(
            "DFlash2DraftModel",
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=3,
        )
    )
    model.save_pretrained(tmp_path)
    loaded, info = DFlash2DraftModel.from_pretrained(
        tmp_path, attn_implementation="eager", output_loading_info=True
    )
    assert not info["missing_keys"]
    assert not info["unexpected_keys"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)


def test_dspark_builds_all_markov_head_variants():
    classes = {
        "vanilla": VanillaMarkovHead,
        "gated": GatedMarkovHead,
        "rnn": RNNMarkovHead,
    }
    for kind, expected in classes.items():
        config = tiny_config(
            "DSparkDraftModel",
            projector_type="dspark",
            markov_rank=4,
            markov_head_type=kind,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        model = DSparkDraftModel(config)
        assert isinstance(model.markov_head, expected)
        assert model.confidence_head is not None


def test_dspark_anchor_is_unsupervised_and_suffix_is_not_left_shifted():
    wrapper = object.__new__(OnlineDSparkModel)
    nn.Module.__init__(wrapper)
    wrapper.block_size = 4
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    labels, mask, indices = wrapper._build_dspark_labels_and_mask(
        input_ids,
        loss_mask,
        anchor_positions=torch.tensor([[1]]),
        block_keep_mask=torch.tensor([[True]]),
    )
    assert indices.tolist() == [[[1, 2, 3, 4]]]
    assert labels.tolist() == [[[11, 12, 13, 14]]]
    assert mask.tolist() == [[[False, True, True, True]]]
    anchor = input_ids[:, 1].view(1, 1, 1)
    predecessors = torch.cat([anchor, labels[:, :, :-1]], dim=-1)
    assert predecessors[0, 0, 1].item() == 11
    assert 15 not in labels.flatten().tolist()


def test_dspark_suffix_mask_stops_after_first_gap():
    wrapper = object.__new__(OnlineDSparkModel)
    nn.Module.__init__(wrapper)
    wrapper.block_size = 4
    input_ids = torch.arange(8).unsqueeze(0)
    loss_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.float32)
    _, mask, _ = wrapper._build_dspark_labels_and_mask(
        input_ids,
        loss_mask,
        anchor_positions=torch.tensor([[1]]),
        block_keep_mask=torch.tensor([[True]]),
    )
    assert mask.tolist() == [[[False, True, False, False]]]


def test_dspark_decay_starts_at_first_supervised_slot():
    wrapper = object.__new__(OnlineDSparkModel)
    nn.Module.__init__(wrapper)
    wrapper.block_size = 4
    wrapper.loss_decay_gamma = 1.0
    weights = wrapper._dspark_loss_weight_mask(
        torch.tensor([[[False, True, True, True]]])
    )
    torch.testing.assert_close(
        weights,
        torch.tensor(
            [
                [
                    [
                        0.0,
                        1.0,
                        torch.exp(torch.tensor(-1.0)),
                        torch.exp(torch.tensor(-2.0)),
                    ]
                ]
            ]
        ),
    )


def test_dflash2_expected_accept_len_uses_selector_distribution():
    wrapper = object.__new__(OnlineDFlashModel)
    nn.Module.__init__(wrapper)
    wrapper.lm_head = nn.Identity()
    target_probs = torch.tensor([0.5, 0.3, 0.2])
    target_logits = target_probs.log().view(1, 1, 1, 3).expand(1, 1, 3, 3)
    unary = _DFlashUnaryDiagnostics(
        probabilities=target_probs.view(1, 1, 1, 3).expand(1, 1, 3, 3),
        hard_label_probability=torch.zeros(1, 1, 3),
        predicted_ids=torch.zeros(1, 1, 3, dtype=torch.long),
        topk_logits=torch.zeros(1, 1, 3, 2),
        candidate_ids=torch.tensor([0, 1]).view(1, 1, 1, 2).expand(1, 1, 3, 2),
        target_is_candidate=torch.ones(1, 1, 3, dtype=torch.bool),
    )
    selector = _DFlashSelectorDiagnostics(
        selected_ids=torch.zeros(1, 1, 3, dtype=torch.long),
        covered_mask=torch.ones(1, 1, 3),
        logits=torch.tensor([0.625, 0.375]).log().view(1, 1, 1, 2).expand(1, 1, 3, 2),
    )
    terms = wrapper._dflash_teacher_terms(
        unary=unary,
        selector=selector,
        greedy_ids=None,
        aligned_target_hidden=target_logits,
        weight_mask=torch.tensor([[[0.0, 1.0, 1.0]]]),
        position_den=torch.tensor([0.0, 1.0, 1.0]),
    )
    # Sparse selector overlap is 0.8 per slot: 1 + 0.8 + 0.8^2.
    torch.testing.assert_close(
        terms.final_expected_accepted_length_num, torch.tensor(2.44)
    )
    # Unary q equals target p, proving the final metric did not use unary q.
    torch.testing.assert_close(terms.expected_accepted_length_num, torch.tensor(3.0))


def test_family_packet_has_no_logits_and_optional_last_hidden():
    spec = FamilyPacketSpec(2, 8, 48, 16, True)
    packet = FamilyBatchPacket.empty(spec, device="cpu")
    assert packet.target_context_hidden.shape == (2, 8, 48)
    assert packet.target_last_hidden.shape == (2, 8, 16)
    assert all(tensor.shape[0] == 2 for tensor in packet.tensors())
    assert not hasattr(packet, "logits")


def test_ratio_family_only_builds_top16_recall_per_position():
    metrics = {}
    numerators = torch.tensor([0.0, 1.0, 2.0])
    denominators = torch.tensor([0.0, 2.0, 4.0])

    OnlineDFlashModel._add_ratio_family(
        metrics,
        "dflash/hard_label/unary_top1_accuracy",
        numerators,
        denominators,
    )
    OnlineDFlashModel._add_ratio_family(
        metrics,
        "dflash/hard_label/unary_top16_recall",
        numerators,
        denominators,
    )

    assert {name for name in metrics if name.startswith("position_")} == {
        "position_1/hard_label/unary_top16_recall",
        "position_2/hard_label/unary_top16_recall",
    }


def test_generated_defaults_and_selector_schedule():
    source = tiny_config()
    source.num_hidden_layers = 5
    source.num_target_layers = 36
    source.layer_types = ["full_attention"] * 5
    args = SimpleNamespace(
        target_model_path="unused",
        trust_remote_code=False,
        num_draft_layers=5,
        block_size=16,
        target_layer_ids=None,
        attention_backend="eager",
        attention_mode="gqa",
        mask_token_id=None,
        conv_kernel_size=2,
        conv_group_size=16,
        selector_rank=256,
        selector_top_k=16,
        markov_head_type="vanilla",
        markov_rank=256,
        confidence_head=True,
        confidence_head_with_markov=True,
        confidence_loss_alpha=1.0,
        backbone_conv=False,
        selector_loss_alpha=1.0,
        selector_warmup_ratio=0.2,
        selector_ramp_ratio=0.3,
    )
    config = build_family_config(args, "dflash2", source_config=source)
    assert config.architectures == ["DFlash2DraftModel"]
    assert config.dflash2_config["selector_rank"] == 256
    assert len(config.dflash2_config["target_layer_ids"]) == 5
    assert selector_alpha(args, 0, 10) == 0
    assert selector_alpha(args, 2, 10) == 1 / 3
    assert selector_alpha(args, 5, 10) == 1


def test_qwen35_normalized_config_builds_dflash_draft():
    normalized = _qwen35_text_dict_to_qwen3_config(
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "max_position_embeddings": 64,
            },
        }
    )
    normalized.num_target_layers = 4
    args = SimpleNamespace(
        target_model_path="unused",
        trust_remote_code=False,
        num_draft_layers=1,
        block_size=4,
        target_layer_ids=None,
        attention_backend="eager",
        attention_mode="gqa",
        mask_token_id=31,
        conv_kernel_size=2,
        conv_group_size=4,
        selector_rank=4,
        selector_top_k=3,
        markov_head_type="vanilla",
        markov_rank=4,
        confidence_head=True,
        confidence_head_with_markov=True,
        confidence_loss_alpha=1.0,
        backbone_conv=False,
    )
    config = build_family_config(args, "dflash", source_config=normalized)
    assert config.target_source_model_type == "qwen3_5"
    assert isinstance(DFlashDraftModel(config), DFlashDraftModel)
