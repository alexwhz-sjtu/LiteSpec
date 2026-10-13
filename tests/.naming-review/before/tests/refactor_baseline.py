"""Capture deterministic CPU model outputs for a structural refactor."""

import importlib
import sys

import torch
from transformers import Qwen3Config


def capture(package="litespec", *, models_only=False, config_keys=None):
    result = {}
    for algorithm, architecture in (
        ("dflash", "DFlashDraftModel"),
        ("dflash2", "DFlash2DraftModel"),
        ("dspark", "DSparkDraftModel"),
        ("dlite", "DLiteDraftModel"),
    ):
        torch.manual_seed(20261010)
        cls = getattr(
            importlib.import_module(f"{package}.modeling.draft.{algorithm}"),
            architecture,
        )
        method = dict(
            block_size=4,
            target_layer_ids=[1],
            mask_token_id=31,
            attention_mode="gqa",
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=4,
            selector_top_k=3,
            markov_rank=4,
            markov_head_type="rnn",
            projector_type="dspark" if algorithm == "dspark" else None,
            enable_confidence_head=True,
            confidence_head_with_markov=True,
        )
        if algorithm == "dlite":
            method = dict(
                architecture_version="dlite_v1",
                model_role="student",
                chs_num_layers=1,
                target_layer_ids=[1],
                sequential_head="rnn",
                sequential_rank=4,
                mask_token_id=31,
            )
        key = (config_keys or {}).get(algorithm, f"{algorithm}_config")
        config = Qwen3Config(
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
            **{key: method},
        )
        config._attn_implementation = "eager"
        model = cls(config)
        if models_only:
            result[algorithm] = model
            continue
        result[algorithm] = {
            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}
        }
        # Exercise the shared backbone using a dense, unrestricted tiny mask.
        noise = torch.randn(1, 4, 16)
        target = torch.randn(1, 1, 16)
        if algorithm == "dlite":
            target = target.view(1, 1, 1, 16)
        output = model(
            position_ids=torch.arange(4 if algorithm == "dlite" else 5).view(1, -1),
            noise_embedding=noise,
            target_hidden=target,
            attention_mask=torch.zeros(1, 1, 4, 5),
            rotary_position_ids=torch.arange(5).view(1, -1),
            **(
                {"history_hidden": torch.empty(1, 1, 0, 16)}
                if algorithm == "dlite"
                else {}
            ),
        )
        hidden = (
            output.last_hidden_state if hasattr(output, "last_hidden_state") else output
        )
        loss = hidden.float().square().mean()
        loss.backward()
        result[algorithm].update(
            output=hidden.detach(),
            loss=loss.detach(),
            gradients={
                k: p.grad.detach().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            },
        )
    return result


def capture_training(models, package="litespec", *, wrappers=None):
    """CPU objective parity, independent from the attention backend benchmark."""
    wrappers = wrappers or {
        "dflash": ("dflash", "OnlineDFlashModel"),
        "dflash2": ("dflash2", "OnlineDFlash2Model"),
        "dspark": ("dspark", "OnlineDSparkModel"),
        "dlite": ("dlite", "OnlineDLiteModel"),
    }
    result = {}
    for algorithm, model in models.items():
        torch.manual_seed(20261010)
        module, name = wrappers[algorithm]
        cls = getattr(importlib.import_module(f"{package}.core.{module}"), name)
        wrapper = cls(
            model,
            torch.nn.Linear(16, 32, bias=False),
            torch.nn.Embedding(32, 16),
            mask_token_id=31,
            block_size=4,
            num_anchors=3,
            attention_backend="flex_attention" if algorithm == "dlite" else "eager",
            **({} if algorithm == "dlite" else {"objective_chunk_blocks": 2}),
        )
        ids = torch.arange(12).view(1, -1)
        if algorithm == "dlite":
            batch = wrapper.prepare_batch(
                ids,
                {1: torch.randn(1, 12, 16)},
                torch.ones_like(ids),
                anchor_positions=torch.tensor([[0, 1, 4]]),
                block_keep_mask=torch.ones(1, 3, dtype=torch.bool),
            )
            hidden = torch.randn(1, 3, 3, 16, requires_grad=True)
            output = wrapper.compute_supervised_loss(
                hidden, batch, torch.randn(1, 3, 3, 32)
            )
        else:
            output = wrapper(
                ids,
                torch.randn(1, 12, 16),
                torch.ones_like(ids),
                target_last_hidden_states=torch.randn(1, 12, 16),
            )
        output.loss.backward()
        result[algorithm] = {
            "loss": output.loss.detach(),
            "numerator": output.loss_terms.numerator.detach(),
            "denominator": output.loss_terms.denominator.detach(),
            "gradients": {
                k: p.grad.detach().clone()
                for k, p in model.named_parameters()
                if p.grad is not None
            },
        }
        if algorithm == "dlite":
            result[algorithm]["input_gradient"] = hidden.grad.detach().clone()
    return result


if __name__ == "__main__":
    torch.save(capture(sys.argv[1]), sys.argv[2])
