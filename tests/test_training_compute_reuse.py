"""Verify graph sharing, metrics, and checkpoint replay after projection reuse."""

import copy
from unittest import mock

import pytest
import torch
from torch import nn

from refactor_baseline import capture
from litespec.core.dlite import OnlineDLiteModel, compute_stage1_distillation_loss
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dflash2 import OnlineDFlash2Model


def test_transition_shares_serial_head_and_preserves_combined_gradients():
    torch.manual_seed(10)
    draft = capture(models_only=True)["dlite"]
    wrapper = OnlineDLiteModel(
        draft,
        nn.Linear(16, 32, bias=False),
        nn.Embedding(32, 16),
        mask_token_id=31,
        block_size=4,
    )
    reference = copy.deepcopy(wrapper)
    ids = torch.arange(12).view(1, -1)
    batch = wrapper.prepare_batch(
        ids,
        {1: torch.randn(1, 12, 16)},
        torch.ones_like(ids),
        anchor_positions=torch.tensor([[0, 4]]),
        block_keep_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    hidden = torch.randn(1, 2, 3, 16, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_()
    target = torch.randn(1, 2, 3, 32)
    teacher = torch.randn(6, 32)
    head = wrapper.draft_model.sequential_head
    with (
        mock.patch.object(wrapper, "forward_backbone", return_value=hidden),
        mock.patch.object(head, "project_logits", wraps=head.project_logits) as project,
    ):
        _, logits, supervised = wrapper(
            prepared_batch=batch,
            seq_len=12,
            target_prefill_logits=target,
            target_logits_are_gathered=True,
            return_transition_outputs=True,
        )
        assert project.call_count == 1
    logits_ref = reference.compute_serial_logits(hidden_ref, batch)
    supervised_ref = reference.compute_supervised_loss(hidden_ref, batch, target)

    def objective(logits, supervised):
        kl = compute_stage1_distillation_loss(
            student_serial_logits=logits,
            teacher_serial_logits=teacher,
            raw_weight_mask=batch.raw_weight_mask,
            kl_weight=1.0,
            loss_decay_gamma=2.0,
        )[2]
        return 0.3 * kl + 0.7 * supervised.loss_terms.numerator

    objective(logits, supervised).backward()
    objective(logits_ref, supervised_ref).backward()
    torch.testing.assert_close(logits, logits_ref, rtol=0, atol=0)
    torch.testing.assert_close(supervised.loss, supervised_ref.loss, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad)
    for actual, expected in zip(wrapper.parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("algorithm", ["dflash", "dflash2"])
@pytest.mark.parametrize("chunk_size", [0, 1])
def test_logging_reuses_projection_and_never_replays_metrics(algorithm, chunk_size):
    torch.manual_seed(13)
    draft = capture(models_only=True)[algorithm]
    cls = OnlineDFlashModel if algorithm == "dflash" else OnlineDFlash2Model
    wrapper = cls(
        draft,
        nn.Linear(16, 32, bias=False).requires_grad_(False),
        nn.Embedding(32, 16).requires_grad_(False),
        mask_token_id=31,
        block_size=4,
        attention_backend="sdpa",
        objective_chunk_blocks=chunk_size,
    )
    reference = copy.deepcopy(wrapper)
    hidden = torch.randn(1, 2, 4, 16, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_()
    ids = torch.arange(16).view(1, -1)
    target = torch.randn(1, 16, 16)
    anchors = torch.tensor([[0, 7]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    # Reference deliberately projects again, reproducing the previous metric
    # path while using exactly the same anchors, logits, and parameter state.
    original_metrics = reference._dflash_metric_chunk_terms

    def separate_projection(*args, **kwargs):
        kwargs.pop("logits", None)
        return original_metrics(*args, **kwargs)

    outputs = []
    for model, states, old in ((wrapper, hidden, False), (reference, hidden_ref, True)):
        with (
            mock.patch.object(
                model,
                "_forward_draft_blocks",
                return_value=(
                    anchors,
                    keep,
                    states.reshape(1, 8, 16),
                ),
            ),
            mock.patch.object(
                model.lm_head, "forward", wraps=model.lm_head.forward
            ) as head,
            mock.patch.object(
                model,
                "_dflash_metric_chunk_terms",
                wraps=separate_projection if old else model._dflash_metric_chunk_terms,
            ) as metrics,
        ):
            output = model(
                ids, target, torch.ones_like(ids), target_last_hidden_states=target
            )
            chunks = 2 if chunk_size else 1
            # One draft + one teacher projection; old logging added a draft one.
            assert head.call_count == chunks * (3 if old else 2)
            output.loss_terms.numerator.backward()
            assert metrics.call_count == chunks
            assert head.call_count == chunks * (3 if old else 2) + (
                chunks if chunk_size else 0
            )
            outputs.append(output)
    torch.testing.assert_close(outputs[0].loss, outputs[1].loss, rtol=0, atol=0)
    assert outputs[0].ratio_metrics.keys() == outputs[1].ratio_metrics.keys()
    for name in outputs[0].ratio_metrics:
        a, b = outputs[0].ratio_metrics[name], outputs[1].ratio_metrics[name]
        torch.testing.assert_close(a.numerator, b.numerator, rtol=0, atol=0)
        torch.testing.assert_close(a.denominator, b.denominator, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=0, atol=0)
    for actual, expected in zip(wrapper.parameters(), reference.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_master_copy_preserves_bfloat16_rounding(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    master = torch.randn(257, 131, device=device)
    master.flatten()[:5] = torch.tensor(
        [0.0, float("inf"), -float("inf"), 1e-38, -1e-38], device=device
    )
    destination = torch.empty_like(master, dtype=torch.bfloat16)
    destination.copy_(master)
    torch.testing.assert_close(destination, master.to(torch.bfloat16), rtol=0, atol=0)
