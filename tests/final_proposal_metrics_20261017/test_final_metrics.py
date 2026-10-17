import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


from test_implementation import config, fixture
from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.metrics import acceptance_metric_terms
from litespec.core.composable.metrics import final_token_accuracy_terms
from litespec.modeling.draft.composable.contracts import DraftOutput
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.draft.composable import ModelInputs, SharedOps
from litespec.modeling.draft.dflash2 import CandidateSelector
from litespec.training.logging import build_training_log_payload


def test_selector_walk_and_sparse_overlap_use_corrected_distribution():
    selector = CandidateSelector(hidden_size=1, vocab_size=3, state_rank=1,
                                 top_k=2, initializer_range=0.02)
    with torch.no_grad():
        selector.hidden_projection.weight.fill_(1)
        selector.predecessor_codebook.copy_(torch.tensor([[-1.], [-1.], [1.]]))
        selector.successor_codebook.copy_(torch.tensor([[0.], [2.], [-2.]]))
    labels = torch.tensor([[[2, 1, 2], [1, 2, 1], [2, 1, 2]]])
    previous = torch.cat((torch.zeros(1, 3, 1, dtype=torch.long), labels[..., :-1]), -1)
    hidden = torch.ones(1, 3, 3, 1, requires_grad=True)
    base = torch.tensor([-9., 0.2, 0.]).expand(1, 3, 3, 3)
    unary, candidates = base.topk(2, -1)
    scores = selector.score_candidates(
        candidate_ids=candidates, unary_logits=unary, hidden_states=hidden,
        predecessor_ids=previous,
    )
    p = torch.tensor([0.4, 0.35, 0.25]).expand_as(base)
    prediction = SimpleNamespace(base_logits=base, candidate_logits=scores,
                                 candidate_ids=candidates, draft_hidden_states=hidden)
    mask = torch.tensor([[[1., 1., 1.], [1., 0., 1.], [0., 0., 0.]]])
    walk, expected, denominator = acceptance_metric_terms(
        SimpleNamespace(candidate_selector=selector), prediction, p.log(),
        labels, previous, mask,
    )
    # Greedy path is [2,1,2], unlike unary [1,1,1] and TF row argmax [2,2,1].
    assert walk.item() == 5.0  # first block length 4; second stops immediately (1)
    assert denominator.item() == 2.0
    dense_q = torch.zeros_like(p).scatter(-1, candidates, scores.float().softmax(-1))
    overlap = torch.minimum(p, dense_q).sum(-1)
    manual = (1 + (overlap * (mask > 0.5)).cumprod(-1).sum(-1))
    manual = (manual * (mask > 0.5).any(-1)).sum()
    torch.testing.assert_close(expected, manual)
    assert not any(x.requires_grad for x in (walk, expected, denominator))
    # Target mass outside top-K is retained, rather than renormalizing p on K.
    assert overlap.max().item() <= 0.6 + 1e-6


def test_selector_accuracy_counts_final_reranking_and_uncovered_gold():
    # Selector fixes unary's first row and preserves the second; an uncovered
    # third label must stay in the denominator.
    prediction = DraftOutput(
        draft_hidden_states=torch.zeros(1, 1, 3, 2),
        base_logits=torch.tensor([[[[9., 0., -9.], [0., 9., -9.], [9., 0., -9.]]]]),
        candidate_ids=torch.tensor([[[[0, 1], [1, 0], [0, 1]]]]),
        candidate_logits=torch.tensor([[[[0., 9.], [9., 0.], [9., 0.]]]], requires_grad=True),
    )
    labels = torch.tensor([[[1, 1, 2]]])
    num, den = final_token_accuracy_terms(prediction, labels, torch.ones_like(labels))
    assert (num.item(), den.item()) == (2., 3.)
    assert not num.requires_grad and not den.requires_grad
    prediction.candidate_logits = -prediction.candidate_logits
    num, den = final_token_accuracy_terms(prediction, labels, torch.ones_like(labels))
    assert (num.item(), den.item()) == (0., 3.)


def test_dense_accuracy_uses_final_logits_instead_of_base_and_masks_padding():
    prediction = DraftOutput(
        draft_hidden_states=torch.zeros(1, 1, 3, 2),
        base_logits=torch.tensor([[[[9., 0.], [9., 0.], [9., 0.]]]]),
        draft_logits=torch.tensor([[[[0., 9.], [9., 0.], [9., 0.]]]], requires_grad=True),
    )
    num, den = final_token_accuracy_terms(
        prediction, torch.ones(1, 1, 3, dtype=torch.long), torch.tensor([[[1., 0., 0.]]])
    )
    assert (num.item(), den.item()) == (1., 1.)


def test_missing_final_logits_cannot_silently_report_unary_accuracy():
    prediction = DraftOutput(torch.zeros(1, 1, 1, 2), base_logits=torch.zeros(1, 1, 1, 3))
    with pytest.raises(ValueError, match="Final proposal logits"):
        final_token_accuracy_terms(prediction, torch.zeros(1, 1, 1), torch.ones(1, 1, 1))


@pytest.mark.parametrize("proposal", ["parallel", "selector", "markov", "rnn", "sequential_rnn"])
@pytest.mark.parametrize("objective", ["hard_ce", "tv", "dpace"])
def test_logged_metrics_preserve_loss_gradients_and_match_independent_reference(proposal, objective):
    torch.manual_seed(743)
    cfg = config("dflare", proposal, objective, heterogeneous=True, window=True)
    model, embedding, head, ids, hidden, teacher, mask = fixture(cfg)
    # Activate recursive corrections so base logits and actual path can differ.
    with torch.no_grad():
        if hasattr(model.proposal, "candidate_selector"):
            model.proposal.candidate_selector.successor_codebook.normal_(std=0.4)
        if hasattr(model.proposal, "markov_head"):
            model.proposal.markov_head.markov_w2.weight.normal_(std=0.4)
    other = copy.deepcopy(model)
    layout = prepare_layout(ids, mask, cfg.block_size, 4)
    plain = ComposableTrainingModel(model, head, embedding, 0, num_anchors=4,
                                    objective_chunk_blocks=0)
    logged = ComposableTrainingModel(other, head, embedding, 0, num_anchors=4,
                                     objective_chunk_blocks=1)
    off = plain(ids, hidden, mask, teacher, layout=layout)
    on = logged(ids, hidden, mask, teacher, layout=layout, collect_detailed_metrics=True)
    torch.testing.assert_close(off.loss, on.loss)
    off.loss.backward()
    on.loss.backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5, msg=name)
    assert "expected_accept_len" not in off.ratio_metrics
    # Independently run the public proposal API on the identical fixed blocks.
    with torch.no_grad():
        noise = ids.new_zeros(ids.size(0), 4, cfg.block_size)
        noise[..., 0] = ids.gather(1, layout.anchor_positions).masked_fill(
            ~layout.block_keep_mask, 0)
        inputs = ModelInputs(hidden, embedding(noise.flatten(1)),
                             layout.anchor_positions, layout.block_keep_mask,
                             previous_token_ids=layout.previous_token_ids)
        output = other(inputs, shared_ops=SharedOps(embedding, head))
        inputs.mode = "sample"
        inputs.anchor_token_ids = layout.previous_token_ids[..., 0]
        greedy = other(inputs, shared_ops=SharedOps(embedding, head))
        valid = layout.prediction_mask > 0.5
        block_valid = valid.any(-1)
        hits = (greedy.draft_token_ids == layout.labels) & valid
        walk_num = ((1 + hits.long().cumprod(-1).sum(-1)) * block_valid).sum()
        indices = (layout.label_indices - 1).clamp_min(0)
        last = teacher.gather(1, indices.flatten(1)[..., None].expand(-1, -1, 32))
        p = head(last).reshape(*layout.labels.shape, -1).float().softmax(-1)
        if proposal == "selector":
            q = torch.zeros_like(p).scatter(-1, output.candidate_ids,
                                            output.candidate_logits.float().softmax(-1))
        else:
            q = output.draft_logits.float().softmax(-1)
        overlap = torch.minimum(p, q).sum(-1)
        expected_num = ((1 + (overlap * valid).cumprod(-1).sum(-1)) * block_valid).sum()
        final_ids = q.argmax(-1)
        accuracy_num = ((final_ids == layout.labels) & valid).sum()
    accuracy = on.ratio_metrics["hard_label/token_accuracy"]
    torch.testing.assert_close(accuracy.numerator, accuracy_num.float())
    torch.testing.assert_close(accuracy.denominator, valid.sum().float())
    for name, reference in [
        ("hard_label/block_accepted_length", walk_num),
        ("expected_accept_len", expected_num),
    ]:
        actual = on.ratio_metrics[name]
        torch.testing.assert_close(actual.numerator.float(), reference.float())
        torch.testing.assert_close(actual.denominator.float(), block_valid.sum().float())
        assert not actual.numerator.requires_grad and not actual.denominator.requires_grad
    reduced = {k: v.numerator / v.denominator.clamp_min(1e-12)
               for k, v in on.ratio_metrics.items()}
    payload = build_training_log_payload(reduced, 0.001, detail_prefix="composable")
    assert 1 <= payload["train/walk_accepted_length"] <= cfg.block_size
    assert 1 <= payload["train/expected_accept_len"] <= cfg.block_size


def test_empty_supervision_and_missing_teacher_are_explicit():
    model, embedding, head, ids, hidden, teacher, mask = fixture(config(proposal="selector"))
    wrapper = ComposableTrainingModel(model, head, embedding, 0, num_anchors=3,
                                      objective_chunk_blocks=1)
    output = wrapper(ids, hidden, mask * 0, teacher, collect_detailed_metrics=True)
    for name in ("hard_label/block_accepted_length", "expected_accept_len"):
        assert output.ratio_metrics[name].numerator == 0
        assert output.ratio_metrics[name].denominator == 0
    with pytest.raises(ValueError, match="Acceptance metrics require"):
        wrapper(ids, hidden, mask, collect_detailed_metrics=True)


def test_nonlogging_steps_do_not_compute_acceptance(monkeypatch):
    import litespec.core.composable.training_model as module
    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance metrics should be gated")
    monkeypatch.setattr(module, "acceptance_metric_terms", forbidden)
    model, embedding, head, ids, hidden, teacher, mask = fixture(config())
    wrapper = ComposableTrainingModel(model, head, embedding, 0, num_anchors=2)
    output = wrapper(ids, hidden, mask)
    output.loss.backward()
    assert "expected_accept_len" not in output.ratio_metrics


def test_bf16_block_counts_remain_exact_across_chunks():
    from torch import nn
    from litespec.modeling.draft.composable.contracts import BackboneOutput
    from litespec.modeling.draft.composable.proposal.parallel import ParallelProposal

    class TinyDraft(nn.Module):
        block_size = 2
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=4, block_size=2,
                composable_config={"proposal": {"type": "parallel"},
                                   "objective": {"type": "hard_ce"}},
            )
            self.proposal = ParallelProposal(self.config)

        def forward(self, inputs, shared_ops=None):
            features = BackboneOutput(inputs.draft_embeddings)
            output = self.proposal(features, SimpleNamespace(inputs=inputs), shared_ops)
            output.backbone_hidden_states = features.draft_hidden_states
            return output

    embedding = nn.Embedding(3, 4).bfloat16().requires_grad_(False)
    head = nn.Linear(4, 3, bias=False).bfloat16().requires_grad_(False)
    head.weight.zero_()
    ids = torch.zeros(1, 600, dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.float32)
    hidden = torch.zeros(1, 600, 4, dtype=torch.bfloat16)
    wrapper = ComposableTrainingModel(TinyDraft(), head, embedding, 0,
                                      num_anchors=511, objective_chunk_blocks=37)
    output = wrapper(ids, hidden, mask, hidden, collect_detailed_metrics=True)
    for key in ("hard_label/block_accepted_length", "expected_accept_len"):
        terms = output.ratio_metrics[key]
        assert terms.denominator.dtype == torch.float32
        assert terms.denominator.item() == 511
        torch.testing.assert_close(terms.numerator / terms.denominator, torch.tensor(2.0))
    accuracy = output.ratio_metrics["hard_label/token_accuracy"]
    assert accuracy.numerator.dtype == accuracy.denominator.dtype == torch.float32
    assert accuracy.numerator.item() == accuracy.denominator.item() == 511
