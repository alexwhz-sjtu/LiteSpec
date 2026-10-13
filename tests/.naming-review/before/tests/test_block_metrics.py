import pytest
import torch

from litespec.core.outputs import RatioTerms
from litespec.training.logging import TrainingMetricAccumulator
from litespec.core.outputs import TrainingOutput
from litespec.core.block_metrics import block_accepted_length_terms
from litespec.training.logging import build_training_log_payload
from litespec.core.block_metrics import distribution_acceptance_probability
from litespec.core.block_metrics import expected_accept_len_terms
from litespec.training.logging import reduce_training_metrics
from litespec.core.block_metrics import sparse_distribution_acceptance_probability


def test_training_log_payload_reserves_train_for_comparable_metrics():
    metrics = {
        "objective/loss": torch.tensor(1.5),
        "hard_label/token_accuracy": torch.tensor(0.4),
        "hard_label/block_accepted_length": torch.tensor(3.0),
        "expected_accept_len": torch.tensor(2.5),
    }

    payload = build_training_log_payload(
        metrics,
        3e-4,
        detail_metrics={"micro_step": 7, "grad_norm": 0.8},
    )

    assert {name for name in payload if name.startswith("train/")} == {
        "train/loss",
        "train/learning_rate",
        "train/walk_accepted_length",
        "train/expected_accept_len",
    }
    assert payload["train/walk_accepted_length"] == 3.0
    assert payload["detail/hard_label/token_accuracy"] == pytest.approx(0.4)
    assert payload["detail/micro_step"] == 7


def test_block_accepted_length_stops_at_first_failure():
    predicted = torch.tensor([[[1, 9, 3], [4, 5, 6]]])
    target = torch.tensor([[[1, 2, 3], [4, 5, 0]]])
    supervised = torch.tensor([[[True, True, True], [True, True, False]]])
    terms = block_accepted_length_terms(predicted, target, supervised)
    # First block: anchor + one hit. Second: anchor + two hits.
    assert terms.numerator.item() == 5
    assert terms.denominator.item() == 2


def test_expected_accept_len_uses_prefix_products():
    acceptance = torch.tensor([[[0.5, 0.25, 0.2], [0.8, 0.7, 0.6]]])
    supervised = torch.tensor([[[True, True, False], [True, True, True]]])
    terms = expected_accept_len_terms(acceptance, supervised)
    expected = (1 + 0.5 + 0.5 * 0.25) + (1 + 0.8 + 0.8 * 0.7 + 0.8 * 0.7 * 0.6)
    torch.testing.assert_close(terms.numerator, torch.tensor(expected))
    assert terms.denominator.item() == 2


def test_dense_acceptance_probability_is_distribution_overlap():
    target = torch.log(torch.tensor([[[0.7, 0.2, 0.1]]]))
    draft = torch.log(torch.tensor([[[0.4, 0.4, 0.2]]]))
    actual = distribution_acceptance_probability(target, draft)
    torch.testing.assert_close(actual, torch.tensor([[0.7]]))


def test_expected_acceptance_identical_and_disjoint_distributions():
    target = torch.tensor([[[[0.0, -torch.inf]] * 3]])
    for draft, expected in ((target, 4.0), (target.flip(-1), 1.0)):
        acceptance = distribution_acceptance_probability(target, draft)
        terms = expected_accept_len_terms(
            acceptance, torch.ones_like(acceptance, dtype=torch.bool)
        )
        assert terms.numerator / terms.denominator == expected


def test_acceptance_supervision_gap_ends_the_prefix():
    labels = torch.ones(1, 2, 3, dtype=torch.long)
    supervision = torch.tensor([[[True, False, True], [False, True, True]]])
    terms = block_accepted_length_terms(labels, labels, supervision)
    assert terms.numerator == 3 and terms.denominator == 2


def test_sparse_acceptance_probability_includes_target_mass_outside_candidates():
    target = torch.log(torch.tensor([[[0.5, 0.3, 0.2]]]))
    candidate_ids = torch.tensor([[[0, 2]]])
    candidate_logits = torch.log(torch.tensor([[[0.6, 0.4]]]))
    actual = sparse_distribution_acceptance_probability(
        target, candidate_logits, candidate_ids
    )
    # min(0.5, 0.6) + min(0.2, 0.4); target token 1 is absent from q.
    torch.testing.assert_close(actual, torch.tensor([[0.7]]))


def test_reduce_training_metrics_only_keeps_top16_recall_per_position():
    output = TrainingOutput(
        loss=torch.tensor(2.0),
        loss_terms=RatioTerms(torch.tensor(6.0), torch.tensor(3.0)),
        ratio_metrics={
            "hard_label/token_accuracy": RatioTerms(
                torch.tensor(3.0), torch.tensor(4.0)
            ),
            "position_1/dflash/hard_label/unary_top16_recall": RatioTerms(
                torch.tensor(2.0), torch.tensor(4.0)
            ),
            "position_1/dflash/hard_label/unary_top1_accuracy": RatioTerms(
                torch.tensor(1.0), torch.tensor(4.0)
            ),
            "position_2/objective/ce": RatioTerms(torch.tensor(8.0), torch.tensor(4.0)),
        },
        sum_metrics={
            "dflash/hard_label/supervised_count": torch.tensor(9.0),
            "position_1/dflash/hard_label/supervised_count": torch.tensor(5.0),
        },
    )

    reduced = reduce_training_metrics(output)

    assert set(reduced) == {
        "objective/loss",
        "hard_label/token_accuracy",
        "position_1/dflash/hard_label/unary_top16_recall",
        "dflash/hard_label/supervised_count",
    }
    torch.testing.assert_close(
        reduced["position_1/dflash/hard_label/unary_top16_recall"],
        torch.tensor(0.5),
    )


def test_training_metric_accumulator_uses_full_optimizer_window():
    accumulator = TrainingMetricAccumulator()
    for numerator, denominator, correct in ((6.0, 3.0, 1.0), (8.0, 4.0, 3.0)):
        accumulator.update(
            TrainingOutput(
                loss=torch.tensor(0.0),
                loss_terms=RatioTerms(
                    torch.tensor(numerator), torch.tensor(denominator)
                ),
                ratio_metrics={
                    "hard_label/token_accuracy": RatioTerms(
                        torch.tensor(correct), torch.tensor(4.0)
                    )
                },
            )
        )

    reduced = accumulator.reduce()

    torch.testing.assert_close(reduced["objective/loss"], torch.tensor(2.0))
    torch.testing.assert_close(reduced["hard_label/token_accuracy"], torch.tensor(0.5))
    accumulator.reset()
    assert not accumulator.ratio_metrics
