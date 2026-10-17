import pytest
from litespec.training.recipes.composable import ADAPTER, parse_args
from litespec.training.optimizer import optimizer_step_count

def args(extra=()):
    return parse_args(argv=[
        "--target-model-path", "unused", "--train-data-path", "unused",
        "--output-dir", "unused", *extra,
    ])

def test_full_schedule_unchanged_by_default():
    phase = ADAPTER.phases(args())[0]
    assert phase.batch_count(1000) == 6000
    assert optimizer_step_count([phase], 1000, 1) == 6000

def test_limit_counts_optimizer_updates_and_preserves_lr_schedule():
    phase = ADAPTER.phases(args(["--max-steps", "500", "--accumulation-steps", "3"]))[0]
    assert phase.batch_count(1000) == 1500
    assert optimizer_step_count([phase], 1000, 3) == 2000

def test_limit_does_not_extend_epochs():
    phase = ADAPTER.phases(args(["--max-steps", "500"]))[0]
    assert phase.batch_count(10) == 60

@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_limit_is_rejected(value):
    with pytest.raises(SystemExit):
        args(["--max-steps", value])
