"""Old adapters do not enter composable window hooks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch
from litespec.core.outputs import RatioTerms, TrainingOutput
from litespec.training.algorithms import Phase, TrainingSession
from litespec.training.runner import run_draft


def test_legacy_runner_unchanged():
    root = Path(__file__).resolve().parent
    path = root / "legacy_runner_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "litespec.training._composable_baseline_runner", path
    )
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    def execute(runner):
        torch.manual_seed(19)
        model = torch.nn.Linear(1, 1, bias=False)
        weights = [1.0, 3.0, 2.0]

        class Optimizer:
            def __init__(self):
                self.inner = torch.optim.SGD(model.parameters(), lr=0.1)

            def scale_model_gradients(self, scale):
                for p in model.parameters():
                    p.grad.mul_(scale)

            def step(self):
                self.inner.step()
                self.inner.zero_grad()
                return 0.0

        class Adapter:
            algorithm = "legacy_fixture"
            modes = ()
            phase_barrier = False

            def enter_phase(self, session, phase):
                pass

            def compute_step(self, session, data, **kwargs):
                loss = (model(torch.tensor([[data + 1.0]])) - 1).square().sum()
                numerator = loss * weights[data]
                denominator = numerator.new_tensor(weights[data])
                return TrainingOutput(
                    numerator / denominator, RatioTerms(numerator, denominator)
                )

        args = SimpleNamespace(
            accumulation_steps=2, log_interval=1, save_interval=2, final_save=True
        )
        session = TrainingSession(
            args, Adapter(), [Phase("legacy_fixture", 1, 1, max_batches=3)]
        )
        session.optimizer = Optimizer()
        session.total_steps = 2
        session.steps_per_epoch = 3
        session.tracker = SimpleNamespace(log=lambda *a, **kw: None)
        events = []
        runner(
            session,
            batch_source=lambda s, p: iter(range(3)),
            save=lambda s, n: events.append(
                (n, s.state.global_step, s.state.optimizer_step)
            ),
            log=lambda *a: None,
        )
        return model.weight.detach().clone(), vars(session.state), events

    current = execute(run_draft)
    original = execute(baseline.run_draft)
    torch.testing.assert_close(current[0], original[0], rtol=0, atol=0)
    assert current[1:] == original[1:]
