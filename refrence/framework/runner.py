# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/training/runner.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/training/runner.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Runner backpropagates numerator/accumulation_steps and normalizes gradients once per optimizer window.


# BEGIN OptimizerWindow — upstream lines 28-69
class OptimizerWindow:
    """Sum loss terms, normalize FSDP gradients once, and aggregate telemetry."""

    def __init__(self, optimizer, accumulation_steps, process_group=None):
        from .logging import TrainingMetricAccumulator

        self.optimizer = optimizer
        self.accumulation_steps = int(accumulation_steps)
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        self.process_group = process_group
        self.metrics = TrainingMetricAccumulator()
        self.micro_steps = 0
        self.denominator = None

    def backward(self, output):
        self.metrics.update(output)
        self.backward_terms(output.loss_terms.numerator, output.loss_terms.denominator)

    def backward_terms(self, numerator, denominator):
        (numerator / self.accumulation_steps).backward()
        value = denominator.detach()
        self.denominator = (
            value.clone() if self.denominator is None else self.denominator + value
        )
        self.micro_steps += 1

    def step(self):
        from .optimizer import normalize_accumulated_gradients

        if not self.micro_steps:
            raise RuntimeError("Cannot step an empty accumulation window")
        normalize_accumulated_gradients(
            self.optimizer,
            self.denominator,
            self.accumulation_steps,
            self.process_group,
        )
        norm = self.optimizer.step()
        self.micro_steps = 0
        self.denominator = None
        return norm

# END OptimizerWindow
