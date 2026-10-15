# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: framework_integration_baseline
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/outputs.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/outputs.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Existing single differentiable numerator and additive detached denominator.


# BEGIN RatioTerms — upstream lines 8-14
@dataclass(frozen=True)
class RatioTerms:
    numerator: torch.Tensor
    denominator: torch.Tensor

    def detached(self) -> "RatioTerms":
        return RatioTerms(self.numerator.detach(), self.denominator.detach())

# END RatioTerms

# BEGIN TrainingOutput — upstream lines 17-22
@dataclass
class TrainingOutput:
    loss: torch.Tensor
    loss_terms: RatioTerms
    ratio_metrics: dict[str, RatioTerms] = field(default_factory=dict)
    sum_metrics: dict[str, torch.Tensor] = field(default_factory=dict)

# END TrainingOutput
