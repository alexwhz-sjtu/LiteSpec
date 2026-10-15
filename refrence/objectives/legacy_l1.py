# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: existing_framework
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/losses.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/losses.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Retain as a scale reference; no changes to legacy algorithm paths.


# BEGIN _l1 — upstream lines 32-35
def _l1(logits, teacher_logits, weights, diagnostics=False):
    target = torch.softmax(teacher_logits.detach().float(), dim=-1)
    draft = torch.softmax(logits.float(), dim=-1)
    return _l1_probs(draft, target, weights, diagnostics)

# END _l1

# BEGIN _l1_probs — upstream lines 38-48
def _l1_probs(draft, target, weights, diagnostics=False):
    distance = (draft - target).abs().sum(dim=-1)
    result = ((distance * weights).sum(), distance.detach())
    if diagnostics:
        # Detached diagnostics must not save autograd state: logging-only
        # maxima are skipped when the objective is checkpoint-recomputed.
        with torch.no_grad():
            target_top1 = target.max(dim=-1).values
            draft_top1 = draft.max(dim=-1).values
        return (*result, target_top1, draft_top1)
    return result

# END _l1_probs
