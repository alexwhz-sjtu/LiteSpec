# REFERENCE EXCERPT ONLY; not a runnable module and never imported by composable.
# Classification: existing_framework
# Repository: alexwhz-sjtu/LiteSpec
# Revision: 7fa8b08ddaa29d619dc193d073185e197f77b230
# Source: litespec/core/losses.py
# Archived full source: /data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp/tests/composable_reference_sources/alexwhz-sjtu__LiteSpec/7fa8b08ddaa29d619dc193d073185e197f77b230/litespec/core/losses.py
# License / upstream notices: refrence/licenses/litespec-LICENSE; retained source headers below.
# Original variable names and bodies retained. Integration naming lives in naming-map.json.
# Target-to-draft KL with frozen teacher and unchanged softmax/precision boundaries.


# BEGIN _kl — upstream lines 57-62
def _kl(logits, teacher_logits, weights):
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    values = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (values * weights).sum()

# END _kl

# BEGIN _kl_student — upstream lines 65-68
def _kl_student(logits, teacher_log_probs, teacher_probs, weights):
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    values = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (values * weights).sum()

# END _kl_student

# BEGIN weighted_kl — upstream lines 141-150
def weighted_kl(logits, teacher_logits, weights):
    """Additive teacher-to-student KL; teacher is always detached."""
    if _should_compile(logits):
        # Materialize the frozen teacher once so AOT does not replay its
        # normalization during backward.
        with torch.no_grad():
            teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
            teacher_probs = teacher_log_probs.exp()
        return _run(_kl_student, logits, teacher_log_probs, teacher_probs, weights)
    return _run(_kl, logits, teacher_logits, weights)

# END weighted_kl
