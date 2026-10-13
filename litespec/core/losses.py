"""Shared weighted losses with autograd-generated Triton forward/backward.

The eager functions are the numerical reference, including input-dtype CE
semantics and FP32 distribution losses. CPU and small workloads stay eager.
Set LITESPEC_LOSS_BACKEND=torch to compare an entire run against the reference,
or inductor to explicitly exercise compilation in numerical/throughput tests.
"""

from functools import lru_cache
import os

import torch
import torch.nn.functional as F


def _ce(logits, labels, weights):
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).reshape_as(labels)
    return (values * weights).sum(), values.detach()


def _ce_l1(logits, labels, teacher_logits, weights):
    ce, values = _ce(logits, labels, weights)
    target = torch.softmax(teacher_logits.detach().float(), dim=-1)
    draft = torch.softmax(logits.float(), dim=-1)
    distance = (draft - target).abs().sum(dim=-1)
    return ce, (distance * weights).sum(), values, distance.detach()


def _kl(logits, teacher_logits, weights):
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(logits.float(), dim=-1)
    values = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (values * weights).sum()


def _bce(logits, targets, weights):
    values = F.binary_cross_entropy_with_logits(
        logits.float(), targets.detach(), reduction="none"
    )
    return (values * weights).sum()


@lru_cache(None)
def _compiled(function):
    # AOTAutograd derives backward from the same reference expression, rather
    # than maintaining a separate handwritten gradient/normalization policy.
    return torch.compile(function, fullgraph=True, dynamic=True, backend="inductor")


def _run(function, *args):
    backend = os.environ.get("LITESPEC_LOSS_BACKEND", "auto")
    if backend not in ("auto", "torch", "inductor"):
        raise ValueError(f"Unknown LITESPEC_LOSS_BACKEND: {backend}")
    logits = args[0]
    enabled = logits.is_cuda and backend != "torch"
    # Amortize compilation/dispatch on vocab reductions. Tiny selector CE and
    # confidence heads remain eager unless explicitly requested for comparison.
    if enabled and (backend == "inductor" or logits.numel() >= 65536):
        return _compiled(function)(*args)
    return function(*args)


def weighted_cross_entropy(logits, labels, weights):
    """Return an additive numerator and detached, unweighted per-token CE."""
    return _run(_ce, logits, labels, weights)


def weighted_ce_l1(logits, labels, teacher_logits, weights):
    """Return CE/L1 numerators, detached per-token CE and L1 distance."""
    return _run(_ce_l1, logits, labels, teacher_logits, weights)


def weighted_kl(logits, teacher_logits, weights):
    """Additive teacher-to-student KL; teacher is always detached."""
    return _run(_kl, logits, teacher_logits, weights)


def weighted_binary_cross_entropy(logits, targets, weights):
    return _run(_bce, logits, targets, weights)
