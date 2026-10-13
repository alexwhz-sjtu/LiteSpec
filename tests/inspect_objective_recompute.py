"""Count actual objective forward replay during backward on tiny CPU models.

Run: PYTHONPATH=. .venv/bin/python tests/inspect_objective_recompute.py
No target model, attention execution, or training dataset is required.
"""

from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from test_dflash_family import tiny_config
from litespec.core.chunking import checkpointed_chunk_reduce
from litespec.core.dflash import OnlineDFlashModel
from litespec.core.dflash2 import OnlineDFlash2Model
from litespec.core.dspark import OnlineDSparkModel
from litespec.modeling.draft.dflash import DFlashDraftModel
from litespec.modeling.draft.dflash2 import DFlash2DraftModel
from litespec.modeling.draft.dspark import DSparkDraftModel


def inspect(algorithm, chunk_size):
    torch.manual_seed(42)
    if algorithm == "dflash":
        draft = DFlashDraftModel(tiny_config())
        wrapper_cls = OnlineDFlashModel
    elif algorithm == "dflash2":
        draft = DFlash2DraftModel(tiny_config(
            "DFlash2DraftModel", conv_kernel_size=2, conv_group_size=4,
            selector_rank=4, selector_top_k=3,
        ))
        wrapper_cls = OnlineDFlash2Model
    else:
        draft = DSparkDraftModel(tiny_config(
            "DSparkDraftModel", projector_type="dspark", markov_rank=4,
            markov_head_type="rnn", enable_confidence_head=True,
            confidence_head_with_markov=True,
        ))
        wrapper_cls = OnlineDSparkModel
    head = nn.Linear(16, 32, bias=False).requires_grad_(False)
    wrapper = wrapper_cls(
        draft, head, nn.Embedding(32, 16).requires_grad_(False), 31,
        block_size=4, attention_backend="sdpa", loss_decay_gamma=2.0,
    )
    hidden = torch.randn(1, 2, 4, 16, requires_grad=True)
    labels = torch.randint(0, 32, (1, 2, 4))
    previous = torch.randint(0, 32, labels.shape)
    weights = torch.ones_like(labels, dtype=torch.float32)
    weights[:, :, 0] = 0
    if algorithm == "dspark":
        function = wrapper._dspark_objective_chunk_terms
        inputs = (hidden, previous, labels, weights, weights.bool(),
                  torch.randn_like(hidden))
    else:
        function = wrapper._dflash_objective_chunk_terms
        inputs = (hidden, labels, weights, previous)
    phase = "forward"
    counts = {"forward": Counter(), "backward_replay": Counter()}

    def counted_function(*args):
        counts[phase]["objective_enter"] += 1
        result = function(*args)
        counts[phase]["objective_complete"] += 1
        return result

    def record_head(module, args):
        kind = "draft_projection" if torch.is_grad_enabled() else "teacher_projection"
        counts[phase][kind] += 1

    def counted_op(name, original):
        def call(*args, **kwargs):
            kind = "grad" if torch.is_grad_enabled() else "no_grad"
            counts[phase][f"{name}/{kind}"] += 1
            return original(*args, **kwargs)
        return call

    with ExitStack() as stack:
        handle = head.register_forward_pre_hook(record_head)
        stack.callback(handle.remove)
        for owner, name in (
            (F, "cross_entropy"),
            (torch, "softmax"),
            (F, "binary_cross_entropy_with_logits"),
        ):
            stack.enter_context(
                patch.object(owner, name, counted_op(name, getattr(owner, name)))
            )
        terms = checkpointed_chunk_reduce(
            counted_function, *inputs, chunk_size=chunk_size, dim=1,
        )
        loss = (
            0.1 * terms[0] + 0.9 * terms[1] + terms[2]
            if algorithm == "dspark" else terms[0] + terms[5]
        )
        phase = "backward_replay"
        loss.backward()
    print(f"{algorithm} chunk_size={chunk_size}: "
          f"{ {key: dict(value) for key, value in counts.items()} }")
    return loss.detach(), hidden.grad.detach()


if __name__ == "__main__":
    torch.set_num_threads(1)
    for algorithm in ("dflash", "dflash2", "dspark"):
        checkpointed = inspect(algorithm, 1)
        uncheckpointed = inspect(algorithm, 0)
        for actual, expected in zip(checkpointed, uncheckpointed):
            torch.testing.assert_close(actual, expected)
        print(f"{algorithm}: loss and hidden gradient match")
