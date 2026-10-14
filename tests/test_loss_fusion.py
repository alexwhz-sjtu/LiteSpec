"""Numerical/autograd contracts for shared losses, including compiled CUDA."""

import pytest
import torch

from litespec.core import losses


@pytest.mark.parametrize("kind", ["ce", "ce_l1", "kl", "bce"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 65), (17, 513), (128, 32768)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_loss_values_and_gradients(monkeypatch, kind, dtype, shape):
    torch.manual_seed(17)
    logits = (torch.randn(shape, device="cuda") * 3).to(dtype).requires_grad_()
    teacher = torch.randn_like(logits, requires_grad=True)
    labels = torch.randint(shape[-1], (shape[0],), device="cuda")
    weights = torch.rand(shape[0], device="cuda")
    weights[::3] = 0
    if shape[0] == 1:
        weights.fill_(1)
    if kind == "bce":
        teacher = teacher.sigmoid().detach().requires_grad_()
        weights = weights[:, None].expand_as(logits)

    def evaluate():
        if kind == "ce":
            return losses.weighted_cross_entropy(logits, labels, weights)
        if kind == "ce_l1":
            return losses.weighted_ce_l1(
                logits, labels, teacher, weights, diagnostics=True
            )
        if kind == "kl":
            return (losses.weighted_kl(logits, teacher, weights),)
        return (losses.weighted_binary_cross_entropy(logits, teacher, weights),)

    monkeypatch.setenv("LITESPEC_LOSS_BACKEND", "torch")
    expected = evaluate()
    objective = expected[0] + 0.7 * expected[1] if kind == "ce_l1" else expected[0]
    expected_grad, expected_teacher_grad = torch.autograd.grad(
        objective * 0.37, (logits, teacher), allow_unused=True
    )
    monkeypatch.setenv("LITESPEC_LOSS_BACKEND", "inductor")
    actual = evaluate()
    objective = actual[0] + 0.7 * actual[1] if kind == "ce_l1" else actual[0]
    actual_grad, actual_teacher_grad = torch.autograd.grad(
        objective * 0.37, (logits, teacher), allow_unused=True
    )
    assert expected_teacher_grad is None and actual_teacher_grad is None
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    # BF16 output gradients have one final representational rounding. Test both
    # FP32 kernels tightly and BF16 gradients at the dtype's resolution.
    torch.testing.assert_close(
        actual_grad,
        expected_grad,
        rtol=3e-5 if dtype == torch.float32 else 0.016,
        atol=2e-6 if dtype == torch.float32 else 2e-5,
    )


def test_loss_weights_and_teacher_detachment(monkeypatch):
    monkeypatch.setenv("LITESPEC_LOSS_BACKEND", "torch")
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 2.0, 0.0]], requires_grad=True)
    teacher = x.detach().clone().requires_grad_()
    ce, l1, _, distance = losses.weighted_ce_l1(
        x, torch.tensor([2, 0]), teacher, torch.tensor([1.0, 0.0])
    )
    assert l1.item() == 0
    assert torch.count_nonzero(distance) == 0
    (ce + l1).backward()
    assert torch.count_nonzero(x.grad[1]) == 0
    assert teacher.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_losses_inside_chunk_checkpoint(monkeypatch):
    from litespec.core.chunking import checkpointed_chunk_reduce

    torch.manual_seed(31)
    hidden = torch.randn(7, 16, device="cuda", requires_grad=True)
    head = torch.randn(513, 16, device="cuda")
    teacher = torch.randn(7, 513, device="cuda")
    labels = torch.randint(513, (7,), device="cuda")
    weights = torch.rand(7, device="cuda")

    def objective(x, target, ids, w):
        logits = x @ head.T
        ce, l1, *_ = losses.weighted_ce_l1(logits, ids, target, w)
        kl = losses.weighted_kl(logits, target, w)
        return ce, l1, kl

    gradients = []
    for backend, chunk_size in (("torch", 0), ("inductor", 3)):
        monkeypatch.setenv("LITESPEC_LOSS_BACKEND", backend)
        terms = checkpointed_chunk_reduce(
            objective,
            hidden,
            teacher,
            labels,
            weights,
            chunk_size=chunk_size,
        )
        gradient = torch.autograd.grad(
            0.1 * terms[0] + terms[1] + 0.7 * terms[2], hidden
        )[0]
        gradients.append(gradient)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=5e-5, atol=1e-5)


@pytest.mark.parametrize(
    "case", ["equal", "near_equal", "zero_weights", "extreme", "autocast"]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_loss_boundary_gradients(monkeypatch, case):
    torch.manual_seed(47)
    # Exercise strided inputs and a non-power-of-two vocabulary as well.
    logits = torch.randn(513, 7, device="cuda").T.requires_grad_()
    teacher = (
        logits.detach().clone()
        if case in ("equal", "near_equal")
        else torch.randn_like(logits)
    )
    if case == "near_equal":
        teacher[0, 0] += 1e-5
    if case == "extreme":
        logits = (logits.detach() * 100).requires_grad_()
        teacher = teacher * 100
    teacher.requires_grad_()
    labels = torch.randint(513, (7,), device="cuda")
    weights = (
        torch.zeros(7, device="cuda")
        if case == "zero_weights"
        else torch.rand(7, device="cuda")
    )
    outputs = []
    for backend in ("torch", "inductor"):
        monkeypatch.setenv("LITESPEC_LOSS_BACKEND", backend)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=case == "autocast"):
            ce, l1, *_ = losses.weighted_ce_l1(logits, labels, teacher, weights)
            kl = losses.weighted_kl(logits, teacher, weights)
            # Confidence heads have FP32 targets even with BF16 logits.
            bce = losses.weighted_binary_cross_entropy(
                logits.to(torch.bfloat16), teacher.float().sigmoid(), weights[:, None]
            )
        gradients = torch.autograd.grad(
            ce + l1 + kl + bce, (logits, teacher), allow_unused=True
        )
        # AOT may donate backward buffers; inspect the standalone L1 gradient
        # on a fresh graph, just as each training forward has its own graph.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=case == "autocast"):
            _, standalone_l1, *_ = losses.weighted_ce_l1(
                logits, labels, teacher, weights
            )
        l1_gradient = torch.autograd.grad(standalone_l1, logits)[0]
        assert gradients[1] is None
        if case in ("equal", "zero_weights"):
            assert l1.item() == 0
        if case == "zero_weights":
            assert torch.count_nonzero(gradients[0]) == 0
        outputs.append((torch.stack((ce, l1, kl, bce)), gradients[0], l1_gradient))
    torch.testing.assert_close(outputs[1][0], outputs[0][0], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(outputs[1][1], outputs[0][1], rtol=3e-5, atol=2e-5)
    torch.testing.assert_close(outputs[1][2], outputs[0][2], rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("kind", ["ce", "ce_l1"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_loss_preserves_bf16_projection_gradient(monkeypatch, kind):
    # DLite's BF16 head is cast to FP32 before CE. FP32-close gradients can
    # straddle a BF16 rounding midpoint and alter the first optimizer update.
    torch.manual_seed(61)
    logits = (torch.randn(32, 151936, device="cuda") * 0.25).bfloat16()
    # Equal distributions make the L1 gradient zero, isolating CE through
    # both public APIs while still exercising the combined compiled path.
    teacher = logits.detach().clone()
    labels = torch.randint(logits.size(-1), (logits.size(0),), device="cuda")
    weights = torch.rand(logits.size(0), device="cuda")
    gradients = []
    values = []
    for backend in ("torch", "inductor"):
        monkeypatch.setenv("LITESPEC_LOSS_BACKEND", backend)
        head_output = logits.detach().clone().requires_grad_(True)
        if kind == "ce":
            ce, _ = losses.weighted_cross_entropy(head_output.float(), labels, weights)
            objective = 0.1 * ce
        else:
            ce, l1, *_ = losses.weighted_ce_l1(
                head_output.float(), labels, teacher, weights
            )
            objective = 0.1 * ce + l1
        values.append(objective.detach())
        gradients.append(torch.autograd.grad(objective, head_output)[0])
    torch.testing.assert_close(values[1], values[0], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
