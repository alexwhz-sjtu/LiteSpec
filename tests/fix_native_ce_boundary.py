from pathlib import Path
p=Path('/diancpfs/user/wanghanzhen/LiteSpec/litespec/core/losses.py')
s=p.read_text()
s=s.replace('Low-precision CE and L1 softmax stay native to preserve rounding/subgradients;\nthe eligible reductions use Inductor-generated Triton and first-order autograd.', 'CE and L1 softmax stay native to preserve rounding/subgradients, including\nFP32 logits cast from a BF16 head. Eligible distribution reductions use\nInductor-generated Triton and first-order autograd.')
old='''    if logits.dtype != torch.float32:
        # PyTorch 2.9 Inductor promotes low-precision CE intermediates even
        # with emulate_precision_casts. Keep native CE's rounding contract.
        return _ce(logits, labels, weights)
    return _run(_ce, logits, labels, weights)'''
new='''    # FP32 logits may have been cast from a BF16 projection. Compiling CE's
    # log-softmax can shift its FP32 gradient across BF16 rounding midpoints,
    # changing model gradients and optimizer updates despite a close loss.
    _should_compile(logits)  # Validate the configured backend on this path.
    return _ce(logits, labels, weights)'''
assert s.count(old)==1;s=s.replace(old,new)
old='''    if logits.dtype != torch.float32:
        ce, values = _ce(logits, labels, weights)
        l1 = _run(_l1_probs, draft, target, weights, diagnostics)
        return ce, l1[0], values, *l1[1:]
    return _run(_ce_l1_probs, logits, labels, draft, target, weights, diagnostics)'''
new='''    # The dtype at this API cannot tell whether logits came from a BF16 head.
    # Keep native CE forward/backward for every dtype, and compile only L1's
    # reduction after the native probability boundary.
    ce, values = _ce(logits, labels, weights)
    l1 = _run(_l1_probs, draft, target, weights, diagnostics)
    return ce, l1[0], values, *l1[1:]'''
assert s.count(old)==1;s=s.replace(old,new)
p.write_text(s)
