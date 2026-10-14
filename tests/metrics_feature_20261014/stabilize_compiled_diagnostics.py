from pathlib import Path
p=Path('/diancpfs/user/wanghanzhen/LiteSpec/litespec/core/losses.py')
s=p.read_text()
old='''    l1 = _run(_l1_probs, draft, target, weights, diagnostics)
    return ce, l1[0], values, *l1[1:]'''
new='''    # Keep the training kernel identical on log/non-log steps and checkpoint
    # replay. Logging-only maxima must not specialize its reduction/backward.
    l1 = _run(_l1_probs, draft, target, weights, False)
    result = (ce, l1[0], values, *l1[1:])
    if diagnostics:
        with torch.no_grad():
            return (*result, target.max(dim=-1).values, draft.max(dim=-1).values)
    return result'''
assert s.count(old)==1;p.write_text(s.replace(old,new))
