from pathlib import Path
import shutil
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
p=root/'litespec/core/losses.py';backup=root/'tests/metrics_feature_20261014/before/litespec/core/losses.py'
shutil.copy2(p,backup)
s=p.read_text()
old='''    if diagnostics:
        return (
            *result,
            target.max(dim=-1).values.detach(),
            draft.max(dim=-1).values.detach(),
        )'''
new='''    if diagnostics:
        # Detached diagnostics must not save autograd state: logging-only
        # maxima are skipped when the objective is checkpoint-recomputed.
        with torch.no_grad():
            target_top1 = target.max(dim=-1).values
            draft_top1 = draft.max(dim=-1).values
        return (*result, target_top1, draft_top1)'''
assert s.count(old)==1;p.write_text(s.replace(old,new))
p=root/'tests/test_training_runner.py'
s=p.read_text();start=s.index('def test_real_dlite_recipe_objectives_on_cpu(');end=s.index('\n\n@pytest.mark.parametrize("algorithm"',start)
block=s[start:end]
block=block.replace('def test_real_dlite_recipe_objectives_on_cpu(monkeypatch, phase_name, disaggregate):',
'''@pytest.mark.parametrize("collect_detailed_metrics", [False, True])
def test_real_dlite_recipe_objectives_on_cpu(
    monkeypatch, phase_name, disaggregate, collect_detailed_metrics
):''')
block=block.replace('output = base.compute_step(session, batch, microbatch_start=0, microbatch_end=1)',
'''output = base.compute_step(
            session, batch, microbatch_start=0, microbatch_end=1,
            collect_detailed_metrics=collect_detailed_metrics,
        )''')
block=block.replace('''                loss_mask=torch.ones_like(ids),
            ),
        )''','''                loss_mask=torch.ones_like(ids),
            ),
            collect_detailed_metrics=collect_detailed_metrics,
        )''')
block=block.replace('''    else:
        assert "expected_accept_len" in output.ratio_metrics''','''    elif collect_detailed_metrics:
        assert "expected_accept_len" in output.ratio_metrics
    if not collect_detailed_metrics:
        assert not output.ratio_metrics''')
assert 'collect_detailed_metrics=collect_detailed_metrics' in block
p.write_text(s[:start]+block+s[end:])
