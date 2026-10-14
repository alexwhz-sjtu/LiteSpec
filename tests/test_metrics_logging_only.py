"""Logging diagnostics must not change objectives, gradients or checkpoint replay."""
import copy
from unittest import mock

import pytest
import torch
from torch import nn

from refactor_baseline import capture
from litespec.core import dlite, dflash, dflash2, dspark
from litespec.training.runner import run_draft
from test_training_runner import make_session, source


def gradients(module):
    return {name: None if p.grad is None else p.grad.clone()
            for name, p in module.named_parameters()}


def assert_gradients_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        assert (actual[name] is None) == (expected[name] is None), name
        if actual[name] is not None:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize('transition', [False, True])
@pytest.mark.parametrize('l1_weight', [0., 1.])
def test_dlite_skips_diagnostics_without_changing_loss_or_gradients(transition, l1_weight):
    torch.manual_seed(33)
    template = dlite.OnlineDLiteModel(
        capture(models_only=True)['dlite'], nn.Linear(16, 32, bias=False),
        nn.Embedding(32, 16), mask_token_id=31, block_size=4,
        l1_loss_weight=l1_weight)
    ids = torch.arange(12).view(1, -1)
    context = {1: torch.randn(1, 12, 16)}
    teacher_logits = torch.randn(1, 2, 3, 32)
    prediction = torch.randn(1, 2, 3, 16)
    outputs=[]
    for collect in [True, False]:
        online=copy.deepcopy(template)
        batch=online.prepare_batch(ids,context,torch.ones_like(ids),
            anchor_positions=torch.tensor([[0,4]]),block_keep_mask=torch.ones(1,2,dtype=torch.bool))
        hidden=prediction.clone().requires_grad_(True)
        with mock.patch.object(online,'forward_backbone',return_value=hidden), \
             mock.patch.object(dlite,'block_accepted_length_terms',wraps=dlite.block_accepted_length_terms) as accepted, \
             mock.patch.object(dlite,'expected_accept_len_terms',wraps=dlite.expected_accept_len_terms) as expected:
            result=online(prepared_batch=batch,seq_len=12,target_prefill_logits=teacher_logits,
                target_logits_are_gathered=True,return_transition_outputs=transition,
                collect_detailed_metrics=collect)
            output=result[2] if transition else result
            output.loss_terms.numerator.backward()
            assert accepted.call_count==int(collect)
            assert expected.call_count==int(collect)
        assert bool(output.ratio_metrics)==collect
        outputs.append((output.loss_terms.numerator.detach(),output.loss_terms.denominator,
                        gradients(online),hidden.grad.clone()))
    torch.testing.assert_close(outputs[1][0],outputs[0][0],rtol=0,atol=0)
    torch.testing.assert_close(outputs[1][1],outputs[0][1],rtol=0,atol=0)
    assert_gradients_equal(outputs[1][2],outputs[0][2])
    torch.testing.assert_close(outputs[1][3],outputs[0][3],rtol=0,atol=0)


@pytest.mark.parametrize('algorithm', ['dflash','dflash2','dspark'])
@pytest.mark.parametrize('chunk_size', [0,1])
def test_dflash_family_logging_gate_preserves_gradients_and_skips_replay(algorithm,chunk_size):
    torch.manual_seed(35)
    classes=dict(dflash=dflash.OnlineDFlashModel,dflash2=dflash2.OnlineDFlash2Model,
                 dspark=dspark.OnlineDSparkModel)
    template=classes[algorithm](capture(models_only=True)[algorithm],
        nn.Linear(16,32,bias=False),nn.Embedding(32,16),mask_token_id=31,
        block_size=4,num_anchors=2,attention_backend='eager',objective_chunk_blocks=chunk_size)
    template.lm_head.requires_grad_(False);template.embed_tokens.requires_grad_(False)
    ids=torch.arange(12).view(1,-1);context=torch.randn(1,12,16);teacher=torch.randn(1,12,16)
    outputs=[]
    for collect in [True,False]:
        online=copy.deepcopy(template);torch.manual_seed(7)
        diagnostic=dspark if algorithm=='dspark' else None
        patches=[]
        if diagnostic:
            patches=[mock.patch.object(dspark,'block_accepted_length_terms',wraps=dspark.block_accepted_length_terms),
                     mock.patch.object(dspark,'weighted_ce_l1',wraps=dspark.weighted_ce_l1)]
        else:
            patches=[mock.patch.object(online,'_dflash_metric_chunk_terms',wraps=online._dflash_metric_chunk_terms)]
        from contextlib import ExitStack
        with ExitStack() as stack:
            spies=[stack.enter_context(patch) for patch in patches]
            output=online(input_ids=ids,hidden_states=context,loss_mask=torch.ones_like(ids),
                target_last_hidden_states=teacher,collect_detailed_metrics=collect)
            before=spies[0].call_count
            output.loss_terms.numerator.backward()
            assert bool(before)==collect
            assert spies[0].call_count==before,'Diagnostic work repeated in checkpoint backward'
            if diagnostic:
                assert bool(next(iter(spies[1].call_args_list)).kwargs['diagnostics'])==collect
                if collect and chunk_size:
                    assert any(not c.kwargs['diagnostics'] for c in spies[1].call_args_list)
        assert bool(output.ratio_metrics)==collect
        outputs.append((output.loss_terms.numerator.detach(),output.loss_terms.denominator,gradients(online)))
    torch.testing.assert_close(outputs[1][0],outputs[0][0],rtol=0,atol=0)
    torch.testing.assert_close(outputs[1][1],outputs[0][1],rtol=0,atol=0)
    assert_gradients_equal(outputs[1][2],outputs[0][2])


@pytest.mark.parametrize('disaggregate', [False,True])
@pytest.mark.parametrize('interval', [2,99])
def test_runner_collects_all_microbatches_in_regular_and_forced_tail_log_windows(disaggregate,interval):
    session=make_session(disaggregate=disaggregate)
    from dataclasses import replace
    session.args.log_interval=interval
    session.phases=[replace(session.phases[0],epochs=3,log_at_schedule_end=False)]
    session.total_steps=3
    logged=[]
    run_draft(session,batch_source=source,save=lambda *a:None,
              log=lambda s,w,n:logged.append(s.state.optimizer_step))
    # Nine batches, accumulation four: step 2 can be a regular full-window
    # log and step 3 is a forced partial-window log outside that interval.
    assert logged==([2,3] if interval==2 else [3])
    for _,_,optimizer_step,_,_,collect in session.adapter.calls:
        assert collect==(optimizer_step==2 or (interval==2 and optimizer_step==1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape', [(7,513),(32,151936)])
@pytest.mark.parametrize('chunk_size', [0,4])
def test_compiled_confidence_targets_and_gradients_are_identical_on_log_steps(monkeypatch,dtype,shape,chunk_size):
    from litespec.core import losses
    from litespec.core.chunking import checkpointed_chunk_reduce,is_chunk_recomputing
    monkeypatch.setenv('LITESPEC_LOSS_BACKEND','inductor')
    torch.manual_seed(39)
    raw=torch.randn(shape,device='cuda').to(dtype)
    teacher=(torch.randn(shape,device='cuda')*3).to(dtype)
    labels=torch.randint(shape[-1],(shape[0],),device='cuda')
    weights=torch.rand(shape[0],device='cuda');weights[::3]=0
    confidence=torch.randn(shape[0],device='cuda').to(dtype)
    outputs=[]
    for collect in [True,False]:
        x=raw.clone().requires_grad_(True);c=confidence.clone().requires_grad_(True)
        targets=[]
        def objective(logits,pred,target,ids,w):
            terms=losses.weighted_ce_l1(logits.float(),ids,target,w,
                diagnostics=collect and not is_chunk_recomputing())
            probability=(1-.5*terms[3]).clamp(0,1)
            targets.append(probability.detach().clone())
            bce=losses.weighted_binary_cross_entropy(pred.float(),probability,w)
            return terms[0],terms[1],bce
        terms=checkpointed_chunk_reduce(objective,x,c,teacher,labels,weights,
                                       chunk_size=chunk_size)
        total=.1*terms[0]+.9*terms[1]+terms[2]
        grad=torch.autograd.grad(total,(x,c))
        outputs.append((total.detach(),grad,torch.cat(targets[:(shape[0]+(chunk_size or shape[0])-1)//(chunk_size or shape[0])])) )
    torch.testing.assert_close(outputs[1][0],outputs[0][0],rtol=0,atol=0)
    for a,b in zip(outputs[1][1],outputs[0][1]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(outputs[1][2],outputs[0][2],rtol=0,atol=0)


@pytest.mark.parametrize('scheduled_final', [False,True])
def test_runner_scheduled_final_full_window_metrics(scheduled_final):
    from dataclasses import replace
    session=make_session()
    session.args.accumulation_steps=2
    session.args.log_interval=99
    session.phases=[replace(session.phases[0],log_at_schedule_end=scheduled_final)]
    session.total_steps=3
    logged=[]
    run_draft(session,batch_source=source,save=lambda *a:None,
              log=lambda s,w,n:logged.append(s.state.optimizer_step))
    assert logged==([3] if scheduled_final else [])
    for _,_,optimizer_step,_,_,collect in session.adapter.calls:
        assert collect==(scheduled_final and optimizer_step==2)
