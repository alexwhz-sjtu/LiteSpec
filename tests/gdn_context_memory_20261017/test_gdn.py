"""GDN semantics, independent oracles and framework regression checks."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from litespec.modeling.draft.composable import (
    ComposableDraftConfig, ComposableDraftModel, ModelInputs, DraftState, SharedOps,
)
from litespec.modeling.draft.composable.backbone.attention.gated_deltanet import l2norm
from litespec.modeling.draft.composable.backbone.gdn_events import build_event_layout
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.batch import prepare_layout
from litespec.training.optimizer import register_optimizer_parameters


def config(qkernel=2, ckernel=4, backend="reference", context="dflash", proposal="parallel", hv=2, dv=8):
    return ComposableDraftConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_target_layers=8, vocab_size=23, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, block_size=4,
        composable_config={
            "context": {"type": context, "target_layer_ids": [1, 5]},
            "backbone": {
                "type": "gdn_context_memory", "layers": [{}, {}],
                "linear_num_key_heads": 2, "linear_num_value_heads": hv,
                "linear_key_head_dim": 8, "linear_value_head_dim": dv,
                "query_conv_kernel_size": qkernel, "context_conv_kernel_size": ckernel,
                "backend": backend,
            },
            "proposal": {"type": proposal, "rank": 8, "top_k": 5},
            "objective": {"type": "hard_ce"},
        },
    )


def make_inputs(n=2):
    return ModelInputs(
        torch.randn(n, 10, 64), torch.randn(n, 12, 32),
        torch.tensor([[7, 2, 2]]).expand(n, -1).clone(),
        torch.ones(n, 3, dtype=torch.bool), mode="features",
    )


def direct_prefix_read(m, x, c, inputs, block_size):
    """Independent implementation: explicitly recompute each small prefix."""
    n, total, _ = x.shape
    anchors = inputs.anchor_positions.size(1)
    outputs = []
    for b in range(n):
        rows = []
        for a in range(anchors):
            if not inputs.block_keep_mask[b, a]:
                rows.append(x.new_zeros(block_size, m.hv, m.dv))
                continue
            p = inputs.anchor_positions[b, a]
            pos = torch.arange(c.size(1)) if inputs.context_position_ids is None else inputs.context_position_ids[b]
            valid = torch.ones_like(pos, dtype=torch.bool) if inputs.context_valid_mask is None else inputs.context_valid_mask[b]
            docs = torch.zeros_like(pos) if inputs.document_ids is None else inputs.document_ids[b]
            doc = 0 if inputs.anchor_document_ids is None else inputs.anchor_document_ids[b, a]
            features = c[b, valid & (pos < p) & (docs == doc)]
            raw = F.linear(features, m.in_proj_kv.weight)
            if len(raw):
                kv = F.silu(F.conv1d(F.pad(raw.T[None], (m.context_kernel-1, 0)), m.context_conv.weight, groups=raw.size(-1)))[0].T
            else:
                kv = raw
            key, value = kv.split((m.key_dim, m.value_dim), -1)
            key = l2norm(key.reshape(-1, m.hk, m.dk)).repeat_interleave(m.hv//m.hk, 1)
            value = value.reshape(-1, m.hv, m.dv)
            beta = F.linear(features, m.in_proj_b.weight).sigmoid()
            g = -m.A_log.float().exp() * F.softplus(F.linear(features, m.in_proj_a.weight).float()+m.dt_bias.float())
            state = x.new_zeros(m.hv, m.dk, m.dv)
            for t in range(len(features)):
                state = state * g[t].exp()[:, None, None]
                old = torch.einsum("hk,hkv->hv", key[t], state)
                state = state + key[t][:, :, None] * (beta[t][:, None]*(value[t]-old))[:, None, :]
            block = x[b, a*block_size:(a+1)*block_size]
            raw_q = F.linear(block, m.in_proj_q.weight)
            query = F.silu(F.conv1d(F.pad(raw_q.T[None], (m.q_kernel-1, 0)), m.query_conv.weight, groups=m.key_dim))[0].T
            query = l2norm(query.reshape(block_size, m.hk, m.dk)).repeat_interleave(m.hv//m.hk, 1)/(m.dk**.5)
            rows.append(torch.einsum("lhk,hkv->lhv", query, state))
        outputs.append(torch.stack(rows).flatten(0, 1))
    out = torch.stack(outputs)
    z = F.linear(x, m.in_proj_z.weight).reshape_as(out)
    normed = out * torch.rsqrt(out.square().mean(-1, keepdim=True)+m.norm.eps)
    gated = normed*m.norm.weight*F.silu(z)
    return F.linear(gated.flatten(-2), m.out_proj.weight)


@pytest.mark.parametrize("qkernel", [1, 2, 4])
@pytest.mark.parametrize("ckernel", [1, 2, 4, 8])
def test_event_prefix_output_and_gradients(qkernel, ckernel):
    torch.manual_seed(11)
    model = ComposableDraftModel(config(qkernel, ckernel, hv=4, dv=12))
    m = model.backbone.layers[0].mixer
    with torch.no_grad():
        m.A_log.fill_(-2); m.dt_bias.fill_(-1)
    inputs = make_inputs(1)
    x = inputs.draft_embeddings.requires_grad_()
    c = torch.randn(1, 10, 32, requires_grad=True)
    layout = build_event_layout(inputs, 4)
    actual, _ = m(x, c, layout, 4)
    expected = direct_prefix_read(m, x, c, inputs, 4)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weight = torch.randn_like(actual)
    parameters = (x, c, *m.parameters())
    ga = torch.autograd.grad((actual*weight).sum(), parameters, retain_graph=True)
    gb = torch.autograd.grad((expected*weight).sum(), parameters)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("context", ["dflash", "dflare"])
@pytest.mark.parametrize("proposal", ["parallel", "selector"])
def test_training_gradients_and_optimizer(context, proposal):
    torch.manual_seed(12)
    model = ComposableDraftModel(config(context=context, proposal=proposal))
    embedding, head = nn.Embedding(23, 32), nn.Linear(32, 23, bias=False)
    embedding.requires_grad_(False); head.requires_grad_(False)
    wrapper = ComposableTrainingModel(model, head, embedding, 0, num_anchors=3, objective_chunk_blocks=1)
    result = wrapper(torch.randint(23, (2, 10)), torch.randn(2, 10, 64), torch.ones(2, 10))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    _, specs = register_optimizer_parameters(model)
    by_name = {s.name: s.optimizer for s in specs}
    for name, kind in by_name.items():
        if any(k in name for k in ("position_embedding", "query_conv", "context_conv", "A_log", "dt_bias")):
            assert kind == "adamw", name
    assert by_name["backbone.layers.0.mixer.in_proj_q.weight"] == "muon"


def test_position_tables_and_exact_injection():
    model = ComposableDraftModel(config())
    layers = model.backbone.layers
    assert layers[0].position_embedding.weight.data_ptr() != layers[1].position_embedding.weight.data_ptr()
    assert sum(isinstance(m, nn.Embedding) for m in model.modules()) == 2
    inputs = make_inputs(1)
    captured = {}
    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(lambda m, args, i=i: captured.update({(i,"x"): args[0].detach().clone()})))
        hooks.append(layer.input_norm.register_forward_pre_hook(lambda m, args, i=i: captured.update({(i,"u"): args[0].detach().clone()})))
    model(inputs)
    for i, layer in enumerate(layers):
        expected = (captured[i,"x"].reshape(1,3,4,32)+layer.position_embedding.weight).reshape(1,12,32)
        torch.testing.assert_close(captured[i,"u"], expected)
    for hook in hooks: hook.remove()


@pytest.mark.parametrize("ckernel", [1, 2, 4, 8])
def test_cached_multitoken_and_read_only(ckernel):
    torch.manual_seed(13)
    model = ComposableDraftModel(config(ckernel=ckernel)).eval()
    c, x = torch.randn(2, 8, 64), torch.randn(2, 4, 32)
    state = DraftState()
    def run(start, end, cached=True):
        return model(ModelInputs(c[:,start:end],x,torch.full((2,1),end),torch.ones(2,1,dtype=torch.bool),context_position_ids=torch.arange(start,end)[None].expand(2,-1),draft_state=state if cached else None,mode="features"))
    with torch.no_grad():
        run(0,3)
        actual = run(3,7).backbone_hidden_states
        expected = run(0,7,False).backbone_hidden_states
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        before = copy.deepcopy(state)
        again = run(7,7).backbone_hidden_states
        torch.testing.assert_close(again, actual)
        for i, layer_state in state.backbone_state.layers.items():
            torch.testing.assert_close(layer_state.recurrent_state, before.backbone_state.layers[i].recurrent_state)
            assert layer_state.recurrent_state.dtype == torch.float32
            if ckernel == 1: assert layer_state.kv_conv_state is None
            else:
                assert layer_state.kv_conv_state.size(-1) == ckernel-1
                torch.testing.assert_close(layer_state.kv_conv_state,before.backbone_state.layers[i].kv_conv_state)
    with pytest.raises(ValueError, match="duplicate, missing"):
        run(3,7)
    with pytest.raises(ValueError, match="snapshot"):
        state.rollback(6)
    state.rollback(7)


def test_future_document_padding_and_block_isolation():
    torch.manual_seed(15)
    model = ComposableDraftModel(config()).eval()
    inputs = make_inputs(1)
    inputs.anchor_positions = torch.tensor([[3,8,8]])
    inputs.document_ids = torch.tensor([[0,0,0,0,1,1,1,1,1,1]])
    inputs.anchor_document_ids = torch.tensor([[0,1,1]])
    inputs.context_valid_mask = torch.tensor([[1,1,1,1,0,1,1,1,0,0]],dtype=torch.bool)
    with torch.no_grad():
        expected = model(inputs).backbone_hidden_states.reshape(1,3,4,32)
        changed = copy.deepcopy(inputs)
        changed.target_hidden_states[:,8:] += 100
        torch.testing.assert_close(model(changed).backbone_hidden_states.reshape_as(expected),expected)
        changed = copy.deepcopy(inputs)
        changed.target_hidden_states[:,:4] += 5*torch.randn_like(changed.target_hidden_states[:,:4])
        actual = model(changed).backbone_hidden_states.reshape_as(expected)
        torch.testing.assert_close(actual[:,1:],expected[:,1:])
        changed = copy.deepcopy(inputs)
        changed.draft_embeddings[:,:4] += torch.randn_like(changed.draft_embeddings[:,:4])
        torch.testing.assert_close(model(changed).backbone_hidden_states.reshape_as(expected)[:,1:],expected[:,1:])


def test_all_invalid_zero_loss_and_gradients():
    model = ComposableDraftModel(config())
    wrapper = ComposableTrainingModel(model, nn.Linear(32,23,bias=False),nn.Embedding(23,32),0,num_anchors=3)
    result = wrapper(torch.randint(23,(2,10)),torch.randn(2,10,64),torch.zeros(2,10))
    assert result.loss.item() == 0
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpoint_and_loss_chunk_gradients():
    torch.manual_seed(16)
    first = ComposableDraftModel(config(proposal="selector"))
    second = copy.deepcopy(first)
    second.backbone.gradient_checkpointing = True
    ids, c, mask = torch.randint(23,(2,10)),torch.randn(2,10,64),torch.ones(2,10)
    layout = prepare_layout(ids,mask,4,3)
    embedding, head = nn.Embedding(23,32),nn.Linear(32,23,bias=False)
    values=[]
    for model,chunk in [(first,0),(second,1)]:
        loss=ComposableTrainingModel(model,head,embedding,0,num_anchors=3,objective_chunk_blocks=chunk)(ids,c,mask,layout=layout).loss
        values.append(loss.detach());loss.backward()
    torch.testing.assert_close(*values)
    for (name,p),(_,q) in zip(first.named_parameters(),second.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5,msg=name)


@pytest.mark.parametrize("key,value",[("query_conv_kernel_size",0),("context_conv_kernel_size",False),("context_conv_kernel_size",1.5),("linear_num_value_heads",3)])
def test_invalid_configuration(key,value):
    cfg=config();recipe=copy.deepcopy(cfg.composable_config);recipe["backbone"][key]=value
    with pytest.raises(ValueError):
        ComposableDraftConfig(hidden_size=32,num_hidden_layers=2,composable_config=recipe)


def test_checkpoint_export(tmp_path):
    from litespec.modeling.draft.composable.export import export_composable
    from transformers import AutoModel
    torch.manual_seed(17)
    model=ComposableDraftModel(config()).eval();inputs=make_inputs(1)
    expected=model(inputs).backbone_hidden_states
    model.save_pretrained(tmp_path/'normal')
    restored=ComposableDraftModel.from_pretrained(tmp_path/'normal').eval()
    torch.testing.assert_close(restored(inputs).backbone_hidden_states,expected)
    export_composable(model,tmp_path/'export')
    restored=AutoModel.from_pretrained(tmp_path/'export',trust_remote_code=True).eval()
    torch.testing.assert_close(restored(inputs).backbone_hidden_states,expected)


def test_full_schedule_stop_cap_resume_contract(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from litespec.training.algorithms import RunState
    from litespec.training import checkpoint as module
    cursor=RunState(phase_index=0,phase_step=20,global_step=20,optimizer_step=20,epoch=0,next_batch=20)
    old_plan=[{"name":"composable","epochs":6,"max_batches":20,"schedule_full_epochs":True}]
    expected={"phase_plan":[{**old_plan[0],"max_batches":22}],"scheduler_total_steps":6000}
    saved={"phase_plan":old_plan,"scheduler_total_steps":6000,"cursor":asdict(cursor),"optimizer_step":20}
    session=SimpleNamespace(resume_state=saved,phases=[SimpleNamespace(batch_count=lambda _:22)],steps_per_epoch=1000,args=SimpleNamespace(accumulation_steps=1))
    monkeypatch.setattr(module,'session_metadata',lambda _:expected)
    module.validate_session_resume(session)
    assert session.state.optimizer_step==20
    expected['phase_plan'][0]['epochs']=7
    with pytest.raises(ValueError,match='phase_plan'):module.validate_session_resume(session)
    expected['phase_plan'][0]['epochs']=6
    session.phases=[SimpleNamespace(batch_count=lambda _:19)]
    with pytest.raises(ValueError,match='cursors'):module.validate_session_resume(session)
    session.phases=[SimpleNamespace(batch_count=lambda _:22)]
    old_plan[0]['schedule_full_epochs']=False
    expected['phase_plan'][0]['schedule_full_epochs']=False
    with pytest.raises(ValueError,match='phase_plan'):module.validate_session_resume(session)


def test_empty_prefix_document_reset_and_special_initialization():
    torch.manual_seed(29)
    model=ComposableDraftModel(config()).eval()
    for layer in model.backbone.layers:
        torch.testing.assert_close(layer.mixer.dt_bias,torch.ones_like(layer.mixer.dt_bias))
        assert ((layer.mixer.A_log.exp()>0)&(layer.mixer.A_log.exp()<=16)).all()
    context=torch.randn(1,5,64);query=torch.randn(1,4,32)
    state=DraftState()
    def run(start,end,document,cached=True):
        return model(ModelInputs(context[:,start:end],query,torch.tensor([[end]]),torch.ones(1,1,dtype=torch.bool),context_position_ids=torch.arange(start,end)[None],document_ids=torch.full((1,end-start),document),anchor_document_ids=torch.tensor([[document]]),draft_state=state if cached else None,mode='features')).backbone_hidden_states
    with torch.no_grad():
        torch.testing.assert_close(run(0,0,0),run(0,0,0,False))
        for end in range(1,6):
            incremental=run(end-1,end,0)
            torch.testing.assert_close(incremental,run(0,end,0,False),atol=4e-6,rtol=4e-5)
        torch.testing.assert_close(run(0,3,1),run(0,3,1,False),atol=4e-6,rtol=4e-5)
        assert state.backbone_state.document_ids.item()==1


def test_gradient_accumulation_uses_window_denominators():
    torch.manual_seed(33)
    full=ComposableDraftModel(config(proposal='selector'))
    accumulated=copy.deepcopy(full)
    head=nn.Linear(32,23,bias=False).requires_grad_(False)
    embedding=nn.Embedding(23,32).requires_grad_(False)
    first=ComposableTrainingModel(full,head,embedding,0,num_anchors=3,objective_chunk_blocks=2)
    second=ComposableTrainingModel(accumulated,head,embedding,0,num_anchors=3,objective_chunk_blocks=1)
    ids=torch.randint(23,(2,12));c=torch.randn(2,12,64);mask=torch.ones(2,12);mask[1,6:]=0
    layout=prepare_layout(ids,mask,4,3)
    totals=torch.stack((torch.tensor(2.),first.objective.native_denominator(layout),first.selector_objective.native_denominator(layout))).double()
    global_loss=first(ids,c,mask,layout=layout,window_denominators=totals).loss
    global_loss.backward()
    micro_losses=[]
    for sample in range(2):
        loss=second(ids[sample:sample+1],c[sample:sample+1],mask[sample:sample+1],layout=layout.batch_slice(sample,sample+1),window_denominators=totals).loss/2
        micro_losses.append(loss.detach());loss.backward()
    torch.testing.assert_close(sum(micro_losses),global_loss.detach(),atol=1e-6,rtol=1e-5)
    for (name,p),(_,q) in zip(full.named_parameters(),accumulated.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=4e-6,rtol=4e-5,msg=name)
