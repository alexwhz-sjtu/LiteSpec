"""Run after installing the pinned GDN extra; includes an actual FLA backward."""

import ast
import json
import os
from pathlib import Path

os.environ.setdefault("FLA_TILELANG", "1")

import pytest
import torch
from torch.nn import functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from litespec.modeling.draft.composable.backbone.attention.gated_deltanet import reference_event_scan


def official_recurrence():
    path = Path("refrence/backbone/gated_deltanet.py")
    tree = ast.parse(path.read_text())
    wanted = {"l2norm", "torch_recurrent_gated_delta_rule", "Qwen3_5RMSNormGated"}
    scope = {"torch": torch, "F": F, "nn": torch.nn}
    module = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted], type_ignores=[])
    exec(compile(module, str(path), "exec"), scope)
    return scope


def test_official_reference_math():
    torch.manual_seed(20)
    q,k,v=[torch.randn(1,13,2,8,requires_grad=True) for _ in range(3)]
    g=(-torch.rand(1,13,2)).requires_grad_()
    beta=torch.rand(1,13,2,requires_grad=True)
    initial=torch.randn(1,2,8,8,requires_grad=True)
    oracle=official_recurrence()
    expected,es=oracle["torch_recurrent_gated_delta_rule"](q,k,v,g,beta,initial,True,True)
    actual,s=reference_event_scan(q,k,v,g,beta,torch.tensor([0,13]),initial)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(s,es,atol=1e-6,rtol=1e-5)
    w=torch.randn_like(actual)
    ga=torch.autograd.grad((actual*w).sum()+s.sum(),(q,k,v,g,beta,initial),retain_graph=True)
    gb=torch.autograd.grad((expected*w).sum()+es.sum(),(q,k,v,g,beta,initial))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("dim",[32,128])
@pytest.mark.parametrize("value_heads",[2,4])
def test_fla_read_events_forward_and_backward(dim,value_heads):
    torch.manual_seed(21)
    device="cuda"
    # Context writes at every fourth event; three read-only queries follow.
    e=67
    q=torch.randn(1,e,2,dim,device=device,dtype=torch.bfloat16,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.randn(1,e,value_heads,dim,device=device,dtype=torch.bfloat16,requires_grad=True)
    gate=(-torch.rand(1,e,value_heads,device=device)*.1).requires_grad_()
    beta=torch.rand(1,e,value_heads,device=device,dtype=torch.bfloat16,requires_grad=True)
    writes=(torch.arange(e,device=device)%4==0)[None,:,None]
    qe=q*(~writes)[...,None]
    ke=k*writes[...,None];ve=v*writes[...,None]
    g=gate*writes;b=beta*writes
    # Two packed documents, including a non-chunk-aligned boundary.
    cpu=torch.tensor([0,29,e]);cu=cpu.to(device)
    actual,state=chunk_gated_delta_rule(qe,ke,ve,g,b,use_qk_l2norm_in_kernel=True,output_final_state=True,state_v_first=False,cu_seqlens=cu,cu_seqlens_cpu=cpu,use_beta_sigmoid_in_kernel=False,use_gate_in_kernel=False)
    expected,refstate=reference_event_scan(qe,ke,ve,g,b,cpu)
    weight=torch.randn_like(expected.float())/e
    ga=torch.autograd.grad((actual.float()*weight).sum(),(q,k,v,gate,beta),retain_graph=True)
    gb=torch.autograd.grad((expected.float()*weight).sum(),(q,k,v,gate,beta))
    def error(a,b):
        return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8))
    errors={"output":error(actual,expected),"state":error(state,refstate)}
    errors.update({name:error(a,b) for name,a,b in zip(["q","k","v","g","beta"],ga,gb)})
    print(json.dumps({"dim":dim,"value_heads":value_heads,"relative_l2_errors":errors}))
    assert all(torch.isfinite(x).all() for x in (actual,state,*ga))
    assert max(errors.values())<.035,errors


def test_fla_initial_state_gradient_and_continuation():
    torch.manual_seed(22)
    q,k,v=[torch.randn(1,5,2,32,device="cuda",dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(1,5,2,device="cuda");b=torch.rand(1,5,2,device="cuda",dtype=torch.bfloat16)
    state=torch.randn(1,2,32,32,device="cuda",requires_grad=True)
    actual,final=chunk_gated_delta_rule(q,k,v,g,b,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True,state_v_first=False)
    expected,ef=reference_event_scan(q,k,v,g,b,torch.tensor([0,5]),state)
    ga=torch.autograd.grad(actual.float().sum()+final.sum(),state,retain_graph=True)[0]
    gb=torch.autograd.grad(expected.float().sum()+ef.sum(),state)[0]
    assert float((ga-gb).norm()/gb.norm())<.035


@pytest.mark.parametrize('context_kernel',[1,4,8])
def test_whole_model_cuda_cache_and_rectangular_state(context_kernel):
    import copy
    from test_gdn import config
    from litespec.modeling.draft.composable import ComposableDraftModel, ModelInputs, DraftState
    torch.manual_seed(27)
    cfg=config(ckernel=context_kernel,backend='fla',hv=4,dv=64)
    cfg.composable_config['backbone']['linear_key_head_dim']=32
    model=ComposableDraftModel(cfg).cuda().bfloat16().eval()
    c=torch.randn(2,9,64,device='cuda',dtype=torch.bfloat16)
    x=torch.randn(2,4,32,device='cuda',dtype=torch.bfloat16)
    state=DraftState()
    def run(start,end,cached=True):
        inp=ModelInputs(c[:,start:end],x,torch.full((2,1),end,device='cuda'),torch.ones(2,1,device='cuda',dtype=torch.bool),context_position_ids=torch.arange(start,end,device='cuda')[None].expand(2,-1),draft_state=state if cached else None,mode='features')
        return model(inp).backbone_hidden_states
    with torch.no_grad():
        run(0,3);run(3,4)
        actual=run(4,8);full=run(0,8,False)
        torch.testing.assert_close(actual,full,atol=.03,rtol=.03)
        snapshot=copy.deepcopy(state)
        repeat=run(8,8)
        torch.testing.assert_close(repeat,actual,atol=.03,rtol=.03)
        for i,layer in state.backbone_state.layers.items():
            torch.testing.assert_close(layer.recurrent_state,snapshot.backbone_state.layers[i].recurrent_state,atol=0,rtol=0)
            assert layer.recurrent_state.shape==(2,4,32,64)
            if context_kernel==1:assert layer.kv_conv_state is None
            else:torch.testing.assert_close(layer.kv_conv_state,snapshot.backbone_state.layers[i].kv_conv_state,atol=0,rtol=0)
