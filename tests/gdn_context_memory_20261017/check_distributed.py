"""Two-rank FSDP/Muon update and global-denominator parity, including zero rank."""

import copy
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("FLA_TILELANG", "1")
import torch
import torch.distributed as dist
from torch import nn

from test_gdn import config
from litespec.modeling.draft.composable import ComposableDraftModel
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.core.composable.batch import prepare_layout
from litespec.training.setup import wrap_training_model
from litespec.training.fsdp_layout import build_fsdp_layouts
from litespec.training.optimizer import register_optimizer_parameters, BF16Optimizer


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    results=[]
    for zero_rank in (False,True):
        torch.manual_seed(31)
        cfg=config(backend='fla',proposal='selector')
        cfg.composable_config['backbone'].update(linear_key_head_dim=32,linear_value_head_dim=32)
        model=ComposableDraftModel(cfg).to(device='cuda',dtype=torch.bfloat16)
        model.backbone.gradient_checkpointing=zero_rank
        baseline=copy.deepcopy(model)
        baseline.backbone.gradient_checkpointing=False
        for layer in baseline.backbone.layers:layer.mixer.backend='reference'
        embedding=nn.Embedding(23,32).to(device='cuda',dtype=torch.bfloat16).requires_grad_(False)
        head=nn.Linear(32,23,bias=False).to(device='cuda',dtype=torch.bfloat16).requires_grad_(False)
        wrapper=ComposableTrainingModel(model,head,embedding,0,num_anchors=3,objective_chunk_blocks=1)
        reference=ComposableTrainingModel(baseline,head,embedding,0,num_anchors=3,objective_chunk_blocks=0)
        parameters,specs=register_optimizer_parameters(model)
        wrapped=wrap_training_model(wrapper,SimpleNamespace(lm_head=head,embed_tokens=embedding))
        layouts=build_fsdp_layouts(wrapped,parameters,specs)
        optimizer=BF16Optimizer(model,lr=.001,warmup_ratio=0,total_steps=5,max_grad_norm=1,
            parameters=parameters,parameter_specs=specs,parameter_layouts=layouts,
            muon_config={'weight_decay':0,'adamw_weight_decay':0,'adamw_lr':.001})
        torch.manual_seed(32)
        ids=torch.randint(23,(2,12));mask=torch.ones(2,12)
        if zero_rank:mask[1]=0
        context=torch.randn(2,12,64).to(device='cuda',dtype=torch.bfloat16)
        global_layout=prepare_layout(ids,mask,4,3)
        local_layout=global_layout.batch_slice(rank,rank+1).to('cuda')
        totals=torch.stack((torch.tensor(1.,device='cuda'),wrapper.objective.native_denominator(local_layout),wrapper.selector_objective.native_denominator(local_layout))).double()
        dist.all_reduce(totals)
        loss=wrapped(input_ids=ids[rank:rank+1].cuda(),target_hidden_states=context[rank:rank+1],loss_mask=mask[rank:rank+1].cuda(),layout=local_layout,window_denominators=totals,context_valid_mask=torch.ones(1,12,device='cuda',dtype=torch.bool)).loss
        assert torch.isfinite(loss)
        loss.backward()
        ref_loss=reference(ids.cuda(),context,mask.cuda(),layout=global_layout.to('cuda'),window_denominators=totals).loss
        ref_loss.backward()
        refs=dict(baseline.named_parameters());errors=torch.zeros(2,device='cuda',dtype=torch.float64)
        for parameter,spec,layout in zip(parameters,specs,layouts):
            offset,length=layout.shards[rank]
            if not length:continue
            expected=refs[spec.name].grad.flatten()[offset:offset+length].float()
            assert parameter.grad is not None,spec.name
            actual=parameter.grad.flatten().float()
            errors[0]+=(actual-expected).double().square().sum()
            errors[1]+=expected.double().square().sum()
        dist.all_reduce(errors)
        relative=float((errors[0]/errors[1].clamp_min(1e-20)).sqrt())
        assert relative<.065,relative
        before=[p.detach().clone() for p in optimizer.fp32_params]
        grad_norm=optimizer.step()
        changed=sum(float((p-b).abs().sum()) for p,b in zip(optimizer.fp32_params,before))
        counts=torch.tensor([changed],device='cuda');dist.all_reduce(counts)
        assert counts.item()>0 and torch.isfinite(torch.tensor(grad_norm))
        results.append({'zero_supervision_rank':zero_rank,'checkpointing':zero_rank,'gradient_relative_l2':relative,'global_parameter_change_l1':counts.item(),'grad_norm':grad_norm})
        del wrapped,optimizer,model,baseline,reference,wrapper,parameters
        torch.cuda.empty_cache();dist.barrier()
    if rank==0:
        root=Path('tests/gdn_context_memory_20261017')
        (root/'distributed_result.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
