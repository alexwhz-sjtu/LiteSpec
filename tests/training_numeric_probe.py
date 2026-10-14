"""Probe identical actual loss inputs and persist first-step FSDP gradients."""
import gc,json,os
from pathlib import Path
import torch
import torch.nn.functional as F

def compare(a,b):
 delta=(a-b).abs();ab=a.bfloat16();bb=b.bfloat16()
 return dict(elements=a.numel(),max_abs=delta.max().item(),l2_delta=delta.norm().item(),reference_l2=a.norm().item(),fp32_different=int((a!=b).sum()),bf16_different=int((ab!=bb).sum()),bf16_max_abs=(ab.float()-bb.float()).abs().max().item())

def diagnose(logits, labels, teacher, weights, directory):
 from litespec.core import losses
 directory=Path(directory);directory.mkdir(exist_ok=True)
 torch.save(dict(logits=logits[:32].detach().cpu(),labels=labels[:32].detach().cpu(),teacher=teacher[:32].detach().cpu(),weights=weights[:32].detach().cpu()),directory/'actual_loss_rows32.pt')
 results={};references={}
 def evaluate(name,kind,ce_compiled=False,l1_compiled=False,combined=False):
  x=logits.detach().clone().requires_grad_(True)
  if combined:
   target=torch.softmax(teacher.detach().float(),-1);draft=torch.softmax(x.float(),-1)
   out=losses._compiled(losses._ce_l1_probs)(x,labels,draft,target,weights)
   objective=.1*out[0]+out[1];values=[out[0].item(),out[1].item()]
  else:
   objective=0.;values=[]
   if kind in ('ce','both'):
    ce=(losses._compiled(losses._ce) if ce_compiled else losses._ce)(x,labels,weights)[0]
    objective=objective+.1*ce;values.append(ce.item())
   if kind in ('l1','both'):
    if l1_compiled:
     target=torch.softmax(teacher.detach().float(),-1);draft=torch.softmax(x.float(),-1)
     l1=losses._compiled(losses._l1_probs)(draft,target,weights)[0]
    else:l1=losses._l1(x,teacher,weights)[0]
    objective=objective+l1;values.append(l1.item())
  grad=torch.autograd.grad(objective,x)[0].detach()
  results[name]=dict(values=values,gradient_l2=grad.norm().item())
  if name in ('native_ce','native_l1','native_both'):
   references[kind]=grad
  else:
   results[name]['vs_native']=compare(references[kind],grad)
  print('LOSS_COMPONENT_PROBE '+json.dumps({name:results[name]}),flush=True)
  (directory/'component_probe.json').write_text(json.dumps(dict(shape=list(logits.shape),logit_dtype=str(logits.dtype),teacher_dtype=str(teacher.dtype),upstream=dict(ce=.1,l1=1.),results=results),indent=2))
  del x,objective;gc.collect()
 evaluate('native_ce','ce')
 evaluate('compiled_ce','ce',ce_compiled=True)
 evaluate('native_l1','l1')
 evaluate('compiled_l1','l1',l1_compiled=True)
 evaluate('native_both','both')
 evaluate('compiled_combined','both',combined=True)
 evaluate('native_ce_compiled_l1','both',l1_compiled=True)
 evaluate('compiled_ce_native_l1','both',ce_compiled=True)
 del references;gc.collect()

def install(session,runner,output,rank,component_probe=False,variant=None):
 import litespec.core.dlite as dlite
 from litespec.core import losses
 output=Path(output);(output/'numeric').mkdir(exist_ok=True)
 original_loss=dlite.weighted_ce_l1
 done=False
 def loss(logits,labels,teacher,weights,**kwargs):
  nonlocal done
  if component_probe and rank==0 and not done:
   done=True
   diagnose(logits,labels,teacher,weights,output/'numeric')
  if variant=='native_ce_compiled_l1':
   target=torch.softmax(teacher.detach().float(),-1);draft=torch.softmax(logits.float(),-1)
   ce,values=losses._ce(logits,labels,weights)
   l1=losses._compiled(losses._l1_probs)(draft,target,weights,kwargs.get('diagnostics',False))
   return ce,l1[0],values,*l1[1:]
  return original_loss(logits,labels,teacher,weights,**kwargs)
 dlite.weighted_ce_l1=loss
 original_step=runner.OptimizerWindow.step
 def step(window):
  optimizer=window.optimizer
  before=dict(names=optimizer.parameter_names,gradients=[None if p.grad is None else p.grad.detach().cpu() for p in optimizer.model_params])
  torch.save(before,output/'numeric'/f'gradients_rank_{rank}.pt');del before
  norm=original_step(window)
  torch.save(dict(names=optimizer.parameter_names,masters=[p.detach().cpu() for p in optimizer.fp32_params]),output/'numeric'/f'masters_rank_{rank}.pt')
  return norm
 runner.OptimizerWindow.step=step
