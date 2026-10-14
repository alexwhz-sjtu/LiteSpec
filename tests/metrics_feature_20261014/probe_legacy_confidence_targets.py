"""Check diagnostic extraction preserves the previous CUDA confidence target."""
import importlib.util,json,os,torch
from pathlib import Path
from litespec.core import losses
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');run=root/'tests/metrics_feature_20261014'
spec=importlib.util.spec_from_file_location('losses_metrics_before',run/'before/litespec/core/losses.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
torch.cuda.set_device(0);os.environ['LITESPEC_LOSS_BACKEND']='inductor';torch.manual_seed(53)
records=[]
for dtype in [torch.float32,torch.bfloat16]:
 shape=(32,151936);raw=torch.randn(shape,device='cuda').to(dtype)
 target=(torch.randn(shape,device='cuda')*3).to(dtype)
 confidence=torch.randn(shape[0],device='cuda').to(dtype)
 labels=torch.randint(shape[-1],(shape[0],),device='cuda');w=torch.rand(shape[0],device='cuda');w[::3]=0
 outputs=[]
 for module in [before,losses]:
  x=raw.clone().requires_grad_(True);c=confidence.clone().requires_grad_(True)
  ce,l1,_,distance,*_=module.weighted_ce_l1(x.float(),labels,target,w,diagnostics=True)
  probability=(1-.5*distance).clamp(0,1)
  bce=module.weighted_binary_cross_entropy(c.float(),probability,w)
  objective=.1*ce+.9*l1+bce
  g=torch.autograd.grad(objective,(x,c));outputs.append((objective.detach(),probability.detach(),g))
 differences=dict(objective_abs=abs((outputs[0][0]-outputs[1][0]).item()),probability_max_abs=(outputs[0][1]-outputs[1][1]).abs().max().item(),logit_gradient_max_abs=(outputs[0][2][0].float()-outputs[1][2][0].float()).abs().max().item(),confidence_gradient_max_abs=(outputs[0][2][1].float()-outputs[1][2][1].float()).abs().max().item())
 records.append(dict(dtype=str(dtype),shape=shape,**differences))
 assert differences['probability_max_abs']<=1e-7 and differences['confidence_gradient_max_abs']<=1e-7 and differences['logit_gradient_max_abs']==0, differences
result=dict(passed=True,cases=records);(run/'legacy_confidence_parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
