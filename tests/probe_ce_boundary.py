import json,torch
import torch.nn.functional as F
from pathlib import Path
from litespec.core import losses
from training_numeric_probe import compare
root=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014')
data={k:v.cuda() for k,v in torch.load(root/'torch/numeric/actual_loss_rows32.pt',weights_only=True).items()}
torch.cuda.set_device(0)
def logsoftmax(x):return F.log_softmax(x,-1)
def gather_reduce(lp,labels,w):return (-lp.gather(-1,labels.unsqueeze(-1)).squeeze(-1)*w).sum()
results={};reference=None
for name in ['native_ce','compiled_ce','native_logsoftmax_native_nll','compiled_logsoftmax_native_nll','native_logsoftmax_compiled_gather']:
 x=data['logits'].clone().requires_grad_(True)
 if name=='native_ce':value=losses._ce(x,data['labels'],data['weights'])[0]
 elif name=='compiled_ce':value=losses._compiled(losses._ce)(x,data['labels'],data['weights'])[0]
 elif name=='native_logsoftmax_compiled_gather':value=losses._compiled(gather_reduce)(logsoftmax(x),data['labels'],data['weights'])
 else:
  lp=(losses._compiled(logsoftmax) if name.startswith('compiled') else logsoftmax)(x)
  value=(F.nll_loss(lp,data['labels'],reduction='none')*data['weights']).sum()
 grad=torch.autograd.grad(.1*value,x)[0].detach()
 if reference is None:reference=grad
 results[name]=dict(value=value.item(),vs_native=compare(reference,grad))
 print(name,results[name],flush=True)
a=logsoftmax(data['logits']);b=losses._compiled(logsoftmax)(data['logits'])
results['logsoftmax_values']=compare(a,b)
(root/'ce_boundary_probe.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
