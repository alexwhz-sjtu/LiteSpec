"""Inspect the native CUDA CE autograd buffers on captured training rows."""
import json,torch
import torch.nn.functional as F
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
run=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014'
torch.cuda.set_device(0)
data=torch.load(run/'torch/numeric/actual_loss_rows32.pt',weights_only=True)
x=data['logits'].cuda().detach().requires_grad_(True)
labels=data['labels'].cuda();weights=data['weights'].cuda();saved=[]
def pack(t):
 saved.append(dict(shape=list(t.shape),dtype=str(t.dtype),bytes=t.numel()*t.element_size(),storage_ptr=t.untyped_storage().data_ptr(),is_logits_storage=t.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()))
 return t
with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
 values=F.cross_entropy(x,labels,reduction='none')
 loss=(values*weights).sum()
 loss.backward()
result=dict(input_shape=list(x.shape),saved_tensors=saved)
(run/'native_ce_saved_tensors.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
