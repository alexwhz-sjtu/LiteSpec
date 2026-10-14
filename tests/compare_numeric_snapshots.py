import json,torch
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014')
torch.set_num_threads(4)
result={}
for name in ['auto','native_ce_compiled_l1']:
 result[name]={}
 for rank in [0,1]:
  per_rank={}
  for filename,key in [('gradients','gradients'),('masters','masters')]:
   a=torch.load(root/'torch/numeric'/f'{filename}_rank_{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
   b=torch.load(root/name/'numeric'/f'{filename}_rank_{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
   assert a['names']==b['names']
   differences=[];elements=changed=bf16_changed=0;max_abs=0.
   for param,x,y in zip(a['names'],a[key],b[key],strict=True):
    assert (x is None)==(y is None)
    if x is None or not x.numel():continue
    different=int((x!=y).sum());bf16=int((x.bfloat16()!=y.bfloat16()).sum())
    elements+=x.numel();changed+=different;bf16_changed+=bf16
    if different:
     delta=(x.float()-y.float()).abs();maximum=delta.max().item();max_abs=max(max_abs,maximum)
     flat_index=int(delta.flatten().argmax());differences.append(dict(name=param,shape=list(x.shape),dtype=str(x.dtype),changed=different,bf16_changed=bf16,max_abs=maximum,reference_at_max=x.flatten()[flat_index].item(),actual_at_max=y.flatten()[flat_index].item(),flat_index_at_max=flat_index))
   per_rank[key]=dict(elements=elements,changed=changed,bf16_changed=bf16_changed,max_abs=max_abs,affected_parameters=len(differences),differences=differences)
   del a,b
  result[name][str(rank)]=per_rank
(root/'snapshot_comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps({name:{rank:{kind:{k:v for k,v in stats.items() if k!='differences'} for kind,stats in kinds.items()} for rank,kinds in ranks.items()} for name,ranks in result.items()},indent=2))
