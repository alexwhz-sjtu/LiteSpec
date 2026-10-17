import gc,json,re
from pathlib import Path
import torch
ROOT=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp')
TASK=ROOT/'tests/gdn_context_memory_20261017'
p20=ROOT/'output/composable_gdn_smoke_20261017/epoch_0_step_20'
p22=ROOT/'output/composable_gdn_smoke_20261017_1/epoch_0_step_22'
def tensors(value):
 if isinstance(value,torch.Tensor):yield value
 elif isinstance(value,dict):
  for v in value.values():yield from tensors(v)
 elif isinstance(value,(list,tuple)):
  for v in value:yield from tensors(v)
def load(path):return torch.load(path,map_location='cpu',weights_only=False)
a=load(p20/'training_state.pt');b=load(p22/'training_state.pt')
assert a['cursor']['optimizer_step']==20 and b['cursor']['optimizer_step']==22
assert a['scheduler_total_steps']==b['scheduler_total_steps']==59724
assert a['train_data_identity']==b['train_data_identity']
assert a['composable_recipe']==b['composable_recipe']
records=[];positions={}
for rank in range(8):
 old=load(p20/f'training_state_rank_{rank:05d}.pt');new=load(p22/f'training_state_rank_{rank:05d}.pt')
 values=list(tensors(new['optimizer_state_dict']))+new['fp32_master_params']
 assert all(torch.isfinite(t).all() for t in values if t.is_floating_point())
 assert all(t.dtype==torch.float32 for t in new['fp32_master_params'])
 update=0.
 for name,p,q in zip(new['optimizer_parameter_names'],old['fp32_master_params'],new['fp32_master_params']):
  delta=float((p-q).abs().sum());update+=delta
  if 'position_embedding' in name:positions[name]=positions.get(name,0.)+delta
 assert update>0
 records.append({'rank':rank,'all_optimizer_tensors_finite':True,'fp32_master_update_l1':update})
 del old,new,values;gc.collect()
assert len(positions)==5 and all(v>0 for v in positions.values()),positions
steps=[]
for file in ['train_20.log','resume_2.log']:
 text=(TASK/file).read_text()
 for match in re.finditer(r'composable optimizer_step=(\d+) micro_step=(\d+) loss=([\d.eE+\-]+)',text):
  steps.append({'optimizer_step':int(match[1]),'micro_step':int(match[2]),'loss':float(match[3])})
assert [row['optimizer_step'] for row in steps]==list(range(1,23)),steps
assert all(torch.isfinite(torch.tensor(row['loss'])) for row in steps)
result={'state':'passed','checkpoint_20':str(p20),'checkpoint_22':str(p22),'scheduler_total_steps':b['scheduler_total_steps'],'scheduler_20':a['scheduler_state_dict'],'scheduler_22':b['scheduler_state_dict'],'optimizer_shards':records,'position_embedding_fp32_master_update_l1_during_resume':positions,'steps':steps,'resources':json.loads((TASK/'training_result.json').read_text())}
(TASK/'training_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['scheduler_20','scheduler_22','resources']},indent=2))
