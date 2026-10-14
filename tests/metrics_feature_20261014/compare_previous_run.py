import json
from pathlib import Path
p=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu')
for rank in [0,1]:
 paths=[p/'fixed_ce_warmup200_steps300/auto_b/telemetry',p/'logging_metrics_warmup200_steps300/all_a/telemetry']
 initials=[json.loads((x/f'init_rank_{rank}.json').read_text()) for x in paths]
 print('rank',rank,'initial_masters_equal',initials[0]['initial_master_sha256']==initials[1]['initial_master_sha256'])
 rows=[[json.loads(s) for s in (x/f'steps_rank_{rank}.jsonl').read_text().splitlines()] for x in paths]
 for field in ['input_sha256','anchors_sha256']:
  print(field,all(a[field]==b[field] for a,b in zip(*rows)))
 for a,b in zip(*rows):
  if a['grad_norm']!=b['grad_norm'] or a['global_weighted_loss']!=b['global_weighted_loss']:
   print('first numerical change',a['step'],'loss',a['global_weighted_loss'],b['global_weighted_loss'],'norm',a['grad_norm'],b['grad_norm']);break
 print('max loss difference',max(abs(a['global_weighted_loss']-b['global_weighted_loss']) for a,b in zip(*rows)))
 keys=set(initials[0]['args'])|set(initials[1]['args'])
 print('args changed', {k:[initials[0]['args'].get(k),initials[1]['args'].get(k)] for k in keys if initials[0]['args'].get(k)!=initials[1]['args'].get(k)})
