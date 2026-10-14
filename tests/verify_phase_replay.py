import json
from pathlib import Path
parent=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu')
run=parent/'phase_diagnosis_20261014';checks={}
for backend,reference in [('auto','auto_b'),('torch','torch_a')]:
 checks[backend]={}
 for rank in [0,1]:
  paths=[run/backend/'telemetry',parent/'warmup200_steps300'/reference/'telemetry']
  data=[[json.loads(x) for x in (p/f'steps_rank_{rank}.jsonl').read_text().splitlines()] for p in paths]
  assert len(data[0])==len(data[1])==300
  initials=[json.loads((p/f'init_rank_{rank}.json').read_text()) for p in paths]
  same_inputs=all((a['input_sha256'],a['anchors_sha256'])==(b['input_sha256'],b['anchors_sha256']) for a,b in zip(*data,strict=True))
  maxloss=max(abs(a['global_weighted_loss']-b['global_weighted_loss']) for a,b in zip(*data,strict=True))
  maxnorm=max(abs(a['grad_norm']-b['grad_norm']) for a,b in zip(*data,strict=True))
  assert same_inputs and maxloss==0 and maxnorm==0 and initials[0]['initial_master_sha256']==initials[1]['initial_master_sha256']
  checks[backend][str(rank)]=dict(initial_equal=True,inputs_and_anchors_equal=True,max_loss_abs=maxloss,max_grad_norm_abs=maxnorm)
summary=json.loads((run/'phase_summary.json').read_text());summary['instrumentation_replay_checks']=checks
(run/'phase_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(checks,indent=2))
