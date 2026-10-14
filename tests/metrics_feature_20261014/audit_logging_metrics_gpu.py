import csv,json,statistics
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
run=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/logging_metrics_warmup200_steps300'
names=['all_a','logging_b','logging_b_repeat','all_a_repeat']
samples=[json.loads(line) for line in (run/'gpu_monitor.jsonl').read_text().splitlines()]
result={}
for name in names:
 initials=[json.loads((run/name/'telemetry'/f'init_rank_{rank}.json').read_text()) for rank in [0,1]]
 expected={i['gpu_process_identity']['gpu_uuid']:{i['gpu_process_identity']['driver_pid']} for i in initials}
 rows=[json.loads(line) for line in (run/name/'telemetry/steps_rank_0.jsonl').read_text().splitlines()]
 normal=[r for r in rows if not r['warmup'] and not r['new_graphs_on_any_rank'] and not r['logging_step']]
 start=min(r['wall_time_started'] for r in normal)+1;end=max(r['wall_time_ended'] for r in normal)-1
 stable=[s for s in samples if s['state'].get('name')=='running_'+name and start<=s['time']<=end]
 assert stable,f'No monitoring samples: {name}'
 violations=[];clocks={u:[] for u in expected}
 for s in stable:
  observed={u:set() for u in expected}
  if s.get('processes_exit')!=0 or s.get('gpu_exit')!=0:
   violations.append(dict(time=s['time'],reason='query error'));continue
  for row in csv.reader(s['processes'].splitlines()):
   if len(row)>=2 and row[1].strip() in observed:observed[row[1].strip()].add(int(row[0]))
  if observed!=expected:violations.append(dict(time=s['time'],observed={u:sorted(v) for u,v in observed.items()}))
  for row in csv.reader(s['gpu'].splitlines()):
   if len(row)>=6 and row[1].strip() in clocks:
    try:clocks[row[1].strip()].append(float(row[5]))
    except ValueError:pass
 gaps=[b['time']-a['time'] for a,b in zip(stable,stable[1:])]
 result[name]=dict(passed=not violations,samples=len(stable),max_sample_gap_seconds=max(gaps,default=None),expected_driver_pids={u:sorted(v) for u,v in expected.items()},violations=violations,sm_clock_mhz={u:dict(min=min(v),max=max(v),mean=statistics.mean(v)) for u,v in clocks.items() if v})
out=dict(passed=all(v['passed'] for v in result.values()),trials=result,limitation='Sampling cannot exclude external activity shorter than the interval.')
(run/'gpu_isolation_audit.json').write_text(json.dumps(out,indent=2))
summary=json.loads((run/'summary.json').read_text());summary['gpu_isolation_audit']=out
(run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(out,indent=2));assert out['passed']
