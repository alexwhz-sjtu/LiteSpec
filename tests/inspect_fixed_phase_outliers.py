import json,statistics
from pathlib import Path
p=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300')
s=json.loads((p/'summary.json').read_text());details={}
for name,m in s['measurements'].items():
 rows=[[json.loads(x) for x in (p/name/'telemetry'/f'phases_rank_{r}.jsonl').read_text().splitlines()] for r in [0,1]]
 selected=[max(pair,key=lambda r:r['rank_step_ms']) for pair in zip(*rows) if pair[0]['step'] in s['common_measurement_steps']]
 top=sorted(selected,key=lambda r:-r['stream_timeline_ms']['load_data'])[:5]
 details[name]=dict(median_exposed_data_wait_ms=statistics.median(r['stream_timeline_ms']['load_data'] for r in selected),mean_exposed_data_wait_ms=m['stages']['load_data']['mean_ms'],top_data_wait=[dict(step=r['step'],data_ms=r['stream_timeline_ms']['load_data'],wall_ms=r['rank_step_ms']) for r in top])
 print(name,{k:v for k,v in m.items() if k!='stages'},details[name])
s['data_wait_distribution']=details
(p/'summary.json').write_text(json.dumps(s,indent=2))
