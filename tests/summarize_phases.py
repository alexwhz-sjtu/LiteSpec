import json,sqlite3,statistics,collections
from pathlib import Path
run=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014')
stages=['load_data','prepare','forward','loss','backward','update']
summary={}
for backend in ['auto','torch']:
 rows=[[json.loads(x) for x in (run/backend/'telemetry'/f'phases_rank_{r}.jsonl').read_text().splitlines()] for r in [0,1]]
 selected=[]
 for a,b in zip(*rows,strict=True):
  assert a['step']==b['step']
  if any(r['profiled'] or r['logging_step'] or r['new_graphs'] for r in [a,b]):continue
  selected.append(max([a,b],key=lambda r:r['rank_step_ms']))
 totals={s:sum(r['stream_timeline_ms'][s] for r in selected) for s in stages};total=sum(totals.values())
 summary[backend]=dict(samples=len(selected),steps=[r['step'] for r in selected],mean_wall_ms=statistics.mean(r['rank_step_ms'] for r in selected),median_wall_ms=statistics.median(r['rank_step_ms'] for r in selected),stages={s:dict(mean_stream_ms=totals[s]/len(selected),share_percent=100*totals[s]/total,mean_host_ms=statistics.mean(r['host_ms'][s] for r in selected)) for s in stages},attributed_mean_ms=total/len(selected),instrumentation_mean_ms=statistics.mean(r['stream_timeline_ms']['instrumentation'] for r in selected),unattributed_sync_and_boundary_mean_ms=statistics.mean(r['rank_step_ms']-sum(r['stream_timeline_ms'].values()) for r in selected))
# A CUDA runtime call is associated to its exclusive CPU NVTX stage, then to kernels by process+correlation.
c=sqlite3.connect(run/'stable_240_242.sqlite');c.row_factory=sqlite3.Row
strings=dict(c.execute('select id,value from StringIds'))
ranges=collections.defaultdict(list)
for r in c.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null'):
 label=r['text'] or strings.get(r['textId'],'')
 if label.startswith('train/') and label[6:] in stages+['instrumentation']:
  ranges[r['globalTid']>>24].append((r['start'],r['end'],label[6:]))
api={}
for r in c.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME'):
 phase=next((s for a,b,s in ranges[r['globalTid']>>24] if a<=r['start']<=b), 'outside_stage')
 api[(r['globalTid']>>24,r['correlationId'])]=(phase,strings[r['nameId']],r['end']-r['start'])
kernels=[]
for r in c.execute('select start,end,deviceId,globalPid,correlationId,demangledName,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL'):
 phase=api.get((r['globalPid']>>24,r['correlationId']),('unattributed',None,None))[0]
 kernels.append(dict(start=r['start'],end=r['end'],device=r['deviceId'],stage=phase,name=strings[r['demangledName']],duration_us=(r['end']-r['start'])/1000,graph_node=r['graphNodeId']))
def stats(ks):
 total=sum(k['duration_us'] for k in ks)
 return dict(count=len(ks),summed_kernel_ms=total/1000,under10us_count=sum(k['duration_us']<10 for k in ks),under20us_count=sum(k['duration_us']<20 for k in ks),under10us_summed_percent=100*sum(k['duration_us'] for k in ks if k['duration_us']<10)/total if total else 0,graph_node_nonzero=sum(bool(k['graph_node']) for k in ks))
groups=collections.defaultdict(list)
for k in kernels:groups[(k['stage'],k['name'])].append(k)
trace=dict(kernel_stats=stats(kernels),by_device={str(d):stats([k for k in kernels if k['device']==d]) for d in {k['device'] for k in kernels}},by_stage={s:stats([k for k in kernels if k['stage']==s]) for s in sorted({k['stage'] for k in kernels})},top_launch_counts=[dict(stage=s,name=n,**stats(ks)) for (s,n),ks in sorted(groups.items(),key=lambda item:-len(item[1]))[:35]],top_gpu_work=[dict(stage=s,name=n,**stats(ks)) for (s,n),ks in sorted(groups.items(),key=lambda item:-sum(k['duration_us'] for k in item[1]))[:25]],graph_api_calls=[dict(name=name,count=count) for name,count in c.execute("select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like '%Graph%' group by s.value")])
result=dict(method='Exclusive CUDA-event default-stream timeline (includes queued work, dependency waits and host-induced idle), critical rank selected per step; NVTX CUDA correlation is separate kernel work accounting, not wall shares.',warmup_steps=200,measurement_steps=[201,300],phase_summary=summary,stable_trace_steps=[240,241,242],trace=trace)
(run/'phase_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
