import hashlib,json,os,subprocess,threading,time,statistics
from pathlib import Path
import yaml
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');os.chdir(root)
parent=root/'tests/optimization_handoff_runs/20261013_cuda13_203502'
run=parent/'newhost_2gpu/logging_metrics_warmup200_steps300';run.mkdir(exist_ok=False)
env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='1',CUDA_HOME='/usr/local/cuda-13.0',CUDA_VISIBLE_DEVICES='0,1',NCCL_DEBUG='WARN',TORCHINDUCTOR_COMPILE_THREADS='1',TRITON_CACHE_ROOT=str(run/'triton-cache'),TORCHINDUCTOR_CACHE_ROOT=str(run/'inductor-cache'),TORCHINDUCTOR_CACHE_DIR=str(run/'inductor-cache'),TVM_FFI_CACHE_ROOT=str(run/'tvm-cache'))
env['PATH']=str(root/'.venv/bin')+':/usr/local/cuda-13.0/bin:/usr/local/bin:'+env['PATH']
stop=threading.Event();state={}
def report(name,**kwargs):
 state.clear();state.update(name=name,time=time.time(),controller_pid=os.getpid(),**kwargs)
 p=run/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(run/'state.json');print(json.dumps(state),flush=True)
def monitor():
 with (run/'gpu_monitor.jsonl').open('x',buffering=1) as log:
  while not stop.is_set():
   entry=dict(time=time.time(),state=dict(state))
   for key,query in [('gpu','--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.current.sm'),('processes','--query-compute-apps=pid,gpu_uuid,used_memory')]:
    try:
     q=subprocess.run(['nvidia-smi',query,'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
     entry[key]=q.stdout;entry[key+'_error']=q.stderr;entry[key+'_exit']=q.returncode
    except subprocess.TimeoutExpired:entry[key+'_error']='timeout'
   log.write(json.dumps(entry)+'\n');stop.wait(1)
try:
 regression=root/'tests/metrics_feature_20261014/cuda_loss_checks.log'
 assert '55 passed' in regression.read_text()
 assert 'return _run(_ce_l1_probs' not in (root/'litespec/core/losses.py').read_text()
 (run/'losses_fixed.py').write_text((root/'litespec/core/losses.py').read_text())
 watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
 names=['all_a','logging_b','logging_b_repeat','all_a_repeat']
 for name in names:
  backend='auto'
  idle=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
  if idle.strip():raise RuntimeError('GPUs busy: '+idle)
  config=yaml.safe_load((parent/'newhost_2gpu/smoke_auto.yaml').read_text());config['launcher']['output_dir']=str(run/name)
  path=run/(name+'.yaml');path.write_text(yaml.safe_dump(config,sort_keys=False))
  cmd=[str(root/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=2','tests/benchmark_training_metrics.py','--config',str(path),'--steps','300','--warmup-steps','200','--final-masters-hash','--verified-cache-record',str(parent/'cache_pipeline/full_validation.json')]
  if name.startswith('all'):cmd+=['--metrics-every-step']
  with (run/(name+'.log')).open('x') as log:
   p=subprocess.Popen(cmd,env=dict(env,LITESPEC_LOSS_BACKEND=backend),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   report('running_'+name,worker_pid=p.pid,command=cmd)
   code=p.wait()
  if code:raise RuntimeError(f'{name} exited {code}')
  report('completed_'+name)
 all_rows={name:[[json.loads(line) for line in (run/name/'telemetry'/f'steps_rank_{rank}.jsonl').read_text().splitlines()] for rank in [0,1]] for name in names}
 comparisons={}
 for rank in [0,1]:
  initial=[json.loads((run/name/'telemetry'/f'init_rank_{rank}.json').read_text()) for name in names]
  final=[json.loads((run/name/'telemetry'/f'complete_rank_{rank}.json').read_text()) for name in names]
  assert len({r['initial_master_sha256'] for r in initial})==1
  assert all(r['optimizer_steps']==300 for r in final)
  baseline=all_rows[names[0]][rank]
  for name in names[1:]:
   assert [(r['input_sha256'],r['anchors_sha256']) for r in baseline]==[(r['input_sha256'],r['anchors_sha256']) for r in all_rows[name][rank]]
  comparisons[str(rank)]=dict(initial_equal=True,inputs_and_anchors_equal=True,final_master_sha256=[r['final_master_sha256'] for r in final],final_masters_exactly_equal=len({r['final_master_sha256'] for r in final})==1,max_loss_abs=max(abs(a['global_weighted_loss']-b['global_weighted_loss']) for name in names[1:] for a,b in zip(baseline,all_rows[name][rank])),max_grad_norm_abs=max(abs(a['grad_norm']-b['grad_norm']) for name in names[1:] for a,b in zip(baseline,all_rows[name][rank])))
 for name in names:
  for rank in [0,1]:
   for row in all_rows[name][rank]:
    expected=(name.startswith('all') or row['step'] % 50==0)
    actual=row['detailed_metric_calls']
    assert actual==({'block_accepted_length_terms':1,'expected_accept_len_terms':1} if expected else {}), (name,rank,row['step'],actual)
 common=set.intersection(*[{r['step'] for r in all_rows[name][0] if not r['warmup'] and not r['profiled'] and not r['logging_step'] and not r['new_graphs_on_any_rank']} for name in names])
 measurements={}
 for name in names:
  rows=[r for r in all_rows[name][0] if r['step'] in common];duration=sum(r['slowest_rank_step_ms'] for r in rows)/1000
  phase_rows=[[json.loads(x) for x in (run/name/'telemetry'/f'phases_rank_{rank}.jsonl').read_text().splitlines()] for rank in [0,1]]
  selected=[max([a,b],key=lambda r:r['rank_step_ms']) for a,b in zip(*phase_rows) if a['step'] in common]
  stages=['load_data','prepare','forward','loss','backward','update'];totals={s:sum(r['stream_timeline_ms'][s] for r in selected) for s in stages};total=sum(totals.values())
  measurements[name]=dict(samples=len(rows),median_ms=statistics.median(r['slowest_rank_step_ms'] for r in rows),mean_ms=statistics.mean(r['slowest_rank_step_ms'] for r in rows),input_tokens=sum(r['global_input_tokens'] for r in rows),duration_seconds=duration,input_tokens_per_s=sum(r['global_input_tokens'] for r in rows)/duration,supervised_tokens_per_s=sum(r['global_supervised_tokens'] for r in rows)/duration,peak_allocated_gib=max(r['peak_allocated_gib'] for rank in [0,1] for r in all_rows[name][rank]),peak_reserved_gib=max(r['peak_reserved_gib'] for rank in [0,1] for r in all_rows[name][rank]),stages={s:dict(mean_ms=totals[s]/len(selected),share_percent=100*totals[s]/total) for s in stages})
 pooled={}
 for backend in ['all','logging']:
  subset=[v for n,v in measurements.items() if n.startswith(backend)]
  pooled[backend]=sum(v['input_tokens'] for v in subset)/sum(v['duration_seconds'] for v in subset)
 result=dict(loss_backend='auto',warmup_steps=200,steps_per_trial=300,order=names,common_measurement_steps=sorted(common),comparisons=comparisons,measurements=measurements,pooled_input_tokens_per_s=pooled,pooled_speedup_percent=100*(pooled['logging']/pooled['all']-1))
 (run/'summary.json').write_text(json.dumps(result,indent=2));report('complete',comparisons=comparisons,pooled=pooled)
except BaseException as e:report('failed',error=repr(e));raise
finally:
 stop.set()
 if 'watcher' in globals():watcher.join(timeout=15)
