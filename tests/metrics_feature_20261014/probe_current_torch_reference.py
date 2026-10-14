import json,os,subprocess
from pathlib import Path
import yaml
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');os.chdir(root)
parent=root/'tests/optimization_handoff_runs/20261013_cuda13_203502'
cache=parent/'newhost_2gpu/logging_metrics_warmup200_steps300'
run=root/'tests/metrics_feature_20261014/torch_reference_shared_cache3'
config=yaml.safe_load((cache/'all_a.yaml').read_text());config['launcher']['output_dir']=str(run)
path=root/'tests/metrics_feature_20261014/torch_reference_shared_cache3.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='1',CUDA_HOME='/usr/local/cuda-13.0',CUDA_VISIBLE_DEVICES='0,1',NCCL_DEBUG='WARN',LITESPEC_LOSS_BACKEND='torch',TORCHINDUCTOR_COMPILE_THREADS='1',TRITON_CACHE_ROOT=str(cache/'triton-cache'),TORCHINDUCTOR_CACHE_ROOT=str(cache/'inductor-cache'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor-cache'),TVM_FFI_CACHE_ROOT=str(cache/'tvm-cache'))
env['PATH']=str(root/'.venv/bin')+':/usr/local/cuda-13.0/bin:/usr/local/bin:'+env['PATH']
cmd=[str(root/'.venv/bin/torchrun'),'--standalone','--nproc_per_node=2','tests/benchmark_training_metrics.py','--config',str(path),'--steps','3','--warmup-steps','200','--metrics-every-step','--verified-cache-record',str(parent/'cache_pipeline/full_validation.json')]
with (run.parent/'torch_reference_shared_cache3.log').open('x') as log:
 subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,start_new_session=True)
a=[json.loads(x) for x in (run/'telemetry/steps_rank_0.jsonl').read_text().splitlines()]
b=[json.loads(x) for x in (cache/'all_a/telemetry/steps_rank_0.jsonl').read_text().splitlines()][:3]
result=dict(steps=[dict(step=x['step'],torch_norm=x['grad_norm'],auto_norm=y['grad_norm'],loss_abs=abs(x['global_weighted_loss']-y['global_weighted_loss'])) for x,y in zip(a,b)])
(run.parent/'torch_reference_shared_cache3.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
