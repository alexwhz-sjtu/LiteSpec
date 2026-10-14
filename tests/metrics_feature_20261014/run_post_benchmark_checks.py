import json,os,subprocess,time
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');os.chdir(root)
run=root/'tests/metrics_feature_20261014';state=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/logging_metrics_warmup200_steps300/state.json'
while True:
 s=json.loads(state.read_text())
 if s['name']=='failed':raise RuntimeError(str(s))
 if s['name']=='complete':break
 time.sleep(3)
env=dict(os.environ,PYTHONPATH=str(root),OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(run/'inductor-cache'))
commands=[('scheduling_checks',[str(root/'.venv/bin/python'),'-m','pytest','-q','tests/test_metrics_logging_only.py','-k','runner_','tests/test_public_config.py']),('legacy_confidence',[str(root/'.venv/bin/python'),'tests/metrics_feature_20261014/probe_legacy_confidence_targets.py'])]
for name,cmd in commands:
 with (run/(name+'.log')).open('x') as log:
  subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,start_new_session=True)
(run/'post_checks_complete.json').write_text(json.dumps(dict(complete=True,time=time.time())))
