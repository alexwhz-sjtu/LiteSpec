import json,os,subprocess,time
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');os.chdir(root)
parent=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu'
while True:
 state=json.loads((parent/'fixed_ce_warmup200_steps300/state.json').read_text())
 if state['name']=='failed':raise RuntimeError(str(state))
 if state['name']=='complete':break
 time.sleep(3)
env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
log_path=parent/'numeric_diagnosis_20261014/related_training_tests_gpu_available.log'
with log_path.open('x') as log:
 p=subprocess.Popen([str(root/'.venv/bin/python'),'-m','pytest','-q','tests/test_training_compute_reuse.py','tests/test_stage1_loss.py','tests/test_training_runner.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 code=p.wait()
result=dict(exit_code=code,log=str(log_path),time=time.time())
(parent/'numeric_diagnosis_20261014/final_related_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if code:raise RuntimeError('Related tests failed')
