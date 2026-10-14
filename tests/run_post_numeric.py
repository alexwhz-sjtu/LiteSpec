import json,subprocess,os,time
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec');os.chdir(root)
p=root/'tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014'
while True:
 s=json.loads((p/'state.json').read_text())
 if s['name']=='failed':raise RuntimeError(str(s))
 if s['name']=='complete':break
 time.sleep(3)
env=dict(os.environ,PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES='0',TORCHINDUCTOR_COMPILE_THREADS='1',TORCHINDUCTOR_CACHE_DIR=str(p/'boundary-inductor'),TRITON_CACHE_DIR=str(p/'boundary-triton'))
for script in ['compare_numeric_snapshots.py','probe_ce_boundary.py','probe_muon_cudagraph.py']:
 with (p/(script+'.log')).open('x') as log:
  subprocess.run([str(root/'.venv/bin/python'),'tests/'+script],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,start_new_session=True)
(p/'post_complete.json').write_text(json.dumps(dict(complete=True,time=time.time())))
