"""Task-owned 20-step training and two-step checkpoint continuation."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp')
TASK=ROOT/'tests/gdn_context_memory_20261017'
OUTPUT=ROOT/'output/composable_gdn_smoke_20261017'
ENV={**os.environ,'FLA_TILELANG':'1','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}


def save(value):
    temp=TASK/'training_status.tmp'
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(TASK/'training_status.json')


def run(name,extra):
    command=['bash','scripts/composable/run_training_composable.sh','scripts/config/composable_gdn.yaml',*extra]
    start=time.monotonic();peak=[0]*8;samples=[]
    with (TASK/(name+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state={'stage':name,'state':'running','pid':process.pid,'command':command,'started_at_utc':datetime.now(timezone.utc).isoformat()}
        save(state)
        while process.poll() is None:
            raw=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
            if raw.returncode==0:
                for row in raw.stdout.splitlines():
                    i,m=map(int,row.split(','));peak[i]=max(peak[i],m)
                if len(samples)<2000:samples.append({'seconds':round(time.monotonic()-start,1),'gpu_memory_used_mib':list(peak)})
            state.update(elapsed_seconds=round(time.monotonic()-start,2),peak_gpu_memory_used_mib=peak)
            save(state);time.sleep(2)
    state.update(state='complete' if process.returncode==0 else 'failed',returncode=process.returncode,elapsed_seconds=round(time.monotonic()-start,2))
    (TASK/(name+'_result.json')).write_text(json.dumps({**state,'samples':samples},indent=2)+'\n')
    save(state)
    if process.returncode:raise RuntimeError(f'{name} failed; see {TASK/(name+".log")}')
    return state


if __name__=='__main__':
    assert not OUTPUT.exists(),'Refuse to overwrite an existing training output'
    first=run('train_20',[])
    checkpoints=list(OUTPUT.glob('*step_20'))
    checkpoints=[p for p in checkpoints if (p/'training_state.pt').is_file()]
    assert len(checkpoints)==1,checkpoints
    checkpoint=checkpoints[0]
    second=run('resume_2',['--resume-from',str(checkpoint),'--max-steps','22','--save-interval','22'])
    summary={'state':'complete','first':first,'resume':second,'resume_checkpoint':str(checkpoint)}
    (TASK/'training_result.json').write_text(json.dumps(summary,indent=2)+'\n')
    save(summary)
