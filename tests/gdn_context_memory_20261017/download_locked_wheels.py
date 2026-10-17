"""Fetch exact existing locked artifacts with parallel verified HTTP ranges."""
import concurrent.futures, hashlib, os, time, tomllib
from pathlib import Path
import requests
root=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp')
dest=root/'tests/gdn_context_memory_20261017/wheels';dest.mkdir(exist_ok=True)
entries=[p for p in tomllib.loads((root/'uv.lock').read_text())['package'] if p['name']=='sgl-kernel' and 'url' in p['source']]
def fetch_package(p):
 url=p['source']['url']; target=dest/url.split('/')[-1].replace('%2B','+')
 digest=p['wheels'][0]['hash'].split(':')[1]
 headers=requests.head(url,allow_redirects=True,timeout=30).headers
 size=int(headers['Content-Length']);step=2**21
 temp=target.with_suffix('.part'); fd=os.open(temp,os.O_RDWR|os.O_CREAT,0o644);os.ftruncate(fd,size)
 def part(start):
  end=min(start+step,size)-1
  for attempt in range(4):
   try:
    response=requests.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=180)
    response.raise_for_status();data=response.content
    assert response.status_code==206 and len(data)==end-start+1,(response.status_code,len(data),response.headers)
    assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    os.pwrite(fd,data,start);return len(data)
   except Exception:
    if attempt==3:raise
    time.sleep(attempt+1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
  done=0
  for length in pool.map(part,range(0,size,step)):
   done+=length
   if done%(step*10)==0:print(target.name,done,size,flush=True)
 os.close(fd)
 actual=hashlib.sha256(temp.read_bytes()).hexdigest()
 assert actual==digest,(actual,digest)
 temp.rename(target);print('VERIFIED',target,actual,flush=True)
for entry in entries:fetch_package(entry)
