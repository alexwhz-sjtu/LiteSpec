"""Retarget verified local wheel sources to their identical existing locked URLs."""
import re,tomllib
from pathlib import Path
root=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp');task=root/'tests/gdn_context_memory_20261017'
old=(task/'before/uv.lock').read_text();new=(task/'resolution/uv.lock').read_text()
def records(text):
 blocks=text.split('[[package]]')
 return [(tomllib.loads('[[package]]'+b)['package'][0],b) for b in blocks[1:]]
originals={p['source']['url'].split('/')[-1].replace('%2B','+'):(p,b) for p,b in records(old) if p['name']=='sgl-kernel' and 'url' in p['source']}
for p,block in records(new):
 if p['name']=='sgl-kernel' and 'path' in p['source']:
  filename=Path(p['source']['path']).name;original,oldblock=originals[filename]
  assert p['version']==original['version']
  assert not p.get('dependencies') and not original.get('dependencies')
  new=new.replace('[[package]]'+block,'[[package]]'+oldblock)
for filename,(p,_) in originals.items():
 pattern=r'path = "[^"\n]*'+re.escape(filename)+r'"'
 new=re.sub(pattern,'url = "'+p['source']['url']+'"',new)
assert 'tests/gdn_context_memory_20261017' not in new
previous={}
for p in tomllib.loads(old)['package']:previous.setdefault(p['name'],set()).add(p.get('version'))
current={}
for p in tomllib.loads(new)['package']:current.setdefault(p['name'],set()).add(p.get('version'))
for name in ['torch','torchvision','torchaudio','transformers','flash-linear-attention','triton','sglang','causal-conv1d']:
 assert previous[name]==current[name],(name,previous[name],current[name])
assert current['tilelang']=={'0.1.15'}
print({k:(sorted(previous.get(k,set())),sorted(v)) for k,v in current.items() if previous.get(k)!=v})
(task/'uv.lock.candidate').write_text(new)
(root/'uv.lock').write_text(new)
