"""Resolve using byte-verified local copies of unchanged locked URL wheels."""
from pathlib import Path
import hashlib,tomllib
root=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp')
dst=root/'tests/gdn_context_memory_20261017/resolution'
text=(root/'pyproject.toml').read_text()
for p in tomllib.loads((root/'uv.lock').read_text())['package']:
 if p['name']=='sgl-kernel' and 'url' in p['source']:
  url=p['source']['url'];wheel=root/'tests/gdn_context_memory_20261017/wheels'/url.split('/')[-1].replace('%2B','+')
  assert hashlib.sha256(wheel.read_bytes()).hexdigest()==p['wheels'][0]['hash'].split(':')[1]
  text=text.replace('url = "'+url+'"','path = "'+str(wheel)+'"')
(dst/'pyproject.toml').write_text(text)
(dst/'uv.lock').write_bytes((root/'uv.lock').read_bytes())
