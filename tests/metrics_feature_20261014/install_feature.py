import ast,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/diancpfs/user/wanghanzhen/LiteSpec')
run=root/'tests/metrics_feature_20261014'
files=sorted((run/'source/litespec').rglob('*.py'))
relative=[str(p.relative_to(run/'source')) for p in files]
assert subprocess.run(['git','diff','--quiet','--',*relative],cwd=root).returncode==0,'Production files changed since source inspection'
record={}
for source,relative_path in zip(files,relative,strict=True):
 ast.parse(source.read_text())
 target=root/relative_path;backup=run/'before'/relative_path
 backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
 record[relative_path]=dict(before_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),after_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
for source,relative_path in zip(files,relative,strict=True):shutil.copy2(source,root/relative_path)
(run/'installed_files.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
