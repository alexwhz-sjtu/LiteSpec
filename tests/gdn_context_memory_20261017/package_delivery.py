"""Package only implementation, reproducible checks, and small validation evidence."""
from pathlib import Path
import difflib,hashlib,importlib.metadata,json,tarfile
root=Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp');task=root/'tests/gdn_context_memory_20261017'
modified=[str(p.relative_to(task/'before')) for p in (task/'before').rglob('*') if p.is_file()]
added=['litespec/modeling/draft/composable/backbone/gdn.py','litespec/modeling/draft/composable/backbone/gdn_events.py','configs/composable_gdn.json','scripts/config/composable_gdn.yaml']
files=sorted(set(modified+added));chunks=[]
for rel in files:
 old=task/'before'/rel;new=root/rel
 chunks.extend(difflib.unified_diff(old.read_text().splitlines(True) if old.exists() else [],new.read_text().splitlines(True),fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel))
(task/'implementation.patch').write_text(''.join(chunks))
validation={
 'remote_host':'Inf_H800_2','repository':str(root),
 'production_files':{rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in files},
 'dependencies':{name:importlib.metadata.version(name) for name in ['torch','transformers','flash-linear-attention','tilelang','triton']},
 'tests':{name:(task/log).read_text().strip().splitlines()[-1] for name,log in {'cpu_semantics':'pytest_cpu_final.log','gpu_numerics':'pytest_gpu.log','gpu_cache':'pytest_gpu_cache.log','legacy_mha_mla':'pytest_legacy_2.log'}.items()},
 'distributed':json.loads((task/'distributed_result.json').read_text()),
 'checkpoint':json.loads((task/'checkpoint_verification.json').read_text()),
 'training':json.loads((task/'training_verification.json').read_text()),
 'lock_check':(task/'lock_check.log').read_text(),
 'dependency_check':(task/'dependency_check.log').read_text(),
}
(task/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
extra=['test_gdn.py','test_gpu.py','check_distributed.py','run_smoke.py','verify_checkpoint.py','verify_training_state.py','validation.json','implementation.patch','training_result.json','training_verification.json','checkpoint_verification.json','distributed_result.json','core_versions_before.json','core_versions_after.json','pytest_cpu_final.log','pytest_gpu.log','pytest_gpu_cache.log','pytest_legacy_2.log','train_20.log','resume_2.log','lock_check.log','dependency_check.log','lock_resolution.log','lock_sync_dryrun.log']
with tarfile.open(task/'gdn_implementation_delivery.tar.gz','w:gz') as archive:
 for rel in files:archive.add(root/rel,arcname=rel)
 for name in extra:archive.add(task/name,arcname='tests/gdn_context_memory_20261017/'+name)
print(json.dumps({'files':len(files),'bundle_bytes':(task/'gdn_implementation_delivery.tar.gz').stat().st_size,'validation':str(task/'validation.json')},indent=2))
