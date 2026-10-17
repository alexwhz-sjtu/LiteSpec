"""Apply a reference/docstring-only payload after checking the observed baseline."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

REPO = Path('/data/wanghanzhen/projects/SpecDecoding/LiteSpec_exp')
TASK = REPO / 'tests/qwen35_gated_deltanet_20261015'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
baseline = json.loads((TASK / 'baseline.json').read_text())
staging = TASK / 'staging'
staging.mkdir(exist_ok=True)
with tarfile.open(TASK / 'payload.tar.gz') as archive:
    for member in archive.getmembers():
        rel = Path(member.name)
        if member.isdir():
            continue
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())

conflicts = []
for rel, old_hash in baseline.items():
    path = REPO / rel
    if path.exists():
        # Identical archives from an earlier task can safely be reused.
        if sha(path) != old_hash and sha(path) != sha(staging / rel):
            conflicts.append(rel)
    elif old_hash is not None:
        conflicts.append(rel)
assert not conflicts, f'Concurrent edits; no files written: {conflicts}'

protected = {}
for pattern in ['litespec/**/*.py', 'refrence/**/*', 'configs/**/*', 'scripts/config/**/*', 'pyproject.toml', 'uv.lock']:
    for path in REPO.glob(pattern):
        if path.is_file() and str(path.relative_to(REPO)) not in baseline:
            protected[str(path.relative_to(REPO))] = sha(path)
(TASK / 'protected-before.json').write_text(json.dumps(protected, indent=2) + '\n')

for rel in baseline:
    source, target = staging / rel, REPO / rel
    if target.exists() and sha(target) == sha(source):
        continue
    if target.exists():
        backup = TASK / 'before' / rel
        assert not backup.exists(), f'Existing backup: {rel}'
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(target.read_bytes())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())

for rel, expected in protected.items():
    assert sha(REPO / rel) == expected, f'Unrelated file changed: {rel}'
for rel in baseline:
    assert sha(REPO / rel) == sha(staging / rel), rel

production = [REPO / rel for rel in baseline if rel.startswith('litespec/')]
for path in production:
    tree = ast.parse(path.read_text())
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str)

manifest = json.loads((REPO / 'refrence/manifest.json').read_text())
new_entries = [e for e in manifest['entries'] if e.get('classification') == 'official_model_implementation']
fragments_verified = 0
for entry in new_entries:
    excerpt = REPO / 'refrence' / entry['path']
    source = Path(entry['full_source_path'])
    assert sha(source) == entry['full_source_sha256']
    assert sha(excerpt) == entry['excerpt_sha256']
    ast.parse(excerpt.read_text())
    lines = source.read_text().splitlines(keepends=True)
    for fragment in entry['fragments']:
        body = ''.join(lines[fragment['start_line'] - 1:fragment['end_line']])
        assert hashlib.sha256(body.encode()).hexdigest() == fragment['sha256']
        assert body in excerpt.read_text()
        fragments_verified += 1
for source in manifest['qwen35_official_sources']:
    assert sha(REPO / source['archive_path']) == source['sha256']
scaffold = json.loads((REPO / 'refrence/scaffold-manifest.json').read_text())
for entry in scaffold['files']:
    if entry['path'] in baseline:
        assert sha(REPO / entry['path']) == entry['sha256']
for path in (REPO / 'refrence').rglob('*.json'):
    json.loads(path.read_text())

result = {
    'scope': 'reference_and_docstring_only',
    'workspace': str(REPO),
    'files_applied_or_verified': len(baseline),
    'production_docstring_only_files': len(production),
    'reference_excerpts_verified': len(new_entries),
    'exact_fragments_verified': fragments_verified,
    'pinned_source_archives_verified': len(manifest['qwen35_official_sources']),
    'unrelated_files_unchanged': len(protected),
    'runtime_training_or_gpu_tests_run': False,
    'files': sorted(baseline),
}
(TASK / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2))
