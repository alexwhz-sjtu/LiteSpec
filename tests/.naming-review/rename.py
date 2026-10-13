"""One-off, scope-aware naming migration; artifacts remain under tests."""
import ast
from pathlib import Path
import re


def sources():
    return [
        p for root in ('litespec', 'scripts', 'evaluation', 'tests')
        for p in Path(root).rglob('*.py')
        if not any(part.startswith('.') or part == '__pycache__' for part in p.parts)
    ]


def replace_exact(path, old, new):
    p = Path(path)
    text = p.read_text(encoding='utf-8-sig')
    assert old in text, (path, old)
    p.write_text(text.replace(old, new))


def edit_ast(path, select):
    p = Path(path)
    original = p.read_bytes()
    bom = b'\xef\xbb\xbf' if original.startswith(b'\xef\xbb\xbf') else b''
    data = original[len(bom):]
    tree = ast.parse(data.decode())
    lines = data.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = {}
    for node, old, new, attribute in select(tree):
        if attribute:
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            start = end - len(old.encode())
        else:
            start = offsets[node.lineno - 1] + node.col_offset
            end = start + len(old.encode())
        assert data[start:end].decode() == old, (path, node.lineno, old, data[start:end])
        assert (start,end) not in edits or edits[start,end] == new.encode()
        edits[start,end] = new.encode()
    for (start, end), new in sorted(edits.items(), reverse=True):
        data = data[:start] + new + data[end:]
    if bom + data != original:
        ast.parse(data.decode())
        p.write_bytes(bom + data)


def local(path, function, mapping):
    """Rename names/arguments only, preserving attributes, keys and call keywords."""
    def select(tree):
        if '.' in function:
            owner, method = function.split('.')
            scopes = [n for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == owner
                      for n in c.body if isinstance(n, ast.FunctionDef) and n.name == method]
        else:
            scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
        assert scopes, (path, function)
        for scope in scopes:
            bound = {n.id for n in ast.walk(scope) if isinstance(n, ast.Name)}
            bound |= {n.arg for n in ast.walk(scope) if isinstance(n, ast.arg)}
            for old, new in mapping.items():
                assert old in bound, (path, function, old)
                assert new not in bound, (path, function, new, 'already bound')
            for n in ast.walk(scope):
                old = n.id if isinstance(n, ast.Name) else n.arg if isinstance(n, ast.arg) else None
                if old in mapping:
                    yield n, old, mapping[old], False
    edit_ast(path, select)


# These symbols uniquely identify the intended algorithms, including __all__
# entries and documentation references. Ordinary variables are handled by AST.
symbols = {
    'DraftBatchPacket': 'DLiteBatchPacket',
    'DraftPacketSpec': 'DLitePacketSpec',
    'FamilyBatchPacket': 'DFlashFamilyBatchPacket',
    'FamilyPacketSpec': 'DFlashFamilyPacketSpec',
    'FamilyTrainingAdapter': 'DFlashFamilyTrainingAdapter',
    'FAMILY_ALGORITHMS': 'DFLASH_FAMILY_ALGORITHMS',
}
for name in ('add_family_args', 'validate_family_args', 'parse_family_args',
             'build_family_config', 'build_family_draft', 'load_family_draft',
             'build_family_components', 'build_family_dataloader',
             'prepare_family_features', 'family_training_metadata'):
    symbols[name] = name.replace('family', 'dflash_family')
paths = sources() + list(Path('.').glob('*.md')) + list(Path('docs').rglob('*.md')) + list(Path('scripts').rglob('*.md'))
pattern = re.compile(r'\b(' + '|'.join(symbols) + r')\b')
for p in paths:
    original = p.read_bytes()
    data = pattern.sub(lambda m: symbols[m[0]], original.decode('utf-8')).encode()
    if data != original:
        p.write_bytes(data)

# Runtime session attributes are not model parameters or persisted fields.
for p in sources():
    def select(tree):
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'session':
                if n.attr in ('loader', 'resume', 'fsdp'):
                    yield n, n.attr, {'loader':'dataloader','resume':'resume_state','fsdp':'fsdp_model'}[n.attr], True
    edit_ast(p, select)
for old,new in [('loader','dataloader'),('resume','resume_state'),('fsdp','fsdp_model')]:
    replace_exact('litespec/training/algorithms.py', f'    {old}:', f'    {new}:')

transport = 'litespec/distributed/transport.py'
for name in ('source', 'destination', 'packet'):
    replace_exact(transport, f'{name}: DLiteBatchPacket', f'{name}: DLiteBatchPacket | DFlashFamilyBatchPacket')
for fn in ('send','receive','wait_receive'):
    mapping={'meta':'packet_header'}
    if fn != 'send':mapping['metas']='packet_headers'
    local(transport,fn,mapping)
local(transport,'build_node_routes',{'producer_batch':'producer_batch_size','draft_batch':'draft_batch_size',
    'p0':'producer_start','p1':'producer_end','d0':'draft_start','d1':'draft_end'})
# Empty packet factories share the same shape vocabulary.
local(transport,'empty',{'b':'bsz','length':'seq_len'})

common='litespec/training/recipes/_dflash_common.py'
local(common,'prepare_dflash_family_features',{'hidden':'target_hidden_states','context':'target_context_hidden','last':'target_last_hidden'})
local(common,'_target_packet',{'output':'target_output','hidden':'target_hidden_states','context':'target_context_hidden','last':'target_last_hidden'})
local(common,'compute_step',{'context_hidden':'target_context_hidden'})
local(common,'_packet_spec',{'local_batch':'local_batch_size'})
local(common,'selector_alpha',{'target':'target_selector_alpha','warmup':'warmup_steps','ramp':'ramp_steps'})
dlite='litespec/training/recipes/dlite.py'
local(dlite,'_produce_packet',{'output':'target_output','hidden':'target_hidden_states','history':'target_history_hidden',
    'prediction':'target_prediction_hidden','final_layer':'final_layer_id','final_hidden':'target_last_hidden'})
local(dlite,'_prediction_hidden',{'final_hidden':'target_last_hidden'})
local(dlite,'_packet_spec',{'local_batch':'local_batch_size'})
local(dlite,'target_proxies',{'capture':'capture_layer_ids'})
for fn in ('_supervised_step','_distillation_step','_transition_step','_packet_step'):
    local(dlite,fn,{'fsdp':'fsdp_model'})
for fn in ('_supervised_step','_distillation_step','_transition_step'):
    local(dlite,fn,{'hidden_states':'target_hidden_states'})
local('litespec/training/setup.py','_build_target_loader',{'local_batch':'local_batch_size'})
local('litespec/training/setup.py','_build_processed_dataset',{'raw':'raw_dataset'})
local('litespec/training/checkpoint.py','load_distributed_training_state',{'required':'required_state_path'})
local('litespec/training/config.py','validate_common_args',{'required':'required_arg_names','missing':'missing_arg_names'})
local('scripts/config/training_launcher.py','validate_common_settings',{'required':'required_arg_names','missing':'missing_arg_names'})
local('litespec/modeling/config_utils.py','_qwen35_text_dict_to_qwen3_config',{'required':'required_field_names','missing':'missing_field_names'})
local('litespec/data/hidden_cache.py','load_hidden_cache_manifest',{'stored':'stored_layer_ids','missing':'missing_layer_ids','needed':'needed_layer_ids'})

core='litespec/core/_dflash_common.py'
local(core,'_create_noise_embed',{'n':'num_blocks','bs':'block_size'})
local(core,'_create_position_ids',{'n_blocks':'num_blocks','pos_ids':'position_ids'})
local(core,'_add_position_ratios',{'family':'metric_path'})
local(core,'_forward_draft_blocks',{'output_hidden':'draft_hidden_states'})
for path,fn in [(core,'_dflash_objective_chunk_terms'),(core,'_dflash_metric_chunk_terms'),
    (core,'_dflash_unary_diagnostics'),(core,'_dflash_greedy_diagnostics'),
    ('litespec/core/dflash2.py','_selector_chunk_terms'),('litespec/core/dflash2.py','_dflash2_selector_diagnostics'),
    ('litespec/core/dspark.py','_dspark_objective_chunk_terms')]:
    local(path,fn,{'hidden':'draft_block_hidden_states'})
for path,fn in [('litespec/core/dflash.py','forward'),('litespec/core/dspark.py','_compute_dspark_loss')]:
    local(path,fn,{'output_hidden':'draft_hidden_states','hidden_4d':'draft_block_hidden_states'})
local('litespec/core/dspark.py','forward',{'output_hidden':'draft_hidden_states'})
# Private helper keyword arguments must follow their renamed parameters.
private_calls={
    '_dflash_objective_chunk_terms':{'hidden':'draft_block_hidden_states'},
    '_dflash_metric_chunk_terms':{'hidden':'draft_block_hidden_states'},
    '_dflash_unary_diagnostics':{'hidden':'draft_block_hidden_states'},
    '_dflash_greedy_diagnostics':{'hidden':'draft_block_hidden_states'},
    '_selector_chunk_terms':{'hidden':'draft_block_hidden_states'},
    '_dflash2_selector_diagnostics':{'hidden':'draft_block_hidden_states'},
    '_dspark_objective_chunk_terms':{'hidden':'draft_block_hidden_states'},
    '_compute_dspark_loss':{'output_hidden':'draft_hidden_states'},
    '_prediction_hidden':{'final_hidden':'target_last_hidden'},
}
for p in sources():
    def select(tree):
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                fn=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else None
                for kw in n.keywords:
                    if kw.arg in private_calls.get(fn,{}):
                        yield kw,kw.arg,private_calls[fn][kw.arg],False
    edit_ast(p,select)
local('litespec/core/block_training.py','gather_block_values',{'safe':'safe_indices'})
for fn in ('create_dflash_sdpa_mask','create_dflash_flex_mask'):
    local('litespec/core/attention_masks.py',fn,{'batch':'bsz','blocks':'num_blocks','context':'context_mask','draft':'draft_mask'})
local('litespec/core/attention_masks.py','create_dflash_flex_mask',{'safe':'safe_block_indices'})
local('evaluation/benchmark.py','flatten_specbench_turns',{'out':'expanded_turns','t':'turn'})
local('evaluation/benchmark.py','format_multifieldqa_en_mixup_prompt',{'ctx':'context','inp':'instruction'})

# Test names referring specifically to DFlash algorithms, not ratio groups.
for p in sources():
    if p.parts[0]!='tests':continue
    text=p.read_text()
    updated=re.sub(r'\btest_(real_|repository_)?family_',lambda m:m[0].replace('family_','dflash_family_'),text)
    if updated!=text:p.write_text(updated)
print('Applied scoped naming migration')
