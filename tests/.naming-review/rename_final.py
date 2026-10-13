from pathlib import Path
exec(Path('tests/.naming-review/rename.py').read_text().split('# These symbols')[0])

local('litespec/training/setup.py','_build_target_loader',{
    'loader':'dataloader','steps':'steps_per_epoch','size':'dataset_size'})
local('litespec/training/recipes/dlite.py','required_target_layers',{'layers':'required_layer_ids'})
local('litespec/training/recipes/dlite.py','_select_packet_layers',{
    'union_ids':'available_layer_ids','requested_ids':'required_layer_ids','indices':'layer_indices'})
local('litespec/modeling/draft/dlite.py','_build_shared_context_kv',{'current_ctx':'current_chs_hidden'})
local('litespec/core/_dflash_common.py','_forward_draft_blocks',{'hidden_states':'target_context_hidden'})
local('tests/distributed_dlite_disaggregate_smoke.py','main',{'local_batch':'local_batch_size'})
local('tests/distributed_muon_smoke.py','run_case',{'fsdp':'fsdp_model'})
local('tests/benchmark_fsdp_muon.py','main',{'fsdp':'fsdp_model'})

renames={
    '_forward_draft_blocks':{'hidden_states':'target_context_hidden'},
    '_select_packet_layers':{'union_ids':'available_layer_ids','requested_ids':'required_layer_ids'},
}
for p in sources():
    def select(tree):
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                fn=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else None
                for kw in n.keywords:
                    if kw.arg in renames.get(fn,{}):
                        yield kw,kw.arg,renames[fn][kw.arg],False
    edit_ast(p,select)

print('Completed cross-function naming pass')
