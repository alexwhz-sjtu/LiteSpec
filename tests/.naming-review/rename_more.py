from pathlib import Path
exec(Path('tests/.naming-review/rename.py').read_text().split('# These symbols')[0])

replace_exact('tests/test_training_runner.py','resumed.resume','resumed.resume_state')

for fn in ('_aligned_target_hidden','_dflash_objective_chunk_terms','_dflash_unary_diagnostics',
           '_dflash_greedy_diagnostics','_dflash_teacher_terms'):
    local('litespec/core/_dflash_common.py',fn,{'batch_size':'bsz'})
for fn in ('_aligned_target_hidden','_dspark_objective_chunk_terms','_compute_dspark_loss'):
    local('litespec/core/dspark.py',fn,{'batch_size':'bsz'})
local('litespec/distributed/transport.py','DLiteBatchPacket.empty',{
    'n':'num_anchors','h':'hidden_size','c':'num_target_layers','k':'prediction_length'})
local('litespec/modeling/draft/dlite.py','DLiteDraftModel.__init__',{
    'h':'hidden_size','required':'required_field_names','missing':'missing_field_names'})
local('litespec/modeling/draft/dlite.py','_apply_chs_depth_embedding',{
    'h':'hidden_size','s_len':'num_chs_slots'})
local('litespec/modeling/draft/dlite.py','_fuse_target_hidden',{
    'n_blk':'num_blocks','current_ctx':'current_chs_hidden','ctx':'context_hidden'})
local('litespec/modeling/draft/dlite.py','build_target_layer_ids',{
    'cap':'num_interior_layers','picked':'selected_layer_ids','result':'target_layer_ids'})
local('litespec/modeling/draft/dlite.py','sample_block_tokens',{'batch_size':'bsz'})
local('litespec/modeling/draft/dflash.py','validate_dflash_mla_config',{
    'required':'required_field_names','missing':'missing_field_names'})
local('litespec/modeling/draft/dflash2.py','_convolve',{'batch_size':'bsz','sequence_length':'seq_len'})
local('litespec/modeling/draft/dflash2.py','_propose_draft_tokens',{'hidden':'draft_prediction_hidden'})
local('litespec/modeling/draft/dspark.py','sample_block_tokens',{'batch_size':'bsz'})
local('litespec/data/hidden_cache.py','RegenFullCollator.__call__',{'sequence_length':'seq_len'})
local('litespec/layers/lm_head.py','forward',{'full':'full_logits'})
local('litespec/layers/ring/utils.py','send_recv',{'res':'received_tensor'})
local('litespec/training/fsdp_layout.py','build_fsdp_layouts',{'info':'shard_info'})
local('litespec/training/muon.py','_orthogonalize_and_return',{
    'full':'full_matrices','result':'orthogonalized_matrices'})
local('litespec/training/runner.py','iter_microbatch_slices',{
    'size':'local_batch_size','micro':'micro_batch_size'})
local('litespec/training/runner.py','run_target_phase',{'loader':'dataloader'})
local('litespec/training/runner.py','log_window',{'reduced':'reduced_metrics'})
local('scripts/config/training_launcher.py','load_yaml_config',{'missing':'missing_field_names'})
local('scripts/data/regenerate_train_data.py','validate_record',{
    'required':'required_field_names','missing':'missing_field_names'})
local('scripts/data/prepare_hidden_states.py','add_bytes',{'info':'member_info'})
local('scripts/data/prepare_hidden_states.py','sample_payload',{
    'hidden':'target_hidden_states','missing':'missing_layer_ids'})
local('scripts/data/prepare_hidden_states.py','_path_component',{'safe':'safe_component'})
local('scripts/data/prepare_hidden_states.py','run_generation',{'raw':'raw_dataset'})

for p in sources():
    def select(tree):
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_target_phase':
                for kw in n.keywords:
                    if kw.arg=='loader':yield kw,'loader','dataloader',False
    edit_ast(p,select)

print('Applied follow-up naming review')
