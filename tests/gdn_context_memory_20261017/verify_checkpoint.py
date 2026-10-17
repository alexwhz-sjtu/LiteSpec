"""Cold-load the exported real model without importing LiteSpec; verify cache use."""
import copy, importlib, json, sys
from pathlib import Path
import torch
from transformers import AutoModel

path=Path(sys.argv[1]).resolve()
model,loading=AutoModel.from_pretrained(path,trust_remote_code=True,torch_dtype=torch.bfloat16,output_loading_info=True)
assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys']),loading
assert not any(name=='litespec' or name.startswith('litespec.') for name in sys.modules),'Standalone export imported production package'
assert all(torch.isfinite(p).all() for p in model.parameters())
contracts=importlib.import_module(model.__class__.__module__.rsplit('.',1)[0]+'.cs_composable_contracts')
model=model.cuda().eval();cfg=model.config;d=cfg.hidden_size;l=cfg.block_size
assert len(model.backbone.layers)==5
assert len({layer.position_embedding.weight.data_ptr() for layer in model.backbone.layers})==5
assert all(layer.position_embedding.weight.shape==(8,2560) for layer in model.backbone.layers)
torch.manual_seed(31)
c=torch.randn(1,9,len(model.target_layer_ids)*d,device='cuda',dtype=torch.bfloat16)
x=torch.randn(1,l,d,device='cuda',dtype=torch.bfloat16)
state=contracts.DraftState()
def run(start,end,cached=True):
 return model(contracts.ModelInputs(c[:,start:end],x,torch.tensor([[end]],device='cuda'),torch.ones(1,1,device='cuda',dtype=torch.bool),context_position_ids=torch.arange(start,end,device='cuda')[None],draft_state=state if cached else None,mode='features')).backbone_hidden_states
with torch.inference_mode():
 run(0,3);run(3,4);actual=run(4,8);expected=run(0,8,False)
 cache_relative=float((actual.float()-expected.float()).norm()/expected.float().norm())
 assert cache_relative<.035,cache_relative
 snapshot=copy.deepcopy(state);repeat=run(8,8)
 repeat_relative=float((repeat.float()-actual.float()).norm()/actual.float().norm())
 assert repeat_relative<.035,repeat_relative
 for i,layer in state.backbone_state.layers.items():
  assert layer.recurrent_state.dtype==torch.float32
  torch.testing.assert_close(layer.recurrent_state,snapshot.backbone_state.layers[i].recurrent_state,atol=0,rtol=0)
  torch.testing.assert_close(layer.kv_conv_state,snapshot.backbone_state.layers[i].kv_conv_state,atol=0,rtol=0)
 assert torch.isfinite(actual).all()
result={'checkpoint':str(path),'standalone_loading':loading,'all_parameters_finite':True,'independent_layer_position_tables':5,'output_shape':list(actual.shape),'cache_vs_full_max_absolute_error':float((actual-expected).abs().max()),'cache_vs_full_relative_l2_error':cache_relative,'repeat_read_relative_l2_error':repeat_relative,'bf16_relative_l2_acceptance_bound':.035,'readonly_state_unchanged':True,'matrix_shapes':[list(layer.recurrent_state.shape) for layer in state.backbone_state.layers.values()],'query_kernel':cfg.composable_config['backbone']['query_conv_kernel_size'],'context_kernel':cfg.composable_config['backbone']['context_conv_kernel_size']}
# Separately require tight FP32 oracle equivalence on the SAME trained weights.
model.float()
for layer in model.backbone.layers:layer.mixer.backend='reference'
c=c.float();x=x.float();state=contracts.DraftState()
with torch.inference_mode():
 run(0,3);run(3,4);fp32_cached=run(4,8);fp32_full=run(0,8,False);fp32_repeat=run(8,8)
 torch.testing.assert_close(fp32_cached,fp32_full,atol=2e-5,rtol=2e-5)
 torch.testing.assert_close(fp32_repeat,fp32_cached,atol=2e-5,rtol=2e-5)
 result['fp32_reference_cache_max_absolute_error']=float((fp32_cached-fp32_full).abs().max())
 result['fp32_reference_repeat_max_absolute_error']=float((fp32_cached-fp32_repeat).abs().max())
print(json.dumps(result,indent=2))
