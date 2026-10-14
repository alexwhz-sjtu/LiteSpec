"""Capture only the static pure-CUDA Newton-Schulz optimizer subroutine."""
import json,time,statistics,torch
from pathlib import Path
from litespec.training.muon import newton_schulz
out=Path('/diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/muon_cudagraph_probe.json')
torch.cuda.set_device(0);torch.manual_seed(42)
results=[]
for shape in [(1,640,640),(1,4096,4096),(1,151936,320)]:
 x=torch.randn(shape,device='cuda',dtype=torch.bfloat16)
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(200):newton_schulz(x)
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 expected=newton_schulz(x);graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):actual=newton_schulz(x)
 graph.replay();torch.cuda.synchronize()
 torch.testing.assert_close(actual,expected,rtol=0,atol=0)
 records={}
 for mode in ['eager','graph','graph','eager']:
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  t=time.perf_counter();start.record()
  for _ in range(100):
   if mode=='eager':newton_schulz(x)
   else:graph.replay()
  end.record();torch.cuda.synchronize()
  records.setdefault(mode,[]).append(dict(wall_ms_per_call=(time.perf_counter()-t)*10,stream_ms_per_call=start.elapsed_time(end)/100))
 results.append(dict(shape=shape,exact_output_equal=True,measurements=records))
 del actual,expected,x,graph;torch.cuda.empty_cache()
result=dict(warmup=200,measured_calls_per_trial=100,trial_order=['eager','graph','graph','eager'],scope='Static Newton-Schulz only; no FSDP, NCCL, parameter mutation, CPU gradient norm or end-to-end speedup claim.',results=results)
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
