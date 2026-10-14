# AOT ID: ['8_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: tests/metrics_feature_20261014/inductor-cache/c2/cc2kcfwdw66c23el2dvsr34736krp5c2prsd6g2q72lwyevvfx3l.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul]
# Source node to ATen node mapping:
# Graph fragment:
#   %sub_7 : Tensor "f32[1, s13][s13, 1]cuda:0" = PlaceHolder[target=sub_7]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_4 : Tensor "f32[1, s13][1, 0]cuda:0" = PlaceHolder[target=primals_4]
#   %expand : Tensor "f32[1, s13][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [1, %primals_1]), kwargs = {})
#   %mul_10 : Tensor "f32[1, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_4), kwargs = {})
#   %mul_11 : Tensor "f32[1, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %mul_10), kwargs = {})
#   return %mul_11
triton_poi_fused_expand_mul_0 = async_compile.triton('triton_poi_fused_expand_mul_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_expand_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_expand_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp5 = tmp2 * tmp4
    tmp6 = tmp0 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_4, sub_7, tangents_1 = args
        args.clear()
        s13 = primals_1
        assert_size_stride(primals_4, (1, s13), (1, 0))
        assert_size_stride(sub_7, (1, s13), (s13, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = sub_7; del sub_7  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.expand, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused_expand_mul_0.run(buf0, tangents_1, primals_4, s13, stream=stream0)
            del primals_4
            del tangents_1
        return (None, buf0, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 65
    primals_4 = rand_strided((1, 65), (1, 0), device='cuda:0', dtype=torch.float32)
    sub_7 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_4, sub_7, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
