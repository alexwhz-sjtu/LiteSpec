# AOT ID: ['1_inference']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/cache_pipeline/server_7/inductor/v7/cv7tvyqpqvmccw42syfyyd7rhoowcjoreavx6tf4a6vphyypuk5m.py
# Topologically Sorted Source Nodes: [lt, neg, clamp, getitem, where], Original ATen: [aten.lt, aten.neg, aten.clamp, aten.index, aten.where, aten.copy_]
# Source node to ATen node mapping:
#   clamp => clamp_min
#   getitem => index
#   lt => lt
#   neg => neg
#   where => where
# Graph fragment:
#   %copy_ : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=copy_]
#   %arg2_1 : Tensor "i64[s35][1]cuda:0" = PlaceHolder[target=arg2_1]
#   %where : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=where]
#   %lt : Tensor "b8[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%arg0_1, 0), kwargs = {})
#   %neg : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%arg0_1,), kwargs = {})
#   %clamp_min : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%neg, 0), kwargs = {})
#   %index : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%clamp_min]), kwargs = {})
#   %where : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %index, %arg0_1), kwargs = {})
#   %copy_ : Tensor "i64[1][1]cuda:0"[num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%arg0_1, %where), kwargs = {})
#   return %where,%buf5
triton_poi_fused_clamp_copy__index_lt_neg_where_0 = async_compile.triton('triton_poi_fused_clamp_copy__index_lt_neg_where_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr1': '*i64', 'ks0': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_copy__index_lt_neg_where_0', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_copy__index_lt_neg_where_0(in_ptr0, in_ptr1, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = -tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp2)
    tmp6 = ks0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < ks0), "index out of bounds: 0 <= tmp9 < ks0")
    tmp11 = tl.load(in_ptr1 + (tmp9), None, eviction_policy='evict_last')
    tmp12 = tl.where(tmp3, tmp11, tmp1)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
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
        arg0_1, arg1_1, arg2_1 = args
        args.clear()
        s35 = arg1_1
        assert_size_stride(arg0_1, (1, ), (1, ))
        assert_size_stride(arg2_1, (s35, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            # Topologically Sorted Source Nodes: [lt, neg, clamp, getitem, where], Original ATen: [aten.lt, aten.neg, aten.clamp, aten.index, aten.where, aten.copy_]
            stream0 = get_raw_stream(0)
            triton_poi_fused_clamp_copy__index_lt_neg_where_0.run(arg0_1, arg2_1, arg0_1, s35, 1, stream=stream0)
            del arg0_1
            del arg2_1
        return ()

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg1_1 = 144
    arg2_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
