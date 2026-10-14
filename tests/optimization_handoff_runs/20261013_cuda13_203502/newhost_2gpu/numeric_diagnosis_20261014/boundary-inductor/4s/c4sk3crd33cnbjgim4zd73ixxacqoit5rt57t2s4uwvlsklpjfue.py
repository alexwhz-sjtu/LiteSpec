# AOT ID: ['1_backward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/vv/cvvnamsnxzzcqmlxef5imxaiif2xhy7rwmyco36mtoef7ka547iw.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "f32[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %sum_2 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [-1], True), kwargs = {})
#   return %buf0
triton_red_fused__log_softmax_backward_data_0 = async_compile.triton('triton_red_fused__log_softmax_backward_data_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_0(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 19)
    x1 = xindex // 19
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x0*((18 + ks0) // 19)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/7z/c7zgspgmytlgwqdm2inshqmzbfxijdktm2gotuvnsco7grcuembo.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf0 : Tensor "f32[s77, 1, 19][19, 19*s77, 1]cuda:0" = PlaceHolder[target=buf0]
#   %sum_2 : Tensor "f32[s77, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%tangents_1, [-1], True), kwargs = {})
#   return %sum_2
triton_per_fused__log_softmax_backward_data_1 = async_compile.triton('triton_per_fused__log_softmax_backward_data_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 19
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 19*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/r3/cr3n63wlp77msn3z3nr2nnqpj5foifuoq23laizbck34iygy7fbr.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
# Source node to ATen node mapping:
# Graph fragment:
#   %tangents_1 : Tensor "f32[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %sub_1 : Tensor "f32[s77, s27][s27, 1]cuda:0" = PlaceHolder[target=sub_1]
#   %sum_2 : Tensor "f32[s77, 1][1, s77]cuda:0" = PlaceHolder[target=sum_2]
#   %exp_1 : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %mul_6 : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_2), kwargs = {})
#   %sub_6 : Tensor "f32[s77, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%tangents_1, %mul_6), kwargs = {})
#   return %sub_6
triton_poi_fused__log_softmax_backward_data_2 = async_compile.triton('triton_poi_fused__log_softmax_backward_data_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__log_softmax_backward_data_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__log_softmax_backward_data_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = libdevice.exp(tmp1)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 - tmp4
    tl.store(out_ptr0 + (x2), tmp5, xmask)
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
        primals_2, primals_1, sub_1, tangents_1 = args
        args.clear()
        s27 = primals_2
        s77 = primals_1
        assert_size_stride(sub_1, (s77, s27), (s27, 1))
        assert_size_stride(tangents_1, (s77, s27), (s27, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s77, 1, 19), (19, 19*s77, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
            triton_red_fused__log_softmax_backward_data_0_xnumel = 19*s77
            triton_red_fused__log_softmax_backward_data_0_r0_numel = (18 + s27) // 19
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_backward_data_0.run(tangents_1, buf0, s27, triton_red_fused__log_softmax_backward_data_0_xnumel, triton_red_fused__log_softmax_backward_data_0_r0_numel, stream=stream0)
            buf1 = empty_strided_cuda((s77, 1), (1, s77), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax_backward_data_1.run(buf0, buf1, s77, 19, stream=stream0)
            del buf0
            buf2 = empty_strided_cuda((s77, s27), (s27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten._log_softmax_backward_data]
            triton_poi_fused__log_softmax_backward_data_2_xnumel = s27*s77
            stream0 = get_raw_stream(0)
            triton_poi_fused__log_softmax_backward_data_2.run(tangents_1, sub_1, buf1, buf2, s27, triton_poi_fused__log_softmax_backward_data_2_xnumel, stream=stream0)
            del buf1
            del sub_1
            del tangents_1
        return (None, None, buf2, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = 151936
    primals_1 = 32
    sub_1 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_2, primals_1, sub_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
