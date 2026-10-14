# AOT ID: ['6_forward']
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/yx/cyxdyh7m6cgtgeisnxlqv7z5tax7qlniz6i4r66e5eodytx7yjxs.py
# Topologically Sorted Source Nodes: [values, mul, sum_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_7
#   sum_1 => sum_1
#   values => abs_1, exp, full_default, log1p, minimum, mul_4, neg, sub_4, sub_5, sub_6
# Graph fragment:
#   %primals_4 : Tensor "f32[s24, s13][s13, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[s24, s13][s13, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_5 : Tensor "f32[s24, s13][1, 0]cuda:0" = PlaceHolder[target=primals_5]
#   %sub_4 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %primals_4), kwargs = {})
#   %mul_4 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %primals_3), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %primals_3), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%primals_3,), kwargs = {})
#   %neg : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_5 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_6 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %sub_5), kwargs = {})
#   %mul_7 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %primals_5), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_7,), kwargs = {})
#   return %buf1
triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0 = async_compile.triton('triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2, 'r0_': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = r0_1 + x0*((1 + ks0*ks1) // 2)
        tmp1 = ks0*ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (((r0_1 + x0*((1 + ks0*ks1) // 2)) % (ks0*ks1))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = 1.0
        tmp5 = tmp4 - tmp3
        tmp6 = tl.load(in_ptr1 + (((r0_1 + x0*((1 + ks0*ks1) // 2)) % (ks0*ks1))), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tmp5 * tmp6
        tmp8 = 0.0
        tmp9 = triton_helpers.minimum(tmp8, tmp6)
        tmp10 = tl_math.abs(tmp6)
        tmp11 = -tmp10
        tmp12 = libdevice.exp(tmp11)
        tmp13 = libdevice.log1p(tmp12)
        tmp14 = tmp9 - tmp13
        tmp15 = tmp7 - tmp14
        tmp16 = tl.load(in_ptr2 + ((((r0_1 + x0*((1 + ks0*ks1) // 2)) // ks0) % ks1)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/7c/c7c43viybwbffwz35cx6sawda665vp5g6hlogvcxp3o4mr34y2jk.py
# Topologically Sorted Source Nodes: [values, mul, sum_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_7
#   sum_1 => sum_1
#   values => abs_1, exp, full_default, log1p, minimum, mul_4, neg, sub_4, sub_5, sub_6
# Graph fragment:
#   %buf1 : Tensor "f32[2][1]cuda:0" = PlaceHolder[target=buf1]
#   %sub_4 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %primals_4), kwargs = {})
#   %mul_4 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %primals_3), kwargs = {})
#   %full_default : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %minimum : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.minimum.default](args = (%full_default, %primals_3), kwargs = {})
#   %abs_1 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%primals_3,), kwargs = {})
#   %neg : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%abs_1,), kwargs = {})
#   %exp : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %log1p : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %sub_5 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%minimum, %log1p), kwargs = {})
#   %sub_6 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, %sub_5), kwargs = {})
#   %mul_7 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %primals_5), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_7,), kwargs = {})
#   return %sum_1
triton_per_fused_binary_cross_entropy_with_logits_mul_sum_1 = async_compile.triton('triton_per_fused_binary_cross_entropy_with_logits_mul_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r0_': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_binary_cross_entropy_with_logits_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'r0_': 2}}
)
@triton.jit
def triton_per_fused_binary_cross_entropy_with_logits_mul_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_0), None)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/7l/c7lxthwwpxw5osruo5ldkefkce33wdqkokpr4qs7qdo52s7r45ow.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.sub]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_3 : Tensor "f32[s24, s13][s13, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[s24, s13][s13, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %sigmoid : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%primals_3,), kwargs = {})
#   %sub_11 : Tensor "f32[s24, s13][s13, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sigmoid, %primals_4), kwargs = {})
#   return %sub_11
triton_poi_fused_sigmoid_sub_2 = async_compile.triton('triton_poi_fused_sigmoid_sub_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_sub_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp3 = tmp1 - tmp2
    tl.store(out_ptr0 + (x0), tmp3, xmask)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        s88 = primals_1
        s3 = primals_2
        primals_3_size = primals_3.size()
        s24 = primals_3_size[0]
        s13 = primals_3_size[1]
        assert_size_stride(primals_3, (s24, s13), (s13, 1))
        assert_size_stride(primals_4, (s24, s13), (s13, 1))
        assert_size_stride(primals_5, (s24, s13), (1, 0))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((2, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [values, mul, sum_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
            triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0_r0_numel = (1 + s13*s24) // 2
            stream0 = get_raw_stream(0)
            triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0.run(primals_4, primals_3, primals_5, buf1, s13, s24, 2, triton_red_fused_binary_cross_entropy_with_logits_mul_sum_0_r0_numel, stream=stream0)
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [values, mul, sum_1], Original ATen: [aten.binary_cross_entropy_with_logits, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_binary_cross_entropy_with_logits_mul_sum_1.run(buf1, buf2, 1, 2, stream=stream0)
            del buf1
            buf3 = empty_strided_cuda((s24, s13), (s13, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.sigmoid, aten.sub]
            triton_poi_fused_sigmoid_sub_2_xnumel = s13*s24
            stream0 = get_raw_stream(0)
            triton_poi_fused_sigmoid_sub_2.run(primals_3, primals_4, buf3, triton_poi_fused_sigmoid_sub_2_xnumel, stream=stream0)
            del primals_3
            del primals_4
        return (buf2, primals_5, buf3, s24, s13, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 17
    primals_2 = 513
    primals_3 = rand_strided((17, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((17, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((17, 513), (1, 0), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
