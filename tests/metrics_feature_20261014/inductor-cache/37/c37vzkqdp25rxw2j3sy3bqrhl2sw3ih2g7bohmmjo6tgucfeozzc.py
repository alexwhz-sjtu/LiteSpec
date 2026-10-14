# AOT ID: ['0_forward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/metrics_feature_20261014/inductor-cache/n5/cn5tjmvwqxt2p3fed6or4zlykglh7qwy4k44n4zen7wemtcfvjjj.py
# Topologically Sorted Source Nodes: [sub, abs_1, distance, max_1, max_2], Original ATen: [aten.sub, aten.abs, aten.sum, aten.max]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   distance => sum_1
#   max_1 => max_1
#   max_2 => max_2
#   sub => sub
# Graph fragment:
#   %primals_3 : Tensor "f32[s36, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[s36, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %sub : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %primals_4), kwargs = {})
#   %abs_1 : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[s36][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [-1]), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%primals_4, -1), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%primals_3, -1), kwargs = {})
#   return %buf0,%buf3,%buf6
triton_red_fused_abs_max_sub_sum_0 = async_compile.triton('triton_red_fused_abs_max_sub_sum_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_max_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_max_sub_sum_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 19)
    x1 = xindex // 19
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp18 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x0*((18 + ks0) // 19)
        tmp1 = ks0
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.abs(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
        tmp12 = tl.load(in_ptr1 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=float("-inf"))
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = triton_helpers.maximum(_tmp14, tmp13)
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr0 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=float("-inf"))
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
        tmp19 = triton_helpers.maximum(_tmp18, tmp17)
        _tmp18 = tl.where(r0_mask & xmask, tmp19, _tmp18)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp14 = triton_helpers.max2(_tmp14, 1)[:, None]
    tmp18 = triton_helpers.max2(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
    tl.store(out_ptr2 + (x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/metrics_feature_20261014/inductor-cache/fk/cfkaxgycuw4zojaj76zgemkv3nodhzxzbdmb3osp227fqfztrdej.py
# Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   distance => sum_1
#   sub => sub
# Graph fragment:
#   %buf0 : Tensor "f32[s36, 19][19, 1]cuda:0" = PlaceHolder[target=buf0]
#   %sub : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %primals_4), kwargs = {})
#   %abs_1 : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub,), kwargs = {})
#   %sum_1 : Tensor "f32[s36][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [-1]), kwargs = {})
#   return %sum_1
triton_per_fused_abs_sub_sum_1 = async_compile.triton('triton_per_fused_abs_sub_sum_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_sub_sum_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/metrics_feature_20261014/inductor-cache/qe/cqe2wfn7a35jotd4k5iynlcpq7rmlk6jomt6akslzgyxvdi3iqdg.py
# Topologically Sorted Source Nodes: [mul, sum_2], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_5
#   sum_2 => sum_2
# Graph fragment:
#   %sum_1 : Tensor "f32[s36][1]cuda:0" = PlaceHolder[target=sum_1]
#   %primals_5 : Tensor "f32[s36][1]cuda:0" = PlaceHolder[target=primals_5]
#   %mul_5 : Tensor "f32[s36][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, %primals_5), kwargs = {})
#   %sum_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_5,), kwargs = {})
#   return %sum_2
triton_red_fused_mul_sum_2 = async_compile.triton('triton_red_fused_mul_sum_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_2(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/metrics_feature_20261014/inductor-cache/b2/cb2kjcklifmdfmmuoo2ysydf4asaiu2u4z75jhnmvfuazqygfezi.py
# Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
# Graph fragment:
#   %buf3 : Tensor "f32[s36, 19][19, 1]cuda:0" = PlaceHolder[target=buf3]
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%primals_4, -1), kwargs = {})
#   return %getitem
triton_per_fused_max_3 = async_compile.triton('triton_per_fused_max_3', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_max_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_max_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/metrics_feature_20261014/inductor-cache/gl/cglrlz72oglrcalr3vhem7q2ecqszcl2mcqzunr6aehrjn4qrcmh.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub, aten.sgn]
# Source node to ATen node mapping:
#   sub => sub
# Graph fragment:
#   %primals_3 : Tensor "f32[s36, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_4 : Tensor "f32[s36, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %sub : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %primals_4), kwargs = {})
#   %sign : Tensor "f32[s36, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub,), kwargs = {})
#   return %sign
triton_poi_fused_sgn_sub_4 = async_compile.triton('triton_poi_fused_sgn_sub_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sgn_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sgn_sub_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tmp3 < tmp2
    tmp5 = tmp4.to(tl.int8)
    tmp6 = tmp2 < tmp3
    tmp7 = tmp6.to(tl.int8)
    tmp8 = tmp5 - tmp7
    tmp9 = tmp8.to(tmp2.dtype)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
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
        s36 = primals_1
        s12 = primals_2
        assert_size_stride(primals_3, (s36, s12), (s12, 1))
        assert_size_stride(primals_4, (s36, s12), (s12, 1))
        assert_size_stride(primals_5, (s36, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s36, 19), (19, 1), torch.float32)
            buf3 = empty_strided_cuda((s36, 19), (19, 1), torch.float32)
            buf6 = empty_strided_cuda((s36, 19), (19, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sub, abs_1, distance, max_1, max_2], Original ATen: [aten.sub, aten.abs, aten.sum, aten.max]
            triton_red_fused_abs_max_sub_sum_0_xnumel = 19*s36
            triton_red_fused_abs_max_sub_sum_0_r0_numel = (18 + s12) // 19
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_max_sub_sum_0.run(primals_3, primals_4, buf0, buf3, buf6, s12, triton_red_fused_abs_max_sub_sum_0_xnumel, triton_red_fused_abs_max_sub_sum_0_r0_numel, stream=stream0)
            buf1 = empty_strided_cuda((s36, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_abs_sub_sum_1.run(buf0, buf1, s36, 19, stream=stream0)
            del buf0
            buf2 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, sum_2], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_2.run(buf1, primals_5, buf2, 1, s36, stream=stream0)
            buf4 = empty_strided_cuda((s36, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused_max_3.run(buf3, buf4, s36, 19, stream=stream0)
            del buf3
            buf7 = empty_strided_cuda((s36, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [max_2], Original ATen: [aten.max]
            stream0 = get_raw_stream(0)
            triton_per_fused_max_3.run(buf6, buf7, s36, 19, stream=stream0)
            del buf6
            buf9 = empty_strided_cuda((s36, s12), (s12, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub, aten.sgn]
            triton_poi_fused_sgn_sub_4_xnumel = s12*s36
            stream0 = get_raw_stream(0)
            triton_poi_fused_sgn_sub_4.run(primals_3, primals_4, buf9, triton_poi_fused_sgn_sub_4_xnumel, stream=stream0)
            del primals_3
            del primals_4
        return (buf2, buf1, buf4, buf7, primals_5, buf9, s36, s12, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 151936
    primals_3 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
