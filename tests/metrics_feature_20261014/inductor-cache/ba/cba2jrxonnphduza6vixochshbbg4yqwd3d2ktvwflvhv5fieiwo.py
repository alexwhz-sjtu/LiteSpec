# AOT ID: ['14_forward']
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


# kernel path: tests/metrics_feature_20261014/inductor-cache/kx/ckxycvskdu2f3nlsmwln5nzuiyyna7j2ylszddaht5z6bz5ax54p.py
# Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   student_log_probs => clone
# Graph fragment:
#   %primals_3 : Tensor "f32[s45, s3][1, s45]cuda:0" = PlaceHolder[target=primals_3]
#   %clone : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%primals_3,), kwargs = {memory_format: torch.contiguous_format})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%clone, -1), kwargs = {})
#   return %buf0
triton_red_fused__log_softmax_prepare_softmax_online_0 = async_compile.triton('triton_red_fused__log_softmax_prepare_softmax_online_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_prepare_softmax_online_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_prepare_softmax_online_0(in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x1*((4 + ks1) // 5)
        tmp1 = ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + ks0*r0_2 + ks0*x1*((4 + ks1) // 5)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: tests/metrics_feature_20261014/inductor-cache/we/cwee4jq6jenkmaeioxto7ip7vrdyo6hozpk7kjcadl5iw4dkpnxl.py
# Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   student_log_probs => clone
# Graph fragment:
#   %buf0 : Tensor "f32[s45, 1, 5][1, 5*s45, s45]cuda:0" = PlaceHolder[target=buf0]
#   %clone : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%primals_3,), kwargs = {memory_format: torch.contiguous_format})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%clone, -1), kwargs = {})
#   return %buf1
triton_per_fused__log_softmax_prepare_softmax_online_1 = async_compile.triton('triton_per_fused__log_softmax_prepare_softmax_online_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_prepare_softmax_online_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_prepare_softmax_online_1(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: tests/metrics_feature_20261014/inductor-cache/x5/cx5iwdtgsty7i65ebszquwcfvtbzhwizk643mipo35yxtrrs3a5b.py
# Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   student_log_probs => clone
# Graph fragment:
#   %primals_3 : Tensor "f32[s45, s3][1, s45]cuda:0" = PlaceHolder[target=primals_3]
#   %buf1 : Tensor "f32[s45, 1][1, s45]cuda:0" = PlaceHolder[target=buf1]
#   %clone : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%primals_3,), kwargs = {memory_format: torch.contiguous_format})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%clone, -1), kwargs = {})
#   return %buf2
triton_red_fused__log_softmax_prepare_softmax_online_2 = async_compile.triton('triton_red_fused__log_softmax_prepare_softmax_online_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_prepare_softmax_online_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_prepare_softmax_online_2(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + x1*((4 + ks1) // 5)
        tmp1 = ks1
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + ks0*r0_2 + ks0*x1*((4 + ks1) // 5)), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = libdevice.exp(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: tests/metrics_feature_20261014/inductor-cache/7o/c7oxx4d25anfctncdvw725youhwzkmb3dhev3wr57t5yikadnhmt.py
# Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
# Source node to ATen node mapping:
#   student_log_probs => clone, log
# Graph fragment:
#   %buf2 : Tensor "f32[s45, 1, 5][1, 5*s45, s45]cuda:0" = PlaceHolder[target=buf2]
#   %getitem_1 : Tensor "f32[s45, 1][1, s45]cuda:0" = PlaceHolder[target=getitem_1]
#   %clone : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%primals_3,), kwargs = {memory_format: torch.contiguous_format})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%clone, -1), kwargs = {})
#   %log : Tensor "f32[s45, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   return %getitem_1,%log
triton_per_fused__log_softmax_prepare_softmax_online_3 = async_compile.triton('triton_per_fused__log_softmax_prepare_softmax_online_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_prepare_softmax_online_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_prepare_softmax_online_3(in_out_ptr0, in_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 5
    R0_BLOCK: tl.constexpr = 8
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
    tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl_math.log(tmp4)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: tests/metrics_feature_20261014/inductor-cache/rj/crjwjgzjzau3eklf44zwlygq3apc6gxcew7hbkh445d6g6u6gmg4.py
# Topologically Sorted Source Nodes: [student_log_probs, sub, mul, values], Original ATen: [aten._log_softmax, prims.prepare_softmax_online, aten.sub, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_6
#   student_log_probs => clone, sub_1
#   sub => sub_6
#   values => sum_2
# Graph fragment:
#   %primals_5 : Tensor "f32[s45, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_4 : Tensor "f32[s45, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[s45, s3][1, s45]cuda:0" = PlaceHolder[target=primals_3]
#   %buf1 : Tensor "f32[s45, 1][1, s45]cuda:0" = PlaceHolder[target=buf1]
#   %log : Tensor "f32[s45, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %clone : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%primals_3,), kwargs = {memory_format: torch.contiguous_format})
#   %getitem : Tensor "f32[s45, 1][1, 1]cuda:0"[num_users=2] = call_function[target=operator.getitem](args = (%prepare_softmax_online_default, 0), kwargs = {})
#   %sub_tensor : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem), kwargs = {})
#   %sub_1 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %sub_6 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_4, %sub_1), kwargs = {})
#   %mul_6 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %sub_6), kwargs = {})
#   %sum_2 : Tensor "f32[s45][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [-1]), kwargs = {})
#   return %sum_2
triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_4 = async_compile.triton('triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (x0 + ks1*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tmp2 - tmp3
        tmp6 = tmp4 - tmp5
        tmp7 = tmp1 - tmp6
        tmp8 = tmp0 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: tests/metrics_feature_20261014/inductor-cache/2p/c2p3extj2ztbwui6bmynspfnfttughic2axcglp65ycc77glbjxi.py
# Topologically Sorted Source Nodes: [mul_1, sum_2], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul_1 => mul_10
#   sum_2 => sum_3
# Graph fragment:
#   %sum_2 : Tensor "f32[s45][1]cuda:0" = PlaceHolder[target=sum_2]
#   %primals_6 : Tensor "f32[s45][1]cuda:0" = PlaceHolder[target=primals_6]
#   %mul_10 : Tensor "f32[s45][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %primals_6), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_10,), kwargs = {})
#   return %sum_3
triton_red_fused_mul_sum_5 = async_compile.triton('triton_red_fused_mul_sum_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_5(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
        args.clear()
        s88 = primals_1
        s3 = primals_2
        primals_3_size = primals_3.size()
        s45 = primals_3_size[0]
        assert_size_stride(primals_3, (s45, s3), (1, s45))
        assert_size_stride(primals_4, (s45, s3), (s3, 1))
        assert_size_stride(primals_5, (s45, s3), (s3, 1))
        assert_size_stride(primals_6, (s45, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s45, 1, 5), (1, 5*s45, s45), torch.float32)
            # Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
            triton_red_fused__log_softmax_prepare_softmax_online_0_xnumel = 5*s45
            triton_red_fused__log_softmax_prepare_softmax_online_0_r0_numel = (4 + s3) // 5
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_prepare_softmax_online_0.run(primals_3, buf0, s45, s3, triton_red_fused__log_softmax_prepare_softmax_online_0_xnumel, triton_red_fused__log_softmax_prepare_softmax_online_0_r0_numel, stream=stream0)
            buf1 = empty_strided_cuda((s45, 1), (1, s45), torch.float32)
            # Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax_prepare_softmax_online_1.run(buf0, buf1, s45, s45, 5, stream=stream0)
            buf2 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
            triton_red_fused__log_softmax_prepare_softmax_online_2_xnumel = 5*s45
            triton_red_fused__log_softmax_prepare_softmax_online_2_r0_numel = (4 + s3) // 5
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_prepare_softmax_online_2.run(primals_3, buf1, buf2, s45, s3, triton_red_fused__log_softmax_prepare_softmax_online_2_xnumel, triton_red_fused__log_softmax_prepare_softmax_online_2_r0_numel, stream=stream0)
            buf3 = empty_strided_cuda((s45, 1), (1, s45), torch.float32)
            buf4 = reinterpret_tensor(buf3, (s45, 1), (1, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [student_log_probs], Original ATen: [aten._log_softmax, prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax_prepare_softmax_online_3.run(buf4, buf2, s45, s45, 5, stream=stream0)
            del buf2
            buf5 = empty_strided_cuda((s45, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [student_log_probs, sub, mul, values], Original ATen: [aten._log_softmax, prims.prepare_softmax_online, aten.sub, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_4.run(primals_5, primals_4, primals_3, buf1, buf4, buf5, s3, s45, s45, s3, stream=stream0)
            del primals_4
            buf6 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul_1, sum_2], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_5.run(buf5, primals_6, buf6, 1, s45, stream=stream0)
            del buf5
        return (buf6, primals_3, primals_5, primals_6, reinterpret_tensor(buf1, (s45, 1), (1, 1), 0), buf4, s45, s3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 7
    primals_2 = 513
    primals_3 = rand_strided((7, 513), (1, 7), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((7, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((7, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
