# AOT ID: ['1_forward']
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/te/ctexwwfeeiye7lyxgvqb3im37dvlc5ufg37amm3fh7ggnyddf7zn.py
# Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_3 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_3, 1), kwargs = {})
#   return %buf0
triton_red_fused_prepare_softmax_online_0 = async_compile.triton('triton_red_fused_prepare_softmax_online_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_prepare_softmax_online_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_prepare_softmax_online_0(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 19)
    x1 = xindex // 19
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
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
        tmp3 = tl.load(in_ptr0 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=float("-inf"))
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/7j/c7j5f2p74u34x5npowazrcfntd4gjxea6qbmfdy5o5fhfw3lxoxp.py
# Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
# Source node to ATen node mapping:
# Graph fragment:
#   %buf0 : Tensor "f32[s88, 1, 19][19, 19*s88, 1]cuda:0" = PlaceHolder[target=buf0]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_3, 1), kwargs = {})
#   return %buf1
triton_per_fused_prepare_softmax_online_1 = async_compile.triton('triton_per_fused_prepare_softmax_online_1', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_prepare_softmax_online_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_prepare_softmax_online_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/ju/cjua5s3mebs4zogxpufolc4q457k2qdat5s6zuvyhokptpax4ro7.py
# Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_3 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %buf1 : Tensor "f32[s88, 1][1, s88]cuda:0" = PlaceHolder[target=buf1]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_3, 1), kwargs = {})
#   return %buf2
triton_red_fused_prepare_softmax_online_2 = async_compile.triton('triton_red_fused_prepare_softmax_online_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_prepare_softmax_online_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_prepare_softmax_online_2(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/mg/cmgzunh7cmdtig5b7xo4x3qfmopv56fz5ixs4bez6abkovlkcsvz.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1, gather, log, ne_2, neg, squeeze, sub_4, unsqueeze, where, where_1
# Graph fragment:
#   %buf2 : Tensor "f32[s88, 1, 19][19, 19*s88, 1]cuda:0" = PlaceHolder[target=buf2]
#   %getitem_1 : Tensor "f32[s88, 1][1, s88]cuda:0" = PlaceHolder[target=getitem_1]
#   %primals_4 : Tensor "i64[s88][1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %buf1 : Tensor "f32[s88, 1][1, s88]cuda:0" = PlaceHolder[target=buf1]
#   %log : Tensor "f32[s88, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_3, 1), kwargs = {})
#   %getitem : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=2] = call_function[target=operator.getitem](args = (%prepare_softmax_online_default, 0), kwargs = {})
#   %sub_tensor : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %getitem), kwargs = {})
#   %log : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_4 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne_2 : Tensor "b8[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_4, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %primals_4, %full_default), kwargs = {})
#   %unsqueeze : Tensor "i64[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_4, 1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_1), kwargs = {})
#   return %getitem_1,%log,%where_1
triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3 = async_compile.triton('triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl_math.log(tmp4)
    tmp7 = tl.full([1, 1], -100, tl.int64)
    tmp8 = tmp6 != tmp7
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tl.where(tmp8, tmp6, tmp9)
    tmp11 = ks0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tmp14) & (tmp14 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp14 < ks0")
    tmp16 = tl.load(in_ptr2 + (tmp14 + ks0*x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 - tmp5
    tmp20 = -tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp8, tmp20, tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr0 + (x0), tmp22, xmask)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/t3/ct3eob7youoowobbmue76bqwst7abjfr3jfus6lp6rs33wbrlxa3.py
# Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   distance => sum_3
#   sub => sub_14
# Graph fragment:
#   %primals_6 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %sub_14 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_6, %primals_7), kwargs = {})
#   %abs_1 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_14,), kwargs = {})
#   %sum_3 : Tensor "f32[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [-1]), kwargs = {})
#   return %buf7
triton_red_fused_abs_sub_sum_4 = async_compile.triton('triton_red_fused_abs_sub_sum_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_sub_sum_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_sub_sum_4(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp4 = tl.load(in_ptr1 + (r0_2 + ks0*x1 + x0*((18 + ks0) // 19)), r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl_math.abs(tmp5)
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/2f/c2fyipyuedlswchx6ooivgwedk6pelzma4cmd4bld5id63ia4sbc.py
# Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   distance => sum_3
#   sub => sub_14
# Graph fragment:
#   %buf7 : Tensor "f32[s88, 19][19, 1]cuda:0" = PlaceHolder[target=buf7]
#   %sub_14 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_6, %primals_7), kwargs = {})
#   %abs_1 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_14,), kwargs = {})
#   %sum_3 : Tensor "f32[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [-1]), kwargs = {})
#   return %sum_3
triton_per_fused_abs_sub_sum_5 = async_compile.triton('triton_per_fused_abs_sub_sum_5', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_sub_sum_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_abs_sub_sum_5(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/av/cav6ppvwlivmbv5hjzvcjcdgvd3i3t3hf3rht7xqnoi4rokycerb.py
# Topologically Sorted Source Nodes: [mul, ce, mul_1, sum_3], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   ce => sum_2
#   mul => mul_12
#   mul_1 => mul_21
#   sum_3 => sum_4
# Graph fragment:
#   %where_1 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=where_1]
#   %primals_5 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sum_3 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=sum_3]
#   %mul_12 : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %primals_5), kwargs = {})
#   %sum_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_12,), kwargs = {})
#   %mul_21 : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_3, %primals_5), kwargs = {})
#   %sum_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_21,), kwargs = {})
#   return %sum_2,%sum_4
triton_red_fused_mul_sum_6 = async_compile.triton('triton_red_fused_mul_sum_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
        tmp7 = tmp6 * tmp1
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp9, None)
''', device_str='cuda')


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/fy/cfyiogic4zr2jhusz6og7tp7zqjl5at2udlxe5njcbeheu5x3upc.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub, aten.sgn]
# Source node to ATen node mapping:
#   sub => sub_14
# Graph fragment:
#   %primals_6 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_7 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %sub_14 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_6, %primals_7), kwargs = {})
#   %sign : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_14,), kwargs = {})
#   return %sign
triton_poi_fused_sgn_sub_7 = async_compile.triton('triton_poi_fused_sgn_sub_7', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sgn_sub_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sgn_sub_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
        args.clear()
        s88 = primals_1
        s3 = primals_2
        assert_size_stride(primals_3, (s88, s3), (s3, 1))
        assert_size_stride(primals_4, (s88, ), (1, ))
        assert_size_stride(primals_5, (s88, ), (1, ))
        assert_size_stride(primals_6, (s88, s3), (s3, 1))
        assert_size_stride(primals_7, (s88, s3), (s3, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s88, 1, 19), (19, 19*s88, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
            triton_red_fused_prepare_softmax_online_0_xnumel = 19*s88
            triton_red_fused_prepare_softmax_online_0_r0_numel = (18 + s3) // 19
            stream0 = get_raw_stream(0)
            triton_red_fused_prepare_softmax_online_0.run(primals_3, buf0, s3, triton_red_fused_prepare_softmax_online_0_xnumel, triton_red_fused_prepare_softmax_online_0_r0_numel, stream=stream0)
            buf1 = empty_strided_cuda((s88, 1), (1, s88), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
            stream0 = get_raw_stream(0)
            triton_per_fused_prepare_softmax_online_1.run(buf0, buf1, s88, 19, stream=stream0)
            buf2 = buf0; del buf0  # reuse
            # Topologically Sorted Source Nodes: [], Original ATen: [prims.prepare_softmax_online]
            triton_red_fused_prepare_softmax_online_2_xnumel = 19*s88
            triton_red_fused_prepare_softmax_online_2_r0_numel = (18 + s3) // 19
            stream0 = get_raw_stream(0)
            triton_red_fused_prepare_softmax_online_2.run(primals_3, buf1, buf2, s3, triton_red_fused_prepare_softmax_online_2_xnumel, triton_red_fused_prepare_softmax_online_2_r0_numel, stream=stream0)
            buf3 = empty_strided_cuda((s88, 1), (1, s88), torch.float32)
            buf4 = reinterpret_tensor(buf3, (s88, 1), (1, 1), 0); del buf3  # reuse
            buf5 = empty_strided_cuda((s88, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream0 = get_raw_stream(0)
            triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3.run(buf4, buf2, primals_4, primals_3, buf1, buf5, s3, s88, 19, stream=stream0)
            buf7 = reinterpret_tensor(buf2, (s88, 19), (19, 1), 0); del buf2  # reuse
            # Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
            triton_red_fused_abs_sub_sum_4_xnumel = 19*s88
            triton_red_fused_abs_sub_sum_4_r0_numel = (18 + s3) // 19
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_sub_sum_4.run(primals_6, primals_7, buf7, s3, triton_red_fused_abs_sub_sum_4_xnumel, triton_red_fused_abs_sub_sum_4_r0_numel, stream=stream0)
            buf8 = empty_strided_cuda((s88, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [sub, abs_1, distance], Original ATen: [aten.sub, aten.abs, aten.sum]
            stream0 = get_raw_stream(0)
            triton_per_fused_abs_sub_sum_5.run(buf7, buf8, s88, 19, stream=stream0)
            del buf7
            buf6 = empty_strided_cuda((), (), torch.float32)
            buf9 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, ce, mul_1, sum_3], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_6.run(buf5, primals_5, buf8, buf6, buf9, 1, s88, stream=stream0)
            buf10 = empty_strided_cuda((s88, s3), (s3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub, aten.sgn]
            triton_poi_fused_sgn_sub_7_xnumel = s3*s88
            stream0 = get_raw_stream(0)
            triton_poi_fused_sgn_sub_7.run(primals_6, primals_7, buf10, triton_poi_fused_sgn_sub_7_xnumel, stream=stream0)
            del primals_6
            del primals_7
        return (buf6, buf9, buf5, buf8, primals_3, primals_4, primals_5, reinterpret_tensor(buf1, (s88, 1), (1, 1), 0), buf4, buf10, s88, s3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 151936
    primals_3 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
