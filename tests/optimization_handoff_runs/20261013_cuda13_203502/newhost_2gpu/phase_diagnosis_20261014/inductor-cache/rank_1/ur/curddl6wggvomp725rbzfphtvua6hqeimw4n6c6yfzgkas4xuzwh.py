# AOT ID: ['0_inference']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/inductor-cache/rank_1/cm/ccmbjw4c4dwln73x274kckt4qrqc3coelpaivjhieihwa26yrkkh.py
# Topologically Sorted Source Nodes: [b, m, qb, safe_qb, n, lt, anchor, sub, ge, and_2, sub_1, le_1, history, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, or_1, is_q, sub_3, kv_qb, eq_1, and_7, sub_4, slot, safe_slot, lt_2, index_2, safe_kb, child_3, invert, padded_token, invert_1, query, or_2], Original ATen: [aten.arange, aten.floor_divide, aten.clamp, aten.lt, aten.view, aten.index, aten.sub, aten.ge, aten.bitwise_and, aten.le, aten.eq, aten.bitwise_or, aten.remainder, aten.bitwise_not]
# Source node to ATen node mapping:
#   anchor => index_1, view_1
#   and_2 => bitwise_and_1
#   and_7 => bitwise_and_6
#   b => iota
#   child_3 => clamp_max_3
#   chs_valid => bitwise_and_4
#   eq => eq, view_4
#   eq_1 => eq_1, view_6
#   floor_divide_1 => div_1
#   ge => ge, view_2
#   ge_1 => ge_1
#   history => bitwise_and_2
#   index_2 => index_2, view_5
#   invert => bitwise_not
#   invert_1 => bitwise_not_1
#   is_chs => bitwise_and_3
#   is_q => ge_2
#   kv_qb => div_2
#   le_1 => le_1, view_3
#   lt => lt
#   lt_1 => lt_1
#   lt_2 => lt_2
#   m => iota_2
#   n => iota_3
#   or_1 => bitwise_or
#   or_2 => bitwise_or_1
#   padded_token => bitwise_and_5
#   qb => div
#   query => bitwise_and_7, view_7
#   safe_kb => clamp_max_1, clamp_min_1
#   safe_qb => clamp_max, clamp_min
#   safe_slot => clamp_max_2, clamp_min_2
#   slot => remainder
#   sub => sub
#   sub_1 => sub_1
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
# Graph fragment:
#   %arg1_1 : Tensor "i64[1, 512][512, 1]cuda:1" = PlaceHolder[target=arg1_1]
#   %arg2_1 : Tensor "b8[1, 512, 1][512, 1, 1]cuda:1" = PlaceHolder[target=arg2_1]
#   %iota : Tensor "i64[1][1]cuda:1"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %iota_2 : Tensor "i64[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4096,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %div : Tensor "i64[4096][1]cuda:1"[num_users=5] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 8), kwargs = {rounding_mode: floor})
#   %clamp_min : Tensor "i64[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 0), kwargs = {})
#   %clamp_max : Tensor "i64[4096][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 511), kwargs = {})
#   %iota_3 : Tensor "i64[6656][1]cuda:1"[num_users=9] = call_function[target=torch.ops.prims.iota.default](args = (6656,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %lt : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, 0), kwargs = {})
#   %view_1 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, 4096][4096, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg1_1, [%view_1, %clamp_max]), kwargs = {})
#   %sub : Tensor "i64[1, 4096][4096, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_1, 1), kwargs = {})
#   %view_2 : Tensor "i64[1, 4096, 1][4096, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub, [1, 4096, 1]), kwargs = {})
#   %ge : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%iota_3, %view_2), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt, %ge), kwargs = {})
#   %sub_1 : Tensor "i64[1, 4096][4096, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_1, 2), kwargs = {})
#   %view_3 : Tensor "i64[1, 4096, 1][4096, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_1, [1, 4096, 1]), kwargs = {})
#   %le_1 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%iota_3, %view_3), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_1, %le_1), kwargs = {})
#   %ge_1 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, 0), kwargs = {})
#   %lt_1 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, 2560), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %sub_2 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 0), kwargs = {})
#   %div_1 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_2, 5), kwargs = {rounding_mode: floor})
#   %view_4 : Tensor "i64[4096, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4096, 1]), kwargs = {})
#   %eq : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_1, %view_4), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %eq), kwargs = {})
#   %bitwise_or : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_2, %bitwise_and_4), kwargs = {})
#   %ge_2 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, 2560), kwargs = {})
#   %sub_3 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 2560), kwargs = {})
#   %div_2 : Tensor "i64[6656][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_3, 8), kwargs = {rounding_mode: floor})
#   %view_6 : Tensor "i64[4096, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4096, 1]), kwargs = {})
#   %eq_1 : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_2, %view_6), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_2, %eq_1), kwargs = {})
#   %sub_4 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 2560), kwargs = {})
#   %remainder : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%sub_4, 8), kwargs = {})
#   %clamp_min_2 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%remainder, 0), kwargs = {})
#   %clamp_max_2 : Tensor "i64[6656][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 7), kwargs = {})
#   %lt_2 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%clamp_max_2, 1), kwargs = {})
#   %view_5 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %clamp_min_1 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_2, 0), kwargs = {})
#   %clamp_max_1 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 511), kwargs = {})
#   %clamp_max_3 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_max_2, 0), kwargs = {})
#   %index_2 : Tensor "b8[1, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg2_1, [%view_5, %clamp_max_1, %clamp_max_3]), kwargs = {})
#   %bitwise_not : Tensor "b8[1, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%index_2,), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[1, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt_2, %bitwise_not), kwargs = {})
#   %bitwise_not_1 : Tensor "b8[1, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and_5,), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, 6656][6656, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_not_1, [1, 1, 6656]), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %view_7), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %bitwise_and_7), kwargs = {})
#   return %bitwise_or_1
triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0 = async_compile.triton('triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27262976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6656)
    x1 = xindex // 6656
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (((511) * ((511) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (511)))), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (((511) * ((511) <= (((0) * ((0) >= ((-320) + (x0 // 8))) + ((-320) + (x0 // 8)) * (((-320) + (x0 // 8)) > (0))))) + (((0) * ((0) >= ((-320) + (x0 // 8))) + ((-320) + (x0 // 8)) * (((-320) + (x0 // 8)) > (0)))) * ((((0) * ((0) >= ((-320) + (x0 // 8))) + ((-320) + (x0 // 8)) * (((-320) + (x0 // 8)) > (0)))) < (511)))), None, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp4 = tl.full([1], 1, tl.int64)
    tmp5 = tmp3 - tmp4
    tmp6 = tmp0 >= tmp5
    tmp7 = tmp2 & tmp6
    tmp8 = tl.full([1], 2, tl.int64)
    tmp9 = tmp3 - tmp8
    tmp10 = tmp0 <= tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tmp0 >= tmp1
    tmp13 = tl.full([1], 2560, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = x0 // 5
    tmp17 = x1 // 8
    tmp18 = tmp16 == tmp17
    tmp19 = tmp15 & tmp18
    tmp20 = tmp11 | tmp19
    tmp21 = tmp0 >= tmp13
    tmp22 = (-320) + (x0 // 8)
    tmp23 = tmp22 == tmp17
    tmp24 = tmp21 & tmp23
    tmp25 = (-2560) + x0
    tmp26 = tl.full([1], 8, tl.int64)
    tmp27 = (tmp25 % tmp26)
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = tmp27 != tmp28
    tmp30 = (libdevice.signbit(tmp27) != 0) if (tmp27).dtype is tl.float32 else tmp27 < 0
    tmp31 = (libdevice.signbit(tmp26) != 0) if (tmp26).dtype is tl.float32 else tmp26 < 0
    tmp32 = tmp30 != tmp31
    tmp33 = tmp29 & tmp32
    tmp34 = tmp27 + tmp26
    tmp35 = tl.where(tmp33, tmp34, tmp27)
    tmp36 = triton_helpers.maximum(tmp35, tmp1)
    tmp37 = tl.full([1], 7, tl.int64)
    tmp38 = triton_helpers.minimum(tmp36, tmp37)
    tmp39 = tmp38 < tmp4
    tmp40 = triton_helpers.minimum(tmp38, tmp1)
    tmp42 = tmp41 == 0
    tmp43 = tmp39 & tmp42
    tmp44 = tmp43 == 0
    tmp45 = tmp24 & tmp44
    tmp46 = tmp20 | tmp45
    tl.store(out_ptr0 + (x2), tmp46, None)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/inductor-cache/rank_1/2w/c2wsiu63sniegsk4ttacqr45dpxtm645tk6py4jsoasb2l5idtal.py
# Topologically Sorted Source Nodes: [b, index, m, qb, safe_qb, le, valid_block, and_9, n, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, invert_2, and_10, le_2, and_11, batched_outputs, batched_outputs_3, mask_2, mask_3, mask_block_sum], Original ATen: [aten.arange, aten.view, aten.floor_divide, aten.clamp, aten.index, aten.le, aten.bitwise_and, aten.ge, aten.lt, aten.sub, aten.eq, aten.bitwise_not, aten.bitwise_or, aten.permute, aten.sum]
# Source node to ATen node mapping:
#   and_10 => bitwise_and_9, view_9
#   and_11 => bitwise_and_10, view_10
#   and_9 => bitwise_and_8, view_8
#   b => iota
#   batched_outputs => bitwise_or_2
#   batched_outputs_3 => view_11
#   chs_valid => bitwise_and_4
#   eq => eq, view_4
#   floor_divide_1 => div_1
#   ge_1 => ge_1
#   index => index, view
#   invert_2 => bitwise_not_2
#   is_chs => bitwise_and_3
#   le => le
#   le_2 => le_2
#   lt_1 => lt_1
#   m => iota_2
#   mask_2 => view_12
#   mask_3 => permute
#   mask_block_sum => sum_1
#   n => iota_3
#   qb => div
#   safe_qb => clamp_max, clamp_min
#   sub_2 => sub_2
#   valid_block => bitwise_and
# Graph fragment:
#   %arg0_1 : Tensor "b8[1, 512][512, 1]cuda:1" = PlaceHolder[target=arg0_1]
#   %bitwise_or_1 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1" = PlaceHolder[target=bitwise_or_1]
#   %iota : Tensor "i64[1][1]cuda:1"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %view : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %iota_2 : Tensor "i64[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4096,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %div : Tensor "i64[4096][1]cuda:1"[num_users=5] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 8), kwargs = {rounding_mode: floor})
#   %clamp_min : Tensor "i64[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 0), kwargs = {})
#   %clamp_max : Tensor "i64[4096][1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 511), kwargs = {})
#   %index : Tensor "b8[1, 4096][4096, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg0_1, [%view, %clamp_max]), kwargs = {})
#   %le : Tensor "b8[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%div, 511), kwargs = {})
#   %bitwise_and : Tensor "b8[1, 4096][4096, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%index, %le), kwargs = {})
#   %view_8 : Tensor "b8[1, 4096, 1][4096, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and, [1, 4096, 1]), kwargs = {})
#   %iota_3 : Tensor "i64[6656][1]cuda:1"[num_users=9] = call_function[target=torch.ops.prims.iota.default](args = (6656,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %ge_1 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, 0), kwargs = {})
#   %lt_1 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, 2560), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_1, %lt_1), kwargs = {})
#   %sub_2 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 0), kwargs = {})
#   %div_1 : Tensor "i64[6656][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_2, 5), kwargs = {rounding_mode: floor})
#   %view_4 : Tensor "i64[4096, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [4096, 1]), kwargs = {})
#   %eq : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_1, %view_4), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[4096, 6656][6656, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %eq), kwargs = {})
#   %bitwise_and_8 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%view_8, %bitwise_or_1), kwargs = {})
#   %bitwise_not_2 : Tensor "b8[1, 4096][4096, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and,), kwargs = {})
#   %view_9 : Tensor "b8[1, 4096, 1][4096, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_not_2, [1, 4096, 1]), kwargs = {})
#   %bitwise_and_9 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %view_9), kwargs = {})
#   %le_2 : Tensor "b8[4096][1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%div, 511), kwargs = {})
#   %view_10 : Tensor "b8[4096, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%le_2, [4096, 1]), kwargs = {})
#   %bitwise_and_10 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_9, %view_10), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[1, 4096, 6656][27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_8, %bitwise_and_10), kwargs = {})
#   %view_11 : Tensor "b8[1, 1, 4096, 6656][27262976, 27262976, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_or_2, [1, 1, 4096, 6656]), kwargs = {})
#   %view_12 : Tensor "b8[1, 1, 32, 128, 52, 128][27262976, 27262976, 851968, 6656, 128, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%expand, [1, 1, 32, 128, 52, 128]), kwargs = {})
#   %permute : Tensor "b8[1, 1, 32, 52, 128, 128][27262976, 27262976, 851968, 128, 6656, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %sum_1 : Tensor "i64[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   return %sum_1
triton_red_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_permute_sub_sum_view_1 = async_compile.triton('triton_red_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_permute_sub_sum_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 16384},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_permute_sub_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 26624, 'r0_': 27262976}}
)
@triton.jit
def triton_red_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_permute_sub_sum_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1664
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = xindex // 52
    x0 = (xindex % 52)
    _tmp22 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index // 128
        r0_2 = (r0_index % 128)
        tmp0 = tl.load(in_ptr0 + (((511) * ((511) <= (((0) * ((0) >= (16*x1 + (r0_3 // 8))) + (16*x1 + (r0_3 // 8)) * ((16*x1 + (r0_3 // 8)) > (0))))) + (((0) * ((0) >= (16*x1 + (r0_3 // 8))) + (16*x1 + (r0_3 // 8)) * ((16*x1 + (r0_3 // 8)) > (0)))) * ((((0) * ((0) >= (16*x1 + (r0_3 // 8))) + (16*x1 + (r0_3 // 8)) * ((16*x1 + (r0_3 // 8)) > (0)))) < (511)))), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp5 = tl.load(in_ptr1 + (r0_2 + 128*x0 + 6656*r0_3 + 851968*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = 16*x1 + (r0_3 // 8)
        tmp2 = tl.full([1, 1], 511, tl.int64)
        tmp3 = tmp1 <= tmp2
        tmp4 = tmp0 & tmp3
        tmp6 = tmp4 & tmp5
        tmp7 = r0_2 + 128*x0
        tmp8 = tl.full([1, 1], 0, tl.int64)
        tmp9 = tmp7 >= tmp8
        tmp10 = tl.full([1, 1], 2560, tl.int64)
        tmp11 = tmp7 < tmp10
        tmp12 = tmp9 & tmp11
        tmp13 = (r0_2 + 128*x0) // 5
        tmp14 = tmp13 == tmp1
        tmp15 = tmp12 & tmp14
        tmp16 = tmp4 == 0
        tmp17 = tmp15 & tmp16
        tmp18 = tmp17 & tmp3
        tmp19 = tmp6 | tmp18
        tmp20 = tmp19.to(tl.int64)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(r0_mask & xmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp22, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/inductor-cache/rank_1/nr/cnregwf5jzttldwknr3jjnziitdalgcaisq4f5n7z5oowcrkirht.py
# Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   dense_mask_4 => full_default_3
# Graph fragment:
#   %full_default_3 : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 32, 53], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   return %index_put_1
triton_poi_fused_new_zeros_2 = async_compile.triton('triton_poi_fused_new_zeros_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 13568}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/inductor-cache/rank_1/vb/cvbcwdtgy33ztz66josjqamljpkfyslytswr5sjxmakuwrjxrezd.py
# Topologically Sorted Source Nodes: [gt, lt_3, partial_blocks, partial_blocks_1, dense_mask, col_indices, full_blocks, full_blocks_1, dense_mask_1, col_indices_1, dense_mask_2, setitem, arange_4, row_indices, col_range, num_blocks_in_row, child_4, unsqueeze_1, index_mask, child_5, valid_indices, dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, num_blocks_in_row_1, child_8, unsqueeze_3, index_mask_1, child_9, valid_indices_1], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort, aten.eq, aten.new_zeros, aten.arange, aten.unsqueeze, aten.sum, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
# Source node to ATen node mapping:
#   arange_4 => iota_4
#   arange_6 => iota_8
#   child_4 => convert_element_type_3
#   child_5 => convert_element_type_4
#   child_8 => convert_element_type_6
#   child_9 => convert_element_type_7
#   col_indices => sort
#   col_indices_1 => sort_1
#   col_range => iota_5
#   col_range_1 => iota_9
#   dense_mask => convert_element_type_2
#   dense_mask_1 => convert_element_type_5
#   dense_mask_2 => full_default
#   dense_mask_4 => full_default_3
#   full_blocks => eq_2
#   full_blocks_1 => convert_element_type_1
#   gt => gt
#   index_mask => lt_4
#   index_mask_1 => lt_5
#   lt_3 => lt_3
#   num_blocks_in_row => sum_2
#   num_blocks_in_row_1 => sum_3
#   partial_blocks => bitwise_and_11
#   partial_blocks_1 => convert_element_type
#   row_indices => unsqueeze
#   row_indices_1 => unsqueeze_7
#   setitem => full_default_2, index_put, iota_6, iota_7, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6
#   setitem_1 => full_default_5, index_put_1, iota_10, iota_11, unsqueeze_10, unsqueeze_11, unsqueeze_12, unsqueeze_13, unsqueeze_9
#   unsqueeze_1 => unsqueeze_1
#   unsqueeze_3 => unsqueeze_8
#   valid_indices => full_default_1, where
#   valid_indices_1 => full_default_4, where_1
# Graph fragment:
#   %sum_1 : Tensor "i64[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1" = PlaceHolder[target=sum_1]
#   %sum_2 : Tensor "i64[1, 1, 32][32, 32, 1]cuda:1" = PlaceHolder[target=sum_2]
#   %sum_3 : Tensor "i64[1, 1, 32][32, 32, 1]cuda:1" = PlaceHolder[target=sum_3]
#   %buf3 : Tensor "i16[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1" = PlaceHolder[target=buf3]
#   %convert_element_type_3 : Tensor "i32[1, 1, 32][32, 32, 1]cuda:1" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_4 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1" = PlaceHolder[target=convert_element_type_4]
#   %index_put : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1" = PlaceHolder[target=index_put]
#   %buf5 : Tensor "i16[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1" = PlaceHolder[target=buf5]
#   %convert_element_type_6 : Tensor "i32[1, 1, 32][32, 32, 1]cuda:1" = PlaceHolder[target=convert_element_type_6]
#   %convert_element_type_7 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1" = PlaceHolder[target=convert_element_type_7]
#   %index_put_1 : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1" = PlaceHolder[target=index_put_1]
#   %gt : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt_3 : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and_11 : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt_3), kwargs = {})
#   %convert_element_type : Tensor "i8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_11, torch.int8), kwargs = {})
#   %convert_element_type_2 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int32), kwargs = {})
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%convert_element_type_2,), kwargs = {stable: True, descending: True})
#   %eq_2 : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : Tensor "i8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_2, torch.int8), kwargs = {})
#   %convert_element_type_5 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.int32), kwargs = {})
#   %sort_1 : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%convert_element_type_5,), kwargs = {stable: True, descending: True})
#   %full_default : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 32, 53], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %iota_7 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_7, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, -1), kwargs = {})
#   %iota_6 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_6, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %iota_4 : Tensor "i32[32][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:1, requires_grad: False})
#   %unsqueeze : Tensor "i32[32, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_4, -1), kwargs = {})
#   %iota_5 : Tensor "i32[52][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (52,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:1, requires_grad: False})
#   %sum_2 : Tensor "i64[1, 1, 32][32, 32, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[1, 1, 32][32, 32, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.int32), kwargs = {})
#   %unsqueeze_1 : Tensor "i32[1, 1, 32, 1][32, 32, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 3), kwargs = {})
#   %lt_4 : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_5, %unsqueeze_1), kwargs = {})
#   %convert_element_type_4 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.int32), kwargs = {})
#   %full_default_1 : Tensor "i32[][]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 52), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %where : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_4, %convert_element_type_4, %full_default_1), kwargs = {})
#   %full_default_2 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %index_put : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%unsqueeze_6, %unsqueeze_3, %unsqueeze, %where], %full_default_2), kwargs = {})
#   %full_default_3 : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 32, 53], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %iota_11 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %unsqueeze_11 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_11, -1), kwargs = {})
#   %unsqueeze_12 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_11, -1), kwargs = {})
#   %unsqueeze_13 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_12, -1), kwargs = {})
#   %iota_10 : Tensor "i64[1][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:1, requires_grad: False})
#   %unsqueeze_9 : Tensor "i64[1, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_10, -1), kwargs = {})
#   %unsqueeze_10 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_9, -1), kwargs = {})
#   %iota_8 : Tensor "i32[32][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:1, requires_grad: False})
#   %unsqueeze_7 : Tensor "i32[32, 1][1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_8, -1), kwargs = {})
#   %iota_9 : Tensor "i32[52][1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (52,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:1, requires_grad: False})
#   %sum_3 : Tensor "i64[1, 1, 32][32, 32, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_5, [-1]), kwargs = {})
#   %convert_element_type_6 : Tensor "i32[1, 1, 32][32, 32, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_3, torch.int32), kwargs = {})
#   %unsqueeze_8 : Tensor "i32[1, 1, 32, 1][32, 32, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_6, 3), kwargs = {})
#   %lt_5 : Tensor "b8[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_9, %unsqueeze_8), kwargs = {})
#   %convert_element_type_7 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_3, torch.int32), kwargs = {})
#   %full_default_4 : Tensor "i32[][]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 52), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %where_1 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_5, %convert_element_type_7, %full_default_4), kwargs = {})
#   %full_default_5 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:1, pin_memory: False})
#   %index_put_1 : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_3, [%unsqueeze_13, %unsqueeze_10, %unsqueeze_7, %where_1], %full_default_5), kwargs = {})
#   return %buf3,%buf5,%sum_2,%sum_3,%convert_element_type_3,%convert_element_type_6,%convert_element_type_4,%buf10,%convert_element_type_7,%buf17
triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_3 = async_compile.triton('triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr4': '*i32', 'out_ptr5': '*i32', 'out_ptr6': '*i32', 'out_ptr7': '*i32', 'out_ptr8': '*i32', 'out_ptr9': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_3', 'mutated_arg_names': ['out_ptr7', 'out_ptr9'], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_3(in_ptr0, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32
    r0_numel = 52
    R0_BLOCK: tl.constexpr = 64
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 52*x0), r0_mask & xmask, other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 > tmp1
    tmp3 = tl.full([1, 1], 16384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tmp5.to(tl.int8)
    tmp7 = tmp6.to(tl.int32)
    tmp8 = r0_1
    tmp9 = tmp8.to(tl.int16)
    tmp10 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp11 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12, tmp13, = triton_helpers.sort_with_index(tmp10, tmp11, rnumel, 1, stable=True, descending=True)
    tmp14 = tmp0 == tmp3
    tmp15 = tmp14.to(tl.int8)
    tmp16 = tmp15.to(tl.int32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp18, tmp19, = triton_helpers.sort_with_index(tmp17, tmp11, rnumel, 1, stable=True, descending=True)
    tmp20 = tmp7.to(tl.int64)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(r0_mask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None].to(tl.int64)
    tmp25 = tmp16.to(tl.int64)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.where(r0_mask & xmask, tmp26, 0)
    tmp29 = tl.sum(tmp28, 1)[:, None].to(tl.int64)
    tmp30 = tmp24.to(tl.int32)
    tmp31 = tmp29.to(tl.int32)
    tmp32 = tmp13.to(tl.int64)
    tmp33 = tmp32.to(tl.int32)
    tmp34 = tmp8 < tmp30
    tmp35 = tl.full([1, 1], 52, tl.int32)
    tmp36 = tl.where(tmp34, tmp33, tmp35)
    tmp37 = tl.full([XBLOCK, R0_BLOCK], 53, tl.int32)
    tmp38 = tmp36 + tmp37
    tmp39 = tmp36 < 0
    tmp40 = tl.where(tmp39, tmp38, tmp36)
    tl.device_assert(((0 <= tmp40) & (tmp40 < 53)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp40 < 53")
    tmp42 = tl.full([1, 1], 1, tl.int32)
    tmp43 = tmp19.to(tl.int64)
    tmp44 = tmp43.to(tl.int32)
    tmp45 = tmp8 < tmp31
    tmp46 = tl.where(tmp45, tmp44, tmp35)
    tmp47 = tmp46 + tmp37
    tmp48 = tmp46 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp46)
    tl.device_assert(((0 <= tmp49) & (tmp49 < 53)) | ~(r0_mask & xmask), "index out of bounds: 0 <= tmp49 < 53")
    tl.store(out_ptr4 + (x0), tmp30, xmask)
    tl.store(out_ptr5 + (x0), tmp31, xmask)
    tl.store(out_ptr6 + (r0_1 + 52*x0), tmp33, r0_mask & xmask)
    tl.store(out_ptr7 + (tl.broadcast_to(tmp40 + 53*x0, [XBLOCK, R0_BLOCK])), tmp42, r0_mask & xmask)
    tl.store(out_ptr8 + (r0_1 + 52*x0), tmp44, r0_mask & xmask)
    tl.store(out_ptr9 + (tl.broadcast_to(tmp49 + 53*x0, [XBLOCK, R0_BLOCK])), tmp42, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/phase_diagnosis_20261014/inductor-cache/rank_1/vq/cvqtexkew4phduoq2vwvz2tnbscksrzaj7fkq2zzfikw3aptexqs.py
# Topologically Sorted Source Nodes: [batched_outputs_4, transpose, col_indices_2, q_indices, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy, aten.sum]
# Source node to ATen node mapping:
#   batched_outputs_4 => clone_4, slice_2
#   col_indices_2 => sort_2
#   num_blocks_in_row_2 => sum_4
#   q_indices => clone_6, convert_element_type_9
#   q_num_blocks => convert_element_type_8
#   transpose => permute_1
# Graph fragment:
#   %buf10 : Tensor "i32[1, 1, 32, 53][1696, 1696, 53, 1]cuda:1" = PlaceHolder[target=buf10]
#   %buf12 : Tensor "i16[1, 1, 52, 32][1664, 1664, 32, 1]cuda:1" = PlaceHolder[target=buf12]
#   %sum_4 : Tensor "i64[1, 1, 52][52, 52, 1]cuda:1" = PlaceHolder[target=sum_4]
#   %slice_2 : Tensor "i32[1, 1, 32, 52][1696, 1696, 53, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put, 3, 0, 52), kwargs = {})
#   %clone_4 : Tensor "i32[1, 1, 32, 52][1664, 1664, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_2,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_1 : Tensor "i32[1, 1, 52, 32][1664, 1664, 1, 52]cuda:1"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_4, [0, 1, 3, 2]), kwargs = {})
#   %sort_2 : [num_users=1] = call_function[target=torch.ops.aten.sort.stable](args = (%permute_1,), kwargs = {stable: True, descending: True})
#   %convert_element_type_9 : Tensor "i32[1, 1, 52, 32][1664, 1664, 1, 52]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_5, torch.int32), kwargs = {})
#   %clone_6 : Tensor "i32[1, 1, 52, 32][1664, 1664, 32, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_9,), kwargs = {memory_format: torch.contiguous_format})
#   %sum_4 : Tensor "i64[1, 1, 52][52, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_1, [-1]), kwargs = {})
#   %convert_element_type_8 : Tensor "i32[1, 1, 52][52, 52, 1]cuda:1"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_4, torch.int32), kwargs = {})
#   return %buf12,%sum_4,%clone_6,%convert_element_type_8
triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4 = async_compile.triton('triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr2': '*i32', 'out_ptr3': '*i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 416, 'r0_': 13312}}
)
@triton.jit
def triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4(in_ptr0, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 52
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 53*r0_1), xmask, other=0.0)
    tmp1 = r0_1
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, None, 1, stable=True, descending=True)
    tmp7 = tmp0.to(tl.int64)
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.int64)
    tmp12 = tmp6.to(tl.int64)
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tmp11.to(tl.int32)
    tl.store(out_ptr2 + (r0_1 + 32*x0), tmp13, xmask)
    tl.store(out_ptr3 + (x0), tmp14, xmask)
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
        assert_size_stride(arg0_1, (1, 512), (512, 1))
        assert_size_stride(arg1_1, (1, 512), (512, 1))
        assert_size_stride(arg2_1, (1, 512, 1), (512, 1, 1))
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_cuda((1, 4096, 6656), (27262976, 6656, 1), torch.bool)
            # Topologically Sorted Source Nodes: [b, m, qb, safe_qb, n, lt, anchor, sub, ge, and_2, sub_1, le_1, history, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, or_1, is_q, sub_3, kv_qb, eq_1, and_7, sub_4, slot, safe_slot, lt_2, index_2, safe_kb, child_3, invert, padded_token, invert_1, query, or_2], Original ATen: [aten.arange, aten.floor_divide, aten.clamp, aten.lt, aten.view, aten.index, aten.sub, aten.ge, aten.bitwise_and, aten.le, aten.eq, aten.bitwise_or, aten.remainder, aten.bitwise_not]
            stream1 = get_raw_stream(1)
            triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0.run(arg1_1, arg2_1, buf0, 27262976, stream=stream1)
            del arg1_1
            del arg2_1
            buf1 = empty_strided_cuda((1, 1, 32, 52), (1664, 1664, 52, 1), torch.int64)
            # Topologically Sorted Source Nodes: [b, index, m, qb, safe_qb, le, valid_block, and_9, n, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, invert_2, and_10, le_2, and_11, batched_outputs, batched_outputs_3, mask_2, mask_3, mask_block_sum], Original ATen: [aten.arange, aten.view, aten.floor_divide, aten.clamp, aten.index, aten.le, aten.bitwise_and, aten.ge, aten.lt, aten.sub, aten.eq, aten.bitwise_not, aten.bitwise_or, aten.permute, aten.sum]
            stream1 = get_raw_stream(1)
            triton_red_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_permute_sub_sum_view_1.run(arg0_1, buf0, buf1, 1664, 16384, stream=stream1)
            del arg0_1
            del buf0
            buf16 = empty_strided_cuda((1, 1, 32, 53), (1696, 1696, 53, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
            stream1 = get_raw_stream(1)
            triton_poi_fused_new_zeros_2.run(buf16, 1696, stream=stream1)
            buf9 = empty_strided_cuda((1, 1, 32, 53), (1696, 1696, 53, 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
            stream1 = get_raw_stream(1)
            triton_poi_fused_new_zeros_2.run(buf9, 1696, stream=stream1)
            buf7 = empty_strided_cuda((1, 1, 32), (32, 32, 1), torch.int32)
            buf14 = empty_strided_cuda((1, 1, 32), (32, 32, 1), torch.int32)
            buf8 = empty_strided_cuda((1, 1, 32, 52), (1664, 1664, 52, 1), torch.int32)
            buf15 = empty_strided_cuda((1, 1, 32, 52), (1664, 1664, 52, 1), torch.int32)
            # Topologically Sorted Source Nodes: [gt, lt_3, partial_blocks, partial_blocks_1, dense_mask, col_indices, full_blocks, full_blocks_1, dense_mask_1, col_indices_1, dense_mask_2, setitem, arange_4, row_indices, col_range, num_blocks_in_row, child_4, unsqueeze_1, index_mask, child_5, valid_indices, dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, num_blocks_in_row_1, child_8, unsqueeze_3, index_mask_1, child_9, valid_indices_1], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort, aten.eq, aten.new_zeros, aten.arange, aten.unsqueeze, aten.sum, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
            stream1 = get_raw_stream(1)
            triton_per_fused__to_copy_arange_bitwise_and_eq_gt_index_put_lt_new_zeros_scalar_tensor_sort_sum_unsqueeze_view_where_3.run(buf1, buf7, buf14, buf8, buf9, buf15, buf16, 32, 52, stream=stream1)
            del buf1
            buf23 = empty_strided_cuda((1, 1, 52, 32), (1664, 1664, 32, 1), torch.int32)
            buf25 = empty_strided_cuda((1, 1, 52), (52, 52, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, col_indices_2, q_indices, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy, aten.sum]
            stream1 = get_raw_stream(1)
            triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4.run(buf9, buf23, buf25, 52, 32, stream=stream1)
            del buf9
            buf20 = empty_strided_cuda((1, 1, 52, 32), (1664, 1664, 32, 1), torch.int32)
            buf22 = empty_strided_cuda((1, 1, 52), (52, 52, 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, col_indices_3, full_q_indices, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort, aten._to_copy, aten.sum]
            stream1 = get_raw_stream(1)
            triton_per_fused__to_copy_clone_slice_sort_sum_transpose_4.run(buf16, buf20, buf22, 52, 32, stream=stream1)
            del buf16
        return (buf20, buf22, buf23, buf25, buf15, buf14, buf8, buf7, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 512), (512, 1), device='cuda:1', dtype=torch.bool)
    arg1_1 = rand_strided((1, 512), (512, 1), device='cuda:1', dtype=torch.int64)
    arg2_1 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:1', dtype=torch.bool)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
