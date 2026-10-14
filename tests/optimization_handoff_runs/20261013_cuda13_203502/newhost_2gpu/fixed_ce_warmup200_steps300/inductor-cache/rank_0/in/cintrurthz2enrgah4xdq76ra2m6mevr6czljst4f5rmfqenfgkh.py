# AOT ID: ['2_inference']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/fr/cfre3l4wggrjkwsqnj4qyeiipoz46zg6wcn2urn4tra6jibj5dbr.py
# Topologically Sorted Source Nodes: [b, m, qb, safe_qb, n, lt, anchor, sub, ge, and_2, sub_1, le_1, history, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, or_1, is_q, sub_3, kv_qb, eq_1, and_7, sub_4, slot, safe_slot, lt_2, index_2, safe_kb, child_3, invert, padded_token, invert_1, query, or_2], Original ATen: [aten.arange, aten.floor_divide, aten.clamp, aten.lt, aten.view, aten.index, aten.sub, aten.ge, aten.bitwise_and, aten.le, aten.eq, aten.bitwise_or, aten.remainder, aten.bitwise_not]
# Source node to ATen node mapping:
#   anchor => index_1, view_1
#   and_2 => bitwise_and_1
#   and_7 => bitwise_and_6
#   b => iota
#   child_3 => clamp_max_3
#   chs_valid => bitwise_and_4
#   eq => eq_27, view_4
#   eq_1 => eq_45, view_6
#   floor_divide_1 => div_1
#   ge => ge_2, view_2
#   ge_1 => ge_3
#   history => bitwise_and_2
#   index_2 => index_2, view_5
#   invert => bitwise_not
#   invert_1 => bitwise_not_1
#   is_chs => bitwise_and_3
#   is_q => ge_4
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
#   sub => sub_13
#   sub_1 => sub_20
#   sub_2 => sub_30
#   sub_3 => sub_39
#   sub_4 => sub_42
# Graph fragment:
#   %arg6_1 : Tensor "i64[1, s96][s96, 1]cuda:0" = PlaceHolder[target=arg6_1]
#   %arg9_1 : Tensor "b8[1, s95, 1][s95, 1, 1]cuda:0" = PlaceHolder[target=arg9_1]
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %iota_2 : Tensor "i64[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %div : Tensor "i64[s12][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 8), kwargs = {rounding_mode: floor})
#   %clamp_min : Tensor "i64[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 0), kwargs = {})
#   %clamp_max : Tensor "i64[s12][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, %arg2_1), kwargs = {})
#   %iota_3 : Tensor "i64[s37][1]cuda:0"[num_users=9] = call_function[target=torch.ops.prims.iota.default](args = (%arg1_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %lt : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, 0), kwargs = {})
#   %view_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %index_1 : Tensor "i64[1, s12][Max(1, s12), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg6_1, [%view_1, %clamp_max]), kwargs = {})
#   %sub_13 : Tensor "i64[1, s12][Max(1, s12), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_1, 1), kwargs = {})
#   %view_2 : Tensor "i64[1, s12, 1][Max(1, s12), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_13, [1, %arg0_1, 1]), kwargs = {})
#   %ge_2 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Tensor](args = (%iota_3, %view_2), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt, %ge_2), kwargs = {})
#   %sub_20 : Tensor "i64[1, s12][Max(1, s12), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%index_1, 2), kwargs = {})
#   %view_3 : Tensor "i64[1, s12, 1][Max(1, s12), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_20, [1, %arg0_1, 1]), kwargs = {})
#   %le_1 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%iota_3, %view_3), kwargs = {})
#   %bitwise_and_2 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_1, %le_1), kwargs = {})
#   %ge_3 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, 0), kwargs = {})
#   %lt_1 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, %arg7_1), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_3, %lt_1), kwargs = {})
#   %sub_30 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 0), kwargs = {})
#   %div_1 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_30, 5), kwargs = {rounding_mode: floor})
#   %view_4 : Tensor "i64[s12, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [%arg0_1, 1]), kwargs = {})
#   %eq_27 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_1, %view_4), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %eq_27), kwargs = {})
#   %bitwise_or : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_2, %bitwise_and_4), kwargs = {})
#   %ge_4 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, %arg7_1), kwargs = {})
#   %sub_39 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, %arg7_1), kwargs = {})
#   %div_2 : Tensor "i64[s37][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_39, 8), kwargs = {rounding_mode: floor})
#   %view_6 : Tensor "i64[s12, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [%arg0_1, 1]), kwargs = {})
#   %eq_45 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_2, %view_6), kwargs = {})
#   %bitwise_and_6 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_4, %eq_45), kwargs = {})
#   %sub_42 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, %arg7_1), kwargs = {})
#   %remainder : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.remainder.Scalar](args = (%sub_42, 8), kwargs = {})
#   %clamp_min_2 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%remainder, 0), kwargs = {})
#   %clamp_max_2 : Tensor "i64[s37][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 7), kwargs = {})
#   %lt_2 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%clamp_max_2, 1), kwargs = {})
#   %view_5 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %clamp_min_1 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_2, 0), kwargs = {})
#   %clamp_max_1 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, %arg2_1), kwargs = {})
#   %clamp_max_3 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_max_2, 0), kwargs = {})
#   %index_2 : Tensor "b8[1, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg9_1, [%view_5, %clamp_max_1, %clamp_max_3]), kwargs = {})
#   %bitwise_not : Tensor "b8[1, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%index_2,), kwargs = {})
#   %bitwise_and_5 : Tensor "b8[1, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%lt_2, %bitwise_not), kwargs = {})
#   %bitwise_not_1 : Tensor "b8[1, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and_5,), kwargs = {})
#   %view_7 : Tensor "b8[1, 1, s37][Max(1, s37), s37, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_not_1, [1, 1, %arg1_1]), kwargs = {})
#   %bitwise_and_7 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_6, %view_7), kwargs = {})
#   %bitwise_or_1 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_or, %bitwise_and_7), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = triton_helpers.div_floor_integer(xindex,  ks0)
    x2 = xindex
    tl.device_assert(((0 <= tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))) < 0, ks2 + ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))))) & (tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))) < 0, ks2 + ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1)))) < ks2)) | ~(xmask), "index out of bounds: 0 <= tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))) < 0, ks2 + ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1)))) < ks2")
    tmp4 = tl.load(in_ptr0 + (tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))) < 0, ks2 + ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks1))))), xmask, eviction_policy='evict_last')
    tl.device_assert(((0 <= tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))) < 0, ks4 + ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))))) & (tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))) < 0, ks4 + ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1)))) < ks4)) | ~(xmask), "index out of bounds: 0 <= tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))) < 0, ks4 + ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1)))) < ks4")
    tmp43 = tl.load(in_ptr1 + (tl.where(((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))) < 0, ks4 + ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))), ((ks1) * ((ks1) <= (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8))) + (triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) * ((triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)) > (0)))) < (ks1))))), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 - tmp5
    tmp7 = tmp0 >= tmp6
    tmp8 = tmp2 & tmp7
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp4 - tmp9
    tmp11 = tmp0 <= tmp10
    tmp12 = tmp8 & tmp11
    tmp13 = tmp0 >= tmp1
    tmp14 = ks3
    tmp15 = tmp0 < tmp14
    tmp16 = tmp13 & tmp15
    tmp17 = x0 // 5
    tmp18 = x1 // 8
    tmp19 = tmp17 == tmp18
    tmp20 = tmp16 & tmp19
    tmp21 = tmp12 | tmp20
    tmp22 = tmp0 >= tmp14
    tmp23 = triton_helpers.div_floor_integer(x0 + ((-1)*ks3),  8)
    tmp24 = tmp23 == tmp18
    tmp25 = tmp22 & tmp24
    tmp26 = x0 + ((-1)*ks3)
    tmp27 = tl.full([1], 8, tl.int64)
    tmp28 = (tmp26 % tmp27)
    tmp29 = tl.full([1], 0, tl.int32)
    tmp30 = tmp28 != tmp29
    tmp31 = (libdevice.signbit(tmp28) != 0) if (tmp28).dtype is tl.float32 else tmp28 < 0
    tmp32 = (libdevice.signbit(tmp27) != 0) if (tmp27).dtype is tl.float32 else tmp27 < 0
    tmp33 = tmp31 != tmp32
    tmp34 = tmp30 & tmp33
    tmp35 = tmp28 + tmp27
    tmp36 = tl.where(tmp34, tmp35, tmp28)
    tmp37 = triton_helpers.maximum(tmp36, tmp1)
    tmp38 = tl.full([1], 7, tl.int64)
    tmp39 = triton_helpers.minimum(tmp37, tmp38)
    tmp40 = tmp39 < tmp5
    tmp42 = triton_helpers.minimum(tmp39, tmp1)
    tmp44 = tmp43 == 0
    tmp45 = tmp40 & tmp44
    tmp46 = tmp45 == 0
    tmp47 = tmp25 & tmp46
    tmp48 = tmp21 | tmp47
    tl.store(out_ptr0 + (x2), tmp48, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/nu/cnuhgx7idvgo4mo27djteqddovhluia3exw7tqcspyruzdk4dh57.py
# Topologically Sorted Source Nodes: [b, index, m, qb, safe_qb, le, valid_block, and_9, n, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, invert_2, and_10, le_2, and_11, batched_outputs, batched_outputs_3, mask_1], Original ATen: [aten.arange, aten.view, aten.floor_divide, aten.clamp, aten.index, aten.le, aten.bitwise_and, aten.ge, aten.lt, aten.sub, aten.eq, aten.bitwise_not, aten.bitwise_or, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   and_10 => bitwise_and_9, view_9
#   and_11 => bitwise_and_10, view_10
#   and_9 => bitwise_and_8, view_8
#   b => iota
#   batched_outputs => bitwise_or_2
#   batched_outputs_3 => view_11
#   chs_valid => bitwise_and_4
#   eq => eq_27, view_4
#   floor_divide_1 => div_1
#   ge_1 => ge_3
#   index => index, view
#   invert_2 => bitwise_not_2
#   is_chs => bitwise_and_3
#   le => le
#   le_2 => le_2
#   lt_1 => lt_1
#   m => iota_2
#   mask_1 => constant_pad_nd
#   n => iota_3
#   qb => div
#   safe_qb => clamp_max, clamp_min
#   sub_2 => sub_30
#   valid_block => bitwise_and
# Graph fragment:
#   %arg4_1 : Tensor "b8[1, s94][s94, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %bitwise_or_1 : Tensor "b8[1, s12, s37][s12*s37, s37, 1]cuda:0" = PlaceHolder[target=bitwise_or_1]
#   %iota : Tensor "i64[1][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [1, 1]), kwargs = {})
#   %iota_2 : Tensor "i64[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%arg0_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %div : Tensor "i64[s12][1]cuda:0"[num_users=5] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%iota_2, 8), kwargs = {rounding_mode: floor})
#   %clamp_min : Tensor "i64[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div, 0), kwargs = {})
#   %clamp_max : Tensor "i64[s12][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, %arg2_1), kwargs = {})
#   %index : Tensor "b8[1, s12][Max(1, s12), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%arg4_1, [%view, %clamp_max]), kwargs = {})
#   %le : Tensor "b8[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%div, %arg2_1), kwargs = {})
#   %bitwise_and : Tensor "b8[1, s12][Max(1, s12), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%index, %le), kwargs = {})
#   %view_8 : Tensor "b8[1, s12, 1][Max(1, s12), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and, [1, %arg0_1, 1]), kwargs = {})
#   %iota_3 : Tensor "i64[s37][1]cuda:0"[num_users=9] = call_function[target=torch.ops.prims.iota.default](args = (%arg1_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %ge_3 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.ge.Scalar](args = (%iota_3, 0), kwargs = {})
#   %lt_1 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%iota_3, %arg7_1), kwargs = {})
#   %bitwise_and_3 : Tensor "b8[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%ge_3, %lt_1), kwargs = {})
#   %sub_30 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%iota_3, 0), kwargs = {})
#   %div_1 : Tensor "i64[s37][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Tensor_mode](args = (%sub_30, 5), kwargs = {rounding_mode: floor})
#   %view_4 : Tensor "i64[s12, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%div, [%arg0_1, 1]), kwargs = {})
#   %eq_27 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%div_1, %view_4), kwargs = {})
#   %bitwise_and_4 : Tensor "b8[s12, s37][Max(1, s37), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_3, %eq_27), kwargs = {})
#   %bitwise_and_8 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%view_8, %bitwise_or_1), kwargs = {})
#   %bitwise_not_2 : Tensor "b8[1, s12][Max(1, s12), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_not.default](args = (%bitwise_and,), kwargs = {})
#   %view_9 : Tensor "b8[1, s12, 1][Max(1, s12), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_not_2, [1, %arg0_1, 1]), kwargs = {})
#   %bitwise_and_9 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_4, %view_9), kwargs = {})
#   %le_2 : Tensor "b8[s12][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%div, %arg2_1), kwargs = {})
#   %view_10 : Tensor "b8[s12, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%le_2, [%arg0_1, 1]), kwargs = {})
#   %bitwise_and_10 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and_9, %view_10), kwargs = {})
#   %bitwise_or_2 : Tensor "b8[1, s12, s37][Max(1, s12)*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_or.Tensor](args = (%bitwise_and_8, %bitwise_and_10), kwargs = {})
#   %view_11 : Tensor "b8[1, 1, s12, s37][Max(1, s12)*Max(1, s37), s12*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_or_2, [1, 1, %arg0_1, %arg1_1]), kwargs = {})
#   %constant_pad_nd : Tensor "b8[1, 1, 128*(((s12 + 127)//128)), 128*(((s37 + 127)//128))][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s37 + 127)//128))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%expand, [0, %sub_86, 0, %sub_88], 0.0), kwargs = {})
#   return %constant_pad_nd
triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1 = async_compile.triton('triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'ks5': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = x1
    tmp1 = ks1
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = ks2
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tl.device_assert(((0 <= tl.where(((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))) < 0, ks4 + ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))), ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))))) & (tl.where(((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))) < 0, ks4 + ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))), ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3)))) < ks4)) | ~(tmp6), "index out of bounds: 0 <= tl.where(((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))) < 0, ks4 + ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))), ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3)))) < ks4")
    tmp8 = tl.load(in_ptr0 + (tl.where(((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))) < 0, ks4 + ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))), ((ks3) * ((ks3) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (ks3))))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp9 = x1 // 8
    tmp10 = tl.broadcast_to(ks3, [XBLOCK])
    tmp11 = tmp9 <= tmp10
    tmp12 = tmp8 & tmp11
    tmp13 = tl.load(in_ptr1 + (x0 + ks2*x1), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp14 = tmp12 & tmp13
    tmp15 = x0
    tmp16 = tl.full([1], 0, tl.int64)
    tmp17 = tmp15 >= tmp16
    tmp18 = tl.broadcast_to(ks5, [XBLOCK])
    tmp19 = tmp15 < tmp18
    tmp20 = tmp17 & tmp19
    tmp21 = x0 // 5
    tmp22 = tmp21 == tmp9
    tmp23 = tmp20 & tmp22
    tmp24 = tmp12 == 0
    tmp25 = tmp23 & tmp24
    tmp26 = tmp25 & tmp11
    tmp27 = tmp14 | tmp26
    tmp28 = tl.full(tmp27.shape, False, tmp27.dtype)
    tmp29 = tl.where(tmp6, tmp27, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, None)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/yf/cyfblgk4afgjlgm7n5ee24qzuefsrkf4yyfuk2bfvwukmokqcg6z.py
# Topologically Sorted Source Nodes: [mask_2, mask_3, mask_block_sum, gt, lt_3, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.view, aten.permute, aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.eq]
# Source node to ATen node mapping:
#   dense_mask => convert_element_type_2
#   dense_mask_1 => convert_element_type_5
#   full_blocks => eq_86
#   full_blocks_1 => convert_element_type_1
#   gt => gt
#   lt_3 => lt_3
#   mask_2 => view_12
#   mask_3 => permute
#   mask_block_sum => sum_1
#   partial_blocks => bitwise_and_11
#   partial_blocks_1 => convert_element_type
# Graph fragment:
#   %constant_pad_nd : Tensor "b8[1, 1, 128*(((s12 + 127)//128)), 128*(((s37 + 127)//128))][16384*(((s12 + 127)//128))*(((s37 + 127)//128)), 16384*(((s12 + 127)//128))*(((s37 + 127)//128)), 128*(((s37 + 127)//128)), 1]cuda:0" = PlaceHolder[target=constant_pad_nd]
#   %sum_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][(((s12 + 127)//128))*(((s37 + 127)//128)), (((s12 + 127)//128))*(((s37 + 127)//128)), ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=sum_1]
#   %view_12 : Tensor "b8[1, 1, ((s12 + 127)//128), 128, ((s37 + 127)//128), 128][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), 128*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s37 + 127)//128))), 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%constant_pad_nd, [1, 1, %floordiv_3, 128, %floordiv_2, 128]), kwargs = {})
#   %permute : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128), 128, 128][Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), Max(1, 128*(((s12 + 127)//128)))*Max(1, 128*(((s37 + 127)//128))), 128*Max(1, 128*(((s37 + 127)//128))), 128, Max(1, 128*(((s37 + 127)//128))), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 1, 2, 4, 3, 5]), kwargs = {})
#   %sum_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute, [-2, -1]), kwargs = {})
#   %gt : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%sum_1, 0), kwargs = {})
#   %lt_3 : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %bitwise_and_11 : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%gt, %lt_3), kwargs = {})
#   %convert_element_type : Tensor "i8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%bitwise_and_11, torch.int8), kwargs = {})
#   %convert_element_type_2 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type, torch.int32), kwargs = {})
#   %eq_86 : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%sum_1, 16384), kwargs = {})
#   %convert_element_type_1 : Tensor "i8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%eq_86, torch.int8), kwargs = {})
#   %convert_element_type_5 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_1, torch.int32), kwargs = {})
#   return %sum_1,%convert_element_type_2,%convert_element_type_5
triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2 = async_compile.triton('triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'out_ptr1': '*i32', 'out_ptr2': '*i32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2(in_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    x4 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 128)
        r0_3 = r0_index // 128
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x0 + 128*r0_3*(triton_helpers.div_floor_integer(127 + ks1,  128)) + 16384*x1*(triton_helpers.div_floor_integer(127 + ks1,  128))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.int1)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tl.full([1, 1], 0, tl.int64)
    tmp6 = tmp3 > tmp5
    tmp7 = tl.full([1, 1], 16384, tl.int64)
    tmp8 = tmp3 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tmp9.to(tl.int8)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = tmp3 == tmp7
    tmp13 = tmp12.to(tl.int8)
    tmp14 = tmp13.to(tl.int32)
    tl.store(out_ptr1 + (x4), tmp11, xmask)
    tl.store(out_ptr2 + (x4), tmp14, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/vw/cvwqrjcfrhaexr3d3nljmo5huyfbhkjkobj6d55nhhjbedme5hht.py
# Topologically Sorted Source Nodes: [num_blocks_in_row, child_4], Original ATen: [aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   child_4 => convert_element_type_3
#   num_blocks_in_row => sum_2
# Graph fragment:
#   %convert_element_type_2 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][(((s12 + 127)//128))*(((s37 + 127)//128)), (((s12 + 127)//128))*(((s37 + 127)//128)), ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=convert_element_type_2]
#   %sum_2 : Tensor "i64[1, 1, ((s12 + 127)//128)][((s12 + 127)//128), ((s12 + 127)//128), 1]cuda:0" = PlaceHolder[target=sum_2]
#   %sum_2 : Tensor "i64[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_2, [-1]), kwargs = {})
#   %convert_element_type_3 : Tensor "i32[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_2, torch.int32), kwargs = {})
#   return %sum_2,%convert_element_type_3
triton_red_fused__to_copy_sum_3 = async_compile.triton('triton_red_fused__to_copy_sum_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr1': '*i32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_sum_3(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tmp3.to(tl.int32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/t3/ct3yzvjsw2ndah4modyncl63g2xa6v2og7bexgybgzktn2a47pyw.py
# Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   dense_mask_2 => full_default
# Graph fragment:
#   %full_default : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, %floordiv_3, %add_212], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   return %index_put
triton_poi_fused_new_zeros_4 = async_compile.triton('triton_poi_fused_new_zeros_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/b2/cb2li3wkwtnljrqyfepuxwp3yjhu33fl6n4jahms2jnjeqnn76xj.py
# Topologically Sorted Source Nodes: [dense_mask_2, setitem, arange_4, row_indices, col_range, unsqueeze_1, index_mask, child_5, valid_indices], Original ATen: [aten.new_zeros, aten.arange, aten.unsqueeze, aten.lt, aten._to_copy, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
# Source node to ATen node mapping:
#   arange_4 => iota_4
#   child_5 => convert_element_type_4
#   col_range => iota_5
#   dense_mask_2 => full_default
#   index_mask => lt_4
#   row_indices => unsqueeze
#   setitem => full_default_1, index_put, iota_6, iota_7, unsqueeze_2, unsqueeze_3, unsqueeze_4, unsqueeze_5, unsqueeze_6
#   unsqueeze_1 => unsqueeze_1
#   valid_indices => scalar_tensor, where
# Graph fragment:
#   %getitem_1 : Tensor "i64[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %convert_element_type_3 : Tensor "i32[1, 1, ((s12 + 127)//128)][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0" = PlaceHolder[target=convert_element_type_3]
#   %convert_element_type_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0" = PlaceHolder[target=convert_element_type_4]
#   %index_put : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][((((s37 + 127)//128)) + 1)*(((s12 + 127)//128)), ((((s37 + 127)//128)) + 1)*(((s12 + 127)//128)), (((s37 + 127)//128)) + 1, 1]cuda:0" = PlaceHolder[target=index_put]
#   %full_default : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, %floordiv_3, %add_212], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %iota_7 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_4 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_7, -1), kwargs = {})
#   %unsqueeze_5 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_4, -1), kwargs = {})
#   %unsqueeze_6 : Tensor "i64[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_5, -1), kwargs = {})
#   %iota_6 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_6, -1), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, 1][1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze_2, -1), kwargs = {})
#   %iota_4 : Tensor "i32[((s12 + 127)//128)][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_3,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i32[((s12 + 127)//128), 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota_4, -1), kwargs = {})
#   %iota_5 : Tensor "i32[((s37 + 127)//128)][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (%floordiv_2,), kwargs = {start: 0, step: 1, dtype: torch.int32, device: cuda:0, requires_grad: False})
#   %unsqueeze_1 : Tensor "i32[1, 1, ((s12 + 127)//128), 1][Max(1, ((s12 + 127)//128)), Max(1, ((s12 + 127)//128)), 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 3), kwargs = {})
#   %lt_4 : Tensor "b8[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%iota_5, %unsqueeze_1), kwargs = {})
#   %convert_element_type_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_1, torch.int32), kwargs = {})
#   %scalar_tensor : Tensor "i32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.scalar_tensor.default](args = (%floordiv_2,), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0})
#   %where : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt_4, %convert_element_type_4, %scalar_tensor), kwargs = {})
#   %full_default_1 : Tensor "i32[1, 1, 1, 1][1, 1, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default, [%unsqueeze_6, %unsqueeze_3, %unsqueeze, %where], %full_default_1), kwargs = {})
#   return %convert_element_type_4,%buf15
triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5 = async_compile.triton('triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i32', 'out_ptr0': '*i32', 'out_ptr1': '*i32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x0 + x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tmp3 = x0
    tmp4 = tmp3 < tmp2
    tmp5 = ks0
    tmp6 = tl.where(tmp4, tmp1, tmp5)
    tmp7 = 1 + ks0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tl.device_assert(((0 <= tmp10) & (tmp10 < 1 + (triton_helpers.div_floor_integer(127 + ks1,  128)))) | ~(xmask), "index out of bounds: 0 <= tmp10 < 1 + (triton_helpers.div_floor_integer(127 + ks1,  128))")
    tmp12 = tl.full([1], 1, tl.int32)
    tl.store(out_ptr0 + (x0 + x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), tmp1, xmask)
    tl.store(out_ptr1 + (tmp10 + x1 + ks0*x1), tmp12, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/l7/cl7o5wxrn442s4x5h7unzxpm6ag7miucphcy6ojxgzmekawrvtzt.py
# Topologically Sorted Source Nodes: [batched_outputs_4], Original ATen: [aten.slice, aten.clone]
# Source node to ATen node mapping:
#   batched_outputs_4 => clone_4, slice_4
# Graph fragment:
#   %buf15 : Tensor "i32[1, 1, ((s12 + 127)//128), (((s37 + 127)//128)) + 1][((((s37 + 127)//128)) + 1)*(((s12 + 127)//128)), ((((s37 + 127)//128)) + 1)*(((s12 + 127)//128)), (((s37 + 127)//128)) + 1, 1]cuda:0" = PlaceHolder[target=buf15]
#   %slice_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put, 3, 0, %floordiv_2), kwargs = {})
#   %clone_4 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_4,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_4
triton_poi_fused_clone_slice_6 = async_compile.triton('triton_poi_fused_clone_slice_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr0': '*i32', 'ks0': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_slice_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_slice_6(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + x1 + ks0*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/dt/cdtkdtdwokfxuophlugl5kyht2wynt7n7cj7oynreletkaq36z4s.py
# Topologically Sorted Source Nodes: [full_q_indices], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   full_q_indices => clone_9, convert_element_type_11
# Graph fragment:
#   %getitem_7 : Tensor "i64[1, 1, ((s37 + 127)//128), ((s12 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), 1, Max(1, ((s37 + 127)//128))]cuda:0" = PlaceHolder[target=getitem_7]
#   %convert_element_type_11 : Tensor "i32[1, 1, ((s37 + 127)//128), ((s12 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), 1, Max(1, ((s37 + 127)//128))]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_7, torch.int32), kwargs = {})
#   %clone_9 : Tensor "i32[1, 1, ((s37 + 127)//128), ((s12 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_11,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_9
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i32', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x1 + x0*((1) * ((1) >= (ks1)) + (ks1) * ((ks1) > (1)))), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int32)
    tl.store(out_ptr0 + (x0 + x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), tmp1, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/pt/cptsn2fdfjznho5p3gaaxttzmbfode4h3ppkidbckzvggfivrts2.py
# Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
# Source node to ATen node mapping:
#   batched_outputs_6 => clone_7, slice_8
#   full_q_num_blocks => convert_element_type_10
#   num_blocks_in_row_3 => sum_5
#   transpose_1 => permute_2
# Graph fragment:
#   %clone_7 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][(((s12 + 127)//128))*(((s37 + 127)//128)), 1, ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=clone_7]
#   %sum_5 : Tensor "i64[1, 1, ((s37 + 127)//128)][((s37 + 127)//128), ((s37 + 127)//128), 1]cuda:0" = PlaceHolder[target=sum_5]
#   %slice_8 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1)*Max(1, ((s12 + 127)//128)), Max(1, (((s37 + 127)//128)) + 1), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%index_put_1, 3, 0, %floordiv_2), kwargs = {})
#   %clone_7 : Tensor "i32[1, 1, ((s12 + 127)//128), ((s37 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_8,), kwargs = {memory_format: torch.contiguous_format})
#   %permute_2 : Tensor "i32[1, 1, ((s37 + 127)//128), ((s12 + 127)//128)][Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), Max(1, ((s12 + 127)//128))*Max(1, ((s37 + 127)//128)), 1, Max(1, ((s37 + 127)//128))]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%clone_7, [0, 1, 3, 2]), kwargs = {})
#   %sum_5 : Tensor "i64[1, 1, ((s37 + 127)//128)][Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%permute_2, [-1]), kwargs = {})
#   %convert_element_type_10 : Tensor "i32[1, 1, ((s37 + 127)//128)][Max(1, ((s37 + 127)//128)), Max(1, ((s37 + 127)//128)), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sum_5, torch.int32), kwargs = {})
#   return %sum_5,%convert_element_type_10
triton_red_fused__to_copy_clone_slice_sum_transpose_8 = async_compile.triton('triton_red_fused__to_copy_clone_slice_sum_transpose_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r0_': 32},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i32', 'out_ptr1': '*i32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_clone_slice_sum_transpose_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_clone_slice_sum_transpose_8(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.int64)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tmp5 = tmp3.to(tl.int32)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
        args.clear()
        s12 = arg0_1
        s37 = arg1_1
        s68 = arg2_1
        s50 = arg3_1
        s96 = arg5_1
        s39 = arg7_1
        s95 = arg8_1
        arg4_1_size = arg4_1.size()
        s94 = arg4_1_size[1]
        assert_size_stride(arg4_1, (1, s94), (s94, 1))
        assert_size_stride(arg6_1, (1, s96), (s96, 1))
        assert_size_stride(arg9_1, (1, s95, 1), (s95, 1, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, s12, s37), (s12*s37, s37, 1), torch.bool)
            # Topologically Sorted Source Nodes: [b, m, qb, safe_qb, n, lt, anchor, sub, ge, and_2, sub_1, le_1, history, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, or_1, is_q, sub_3, kv_qb, eq_1, and_7, sub_4, slot, safe_slot, lt_2, index_2, safe_kb, child_3, invert, padded_token, invert_1, query, or_2], Original ATen: [aten.arange, aten.floor_divide, aten.clamp, aten.lt, aten.view, aten.index, aten.sub, aten.ge, aten.bitwise_and, aten.le, aten.eq, aten.bitwise_or, aten.remainder, aten.bitwise_not]
            triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0_xnumel = s12*s37
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0.run(arg6_1, arg9_1, buf0, s37, s68, s96, s39, s95, triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0_xnumel, stream=stream0)
            del arg6_1
            del arg9_1
            ps0 = 128*((127 + s37) // 128)
            buf1 = empty_strided_cuda((1, 1, 128*((127 + s12) // 128), 128*((127 + s37) // 128)), (16384*((127 + s12) // 128)*((127 + s37) // 128), 16384*((127 + s12) // 128)*((127 + s37) // 128), 128*((127 + s37) // 128), 1), torch.bool)
            # Topologically Sorted Source Nodes: [b, index, m, qb, safe_qb, le, valid_block, and_9, n, ge_1, lt_1, is_chs, sub_2, floor_divide_1, eq, chs_valid, invert_2, and_10, le_2, and_11, batched_outputs, batched_outputs_3, mask_1], Original ATen: [aten.arange, aten.view, aten.floor_divide, aten.clamp, aten.index, aten.le, aten.bitwise_and, aten.ge, aten.lt, aten.sub, aten.eq, aten.bitwise_not, aten.bitwise_or, aten.constant_pad_nd]
            triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1_xnumel = 16384*((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1.run(arg4_1, buf0, buf1, ps0, s12, s37, s68, s94, s39, triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1_xnumel, stream=stream0)
            del arg4_1
            del buf0
            ps1 = (127 + s37) // 128
            buf3 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), ((127 + s12) // 128)*((127 + s37) // 128), (127 + s37) // 128, 1), torch.int32)
            buf7 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), ((127 + s12) // 128)*((127 + s37) // 128), (127 + s37) // 128, 1), torch.int32)
            # Topologically Sorted Source Nodes: [mask_2, mask_3, mask_block_sum, gt, lt_3, partial_blocks, partial_blocks_1, dense_mask, full_blocks, full_blocks_1, dense_mask_1], Original ATen: [aten.view, aten.permute, aten.sum, aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.eq]
            triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2.run(buf1, buf3, buf7, ps1, s37, triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_2_xnumel, 16384, stream=stream0)
            del buf1
            # Topologically Sorted Source Nodes: [gt, lt_3, partial_blocks, partial_blocks_1, dense_mask, col_indices], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort]
            buf4 = torch.ops.aten.sort.stable(buf3, stable=True, dim=3, descending=True)
            buf6 = buf4[1]
            assert_size_stride(buf6, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), 'torch.ops.aten.sort.stable')
            assert_alignment(buf6, 16, 'torch.ops.aten.sort.stable')
            del buf4
            # Topologically Sorted Source Nodes: [full_blocks, full_blocks_1, dense_mask_1, col_indices_1], Original ATen: [aten.eq, aten._to_copy, aten.sort]
            buf8 = torch.ops.aten.sort.stable(buf7, stable=True, dim=3, descending=True)
            buf10 = buf8[1]
            assert_size_stride(buf10, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), 'torch.ops.aten.sort.stable')
            assert_alignment(buf10, 16, 'torch.ops.aten.sort.stable')
            del buf8
            buf12 = empty_strided_cuda((1, 1, (127 + s12) // 128), (max(1, (127 + s12) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [num_blocks_in_row, child_4], Original ATen: [aten.sum, aten._to_copy]
            triton_red_fused__to_copy_sum_3_xnumel = (127 + s12) // 128
            triton_red_fused__to_copy_sum_3_r0_numel = (127 + s37) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_sum_3.run(buf3, buf12, ps1, triton_red_fused__to_copy_sum_3_xnumel, triton_red_fused__to_copy_sum_3_r0_numel, stream=stream0)
            buf14 = empty_strided_cuda((1, 1, (127 + s12) // 128, 1 + ((127 + s37) // 128)), (((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128), 1 + ((127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2], Original ATen: [aten.new_zeros]
            triton_poi_fused_new_zeros_4_xnumel = ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_4.run(buf14, triton_poi_fused_new_zeros_4_xnumel, stream=stream0)
            buf13 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_2, setitem, arange_4, row_indices, col_range, unsqueeze_1, index_mask, child_5, valid_indices], Original ATen: [aten.new_zeros, aten.arange, aten.unsqueeze, aten.lt, aten._to_copy, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
            triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5.run(buf6, buf12, buf13, buf14, ps1, s37, triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5_xnumel, stream=stream0)
            del buf6
            buf16 = reinterpret_tensor(buf3, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), 1, (127 + s37) // 128, 1), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [batched_outputs_4], Original ATen: [aten.slice, aten.clone]
            triton_poi_fused_clone_slice_6_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_6.run(buf14, buf16, ps1, triton_poi_fused_clone_slice_6_xnumel, stream=stream0)
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, col_indices_2], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort]
            buf17 = torch.ops.aten.sort.stable(reinterpret_tensor(buf16, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (0, 0, 1, (127 + s37) // 128), 0), stable=True, dim=3, descending=True)
            buf19 = buf17[1]
            assert_size_stride(buf19, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), 1, max(1, (127 + s37) // 128)), 'torch.ops.aten.sort.stable')
            assert_alignment(buf19, 16, 'torch.ops.aten.sort.stable')
            del buf17
            buf21 = empty_strided_cuda((1, 1, (127 + s12) // 128), (max(1, (127 + s12) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [num_blocks_in_row_1, child_8], Original ATen: [aten.sum, aten._to_copy]
            triton_red_fused__to_copy_sum_3_xnumel = (127 + s12) // 128
            triton_red_fused__to_copy_sum_3_r0_numel = (127 + s37) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_sum_3.run(buf7, buf21, ps1, triton_red_fused__to_copy_sum_3_xnumel, triton_red_fused__to_copy_sum_3_r0_numel, stream=stream0)
            buf23 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [dense_mask_4], Original ATen: [aten.new_zeros]
            triton_poi_fused_new_zeros_4_xnumel = ((127 + s12) // 128)*((127 + s37) // 128) + ((127 + s12) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_new_zeros_4.run(buf23, triton_poi_fused_new_zeros_4_xnumel, stream=stream0)
            buf22 = empty_strided_cuda((1, 1, (127 + s12) // 128, (127 + s37) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, unsqueeze_3, index_mask_1, child_9, valid_indices_1], Original ATen: [aten.new_zeros, aten.arange, aten.unsqueeze, aten.lt, aten._to_copy, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
            triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5.run(buf10, buf21, buf22, buf23, ps1, s37, triton_poi_fused__to_copy_arange_index_put_lt_new_zeros_scalar_tensor_unsqueeze_view_where_5_xnumel, stream=stream0)
            del buf10
            buf25 = reinterpret_tensor(buf7, (1, 1, (127 + s12) // 128, (127 + s37) // 128), (((127 + s12) // 128)*((127 + s37) // 128), 1, (127 + s37) // 128, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [batched_outputs_6], Original ATen: [aten.slice, aten.clone]
            triton_poi_fused_clone_slice_6_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused_clone_slice_6.run(buf23, buf25, ps1, triton_poi_fused_clone_slice_6_xnumel, stream=stream0)
            del buf23
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, col_indices_3], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sort]
            buf26 = torch.ops.aten.sort.stable(reinterpret_tensor(buf25, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (0, 0, 1, (127 + s37) // 128), 0), stable=True, dim=3, descending=True)
            buf28 = buf26[1]
            assert_size_stride(buf28, (1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), 1, max(1, (127 + s37) // 128)), 'torch.ops.aten.sort.stable')
            assert_alignment(buf28, 16, 'torch.ops.aten.sort.stable')
            del buf26
            ps2 = (127 + s12) // 128
            buf29 = empty_strided_cuda((1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [full_q_indices], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_7_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf28, buf29, ps2, ps1, triton_poi_fused__to_copy_7_xnumel, stream=stream0)
            del buf28
            buf31 = empty_strided_cuda((1, 1, (127 + s37) // 128), (max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_6, transpose_1, num_blocks_in_row_3, full_q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            triton_red_fused__to_copy_clone_slice_sum_transpose_8_xnumel = (127 + s37) // 128
            triton_red_fused__to_copy_clone_slice_sum_transpose_8_r0_numel = (127 + s12) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_slice_sum_transpose_8.run(buf25, buf31, ps1, triton_red_fused__to_copy_clone_slice_sum_transpose_8_xnumel, triton_red_fused__to_copy_clone_slice_sum_transpose_8_r0_numel, stream=stream0)
            del buf25
            buf32 = empty_strided_cuda((1, 1, (127 + s37) // 128, (127 + s12) // 128), (max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128)*max(1, (127 + s37) // 128), max(1, (127 + s12) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [q_indices], Original ATen: [aten._to_copy]
            triton_poi_fused__to_copy_7_xnumel = ((127 + s12) // 128)*((127 + s37) // 128)
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_7.run(buf19, buf32, ps2, ps1, triton_poi_fused__to_copy_7_xnumel, stream=stream0)
            del buf19
            buf34 = empty_strided_cuda((1, 1, (127 + s37) // 128), (max(1, (127 + s37) // 128), max(1, (127 + s37) // 128), 1), torch.int32)
            # Topologically Sorted Source Nodes: [batched_outputs_4, transpose, num_blocks_in_row_2, q_num_blocks], Original ATen: [aten.slice, aten.clone, aten.transpose, aten.sum, aten._to_copy]
            triton_red_fused__to_copy_clone_slice_sum_transpose_8_xnumel = (127 + s37) // 128
            triton_red_fused__to_copy_clone_slice_sum_transpose_8_r0_numel = (127 + s12) // 128
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_clone_slice_sum_transpose_8.run(buf16, buf34, ps1, triton_red_fused__to_copy_clone_slice_sum_transpose_8_xnumel, triton_red_fused__to_copy_clone_slice_sum_transpose_8_r0_numel, stream=stream0)
            del buf16
        return (buf29, buf31, buf32, buf34, buf22, buf21, buf13, buf12, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 4096
    arg1_1 = 6656
    arg2_1 = 511
    arg3_1 = 512
    arg4_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.bool)
    arg5_1 = 512
    arg6_1 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    arg7_1 = 2560
    arg8_1 = 512
    arg9_1 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.bool)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
