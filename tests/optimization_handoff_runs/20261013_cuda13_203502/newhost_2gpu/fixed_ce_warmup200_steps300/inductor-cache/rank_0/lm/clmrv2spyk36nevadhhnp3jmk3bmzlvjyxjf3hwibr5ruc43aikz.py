# AOT ID: ['3_backward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/2r/c2rnwjrboefygzqriishsdyk77tvnrha225ipaxd55v32tc4di6a.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
# Graph fragment:
#   %getitem : Tensor "bf16[1, 32, s37, 128][4096*s37, 128, 4096, 1]cuda:0" = PlaceHolder[target=getitem]
#   %tangents_1 : Tensor "bf16[1, 32, s37, 128][4096*Max(1, s37), 128*Max(1, s37), 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %buf0 : Tensor "bf16[1, 32, s37][32*s37, s37, 1]cuda:0" = PlaceHolder[target=buf0]
#   %full_default : Tensor "f32[1, 32, s37][32*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, %primals_10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_2, %primals_4, %primals_6, %getitem, %getitem_1, %tangents_1, %full_default, %fw_graph0, %joint_graph0, (%primals_10, %primals_11, %primals_13, %primals_9, %primals_23, %primals_26, %primals_28, %primals_31, %primals_33, %primals_36, 128, 128, %mask_graph0), 0.08838834764831845, {FORCE_USE_FLEX_ATTENTION: True, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_14, %primals_15, %primals_16, %primals_17, %primals_18, %primals_19, %primals_20, %primals_21)), kwargs = {})
#   return %buf0,%buf1
triton_red_fused_zeros_0 = async_compile.triton('triton_red_fused_zeros_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_zeros_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_zeros_0(in_ptr0, in_ptr1, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = triton_helpers.div_floor_integer(xindex,  ks0)
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x1 + 4096*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x0 + 128*x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = tmp4.to(tl.float32)
    tmp7 = 0.0
    tmp8 = tmp6 - tmp7
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_0/xh/cxhx4n4lrluwxvkglt4quncpi5voaqwsoeqytmwplfck6cjpa77h.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
# Graph fragment:
#   %primals_2 : Tensor "bf16[1, 32, s37, 128][4096*s37, 128, 4096, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "bf16[1, 8, s0, 128][1024*s0, 128, 1024, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_6 : Tensor "bf16[1, 8, s0, 128][1024*s0, 128, 1024, 1]cuda:0" = PlaceHolder[target=primals_6]
#   %getitem_1 : Tensor "f32[1, 32, s37][32*Max(1, s37), Max(1, s37), 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 32, s37][32*s37, s37, 1]cuda:0" = PlaceHolder[target=buf1]
#   %tangents_1 : Tensor "bf16[1, 32, s37, 128][4096*Max(1, s37), 128*Max(1, s37), 128, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %getitem_3 : Tensor "bf16[1, 32, s37, 128][4096*s37, 128, 4096, 1]cuda:0" = PlaceHolder[target=getitem_3]
#   %getitem_5 : Tensor "bf16[1, 8, s0, 128][1024*s0, 128, 1024, 1]cuda:0" = PlaceHolder[target=getitem_5]
#   %primals_13 : Tensor "i32[1, 1, s99][s99, s99, 1]cuda:0" = PlaceHolder[target=primals_13]
#   %primals_9 : Tensor "i32[1, 1, s22, s72][s22*s72, s22*s72, s72, 1]cuda:0" = PlaceHolder[target=primals_9]
#   %primals_28 : Tensor "i32[1, 1, s56][s56, s56, 1]cuda:0" = PlaceHolder[target=primals_28]
#   %primals_31 : Tensor "i32[1, 1, s84, s53][s53*s84, s53*s84, s53, 1]cuda:0" = PlaceHolder[target=primals_31]
#   %primals_23 : Tensor "i32[1, 1, s94][s94, s94, 1]cuda:0" = PlaceHolder[target=primals_23]
#   %primals_26 : Tensor "i32[1, 1, s28, s4][s28*s4, s28*s4, s4, 1]cuda:0" = PlaceHolder[target=primals_26]
#   %primals_33 : Tensor "i32[1, 1, s100][s100, s100, 1]cuda:0" = PlaceHolder[target=primals_33]
#   %primals_36 : Tensor "i32[1, 1, s5, s10][s10*s5, s10*s5, s10, 1]cuda:0" = PlaceHolder[target=primals_36]
#   %primals_16 : Tensor "b8[1, s89][s89, 1]cuda:0" = PlaceHolder[target=primals_16]
#   %primals_18 : Tensor "i64[1, s79][s79, 1]cuda:0" = PlaceHolder[target=primals_18]
#   %primals_21 : Tensor "b8[1, s45, 1][s45, 1, 1]cuda:0" = PlaceHolder[target=primals_21]
#   %full_default : Tensor "f32[1, 32, s37][32*Max(1, s37), Max(1, s37), 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 32, %primals_10], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %flex_attention_backward : [num_users=3] = call_function[target=torch.ops.higher_order.flex_attention_backward](args = (%primals_2, %primals_4, %primals_6, %getitem, %getitem_1, %tangents_1, %full_default, %fw_graph0, %joint_graph0, (%primals_10, %primals_11, %primals_13, %primals_9, %primals_23, %primals_26, %primals_28, %primals_31, %primals_33, %primals_36, 128, 128, %mask_graph0), 0.08838834764831845, {FORCE_USE_FLEX_ATTENTION: True, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_14, %primals_15, %primals_16, %primals_17, %primals_18, %primals_19, %primals_20, %primals_21)), kwargs = {})
#   return %getitem_4
triton_tem_fused_zeros_1 = async_compile.triton('triton_tem_fused_zeros_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_Q': '*bf16', 'arg_K': '*bf16', 'arg_V': '*bf16', 'arg_LSE': '*fp32', 'arg_DELTA': '*fp32', 'arg_DO': '*bf16', 'arg_DQ': '*bf16', 'arg_DV': '*bf16', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_Q_NUM_BLKS': '*i32', 'arg_Q_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'arg_FULL_Q_NUM_BLKS': '*i32', 'arg_FULL_Q_IDX': '*i32', 'in_ptr16': '*i1', 'in_ptr17': '*i64', 'in_ptr18': '*i1', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'ks5': 'i32', 'ks6': 'i32', 'ks7': 'i32', 'ks8': 'i32', 'ks9': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]], (15,): [['tt.divisibility', 16]], (16,): [['tt.divisibility', 16]], (17,): [['tt.divisibility', 16]], (18,): [['tt.divisibility', 16]], (19,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_zeros_1', 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'FORCE_USE_FLEX_ATTENTION': True, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False, 'FLOAT32_PRECISION': "'ieee'", 'IS_DIVISIBLE': False, 'SM_SCALE': 0.08838834764831845, 'GQA_SHARED_HEADS': 4, 'HAS_FULL_BLOCKS': True, 'QK_HEAD_DIM': 128, 'QK_HEAD_DIM_ROUNDED': 128, 'V_HEAD_DIM': 128, 'V_HEAD_DIM_ROUNDED': 128, 'SAFE_HEAD_DIM': True, 'BLOCK_M1': 64, 'BLOCK_N1': 128, 'BLOCK_M2': 128, 'BLOCK_N2': 64, 'SPARSE_Q_BLOCK_SIZE': 128, 'SPARSE_KV_BLOCK_SIZE': 128}},

)
@triton.jit
def triton_tem_fused_zeros_1(arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9):
    FORCE_USE_FLEX_ATTENTION : tl.constexpr = True
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 128
    BLOCK_M2 : tl.constexpr = 128
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    DELTA = arg_DELTA
    DO = arg_DO
    DQ = arg_DQ
    DV = arg_DV
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    Q_NUM_BLKS = arg_Q_NUM_BLKS
    Q_IDX = arg_Q_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX
    FULL_Q_NUM_BLKS = arg_FULL_Q_NUM_BLKS
    FULL_Q_IDX = arg_FULL_Q_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT*DO, axis=-1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    # (Modifiable) Performance tuning options
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # Q_NUM_BLKS: The number of Q blocks (that may or may not require masking) for each query.
    # Q_IDX: The indices of Q blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_Q_NUM_BLKS: The number of fully unmasked Q blocks (so we don't need masking) for each query.
    # FULL_Q_IDX: The indices of fully unmasked Q blocks (so we don't need masking) for each query.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qd = 4096*ks0, 128, 4096, 1
    stride_kz, stride_kh, stride_kn, stride_kd = 1024*ks1, 128, 1024, 1
    stride_vz, stride_vh, stride_vn, stride_vd = 1024*ks1, 128, 1024, 1
    stride_doz, stride_doh, stride_dom, stride_dod = 4096*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1))), 128*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1))), 128, 1

    stride_dqz, stride_dqh, stride_dqm, stride_dqd = 4096*ks0, 128, 4096, 1
    stride_dvz, stride_dvh, stride_dvm, stride_dvd = 1024*ks1, 128, 1024, 1

    ZQ = 1
    HQ = 32
    HKV = 8
    Q_LEN = ks0
    ZKV = 1
    KV_LEN = ks1

    MATMUL_PRECISION = Q.dtype.element_ty

    pid = tl.program_id(0).to(INDEX_DTYPE)
    NUM_KV_BLOCKS = tl.cdiv(KV_LEN, BLOCK_N1)
    NUM_Q_BLOCKS = tl.cdiv(Q_LEN, BLOCK_M2)

    off_zq = tl.program_id(1).to(INDEX_DTYPE) # q batch idx
    off_hkv = tl.program_id(2).to(INDEX_DTYPE) # kv head idx
    off_zkv = off_zq % ZKV # kv batch idx

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_zq % SPARSE_Z

    k_adj = (stride_kh * off_hkv + stride_kz * off_zkv).to(tl.int64)
    v_adj = (stride_vh * off_hkv + stride_vz * off_zkv).to(tl.int64)
    # first compute broadcasted dv of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
    # then reduce to dv of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
    dv_adj = (stride_dvh * off_hkv + stride_dvz * off_zq).to(tl.int64)

    # offset K, V, DV pointers for batch/kv-head
    K += k_adj
    V += v_adj
    DV += dv_adj

    RCP_LN2 = 1.44269504
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    if pid >= NUM_KV_BLOCKS:
        off_pid = pid - NUM_KV_BLOCKS
        # THIS BLOCK DOES DQ
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M2)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
        off_hq2 = off_pid // NUM_Q_BLOCKS + off_hkv * GQA_SHARED_HEADS
        start_m2_block = off_pid % NUM_Q_BLOCKS
        off_pid_mask = start_m2_block // SPARSE_Q_MULTIPLE
        stride_kv_num_blks_h = ks2
        stride_kv_idx_h = ks3*ks4
        stride_kv_idx_m = ks4

        sparse_idx_hq2 = off_hq2 % SPARSE_HQ
        sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq2

        sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + off_pid_mask
        sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + off_pid_mask * stride_kv_idx_m  # noqa: B950

        # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
        q_adj2 = (stride_qh * off_hq2 + stride_qz * off_zq).to(tl.int64)
        do_adj2 = (stride_doh * off_hq2 + stride_doz * off_zq).to(tl.int64)
        dq_adj2 = (stride_dqh * off_hq2 + stride_dqz * off_zq).to(tl.int64)
        off_chz2 = ((off_zq * HQ + off_hq2) * Q_LEN).to(tl.int64)

        Q2 = Q + q_adj2
        DO2 = DO + do_adj2
        # TODO: This does not work if DQ is not the same layout as Q (for example,
        # if Q is broadcasted)
        DQ2 = DQ + dq_adj2
        LSE2 = LSE + off_chz2
        DELTA2 = DELTA + off_chz2

        # dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM], dtype=tl.float32)
        dq = tl.zeros([BLOCK_M2, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_m2 = start_m2_block * BLOCK_M2
        offs_m2 = start_m2 + tl.arange(0, BLOCK_M2)

        # load Q and do: they stay in SRAM throughout the inner loop.
        q = load_checked_2d(Q2, offs_m2, offs_k, stride_qm, stride_qd, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)
        do = load_checked_2d(DO2, offs_m2, offs_v, stride_dom, stride_dod, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        if IS_DIVISIBLE:
            Di = tl.load(DELTA2 + offs_m2)
            lse = tl.load(LSE2 + offs_m2)
        else:
            Di = tl.load(DELTA2 + offs_m2, mask=offs_m2 < Q_LEN)
            lse = tl.load(LSE2 + offs_m2, mask=offs_m2 < Q_LEN)
        lse = tl.where(lse == -float("inf"), 0.0, lse)
        lse = lse[:, None]

        # ~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # KV_IDX and KV_NUM_BLKS are always contiguous.
        kv_indices = KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        sparse_kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)

        offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
        dq = bwd_dq_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
            K, V,
            dq, q, do, Di, lse,
            off_zq, off_hq2, offs_m2, offs_n2,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=False,
        )

        if HAS_FULL_BLOCKS:
            # ~~~~~~~~~~~ partial unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
            kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
            kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
            sparse_kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)

            offs_n2 = kv_start + tl.arange(0, BLOCK_N2)
            dq = bwd_dq_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
                K, V,
                dq, q, do, Di, lse,
                off_zq, off_hq2, offs_m2, offs_n2,
                stride_kn, stride_kd, stride_vn, stride_vd,
                kv_indices, sparse_kv_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=True,
            )

        # Write back dQ.
        dq_ptrs = DQ2 + offs_m2[:, None] * stride_dqm + offs_k[None, :] * stride_dqd
        dq *= SM_SCALE
        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dq_ptrs, dq)
        else:
            tl.store(dq_ptrs, dq, mask=(offs_m2[:, None] < Q_LEN) & (offs_k[None, :] < QK_HEAD_DIM))
    else:
        # THIS BLOCK DOES DK & DV
        SPARSE_Q_MULTIPLE = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
        SPARSE_KV_MULTIPLE = (SPARSE_KV_BLOCK_SIZE // BLOCK_N1)

        pid_mask = pid // SPARSE_KV_MULTIPLE

        stride_q_num_blks_h = ks5
        stride_q_idx_h = ks6*ks7
        stride_q_idx_n = ks6


        dv = tl.zeros([BLOCK_N1, V_HEAD_DIM_ROUNDED], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, QK_HEAD_DIM_ROUNDED], dtype=tl.float32)

        start_n1 = pid * BLOCK_N1
        offs_n1 = start_n1 + tl.arange(0, BLOCK_N1)

        # load K and V: they stay in SRAM throughout the inner loop.
        k = load_checked_2d(K, offs_n1, offs_k, stride_kn, stride_kd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)
        v = load_checked_2d(V, offs_n1, offs_v, stride_vn, stride_vd, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)

        if PRESCALE_QK:
            k = (k * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

        for off_g in range(0, GQA_SHARED_HEADS):
            off_hq1 = off_hkv * GQA_SHARED_HEADS + off_g

            # Offset Q, DQ, DO, DELTA & LSE. These inputs are offsetted by query heads.
            q_adj1 = (stride_qh * off_hq1 + stride_qz * off_zq).to(tl.int64)
            do_adj1 = (stride_doh * off_hq1 + stride_doz * off_zq).to(tl.int64)
            dq_adj1 = (stride_dqh * off_hq1 + stride_dqz * off_zq).to(tl.int64)
            off_chz1 = ((off_zq * HQ + off_hq1) * Q_LEN).to(tl.int64)

            Q1 = Q + q_adj1
            DO1 = DO + do_adj1
            # TODO: This does not work if DQ is not the same layout as Q (for example,
            # if Q is broadcasted)
            LSE1 = LSE + off_chz1
            DELTA1 = DELTA + off_chz1

            sparse_idx_hq1 = off_hq1 % SPARSE_HQ
            sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq1

            sparse_q_num_blks_offset = sparse_hz_offset * stride_q_num_blks_h + pid_mask
            sparse_q_idx_offset = sparse_hz_offset * stride_q_idx_h + pid_mask * stride_q_idx_n  # noqa: B950

            # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Q_IDX and Q_NUM_BLKS are always contiguous.
            q_indices = Q_IDX + sparse_q_idx_offset
            q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
            sparse_q_num_blocks = tl.load(Q_NUM_BLKS + sparse_q_num_blks_offset)

            offs_m1 = q_start + tl.arange(0, BLOCK_M1)
            dk, dv = bwd_dkdv_inner(
                arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
                Q1, DO1, DELTA1, LSE1,
                dk, dv, k, v,
                off_zq, off_hq1, offs_n1, offs_m1,
                stride_qm, stride_qd, stride_dom, stride_dod,
                q_indices, sparse_q_num_blocks,
                MATMUL_PRECISION,
                IS_FULL_BLOCKS=False,
            )


            if HAS_FULL_BLOCKS:
                # ~~~~~~~~~~~~~~~ fully unmasked blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # FULL_Q_IDX and FULL_Q_NUM_BLKS are always contiguous.
                q_indices = FULL_Q_IDX + sparse_q_idx_offset
                q_start = tl.load(q_indices) * SPARSE_Q_BLOCK_SIZE # first q block we're loading
                sparse_q_num_blocks = tl.load(FULL_Q_NUM_BLKS + sparse_q_num_blks_offset)

                offs_m1 = q_start + tl.arange(0, BLOCK_M1)
                dk, dv = bwd_dkdv_inner(
                    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
                    Q1, DO1, DELTA1, LSE1,
                    dk, dv, k, v,
                    off_zq, off_hq1, offs_n1, offs_m1,
                    stride_qm, stride_qd, stride_dom, stride_dod,
                    q_indices, sparse_q_num_blocks,
                    MATMUL_PRECISION,
                    IS_FULL_BLOCKS=True,
                )

        # Write back dV and dK.
        dv_ptrs = DV + offs_n1[:, None] * stride_dvm + offs_v[None, :] * stride_dvd

        index_n = offs_n1[:, None]
        index_k = offs_k[None, :]
        index_v = offs_v[None, :]

        if IS_DIVISIBLE and SAFE_HEAD_DIM:
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dv_ptrs, dv, mask=(index_n < KV_LEN) & (index_v < V_HEAD_DIM))

        dk *= SM_SCALE

        if SAFE_HEAD_DIM:
            mask = index_n < KV_LEN
        else:
            mask = (index_n < KV_LEN) & (index_k < QK_HEAD_DIM)

        # first compute broadcasted dk of shape [Bq, Hkv, KV_LEN, V_HEAD_DIM]
        # then reduce to dk of shape [Bkv, Hkv, KV_LEN, V_HEAD_DIM]
        tl.static_assert(dk.shape == [BLOCK_N1, QK_HEAD_DIM_ROUNDED])
        xindex = index_k + 128*index_n + 128*off_hkv*ks1 + 1024*off_zq*ks1
        tl.store(out_ptr0 + (tl.broadcast_to(index_k + 128*off_hkv + 1024*index_n, dk.shape)), dk, mask)

@triton.jit
def bwd_dq_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
    K, V,  # pointers
    dq, q, do, Di, lse,
    off_z, off_hq, offs_m2, offs_n2,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    FORCE_USE_FLEX_ATTENTION : tl.constexpr = True
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 128
    BLOCK_M2 : tl.constexpr = 128
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32

    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N2)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = ks0
    KV_LEN = ks1

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    kT_ptrs = K + offs_n2[None, :] * stride_kn + offs_k[:, None] * stride_kd
    vT_ptrs = V + offs_n2[None, :] * stride_vn + offs_v[:, None] * stride_vd
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

    hi = tl.minimum(sparse_kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N2), 1))

    for start_n in range(0, hi):
        dq = bwd_dq_block_mn(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
            dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
            off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
            stride_kn, stride_kd, stride_vn, stride_vd,
            kv_indices, sparse_kv_num_blocks,
            MATMUL_PRECISION, RCP_LN2,
            IS_FULL_BLOCKS,
        )

        # Increment pointers.
        offset = get_offset_for_next_block(
            start_n, kv_indices, sparse_kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N2, BLOCKS_ARE_CONTIGUOUS
        )

        kT_ptrs += offset * stride_kn
        vT_ptrs += offset * stride_vn

        offs_n2 += offset

    return dq


@triton.jit
def bwd_dq_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
    dq, q, kT_ptrs, vT_ptrs, do, Di, lse, Q_LEN, KV_LEN,
    off_z, off_hq, offs_m2, offs_n2, offs_k, offs_v,
    stride_kn, stride_kd, stride_vn, stride_vd,
    kv_indices, sparse_kv_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS,
):
    FORCE_USE_FLEX_ATTENTION : tl.constexpr = True
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 128
    BLOCK_M2 : tl.constexpr = 128
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32


    # NB reversed order to since K is transposed
    kT = load_checked_2d(kT_ptrs, offs_k, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, KV_LEN)
    qk = tl.dot(q, kT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    pre_mod_scores = qk
    n = get_bounded_indices(offs_n2[None, :], KV_LEN if not IS_DIVISIBLE else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across N dim
    # that the M reads out of bounds for the PIDS spanning the Q_LEN boundary
    m = get_bounded_indices(offs_m2[:, None], Q_LEN if not IS_DIVISIBLE else None)

    tmp0 = (qk)
    post_mod_scores = tmp0




    if not IS_DIVISIBLE:
        post_mod_scores = tl.where(offs_n2[None, :] < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (off_z)
        tmp2 = (m)
        tmp3 = tl.full([1], 8, tl.int32)
        tmp4 = tl.where((tmp2 < 0) != (tmp3 < 0), tl.where(tmp2 % tmp3 != 0, tmp2 // tmp3 - 1, tmp2 // tmp3), tmp2 // tmp3)
        tmp5 = tl.full([1], 0, tl.int32)
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = ks8
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = tl.load(in_ptr16 + tmp8)
        tmp10 = tmp4 <= tmp7
        tmp11 = tmp9 & tmp10
        tmp12 = (n)
        tmp13 = tmp12 < tmp5
        tmp14 = tmp12.to(tl.int64)
        tmp15 = tl.load(in_ptr17 + tmp8)
        tmp16 = tl.full([1], 1, tl.int64)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 >= tmp17
        tmp19 = tmp13 & tmp18
        tmp20 = tl.full([1], 2, tl.int64)
        tmp21 = tmp15 - tmp20
        tmp22 = tmp14 <= tmp21
        tmp23 = tmp19 & tmp22
        tmp24 = tmp12 >= tmp5
        tmp25 = ks9
        tmp26 = tmp12 < tmp25
        tmp27 = tmp24 & tmp26
        tmp28 = tmp12 - tmp5
        tmp29 = tl.full([1], 5, tl.int32)
        tmp30 = tl.where((tmp28 < 0) != (tmp29 < 0), tl.where(tmp28 % tmp29 != 0, tmp28 // tmp29 - 1, tmp28 // tmp29), tmp28 // tmp29)
        tmp31 = tmp30 == tmp4
        tmp32 = tmp27 & tmp31
        tmp33 = tmp23 | tmp32
        tmp34 = tmp12 >= tmp25
        tmp35 = tmp12 - tmp25
        tmp36 = tl.where((tmp35 < 0) != (tmp3 < 0), tl.where(tmp35 % tmp3 != 0, tmp35 // tmp3 - 1, tmp35 // tmp3), tmp35 // tmp3)
        tmp37 = tmp36 == tmp4
        tmp38 = tmp34 & tmp37
        tmp39 = (tmp35 % tmp3)
        tmp40 = tmp39 != tmp5
        tmp41 = (libdevice.signbit(tmp39) != 0) if (tmp39).dtype is tl.float32 else tmp39 < 0
        tmp42 = (libdevice.signbit(tmp3) != 0) if (tmp3).dtype is tl.float32 else tmp3 < 0
        tmp43 = tmp41 != tmp42
        tmp44 = tmp40 & tmp43
        tmp45 = tmp39 + tmp3
        tmp46 = tl.where(tmp44, tmp45, tmp39)
        tmp47 = triton_helpers.maximum(tmp46, tmp5)
        tmp48 = tl.full([1], 7, tl.int32)
        tmp49 = triton_helpers.minimum(tmp47, tmp48)
        tmp50 = tl.full([1], 1, tl.int32)
        tmp51 = tmp49 < tmp50
        tmp52 = triton_helpers.maximum(tmp36, tmp5)
        tmp53 = triton_helpers.minimum(tmp52, tmp7)
        tmp54 = triton_helpers.minimum(tmp49, tmp5)
        tmp55 = tl.load(in_ptr18 + tmp53)
        tmp56 = tmp55 == 0
        tmp57 = tmp51 & tmp56
        tmp58 = tmp57 == 0
        tmp59 = tmp38 & tmp58
        tmp60 = tmp33 | tmp59
        tmp61 = tmp11 & tmp60
        tmp62 = tmp11 == 0
        tmp63 = tmp32 & tmp62
        tmp64 = tmp63 & tmp10
        tmp65 = tmp61 | tmp64
        mask_mod_output = tmp65


        # apply mask for partial masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    p = tl.math.exp2(post_mod_scores - lse)
    # Compute dP and dS.
    # NB reversed order to since V is transposed
    vT = load_checked_2d(vT_ptrs, offs_v, offs_n2, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, V_HEAD_DIM, KV_LEN)

    dp = tl.dot(do, vT, input_precision=FLOAT32_PRECISION)
    ds = p * (dp - Di[:, None])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp66 = (ds)
    grad_scores = tmp66


    if not IS_DIVISIBLE:
        grad_scores = tl.where(offs_n2[None, :] < KV_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if WRITE_DQ:
        scatter_mask = (offs_m2[:, None] < Q_LEN ) & (offs_n2[None, :] < KV_LEN)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = grad_scores

    if not IS_FULL_BLOCKS:
        # (grads) apply mask for partially unmasked block
        ds = tl.where(mask_mod_output, ds, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ds = ds.to(MATMUL_PRECISION)
    # Compute dQ.
    dq += tl.dot(ds, tl.trans(kT), input_precision=FLOAT32_PRECISION)

    return dq


@triton.jit
def bwd_dkdv_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
    Q, DO, DELTA, LSE, # pointers
    dk, dv, k, v,
    off_z, off_hq, offs_n1, offs_m1,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    FORCE_USE_FLEX_ATTENTION : tl.constexpr = True
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 128
    BLOCK_M2 : tl.constexpr = 128
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M1)
    RCP_LN2: tl.constexpr = 1.44269504
    Q_LEN = ks0
    KV_LEN = ks1

    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)

    qT_ptrs = Q + offs_m1[None, :] * stride_qm + offs_k[:, None] * stride_qd
    do_ptrs = DO + offs_m1[:, None] * stride_dom + offs_v[None, :] * stride_dod
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    # The minimum is needed to handle the case where we run with a super large
    # SPARSE_BLOCK_SIZE (i.e. no block-mask!)
    hi = tl.minimum(sparse_q_num_blocks * SPARSE_Q_MULTIPLE, tl.maximum(tl.cdiv(Q_LEN, BLOCK_M1), 1))

    for start_m in range(0, hi):
        dk, dv = bwd_dkdv_block_mn(
            arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
            dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
            off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
            stride_qm, stride_qd, stride_dom, stride_dod,
            q_indices, sparse_q_num_blocks,
            MATMUL_PRECISION, RCP_LN2,
            IS_FULL_BLOCKS,
        )
        # Increment pointers.
        offset = get_offset_for_next_block(
            start_m, q_indices, sparse_q_num_blocks,
            SPARSE_Q_BLOCK_SIZE, SPARSE_Q_MULTIPLE, BLOCK_M1, BLOCKS_ARE_CONTIGUOUS
        )

        qT_ptrs += offset * stride_qm
        do_ptrs += offset * stride_dom
        offs_m1 += offset

    return dk, dv


@triton.jit
def bwd_dkdv_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_DELTA, arg_DO, arg_DQ, arg_DV, arg_KV_NUM_BLKS, arg_KV_IDX, arg_Q_NUM_BLKS, arg_Q_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, arg_FULL_Q_NUM_BLKS, arg_FULL_Q_IDX, in_ptr16, in_ptr17, in_ptr18, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6, ks7, ks8, ks9,
    dk, dv, qT_ptrs, k, v, do_ptrs, DELTA, LSE, Q_LEN, KV_LEN,
    off_z, off_hq, offs_n1, offs_m1, offs_k, offs_v,
    stride_qm, stride_qd, stride_dom, stride_dod,
    q_indices, sparse_q_num_blocks,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS,
):
    FORCE_USE_FLEX_ATTENTION : tl.constexpr = True
    PRESCALE_QK : tl.constexpr = False
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    BLOCKS_ARE_CONTIGUOUS : tl.constexpr = False
    WRITE_DQ : tl.constexpr = True
    OUTPUT_LOGSUMEXP : tl.constexpr = True
    OUTPUT_MAX : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = False
    SM_SCALE : tl.constexpr = 0.08838834764831845
    GQA_SHARED_HEADS : tl.constexpr = 4
    HAS_FULL_BLOCKS : tl.constexpr = True
    QK_HEAD_DIM : tl.constexpr = 128
    QK_HEAD_DIM_ROUNDED : tl.constexpr = 128
    V_HEAD_DIM : tl.constexpr = 128
    V_HEAD_DIM_ROUNDED : tl.constexpr = 128
    SAFE_HEAD_DIM : tl.constexpr = True
    BLOCK_M1 : tl.constexpr = 64
    BLOCK_N1 : tl.constexpr = 128
    BLOCK_M2 : tl.constexpr = 128
    BLOCK_N2 : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32


    # NB reversed order since Q is transposed
    qT = load_checked_2d(qT_ptrs, offs_k, offs_m1, None, None, SAFE_HEAD_DIM, IS_DIVISIBLE, QK_HEAD_DIM, Q_LEN)
    # Load LSE before computing qk to reduce pipeline stall.
    if IS_DIVISIBLE:
        lse = tl.load(LSE + offs_m1)
    else:
        lse = tl.load(LSE + offs_m1, mask=offs_m1 < Q_LEN)
    lse = tl.where(lse == -float("inf"), 0.0, lse)
    qkT = tl.dot(k, qT, input_precision=FLOAT32_PRECISION)
    if not PRESCALE_QK:
        qkT *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    m = get_bounded_indices(offs_m1[None, :], Q_LEN if not IS_DIVISIBLE else None)
    # The boundary check is done for the outer loop, but here it's possible since we're iterating across M dim
    # that the n reads out of bounds for the PIDS spanning the KV_LEN boundary
    n = get_bounded_indices(offs_n1[:, None], KV_LEN if not IS_DIVISIBLE else None)

    pre_mod_scores = qkT
    tmp67 = (qkT)
    post_mod_scores = tmp67



    if not IS_DIVISIBLE:
        post_mod_scores = tl.where(offs_m1[None, :] < Q_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp68 = (off_z)
        tmp69 = (m)
        tmp70 = tl.full([1], 8, tl.int32)
        tmp71 = tl.where((tmp69 < 0) != (tmp70 < 0), tl.where(tmp69 % tmp70 != 0, tmp69 // tmp70 - 1, tmp69 // tmp70), tmp69 // tmp70)
        tmp72 = tl.full([1], 0, tl.int32)
        tmp73 = triton_helpers.maximum(tmp71, tmp72)
        tmp74 = ks8
        tmp75 = triton_helpers.minimum(tmp73, tmp74)
        tmp76 = tl.load(in_ptr16 + tmp75)
        tmp77 = tmp71 <= tmp74
        tmp78 = tmp76 & tmp77
        tmp79 = (n)
        tmp80 = tmp79 < tmp72
        tmp81 = tmp79.to(tl.int64)
        tmp82 = tl.load(in_ptr17 + tmp75)
        tmp83 = tl.full([1], 1, tl.int64)
        tmp84 = tmp82 - tmp83
        tmp85 = tmp81 >= tmp84
        tmp86 = tmp80 & tmp85
        tmp87 = tl.full([1], 2, tl.int64)
        tmp88 = tmp82 - tmp87
        tmp89 = tmp81 <= tmp88
        tmp90 = tmp86 & tmp89
        tmp91 = tmp79 >= tmp72
        tmp92 = ks9
        tmp93 = tmp79 < tmp92
        tmp94 = tmp91 & tmp93
        tmp95 = tmp79 - tmp72
        tmp96 = tl.full([1], 5, tl.int32)
        tmp97 = tl.where((tmp95 < 0) != (tmp96 < 0), tl.where(tmp95 % tmp96 != 0, tmp95 // tmp96 - 1, tmp95 // tmp96), tmp95 // tmp96)
        tmp98 = tmp97 == tmp71
        tmp99 = tmp94 & tmp98
        tmp100 = tmp90 | tmp99
        tmp101 = tmp79 >= tmp92
        tmp102 = tmp79 - tmp92
        tmp103 = tl.where((tmp102 < 0) != (tmp70 < 0), tl.where(tmp102 % tmp70 != 0, tmp102 // tmp70 - 1, tmp102 // tmp70), tmp102 // tmp70)
        tmp104 = tmp103 == tmp71
        tmp105 = tmp101 & tmp104
        tmp106 = (tmp102 % tmp70)
        tmp107 = tmp106 != tmp72
        tmp108 = (libdevice.signbit(tmp106) != 0) if (tmp106).dtype is tl.float32 else tmp106 < 0
        tmp109 = (libdevice.signbit(tmp70) != 0) if (tmp70).dtype is tl.float32 else tmp70 < 0
        tmp110 = tmp108 != tmp109
        tmp111 = tmp107 & tmp110
        tmp112 = tmp106 + tmp70
        tmp113 = tl.where(tmp111, tmp112, tmp106)
        tmp114 = triton_helpers.maximum(tmp113, tmp72)
        tmp115 = tl.full([1], 7, tl.int32)
        tmp116 = triton_helpers.minimum(tmp114, tmp115)
        tmp117 = tl.full([1], 1, tl.int32)
        tmp118 = tmp116 < tmp117
        tmp119 = triton_helpers.maximum(tmp103, tmp72)
        tmp120 = triton_helpers.minimum(tmp119, tmp74)
        tmp121 = triton_helpers.minimum(tmp116, tmp72)
        tmp122 = tl.load(in_ptr18 + tmp120)
        tmp123 = tmp122 == 0
        tmp124 = tmp118 & tmp123
        tmp125 = tmp124 == 0
        tmp126 = tmp105 & tmp125
        tmp127 = tmp100 | tmp126
        tmp128 = tmp78 & tmp127
        tmp129 = tmp78 == 0
        tmp130 = tmp99 & tmp129
        tmp131 = tmp130 & tmp77
        tmp132 = tmp128 | tmp131
        mask_mod_output = tmp132

        # (grads) apply mask for fully masked block
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    pT = tl.math.exp2(post_mod_scores - lse[None, :])
    do = load_checked_2d(do_ptrs, offs_m1, offs_v, None, None, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, V_HEAD_DIM)
    # Compute dV.
    ppT = pT
    dv += tl.dot(ppT.to(MATMUL_PRECISION), do, input_precision=FLOAT32_PRECISION)
    if IS_DIVISIBLE:
        Di = tl.load(DELTA + offs_m1)
    else:
        Di = tl.load(DELTA + offs_m1, mask=offs_m1 < Q_LEN)
    # Compute dP and dS.
    dpT = tl.dot(v, tl.trans(do), input_precision=FLOAT32_PRECISION)
    dsT = pT * (dpT - Di[None, :])
    # ~~~~~~~~~~~~~~~~~~~ Apply joint modification  ~~~~~~~~~~~~~~~~~~~
    tmp133 = (dsT)
    grad_scores = tmp133



    if not IS_DIVISIBLE:
        grad_scores = tl.where(offs_m1[None, :] < Q_LEN, grad_scores, 0.0)

    # ~~~~~~~~~~~~~~~~~~~ Apply other buffer grad writes ~~~~~~~~~~~~~
    if not WRITE_DQ:
        idx_b = off_z
        idx_h = off_hq
        idx_m = m
        idx_n = n
        scatter_mask = (offs_m1[None, :] < Q_LEN) & (offs_n1[:, None] < KV_LEN)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dsT = grad_scores
    if not IS_FULL_BLOCKS:
        # (grads) apply mask for partially unmasked block
        dsT = tl.where(mask_mod_output, dsT, 0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dk += tl.dot(dsT.to(MATMUL_PRECISION), tl.trans(qT), input_precision=FLOAT32_PRECISION)

    return dk, dv

# Utility triton funcs
@triton.jit
def get_offset_for_next_block(
    loop_iter, col_indices, total_blocks,
    SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK,
    BLOCKS_ARE_CONTIGUOUS: tl.constexpr
):
    if BLOCKS_ARE_CONTIGUOUS:
        return BLOCK
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK
    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def get_bounded_indices(indices, max_len=None):
    return indices % max_len if max_len is not None else indices

@triton.jit
def load_checked_block(block_ptr, IS_DIVISIBLE: tl.constexpr, SAFE_HEAD_DIM: tl.constexpr):
  if IS_DIVISIBLE and SAFE_HEAD_DIM:
    return tl.load(block_ptr)
  elif IS_DIVISIBLE and not SAFE_HEAD_DIM:
    return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
  elif not IS_DIVISIBLE and SAFE_HEAD_DIM:
      return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
  else:
      return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def load_checked_2d(
    ptr,
    offs_m,
    offs_n,
    stride_m,
    stride_n,
    IS_DIVISIBLE_M: tl.constexpr,
    IS_DIVISIBLE_N: tl.constexpr,
    M_LEN: tl.constexpr,
    N_LEN: tl.constexpr,
):
    # Calculate final pointer if strides are provided
    if stride_m is not None and stride_n is not None:
        ptr = ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n

    # Handle all masking cases
    if not IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN) & (offs_n[None, :] < N_LEN), other=0.0)
    elif IS_DIVISIBLE_M and not IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_n[None, :] < N_LEN), other=0.0)
    elif not IS_DIVISIBLE_M and IS_DIVISIBLE_N:
        return tl.load(ptr, mask=(offs_m[:, None] < M_LEN), other=0.0)
    else:  # Both divisible
        return tl.load(ptr)
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
        primals_10, primals_11, primals_14, primals_15, primals_17, primals_19, primals_20, primals_7, primals_8, primals_12, primals_22, primals_24, primals_25, primals_27, primals_30, primals_29, primals_32, primals_35, primals_34, primals_2, primals_4, primals_6, primals_9, primals_13, primals_16, primals_18, primals_21, primals_23, primals_26, primals_28, primals_31, primals_33, primals_36, getitem, getitem_1, tangents_1 = args
        args.clear()
        s37 = primals_10
        s0 = primals_11
        s38 = primals_14
        s11 = primals_15
        s79 = primals_17
        s44 = primals_19
        s45 = primals_20
        s22 = primals_7
        s72 = primals_8
        s99 = primals_12
        s94 = primals_22
        s28 = primals_24
        s4 = primals_25
        s56 = primals_27
        s53 = primals_30
        s84 = primals_29
        s100 = primals_32
        s10 = primals_35
        s5 = primals_34
        primals_16_size = primals_16.size()
        s89 = primals_16_size[1]
        assert_size_stride(primals_2, (1, 32, s37, 128), (4096*s37, 128, 4096, 1))
        assert_size_stride(primals_4, (1, 8, s0, 128), (1024*s0, 128, 1024, 1))
        assert_size_stride(primals_6, (1, 8, s0, 128), (1024*s0, 128, 1024, 1))
        assert_size_stride(primals_9, (1, 1, s22, s72), (s22*s72, s22*s72, s72, 1))
        assert_size_stride(primals_13, (1, 1, s99), (s99, s99, 1))
        assert_size_stride(primals_16, (1, s89), (s89, 1))
        assert_size_stride(primals_18, (1, s79), (s79, 1))
        assert_size_stride(primals_21, (1, s45, 1), (s45, 1, 1))
        assert_size_stride(primals_23, (1, 1, s94), (s94, s94, 1))
        assert_size_stride(primals_26, (1, 1, s28, s4), (s28*s4, s28*s4, s4, 1))
        assert_size_stride(primals_28, (1, 1, s56), (s56, s56, 1))
        assert_size_stride(primals_31, (1, 1, s84, s53), (s53*s84, s53*s84, s53, 1))
        assert_size_stride(primals_33, (1, 1, s100), (s100, s100, 1))
        assert_size_stride(primals_36, (1, 1, s5, s10), (s10*s5, s10*s5, s10, 1))
        assert_size_stride(getitem, (1, 32, s37, 128), (4096*s37, 128, 4096, 1))
        assert_size_stride(getitem_1, (1, 32, s37), (32*max(1, s37), max(1, s37), 1))
        assert_size_stride(tangents_1, (1, 32, s37, 128), (4096*max(1, s37), 128*max(1, s37), 128, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, 32, s37), (32*s37, s37, 1), torch.float32)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
            triton_red_fused_zeros_0_xnumel = 32*s37
            stream0 = get_raw_stream(0)
            triton_red_fused_zeros_0.run(getitem, tangents_1, buf1, s37, triton_red_fused_zeros_0_xnumel, 128, stream=stream0)
            del getitem
            buf3 = empty_strided_cuda((1, 32, s37, 128), (4096*s37, 128, 4096, 1), torch.bfloat16)
            buf4 = empty_strided_cuda((1, 8, s0, 128), (1024*s0, 128, 1024, 1), torch.bfloat16)
            buf5 = empty_strided_cuda((1, 8, s0, 128), (1024*s0, 128, 1024, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [], Original ATen: [aten.zeros]
            stream0 = get_raw_stream(0)
            triton_tem_fused_zeros_1.run(primals_2, primals_4, primals_6, getitem_1, buf1, tangents_1, buf3, buf4, primals_13, primals_9, primals_28, primals_31, primals_23, primals_26, primals_33, primals_36, primals_16, primals_18, primals_21, buf5, s37, s0, s99, s22, s72, s56, s53, s84, s38, s44, 4*((127 + s37) // 128) + ((127 + s0) // 128), 1, 8, stream=stream0)
            del buf1
            del getitem_1
            del primals_13
            del primals_16
            del primals_18
            del primals_2
            del primals_21
            del primals_23
            del primals_26
            del primals_28
            del primals_31
            del primals_33
            del primals_36
            del primals_4
            del primals_6
            del primals_9
            del tangents_1
        return (None, buf3, None, buf5, None, buf4, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_10 = 4096
    primals_11 = 6656
    primals_14 = 511
    primals_15 = 512
    primals_17 = 512
    primals_19 = 2560
    primals_20 = 512
    primals_7 = 32
    primals_8 = 52
    primals_12 = 32
    primals_22 = 32
    primals_24 = 32
    primals_25 = 52
    primals_27 = 52
    primals_30 = 32
    primals_29 = 52
    primals_32 = 52
    primals_35 = 32
    primals_34 = 52
    primals_2 = rand_strided((1, 32, 4096, 128), (16777216, 128, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 8, 6656, 128), (6815744, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_6 = rand_strided((1, 8, 6656, 128), (6815744, 128, 1024, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((1, 1, 32, 52), (1664, 1664, 52, 1), device='cuda:0', dtype=torch.int32)
    primals_13 = rand_strided((1, 1, 32), (32, 32, 1), device='cuda:0', dtype=torch.int32)
    primals_16 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.bool)
    primals_18 = rand_strided((1, 512), (512, 1), device='cuda:0', dtype=torch.int64)
    primals_21 = rand_strided((1, 512, 1), (512, 1, 1), device='cuda:0', dtype=torch.bool)
    primals_23 = rand_strided((1, 1, 32), (32, 32, 1), device='cuda:0', dtype=torch.int32)
    primals_26 = rand_strided((1, 1, 32, 52), (1664, 1664, 52, 1), device='cuda:0', dtype=torch.int32)
    primals_28 = rand_strided((1, 1, 52), (52, 52, 1), device='cuda:0', dtype=torch.int32)
    primals_31 = rand_strided((1, 1, 52, 32), (1664, 1664, 32, 1), device='cuda:0', dtype=torch.int32)
    primals_33 = rand_strided((1, 1, 52), (52, 52, 1), device='cuda:0', dtype=torch.int32)
    primals_36 = rand_strided((1, 1, 52, 32), (1664, 1664, 32, 1), device='cuda:0', dtype=torch.int32)
    getitem = rand_strided((1, 32, 4096, 128), (16777216, 128, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 32, 4096), (131072, 4096, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((1, 32, 4096, 128), (16777216, 524288, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_10, primals_11, primals_14, primals_15, primals_17, primals_19, primals_20, primals_7, primals_8, primals_12, primals_22, primals_24, primals_25, primals_27, primals_30, primals_29, primals_32, primals_35, primals_34, primals_2, primals_4, primals_6, primals_9, primals_13, primals_16, primals_18, primals_21, primals_23, primals_26, primals_28, primals_31, primals_33, primals_36, getitem, getitem_1, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
