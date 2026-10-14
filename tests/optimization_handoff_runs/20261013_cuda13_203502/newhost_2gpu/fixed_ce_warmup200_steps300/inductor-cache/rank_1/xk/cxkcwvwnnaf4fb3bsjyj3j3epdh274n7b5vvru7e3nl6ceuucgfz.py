# AOT ID: ['3_forward']
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
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/fixed_ce_warmup200_steps300/inductor-cache/rank_1/mv/cmvzyndzh4fer6oteqvlujyhayzeschi7humdgz4xg7p7kok5llc.py
# Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
# Source node to ATen node mapping:
#   flex_attention => flex_attention
# Graph fragment:
#   %primals_2 : Tensor "bf16[1, 32, s37, 128][4096*s37, 128, 4096, 1]cuda:1" = PlaceHolder[target=primals_2]
#   %primals_4 : Tensor "bf16[1, 8, s0, 128][1024*s0, 128, 1024, 1]cuda:1" = PlaceHolder[target=primals_4]
#   %primals_6 : Tensor "bf16[1, 8, s0, 128][1024*s0, 128, 1024, 1]cuda:1" = PlaceHolder[target=primals_6]
#   %getitem_1 : Tensor "f32[1, 32, s37][32*s37, s37, 1]cuda:1" = PlaceHolder[target=getitem_1]
#   %buf1 : Tensor "f32[1, 32, s37][32*s37, s37, 1]cuda:1" = PlaceHolder[target=buf1]
#   %primals_13 : Tensor "i32[1, 1, s99][s99, s99, 1]cuda:1" = PlaceHolder[target=primals_13]
#   %primals_9 : Tensor "i32[1, 1, s22, s72][s22*s72, s22*s72, s72, 1]cuda:1" = PlaceHolder[target=primals_9]
#   %primals_23 : Tensor "i32[1, 1, s94][s94, s94, 1]cuda:1" = PlaceHolder[target=primals_23]
#   %primals_26 : Tensor "i32[1, 1, s28, s4][s28*s4, s28*s4, s4, 1]cuda:1" = PlaceHolder[target=primals_26]
#   %primals_16 : Tensor "b8[1, s89][s89, 1]cuda:1" = PlaceHolder[target=primals_16]
#   %primals_18 : Tensor "i64[1, s79][s79, 1]cuda:1" = PlaceHolder[target=primals_18]
#   %primals_21 : Tensor "b8[1, s45, 1][s45, 1, 1]cuda:1" = PlaceHolder[target=primals_21]
#   %flex_attention : [num_users=2] = call_function[target=torch.ops.higher_order.flex_attention](args = (%primals_2, %primals_4, %primals_6, %sdpa_score0, (%primals_10, %primals_11, %primals_13, %primals_9, %primals_23, %primals_26, %primals_28, %primals_31, %primals_33, %primals_36, 128, 128, %sdpa_mask0), 0.08838834764831845, {FORCE_USE_FLEX_ATTENTION: True, PRESCALE_QK: False, ROWS_GUARANTEED_SAFE: False, BLOCKS_ARE_CONTIGUOUS: False, WRITE_DQ: True, OUTPUT_LOGSUMEXP: True, OUTPUT_MAX: False}, (), (%primals_14, %primals_15, %primals_16, %primals_17, %primals_18, %primals_19, %primals_20, %primals_21)), kwargs = {})
#   return %getitem
triton_tem_fused_0 = async_compile.triton('triton_tem_fused_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

@triton_heuristics.template(

num_stages=3,
num_warps=8,
triton_meta={'signature': {'arg_Q': '*bf16', 'arg_K': '*bf16', 'arg_V': '*bf16', 'arg_LSE': '*fp32', 'arg_MAX': '*fp32', 'arg_KV_NUM_BLKS': '*i32', 'arg_KV_IDX': '*i32', 'arg_FULL_KV_NUM_BLKS': '*i32', 'arg_FULL_KV_IDX': '*i32', 'in_ptr9': '*i1', 'in_ptr10': '*i64', 'in_ptr11': '*i1', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'ks3': 'i32', 'ks4': 'i32', 'ks5': 'i32', 'ks6': 'i32'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
inductor_meta={'kernel_name': 'triton_tem_fused_0', 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'FORCE_USE_FLEX_ATTENTION': True, 'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': True, 'OUTPUT_MAX': False, 'FLOAT32_PRECISION': "'ieee'", 'IS_DIVISIBLE': False, 'SM_SCALE': 0.08838834764831845, 'GQA_SHARED_HEADS': 4, 'HAS_FULL_BLOCKS': True, 'QK_HEAD_DIM': 128, 'QK_HEAD_DIM_ROUNDED': 128, 'V_HEAD_DIM': 128, 'V_HEAD_DIM_ROUNDED': 128, 'SAFE_HEAD_DIM': True, 'USE_TMA': False, 'BLOCK_M': 128, 'BLOCK_N': 64, 'SPARSE_Q_BLOCK_SIZE': 128, 'SPARSE_KV_BLOCK_SIZE': 128}},

)
@triton.jit
def triton_tem_fused_0(arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6):
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
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    MAX = arg_MAX
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # BLOCKS_ARE_CONTIGUOUS: Is it guaranteed that all blocks in the mask are
    # contiguous? If so, we don't need to do an indirect jump for every block

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = 4096*ks0, 128, 4096, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 1024*ks1, 128, 1024, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 1024*ks1, 128, 1024, 1

    ZQ = 1
    HQ = 32
    Q_LEN = ks0
    ZKV = 1
    KV_LEN = ks1

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0).to(INDEX_DTYPE)
    off_zq = tl.program_id(1).to(INDEX_DTYPE)
    off_hq = tl.program_id(2).to(INDEX_DTYPE)

    # We support two cases for batch dimension. a) (ZKV == ZQ) where off_zkv = off_zq.
    # b) (ZKV == 1 and ZQ > 1) where KV is broadcasted along the batch dimension and off_zkv=0.
    off_zkv = off_zq % ZKV
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_zq * stride_qz + off_hq * stride_qh
    k_offset = off_zkv * stride_kz + off_hkv * stride_kh
    v_offset = off_zkv * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    # Setting up the TMA descriptors for Q, K, V
    desc_q = None
    desc_k = None
    desc_v = None

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_zq % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    stride_kv_num_blks_h = ks2
    stride_kv_idx_h = ks3*ks4
    stride_kv_idx_m = ks4

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM_ROUNDED], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * stride_kv_num_blks_h + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * stride_kv_idx_h + (q_start // SPARSE_Q_MULTIPLE) * stride_kv_idx_m  # noqa: B950
    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    q = load_checked_2d(Q, offs_m, offs_k, stride_qm, stride_qk, IS_DIVISIBLE, SAFE_HEAD_DIM, Q_LEN, QK_HEAD_DIM)

    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))


    # K and V pointers will be passed directly to forward_inner

    offs_n = kv_start + tl.arange(0, BLOCK_N)


    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
        q, K, V,
        desc_k, desc_v, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_zq, off_hq, offs_m[:, None], offs_n[None, :],
        kv_start,
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        stride_kk, stride_kn, stride_vn, stride_vk,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))
        # K and V pointers will be passed directly to forward_inner
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
            q, K, V,
            desc_k, desc_v, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_zq, off_hq, offs_m[:, None], offs_n[None, :],
            kv_start,
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            stride_kk, stride_kn, stride_vn, stride_vk,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_zq = tl.program_id(1).to(INDEX_DTYPE)
    idx_hq = tl.program_id(2).to(INDEX_DTYPE)
    idx_m = offs_m[:, None].to(INDEX_DTYPE)
    idx_d = tl.arange(0, V_HEAD_DIM_ROUNDED)[None, :].to(INDEX_DTYPE)

    mask = (idx_m < Q_LEN) & (idx_d < V_HEAD_DIM)

    tl.static_assert(acc.shape == [BLOCK_M, V_HEAD_DIM_ROUNDED])
    xindex = idx_d + 128*idx_m + 128*idx_hq*ks0 + 4096*idx_zq*ks0
    tl.store(out_ptr0 + (tl.broadcast_to(idx_d + 128*idx_hq + 4096*idx_m, acc.shape)), acc, mask)

    if OUTPUT_LOGSUMEXP:
        off_hz = off_zq * HQ + off_hq
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)

    if OUTPUT_MAX:
        off_hz = off_zq * HQ + off_hq
        max_ptrs = MAX + off_hz * Q_LEN + offs_m
        if IS_DIVISIBLE:
            tl.store(max_ptrs, m_i)
        else:
            tl.store(max_ptrs, m_i, mask=offs_m < Q_LEN)


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


# Common Imports
@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
    q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    kv_offset,
    MATMUL_PRECISION, RCP_LN2,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,

):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
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
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32


    # -- load k --
    # NB reversed order to since K is transposed
    kv_base_offset = kv_start + kv_offset

    # Load K as [BLOCK_N, QK_HEAD_DIM_ROUNDED] then transpose to [QK_HEAD_DIM_ROUNDED, BLOCK_N]
    offs_k = tl.arange(0, QK_HEAD_DIM_ROUNDED)
    offs_n_load = kv_base_offset + tl.arange(0, BLOCK_N)
    k = load_checked_2d(K, offs_n_load, offs_k, stride_kn, stride_kk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, QK_HEAD_DIM)

    k = tl.trans(k)
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
    # which is larger than the actual number of elements. To avoid access memory out of bound,
    # we need to mask out the elements that are out of Q_LEN & KV_LEN.
    m = get_bounded_indices(offs_m, Q_LEN if CHECK_BLOCK_BOUNDARY else None)
    n = get_bounded_indices(offs_n, KV_LEN if CHECK_BLOCK_BOUNDARY else None)

    tmp0 = (qk)
    post_mod_scores = tmp0


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp1 = (off_z)
        tmp2 = (m)
        tmp3 = tl.full([1], 8, tl.int32)
        tmp4 = tl.where((tmp2 < 0) != (tmp3 < 0), tl.where(tmp2 % tmp3 != 0, tmp2 // tmp3 - 1, tmp2 // tmp3), tmp2 // tmp3)
        tmp5 = tl.full([1], 0, tl.int32)
        tmp6 = triton_helpers.maximum(tmp4, tmp5)
        tmp7 = ks5
        tmp8 = triton_helpers.minimum(tmp6, tmp7)
        tmp9 = tl.load(in_ptr9 + tmp8)
        tmp10 = tmp4 <= tmp7
        tmp11 = tmp9 & tmp10
        tmp12 = (n)
        tmp13 = tmp12 < tmp5
        tmp14 = tmp12.to(tl.int64)
        tmp15 = tl.load(in_ptr10 + tmp8)
        tmp16 = tl.full([1], 1, tl.int64)
        tmp17 = tmp15 - tmp16
        tmp18 = tmp14 >= tmp17
        tmp19 = tmp13 & tmp18
        tmp20 = tl.full([1], 2, tl.int64)
        tmp21 = tmp15 - tmp20
        tmp22 = tmp14 <= tmp21
        tmp23 = tmp19 & tmp22
        tmp24 = tmp12 >= tmp5
        tmp25 = ks6
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
        tmp55 = tl.load(in_ptr11 + tmp53)
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


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, False)
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]
    # Calculate offsets for V loading - reuse kv_base_offset from K loading
    offs_v = tl.arange(0, V_HEAD_DIM_ROUNDED)
    v = load_checked_2d(V, offs_n_load, offs_v, stride_vn, stride_vk, IS_DIVISIBLE, SAFE_HEAD_DIM, KV_LEN, V_HEAD_DIM)
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
    q, K, V,
    desc_k, desc_v, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # Offsets needed for TMA loads
    kv_start,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    # Strides for K and V
    stride_kk, stride_kn, stride_vn, stride_vk,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
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
    USE_TMA : tl.constexpr = False
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 128
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 128
    INDEX_DTYPE : tl.constexpr = tl.int32


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    kv_offset = 0

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        # Here IS_DIVISIBLE acts are the start_n = tl.multiple_of(start_n, BLOCK_N) from triton_fused_attention.
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_MAX, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, in_ptr10, in_ptr11, out_ptr0, ks0, ks1, ks2, ks3, ks4, ks5, ks6,
                q, K, V, desc_k, desc_v, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                # Offsets needed for TMA loads
                kv_start,
                kv_offset,
                MATMUL_PRECISION, RCP_LN2,
                # Strides for K and V
                stride_kk, stride_kn, stride_vn, stride_vk,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )



        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N, BLOCKS_ARE_CONTIGUOUS
        )

        offs_n = offs_n + offset
        kv_offset += offset


    return acc, l_i, m_i
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36 = args
        args.clear()
        s50 = primals_1
        s0 = primals_3
        s43 = primals_5
        s22 = primals_7
        s72 = primals_8
        s37 = primals_10
        s71 = primals_11
        s99 = primals_12
        s38 = primals_14
        s11 = primals_15
        s79 = primals_17
        s44 = primals_19
        s45 = primals_20
        s94 = primals_22
        s28 = primals_24
        s4 = primals_25
        s56 = primals_27
        s84 = primals_29
        s53 = primals_30
        s100 = primals_32
        s5 = primals_34
        s10 = primals_35
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
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            buf0 = empty_strided_cuda((1, 32, s37), (32*s37, s37, 1), torch.float32)
            buf1 = empty_strided_cuda((1, 32, s37), (32*s37, s37, 1), torch.float32)
            buf2 = empty_strided_cuda((1, 32, s37, 128), (4096*s37, 128, 4096, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
            stream1 = get_raw_stream(1)
            triton_tem_fused_0.run(primals_2, primals_4, primals_6, buf0, buf1, primals_13, primals_9, primals_23, primals_26, primals_16, primals_18, primals_21, buf2, s37, s0, s99, s22, s72, s38, s44, (127 + s37) // 128, 1, 32, stream=stream1)
            del buf1
        return (buf2, primals_2, primals_4, primals_6, primals_9, primals_13, primals_16, primals_18, primals_21, primals_23, primals_26, primals_28, primals_31, primals_33, primals_36, buf2, buf0, s37, s0, s38, s11, s79, s44, s45, s22, s72, s99, s94, s28, s4, s56, s53, s84, s100, s10, s5, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 1872
    primals_2 = rand_strided((1, 32, 1872, 128), (7667712, 128, 4096, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_3 = 3042
    primals_4 = rand_strided((1, 8, 3042, 128), (3115008, 128, 1024, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_5 = 3042
    primals_6 = rand_strided((1, 8, 3042, 128), (3115008, 128, 1024, 1), device='cuda:1', dtype=torch.bfloat16)
    primals_7 = 15
    primals_8 = 24
    primals_9 = rand_strided((1, 1, 15, 24), (360, 360, 24, 1), device='cuda:1', dtype=torch.int32)
    primals_10 = 1872
    primals_11 = 3042
    primals_12 = 15
    primals_13 = rand_strided((1, 1, 15), (15, 15, 1), device='cuda:1', dtype=torch.int32)
    primals_14 = 233
    primals_15 = 234
    primals_16 = rand_strided((1, 234), (234, 1), device='cuda:1', dtype=torch.bool)
    primals_17 = 234
    primals_18 = rand_strided((1, 234), (234, 1), device='cuda:1', dtype=torch.int64)
    primals_19 = 1170
    primals_20 = 234
    primals_21 = rand_strided((1, 234, 1), (234, 1, 1), device='cuda:1', dtype=torch.bool)
    primals_22 = 15
    primals_23 = rand_strided((1, 1, 15), (15, 15, 1), device='cuda:1', dtype=torch.int32)
    primals_24 = 15
    primals_25 = 24
    primals_26 = rand_strided((1, 1, 15, 24), (360, 360, 24, 1), device='cuda:1', dtype=torch.int32)
    primals_27 = 24
    primals_28 = rand_strided((1, 1, 24), (24, 24, 1), device='cuda:1', dtype=torch.int32)
    primals_29 = 24
    primals_30 = 15
    primals_31 = rand_strided((1, 1, 24, 15), (360, 360, 15, 1), device='cuda:1', dtype=torch.int32)
    primals_32 = 24
    primals_33 = rand_strided((1, 1, 24), (24, 24, 1), device='cuda:1', dtype=torch.int32)
    primals_34 = 24
    primals_35 = 15
    primals_36 = rand_strided((1, 1, 24, 15), (360, 360, 15, 1), device='cuda:1', dtype=torch.int32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
