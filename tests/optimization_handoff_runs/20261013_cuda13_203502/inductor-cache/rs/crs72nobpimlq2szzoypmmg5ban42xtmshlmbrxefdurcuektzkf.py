
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_max_mul_sgn_sub_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_max_mul_sgn_sub_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(r0_mask, tmp9, _tmp8)
        tmp10 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
        tmp13 = tl.full([1, 1], 0, tl.int32)
        tmp14 = tmp13 < tmp2
        tmp15 = tmp14.to(tl.int8)
        tmp16 = tmp2 < tmp13
        tmp17 = tmp16.to(tl.int8)
        tmp18 = tmp15 - tmp17
        tmp19 = tmp18.to(tmp2.dtype)
        tl.store(out_ptr3 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp19, r0_mask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp8, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
    tmp20 = tl.load(in_ptr2 + (0))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, 1])
    tmp22 = tmp5 * tmp21
    tl.store(out_ptr4 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
