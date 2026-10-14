
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
