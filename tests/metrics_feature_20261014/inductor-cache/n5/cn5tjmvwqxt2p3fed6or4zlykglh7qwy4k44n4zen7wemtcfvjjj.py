
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
