
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
