
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_eq_floor_divide_ge_index_le_lt_remainder_sub_view_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2695784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2093)
    x1 = xindex // 2093
    tmp3 = tl.load(in_ptr0 + (((160) * ((160) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (160)))), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr1 + (((160) * ((160) <= (((0) * ((0) >= (triton_helpers.div_floor_integer((-805) + x0,  8))) + (triton_helpers.div_floor_integer((-805) + x0,  8)) * ((triton_helpers.div_floor_integer((-805) + x0,  8)) > (0))))) + (((0) * ((0) >= (triton_helpers.div_floor_integer((-805) + x0,  8))) + (triton_helpers.div_floor_integer((-805) + x0,  8)) * ((triton_helpers.div_floor_integer((-805) + x0,  8)) > (0)))) * ((((0) * ((0) >= (triton_helpers.div_floor_integer((-805) + x0,  8))) + (triton_helpers.div_floor_integer((-805) + x0,  8)) * ((triton_helpers.div_floor_integer((-805) + x0,  8)) > (0)))) < (160)))), xmask, eviction_policy='evict_last').to(tl.int1)
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
    tmp13 = tl.full([1], 805, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = x0 // 5
    tmp17 = x1 // 8
    tmp18 = tmp16 == tmp17
    tmp19 = tmp15 & tmp18
    tmp20 = tmp11 | tmp19
    tmp21 = tmp0 >= tmp13
    tmp22 = triton_helpers.div_floor_integer((-805) + x0,  8)
    tmp23 = tmp22 == tmp17
    tmp24 = tmp21 & tmp23
    tmp25 = (-805) + x0
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
    tl.store(out_ptr0 + (x0 + 2176*x1), tmp46, xmask)
