
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
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
