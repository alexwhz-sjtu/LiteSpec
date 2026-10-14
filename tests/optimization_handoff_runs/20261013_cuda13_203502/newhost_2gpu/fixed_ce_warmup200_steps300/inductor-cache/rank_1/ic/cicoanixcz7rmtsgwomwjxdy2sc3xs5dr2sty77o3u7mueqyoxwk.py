
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'ks3': 'i64', 'ks4': 'i64', 'ks5': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
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
