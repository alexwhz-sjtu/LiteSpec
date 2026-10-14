
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 9191424}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_bitwise_and_bitwise_not_bitwise_or_clamp_constant_pad_nd_eq_floor_divide_ge_index_le_lt_sub_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3063808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 2176
    x0 = (xindex % 2176)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 1288, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 2093, tl.int64)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (((160) * ((160) <= (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0))))) + (((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) * ((((0) * ((0) >= (x1 // 8)) + (x1 // 8) * ((x1 // 8) > (0)))) < (160)))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp8 = x1 // 8
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp8 <= tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tl.load(in_ptr1 + (x2), tmp6, other=0.0).to(tl.int1)
    tmp13 = tmp11 & tmp12
    tmp14 = x0
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = tmp14 >= tmp15
    tmp17 = tl.full([1], 805, tl.int64)
    tmp18 = tmp14 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = x0 // 5
    tmp21 = tmp20 == tmp8
    tmp22 = tmp19 & tmp21
    tmp23 = tmp11 == 0
    tmp24 = tmp22 & tmp23
    tmp25 = tmp24 & tmp10
    tmp26 = tmp13 | tmp25
    tmp27 = tl.full(tmp26.shape, False, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tl.store(out_ptr0 + (x2), tmp28, None)
