
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'out_ptr1': '*i64', 'ks0': 'i64', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_copy__index_lt_neg_where_0', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_copy__index_lt_neg_where_0(in_ptr0, in_ptr1, out_ptr1, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 0, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = -tmp1
    tmp5 = triton_helpers.maximum(tmp4, tmp2)
    tmp6 = ks0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp5 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp5)
    tl.device_assert((0 <= tmp9) & (tmp9 < ks0), "index out of bounds: 0 <= tmp9 < ks0")
    tmp11 = tl.load(in_ptr1 + (tmp9), None, eviction_policy='evict_last')
    tmp12 = tl.where(tmp3, tmp11, tmp1)
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp12, None)
