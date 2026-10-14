
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': None, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 19
    R0_BLOCK: tl.constexpr = 32
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
    tmp0 = tl.load(in_ptr0 + (r0_1 + 19*x0), r0_mask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl_math.log(tmp4)
    tmp7 = tl.full([1, 1], -100, tl.int64)
    tmp8 = tmp6 != tmp7
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tl.where(tmp8, tmp6, tmp9)
    tmp11 = ks0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert(((0 <= tmp14) & (tmp14 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp14 < ks0")
    tmp16 = tl.load(in_ptr2 + (tmp14 + ks0*x0), xmask, eviction_policy='evict_last')
    tmp18 = tmp16 - tmp17
    tmp19 = tmp18 - tmp5
    tmp20 = -tmp19
    tmp21 = 0.0
    tmp22 = tl.where(tmp8, tmp20, tmp21)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr0 + (x0), tmp22, xmask)
