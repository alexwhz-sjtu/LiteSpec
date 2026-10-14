
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = r0_1
        tmp6 = tmp4 == tmp5
        tmp7 = -1.0
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp13 = tmp11 * tmp12
        tmp14 = tl.where(tmp2, tmp13, tmp8)
        tmp15 = tmp9 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask & xmask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp28 = tl.load(in_ptr1 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp34 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp33 = tl.load(in_ptr3 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full([1, 1], -100, tl.int64)
        tmp20 = tmp0 != tmp19
        tmp21 = tl.full([1, 1], 0, tl.int64)
        tmp22 = tl.where(tmp20, tmp0, tmp21)
        tmp23 = r0_1
        tmp24 = tmp22 == tmp23
        tmp25 = -1.0
        tmp26 = 0.0
        tmp27 = tl.where(tmp24, tmp25, tmp26)
        tmp30 = tmp29 * tmp12
        tmp31 = tl.where(tmp20, tmp30, tmp26)
        tmp32 = tmp27 * tmp31
        tmp35 = tmp33 - tmp34
        tmp37 = tmp35 - tmp36
        tmp38 = libdevice.exp(tmp37)
        tmp39 = tmp38 * tmp17
        tmp40 = tmp32 - tmp39
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp40, r0_mask & xmask)
