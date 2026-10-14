
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp11 = tl.load(in_ptr1 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp13 = tl.load(in_ptr2 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp4 = tl.full([1, 1], 0, tl.int64)
        tmp5 = tl.where(tmp3, tmp1, tmp4)
        tmp6 = r0_0
        tmp7 = tmp5 == tmp6
        tmp8 = -1.0
        tmp9 = 0.0
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp15 = tmp12 * tmp14
        tmp16 = tl.where(tmp3, tmp15, tmp9)
        tmp17 = tmp10 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr0 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp32 = tl.load(in_ptr1 + (0))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, R0_BLOCK])
    tmp34 = tl.load(in_ptr2 + (0))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK, R0_BLOCK])
    tmp40 = tl.load(in_ptr4 + (0))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK, R0_BLOCK])
    tmp43 = tl.load(in_ptr5 + (0))
    tmp44 = tl.broadcast_to(tmp43, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp39 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp23 = tl.full([1, 1], -100, tl.int64)
        tmp24 = tmp22 != tmp23
        tmp25 = tl.full([1, 1], 0, tl.int64)
        tmp26 = tl.where(tmp24, tmp22, tmp25)
        tmp27 = r0_0
        tmp28 = tmp26 == tmp27
        tmp29 = -1.0
        tmp30 = 0.0
        tmp31 = tl.where(tmp28, tmp29, tmp30)
        tmp36 = tmp33 * tmp35
        tmp37 = tl.where(tmp24, tmp36, tmp30)
        tmp38 = tmp31 * tmp37
        tmp42 = tmp39 - tmp41
        tmp45 = tmp42 - tmp44
        tmp46 = libdevice.exp(tmp45)
        tmp47 = tmp46 * tmp19
        tmp48 = tmp38 - tmp47
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp48, r0_mask)
