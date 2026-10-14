
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=1, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_unsqueeze_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_unsqueeze_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp22 = tl.load(in_out_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
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
        tmp21 = tmp20 * tmp12
        tmp23 = tmp21 * tmp22
        tl.store(in_out_ptr0 + (r0_1 + ks0*x0), tmp23, r0_mask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp33 = tl.load(in_ptr1 + (0))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, R0_BLOCK])
    tmp39 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp38 = tl.load(in_ptr4 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp24 = tl.full([1, 1], -100, tl.int64)
        tmp25 = tmp0 != tmp24
        tmp26 = tl.full([1, 1], 0, tl.int64)
        tmp27 = tl.where(tmp25, tmp0, tmp26)
        tmp28 = r0_1
        tmp29 = tmp27 == tmp28
        tmp30 = -1.0
        tmp31 = 0.0
        tmp32 = tl.where(tmp29, tmp30, tmp31)
        tmp35 = tmp34 * tmp12
        tmp36 = tl.where(tmp25, tmp35, tmp31)
        tmp37 = tmp32 * tmp36
        tmp40 = tmp38 - tmp39
        tmp42 = tmp40 - tmp41
        tmp43 = libdevice.exp(tmp42)
        tmp44 = tmp43 * tmp17
        tmp45 = tmp37 - tmp44
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp45, r0_mask & xmask)
