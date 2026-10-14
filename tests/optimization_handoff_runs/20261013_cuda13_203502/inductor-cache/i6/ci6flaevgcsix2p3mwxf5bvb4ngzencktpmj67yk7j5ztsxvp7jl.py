# AOT ID: ['13_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/bs/cbspt5pnijx4hlrntgnvoeq4mbjcg2j75eo2rxbhlbb7wpzqqkqx.py
# Topologically Sorted Source Nodes: [cross_entropy, mul, ce], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   ce => sum_2
#   cross_entropy => full_default, full_default_1, gather, log, ne, neg, squeeze, sub_2, unsqueeze, where, where_1
#   mul => mul_7
# Graph fragment:
#   %primals_2 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %getitem_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %primals_4 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=primals_4]
#   %getitem : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %log : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %where_1 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=where_1]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_2, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %getitem), kwargs = {})
#   %log : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_2 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne : Tensor "b8[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_4, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne, %primals_4, %full_default), kwargs = {})
#   %unsqueeze : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_2, 1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ne, %neg, %full_default_1), kwargs = {})
#   %mul_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %primals_5), kwargs = {})
#   %sum_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_7,), kwargs = {})
#   return %getitem,%getitem_1,%log,%where_1,%sum_2
triton_red_fused__log_softmax_mul_nll_loss_forward_prepare_softmax_online_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax_mul_nll_loss_forward_prepare_softmax_online_sub_sum_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_mul_nll_loss_forward_prepare_softmax_online_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_mul_nll_loss_forward_prepare_softmax_online_sub_sum_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp2_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp2_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])

        _tmp2_max_next, _tmp2_sum_next = triton_helpers.online_softmax_combine(
            _tmp2_max, _tmp2_sum, tmp1, False
        )

        _tmp2_max = tl.where(r0_mask, _tmp2_max_next, _tmp2_max)
        _tmp2_sum = tl.where(r0_mask, _tmp2_sum_next, _tmp2_sum)

    tmp2, tmp3 = triton_helpers.online_softmax_reduce(
        _tmp2_max, _tmp2_sum, 1, False)
    tmp2 = tmp2[:, None]
    tmp3 = tmp3[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp2, None)
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, 1])
    tmp22 = tl.load(in_ptr2 + (0))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, 1])
    tmp4 = tl_math.log(tmp3)
    tmp7 = tl.full([1, 1], -100, tl.int64)
    tmp8 = tmp6 != tmp7
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = tl.where(tmp8, tmp6, tmp9)
    tmp11 = ks0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert((0 <= tmp14) & (tmp14 < ks0), "index out of bounds: 0 <= tmp14 < ks0")
    tmp16 = tl.load(in_ptr0 + (tmp14), None, eviction_policy='evict_last')
    tmp17 = tmp16 - tmp2
    tmp18 = tmp17 - tmp4
    tmp19 = -tmp18
    tmp20 = 0.0
    tmp21 = tl.where(tmp8, tmp19, tmp20)
    tmp24 = tmp21 * tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp24, None)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/ug/cugltz6qljbs2m4arhwylrc5fwtc5a3qzfouh7reikkko23nom4e.py
# Topologically Sorted Source Nodes: [sub, abs_1, distance, mul_1, sum_3], Original ATen: [aten.sub, aten.abs, aten.sum, aten.mul, aten.sgn]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   distance => sum_3
#   mul_1 => mul_12
#   sub => sub_5
#   sum_3 => sum_4
# Graph fragment:
#   %primals_7 : Tensor "f32[1, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_7]
#   %primals_8 : Tensor "f32[1, s12][s12, 1]cuda:0" = PlaceHolder[target=primals_8]
#   %sum_3 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sum_3]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %sub_5 : Tensor "f32[1, s12][s12, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_7, %primals_8), kwargs = {})
#   %abs_1 : Tensor "f32[1, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%sub_5,), kwargs = {})
#   %sum_3 : Tensor "f32[1][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [-1]), kwargs = {})
#   %mul_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_3, %primals_5), kwargs = {})
#   %sum_4 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_12,), kwargs = {})
#   %sign : Tensor "f32[1, s12][s12, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sign.default](args = (%sub_5,), kwargs = {})
#   return %sum_3,%sign,%sum_4
triton_red_fused_abs_mul_sgn_sub_sum_1 = async_compile.triton('triton_red_fused_abs_mul_sgn_sub_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_mul_sgn_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_abs_mul_sgn_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int32)
        tmp8 = tmp7 < tmp2
        tmp9 = tmp8.to(tl.int8)
        tmp10 = tmp2 < tmp7
        tmp11 = tmp10.to(tl.int8)
        tmp12 = tmp9 - tmp11
        tmp13 = tmp12.to(tmp2.dtype)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp13, r0_mask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp5, None)
    tmp14 = tl.load(in_ptr2 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, 1])
    tmp16 = tmp5 * tmp15
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp16, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
        args.clear()
        s3 = primals_1
        s52 = primals_3
        s12 = primals_6
        assert_size_stride(primals_2, (1, s3), (s3, 1))
        assert_size_stride(primals_4, (1, ), (1, ))
        assert_size_stride(primals_5, (1, ), (1, ))
        assert_size_stride(primals_7, (1, s12), (s12, 1))
        assert_size_stride(primals_8, (1, s12), (s12, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf2 = buf1; del buf1  # reuse
            buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf6 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [cross_entropy, mul, ce], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_mul_nll_loss_forward_prepare_softmax_online_sub_sum_0.run(buf2, primals_2, primals_4, primals_5, buf0, buf3, buf6, s3, 1, s3, stream=stream0)
            buf4 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf5 = empty_strided_cuda((1, s12), (s12, 1), torch.float32)
            buf7 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [sub, abs_1, distance, mul_1, sum_3], Original ATen: [aten.sub, aten.abs, aten.sum, aten.mul, aten.sgn]
            stream0 = get_raw_stream(0)
            triton_red_fused_abs_mul_sgn_sub_sum_1.run(primals_7, primals_8, primals_5, buf4, buf5, buf7, 1, s12, stream=stream0)
            del primals_7
            del primals_8
        return (buf6, buf7, buf3, buf4, primals_2, primals_4, primals_5, buf0, buf2, buf5, s3, s12, s52, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 513
    primals_2 = rand_strided((1, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = 6
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = 513
    primals_7 = rand_strided((1, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((1, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
