# AOT ID: ['7_forward']
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


# kernel path: tests/metrics_feature_20261014/inductor-cache/vz/cvz6rys7y3ftjjbcwcw6ajkd2pjjkrncg2m33n6jrp3ygnypaqew.py
# Topologically Sorted Source Nodes: [student_log_probs, sub, mul, values, mul_1, sum_2], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_6
#   mul_1 => mul_9
#   student_log_probs => log, sub_1
#   sub => sub_4
#   sum_2 => sum_3
#   values => sum_2
# Graph fragment:
#   %primals_2 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %getitem_1 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem_1]
#   %primals_4 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_3 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %getitem : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %log : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_2 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=sum_2]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_2, -1), kwargs = {})
#   %sub_tensor : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %getitem), kwargs = {})
#   %log : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_1 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %sub_4 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %sub_1), kwargs = {})
#   %mul_6 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_4, %sub_4), kwargs = {})
#   %sum_2 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [-1]), kwargs = {})
#   %mul_9 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_2, %primals_5), kwargs = {})
#   %sum_3 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_9,), kwargs = {})
#   return %getitem_1,%log,%getitem,%sum_2,%sum_3
triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_0 = async_compile.triton('triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp4 = tl_math.log(tmp3)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
    _tmp7_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp7_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])

        _tmp7_max_next, _tmp7_sum_next = triton_helpers.online_softmax_combine(
            _tmp7_max, _tmp7_sum, tmp6, False
        )

        _tmp7_max = tl.where(r0_mask, _tmp7_max_next, _tmp7_max)
        _tmp7_sum = tl.where(r0_mask, _tmp7_sum_next, _tmp7_sum)

    tmp7, tmp8 = triton_helpers.online_softmax_reduce(
        _tmp7_max, _tmp7_sum, 1, False)
    tmp7 = tmp7[:, None]
    tmp8 = tmp8[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp7, None)
    _tmp17 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp9 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp11 - tmp7
        tmp13 = tmp12 - tmp4
        tmp14 = tmp10 - tmp13
        tmp15 = tmp9 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(r0_mask, tmp18, _tmp17)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.load(in_ptr3 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, 1])
    tmp21 = tmp17 * tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5 = args
        args.clear()
        s3 = primals_1
        assert_size_stride(primals_2, (1, s3), (s3, 1))
        assert_size_stride(primals_3, (1, s3), (s3, 1))
        assert_size_stride(primals_4, (1, s3), (s3, 1))
        assert_size_stride(primals_5, (1, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf2 = buf1; del buf1  # reuse
            buf0 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
            buf3 = empty_strided_cuda((1, ), (1, ), torch.float32)
            buf4 = reinterpret_tensor(buf3, (), (), 0); del buf3  # reuse
            # Topologically Sorted Source Nodes: [student_log_probs, sub, mul, values, mul_1, sum_2], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_mul_prepare_softmax_online_sub_sum_0.run(buf2, buf4, primals_2, primals_4, primals_3, primals_5, buf0, 1, s3, stream=stream0)
            del primals_3
        return (buf4, primals_2, primals_4, primals_5, buf0, buf2, s3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 65
    primals_2 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
