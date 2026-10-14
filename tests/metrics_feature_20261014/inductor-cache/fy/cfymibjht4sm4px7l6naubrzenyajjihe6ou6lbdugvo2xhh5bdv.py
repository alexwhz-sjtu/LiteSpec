# AOT ID: ['9_backward']
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


# kernel path: tests/metrics_feature_20261014/inductor-cache/lu/clujcukly2iw4ul2wwsio4dftpgly6j4ykx3cdgx5gx7dtq3pqg4.py
# Topologically Sorted Source Nodes: [float_1, student_log_probs], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.neg, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   student_log_probs => sub_1, sub_2
# Graph fragment:
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_5 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_4 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_2 : Tensor "bf16[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %amax : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=sum_4]
#   %expand : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [1]), kwargs = {})
#   %mul_12 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_5), kwargs = {})
#   %unsqueeze : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_12, -1), kwargs = {})
#   %expand_1 : Tensor "f32[1, s3][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [1, %primals_1]), kwargs = {})
#   %mul_13 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %primals_4), kwargs = {})
#   %neg : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_13,), kwargs = {})
#   %convert_element_type : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_2, torch.float32), kwargs = {})
#   %sub_1 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %amax), kwargs = {})
#   %sub_2 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %exp_1 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_4 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg, [-1], True), kwargs = {})
#   %mul_14 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_8 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %mul_14), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_8, torch.bfloat16), kwargs = {})
#   return %sum_4,%convert_element_type_1
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp5 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tmp1 * tmp3
        tmp6 = tmp4 * tmp5
        tmp7 = -tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr0 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp13 = tl.load(in_ptr1 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp21 = tl.load(in_ptr4 + (0))
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.load(in_ptr5 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp16 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tmp12 * tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = -tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp23 = tmp20 - tmp22
        tmp26 = tmp23 - tmp25
        tmp27 = libdevice.exp(tmp26)
        tmp28 = tmp27 * tmp9
        tmp29 = tmp18 - tmp28
        tmp30 = tmp29.to(tl.float32)
        tl.store(out_ptr1 + (tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK])), tmp30, r0_mask)
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
        primals_1, primals_2, primals_4, primals_5, amax, log, tangents_1 = args
        args.clear()
        s3 = primals_1
        assert_size_stride(primals_2, (1, s3), (s3, 1))
        assert_size_stride(primals_4, (1, s3), (s3, 1))
        assert_size_stride(primals_5, (1, ), (1, ))
        assert_size_stride(amax, (1, 1), (1, 1))
        assert_size_stride(log, (1, 1), (1, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, s3), (s3, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, student_log_probs], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.neg, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0.run(tangents_1, primals_5, primals_4, primals_2, amax, log, buf1, 1, s3, stream=stream0)
            del amax
            del log
            del primals_2
            del primals_4
            del primals_5
            del tangents_1
        return (None, buf1, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 65
    primals_2 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_4, primals_5, amax, log, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
