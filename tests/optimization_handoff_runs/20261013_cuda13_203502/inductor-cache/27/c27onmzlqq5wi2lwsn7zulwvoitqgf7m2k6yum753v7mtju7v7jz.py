# AOT ID: ['11_backward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/xd/cxdvn6nhtfcrr4wznryayer6lf6iaeunp3kyqkfpdo44nwlgkin4.py
# Topologically Sorted Source Nodes: [float_1, student_log_probs], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.neg, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   float_1 => convert_element_type
#   student_log_probs => sub_2, sub_3
# Graph fragment:
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_6 : Tensor "f32[s45][1]cuda:0" = PlaceHolder[target=primals_6]
#   %primals_5 : Tensor "f32[s45, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_3 : Tensor "bf16[s45, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %amax : Tensor "f32[s45, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[s45, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_4 : Tensor "f32[s45, 1][1, s45]cuda:0" = PlaceHolder[target=sum_4]
#   %expand : Tensor "f32[s45][0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [%primals_1]), kwargs = {})
#   %mul_16 : Tensor "f32[s45][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_6), kwargs = {})
#   %unsqueeze : Tensor "f32[s45, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_16, -1), kwargs = {})
#   %expand_1 : Tensor "f32[s45, s3][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze, [%primals_1, %primals_2]), kwargs = {})
#   %mul_17 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %primals_5), kwargs = {})
#   %neg : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.neg.default](args = (%mul_17,), kwargs = {})
#   %convert_element_type : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float32), kwargs = {})
#   %sub_2 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %amax), kwargs = {})
#   %sub_3 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_2, %log), kwargs = {})
#   %exp_1 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_3,), kwargs = {})
#   %sum_4 : Tensor "f32[s45, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%neg, [-1], True), kwargs = {})
#   %mul_18 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_4), kwargs = {})
#   %sub_15 : Tensor "f32[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%neg, %mul_18), kwargs = {})
#   %convert_element_type_1 : Tensor "bf16[s45, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_15, torch.bfloat16), kwargs = {})
#   return %sum_4,%convert_element_type_1
triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp4 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = -tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tl.load(in_ptr0 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp18 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp13 = tl.load(in_ptr2 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tmp11 * tmp2
        tmp14 = tmp12 * tmp13
        tmp15 = -tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 - tmp18
        tmp21 = tmp19 - tmp20
        tmp22 = libdevice.exp(tmp21)
        tmp23 = tmp22 * tmp8
        tmp24 = tmp15 - tmp23
        tmp25 = tmp24.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + ks0*x0), tmp25, r0_mask & xmask)
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
        primals_1, primals_2, primals_3, primals_5, primals_6, amax, log, tangents_1 = args
        args.clear()
        s45 = primals_1
        s3 = primals_2
        assert_size_stride(primals_3, (s45, s3), (s3, 1))
        assert_size_stride(primals_5, (s45, s3), (s3, 1))
        assert_size_stride(primals_6, (s45, ), (1, ))
        assert_size_stride(amax, (s45, 1), (1, 1))
        assert_size_stride(log, (s45, 1), (1, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((s45, s3), (s3, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [float_1, student_log_probs], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.neg, aten._to_copy, aten._log_softmax, aten._log_softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_expand_mul_neg_unsqueeze_0.run(tangents_1, primals_6, primals_5, primals_3, amax, log, buf1, s3, s45, s3, stream=stream0)
            del amax
            del log
            del primals_3
            del primals_5
            del primals_6
            del tangents_1
        return (None, None, buf1, None, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 17
    primals_2 = 513
    primals_3 = rand_strided((17, 513), (513, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_5 = rand_strided((17, 513), (513, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((17, ), (1, ), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((17, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((17, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_5, primals_6, amax, log, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
