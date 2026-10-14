# AOT ID: ['1_backward']
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


# kernel path: tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/regression-inductor/k5/ck525ordznjzorext7zw3hyug6tdpk2ta4m2zjgoazvtwwcgxysb.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1, sub_3, sub_4
# Graph fragment:
#   %primals_4 : Tensor "i64[s88][1]cuda:0" = PlaceHolder[target=primals_4]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_5 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=primals_5]
#   %tangents_2 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_2]
#   %sign : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=sign]
#   %primals_3 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %amax : Tensor "f32[s88, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[s88, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_5 : Tensor "f32[s88, 1][1, s88]cuda:0" = PlaceHolder[target=sum_5]
#   %expand : Tensor "f32[s88][0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_2, [%primals_1]), kwargs = {})
#   %mul_25 : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_5), kwargs = {})
#   %unsqueeze_1 : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_25, -1), kwargs = {})
#   %expand_1 : Tensor "f32[s88, s3][1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_1, [%primals_1, %primals_2]), kwargs = {})
#   %mul_26 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_1, %sign), kwargs = {})
#   %expand_2 : Tensor "f32[s88][0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [%primals_1]), kwargs = {})
#   %mul_27 : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand_2, %primals_5), kwargs = {})
#   %unsqueeze_2 : Tensor "i64[s88, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_4, 1), kwargs = {})
#   %ne_4 : Tensor "b8[s88, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_2, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "i64[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_2, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [%primals_1, %primals_2], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %unsqueeze_3 : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_27, 1), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_3, %full_default_1), kwargs = {})
#   %mul_28 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %sub_3 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %amax), kwargs = {})
#   %sub_4 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_3, %log), kwargs = {})
#   %exp_1 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_5 : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_28, [1], True), kwargs = {})
#   %mul_29 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_5), kwargs = {})
#   %sub_23 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_28, %mul_29), kwargs = {})
#   return %sum_5,%mul_26,%sub_23
triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_unsqueeze_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
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
        primals_1, primals_2, primals_3, primals_4, primals_5, amax, log, sign, tangents_1, tangents_2 = args
        args.clear()
        s88 = primals_1
        s3 = primals_2
        assert_size_stride(primals_3, (s88, s3), (s3, 1))
        assert_size_stride(primals_4, (s88, ), (1, ))
        assert_size_stride(primals_5, (s88, ), (1, ))
        assert_size_stride(amax, (s88, 1), (1, 1))
        assert_size_stride(log, (s88, 1), (1, 1))
        assert_size_stride(sign, (s88, s3), (s3, 1))
        assert_size_stride(tangents_1, (), ())
        assert_size_stride(tangents_2, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = sign; del sign  # reuse
            buf2 = empty_strided_cuda((s88, s3), (s3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.expand, aten.mul, aten.unsqueeze, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax, aten._log_softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_unsqueeze_0.run(buf0, primals_4, tangents_1, primals_5, tangents_2, primals_3, amax, log, buf2, s3, s88, s3, stream=stream0)
            del amax
            del log
            del primals_3
            del primals_4
            del primals_5
            del tangents_1
            del tangents_2
        return (None, None, buf2, None, None, buf0, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 151936
    primals_3 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((32, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    sign = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    tangents_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, amax, log, sign, tangents_1, tangents_2])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
