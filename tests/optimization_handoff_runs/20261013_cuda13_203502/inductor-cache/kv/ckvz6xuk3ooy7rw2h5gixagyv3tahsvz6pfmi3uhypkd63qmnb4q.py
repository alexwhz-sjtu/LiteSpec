# AOT ID: ['0_backward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/x7/cx73ifho5l3egjh65z5hthhzja3yuo2tuxe3ga7k7uggpiebshrp.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.expand, aten.mul, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1, sub_1, sub_2
# Graph fragment:
#   %primals_3 : Tensor "i64[1][1]cuda:0" = PlaceHolder[target=primals_3]
#   %tangents_1 : Tensor "f32[][]cuda:0" = PlaceHolder[target=tangents_1]
#   %primals_4 : Tensor "f32[1][1]cuda:0" = PlaceHolder[target=primals_4]
#   %primals_2 : Tensor "f32[1, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %amax : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=amax]
#   %log : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %sum_3 : Tensor "f32[1, 1][1, 1]cuda:0" = PlaceHolder[target=sum_3]
#   %expand : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%tangents_1, [1]), kwargs = {})
#   %mul_7 : Tensor "f32[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expand, %primals_4), kwargs = {})
#   %unsqueeze_1 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_3, 1), kwargs = {})
#   %ne_2 : Tensor "b8[1, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_1, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %unsqueeze_1, %full_default), kwargs = {})
#   %scatter_upon_const_tensor : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch._inductor.fx_passes.post_grad.scatter_upon_const_tensor](args = (), kwargs = {shape: [1, %primals_1], background_val: 0, dtype: torch.float32, dim: 1, selector: %where_2, val: -1.0})
#   %unsqueeze_2 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mul_7, 1), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %unsqueeze_2, %full_default_1), kwargs = {})
#   %mul_8 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_upon_const_tensor, %where_3), kwargs = {})
#   %sub_1 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %amax), kwargs = {})
#   %sub_2 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_1, %log), kwargs = {})
#   %exp_1 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_3 : Tensor "f32[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [1], True), kwargs = {})
#   %mul_9 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_1, %sum_3), kwargs = {})
#   %sub_5 : Tensor "f32[1, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   return %sum_3,%sub_5
triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_0', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        primals_1, primals_2, primals_3, primals_4, amax, log, tangents_1 = args
        args.clear()
        s3 = primals_1
        assert_size_stride(primals_2, (1, s3), (s3, 1))
        assert_size_stride(primals_3, (1, ), (1, ))
        assert_size_stride(primals_4, (1, ), (1, ))
        assert_size_stride(amax, (1, 1), (1, 1))
        assert_size_stride(log, (1, 1), (1, 1))
        assert_size_stride(tangents_1, (), ())
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf1 = empty_strided_cuda((1, s3), (s3, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.expand, aten.mul, aten.nll_loss_backward, aten.nll_loss_forward, aten._log_softmax, aten._log_softmax_backward_data]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax__log_softmax_backward_data_expand_mul_nll_loss_backward_nll_loss_forward_0.run(primals_3, tangents_1, primals_4, primals_2, amax, log, buf1, 1, s3, stream=stream0)
            del amax
            del log
            del primals_2
            del primals_3
            del primals_4
            del tangents_1
        return (None, buf1, None, None, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 65
    primals_2 = rand_strided((1, 65), (65, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    amax = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    log = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, amax, log, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
