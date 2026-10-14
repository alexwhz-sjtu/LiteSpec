# AOT ID: ['0_forward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/eq/ceqk5wslvjr2ug4b27tpc4tdw72ifx7j3idwolyhelvv5n3e42zg.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => full_default, full_default_1, gather, log, ne_2, neg, squeeze, sub_4, unsqueeze, where, where_1
# Graph fragment:
#   %primals_3 : Tensor "f32[s88, s3][s3, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %getitem_1 : Tensor "f32[s88, 1][1, s88]cuda:0" = PlaceHolder[target=getitem_1]
#   %primals_4 : Tensor "i64[s88][1]cuda:0" = PlaceHolder[target=primals_4]
#   %getitem : Tensor "f32[s88, 1][1, 1]cuda:0" = PlaceHolder[target=getitem]
#   %log : Tensor "f32[s88, 1][1, 1]cuda:0" = PlaceHolder[target=log]
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%primals_3, 1), kwargs = {})
#   %sub_tensor : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %getitem), kwargs = {})
#   %log : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.log.default](args = (%getitem_1,), kwargs = {})
#   %sub_4 : Tensor "f32[s88, s3][s3, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor, %log), kwargs = {})
#   %ne_2 : Tensor "b8[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%primals_4, -100), kwargs = {})
#   %full_default : Tensor "i64[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "i64[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %primals_4, %full_default), kwargs = {})
#   %unsqueeze : Tensor "i64[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%where, 1), kwargs = {})
#   %gather : Tensor "f32[s88, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%sub_4, 1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, 1), kwargs = {})
#   %neg : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %full_default_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : Tensor "f32[s88][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%ne_2, %neg, %full_default_1), kwargs = {})
#   return %getitem,%getitem_1,%log,%where_1
triton_red_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_0 = async_compile.triton('triton_red_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r0_': 262144},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp2_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + ks0*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])

        _tmp2_max_next, _tmp2_sum_next = triton_helpers.online_softmax_combine(
            _tmp2_max, _tmp2_sum, tmp1, False
        )

        _tmp2_max = tl.where(r0_mask & xmask, _tmp2_max_next, _tmp2_max)
        _tmp2_sum = tl.where(r0_mask & xmask, _tmp2_sum_next, _tmp2_sum)

    tmp2, tmp3 = triton_helpers.online_softmax_reduce(
        _tmp2_max, _tmp2_sum, 1, False)
    tmp2 = tmp2[:, None]
    tmp3 = tmp3[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl_math.log(tmp3)
    tmp6 = tl.full([1, 1], -100, tl.int64)
    tmp7 = tmp5 != tmp6
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tl.where(tmp7, tmp5, tmp8)
    tmp10 = ks0
    tmp11 = tmp9 + tmp10
    tmp12 = tmp9 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp9)
    tl.device_assert(((0 <= tmp13) & (tmp13 < ks0)) | ~(xmask), "index out of bounds: 0 <= tmp13 < ks0")
    tmp15 = tl.load(in_ptr0 + (tmp13 + ks0*x0), xmask, eviction_policy='evict_last')
    tmp16 = tmp15 - tmp2
    tmp17 = tmp16 - tmp4
    tmp18 = -tmp17
    tmp19 = 0.0
    tmp20 = tl.where(tmp7, tmp18, tmp19)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)
''', device_str='cuda')


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/inductor-cache/mt/cmtnwpeg4bx5hkfiryymuj6zvgljnxfk6xrip5acsx2ocsgiwlzn.py
# Topologically Sorted Source Nodes: [mul, sum_1], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   mul => mul_12
#   sum_1 => sum_2
# Graph fragment:
#   %where_1 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=where_1]
#   %primals_5 : Tensor "f32[s88][1]cuda:0" = PlaceHolder[target=primals_5]
#   %mul_12 : Tensor "f32[s88][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_1, %primals_5), kwargs = {})
#   %sum_2 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_12,), kwargs = {})
#   return %sum_2
triton_red_fused_mul_sum_1 = async_compile.triton('triton_red_fused_mul_sum_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mul_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mul_sum_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
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
        s88 = primals_1
        s3 = primals_2
        assert_size_stride(primals_3, (s88, s3), (s3, 1))
        assert_size_stride(primals_4, (s88, ), (1, ))
        assert_size_stride(primals_5, (s88, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s88, 1), (1, 1), torch.float32)
            buf1 = empty_strided_cuda((s88, 1), (1, s88), torch.float32)
            buf2 = reinterpret_tensor(buf1, (s88, 1), (1, 1), 0); del buf1  # reuse
            buf3 = empty_strided_cuda((s88, ), (1, ), torch.float32)
            # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [prims.prepare_softmax_online, aten.sub, aten._log_softmax, aten.nll_loss_forward]
            stream0 = get_raw_stream(0)
            triton_red_fused__log_softmax_nll_loss_forward_prepare_softmax_online_sub_0.run(buf2, primals_3, primals_4, buf0, buf3, s3, s88, s3, stream=stream0)
            buf4 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [mul, sum_1], Original ATen: [aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_mul_sum_1.run(buf3, primals_5, buf4, 1, s88, stream=stream0)
        return (buf4, buf3, primals_3, primals_4, primals_5, buf0, buf2, s88, s3, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 512
    primals_2 = 151936
    primals_3 = rand_strided((512, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
