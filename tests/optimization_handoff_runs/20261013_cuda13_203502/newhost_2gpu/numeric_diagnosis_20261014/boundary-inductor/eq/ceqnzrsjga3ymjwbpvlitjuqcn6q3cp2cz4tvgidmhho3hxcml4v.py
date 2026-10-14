# AOT ID: ['2_forward']
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


# kernel path: /diancpfs/user/wanghanzhen/LiteSpec/tests/optimization_handoff_runs/20261013_cuda13_203502/newhost_2gpu/numeric_diagnosis_20261014/boundary-inductor/rr/crrmxu3zrlfbhmitygydbeayhxhr76edxdn3s7fljtfcyw6d4lmc.py
# Topologically Sorted Source Nodes: [unsqueeze, gather, squeeze, neg, mul, sum_1], Original ATen: [aten.unsqueeze, aten.gather, aten.squeeze, aten.neg, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   gather => gather
#   mul => mul_4
#   neg => neg
#   squeeze => squeeze
#   sum_1 => sum_1
#   unsqueeze => unsqueeze
# Graph fragment:
#   %primals_5 : Tensor "i64[s15][1]cuda:0" = PlaceHolder[target=primals_5]
#   %primals_3 : Tensor "f32[s47, s84][s84, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_6 : Tensor "f32[s15][1]cuda:0" = PlaceHolder[target=primals_6]
#   %unsqueeze : Tensor "i64[s15, 1][1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%primals_5, -1), kwargs = {})
#   %gather : Tensor "f32[s15, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.gather.default](args = (%primals_3, -1, %unsqueeze), kwargs = {})
#   %squeeze : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%gather, -1), kwargs = {})
#   %neg : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%squeeze,), kwargs = {})
#   %mul_4 : Tensor "f32[s15][1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %primals_6), kwargs = {})
#   %sum_1 : Tensor "f32[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_4,), kwargs = {})
#   return %sum_1
triton_red_fused_gather_mul_neg_squeeze_sum_unsqueeze_0 = async_compile.triton('triton_red_fused_gather_mul_neg_squeeze_sum_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1, 'r0_': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i64', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gather_mul_neg_squeeze_sum_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'E32C50C3B9CF9CDC47D62CC6811346E52908133A0176777B4DADF595E1971975', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_gather_mul_neg_squeeze_sum_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    _tmp11 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_0 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr2 + (r0_0), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = ks0
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < ks0)) | ~(r0_mask), "index out of bounds: 0 <= tmp4 < ks0")
        tmp6 = tl.load(in_ptr1 + (tmp4 + ks0*r0_0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp7 = -tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(r0_mask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
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
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
        args.clear()
        s47 = primals_1
        s84 = primals_2
        s97 = primals_4
        primals_5_size = primals_5.size()
        s15 = primals_5_size[0]
        assert_size_stride(primals_3, (s47, s84), (s84, 1))
        assert_size_stride(primals_5, (s15, ), (1, ))
        assert_size_stride(primals_6, (s15, ), (1, ))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((), (), torch.float32)
            # Topologically Sorted Source Nodes: [unsqueeze, gather, squeeze, neg, mul, sum_1], Original ATen: [aten.unsqueeze, aten.gather, aten.squeeze, aten.neg, aten.mul, aten.sum]
            stream0 = get_raw_stream(0)
            triton_red_fused_gather_mul_neg_squeeze_sum_unsqueeze_0.run(primals_5, primals_3, primals_6, buf0, s84, 1, s15, stream=stream0)
            del primals_3
        return (buf0, primals_6, reinterpret_tensor(primals_5, (s15, 1), (1, 1), 0), s47, s84, s15, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 32
    primals_2 = 151936
    primals_3 = rand_strided((32, 151936), (151936, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = 32
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.int64)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
